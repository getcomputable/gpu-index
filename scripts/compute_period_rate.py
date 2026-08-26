#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Coverage report + period rate over a published hourly panel series
(METHODOLOGY.md section 6.1; gpu_index.index.period_rate).

READ-ONLY on purpose: this CLI never publishes, never moves a pointer,
never touches an artifact. It reads the published record of one lane
(--config names the panel config, exactly the compute_panel_index.py
convention -- no env-var lane fallback) and prints the section 6.1
report for a half-open period [--start, --end):

  - scheduled / filled stamps, coverage, every gap with its per-stamp
    cause (missed / dark / quarantined / unpublished);
  - the fill-rule period rate (L = 72), with per-gap fill windows and
    per-stamp provenance (observed | filled | dropped_genesis) so a
    filled value is always distinguishable from an observed one;
  - the band verdict against the RECOMMENDED CONTRACT DEFAULTS
    (settles / review / determination) -- defaults a contract may
    replace; the rate computes identically either way.

Because it is read-only it is safe against an ARMED lane (the manual
--sync prohibition does not apply: there is no second writer here).

A still-running period is CLIPPED at the record frontier -- the last
scheduled stamp whose window has closed (its next scheduled mark is at
or before now). The clip is recorded in the report (period.clipped_at)
so a partial period can never present as a settled one.

Bucket reads are bounded: ONE paginated LIST learns the published
stamp set; artifacts GET only for published stamps inside the period
plus a backward walk before it that stops as soon as
FILL_LOOKBACK_HOURS filled stamps are in hand (or genesis is hit).
Unpublished stamps never GET.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpu_index.index.panel_config import (  # noqa: E402
    PanelConfigError,
    load_panel_config,
    panel_schedule,
)
from gpu_index.index.panel_schedule import (  # noqa: E402
    PanelScheduleError,
    date_hour_to_stamp,
    hour_iso_to_stamp,
    stamp_to_hour_iso,
)
from gpu_index.index.period_rate import (  # noqa: E402
    FILL_LOOKBACK_HOURS,
    PeriodRateError,
    classify_artifact,
    period_report,
)
from gpu_index.common.slots import utc_now  # noqa: E402
from gpu_index.common.store import (  # noqa: E402
    BucketConfig,
    get_panel_composite,
    list_panel_observations,
    make_client,
)


def _in_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


_LOG_CLEAN_RE = re.compile(r"[\r\n\x00-\x1f]")


def _log_clean(msg: str) -> str:
    return _LOG_CLEAN_RE.sub(" ", str(msg))


def notice(msg: str) -> None:
    msg = _log_clean(msg)
    print(f"::notice::{msg}" if _in_actions() else f"NOTICE: {msg}", file=sys.stderr)


def error(msg: str) -> None:
    msg = _log_clean(msg)
    print(f"::error::{msg}" if _in_actions() else f"ERROR: {msg}", file=sys.stderr)


def parse_period_stamp(value: str) -> int:
    """Accept 'YYYY-MM-DDTHH' or a bare 'YYYY-MM-DD' (expands to T00 --
    a day-aligned period boundary). Loud on anything else."""
    text = str(value)
    if len(text) == 10:
        try:
            return date_hour_to_stamp(text, 0)
        except ValueError as exc:
            raise PanelScheduleError(
                f"period boundary must be 'YYYY-MM-DD' or 'YYYY-MM-DDTHH', "
                f"got {value!r}: {exc}"
            ) from exc
    return hour_iso_to_stamp(text)


def build_statuses(
    client,
    bucket: str,
    *,
    prefix: str,
    methodology_id: str,
    schedule,
    start_stamp: int,
    end_stamp: int,
    frontier_stamp: int,
    published: set,
):
    """statuses for the period plus exactly the context the report
    needs: GET each published stamp in [start, end); when any period
    stamp is missing, a backward walk before start stops once
    FILL_LOOKBACK_HOURS filled stamps precede the period (or genesis) --
    a fully-covered period performs ZERO extra reads; when the LAST
    period stamp is missing, a forward walk over [end, frontier) stops
    at the first filled stamp (where the record resumes) so the tail
    run's true length is known. Returns (statuses, gets) -- gets is the
    GET count, reported for the read-budget notice."""
    statuses = {}
    gets = 0

    def classify(stamp: int):
        nonlocal gets
        iso = stamp_to_hour_iso(stamp)
        artifact = None
        if iso in published:
            artifact = get_panel_composite(
                client,
                bucket,
                prefix=prefix,
                methodology_id=methodology_id,
                observation=iso,
            )
            gets += 1
        return classify_artifact(artifact)

    scheduled = schedule.scheduled_stamps(start_stamp, end_stamp)
    for stamp in scheduled:
        statuses[stamp] = classify(stamp)
    any_missing = any(statuses[stamp][0] is None for stamp in scheduled)

    if any_missing:
        filled_before = 0
        cursor = schedule.prev_scheduled_stamp(start_stamp)
        while cursor is not None and filled_before < FILL_LOOKBACK_HOURS:
            statuses[cursor] = classify(cursor)
            if statuses[cursor][0] is not None:
                filled_before += 1
            cursor = schedule.prev_scheduled_stamp(cursor)

    if scheduled and statuses[scheduled[-1]][0] is None:
        for stamp in schedule.scheduled_stamps(end_stamp, frontier_stamp):
            statuses[stamp] = classify(stamp)
            if statuses[stamp][0] is not None:
                break
    return statuses, gets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only section 6.1 coverage report + period rate for one "
            "hourly panel lane over [--start, --end)."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="panel config path (config/index_panel_*.json)",
    )
    parser.add_argument(
        "--start",
        help="period start, 'YYYY-MM-DDTHH' or 'YYYY-MM-DD' (inclusive)",
    )
    parser.add_argument(
        "--end",
        help="period end, 'YYYY-MM-DDTHH' or 'YYYY-MM-DD' (exclusive)",
    )
    parser.add_argument(
        "--series",
        action="store_true",
        help="embed the per-stamp series (source/provenance per stamp)",
    )
    parser.add_argument(
        "--out",
        help="write the JSON report here instead of stdout",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate the named config offline and exit (no bucket reads)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_panel_config(Path(args.config))
        schedule = panel_schedule(config)
    except (PanelConfigError, PanelScheduleError) as exc:
        error(f"config refused: {exc}")
        return 2
    prefix = config["bucket_prefix"]
    methodology_id = config["calc"]["methodology_id"]

    if args.check_config:
        notice(f"config ok: {methodology_id} ({args.config})")
        return 0

    if not args.start or not args.end:
        error("--start and --end are required (except with --check-config)")
        return 2

    try:
        start_stamp = parse_period_stamp(args.start)
        end_stamp = parse_period_stamp(args.end)
    except PanelScheduleError as exc:
        error(str(exc))
        return 2

    # Clip a still-running period at the record frontier: the last
    # scheduled stamp whose window has CLOSED. An observation's window
    # closes at the NEXT scheduled mark (era-aware -- on the 4-slot
    # replay grid a slot stays open for six hours), so the last closed
    # stamp is the one BEFORE the latest scheduled stamp at-or-before
    # now: that latest stamp's own next mark is still in the future.
    # Counting a still-open slot as missing would fabricate a gap on
    # every mid-window run.
    now = utc_now()
    now_stamp = date_hour_to_stamp(now.date().isoformat(), now.hour)
    latest_started = schedule.prev_scheduled_stamp(now_stamp + 1)
    last_closed = (
        None
        if latest_started is None
        else schedule.prev_scheduled_stamp(latest_started)
    )
    clipped_at_stamp = None
    if last_closed is None:
        error("nothing has closed yet on this lane's grid -- period is all future")
        return 2
    frontier_stamp = last_closed + 1
    if end_stamp > frontier_stamp:
        clipped_at_stamp = frontier_stamp
        notice(
            f"period end {stamp_to_hour_iso(end_stamp)} is beyond the record "
            f"frontier; clipping at {stamp_to_hour_iso(clipped_at_stamp)}"
        )
        end_stamp = clipped_at_stamp
    if end_stamp <= start_stamp:
        error(
            f"period [{args.start}, {args.end}) is empty after the frontier "
            f"clip -- nothing has closed inside it yet"
        )
        return 2

    try:
        bucket_config = BucketConfig.from_env()
    except Exception as exc:  # BucketPublishError names publish; reword.
        error(
            f"bucket credentials missing or incomplete -- this READ-ONLY "
            f"report needs the artifact-store env "
            f"(GPU_INDEX_S3_* / GPU_INDEX_S3_BUCKET): {exc}"
        )
        return 2
    client = make_client(bucket_config)
    published = list_panel_observations(
        client,
        bucket_config.bucket,
        prefix=prefix,
        methodology_id=methodology_id,
    )

    try:
        statuses, gets = build_statuses(
            client,
            bucket_config.bucket,
            prefix=prefix,
            methodology_id=methodology_id,
            schedule=schedule,
            start_stamp=start_stamp,
            end_stamp=end_stamp,
            frontier_stamp=frontier_stamp,
            published=published,
        )
        report = period_report(
            schedule=schedule,
            start_stamp=start_stamp,
            end_stamp=end_stamp,
            statuses=statuses,
            methodology_id=methodology_id,
            panel_id=config["panel_id"],
            include_series=args.series,
            clipped_at_stamp=clipped_at_stamp,
            frontier_stamp=frontier_stamp,
        )
    except (PeriodRateError, json.JSONDecodeError) as exc:
        # JSONDecodeError: a listed artifact with malformed bytes must
        # refuse like any other unreadable record, never traceback --
        # this read-only CLI is exactly what gets pointed at a suspect
        # record during an incident.
        error(f"report refused: {exc}")
        return 2

    coverage = report["coverage"]
    notice(
        f"{methodology_id} [{report['period']['start']}, "
        f"{report['period']['end']}): {coverage['filled']}/"
        f"{coverage['scheduled']} filled "
        f"({coverage['coverage_ratio']:.2%}), longest gap "
        f"{coverage['longest_gap']}, band {report['band'].upper()}, rate "
        f"{report['period_rate']['value_usd_gpu_hr']} "
        f"(carried {report['period_rate']['stamps_carried']}), "
        f"{gets} artifact reads"
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
        notice(f"report written: {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
