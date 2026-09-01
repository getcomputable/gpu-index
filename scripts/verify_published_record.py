#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Verify PUBLISHED index observations by recompute-and-match.

The public face of ./reproduce: consumes the published record (a local
downloaded copy under GPU_INDEX_DATA_DIR, the public HTTPS front via
GPU_INDEX_PUBLIC_BASE_URL, or an S3-compatible bucket) and, for every
requested observation,

  - verifies the file's envelope digest (artifact_sha256 recomputed from
    the compact canonical JSON of the parsed payload; the file itself is
    pretty-printed by design);
  - rebuilds the per-provider sd-votes from the published receipts and
    recomputes the index value and stability band with the panel
    engine's own vote math, matching the published numbers exactly.

Observations whose contributing receipts are withheld by the disclosure
policy (price_disclosure "withheld") degrade to digest-verification only
and say so; they do not fail the run.

  verify_published_record.py --sku H100 --date 2026-08-22
  verify_published_record.py --sku B200 --date 2026-08-20T04

Exit codes: 0 every checked observation matched (digest OK; degraded
observations are reported but do not fail), 1 any MISMATCH or digest
FAIL, 2 could not verify (the record source is unreachable, no published
day file for the date, or no observation for the SKU/stamp) or usage
error. Never 0 without verifying something.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpu_index.common.bucket import BucketPublishError  # noqa: E402
from gpu_index.published.artifacts import (  # noqa: E402
    ArtifactDigestError,
    PublishedRecordError,
)
from gpu_index.published.reader import PublishedRecordReader  # noqa: E402
from gpu_index.published.verify import (  # noqa: E402
    MIN_DISCLOSURE_WINDOW_DAYS,
    VERDICT_DEGRADED,
    VERDICT_MATCH,
    disclosure_window_warning,
    recompute_observation,
    select_observations,
)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3])$")


def _value_label(value, band) -> str:
    if value is None:
        return "NO-PRINT"
    return f"{value} (band {band})"


def _window_warning(reader, sku: str, date: str):
    """One probe read at the disclosure-window bound; never fatal.

    A whole-day run is the CLI's full-history check, so it also reports
    whether the observable published history behind the verified day
    covers the weighting lookback (MIN_DISCLOSURE_WINDOW_DAYS). The
    probe is a single day-file read; any probe failure is swallowed --
    this warning must never fail or block a verification run.
    """
    probe_date = (
        datetime.date.fromisoformat(date)
        - datetime.timedelta(days=MIN_DISCLOSURE_WINDOW_DAYS - 1)
    ).isoformat()
    try:
        if reader.read_day(probe_date, sku=sku) is not None:
            return None
    except Exception:
        return None  # probe unreadable: claim nothing either way
    return disclosure_window_warning(
        sku=sku, verified_date=date, probe_date=probe_date
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="verify published index observations "
        "(recompute-and-match)"
    )
    parser.add_argument(
        "--sku",
        required=True,
        help="published SKU, e.g. H100, H200, B300, B200",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="YYYY-MM-DD (whole UTC day) or YYYY-MM-DDTHH (one observation)",
    )
    args = parser.parse_args(argv)

    sku = args.sku.upper()
    stamp = None
    if _STAMP_RE.match(args.date):
        stamp = args.date
        date = args.date[:10]
    elif _DATE_RE.match(args.date):
        date = args.date
    else:
        print(
            f"bad date {args.date!r} (expected YYYY-MM-DD or YYYY-MM-DDTHH)",
            file=sys.stderr,
        )
        return 2

    try:
        reader = PublishedRecordReader()
    except (PublishedRecordError, BucketPublishError) as exc:
        print(f"published record: {exc}", file=sys.stderr)
        return 2
    try:
        key = reader.resolve_day_key(date, sku=sku)
        print(f"published record: {key} via {reader.describe()}")
        envelope = reader.read_day(date, sku=sku, resolved_key=key)
    except (httpx.HTTPError, BucketPublishError, OSError) as exc:
        # Fail LOUDLY, never silently: an unreachable record source is
        # "could not verify" (exit 2), one actionable line -- never a
        # traceback and never a 0 that verified nothing.
        print(
            f"could not verify: the published record is unreachable via "
            f"{reader.describe()} ({exc}) -- set GPU_INDEX_PUBLIC_BASE_URL "
            "to a reachable record front, or download the record and point "
            "GPU_INDEX_DATA_DIR at the copy",
            file=sys.stderr,
        )
        return 2
    except ArtifactDigestError as exc:
        print(f"digest FAIL: {exc}", file=sys.stderr)
        print(
            "the file's embedded artifact_sha256 does not match its own "
            "payload; nothing in it is verifiable",
            file=sys.stderr,
        )
        return 1
    except PublishedRecordError as exc:
        print(f"invalid published artifact: {exc}", file=sys.stderr)
        return 1
    if envelope is None:
        print(
            f"no published day file for {date} ({key}): outside the "
            "publisher's trailing window, or not yet published. Pick a "
            "date inside the published window, or run "
            "./reproduce --producer <sku> <date> against a local "
            "producer record.",
            file=sys.stderr,
        )
        return 2

    digest = envelope["artifact_sha256"]
    print(f"digest OK: {digest}")

    observations = select_observations(envelope, sku=sku, stamp=stamp)
    if not observations:
        target = stamp or date
        print(
            f"the published day file has no {sku} observation at {target} "
            f"(day file spans {envelope['meta']['from_observed_at']} .. "
            f"{envelope['meta']['to_observed_at']})",
            file=sys.stderr,
        )
        return 2

    matched = mismatched = degraded = 0
    for observation in observations:
        try:
            check = recompute_observation(observation)
        except PublishedRecordError as exc:
            print(f"invalid published observation: {exc}", file=sys.stderr)
            return 1
        stamp_label = check.observed_at[:16]
        if stamp_label.endswith(":00"):
            # Preserve the historical hour-grain transcript while making
            # quarter-hour observations distinguishable within an hour.
            stamp_label = stamp_label[:13]
        if check.verdict == VERDICT_DEGRADED:
            degraded += 1
            print(
                f"{check.sku} {stamp_label} DEGRADED digest-only "
                f"(withheld: {', '.join(check.withheld_sources)}) digest OK"
            )
        else:
            verdict = (
                "MATCH" if check.verdict == VERDICT_MATCH else "MISMATCH"
            )
            if check.verdict == VERDICT_MATCH:
                matched += 1
            else:
                mismatched += 1
            published = _value_label(
                check.published_value, check.published_band
            )
            if check.status == "no_print":
                recomputed = (
                    "NO-PRINT"
                    if check.verdict == VERDICT_MATCH
                    else "PRINTABLE"
                )
                reason = observation.get("reason")
                published = f"NO-PRINT ({reason})"
            else:
                recomputed = _value_label(
                    check.recomputed_value, check.recomputed_band
                )
            print(
                f"{check.sku} {stamp_label} recomputed {recomputed} "
                f"published {published} {verdict} digest OK"
            )
        for message in check.messages:
            print(f"  {message}")

    total = matched + mismatched + degraded
    print(
        f"summary: {total} observation(s): {matched} MATCH, "
        f"{mismatched} MISMATCH, {degraded} degraded"
    )
    if stamp is None:
        window_note = _window_warning(reader, sku, date)
        if window_note is not None:
            print(f"WARNING: {window_note}")
    if mismatched:
        return 1
    if degraded:
        print(
            f"NOTE: {degraded} observation(s) degraded to digest-only "
            "verification: their contributing prices are withheld by the "
            "published disclosure policy, so the vote recompute cannot "
            "run for them (the file digest still verifies)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
