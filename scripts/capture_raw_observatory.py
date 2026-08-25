#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Capture the raw price observatory — wide-net per-source GPU price prints.

Collection only, for as many chips and sources as have parseable public
surfaces. This script computes nothing, but since the hourly panel mint
(2026-08-23, METHODOLOGY.md) the stored record IS consumed, read-only, by
the six panel-index calc lanes -- a collector or recipe change here CAN
move a contractual panel print, so treat collector edits with basket-lane
care. Capture gaps still can never be backfilled. Fully separate from the
basket lanes: this script never touches any index/<chip>_basket key, and
nothing writes under this lane's prefix but this script.

Scheduling mirrors the basket lanes: fired frequently by a scheduler,
slot idempotency ALWAYS on — any
non-``--force`` run exits 0 when the current (date, slot) is already
captured, so frequent firings record exactly the configured slots per day
and a dead firing self-heals on the next.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpu_index.common.slots import current_slot, is_canonical, utc_now  # noqa: E402
from gpu_index.observatory.catalog import load_sku_catalog  # noqa: E402
from gpu_index.observatory.collect import collect_all  # noqa: E402
from gpu_index.observatory.config import (  # noqa: E402
    load_observatory_config,
    resolve_catalog_path,
)
from gpu_index.observatory.snapshot import build_capture_snapshot  # noqa: E402
from gpu_index.observatory.sources import COLLECTORS  # noqa: E402
from gpu_index.observatory.store import (  # noqa: E402
    BucketConfig,
    make_client,
    make_run_id,
    slot_already_captured,
    slot_hours_present,
    upload_capture_snapshot,
    write_local_snapshot,
)

CAPTURER_VERSION = "capture_raw_observatory/0.1"

# Remote-derived strings (page fragments in exception text, provider labels)
# get printed into job logs, where a newline + '::' sequence is a workflow
# command in GH Actions. Strip control chars and truncate at print time only
# — the stored snapshot keeps raw.
_CONTROL_CHARS_RE = re.compile(r"[\r\n\x00-\x08\x0b-\x1f]")


def _clean(text, limit: int = 300) -> str:
    return _CONTROL_CHARS_RE.sub(" ", str(text))[:limit]


def _in_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def notice(msg: str) -> None:
    print(f"::notice::{msg}" if _in_actions() else f"NOTICE: {msg}")


def warn(msg: str) -> None:
    print(f"::warning::{msg}" if _in_actions() else f"WARNING: {msg}")


def error(msg: str) -> None:
    print(f"::error::{msg}" if _in_actions() else f"ERROR: {msg}")


def print_summary(payload) -> None:
    """Per-source summary lines. Deliberately NOT one line per observation —
    a wide capture holds hundreds of rows (marketplace books, aggregator
    sweeps); the full document is in the snapshot and behind --json."""
    for src in payload["sources"]:
        sid = src["source_id"]
        if src["status"] != "ok":
            print(f"  {sid:<16} {src['status']}: {_clean(src.get('error', ''))}")
            continue
        obs = src["observations"]
        skus = sorted({o["sku"] for o in obs if o["sku"] is not None})
        n_unmapped = sum(1 for o in obs if o["sku_match"] == "unmapped")
        n_implausible = sum(1 for o in obs if o.get("implausible"))
        usd = [
            float(o["price_usd_gpu_hr"])
            for o in obs
            if o["price_usd_gpu_hr"] is not None
        ]
        price_range = (
            f"${min(usd):.4f}-${max(usd):.4f}/GPU-hr" if usd else "no USD prints"
        )
        flags = ""
        if n_unmapped:
            flags += f"  unmapped={n_unmapped}"
        if n_implausible:
            flags += f"  implausible={n_implausible}"
        # Whole sku names only — a mid-name truncation ('RTX_PRO_600')
        # reads as a real sku in job logs.
        shown = skus[:16]
        sku_list = ", ".join(shown) + (
            f", +{len(skus) - len(shown)} more" if len(skus) > len(shown) else ""
        )
        print(
            f"  {sid:<16} ok: {len(obs):>3} obs, {len(skus):>2} skus "
            f"({_clean(sku_list, 400)}), {price_range}{flags}"
        )
        if src.get("partial_errors"):
            for perr in src["partial_errors"]:
                print(f"  {sid:<16} partial: {_clean(perr, 200)}")
    if payload["unmapped_identifiers"]:
        warn(
            "unmapped sku identifiers (grow config/gpu_sku_catalog.json): "
            + _clean("; ".join(payload["unmapped_identifiers"]), 500)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture wide-net per-source GPU price observations "
            "(collection only)."
        )
    )
    parser.add_argument(
        "--slot-gated",
        action="store_true",
        help=(
            "Mark this run as job-driven (notices phrased for the job log). "
            "Slot idempotency itself is ALWAYS on: any non---force run "
            "skips (exit 0) when the slot is already captured."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Capture even if the current slot already has a snapshot",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Collect and print; write nothing"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write the local mirror but skip the bucket (dev)",
    )
    parser.add_argument(
        "--only-source",
        action="append",
        dest="only_sources",
        metavar="NAME",
        help=(
            "Debug: limit collectors (implies --dry-run). Choices: "
            + ", ".join(sorted(COLLECTORS))
        ),
    )
    parser.add_argument(
        "--config", type=Path, help="Override config/raw_observatory.json"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full snapshot JSON"
    )
    args = parser.parse_args()

    if args.only_sources:
        # Fail loud on a typo'd name: silently selecting nothing would exit
        # with the misleading 'every implemented source failed'.
        unknown = sorted(set(args.only_sources) - set(COLLECTORS))
        if unknown:
            parser.error(
                f"--only-source {', '.join(unknown)}: no such collector "
                f"(choices: {', '.join(sorted(COLLECTORS))})"
            )
        if not args.dry_run:
            notice(
                "--only-source implies --dry-run: partial snapshots are never recorded"
            )
            args.dry_run = True

    config = load_observatory_config(args.config)
    catalog = load_sku_catalog(resolve_catalog_path(config))
    now = utc_now()
    slot_date, slot_hour = current_slot(now, config["capture_slots_utc"])
    canonical = is_canonical(slot_hour, config.get("canonical_slot_utc"))
    prefix = config["bucket_prefix"]

    client = None
    bucket = None
    previous_day_empty = None
    if not (args.dry_run or args.local_only):
        bucket_config = BucketConfig.from_env()
        client = make_client(bucket_config)
        bucket = bucket_config.bucket
        # Missed-day alarm FIRST — missed history is the unrecoverable case
        # and must be loud on EVERY firing that sees it, including the cheap
        # already-captured no-ops below. Coverage-based: a day that captured
        # 22:00 but missed the canonical 16:00 must not look healthy just
        # because it is non-empty.
        prev_day = slot_date - timedelta(days=1)
        prev_present = slot_hours_present(
            client, bucket, prefix=prefix, day=prev_day
        )
        prev_missing = sorted(set(config["capture_slots_utc"]) - prev_present)
        previous_day_empty = not prev_present
        if previous_day_empty:
            warn(
                f"raw observatory: previous day {prev_day} has ZERO stored "
                "snapshots — that history is permanently lost (missed-day "
                "alarm; expected on every firing of the lane's FIRST day)"
            )
        elif prev_missing:
            canonical_hour = config.get("canonical_slot_utc")
            canonical_note = (
                " including the CANONICAL slot"
                if canonical_hour in prev_missing
                else ""
            )
            warn(
                f"raw observatory: previous day {prev_day} missed slot(s) "
                f"{prev_missing}{canonical_note} — those observations are "
                "permanently lost (partial-miss alarm)"
            )
        if slot_already_captured(
            client, bucket, prefix=prefix, day=slot_date, slot_hour=slot_hour
        ):
            if args.force:
                notice(
                    f"slot {slot_date} {slot_hour:02d}:00Z already captured — "
                    "--force records another snapshot"
                )
            else:
                notice(
                    f"raw observatory slot {slot_date} {slot_hour:02d}:00Z "
                    "already captured — nothing to do"
                )
                return 0
        elif args.slot_gated and now.hour not in config["capture_slots_utc"]:
            notice(
                f"claiming slot {slot_date} {slot_hour:02d}:00Z at "
                f"{now.strftime('%H:%M')}Z (first firing since the mark)"
            )

    run_id = make_run_id(now)
    # Recorded after the mark hour (a later firing filled the slot)?
    late_fill = not (now.date() == slot_date and now.hour == slot_hour)
    results = collect_all(
        config, COLLECTORS, only=set(args.only_sources or []) or None
    )
    payload = build_capture_snapshot(
        config=config,
        catalog=catalog,
        source_results=results,
        captured_at=now,
        run_id=run_id,
        slot_date=slot_date,
        slot_hour_utc=slot_hour,
        canonical=canonical,
        capturer={
            "job": "github-actions"
            if _in_actions()
            else os.environ.get("OBSERVATORY_CAPTURER_JOB", "manual"),
            "hostname": os.environ.get("HOSTNAME"),
            "git_sha": os.environ.get("GITHUB_SHA"),
            "version": CAPTURER_VERSION,
        },
        previous_day_empty=previous_day_empty,
        late_fill=late_fill,
    )

    print(
        f"raw observatory capture: {payload['capture_date']} "
        f"slot {slot_hour:02d}:00Z run {run_id}"
        f"{' (canonical)' if canonical else ''}"
    )
    print_summary(payload)
    if args.json:
        print(json.dumps(payload, indent=2))

    if not payload["sources_ok"]:
        error("every implemented source failed — snapshot NOT recorded")
        return 1

    n_implemented = sum(
        1
        for s in payload["sources"]
        if s["status"] != "unimplemented"
    )
    min_claim = int(config.get("min_sources_to_claim", 1))
    print(
        f"coverage: {len(payload['sources_ok'])}/{n_implemented} implemented "
        f"sources ok, {payload['observation_count']} observations, "
        f"{len(payload['skus_observed'])} skus "
        f"(minimum {min_claim} sources to claim the slot)"
    )
    if args.dry_run:
        notice("dry run — nothing written")
        return 0

    if not args.only_sources and len(payload["sources_ok"]) < min_claim:
        # Same rule as the basket lanes: a thin snapshot must never CLAIM
        # the slot — recording it would stop every later firing from
        # retrying while most of the net is dark, and earliest-key-wins
        # would make the thin snapshot the one a reader sees.
        error(
            f"only {len(payload['sources_ok'])}/{n_implemented} sources "
            f"succeeded (minimum {min_claim}) — slot NOT claimed so the "
            "next firing retries"
        )
        return 1

    try:
        local_path = write_local_snapshot(payload)
        print(f"local:    {local_path}")
    except OSError as exc:
        # The local mirror is dev/debug convenience; a read-only container
        # filesystem must not cost the slot its bucket record.
        warn(f"local snapshot mirror failed ({exc}) — continuing to bucket upload")
    if not args.local_only:
        outcome = upload_capture_snapshot(
            client, bucket, payload, prefix=prefix, now=now
        )
        print(f"snapshot: {outcome['snapshot_key']}")
        print(
            f"pointer:  {prefix}/latest.json "
            f"published_at={outcome['pointer']['published_at']}"
        )
    if payload["sources_failed"]:
        warn(
            "raw observatory recorded with failed sources: "
            + ", ".join(payload["sources_failed"])
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
