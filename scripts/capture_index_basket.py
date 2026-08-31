#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Capture the index-basket price points — per-source raw prints only.

Collection only: per-source raw prints into the append-only keyspace under
the lane's ``bucket_prefix``; no composite is computed here.

Designed to be fired every ~30 minutes by a scheduler (collection
mechanics: METHODOLOGY.md section 3.5). Slot idempotency is ALWAYS on: any
non-``--force`` run exits 0
when the current (date, slot) is already captured, so frequent firings
record exactly the configured 2-4 snapshots per day and a dead firing
self-heals on the next. ``--slot-gated`` only marks a run as job-driven
for its notices; it does not change gating.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpu_index.common.bucket import get_object_bytes  # noqa: E402
from gpu_index.index.config import load_basket_config  # noqa: E402
from gpu_index.index.screens import apply_jump_screen  # noqa: E402
from gpu_index.common.slots import (  # noqa: E402
    current_slot,
    is_canonical,
    latest_pointer_key,
    utc_now,
)
from gpu_index.index.snapshot import build_capture_snapshot, derive_basis_pairs  # noqa: E402
from gpu_index.index.sources import COLLECTORS  # noqa: E402
from gpu_index.common.store import (  # noqa: E402
    BucketConfig,
    make_client,
    make_run_id,
    slot_already_captured,
    slot_minutes_present,
    upload_capture_snapshot,
    write_local_snapshot,
)

CAPTURER_VERSION = "capture_index_basket/0.1"
DEFAULT_CAPTURE_BUDGET_SECONDS = 600.0
DEFAULT_PER_SOURCE_DEADLINE_SECONDS = 180.0

# Remote-derived strings (page fragments in exception text, provider region
# labels) get printed into GH Actions logs, where a newline + '::' sequence
# is a workflow command (::add-mask::, ::stop-commands::). Strip control
# chars and truncate at print time only — the stored snapshot keeps raw.
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


def _call_with_deadline(fn, *, timeout, deadline):
    """Run a collector with a HARD wall-clock bound.

    Between-source budget checks alone don't cap the source that is already
    in flight (vast makes up to 4 sequential fetches) — and an over-long
    capture must never overrun its schedule: a runner-level kill
    bypasses the warn-only wrapper, and a scheduler concurrency group
    would make a slow capture DELAY the next firing. The worker is
    a daemon thread: an abandoned fetch dies with the process."""
    outcome = {}

    def _target():
        try:
            outcome["value"] = fn(timeout=timeout)
        except BaseException as exc:  # noqa: BLE001 — reported by the caller
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        raise RuntimeError(
            f"source exceeded its {deadline:.0f}s budget share — abandoned mid-fetch"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def collect_all(config, only=None, timeout=None, budget_seconds=None):
    """Run collectors serially under a hard wall-clock budget.

    urllib timeouts bound socket operations, not requests, so the budget is
    the real fence: each source gets min(per-source deadline, remaining
    budget) and is abandoned past it; once the budget is spent, remaining
    sources are recorded as errors (visible holes). The scheduler's own
    job timeout must never be the thing that stops this step.
    """
    timeout = timeout or float(config.get("per_source_timeout_seconds", 30))
    budget = budget_seconds or float(
        config.get("capture_budget_seconds", DEFAULT_CAPTURE_BUDGET_SECONDS)
    )
    per_source_deadline = float(
        config.get("per_source_deadline_seconds", DEFAULT_PER_SOURCE_DEADLINE_SECONDS)
    )
    started_all = time.monotonic()
    results = []
    for src in config["sources"]:
        sid = src["source_id"]
        if only and sid not in only:
            continue
        fn = COLLECTORS.get(sid)
        if fn is None:
            results.append(
                {
                    "source_id": sid,
                    "status": "unimplemented",
                    "error": "collector not implemented yet",
                    "observations": [],
                }
            )
            continue
        remaining = budget - (time.monotonic() - started_all)
        if remaining <= 0:
            results.append(
                {
                    "source_id": sid,
                    "status": "error",
                    "error": f"capture budget ({budget:.0f}s) exhausted before this source ran",
                    "observations": [],
                }
            )
            continue
        started = time.monotonic()
        try:
            result = _call_with_deadline(
                fn, timeout=timeout, deadline=min(remaining, per_source_deadline)
            )
            result["status"] = "ok"
            # The registry key is authoritative — a collector mislabeling its
            # own source_id must not relabel the snapshot entry.
            result["source_id"] = sid
        except Exception as exc:  # noqa: BLE001 — one dead feed must not kill the capture
            result = {
                "source_id": sid,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "observations": [],
            }
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
    return results


def print_summary(payload) -> None:
    for src in payload["sources"]:
        sid = src["source_id"]
        if src["status"] == "ok":
            for o in src["observations"]:
                flag = "  IMPLAUSIBLE" if o.get("implausible") else ""
                if o["price_usd_gpu_hr"] is not None:
                    price = f"{o['price_usd_gpu_hr']:>8.4f} $/GPU-hr"
                elif o["price_native_per_gpu_hr"] is not None:
                    price = (
                        f"{o['price_native_per_gpu_hr']:>8.4f} "
                        f"{_clean(o['currency'], 8)}/GPU-hr (unconverted)"
                    )
                else:
                    price = "no price parsed"
                print(
                    f"  {sid:<14} {_clean(str(o['sku']), 20):<5} "
                    f"{_clean(str(o['tier']), 16):<14} {price}  "
                    f"({o['gpu_count_basis']}x basis, {_clean(str(o['region']), 40)}){flag}"
                )
        else:
            print(f"  {sid:<14} {src['status']}: {_clean(src.get('error', ''))}")
    for pair in payload["basis_pairs"]:
        print(
            f"  basis pair {pair['source_id']}: "
            f"B300/B200 = {pair['ratio_b300_b200']:.4f} "
            f"({pair['b300_usd_gpu_hr']} / {pair['b200_usd_gpu_hr']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture per-source index-basket price points (collection only)."
    )
    parser.add_argument(
        "--slot-gated",
        action="store_true",
        help=(
            "Mark this run as job-driven (notices phrased for the job "
            "log). Slot idempotency itself is ALWAYS on: any "
            "non---force run skips (exit 0) when the slot is already captured."
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
        help=f"Debug: limit collectors (implies --dry-run). Choices: {', '.join(COLLECTORS)}",
    )
    parser.add_argument("--config", type=Path, help="Override config/index_basket.json")
    parser.add_argument("--json", action="store_true", help="Print the full snapshot JSON")
    args = parser.parse_args()

    if args.only_sources and not args.dry_run:
        notice("--only-source implies --dry-run: partial snapshots are never recorded")
        args.dry_run = True

    config = load_basket_config(args.config)
    now = utc_now()
    # Hour-vocabulary lane: marks normalize to minute-of-day for the
    # shared slot machinery (15-min cadence re-base 2026-08-27); keys keep
    # the legacy hour tokens, snapshot bytes are unchanged.
    grid_minutes = [int(h) * 60 for h in config["capture_slots_utc"]]
    canonical_cfg = config.get("canonical_slot_utc")
    canonical_minute = None if canonical_cfg is None else int(canonical_cfg) * 60
    slot_date, slot_minute = current_slot(now, grid_minutes)
    slot_hour = slot_minute // 60
    canonical = is_canonical(slot_minute, canonical_minute)
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
        prev_present = slot_minutes_present(
            client, bucket, prefix=prefix, day=prev_day
        )
        prev_missing = sorted(
            m // 60 for m in set(grid_minutes) - prev_present
        )
        previous_day_empty = not prev_present
        if previous_day_empty:
            warn(
                f"index basket: previous day {prev_day} has ZERO stored "
                "snapshots — that history is permanently lost (missed-day "
                "alarm; expected once on the lane's first day)"
            )
        elif prev_missing:
            canonical_note = (
                " including the CANONICAL slot"
                if canonical_cfg in prev_missing
                else ""
            )
            warn(
                f"index basket: previous day {prev_day} missed slot(s) "
                f"{prev_missing}{canonical_note} — those observations are "
                "permanently lost (partial-miss alarm)"
            )
        if slot_already_captured(
            client,
            bucket,
            prefix=prefix,
            day=slot_date,
            minute_of_day=slot_minute,
        ):
            if args.force:
                notice(
                    f"slot {slot_date} {slot_hour:02d}:00Z already captured — "
                    "--force records another snapshot"
                )
            else:
                notice(
                    f"index basket slot {slot_date} {slot_hour:02d}:00Z already "
                    "captured — nothing to do"
                )
                return 0
        elif args.slot_gated and now.hour not in config["capture_slots_utc"]:
            # Between slots with the current slot not yet captured: capture
            # now (a late fill within the window beats a hole), but say so —
            # the timestamp records the truth either way.
            notice(
                f"claiming slot {slot_date} {slot_hour:02d}:00Z at "
                f"{now.strftime('%H:%M')}Z (first firing since the mark)"
            )

    run_id = make_run_id(now)
    # Recorded after the mark hour (a later firing filled the slot)?
    # late = recorded outside the mark's own wall-clock hour (the
    # original now.hour == slot_hour rule, expressed on the shared
    # minute machinery -- byte-identical for this hour-grid lane).
    late_fill = not (
        now.date() == slot_date
        and slot_minute <= (now.hour * 60 + now.minute) < slot_minute + 60
    )
    results = collect_all(config, only=set(args.only_sources or []) or None)
    payload = build_capture_snapshot(
        config=config,
        source_results=results,
        captured_at=now,
        run_id=run_id,
        slot_date=slot_date,
        slot_hour_utc=slot_hour,
        canonical=canonical,
        capturer={
            "job": "github-actions" if _in_actions() else os.environ.get(
                "BASKET_CAPTURER_JOB", "manual"
            ),
            # The capture host is deliberately not recorded; the field
            # stays for snapshot shape.
            "hostname": None,
            "git_sha": os.environ.get("GITHUB_SHA"),
            "version": CAPTURER_VERSION,
        },
        previous_day_empty=previous_day_empty,
        late_fill=late_fill,
    )

    # Capture screens: delta report (book vs same-machine) + the
    # corroboration-gated jump quarantine, evaluated against the previous
    # stored snapshot. Flags only, via the implausible machinery — raw
    # prices stay recorded. Fail-open: no readable reference -> report-only.
    # Skipped in dry/local runs (no bucket to reference).
    if client is not None:
        # The WHOLE screens block is fail-open: neither a broken reference
        # nor a screen bug may ever cost the slot — an unclaimed slot would
        # re-read the same broken reference every 30 minutes and dark the
        # lane until someone heals the pointer.
        try:
            reference = None
            pointer_raw = get_object_bytes(
                client, bucket, latest_pointer_key(prefix)
            )
            if pointer_raw is not None:
                snapshot_key = json.loads(pointer_raw).get("snapshot_key")
                snap_raw = (
                    get_object_bytes(client, bucket, snapshot_key)
                    if snapshot_key
                    else None
                )
                if snap_raw is not None:
                    reference = json.loads(snap_raw)
            screen_report = apply_jump_screen(payload, reference, config=config)
            # ref_label embeds the reference's capture_date — bucket JSON,
            # i.e. remote-derived: clean before it reaches the job log.
            ref_label = _clean(screen_report["reference"] or "no reference", 40)
            for d in screen_report["deltas"]:
                book = (
                    f"{d['book_pct']:+.1f}%"
                    if d["book_pct"] is not None
                    else "n/a"
                )
                same = (
                    f"{d['same_machine_pct']:+.1f}%"
                    if d["same_machine_pct"] is not None
                    else "n/a"
                )
                note = f"  ({_clean(d['note'], 60)})" if d["note"] else ""
                print(
                    f"  delta {config.get('target_sku', 'B300')} {d['source_id']:<14} "
                    f"book {book:>8}  same-machine {same:>8}  vs {ref_label}{note}"
                )
            for q in screen_report["quarantined"]:
                warn(
                    f"L5 QUARANTINE {q['source_id']}: {config.get('target_sku', 'B300')} "
                    f"print moved {q['book_pct']:+.1f}% vs {ref_label} with only "
                    f"{q['corroborators']} corroborating source(s) — prints "
                    "flagged implausible for THIS capture (raw values recorded)"
                )
            if screen_report.get("quarantine_skipped"):
                warn(
                    "capture screens: "
                    + _clean(screen_report["quarantine_skipped"], 200)
                )
            if screen_report["quarantined"]:
                # Basis pairs were derived pre-screen; a quarantined print
                # must not feed the recorded B300:B200 ratio either.
                payload["basis_pairs"] = derive_basis_pairs(payload["sources"])
        except Exception as exc:  # noqa: BLE001 — screens must never cost the capture
            warn(
                f"capture screens failed ({_clean(exc)}) — capture "
                "proceeds unscreened"
            )

    print(
        f"index basket capture: {payload['capture_date']} "
        f"slot {slot_hour:02d}:00Z run {run_id}"
        f"{' (canonical)' if canonical else ''}"
    )
    print_summary(payload)
    if args.json:
        print(json.dumps(payload, indent=2))

    if not payload["sources_ok"]:
        # Zero usable prints: nothing worth recording, and the job
        # wrapper should say so loudly.
        error("every implemented source failed — snapshot NOT recorded")
        return 1

    basket_ok = payload["basket_sources_ok"]
    basket_role = config.get("basket_role", "b300_basket")
    n_basket = sum(1 for s in config["sources"] if s["role"] == basket_role)
    min_claim = int(config.get("min_basket_sources_to_claim", 1))
    print(
        f"basket coverage: {len(basket_ok)}/{n_basket} constituents ok "
        f"(minimum {min_claim} to claim the slot)"
    )
    if args.dry_run:
        notice("dry run — nothing written")
        return 0

    if not args.only_sources and len(basket_ok) < min_claim:
        # A thin snapshot must never CLAIM the slot —
        # recording it would stop every later firing from retrying while
        # most of the basket is dark (and earliest-key-wins would make the
        # thin snapshot the one a reader sees). Dry runs are exempt above:
        # they record nothing, so there is nothing to gate.
        error(
            f"only {len(basket_ok)}/{n_basket} {basket_role} constituents "
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
        print(f"pointer:  {prefix}/latest.json published_at={outcome['pointer']['published_at']}")
    if payload["sources_failed"]:
        warn(
            "index basket recorded with failed sources: "
            + ", ".join(payload["sources_failed"])
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
