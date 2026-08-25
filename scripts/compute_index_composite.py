#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Compute daily index composites from stored basket raws.

One CLI for every basket lane: the config (default config/index_basket.json,
overridden by --config or BASKET_CONFIG_PATH) supplies keyspace,
methodology, constituents, and the lane
knobs. Replay discipline:

  - **Published days are the authority.** The raw store for a published day
    can legitimately GROW afterwards (a late slot upload racing the compute,
    an earlier-run_id snapshot landing after a later one, an ECB rate
    backfilled after an outage). Re-deriving published days from raw would
    silently rewrite the filter history the immutable series was built on —
    so replay advances each source's window from the PUBLISHED composite's
    recorded prints, never from raw. This is rule R4's "never revised"
    applied to history itself.
  - **Unpublished days derive from raw** (first publication), in date order
    from ``genesis_date`` — filter acceptance is order-dependent and
    containers keep no state.
  - **Drift detection**: for every published day the raw store is still
    compared against the artifact; divergence prints a loud DRIFT warning
    (the composite stands — immutable — but a human should know the raw
    store now tells a different story).
  - **Params pinning (rule D2)**: published days advance the
    filter windows under the ARTIFACT's own embedded calc_params, never
    the live config; and before any NEW day publishes, the live config's
    calc_params are compared against the last published artifact's — any
    drifted key (beyond manual_exclusions, which has its own check) errors
    loudly and refuses to extend the series (mint instead).

Eager compute (rule R4): day D publishes at the first run where D's
canonical (16:00Z) snapshot exists; after the canonical window closes
(D 22:00Z) the nearest existing slot is promoted; a day with no snapshots
records an explicit basket_dark/day_missed artifact once fully closed
(D+1 04:00Z).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpu_index.common.bucket import get_object_bytes, put_bytes  # noqa: E402
from gpu_index.common.jsondiff import field_diffs  # noqa: E402
from gpu_index.index.composite import (  # noqa: E402
    DEFAULT_FILTER_TERMS,
    advance_window,
    calc_params,
    compute_day,
    filter_observation,
    resolve_daily_print,
    resolve_slot_prints,
    select_slot_snapshot,
)
from gpu_index.index.config import load_basket_config  # noqa: E402
from gpu_index.index.fx import ensure_rates, load_stored_rates  # noqa: E402
from gpu_index.index.weights import advance_weight_state, new_weight_state  # noqa: E402
from gpu_index.common.slots import (  # noqa: E402
    latest_pointer_key,
    snapshot_key,
    utc_now,
)
from gpu_index.common.store import (  # noqa: E402
    BucketConfig,
    composite_key,
    get_composite,
    make_client,
    make_run_id,
    read_day_snapshots,
    snapshot_bytes,
    upload_composite,
)


def _in_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


# Artifact-derived strings (e.g. verdict currency labels) must never carry
# GH workflow-command sequences — under GITHUB_ACTIONS a smuggled newline +
# '::' would inject a workflow command into the job. Same rule as
# basket.sources._VAST_LOG_CLEAN_RE: control chars become spaces.
_LOG_CLEAN_RE = re.compile(r"[\r\n\x00-\x1f]")


def _log_clean(msg: str) -> str:
    return _LOG_CLEAN_RE.sub(" ", str(msg))


def notice(msg: str) -> None:
    msg = _log_clean(msg)
    print(f"::notice::{msg}" if _in_actions() else f"NOTICE: {msg}")


def warn(msg: str) -> None:
    msg = _log_clean(msg)
    print(f"::warning::{msg}" if _in_actions() else f"WARNING: {msg}")


def error(msg: str) -> None:
    msg = _log_clean(msg)
    print(f"::error::{msg}" if _in_actions() else f"ERROR: {msg}")


def publish_report(
    client,
    bucket: str,
    *,
    prefix: str,
    methodology_id: str,
    genesis: date,
    now: datetime,
    basket_label: str = "B300 index basket",
) -> str:
    """Render + PUT the HTML dashboard to the ONE mutable key
    under index/: ``<prefix>/report/index.html``, overwritten every firing
    (Cache-Control: no-store keeps the signed-URL bookmark current).

    Called warn-only from main(): any failure here — including the import —
    must never fail composite publishing, so the renderer import stays
    inside this function.
    """
    from gpu_index.index.report import REPORT_WINDOW_DAYS, render_report

    pointer_raw = get_object_bytes(client, bucket, latest_pointer_key(prefix))
    pointer = json.loads(pointer_raw) if pointer_raw is not None else None
    latest_snapshot = None
    if pointer and pointer.get("snapshot_key"):
        snapshot_raw = get_object_bytes(client, bucket, pointer["snapshot_key"])
        if snapshot_raw is not None:
            latest_snapshot = json.loads(snapshot_raw)

    today = now.date()
    start = max(genesis, today - timedelta(days=REPORT_WINDOW_DAYS - 1))
    composites_by_date = {}
    day = start
    while day <= today:
        stored = get_composite(
            client, bucket, prefix=prefix, methodology_id=methodology_id,
            day=day.isoformat(),
        )
        if stored is not None:
            composites_by_date[day.isoformat()] = stored
        day += timedelta(days=1)

    html = render_report(
        pointer=pointer,
        latest_snapshot=latest_snapshot,
        composites_by_date=composites_by_date,
        now=now,
        basket_label=basket_label,
    )
    key = f"{prefix}/report/index.html"
    put_bytes(
        client,
        bucket,
        key,
        html.encode("utf-8"),
        content_type="text/html; charset=utf-8",
        cache_control="no-store",
    )
    return key


def _day_fully_closed(day: date, slots, now: datetime) -> bool:
    """No further snapshot for <day> can appear: its last slot's window ends
    at the first mark of the NEXT day."""
    close_at = datetime.combine(
        day + timedelta(days=1), time(sorted(slots)[0]), tzinfo=timezone.utc
    )
    return now >= close_at


def _canonical_window_closed(
    day: date, canonical_hour: int, slots, now: datetime
) -> bool:
    """The canonical slot's late-fill window ends at the NEXT slot mark."""
    later = [h for h in sorted(slots) if h > canonical_hour]
    if not later:
        return _day_fully_closed(day, slots, now)
    return now >= datetime.combine(day, time(later[0]), tzinfo=timezone.utc)


def advance_windows_from_published(
    stored: dict,
    window_history: dict,
    window_currencies: dict,
    pending_currencies: dict,
) -> None:
    """Advance the replay's filter state from a PUBLISHED day's artifact.

    The filter terms come from the ARTIFACT's own embedded calc_params
    (rule D2: a legacy artifact without the key replays under "usd"
    terms) — never from the live config, so a config edit can never
    silently re-derive the windows published history was built on. Each
    stored chosen print runs through the same filter_observation +
    advance_window state machine compute_day uses (untrusted prints and
    unconfirmed cross-currency prints never entered a window on their day,
    and they don't here either), so full replays and mid-series restarts
    rebuild identical state."""
    artifact_params = stored.get("calc_params") or {}
    filter_terms = str(artifact_params.get("filter_terms", DEFAULT_FILTER_TERMS))
    for entry in stored.get("sources", []):
        if entry.get("chosen") and entry.get("filter"):
            advance_window(
                window_history,
                window_currencies,
                pending_currencies,
                entry["source_id"],
                filter_observation(entry["chosen"], filter_terms=filter_terms),
            )


def advance_weight_state_from_published(stored: dict, weight_state: dict) -> None:
    """Advance the calc_v5 weight state from a PUBLISHED day's artifact.

    R-slots makes this a pure read of pinned facts: the artifact's
    weight_calc block carries the just-closed prior day's slot prints, the
    day's rounded weight vector, and the mode latch VERBATIM — nothing is
    re-derived from raw (which can legitimately grow after publication),
    so a replay carries exactly the series the live run advanced. Legacy
    artifacts (no dynamic_weights in calc_params) advance nothing.
    """
    artifact_params = stored.get("calc_params") or {}
    if "dynamic_weights" not in artifact_params:
        return
    advance_weight_state(
        weight_state,
        day=stored["date"],
        weight_block=stored.get("weight_calc"),
    )


def detect_drift(stored, selection, snapshots, *, day_str, params, fx_records):
    """How the raw store NOW disagrees with the published artifact.

    FX-converted sources compare in NATIVE terms: the day's real ECB rate
    landing after a walked-back publish is routine (R2), and comparing USD
    values would page on it 48x/day forever — the artifact records the rate
    it used, and that record stands.
    """
    msgs = []
    if stored.get("day_missed") and snapshots:
        msgs.append(
            "published as day_missed but the raw store now holds snapshots"
        )
        return msgs
    if not stored.get("day_missed") and not snapshots:
        # THE tripwire case: a published day whose raw evidence vanished.
        msgs.append("raw store now holds NO snapshots for this published day")
        return msgs
    if selection is None:
        return msgs
    snapshot, substituted = selection
    if stored.get("snapshot_run_id") != snapshot.get("run_id"):
        msgs.append(
            f"selection changed: published from run "
            f"{stored.get('snapshot_run_id')}, raw store now yields "
            f"{snapshot.get('run_id')}"
        )
    if stored.get("substituted_from_slot") != substituted:
        msgs.append(
            f"slot substitution changed: published "
            f"{stored.get('substituted_from_slot')} vs raw-store {substituted}"
        )
    entries = {s["source_id"]: s for s in snapshot.get("sources", [])}
    for entry in stored.get("sources", []):
        source_id = entry["source_id"]
        if entry.get("status") == "manually_excluded":
            # calc_v2: the artifact diverging from raw is the POINT of an
            # exclusion — warning on it 48x/day would bury real drift.
            continue
        stored_chosen = entry.get("chosen") or {}
        current_entry = entries.get(source_id)
        # Same resolver as compute_day: a statistic-priced source (the
        # order-book vast rule) re-prices by its statistic here, or every published
        # day would "drift" against the min rule forever.
        current = resolve_daily_print(
            current_entry,
            source_id=source_id,
            params=params,
            sku=params["target_sku"],
            day=day_str,
            fx_records=fx_records,
        )
        if current is None or current.get("fx_unavailable"):
            current = {}
        if stored_chosen.get("fx_rate") is not None:
            published_cmp = (
                stored_chosen.get("native_per_gpu_hr"),
                stored_chosen.get("currency"),
            )
            current_cmp = (
                current.get("native_per_gpu_hr"),
                current.get("currency"),
            )
            if published_cmp != current_cmp:
                msgs.append(
                    f"{source_id}: published native print {published_cmp} vs "
                    f"raw-store {current_cmp}"
                )
        else:
            published = stored_chosen.get("usd_per_gpu_hr")
            current_price = current.get("usd_per_gpu_hr")
            if published != current_price:
                msgs.append(
                    f"{source_id}: published print {published} vs raw-store "
                    f"{current_price}"
                )
    return msgs


# ------------------------------------------------------- verify-published


def verify_published(args, config) -> int:
    """--verify-published <date>: recompute one PUBLISHED day byte-for-byte
    from the record and compare against the stored artifact. Reads only —
    published days are never revised (rule R4), and this mode holds to that
    by construction: no upload, no pointer move, no FX persist, no report.

    Determinism comes from pinning everything to the record, exactly the
    replay semantics the pipeline already defines:
      - filter windows / currency streaks / weight state advance from the
        PUBLISHED artifacts genesis..day-1 (never from raw, which can
        legitimately grow after publication);
      - the day's snapshot is the artifact's own pinned choice
        (snapshot_run_id + substituted_from_slot fetched by exact key),
        never re-selected from the raw store;
      - under dynamic weights the prior day's slot prints come from the
        artifact's pinned weight_calc.slot_prints, never re-derived;
      - FX uses the stored append-only rate records only (no live feed):
        rates for a published date never change, but a rate that was
        WALKED BACK at publish time and backfilled later is the one known
        benign mismatch cause — the diff paths name the fx fields.
    """
    try:
        day = date.fromisoformat(args.verify_published)
    except ValueError:
        error(f"--verify-published {args.verify_published!r}: not a YYYY-MM-DD date")
        return 1
    day_str = day.isoformat()
    params = calc_params(config)
    prefix = config["bucket_prefix"]
    canonical_hour = int(config.get("canonical_slot_utc", 16))
    methodology_id = params["methodology_id"]
    genesis = date.fromisoformat(config["genesis_date"])

    bucket_config = BucketConfig.from_env()
    client = make_client(bucket_config)
    bucket = bucket_config.bucket

    artifact_key = composite_key(prefix, methodology_id, day_str)
    stored_raw = get_object_bytes(client, bucket, artifact_key)
    if stored_raw is None:
        error(
            f"{day_str} is not published under {methodology_id} "
            f"(no {artifact_key}) — derive it with --date {day_str} "
            "--dry-run instead"
        )
        return 1
    stored = json.loads(stored_raw)
    stored_sha = hashlib.sha256(stored_raw).hexdigest()
    index = stored.get("index") or {}
    print(
        f"verify-published {day_str} (methodology {methodology_id})\n"
        f"artifact: {artifact_key}\n"
        "published: "
        + (
            "BASKET_DARK"
            if stored.get("basket_dark")
            else f"{index.get('value_usd_gpu_hr'):.4f} $/GPU-hr "
            f"({index.get('sources_used_count', 0)} sources)"
        )
        + (" [day_missed]" if stored.get("day_missed") else "")
    )

    # Rule D2 heads-up: the recompute runs under the LIVE config (the same
    # config the pipeline extends the series with), so params drift from
    # the artifact guarantees a byte mismatch that is config drift, not
    # tampering — say so before the verdict.
    live_embedded = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
    }
    stored_params = stored.get("calc_params") or {}
    drifted = sorted(
        k
        for k in set(live_embedded) | set(stored_params)
        if live_embedded.get(k) != stored_params.get(k)
    )
    if drifted:
        warn(
            f"live config calc_params drift from the artifact's on key(s) "
            f"{drifted} — a MISMATCH below reflects config drift (rule D2: "
            "mint a new methodology_id), not necessarily tampering"
        )

    # FX: stored append-only records only — the record, not the live feed.
    if params.get("fx_lane", "ecb") == "none":
        fx_records: dict = {}
    else:
        fx_records = load_stored_rates(client, bucket, prefix=prefix)

    # Replay state advance from published artifacts, genesis..day-1 — the
    # exact pin-to-published walk main() does, minus raw reads and writes.
    window_history: dict = {}
    window_currencies: dict = {}
    pending_currencies: dict = {}
    weight_state: dict = new_weight_state()
    prior = genesis
    while prior < day:
        prior_stored = get_composite(
            client,
            bucket,
            prefix=prefix,
            methodology_id=methodology_id,
            day=prior.isoformat(),
        )
        if prior_stored is None:
            warn(
                f"{prior.isoformat()} is unpublished below the target day — "
                "the series publishes in order, so replay state may be "
                "incomplete (expect MISMATCH until the gap is explained)"
            )
        else:
            advance_windows_from_published(
                prior_stored, window_history, window_currencies,
                pending_currencies,
            )
            advance_weight_state_from_published(prior_stored, weight_state)
        prior += timedelta(days=1)

    # The day's snapshot: the artifact's own pinned choice, by exact key.
    substituted = stored.get("substituted_from_slot")
    if stored.get("day_missed"):
        snapshot = None
    else:
        run_id = stored.get("snapshot_run_id")
        slot_hour = canonical_hour if substituted is None else int(substituted)
        pinned_key = snapshot_key(prefix, day, slot_hour, run_id)
        snapshot_raw = get_object_bytes(client, bucket, pinned_key)
        if snapshot_raw is None:
            error(
                f"MISMATCH {day_str}: the artifact's pinned snapshot "
                f"{pinned_key} is MISSING from the raw store — cannot "
                "recompute; the raw evidence for this published day is gone "
                "(the drift tripwire case)"
            )
            return 1
        snapshot = json.loads(snapshot_raw)

    # Dynamic weights: the prior day's slot prints are pinned in the
    # artifact (R-slots) — replay never re-derives them from raw.
    prior_slot_prints = None
    if "dynamic_weights" in params:
        weight_calc = stored.get("weight_calc") or {}
        slot_block = weight_calc.get("slot_prints") or {}
        prior_slot_prints = dict(slot_block.get("slots") or {})

    payload = compute_day(
        config=config,
        day=day_str,
        snapshot=snapshot,
        substituted_from=substituted,
        window_history=window_history,
        fx_records=fx_records,
        window_currencies=window_currencies,
        pending_currencies=pending_currencies,
        weight_state=weight_state,
        prior_slot_prints=prior_slot_prints,
    )
    recomputed_raw = snapshot_bytes(payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    if recomputed_raw == stored_raw:
        print(f"MATCH sha256={stored_sha}")
        return 0

    recomputed_sha = hashlib.sha256(recomputed_raw).hexdigest()
    diffs = field_diffs(stored, payload)
    print(
        f"MISMATCH {day_str}: published sha256={stored_sha} vs "
        f"recomputed sha256={recomputed_sha}"
    )
    if diffs:
        for diff in diffs:
            print(f"  {_log_clean(diff)}")
        if any(".fx_" in d or "fx_as_of" in d for d in diffs):
            notice(
                "diff paths touch fx fields — a rate walked back at publish "
                "time and backfilled later (rule R2) is the known benign "
                "cause; the artifact's recorded rate stands"
            )
    else:
        print(
            "  JSON content is identical; the stored bytes are not the "
            "canonical serialization (sorted keys, 2-space indent, trailing "
            "newline) — the artifact bytes were rewritten"
        )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute daily B300 index composites from stored snapshots."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Compute every missing, currently-computable day from genesis onward",
    )
    parser.add_argument("--date", help="Compute a single day (YYYY-MM-DD)")
    parser.add_argument("--from", dest="from_date", help="Range start (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", help="Range end (YYYY-MM-DD)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and print; write nothing"
    )
    parser.add_argument(
        "--verify-published",
        metavar="DATE",
        help=(
            "Recompute one PUBLISHED day byte-for-byte from the record "
            "(prior artifacts + that day's pinned snapshot) and compare "
            "against the stored artifact — MATCH exits 0, MISMATCH exits 1. "
            "Reads only; writes nothing."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print full payload JSON")
    parser.add_argument("--config", type=Path, help="Override config/index_basket.json")
    args = parser.parse_args()

    if args.verify_published:
        if args.sync or args.date or args.from_date or args.to_date:
            parser.error(
                "--verify-published is its own mode — do not combine it "
                "with --sync/--date/--from/--to"
            )
    elif not (args.sync or args.date or (args.from_date and args.to_date)):
        parser.error(
            "pick a mode: --sync, --date, --from/--to, or --verify-published"
        )

    config = load_basket_config(args.config)

    if args.verify_published:
        return verify_published(args, config)
    params = calc_params(config)
    prefix = config["bucket_prefix"]
    slots = config["capture_slots_utc"]
    canonical_hour = int(config.get("canonical_slot_utc", 16))
    methodology_id = params["methodology_id"]
    genesis = date.fromisoformat(config["genesis_date"])
    now = utc_now()
    today = now.date()

    if args.date:
        targets = {date.fromisoformat(args.date)}
    elif args.from_date:
        start = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date)
        if start > end:
            error(f"--from {start} is after --to {end}")
            return 1
        targets = {start + timedelta(days=i) for i in range((end - start).days + 1)}
    else:
        targets = None  # --sync: any missing computable day

    if targets:
        out_of_range = sorted(t for t in targets if t < genesis or t > today)
        if out_of_range:
            # A typo'd backfill must never look like a successful no-op.
            error(
                f"target day(s) outside the replayable range "
                f"[{genesis}..{today}]: {[d.isoformat() for d in out_of_range]}"
            )
            return 1

    bucket_config = BucketConfig.from_env()
    client = make_client(bucket_config)
    bucket = bucket_config.bucket

    if params.get("fx_lane", "ecb") == "none":
        # USD-only basket: no ECB fetch, no fx/ keyspace
        # under this prefix. Any non-USD print that sneaks in is held out
        # loudly by the fail-closed conversion path, never guessed.
        fx_records: dict = {}
    else:
        fx_records = ensure_rates(
            client, bucket, prefix=prefix, persist=not args.dry_run
        )

    # Sequential replay from genesis. Published days advance history from
    # their own artifacts (pin-to-published); unpublished days derive from
    # the raw store; writes happen only for target days. window_currencies
    # (source -> window currency) and pending_currencies (source -> the D1
    # consecutive-new-currency confirmation streak) ride alongside
    # window_history across days; both reconstruct from published artifacts
    # alone, so replays and mid-series restarts are deterministic.
    window_history: dict = {}
    window_currencies: dict = {}
    pending_currencies: dict = {}
    # calc_v5 (dynamic_weights): the weight series state — trusted daily
    # prints + each day's pinned rounded vector + the fallback->dynamic
    # mode latch. Advanced from published artifacts for published days and
    # by compute_day for new ones; harmlessly inert on legacy configs.
    weight_state: dict = new_weight_state()
    # Raw-snapshot cache for the R-slots prior-day read (one day of lag).
    last_raw_day: date | None = None
    last_raw_snapshots: dict = {}
    # Rule D2: published days pin their calc_params. Before any NEW day
    # publishes, the live config's params are compared against the LAST
    # published artifact's; any drifted key (beyond manual_exclusions,
    # which has its own check above) refuses to extend the series.
    last_published_params: dict | None = None
    params_drift: list = []
    drift_checked = False
    wrote = 0
    exit_code = 0
    skipped_unpublished = False
    exclusion_conflict = False
    day = genesis
    while day <= today:
        day_str = day.isoformat()
        stored = get_composite(
            client, bucket, prefix=prefix, methodology_id=methodology_id, day=day_str
        )
        if stored is not None:
            # calc_v2: the pin-published-days rule made MECHANICAL. A
            # published day pins its exclusion set; a config that now says
            # otherwise (an exclusion added for a published day, or one
            # removed after publication) must never keep extending the
            # series silently — new days would embed calc_params that
            # contradict published artifacts under the same methodology_id.
            stored_excluded = {
                s["source_id"]
                for s in stored.get("sources", [])
                if s.get("status") == "manually_excluded"
            }
            config_excluded = {
                e["source_id"]
                for e in params["manual_exclusions"]
                if e["date"] == day_str
            }
            if stored_excluded != config_excluded:
                error(
                    f"{day_str}: manual_exclusions contradict the PUBLISHED "
                    f"artifact (stored excluded {sorted(stored_excluded)} vs "
                    f"config {sorted(config_excluded)}) — published days pin "
                    "their exclusion set; mint a new methodology_id instead "
                    "of editing it"
                )
                exclusion_conflict = True
                exit_code = 1  # loud even when there is nothing new to publish
            advance_windows_from_published(
                stored, window_history, window_currencies, pending_currencies
            )
            advance_weight_state_from_published(stored, weight_state)
            last_published_params = stored.get("calc_params")
            # Drift scanning is bounded (48 firings/day forever would turn
            # an unbounded genesis scan into the job's dominant cost);
            # history-advance above stays one GET per day regardless.
            if (today - day).days <= params["drift_scan_days"]:
                snapshots = read_day_snapshots(
                    client, bucket, prefix=prefix, day=day
                )
                last_raw_day, last_raw_snapshots = day, snapshots
                selection = select_slot_snapshot(
                    snapshots,
                    canonical_hour=canonical_hour,
                    window_closed=_canonical_window_closed(
                        day, canonical_hour, slots, now
                    ),
                    tie_break=params["promote_tie_break"],
                )
                for msg in detect_drift(
                    stored,
                    selection,
                    snapshots,
                    day_str=day_str,
                    params=params,
                    fx_records=fx_records,
                ):
                    warn(
                        f"DRIFT {day_str}: {msg} — the published composite "
                        "stands (immutable); the raw store now diverges from it"
                    )
            day += timedelta(days=1)
            continue

        snapshots = read_day_snapshots(client, bucket, prefix=prefix, day=day)
        selection = select_slot_snapshot(
            snapshots,
            canonical_hour=canonical_hour,
            window_closed=_canonical_window_closed(day, canonical_hour, slots, now),
            tie_break=params["promote_tie_break"],
        )
        computable = selection is not None or (
            not snapshots and _day_fully_closed(day, slots, now)
        )
        snapshot, substituted = selection if selection else (None, None)
        is_target = (targets is None) or (day in targets)

        # R-slots: a NEW day's weights consume the just-closed prior day's
        # per-slot prints, resolved from raw HERE (compute_day pins them in
        # the artifact so no replay ever re-derives them). The genesis-
        # replay loop computes consecutive days, so the cache turns the
        # extra read into a no-op for every day after the first.
        prior_slot_prints = None
        if computable and "dynamic_weights" in params:
            prior = day - timedelta(days=1)
            if prior < genesis:
                prior_slot_prints = {}
            else:
                if last_raw_day == prior:
                    prior_snapshots = last_raw_snapshots
                else:
                    prior_snapshots = read_day_snapshots(
                        client, bucket, prefix=prefix, day=prior
                    )
                prior_slot_prints = resolve_slot_prints(
                    prior_snapshots,
                    config=config,
                    day=prior.isoformat(),
                    fx_records=fx_records,
                )
        last_raw_day, last_raw_snapshots = day, snapshots

        if computable:
            if not drift_checked and last_published_params is not None:
                # Rule D2, checked once at the first NEW day: the mint
                # rule made mechanical for EVERY param — a live config
                # whose calc_params drift from the last published
                # artifact's must never extend the series.
                drift_checked = True
                live_embedded = {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in params.items()
                }
                keys = set(live_embedded) | set(last_published_params)
                keys.discard("manual_exclusions")  # has its own check above
                params_drift = sorted(
                    k
                    for k in keys
                    if live_embedded.get(k) != last_published_params.get(k)
                )
                if params_drift:
                    error(
                        f"calc_params drift vs the last published "
                        f"{methodology_id} artifact on key(s) "
                        f"{params_drift} — published days pin their "
                        "params; mint a new methodology_id instead of "
                        "editing them"
                    )
                    exit_code = 1
            payload = compute_day(
                config=config,
                day=day_str,
                snapshot=snapshot,
                substituted_from=substituted,
                window_history=window_history,
                fx_records=fx_records,
                window_currencies=window_currencies,
                pending_currencies=pending_currencies,
                weight_state=weight_state,
                prior_slot_prints=prior_slot_prints,
            )
            weight_calc = payload.get("weight_calc") or {}
            if weight_calc.get("switched_on"):
                # The planned methodology transition, loud but not red: the
                # first day the switch quorum held — the series leaves the
                # fallback (opening) weights permanently.
                warn(
                    f"{day_str}: dynamic weighting SWITCHED ON — weights "
                    "leave the opening-weights fallback for predictive "
                    "allocation from this day forward"
                )
            if weight_calc.get("degenerate_allocation"):
                warn(
                    f"{day_str}: weight bounds degenerate for the day's "
                    f"eligible set — "
                    f"{weight_calc.get('degenerate_allocation')} weights "
                    f"published ({weight_calc.get('fallback_reason')})"
                )
            if weight_calc and weight_calc.get("mode") == "fallback":
                # R-quorum-v2 visibility: the switch is count-based —
                # show the defined-vs-required tally and
                # name the unscored sources on every computed day so warm-up
                # progress is a read, not an investigation.
                eligible_ids = sorted(weight_calc.get("weights") or {})
                score_blocks = weight_calc.get("sources") or {}
                defined = [
                    sid
                    for sid in eligible_ids
                    if (score_blocks.get(sid) or {}).get("Q") is not None
                ]
                undefined = sorted(
                    sid
                    for sid, entry in score_blocks.items()
                    if entry.get("Q") is None
                )
                quorum = (params.get("dynamic_weights") or {}).get(
                    "switch_min_eligible", 1
                )
                notice(
                    f"{day_str}: dynamic switch pending — "
                    f"{len(defined)}/{quorum} eligible sources have a "
                    f"defined Q"
                    + (
                        f"; undefined: {', '.join(undefined)}"
                        if undefined
                        else ""
                    )
                )
                # Fallback-parity tripwire (mirror-drift class): fallback
                # mode CLAIMS byte-identical index math to the frozen
                # prior-methodology series, but this series replays from
                # the CURRENT raw store, which can legitimately have grown
                # since the frozen day published. Compare and warn loudly
                # (red, still publishes — the currency-anomaly posture):
                # a mismatch means the raw store tells a different story
                # now, and a human should rule on it before promoting.
                parity_id = config.get("fallback_parity_methodology_id")
                if parity_id:
                    frozen = get_composite(
                        client,
                        bucket,
                        prefix=prefix,
                        methodology_id=str(parity_id),
                        day=day_str,
                    )
                    if frozen is not None:
                        ours = (payload.get("index") or {}).get(
                            "value_usd_gpu_hr"
                        )
                        theirs = (frozen.get("index") or {}).get(
                            "value_usd_gpu_hr"
                        )
                        if ours != theirs:
                            warn(
                                f"{day_str}: FALLBACK PARITY MISMATCH — "
                                f"{methodology_id} computes {ours} but the "
                                f"frozen {parity_id} artifact published "
                                f"{theirs}; the raw store has drifted since "
                                "the frozen day published (its artifact "
                                "stands; this series replays from raw)"
                            )
                            exit_code = 1
            for entry in payload["sources"]:
                verdict = entry.get("filter") or {}
                if verdict.get("untrusted_currency"):
                    # Rule D1: fail-closed and LOUD — held out, window
                    # preserved, firing reddened (still publishes).
                    warn(
                        f"{day_str}: {entry['source_id']} untrusted "
                        f"currency label {verdict.get('currency_label')!r} "
                        "— print held out fail-closed; window preserved"
                    )
                    exit_code = 1
                elif verdict.get("currency_mismatch"):
                    warn(
                        f"{day_str}: {entry['source_id']} recorded currency "
                        f"changed {verdict.get('window_currency')} -> "
                        f"{verdict.get('currency')} — held out "
                        f"({verdict.get('pending_count')}/"
                        f"{verdict.get('confirm_after')} toward "
                        "confirmation); window preserved"
                    )
                    exit_code = 1
                elif verdict.get("currency_confirmed"):
                    # The change is now real: accepted, old window
                    # discarded, new window seeded. Loud but not red — the
                    # preceding mismatch days already reddened firings.
                    warn(
                        f"{day_str}: {entry['source_id']} currency change "
                        f"CONFIRMED {verdict.get('window_currency')} -> "
                        f"{verdict.get('currency')} — window reseeded from "
                        f"the {verdict.get('n_history')} pending prints "
                        "(warm-up restarts)"
                    )
            wrote_this_day = False
            if is_target:
                index = payload.get("index") or {}
                print(
                    f"{day_str}: "
                    + (
                        "BASKET_DARK"
                        if payload["basket_dark"]
                        else f"{index['value_usd_gpu_hr']:.4f} $/GPU-hr "
                        f"({index['sources_used_count']} sources)"
                    )
                    + (" [day_missed]" if payload["day_missed"] else "")
                    + (
                        f" [substituted from slot {substituted:02d}]"
                        if substituted is not None
                        else ""
                    )
                )
                if args.json:
                    print(json.dumps(payload, indent=2))
                if not args.dry_run:
                    if exclusion_conflict:
                        # Refusing to extend the series is what makes the
                        # published-days-pin rule real: fix the config (or
                        # mint) before any new day publishes under it.
                        error(
                            f"{day_str}: NOT published — the config's "
                            "manual_exclusions contradict published "
                            f"{methodology_id} history (see errors above)"
                        )
                        exit_code = 1
                    elif params_drift:
                        # Rule D2: same refusal for ANY drifted param.
                        error(
                            f"{day_str}: NOT published — the live config's "
                            f"calc_params drift from published "
                            f"{methodology_id} history on key(s) "
                            f"{params_drift} (see error above)"
                        )
                        exit_code = 1
                    elif targets is not None and skipped_unpublished:
                        # Writing day D while an earlier computable day is
                        # unpublished would bake filter provenance that the
                        # eventually-published earlier artifacts may not
                        # reproduce (their raw can still grow) — publish in
                        # order instead.
                        error(
                            f"{day_str}: earlier unpublished computable "
                            "day(s) exist — publish in order (run --sync or "
                            "widen the range)"
                        )
                        exit_code = 1
                    else:
                        outcome = upload_composite(
                            client,
                            bucket,
                            payload,
                            prefix=prefix,
                            run_id=make_run_id(now),
                            now=now,
                        )
                        print(f"  -> {outcome['composite_key']}")
                        wrote += 1
                        wrote_this_day = True
            if not wrote_this_day and not args.dry_run:
                skipped_unpublished = True
        elif is_target and targets is not None:
            notice(f"{day_str}: not yet computable (canonical window still open)")
            exit_code = 1
        day += timedelta(days=1)

    print(f"composites written: {wrote} (methodology {methodology_id})")

    # Refresh the bucket dashboard after every real --sync. This
    # is dev-tooling on top of the published artifacts and is WARN-ONLY by
    # design — a report-render bug must never fail composite publishing.
    if args.sync and not args.dry_run:
        try:
            report_key = publish_report(
                client,
                bucket,
                prefix=prefix,
                methodology_id=methodology_id,
                genesis=genesis,
                now=now,
                basket_label=f"{config.get('target_sku', 'B300')} index basket",
            )
            notice(f"report published: {report_key}")
        except Exception as exc:  # noqa: BLE001 — warn-only by design
            warn(f"report render failed - composites unaffected: {exc}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
