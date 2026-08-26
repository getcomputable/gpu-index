#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Compute hourly PANEL index composites from the stored record
(METHODOLOGY.md sections 1, 2, 5, 6).

One CLI for every hourly lane: a REQUIRED --config path names a panel
config (gpu_index.index.panel_config), which supplies the lane's keyspace,
methodology, record sources + cutover, era grids, membership + screens,
and every calc knob. The published methodology documents are the record
of what each lane runs; this open pipeline runs the baked config files
under config/. No env-var config fallback exists on purpose (six lanes
share this entry point; a silently inherited default could run the wrong
lane's law).

Replay discipline -- the compute_index_composite.py doctrine re-minted
per OBSERVATION:

  - **Published observations are the authority.** The record for a
    published observation can legitimately GROW afterwards (a late slot
    upload racing the compute, an earlier-run_id snapshot landing after
    a later one, an ECB rate backfilled after an outage). Replay
    advances each source's filter window and the weight state from the
    PUBLISHED artifact's recorded prints and pinned weight_calc block,
    under the ARTIFACT's own embedded calc_params (rule D2) -- never
    from raw, never from the live config.
  - **Unpublished observations derive from the record** in stamp order:
    filter acceptance, attendance, and the weight-mode latch are all
    order-dependent, and containers keep no state. Publish-in-order is
    enforced: a target observation refuses to publish while an earlier
    computable observation is unpublished.
  - **Replay is BOUNDED (perf stage; amended into the mints before any
    observation published).** The run learns the published stamp set
    from ONE paginated LIST of the composite keyspace (never a
    per-stamp GET walk), takes frontier = the first scheduled stamp not
    yet published, and replays state from window_start =
    max(genesis, frontier - STATE_WINDOW_HOURS) where STATE_WINDOW_HOURS
    = dw history_days*24 + max(forward_horizons) + PRUNE_MARGIN_HOURS --
    exactly the weight state's prune bound, so every state the engine
    can consult (the 20-observation filter window, the 3-print currency
    streak, the jump-screen reference walk-back, attendance over
    history_hours, and the weight prices/vectors) rebuilds fully inside
    the window. Published artifacts INSIDE the window are GET+advanced
    exactly as before; the two pre-window facts that outlive the window
    -- the weight-mode latch and the D2 last-published-params baseline
    -- seed from ONE GET of the latest published artifact strictly
    before window_start. Steady-state cost is therefore O(window), not
    O(series age). The manual-exclusion pin check consequently runs
    over published artifacts WITHIN the window only (a documented
    loosening): older artifacts are immutable and their pins were
    enforced when the frontier passed them.
  - **Closure rule (section 2)**: an observation at scheduled stamp t
    is computable once its window CLOSES at the NEXT scheduled mark
    (era-aware; utc_now decides). There is NO slot promotion and NO
    eager compute: the slot's own snapshot prices the observation, and
    a scheduled hour with no snapshot publishes an explicit
    observation_missed artifact once closed -- never skipped, never
    interpolated, never substituted from a neighboring hour (the daily
    lane's R4 promotion has no meaning on a dense grid).
  - **False-missed guard (harden stage, adversarial F7)**: an
    observation_missed artifact is immutable, so before one publishes
    the slot keys are re-LISTed ONCE with a fresh call (never the run
    cache) -- a transient empty-Contents gateway blip must not pin a
    permanent false missed record; if the key appears on the confirming
    LIST the observation computes normally.
  - **Record quarantine (harden stage, adversarial F6)**: a top-level
    config key ``record_exclusions`` ([{date, hour, reason}]) names
    stamps whose stored record object must NEVER be read -- the escape
    hatch for a poisoned/unparseable snapshot that would otherwise
    crash every firing forever (publish-in-order blocks the lane behind
    it; earliest-key-wins means a later good object cannot shadow it).
    The check runs BEFORE any record read; the observation publishes an
    explicit record_quarantined artifact (index null, observation_missed
    FALSE -- bytes exist, they are quarantined, not missing). The key
    rides calc_params and pins per published observation exactly like
    manual_exclusions; quarantined artifacts are skipped by the drift
    scan (re-parsing the poison is the crash the quarantine fences off)
    and by the jump-reference walk-back (they carry no prints).
  - **Fences**: the D2 calc_params-drift refusal fires once at the
    first NEW observation (any drifted key beyond manual_exclusions /
    record_exclusions -- each pinned per observation by its own check --
    refuses to extend the series; mint instead); manual exclusions
    pin per published observation, hour-scoped entries pinning ONLY
    their observation and date-scoped entries pinning every observation
    of their date (design section 3 item 9).
  - **Drift scan**: published observations within the trailing
    ``drift_scan_observations`` scheduled stamps (a TOP-LEVEL panel
    config key -- operational, never in calc_params) are re-resolved
    from the record via the SAME panel resolver (screens + statistics +
    lowest-eligible) and compared -- warn-only, the artifact stands
    (immutable). FX-converted prints compare in NATIVE terms (a late-
    landing real ECB rate is routine and must not page 48x/day);
    statistic prints are USD by construction and compare in USD;
    manually-excluded seats are skipped (divergence is the point).
    GATED (perf stage): the scan re-reads the record for dozens of
    published observations, so it runs only on the DRIFT_SCAN_HOUR_UTC
    (16:00Z) firing -- one sweep per day, the daily lanes' cadence --
    or when --drift-scan forces it explicitly; every other firing
    advances state from published artifacts alone with zero record
    reads for published stamps.
  - **--max-observations N** is the per-run publish valve (section 5
    perf): a genesis backfill at 24 artifacts/day must not blow the
    job's timeout, so a run stops publishing after N artifacts and
    exits 0 with a notice; publish-in-order means the next firing
    continues the backlog exactly where this one stopped.

NO fallback-parity leg exists here, deliberately: the daily lanes'
``fallback_parity_methodology_id`` tripwire compares same-prefix
same-day artifacts against a frozen predecessor whose index math is
byte-identical in fallback mode. No such predecessor exists for an
hourly lane -- the daily and hourly grids are DIFFERENT INSTRUMENTS
(the hourly series advances filter windows and weights at every
observation where the daily advanced once per day, so values at a
shared wall-clock hour legitimately diverge; that divergence is rule
A1, the same doctrine as the v6 mint, not drift). Panel configs
therefore carry no fallback_parity key and this CLI runs no parity
compare. Mirror-drift protection for the hourly lanes is the D2 pin
plus the drift scan above.

No HTML report ships in v1 (design section 9): the frozen daily
dashboards remain; an hourly report is a follow-up.

FX: when calc.fx_lane == "ecb" the lane reads its OWN prefix's fx/
store (the migrated B300 lane thereby reuses its basket prefix's
existing records; fresh H-panel prefixes seed their own from the 90d
feed on their first run), BOUNDED (perf stage) to
[window_start_date - fx_max_staleness_days, today] -- record GETs never
scale with the fx/ store's age. The live ECB feed is fetched ONLY when
some unpublished observation's day cannot resolve a rate from stored
records after the standard walk-back (so a fully-covered run performs
zero FX egress); the fetched window persists append-only exactly as
before, and the fail-closed staleness fence is untouched. Record reads
are slot-granular: a day's snapshot keys LIST once and only the needed
hours GET, each parsed snapshot stripped to the panel's member entries
and in-sku-union rows before entering a small capped cache (memory
bounded on wide observatory days). fx_lane "none" performs zero FX
egress and prices no non-USD print, mechanically (the USD-only rule).

--verify-published <YYYY-MM-DDTHH> recomputes one PUBLISHED observation
byte-for-byte from the record (prior published artifacts advance the
replay state; the observation's snapshot is the artifact's own pinned
choice fetched by exact key) and compares against the stored artifact:
MATCH exits 0 with the sha256, MISMATCH exits 1 with field-path diffs.
Reads only, writes nothing -- the daily CLI's mode re-minted per
observation (see verify_published_observation).

--check-config validates the named config offline (loader + schedule +
resolved calc_params) and exits 0 without touching bucket credentials
-- the pre-merge sanity read for config-only PRs.

OPS RULE: NEVER run a manual --sync against an ARMED lane. The
scheduled panel-index job owns an armed lane's cadence, and its
no-concurrency fence covers only the job against itself -- a manual
sync from a workstation can race the job's run between the frontier
LIST and its publishes (two writers, one publish-in-order invariant).
Manual --sync is for unarmed lanes and pre-arming backfill only; to
intervene on an armed lane, disarm or suspend the job first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict, deque
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpu_index.index.composite import (  # noqa: E402
    DEFAULT_FILTER_TERMS,
    advance_window,
    filter_observation,
)
from gpu_index.index.fx import ensure_rates, load_stored_rates, rates_cover  # noqa: E402
from gpu_index.index.panel import (  # noqa: E402
    compile_screens,
    compute_observation,
    embedded_calc_params,
    exclusion_applies,
    jump_reference_prints,
    member_eligible_rows,
    panel_calc_params,
    record_exclusion_reason,
    record_source_for,
    resolve_member_print,
)
from gpu_index.index.panel_config import (  # noqa: E402
    PanelConfigError,
    load_panel_config,
    panel_schedule,
)
from gpu_index.index.panel_schedule import (  # noqa: E402
    PanelScheduleError,
    hour_iso_to_stamp,
    stamp_to_date_hour,
    stamp_to_hour_iso,
)
from gpu_index.common.jsondiff import field_diffs  # noqa: E402
from gpu_index.common.slots import snapshot_key, utc_now  # noqa: E402
from gpu_index.common.store import (  # noqa: E402
    BucketConfig,
    day_slot_keys,
    get_object_bytes,
    get_panel_composite,
    get_snapshot_by_key,
    list_panel_observations,
    make_client,
    make_run_id,
    panel_composite_key,
    snapshot_bytes,
    upload_panel_composite,
)
from gpu_index.index.weights import (  # noqa: E402
    MODE_DYNAMIC,
    PRUNE_MARGIN_HOURS,
    advance_panel_weight_state,
    new_weight_state,
    series_print,
)


def _in_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


# The record drift scan's daily firing (module docstring): re-reading the
# record for dozens of published observations 48x/day would be the job's
# dominant cost, so the scan runs on the 16:00Z firing only -- the daily
# lanes' canonical-slot hour -- unless --drift-scan forces it.
DRIFT_SCAN_HOUR_UTC = 16

# Parsed-snapshot cache size. Snapshots are consumed in stamp order, so
# one entry is usually enough (24 observations share a day's keys but
# each HOUR is read once); a few entries absorb the drift scan re-reading
# a neighbor while the compute path walks forward. Snapshots are STRIPPED
# to the panel's member entries and in-sku-union rows before caching, so
# the bound is on count, and memory stays flat on wide observatory days.
_SNAPSHOT_CACHE_ENTRIES = 4


# Artifact-derived strings (verdict currency labels, run ids) must never
# carry GH workflow-command sequences -- under GITHUB_ACTIONS a smuggled
# newline + '::' would inject a workflow command into the job. Same rule
# as compute_index_composite.py: control chars become spaces.
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


def _stamp_datetime(stamp: int) -> datetime:
    day_ordinal, hour = divmod(int(stamp), 24)
    return datetime.combine(
        date.fromordinal(day_ordinal), time(hour), tzinfo=timezone.utc
    )


def _next_scheduled_stamp(schedule, stamp: int) -> int:
    """The scheduled stamp strictly after ``stamp`` -- the mark at which
    the observation at ``stamp`` CLOSES. Always exists on a valid grid
    (the final era is open-ended with >= 1 slot per day, so the gap is
    bounded by 47 hours); a 72h probe window is therefore sufficient,
    and an empty probe is a loud impossibility, never a silent skip."""
    upcoming = schedule.scheduled_stamps(int(stamp) + 1, int(stamp) + 73)
    if not upcoming:
        raise RuntimeError(
            f"no scheduled stamp follows {stamp_to_hour_iso(stamp)} within "
            f"72h -- the era grid is malformed (validation should have "
            f"refused it)"
        )
    return upcoming[0]


def advance_panel_state_from_published(
    stored: dict,
    window_history: dict,
    window_currencies: dict,
    pending_currencies: dict,
    weight_state: dict,
) -> None:
    """Advance the replay's filter windows AND weight state from one
    PUBLISHED observation's artifact -- pure reads of pinned facts.

    Filter terms come from the ARTIFACT's own embedded calc_params
    (rule D2), never the live config. Each stored chosen print runs
    through the SAME filter_observation + advance_window state machine
    compute_observation used. Entries without both chosen and filter
    advance nothing, by construction of the artifact: manually-excluded
    and statistic-held-out seats have no chosen; a quarantined seat has
    chosen but NO filter verdict (its print never existed -- capture
    parity), so the window and any pending streak stay untouched.

    The weight state ingests the observation's trusted prints (every
    chosen whose filter_observation is trusted -- accepted AND fenced;
    the fence holds a print out of the INDEX, never the weight series)
    plus the pinned rounded vector and mode latch from weight_calc, at
    the artifact's own stamp, under the artifact's own dw params -- the
    no-slot_prints replay rule: the sources[] block IS the print record,
    so nothing is ever re-derived from raw (which can legitimately grow
    after publication)."""
    artifact_params = stored.get("calc_params") or {}
    filter_terms = str(artifact_params.get("filter_terms", DEFAULT_FILTER_TERMS))
    prints: dict = {}
    for entry in stored.get("sources", []):
        if entry.get("chosen") and entry.get("filter"):
            observation = filter_observation(
                entry["chosen"], filter_terms=filter_terms
            )
            advance_window(
                window_history,
                window_currencies,
                pending_currencies,
                entry["source_id"],
                observation,
            )
            if observation is not None:
                prints[entry["source_id"]] = series_print(
                    entry["chosen"]["usd_per_gpu_hr"], observation
                )
    weight_calc = stored.get("weight_calc") or {}
    advance_panel_weight_state(
        weight_state,
        obs_stamp=hour_iso_to_stamp(stored["date"]),
        prints=prints,
        vector=weight_calc.get("weights"),
        mode=weight_calc.get("mode", "fallback"),
        dw_params=artifact_params["dynamic_weights"],
    )


def strip_snapshot(payload: dict, *, member_ids, sku_union) -> dict:
    """A stored snapshot reduced to exactly what the panel consumes --
    the member SOURCE ENTRIES (all their fields: status, book_stats, ...)
    with their observation rows filtered to the panel's sku union, plus
    the top-level run_id / late_fill the artifact copies. BYTE-NEUTRAL by
    construction (test-pinned): compute_observation and detect_drift read
    only sources/run_id/late_fill, member_eligible_rows admits only rows
    whose sku is in a member's sku set (a subset of the union), and the
    statistics locate their book_stats through those surviving rows'
    identifiers. The point is MEMORY (change 2): a wide observatory
    snapshot (dozens of sources, thousands of rows) must not sit in the
    per-run cache at full size."""
    sources = []
    for entry in payload.get("sources") or []:
        if (
            not isinstance(entry, dict)
            or entry.get("source_id") not in member_ids
        ):
            continue
        slim = dict(entry)
        slim["observations"] = [
            row
            for row in entry.get("observations") or []
            if isinstance(row, dict) and row.get("sku") in sku_union
        ]
        sources.append(slim)
    return {
        "run_id": payload.get("run_id"),
        "late_fill": payload.get("late_fill"),
        "sources": sources,
    }


def detect_drift(
    stored: dict,
    snapshot,
    *,
    obs_date: str,
    params: dict,
    screens: dict,
    fx_records: dict,
) -> list:
    """How the record NOW disagrees with a published observation.

    Re-resolves every seat via the SAME panel pipeline that priced it
    (member_eligible_rows -> resolve_member_print), so statistic seats
    re-price by their statistic instead of false-drifting against the
    lowest-eligible rule. FX-converted prints compare in NATIVE terms
    (the observation date's real ECB rate landing after a walked-back
    publish is routine); statistic prints and plain USD prints compare
    in USD. Manually-excluded seats are skipped -- the artifact
    diverging from raw is the POINT of an exclusion."""
    msgs = []
    if stored.get("observation_missed") and snapshot is not None:
        msgs.append(
            "published as observation_missed but the record now holds a "
            "snapshot for this slot"
        )
        return msgs
    if not stored.get("observation_missed") and snapshot is None:
        # THE tripwire case: a published observation whose record
        # evidence vanished.
        msgs.append(
            "record now holds NO snapshot for this published observation"
        )
        return msgs
    if snapshot is None:
        return msgs
    if stored.get("snapshot_run_id") != snapshot.get("run_id"):
        msgs.append(
            f"selection changed: published from run "
            f"{stored.get('snapshot_run_id')}, record now yields "
            f"{snapshot.get('run_id')}"
        )
    entries = {s["source_id"]: s for s in snapshot.get("sources", [])}
    for entry in stored.get("sources", []):
        source_id = entry["source_id"]
        if entry.get("status") == "manually_excluded":
            continue
        member_screen = screens["members"].get(source_id)
        if member_screen is None:
            # The membership rides calc_params, so the D2 fence refuses
            # any NEW publish under this config -- but the scan must not
            # crash on the way there.
            msgs.append(
                f"{source_id}: published seat is not a member of the live "
                f"config (members drifted; the D2 fence governs publishes)"
            )
            continue
        stored_chosen = entry.get("chosen") or {}
        rows = member_eligible_rows(
            entries.get(source_id),
            skus=member_screen["skus"],
            reject_patterns=screens["reject"],
            require_patterns=member_screen["require"],
            eligible_tiers=params["eligible_tiers"],
            extra_require=member_screen["extra_require"],
        )
        current = resolve_member_print(
            rows,
            source_entry=entries.get(source_id),
            statistic=member_screen["statistic"],
            statistic_params=params["statistic_params"],
            obs_date=obs_date,
            fx_records=fx_records,
            fx_max_staleness_days=params["fx_max_staleness_days"],
        )
        if (
            current is None
            or current.get("fx_unavailable")
            or current.get("held_out")
        ):
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
                    f"{source_id}: published native print {published_cmp} "
                    f"vs record {current_cmp}"
                )
        else:
            published = stored_chosen.get("usd_per_gpu_hr")
            current_price = current.get("usd_per_gpu_hr")
            if published != current_price:
                msgs.append(
                    f"{source_id}: published print {published} vs record "
                    f"{current_price}"
                )
    return msgs


def _config_excluded_for(exclusions, obs_date: str, obs_hour: int) -> set:
    """The source_ids the live config excludes at one observation -- the
    scope rule is gpu_index.index.panel.exclusion_applies, the SAME predicate the
    engine applies (one home; a forked copy here could silently disagree
    with the engine on exactly the entries this pin polices)."""
    return {
        e["source_id"]
        for e in exclusions
        if exclusion_applies(e, obs_date, obs_hour)
    }


def verify_published_observation(args, config, params, schedule) -> int:
    """--verify-published <YYYY-MM-DDTHH>: recompute one PUBLISHED
    observation byte-for-byte from the record and compare against the
    stored artifact -- the daily CLI's verify_published re-minted per
    OBSERVATION. Reads only: published observations are never revised,
    and this mode holds to that by construction (no upload, no pointer
    move, no FX persist -- stored append-only FX records are the only
    rates read, never the live feed).

    Determinism comes from pinning everything to the record, exactly the
    replay semantics main() defines:
      - filter windows / currency streaks / the weight state advance
        from the PUBLISHED artifacts genesis..target-1 (never from raw,
        which can legitimately grow after publication);
      - the observation's snapshot is the artifact's own pinned choice
        (snapshot_run_id fetched by exact key under the observation's
        record source), never re-selected from the record;
      - a pinned observation_missed / record_quarantined verdict replays
        as recorded (a quarantined stamp's stored object is never read
        -- parsing it is the crash the quarantine fences off; a record
        that GREW a snapshot after a missed publish is the drift scan's
        finding, not a verify failure);
      - the recompute runs under the LIVE config (the same config main()
        extends the series with), so calc_params drift from the artifact
        guarantees a MISMATCH that is config drift (rule D2: mint a new
        methodology_id), not necessarily tampering -- a warning says so
        before the verdict.
    """
    prefix = config["bucket_prefix"]
    methodology_id = params["methodology_id"]
    try:
        target = hour_iso_to_stamp(args.verify_published)
    except PanelScheduleError as exc:
        error(f"--verify-published: {exc}")
        return 1
    stamp_iso = stamp_to_hour_iso(target)
    if not schedule.is_scheduled(target):
        error(
            f"--verify-published {stamp_iso} is not a scheduled "
            f"observation of this lane's era grid"
        )
        return 1
    obs_date, obs_hour = stamp_to_date_hour(target)

    bucket_config = BucketConfig.from_env()
    client = make_client(bucket_config)
    bucket = bucket_config.bucket

    artifact_key = panel_composite_key(prefix, methodology_id, stamp_iso)
    stored_raw = get_object_bytes(client, bucket, artifact_key)
    if stored_raw is None:
        error(
            f"{stamp_iso} is not published under {methodology_id} "
            f"(no {artifact_key}) -- derive it with --observation "
            f"{stamp_iso} --dry-run instead"
        )
        return 1
    stored = json.loads(stored_raw)
    stored_sha = hashlib.sha256(stored_raw).hexdigest()
    index = stored.get("index") or {}
    print(
        f"verify-published {stamp_iso} (methodology {methodology_id})\n"
        f"artifact: {artifact_key}\n"
        "published: "
        + (
            "PANEL_DARK"
            if stored.get("panel_dark")
            else f"{index.get('value_usd_gpu_hr'):.4f} $/GPU-hr "
            f"({index.get('sources_used_count', 0)} sources)"
        )
        + (" [observation_missed]" if stored.get("observation_missed") else "")
        + (
            " [record_quarantined]"
            if stored.get("record_quarantined")
            else ""
        )
    )

    # Rule D2 heads-up before the verdict: the recompute runs under the
    # LIVE config, so params drift from the artifact guarantees a byte
    # mismatch that is config drift, not necessarily tampering.
    live_embedded = embedded_calc_params(params)
    stored_params = stored.get("calc_params") or {}
    drifted = sorted(
        k
        for k in set(live_embedded) | set(stored_params)
        if live_embedded.get(k) != stored_params.get(k)
    )
    if drifted:
        warn(
            f"live config calc_params drift from the artifact's on key(s) "
            f"{drifted} -- a MISMATCH below reflects config drift (rule "
            "D2: mint a new methodology_id), not necessarily tampering"
        )

    # FX: stored append-only records only -- the record, not the live
    # feed (fx_lane none performs zero FX reads, the USD-only rule).
    if params["fx_lane"] == "none":
        fx_records: dict = {}
    else:
        fx_records = load_stored_rates(client, bucket, prefix=prefix)

    # Replay state advance from published artifacts, genesis..target-1 --
    # the exact pin-to-published walk main() does, minus record reads and
    # writes. recent_payloads doubles as the jump-screen reference book.
    window_history: dict = {}
    window_currencies: dict = {}
    pending_currencies: dict = {}
    weight_state: dict = new_weight_state()
    lookback = params["jump_screen"]["reference_max_lookback"]
    recent_payloads: deque = deque(maxlen=lookback)
    for prior in schedule.scheduled_stamps(schedule.genesis_stamp, target):
        prior_stored = get_panel_composite(
            client,
            bucket,
            prefix=prefix,
            methodology_id=methodology_id,
            observation=stamp_to_hour_iso(prior),
        )
        if prior_stored is None:
            warn(
                f"{stamp_to_hour_iso(prior)} is unpublished below the "
                "target observation -- the series publishes in order, so "
                "replay state may be incomplete (expect MISMATCH until "
                "the gap is explained)"
            )
            continue
        advance_panel_state_from_published(
            prior_stored,
            window_history,
            window_currencies,
            pending_currencies,
            weight_state,
        )
        recent_payloads.append(prior_stored)

    # The observation's snapshot: the artifact's own pinned verdict. A
    # missed or quarantined observation replays with snapshot=None; a
    # priced one fetches the pinned snapshot by exact key.
    quarantine = stored.get("record_quarantined")
    if quarantine is not None or stored.get("observation_missed"):
        snapshot = None
    else:
        record_entry = record_source_for(params["record_sources"], obs_date)
        pinned_key = snapshot_key(
            record_entry["prefix"],
            date.fromisoformat(obs_date),
            obs_hour,
            stored.get("snapshot_run_id"),
        )
        snapshot_raw = get_object_bytes(client, bucket, pinned_key)
        if snapshot_raw is None:
            error(
                f"MISMATCH {stamp_iso}: the artifact's pinned snapshot "
                f"{pinned_key} is MISSING from the record -- cannot "
                "recompute; the record evidence for this published "
                "observation is gone (the drift tripwire case)"
            )
            return 1
        snapshot = json.loads(snapshot_raw)

    # Jump-screen reference: the most recent prior non-missed,
    # non-quarantined published artifact within the walk-back window --
    # main()'s rule verbatim.
    reference_prints = None
    reference_label = None
    for prior_payload in reversed(recent_payloads):
        if not prior_payload.get("observation_missed") and not (
            prior_payload.get("record_quarantined")
        ):
            reference_prints = jump_reference_prints(prior_payload)
            reference_label = prior_payload["date"]
            break

    payload = compute_observation(
        config=config,
        obs_stamp=target,
        snapshot=snapshot,
        fx_records=fx_records,
        window_history=window_history,
        window_currencies=window_currencies,
        pending_currencies=pending_currencies,
        weight_state=weight_state,
        reference_prints=reference_prints,
        reference_label=reference_label,
        schedule=schedule,
        calc_params=params,
        compiled_screens=compile_screens(params),
        record_quarantined=quarantine,
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
        f"MISMATCH {stamp_iso}: published sha256={stored_sha} vs "
        f"recomputed sha256={recomputed_sha}"
    )
    if diffs:
        for diff in diffs:
            print(f"  {_log_clean(diff)}")
        if any(".fx_" in d or "fx_as_of" in d for d in diffs):
            notice(
                "diff paths touch fx fields -- a rate walked back at "
                "publish time and backfilled later is the known benign "
                "cause; the artifact's recorded rate stands"
            )
    else:
        print(
            "  JSON content is identical; the stored bytes are not the "
            "canonical serialization (sorted keys, 2-space indent, "
            "trailing newline) -- the artifact bytes were rewritten"
        )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute hourly panel index composites from the stored record."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Panel config path (REQUIRED; no env fallback)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Compute every missing CLOSED observation from genesis onward",
    )
    parser.add_argument(
        "--observation", help="Compute a single observation (YYYY-MM-DDTHH)"
    )
    parser.add_argument(
        "--from", dest="from_obs", help="Range start (YYYY-MM-DDTHH)"
    )
    parser.add_argument("--to", dest="to_obs", help="Range end (YYYY-MM-DDTHH)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute and print; write nothing"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print full payload JSON"
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        dest="max_observations",
        help="Per-run publish valve: stop after N publishes (exit 0; the "
        "next run continues the backlog)",
    )
    parser.add_argument(
        "--drift-scan",
        action="store_true",
        dest="drift_scan",
        help="Force the record drift scan this run (it otherwise runs only "
        "on the 16:00Z firing -- DRIFT_SCAN_HOUR_UTC)",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate the config offline and exit 0 (no bucket access)",
    )
    parser.add_argument(
        "--verify-published",
        metavar="OBSERVATION",
        help=(
            "Recompute one PUBLISHED observation (YYYY-MM-DDTHH) "
            "byte-for-byte from the record (prior artifacts + its pinned "
            "snapshot) and compare against the stored artifact -- MATCH "
            "exits 0, MISMATCH exits 1. Reads only; writes nothing."
        ),
    )
    args = parser.parse_args()

    if args.max_observations is not None and args.max_observations < 1:
        parser.error("--max-observations must be >= 1")
    if args.verify_published:
        if (
            args.sync
            or args.observation
            or args.from_obs
            or args.to_obs
            or args.check_config
        ):
            parser.error(
                "--verify-published is its own mode -- do not combine it "
                "with --sync/--observation/--from/--to/--check-config"
            )
    elif not args.check_config and not (
        args.sync or args.observation or (args.from_obs and args.to_obs)
    ):
        parser.error(
            "pick a mode: --sync, --observation, --from/--to, "
            "--verify-published, or --check-config"
        )

    client = None
    bucket = None
    try:
        config = load_panel_config(args.config)
    except PanelConfigError as exc:
        error(f"panel config refused: {exc}")
        return 1
    params = panel_calc_params(config)
    schedule = panel_schedule(config)
    prefix = config["bucket_prefix"]
    methodology_id = params["methodology_id"]

    if args.check_config:
        # Offline by construction -- nothing above touched BucketConfig
        # or the network, so a creds-free laptop can gate a config-only
        # PR.
        notice(
            f"config OK: panel {config['panel_id']} methodology "
            f"{methodology_id}; {len(config['members'])} members; genesis "
            f"{config['genesis_date']}; {len(config['slot_grids'])} grid "
            f"era(s); no bucket access performed"
        )
        return 0

    if args.verify_published:
        return verify_published_observation(args, config, params, schedule)

    now = utc_now()
    now_stamp = now.date().toordinal() * 24 + now.hour

    targets = None  # --sync: every missing closed observation
    if args.observation or args.from_obs:
        try:
            if args.observation:
                target = hour_iso_to_stamp(args.observation)
                if not schedule.is_scheduled(target):
                    error(
                        f"--observation {args.observation} is not a scheduled "
                        f"observation of this lane's era grid"
                    )
                    return 1
                targets = {target}
            else:
                start = hour_iso_to_stamp(args.from_obs)
                end = hour_iso_to_stamp(args.to_obs)
                if start > end:
                    error(f"--from {args.from_obs} is after --to {args.to_obs}")
                    return 1
                targets = set(schedule.scheduled_stamps(start, end + 1))
                if not targets:
                    error(
                        f"no scheduled observations in "
                        f"[{args.from_obs}..{args.to_obs}]"
                    )
                    return 1
        except PanelScheduleError as exc:
            error(str(exc))
            return 1
        out_of_range = sorted(
            t
            for t in targets
            if t < schedule.genesis_stamp or _stamp_datetime(t) > now
        )
        if out_of_range:
            # A typo'd backfill must never look like a successful no-op.
            error(
                f"target observation(s) outside the replayable range "
                f"[{stamp_to_hour_iso(schedule.genesis_stamp)}..now]: "
                f"{[stamp_to_hour_iso(t) for t in out_of_range]}"
            )
            return 1

    if client is None:
        bucket_config = BucketConfig.from_env()
        client = make_client(bucket_config)
        bucket = bucket_config.bucket

    # The replay domain: every scheduled stamp from genesis whose window
    # has CLOSED. All scheduled stamps <= now_stamp except possibly the
    # last are closed by construction (each closes at the next scheduled
    # stamp, which is itself <= now); only the final one needs the
    # explicit next-mark probe.
    scheduled = list(schedule.iter_scheduled(now_stamp))
    if scheduled and _stamp_datetime(
        _next_scheduled_stamp(schedule, scheduled[-1])
    ) > now:
        closed = scheduled[:-1]
    else:
        closed = scheduled
    closed_set = set(closed)

    # ------------------------------------ bounded replay (LIST frontier)
    # ONE paginated LIST of the composite keyspace learns the published
    # stamp set (module docstring, perf stage) -- the old genesis->now
    # per-stamp GET walk scaled with the series' age.
    published_stamps: set = set()
    for published_iso in list_panel_observations(
        client, bucket, prefix=prefix, methodology_id=methodology_id
    ):
        try:
            published_stamps.add(hour_iso_to_stamp(published_iso))
        except ValueError:
            # Regex-shaped but calendar-invalid (a foreign object in our
            # own keyspace): never a stamp, but worth a page.
            warn(
                f"unparseable observation key {published_iso!r} under the "
                f"{methodology_id} composites keyspace -- ignored"
            )
    frontier = next((s for s in scheduled if s not in published_stamps), None)
    if frontier is None:
        # Every scheduled stamp <= now is published: the frontier is the
        # next future mark; the window still covers the trailing series
        # (drift scan, pin checks) with nothing new to compute.
        frontier = (
            _next_scheduled_stamp(schedule, scheduled[-1])
            if scheduled
            else schedule.genesis_stamp
        )
    dw_live = params["dynamic_weights"]
    state_window_hours = (
        int(dw_live["history_days"]) * 24
        + max(int(x) for x in dw_live["forward_horizons_hours"])
        + PRUNE_MARGIN_HOURS
    )
    window_start = max(schedule.genesis_stamp, frontier - state_window_hours)
    # The stamps this run walks: closed AND inside the state window.
    # Everything older is published (frontier is the FIRST unpublished
    # scheduled stamp) and fully summarized by the seed below.
    replayed = [s for s in closed if s >= window_start]

    if params["fx_lane"] == "none":
        # USD-only lane (the USD-only rule): no ECB fetch, no fx/ keyspace
        # under this prefix. Any non-USD print is held out loudly by the
        # fail-closed conversion path, never guessed.
        fx_records: dict = {}
    else:
        # Stored rates bounded to the replay window plus the walk-back
        # allowance; the live feed fetches ONLY when some unpublished
        # observation's day is unresolvable from stored records after
        # walk-back (module docstring, perf stage).
        fx_from_day = (
            date.fromordinal(window_start // 24)
            - timedelta(days=params["fx_max_staleness_days"])
        ).isoformat()
        fx_records = load_stored_rates(
            client, bucket, prefix=prefix, from_day=fx_from_day
        )
        fx_needed_days = sorted(
            {
                stamp_to_date_hour(s)[0]
                for s in replayed
                if s not in published_stamps
            }
        )
        if fx_needed_days and not rates_cover(
            fx_records,
            fx_needed_days,
            max_staleness_days=params["fx_max_staleness_days"],
        ):
            fx_records = ensure_rates(
                client,
                bucket,
                prefix=prefix,
                persist=not args.dry_run,
                from_day=fx_from_day,
            )

    # Sequential replay state -- the compute_index_composite.py shape:
    # filter windows + currency confirmation streaks + the observation-
    # mode weight state, all reconstructible from published artifacts
    # alone, so replays and mid-series restarts are deterministic.
    window_history: dict = {}
    window_currencies: dict = {}
    pending_currencies: dict = {}
    weight_state: dict = new_weight_state()
    # Jump-screen reference book: the trailing published/computed
    # payloads, bounded by the config's walk-back (the reference for
    # stamp t is the most recent prior non-missed artifact within
    # reference_max_lookback scheduled observations).
    lookback = params["jump_screen"]["reference_max_lookback"]
    recent_payloads: deque = deque(maxlen=lookback)
    screens = compile_screens(params)

    # Pre-window seed (bounded replay, one GET): the two published facts
    # that outlive the state window -- the weight-mode latch (permanent
    # by rule A2, so an in-window rebuild alone could re-run fallback
    # and re-emit switched_on at a wrong stamp) and the D2 baseline (the
    # params fence must hold even when every in-window stamp is
    # unpublished). Everything else the engine consults rebuilds fully
    # inside the window (module docstring).
    last_published_params: dict | None = None
    pre_window = [s for s in published_stamps if s < window_start]
    if pre_window:
        seed = get_panel_composite(
            client,
            bucket,
            prefix=prefix,
            methodology_id=methodology_id,
            observation=stamp_to_hour_iso(max(pre_window)),
        )
        if seed is not None:
            last_published_params = seed.get("calc_params")
            if (seed.get("weight_calc") or {}).get("mode") == MODE_DYNAMIC:
                weight_state["mode"] = MODE_DYNAMIC

    # Slot-granular record reads (perf stage): a day's snapshot keys LIST
    # once per (record prefix, day); only the needed hour's snapshot GETs,
    # and each parsed snapshot is STRIPPED to this panel's member entries
    # and in-sku-union rows before entering the small capped cache --
    # compute_observation and detect_drift read snapshot_run_id/late_fill/
    # sources only, and prints can only come from member rows whose sku is
    # in a member's sku set, so the strip is byte-neutral by construction
    # (test-pinned). read_day_snapshots stays untouched for the daily
    # lanes.
    panel_member_ids = {m["source_id"] for m in params["members"]}
    panel_sku_union = {
        sku for member in params["members"] for sku in member["skus"]
    }
    day_keys_cache: dict = {}
    snapshot_cache: OrderedDict = OrderedDict()

    def _refresh_day_keys(record_prefix: str, day_str: str):
        """One FRESH slot-key LIST, bypassing the run cache -- the
        false-missed guard's read (adversarial review F7): a transient
        empty-Contents gateway blip on the cached LIST must not pin an
        immutable false observation_missed artifact, so the missed
        verdict is confirmed against a second, fresh LIST before it
        publishes. The cache is updated so later hours of the same day
        see the refreshed keys."""
        slot_keys = day_slot_keys(
            client,
            bucket,
            prefix=record_prefix,
            day=date.fromisoformat(day_str),
        )
        day_keys_cache[(record_prefix, day_str)] = slot_keys
        return slot_keys

    def _slot_snapshot(record_prefix: str, day_str: str, hour: int):
        day_key = (record_prefix, day_str)
        slot_keys = day_keys_cache.get(day_key)
        if slot_keys is None:
            slot_keys = _refresh_day_keys(record_prefix, day_str)
        key = slot_keys.get(hour)
        if key is None:
            return None  # slot missing: the observation_missed condition
        if key in snapshot_cache:
            snapshot_cache.move_to_end(key)
            return snapshot_cache[key]
        raw = get_snapshot_by_key(client, bucket, key)
        snapshot = (
            None
            if raw is None
            else strip_snapshot(
                raw, member_ids=panel_member_ids, sku_union=panel_sku_union
            )
        )
        snapshot_cache[key] = snapshot
        if len(snapshot_cache) > _SNAPSHOT_CACHE_ENTRIES:
            snapshot_cache.popitem(last=False)
        return snapshot

    params_drift: list = []
    drift_checked = False
    wrote = 0
    exit_code = 0
    skipped_unpublished = False
    exclusion_conflict = False
    # Drift-scan gating (module docstring): the daily 16:00Z sweep, or an
    # explicit --drift-scan. The bound counts trailing REPLAYED closed
    # observations (== trailing closed observations whenever the ops knob
    # sits inside the state window, its sane range).
    drift_scan_armed = bool(args.drift_scan) or now.hour == DRIFT_SCAN_HOUR_UTC
    drift_scan_observations = int(config["drift_scan_observations"])
    scan_from_index = len(replayed) - drift_scan_observations

    for idx, stamp in enumerate(replayed):
        stamp_iso = stamp_to_hour_iso(stamp)
        obs_date, obs_hour = stamp_to_date_hour(stamp)
        stored = None
        if stamp in published_stamps:
            stored = get_panel_composite(
                client,
                bucket,
                prefix=prefix,
                methodology_id=methodology_id,
                observation=stamp_iso,
            )
            if stored is None:
                # LIST said published, GET says gone -- a race or a
                # deleted object. Recompute is byte-deterministic and the
                # append-only guard arbitrates, so fall through loudly.
                warn(
                    f"{stamp_iso}: listed as published but the artifact "
                    "GET returned nothing -- treating as unpublished"
                )
        if stored is not None:
            # Exclusion pin, per OBSERVATION: a published observation
            # pins its exclusion set under the same scope rule the
            # engine applied (hour-scoped entries pin only their hour).
            stored_excluded = {
                s["source_id"]
                for s in stored.get("sources", [])
                if s.get("status") == "manually_excluded"
            }
            config_excluded = _config_excluded_for(
                params["manual_exclusions"], obs_date, obs_hour
            )
            if stored_excluded != config_excluded:
                error(
                    f"{stamp_iso}: manual_exclusions contradict the "
                    f"PUBLISHED artifact (stored excluded "
                    f"{sorted(stored_excluded)} vs config "
                    f"{sorted(config_excluded)}) -- published observations "
                    "pin their exclusion set; mint a new methodology_id "
                    "instead of editing it"
                )
                exclusion_conflict = True
                exit_code = 1  # loud even with nothing new to publish
            # Record-quarantine pin, same discipline (F6): a published
            # observation pins whether it was quarantined and why; the
            # live config may ADD entries only for unpublished stamps.
            config_quarantine = record_exclusion_reason(
                params["record_exclusions"], obs_date, obs_hour
            )
            stored_quarantine = stored.get("record_quarantined")
            if stored_quarantine != config_quarantine:
                error(
                    f"{stamp_iso}: record_exclusions contradict the "
                    f"PUBLISHED artifact (stored record_quarantined "
                    f"{stored_quarantine!r} vs config "
                    f"{config_quarantine!r}) -- published observations pin "
                    "their quarantine verdict; mint a new methodology_id "
                    "instead of editing it"
                )
                exclusion_conflict = True
                exit_code = 1
            advance_panel_state_from_published(
                stored,
                window_history,
                window_currencies,
                pending_currencies,
                weight_state,
            )
            last_published_params = stored.get("calc_params")
            recent_payloads.append(stored)
            # Drift scanning is bounded in OBSERVATIONS and GATED to the
            # daily 16:00Z sweep / --drift-scan (module docstring);
            # history-advance above stays one GET per published in-window
            # observation regardless, with zero record reads. A
            # record-quarantined observation is SKIPPED outright: the
            # quarantine exists because the stored object is poisoned --
            # the scan re-reading (and json-parsing) it would crash on
            # exactly the object the quarantine fences off, and its
            # divergence from the artifact is the point.
            if (
                drift_scan_armed
                and idx >= scan_from_index
                and not stored.get("record_quarantined")
            ):
                record_entry = record_source_for(
                    params["record_sources"], obs_date
                )
                snapshot = _slot_snapshot(
                    record_entry["prefix"], obs_date, obs_hour
                )
                for msg in detect_drift(
                    stored,
                    snapshot,
                    obs_date=obs_date,
                    params=params,
                    screens=screens,
                    fx_records=fx_records,
                ):
                    warn(
                        f"DRIFT {stamp_iso}: {msg} -- the published "
                        "composite stands (immutable); the record now "
                        "diverges from it"
                    )
            continue

        # ------------------------------------------- unpublished stamp
        if args.max_observations is not None and wrote >= args.max_observations:
            notice(
                f"--max-observations {args.max_observations} reached -- "
                f"{len(replayed) - idx} closed observation(s) remain "
                "unpublished; the next run continues the backlog"
            )
            break

        if not drift_checked and last_published_params is not None:
            # Rule D2, checked once at the first NEW observation: a
            # live config whose calc_params drift from the last
            # published artifact's must never extend the series.
            drift_checked = True
            # embedded_calc_params is the ONE conversion the engine embeds
            # with (gpu_index.index.panel) -- the fence must compare the same bytes.
            live_embedded = embedded_calc_params(params)
            keys = set(live_embedded) | set(last_published_params)
            keys.discard("manual_exclusions")  # has its own check above
            keys.discard("record_exclusions")  # pinned per observation too
            params_drift = sorted(
                k
                for k in keys
                if live_embedded.get(k) != last_published_params.get(k)
            )
            if params_drift:
                error(
                    f"calc_params drift vs the last published "
                    f"{methodology_id} artifact on key(s) {params_drift} "
                    "-- published observations pin their params; mint a "
                    "new methodology_id instead of editing them"
                )
                exit_code = 1

        record_entry = record_source_for(params["record_sources"], obs_date)
        # Record quarantine (F6), checked BEFORE any record read: a
        # quarantined (date, hour)'s stored object must never be fetched
        # or parsed -- the quarantine exists because parsing it crashes.
        quarantine_reason = record_exclusion_reason(
            params["record_exclusions"], obs_date, obs_hour
        )
        if quarantine_reason is not None:
            snapshot = None
            warn(
                f"{stamp_iso}: record quarantined by config "
                f"({quarantine_reason}) -- publishing an explicit "
                "record_quarantined artifact without reading the record"
            )
        else:
            snapshot = _slot_snapshot(record_entry["prefix"], obs_date, obs_hour)
            if snapshot is None:
                # False-missed guard (F7): confirm against ONE fresh LIST
                # before pinning an immutable observation_missed artifact
                # -- a transient empty-Contents gateway blip must not
                # become a permanent false record.
                refreshed = _refresh_day_keys(record_entry["prefix"], obs_date)
                if refreshed.get(obs_hour) is not None:
                    warn(
                        f"{stamp_iso}: slot key appeared on the confirming "
                        "re-LIST (transient empty LIST) -- computing the "
                        "observation instead of publishing missed"
                    )
                    snapshot = _slot_snapshot(
                        record_entry["prefix"], obs_date, obs_hour
                    )

        # Jump-screen reference: the most recent prior NON-MISSED,
        # NON-QUARANTINED artifact within the walk-back window (both
        # carry no prints; a quarantined stamp must not shadow a real
        # reference further back). recent_payloads holds exactly the
        # trailing scheduled observations' payloads -- published ones
        # verbatim, computed-this-run ones by value (byte-deterministic
        # either way, so a later run reconstructs the identical
        # reference).
        reference_prints = None
        reference_label = None
        for prior in reversed(recent_payloads):
            if not prior.get("observation_missed") and not prior.get(
                "record_quarantined"
            ):
                reference_prints = jump_reference_prints(prior)
                reference_label = prior["date"]
                break

        payload = compute_observation(
            config=config,
            obs_stamp=stamp,
            snapshot=snapshot,
            fx_records=fx_records,
            window_history=window_history,
            window_currencies=window_currencies,
            pending_currencies=pending_currencies,
            weight_state=weight_state,
            reference_prints=reference_prints,
            reference_label=reference_label,
            schedule=schedule,
            # Hot-loop invariant: params/screens resolve ONCE per run
            # (compute_observation checks the methodology_id so the
            # amortized path can never price under another lane's law).
            calc_params=params,
            compiled_screens=screens,
            record_quarantined=quarantine_reason,
        )
        recent_payloads.append(payload)

        weight_calc = payload.get("weight_calc") or {}
        if weight_calc.get("switched_on"):
            warn(
                f"{stamp_iso}: computed liveness weights SWITCHED ON -- "
                "the panel leaves its opening weights permanently from "
                "this observation forward"
            )
        if weight_calc.get("degenerate_allocation"):
            warn(
                f"{stamp_iso}: weight bounds degenerate for the "
                f"observation's eligible set -- "
                f"{weight_calc.get('degenerate_allocation')} weights "
                f"published ({weight_calc.get('fallback_reason')})"
            )
        jump_block = payload.get("jump_screen") or {}
        for q in jump_block.get("quarantined") or []:
            # Loud but not red: the verdict is artifact data, the seat
            # re-evaluates against a fresh reference next observation.
            warn(
                f"{stamp_iso}: {q['source_id']} JUMP QUARANTINED (book "
                f"{q['book_pct']}%, corroborators {q['corroborators']}) -- "
                "held out of this observation only (uncorroborated_jump)"
            )
        if jump_block.get("quarantine_skipped"):
            warn(f"{stamp_iso}: {jump_block['quarantine_skipped']}")
        for entry in payload["sources"]:
            verdict = entry.get("filter") or {}
            if verdict.get("untrusted_currency"):
                warn(
                    f"{stamp_iso}: {entry['source_id']} untrusted currency "
                    f"label {verdict.get('currency_label')!r} -- print held "
                    "out fail-closed; window preserved"
                )
                exit_code = 1
            elif verdict.get("currency_mismatch"):
                warn(
                    f"{stamp_iso}: {entry['source_id']} recorded currency "
                    f"changed {verdict.get('window_currency')} -> "
                    f"{verdict.get('currency')} -- held out "
                    f"({verdict.get('pending_count')}/"
                    f"{verdict.get('confirm_after')} toward confirmation); "
                    "window preserved"
                )
                exit_code = 1
            elif verdict.get("currency_confirmed"):
                warn(
                    f"{stamp_iso}: {entry['source_id']} currency change "
                    f"CONFIRMED {verdict.get('window_currency')} -> "
                    f"{verdict.get('currency')} -- window reseeded from "
                    f"the {verdict.get('n_history')} pending prints "
                    "(warm-up restarts)"
                )

        wrote_this_obs = False
        is_target = (targets is None) or (stamp in targets)
        if is_target:
            index = payload.get("index") or {}
            print(
                f"{stamp_iso}: "
                + (
                    "PANEL_DARK"
                    if payload["panel_dark"]
                    else f"{index['value_usd_gpu_hr']:.4f} $/GPU-hr "
                    f"({index['sources_used_count']} sources)"
                )
                + (
                    " [observation_missed]"
                    if payload["observation_missed"]
                    else ""
                )
                + (
                    " [record_quarantined]"
                    if payload.get("record_quarantined")
                    else ""
                )
            )
            if args.json:
                print(json.dumps(payload, indent=2))
            if not args.dry_run:
                if exclusion_conflict:
                    error(
                        f"{stamp_iso}: NOT published -- the config's "
                        "manual_exclusions contradict published "
                        f"{methodology_id} history (see errors above)"
                    )
                    exit_code = 1
                elif params_drift:
                    error(
                        f"{stamp_iso}: NOT published -- the live config's "
                        f"calc_params drift from published {methodology_id} "
                        f"history on key(s) {params_drift} (see error above)"
                    )
                    exit_code = 1
                elif targets is not None and skipped_unpublished:
                    # Writing observation t while an earlier computable
                    # one is unpublished would bake filter/weight
                    # provenance the eventually-published earlier
                    # artifacts may not reproduce -- publish in order.
                    error(
                        f"{stamp_iso}: earlier unpublished computable "
                        "observation(s) exist -- publish in order (run "
                        "--sync or widen the range)"
                    )
                    exit_code = 1
                else:
                    outcome = upload_panel_composite(
                        client,
                        bucket,
                        payload,
                        prefix=prefix,
                        run_id=make_run_id(now),
                        now=now,
                    )
                    print(f"  -> {outcome['composite_key']}")
                    wrote += 1
                    wrote_this_obs = True
        if not wrote_this_obs and not args.dry_run:
            skipped_unpublished = True

    if targets is not None:
        for target in sorted(targets - closed_set):
            # In range (<= now) but the next scheduled mark has not
            # passed: loud exit 1, mirroring the daily lane's
            # not-yet-computable posture.
            notice(
                f"{stamp_to_hour_iso(target)}: not yet closed (an "
                "observation closes at the next scheduled mark)"
            )
            exit_code = 1

    print(f"observations written: {wrote} (methodology {methodology_id})")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
