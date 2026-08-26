# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Pure per-observation composite calculation for the hourly panel lanes.

ONE calculator for all six hourly lanes (METHODOLOGY.md): the migrated B300/B200 lanes and the four
H-series panels run the same functions, parameterized ONLY by panel config
(gpu_index.index.panel_config). Every function here is a pure function of its
inputs -- zero I/O, zero clock reads -- so a lane's whole series is a
deterministic replay of (stored record, stored FX records, config), the
gpu_index.index.composite posture verbatim.

The daily basket engine is REUSED, never forked, wherever the semantics
are identical: filter_observation / advance_window / evaluate_filter /
window_incompatible / the D1 pending-currency streak (the per-source
outlier fence, METHODOLOGY.md section 6.4,
and its fail-closed currency rules), vote_stddev / median_stddev_composite
/ weighted_composite (the calc_v4 median-of-CI-votes aggregate and its
weighted-quantile arithmetic), the volume-weighted-median helper, the
us_ca_verified_host screen, and the R2 EUR->USD conversion. Nothing in
gpu_index.index.composite changes -- the frozen daily series' bytes depend on it.

What is panel-NEW, each a design-doc rule:

  - eligibility runs over the OBSERVATORY record shape per member SKU SET
    (section 3 steps 1-2; the tier screen is the ALLOW-LIST
    calc.eligible_tiers -- methodology section 5 "on-demand only", the
    hourly-mint tier reconciliation), then the panel-level IDENTITY screen
    (step 3: boundary-aware reject tokens on the structured
    sku_identifier, reusing gpu_index.observatory.catalog's normalization/boundary
    machinery -- one home), the per-seat VARIANT rule (step 4:
    require_tokens or declared; fail closed -- a seat whose labels
    satisfy neither prints nothing), and the per-seat EXTRA_REQUIRE
    screen ({key: value} exact-match on the row's structured extra dict,
    e.g. runpod extra.cloud "secure" -- see
    member_eligible_rows for the one documented fail-open);
  - the JUMP SCREEN moves from capture to calc (step 5): same L5
    semantics as gpu_index.index.screens (native-currency ratios, corroboration,
    same-machine delta, starvation stand-down) but the verdict is
    ARTIFACT DATA ONLY -- the record is immutable, a quarantined member is
    held out of this observation (status ``uncorroborated_jump``) and its
    print enters neither the filter window nor the weight series (capture
    parity: a flagged row never became a print);
  - three panel-scoped STATISTICS (section 4) with population-accounting
    gates and thin-book floors, dispatched over the member's PRE-SCREENED
    rows -- a deliberately different signature from the daily registry, so
    the registry contract of the frozen ``vast_vwm_verified_us_ca`` id is
    untouched (see PANEL_STATISTIC_FNS);
  - weights come from the stage-2 OBSERVATION-mode engine
    (gpu_index.index.weights.compute_panel_weights: per-observation R-cutoff on the
    era grid, A2 attendance floor, hour-stamped vectors), advanced with
    each observation's own trusted prints AFTER the vector is computed;
  - the artifact is per OBSERVATION: kind ``index_panel_composite``,
    ``date`` = the fixed-width YYYY-MM-DDTHH stamp (lexicographic ==
    chronological, so the store's no-regress pointer works unchanged),
    with observation_date / observation_hour_utc for readability,
    observation_missed for a scheduled hour with no snapshot (published
    explicitly, never skipped), panel_dark for a below-claim-floor
    observation, and calc_params embedded VERBATIM -- record sources +
    cutover, slot grids per era, every screen, statistic ids + params, dw
    params incl. attendance -- so the D2 refuse-to-extend fence carries
    over. Panel calc_params keys are UNCONDITIONAL: the basket's
    conditional-key discipline protects frozen artifact bytes, and no
    frozen panel bytes exist; a fully-resolved embed is strictly more
    auditable.

Manual exclusions gain an optional ``hour`` (section 3 item 9): a
date-only entry holds out all of that date's observations (the shape the
existing B300/B200 entries need at migration), an hour-scoped entry holds
out exactly one.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gpu_index.index.composite import (
    CURRENCY_CONFIRM_DAYS,
    DEFAULT_COMPOSITE_STATISTIC,
    DEFAULT_FILTER_SIGMA,
    DEFAULT_FILTER_SIGMA_FLOOR,
    DEFAULT_FILTER_TERMS,
    DEFAULT_FILTER_WARMUP,
    DEFAULT_FILTER_WINDOW,
    DEFAULT_MANUAL_VERIFY_PCT,
    MEDIAN_STDDEV_VOTES,
    SOURCE_STATISTIC_FNS,
    _pending_after,
    _weighted_median,
    advance_window,
    evaluate_filter,
    filter_observation,
    median_stddev_composite,
    us_ca_verified_host,
    vote_stddev,
    weighted_composite,
    window_incompatible,
)
from gpu_index.index.fx import (
    DEFAULT_FX_MAX_STALENESS_DAYS,
    FxUnavailableError,
    eur_to_usd,
)
from gpu_index.index.panel_schedule import stamp_to_date_hour, stamp_to_hour_iso
from gpu_index.index.weights import (
    DEFAULT_TARGET_VARIANCE_FLOOR,
    advance_panel_weight_state,
    compute_panel_weights,
    series_print,
)

# One home for the L5 vocabulary and machinery: the quarantine reason
# string and the priceable-currency fence are gpu_index.index.screens' (the calc
# lane must speak the same words the capture lane spoke); the boundary
# matcher is gpu_index.observatory.catalog's (a token must mean the same thing to
# the catalog and to the panel screens -- 'B200' never inside 'GB200',
# 'NVL' never inside 'NVLINK'). The underscore imports are deliberate
# reuse-over-fork of package-internal helpers, same as composite's
# _weighted_median above.
from gpu_index.index.screens import QUARANTINE_REASON, _PRICEABLE_CURRENCIES
from gpu_index.observatory.catalog import _boundary_pattern, normalize_label

PANEL_SCHEMA_VERSION = 1
PANEL_COMPOSITE_KIND = "index_panel_composite"


def _finite_number(value: Any) -> bool:
    """A real, finite number -- the row-eligibility price fence (F9):
    bools are numbers to isinstance but never prices, and json.loads
    happily admits NaN/Infinity literals."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


# ------------------------------------------------- statistics (section 4)
#
# Panel statistics take (rows, source_entry, statistic, params) where
# ``rows`` is the member's PRE-SCREENED eligible row set (sku-set +
# tier-allow-list + implausible + identity + variant + extra_require
# screens already applied) -- the daily
# registry's (source_entry, statistic, sku, interruptible_tiers) contract
# cannot express a sku SET or statistic params, and changing it would
# touch the frozen ``vast_vwm_verified_us_ca`` series' call sites. The two
# registries therefore stay SEPARATE and their id spaces DISJOINT
# (asserted at import below + pinned by test): an id names one computation
# forever, and one id living in two registries with two meanings would be
# mirror drift by construction. A panel member cannot seat a daily-registry
# id (no recorded use exists; rule A3 keeps vast-on-B300 lowest-eligible,
# which is the NO-statistic default).
#
# Return contract: a print dict ({usd_per_gpu_hr, statistic, currency,
# n_eligible_prints, gpu_volume, ...}), a hold-out dict ({"held_out":
# {"reason": ..., counts...}}), or None (no book at all -- the member
# simply has no print, same as the daily statistics' None).

# Chosen priors (METHODOLOGY.md section 6.2); config
# calc.statistic_params overrides per statistic, and the RESOLVED values
# ride calc_params so every artifact pins the floors it was priced under.
PANEL_STATISTIC_PARAM_DEFAULTS: Dict[str, Dict[str, int]] = {
    "vast_vwm_verified_us_ca_v2": {},
    "vast_vwm_verified_us_ca_floor": {
        "min_population_machines": 5,
        "min_population_hosts": 3,
    },
    "lium_vwm_book_floor": {
        "min_population_machines": 5,
        "min_population_miners": 3,
    },
}


def _statistic_floor(params: Dict[str, Any], statistic: str, name: str) -> int:
    """One statistic floor value: the config override when present, else
    the registry's chosen prior (PANEL_STATISTIC_PARAM_DEFAULTS -- the ONE
    home; a literal default inside a statistic body could silently diverge
    from the prior the validator and calc_params embed)."""
    return int(params.get(name, PANEL_STATISTIC_PARAM_DEFAULTS[statistic][name]))


def _accounting_holdout(
    rows: Sequence[Dict[str, Any]],
    source_entry: Optional[Dict[str, Any]],
    statistic: str,
) -> Optional[Dict[str, Any]]:
    """The shared population-accounting gate verdict for the vast
    statistics: the ``no_population_accounting`` hold-out dict when any of
    the member's books lacks the ``verified_us_ca_machines`` proof, else
    None (gate passed). One home so the two vast statistics can never
    drift on the gate's shape or reason string."""
    gap = _population_accounting_gap(rows, source_entry)
    if gap is None:
        return None
    return {
        "held_out": {
            "reason": "no_population_accounting",
            "statistic": statistic,
            "books_missing_accounting": gap,
        }
    }


def _population_accounting_gap(
    rows: Sequence[Dict[str, Any]], source_entry: Optional[Dict[str, Any]]
) -> Optional[List[str]]:
    """The books (by recorded sku_identifier) whose accounting is MISSING,
    or None when every book the member's rows came from carries
    ``verified_us_ca_machines`` -- the proof the population branch ran.

    book_stats is keyed by the vast QUERY name, and every vast row's
    sku_identifier IS its query name (the identity pin in both capture
    lanes), so the rows' own identifiers locate their books in both the
    basket-era and observatory-era records. A row without a string
    identifier cannot locate its book -- fail closed, counted as '?'.
    Presence of the KEY is the gate (a zero count still proves the branch
    ran); a snapshot captured before the branch shipped has no key and the
    member is held out, healing on deploy (design section 4).
    """
    book_stats = (source_entry or {}).get("book_stats")
    stats = book_stats if isinstance(book_stats, dict) else {}
    missing: set = set()
    for row in rows:
        identifier = row.get("sku_identifier")
        if not isinstance(identifier, str) or not identifier:
            missing.add("?")
            continue
        entry = stats.get(identifier)
        if not isinstance(entry, dict) or "verified_us_ca_machines" not in entry:
            missing.add(identifier)
    return sorted(missing) if missing else None


def _vast_verified_book(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """The verified-US/CA USD rows of a pre-screened row set -- the
    verified-US/CA population screen (one home: us_ca_verified_host) plus
    the USD-only rule (a priceless/non-USD row never enters a vast
    statistic)."""
    return [
        row
        for row in rows
        if row.get("price_usd_gpu_hr") is not None
        and us_ca_verified_host(row.get("verification"), row.get("region"))
    ]


def vast_vwm_verified_us_ca_v2(
    rows: Sequence[Dict[str, Any]],
    *,
    source_entry: Optional[Dict[str, Any]],
    statistic: str,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """The frozen vast_vwm_verified_us_ca semantics (volume-weighted median
    of verified-US/CA per-GPU asks, volume = gpu_count_basis) plus the
    POPULATION-ACCOUNTING GATE: the member's books must prove the
    population branch ran (verified_us_ca_machines present in book_stats),
    else held out ``no_population_accounting``. Without the gate, a
    pre-branch snapshot's cheapest-N truncation would silently price the
    statistic on a one-sided-low book -- the truncated-book capture defect
    shape the neutral exclusions record. Fail-closed by construction, deterministic on replay: the gate
    reads only stored bytes."""
    if not rows:
        return None
    holdout = _accounting_holdout(rows, source_entry, statistic)
    if holdout is not None:
        return holdout
    book = [
        (float(row["price_usd_gpu_hr"]), row.get("gpu_count_basis") or 1)
        for row in _vast_verified_book(rows)
    ]
    if not book:
        return None
    return {
        "usd_per_gpu_hr": round(_weighted_median(book), 6),
        "statistic": statistic,
        "currency": "USD",
        "n_eligible_prints": len(book),
        "gpu_volume": sum(w for _, w in book),
    }


def vast_vwm_verified_us_ca_floor(
    rows: Sequence[Dict[str, Any]],
    *,
    source_entry: Optional[Dict[str, Any]],
    statistic: str,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """v2 semantics (accounting gate included) plus POPULATION FLOORS over
    the member's eligible rows across the panel sku set:
    min_population_machines distinct machine_id AND min_population_hosts
    distinct host_id, else held out ``thin_book`` with the counts
    recorded. Rationale (design section 4, live probe 2026-08-23): the
    H100-SXM verified-US/CA book was 3 machines on ONE host -- a median
    over that is one host's price wearing a statistic's clothing. Rows
    without an identity field cannot prove a distinct machine/host and do
    not count toward the floors (fail closed) but still price once the
    floors pass."""
    if not rows:
        return None
    holdout = _accounting_holdout(rows, source_entry, statistic)
    if holdout is not None:
        return holdout
    screened = _vast_verified_book(rows)
    machines = {
        row["machine_id"] for row in screened if row.get("machine_id") is not None
    }
    hosts = {row["host_id"] for row in screened if row.get("host_id") is not None}
    min_machines = _statistic_floor(
        params, "vast_vwm_verified_us_ca_floor", "min_population_machines"
    )
    min_hosts = _statistic_floor(
        params, "vast_vwm_verified_us_ca_floor", "min_population_hosts"
    )
    if len(machines) < min_machines or len(hosts) < min_hosts:
        return {
            "held_out": {
                "reason": "thin_book",
                "statistic": statistic,
                "population_machines": len(machines),
                "population_hosts": len(hosts),
                "min_population_machines": min_machines,
                "min_population_hosts": min_hosts,
            }
        }
    book = [
        (float(row["price_usd_gpu_hr"]), row.get("gpu_count_basis") or 1)
        for row in screened
    ]
    return {
        "usd_per_gpu_hr": round(_weighted_median(book), 6),
        "statistic": statistic,
        "currency": "USD",
        "n_eligible_prints": len(book),
        "gpu_volume": sum(w for _, w in book),
        "population_machines": len(machines),
        "population_hosts": len(hosts),
    }


def lium_vwm_book_floor(
    rows: Sequence[Dict[str, Any]],
    *,
    source_entry: Optional[Dict[str, Any]],
    statistic: str,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Volume-weighted median over the member's eligible DEDUPED machine
    book (cheapest USD print per machine_id); volume = ``extra.gpu_count``
    (the MACHINE size -- gpu_count_basis is pinned 1 on lium rows, so
    reusing the vast volume rule verbatim would weight every machine 1).
    A row whose gpu_count is not an int >= 1, carries no USD price, or has
    no machine_id is SKIPPED AND COUNTED (rows_skipped_* keys), never
    guessed. Floors: min_population_machines distinct machines AND
    min_population_miners distinct truthy ``extra.miner_hotkey``, else
    held out ``thin_book`` with counts. NO verified screen exists on lium
    (probe 2026-08-23: no such field) and no geo screen is applied -- both
    recorded open methodology calls."""
    if not rows:
        return None
    skipped = {"non_usd": 0, "bad_volume": 0, "no_machine_id": 0}
    best: Dict[Any, Tuple[float, int, Any]] = {}
    for row in rows:
        usd = row.get("price_usd_gpu_hr")
        if usd is None:
            skipped["non_usd"] += 1
            continue
        volume = (row.get("extra") or {}).get("gpu_count")
        if not isinstance(volume, int) or isinstance(volume, bool) or volume < 1:
            skipped["bad_volume"] += 1
            continue
        machine = row.get("machine_id")
        if machine is None:
            skipped["no_machine_id"] += 1
            continue
        prev = best.get(machine)
        if prev is None or float(usd) < prev[0]:
            best[machine] = (
                float(usd),
                volume,
                (row.get("extra") or {}).get("miner_hotkey"),
            )
    skip_counts = {
        f"rows_skipped_{key}": count for key, count in skipped.items() if count
    }
    miners = {hotkey for _, _, hotkey in best.values() if hotkey}
    min_machines = _statistic_floor(
        params, "lium_vwm_book_floor", "min_population_machines"
    )
    min_miners = _statistic_floor(
        params, "lium_vwm_book_floor", "min_population_miners"
    )
    if len(best) < min_machines or len(miners) < min_miners:
        return {
            "held_out": {
                "reason": "thin_book",
                "statistic": statistic,
                "population_machines": len(best),
                "population_miners": len(miners),
                "min_population_machines": min_machines,
                "min_population_miners": min_miners,
                **skip_counts,
            }
        }
    book = [(usd, volume) for usd, volume, _ in best.values()]
    return {
        "usd_per_gpu_hr": round(_weighted_median(book), 6),
        "statistic": statistic,
        "currency": "USD",
        "n_eligible_prints": len(book),
        "gpu_volume": sum(volume for _, volume in book),
        "population_machines": len(best),
        "population_miners": len(miners),
        **skip_counts,
    }


PANEL_STATISTIC_FNS = {
    "vast_vwm_verified_us_ca_v2": vast_vwm_verified_us_ca_v2,
    "vast_vwm_verified_us_ca_floor": vast_vwm_verified_us_ca_floor,
    "lium_vwm_book_floor": lium_vwm_book_floor,
}

# An id names ONE computation forever. The panel and daily registries have
# different call signatures, so a shared id would mean two computations
# under one name -- refuse at import (i.e. at load), belt to the test pin.
_ID_OVERLAP = set(PANEL_STATISTIC_FNS) & set(SOURCE_STATISTIC_FNS)
if _ID_OVERLAP:
    raise RuntimeError(
        f"panel statistic ids collide with the daily registry: "
        f"{sorted(_ID_OVERLAP)}"
    )


# ---------------------------------------------------- resolved calc params


def panel_calc_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """The single resolution of a panel lane's calc parameters -- the CLI
    and compute_observation must consume THIS, never re-type defaults
    (gpu_index.index.composite.calc_params doctrine). Embedded VERBATIM in every
    artifact, so the D2 refuse-to-extend fence covers record cutover, era
    grids, membership + variant screens, statistic floors, jump-screen
    thresholds, and every dw knob: changing ANY value without minting a
    new methodology_id is visible in the series."""
    calc = config["calc"]
    members = sorted(config["members"], key=lambda m: m["source_id"])
    member_params: List[Dict[str, Any]] = []
    for member in members:
        entry: Dict[str, Any] = {
            "source_id": str(member["source_id"]),
            "weight": float(member["weight"]),
            "skus": sorted(str(s) for s in member["skus"]),
        }
        if member.get("statistic") is not None:
            entry["statistic"] = str(member["statistic"])
        variant = member.get("variant")
        if variant is not None:
            # mode + require_tokens are print-shaping (methodology);
            # evidence/notes are audit prose and stay config-only.
            embedded = {"mode": str(variant["mode"])}
            if variant["mode"] == "label":
                embedded["require_tokens"] = [
                    str(t) for t in variant["require_tokens"]
                ]
            entry["variant"] = embedded
        extra_require = member.get("extra_require")
        if extra_require is not None:
            # Member-shaping bytes exactly like the variant rule: WHICH
            # recorded rows may print for the seat is methodology, so the
            # screen rides the artifact under the D2 fence.
            entry["extra_require"] = {
                str(k): str(v) for k, v in sorted(extra_require.items())
            }
        member_params.append(entry)
    used_statistics = sorted(
        {m["statistic"] for m in member_params if "statistic" in m}
    )
    statistic_overrides = calc.get("statistic_params") or {}
    statistic_params = {
        stat: {
            name: int((statistic_overrides.get(stat) or {}).get(name, default))
            for name, default in sorted(
                PANEL_STATISTIC_PARAM_DEFAULTS[stat].items()
            )
        }
        for stat in used_statistics
    }
    exclusions = []
    for entry in calc.get("manual_exclusions") or []:
        pinned = {
            "date": str(entry["date"]),
            "source_id": str(entry["source_id"]),
            "reason": str(entry["reason"]),
        }
        if "hour" in entry:
            pinned["hour"] = int(entry["hour"])
        exclusions.append(pinned)
    exclusions.sort(
        key=lambda e: (e["date"], e["source_id"], e.get("hour", -1))
    )
    # Record-quarantine entries (top-level config key; adversarial review
    # F6, amended into the mints before any observation published): an
    # excluded (date, hour) never reads the record and publishes an
    # explicit record_quarantined artifact -- the escape hatch for a
    # poisoned/unparseable snapshot object that would otherwise crash
    # every firing forever (earliest-key-wins means a later good snapshot
    # can never shadow it). Series-shaping, so it rides calc_params; the
    # CLI pins it per published observation exactly like manual
    # exclusions (both are discarded from the plain D2 key compare).
    record_exclusions = sorted(
        (
            {
                "date": str(entry["date"]),
                "hour": int(entry["hour"]),
                "reason": str(entry["reason"]),
            }
            for entry in config.get("record_exclusions") or []
        ),
        key=lambda e: (e["date"], e["hour"]),
    )
    dw = calc["dynamic_weights"]
    screen = calc["jump_screen"]
    return {
        "methodology_id": str(calc["methodology_id"]),
        "genesis_date": str(config["genesis_date"]),
        "record_sources": [
            {
                "kind": str(e["kind"]),
                "prefix": str(e["prefix"]),
                "from_date": str(e["from_date"]),
                **({"to_date": str(e["to_date"])} if "to_date" in e else {}),
            }
            for e in config["record_sources"]
        ],
        "slot_grids": [
            {
                "from_date": str(g["from_date"]),
                "slot_hours_utc": [int(h) for h in g["slot_hours_utc"]],
            }
            for g in config["slot_grids"]
        ],
        "reject_tokens": sorted(
            str(t) for t in config.get("reject_tokens") or []
        ),
        "members": member_params,
        # The tier ALLOW-LIST is index-defining bytes (which recorded rows
        # may price the lane at all); REQUIRED by the loader, no default.
        "eligible_tiers": tuple(str(t) for t in calc["eligible_tiers"]),
        "filter_window": int(calc.get("filter_window", DEFAULT_FILTER_WINDOW)),
        "filter_sigma": float(calc.get("filter_sigma", DEFAULT_FILTER_SIGMA)),
        "filter_warmup": int(calc.get("filter_warmup", DEFAULT_FILTER_WARMUP)),
        "filter_sigma_floor": float(
            calc.get("filter_sigma_floor", DEFAULT_FILTER_SIGMA_FLOOR)
        ),
        "filter_terms": str(calc.get("filter_terms", DEFAULT_FILTER_TERMS)),
        "composite_statistic": str(
            calc.get("composite_statistic", DEFAULT_COMPOSITE_STATISTIC)
        ),
        "manual_verify_pct": float(
            calc.get("manual_verify_pct", DEFAULT_MANUAL_VERIFY_PCT)
        ),
        "min_sources_to_claim": int(calc["min_sources_to_claim"]),
        "fx_lane": str(calc["fx_lane"]),
        "fx_max_staleness_days": int(
            calc.get("fx_max_staleness_days", DEFAULT_FX_MAX_STALENESS_DAYS)
        ),
        "manual_exclusions": exclusions,
        "record_exclusions": record_exclusions,
        # The availability-verified disclosure list SHAPES artifact bytes
        # (the share below), so it is a calc param under the D2 fence --
        # a retune is a versioned methodology change, and the
        # published-stamp recompute path stays byte-deterministic
        # (adversarial review: an unpinned live key here turned the
        # store's idempotent re-PUT into BucketPublishError after any
        # retune). Sorted for canonical bytes.
        "availability_verified_sources": sorted(
            str(sid) for sid in calc.get("availability_verified_sources") or []
        ),
        "jump_screen": {
            "quarantine_pct": float(screen["quarantine_pct"]),
            "corroborate_pct": float(screen["corroborate_pct"]),
            "min_corroborators": int(screen["min_corroborators"]),
            "reference_max_lookback": int(screen["reference_max_lookback"]),
        },
        "statistic_params": statistic_params,
        "dynamic_weights": {
            "scheme": str(dw["scheme"]),
            "lookback_horizons_hours": [
                int(x) for x in dw["lookback_horizons_hours"]
            ],
            "forward_horizons_hours": [
                int(x) for x in dw["forward_horizons_hours"]
            ],
            "history_days": int(dw["history_days"]),
            "half_life_days": float(dw["half_life_days"]),
            "ridge_lambda": float(dw["ridge_lambda"]),
            "gamma": float(dw["gamma"]),
            "weight_min": float(dw["weight_min"]),
            "weight_max": float(dw["weight_max"]),
            "min_train_samples": int(dw["min_train_samples"]),
            "target_variance_floor": float(
                dw.get("target_variance_floor", DEFAULT_TARGET_VARIANCE_FLOOR)
            ),
            "switch_min_eligible": int(dw.get("switch_min_eligible", 1)),
            "max_abs_log_return": (
                float(dw["max_abs_log_return"])
                if dw.get("max_abs_log_return") is not None
                else None
            ),
            "source_weight_caps": {
                str(sid): float(cap)
                for sid, cap in sorted(
                    (dw.get("source_weight_caps") or {}).items()
                )
            },
            # A2: the attendance floor is a weight-methodology param like
            # any other -- it rides the artifact so the switch decision
            # replays from published bytes alone.
            "attendance_floor": float(dw["attendance_floor"]),
            "fallback_weights": {
                m["source_id"]: m["weight"] for m in member_params
            },
        },
        # drift_scan_observations is DELIBERATELY absent: it is an
        # operational knob (how much of the trailing series the warn-only
        # record drift scan re-resolves), not index methodology, so it
        # lives at the panel config's TOP level -- the daily lanes'
        # fallback_parity_methodology_id precedent -- and never rides
        # calc_params/artifact bytes. Amended into the mints before any
        # observation published.
    }


def embedded_calc_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """The artifact-embedded form of a resolved param set: tuples become
    lists (JSON has no tuple), nothing else changes. ONE home shared by
    compute_observation (which embeds it) and the CLI's D2 drift compare
    (which must compare the SAME bytes it would embed -- a forked
    conversion could make the fence blind to, or falsely fire on, a
    tuple-shaped key)."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}


def exclusion_applies(entry: Dict[str, Any], obs_date: str, obs_hour: int) -> bool:
    """Section 3 item 9's ONE scope rule: a date-only exclusion covers
    every observation of its date; an hour-scoped entry covers exactly
    that observation. Shared by the engine (_exclusion_reason) and the
    CLI's published-artifact pin check -- two copies of this predicate
    could silently disagree on exactly the entries the pin exists to
    police."""
    return str(entry["date"]) == str(obs_date) and (
        "hour" not in entry or int(entry["hour"]) == int(obs_hour)
    )


def record_exclusion_reason(
    record_exclusions: Sequence[Dict[str, Any]], obs_date: str, obs_hour: int
) -> Optional[str]:
    """The record-quarantine reason covering one (date, hour), or None.
    Entries are always hour-scoped (a record snapshot is per slot); the
    loader guarantees no duplicates. Shared by the CLI's pre-read check
    and its published-artifact pin check."""
    for entry in record_exclusions:
        if str(entry["date"]) == str(obs_date) and int(entry["hour"]) == int(
            obs_hour
        ):
            return str(entry["reason"])
    return None


def record_source_for(
    record_sources: Sequence[Dict[str, Any]], day: str
) -> Dict[str, Any]:
    """The ONE record source covering ``day`` (config guarantees contiguous
    non-overlapping coverage from genesis, final entry open-ended).
    Canonical ISO strings compare chronologically. Loud on a pre-genesis
    day -- nothing is scheduled there, so nothing may read there."""
    for entry in record_sources:
        if day < entry["from_date"]:
            continue
        if "to_date" in entry and day > entry["to_date"]:
            continue
        return entry
    raise ValueError(f"no record source covers {day!r} (pre-genesis?)")


# --------------------------------------- identity / variant screens (2-4)


def _token_patterns(token: str) -> List[Any]:
    """Compiled boundary patterns for one screen token -- the EXACT
    gpu_index.observatory.catalog compilation: the token passes through
    normalize_label (so 'SXM5' in a label matches token 'SXM' via the
    alpha<->digit split) and its fully-compacted variant also gets a
    pattern when it differs (the catalog's partial-compaction rule)."""
    norm = normalize_label(token)
    patterns = [_boundary_pattern(norm)]
    compact = normalize_label(norm.replace(" ", ""))
    if compact != norm:
        patterns.append(_boundary_pattern(compact))
    return patterns


def compile_screens(params: Dict[str, Any]) -> Dict[str, Any]:
    """Precompiled identity/variant screens for one resolved param set:
    {"reject": [patterns], "members": {sid: {skus, require, declared,
    extra_require, statistic, weight}}}. ``require`` is None when the
    seat has no label rule (broad panels, declared seats) -- distinct
    from an empty list, which would fail every row. ``extra_require`` is
    None when the seat declares none."""
    reject = [
        pattern
        for token in params["reject_tokens"]
        for pattern in _token_patterns(token)
    ]
    members: Dict[str, Dict[str, Any]] = {}
    for member in params["members"]:
        variant = member.get("variant")
        require = None
        declared = False
        if variant is not None:
            if variant["mode"] == "label":
                require = [
                    pattern
                    for token in variant["require_tokens"]
                    for pattern in _token_patterns(token)
                ]
            else:
                declared = True
        members[member["source_id"]] = {
            "skus": set(member["skus"]),
            "require": require,
            "declared": declared,
            "extra_require": member.get("extra_require"),
            "statistic": member.get("statistic"),
            "weight": member["weight"],
        }
    return {"reject": reject, "members": members}


def member_eligible_rows(
    source_entry: Optional[Dict[str, Any]],
    *,
    skus: Any,
    reject_patterns: Sequence[Any],
    require_patterns: Optional[Sequence[Any]],
    eligible_tiers: Sequence[str],
    extra_require: Optional[Dict[str, str]] = None,
    screen_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Design section 3 steps 1-4 for one member: entry status ok; row sku
    in the member's sku set, tier in the ALLOW-LIST ``eligible_tiers``
    (methodology section 5 "on-demand only" -- the hourly-mint tier
    reconciliation: reserved/committed/serverless/from_floor rows are
    ineligible by construction, where the retired exclusion-list screen
    admitted any tier it had not enumerated), implausible falsy; then the
    panel identity screen (reject tokens on sku_identifier -- a row whose
    label carries a rejected variant token is a different instrument,
    however it was filed pre-catalog-change); then the variant rule
    (require_patterns: at least one required token, fail closed -- a
    label that names no variant proves no variant); then the member's
    ``extra_require`` ({key: value} on the row's STRUCTURED extra dict):
    a row whose extra CARRIES a required key must match its value
    exactly (mismatch -> ineligible, counted in ``screen_counts`` under
    ``extra_require_mismatch``), while a row with NO extra dict or
    without the key PASSES. That pass is the ONE deliberate fail-open in
    this screen: basket-era records carry no extra dict at all (the
    basket collectors recorded only the daily lanes' eligible surface -- the
    basket runpod collector requested securePrice alone, so that era is
    secure-only by construction), and the observatory runpod collector
    labels EVERY row with extra.cloud (its source test pins that), so a
    row missing the key can only come from a record era where the
    screened distinction does not exist; failing closed on absence would
    retroactively unseat every basket-era print. Screens read the
    structured sku_identifier / extra only, never prose notes.

    FINITENESS FAIL-CLOSED (adversarial review F9, amended pre-publish):
    a candidate row must carry a FINITE native price (and a finite USD
    price when one is present) or it is excluded and counted
    (``non_finite_price``). json.loads admits NaN/Infinity, and non-USD
    rows skip the capture-side USD plausibility band entirely -- an
    Infinity EUR native would otherwise convert to an Infinity candidate
    and poison the lowest-eligible min. The statistics inherit this
    screen (they run over the returned rows only).

    ``screen_counts`` (when passed) tallies EVERY elimination of a
    sku-matched row by screen: ``tier_ineligible``, ``implausible``,
    ``non_finite_price``, ``identity_rejected``, ``variant_unmatched``,
    ``extra_require_mismatch`` -- the dead-seat visibility record
    (adversarial review F1): a seat whose rows ALL screen out must be
    distinguishable in the artifact from a seat with no rows at all."""
    if (source_entry or {}).get("status") != "ok":
        return []

    def _count(reason: str) -> None:
        if screen_counts is not None:
            screen_counts[reason] = screen_counts.get(reason, 0) + 1

    screening = bool(reject_patterns) or require_patterns is not None
    rows: List[Dict[str, Any]] = []
    for obs in source_entry.get("observations") or []:
        if obs.get("sku") not in skus:
            continue
        if obs.get("tier") not in eligible_tiers:
            _count("tier_ineligible")
            continue
        if obs.get("implausible"):
            _count("implausible")
            continue
        native = obs.get("price_native_per_gpu_hr")
        usd = obs.get("price_usd_gpu_hr")
        if not _finite_number(native) or (
            usd is not None and not _finite_number(usd)
        ):
            _count("non_finite_price")
            continue
        if screening:
            # Hot-loop invariant (review perf stage): normalize the row's
            # identifier ONCE per row and search both compiled screens
            # over that one label (an empty/unnormalizable label matches
            # nothing, so reject passes and require fails -- fail closed,
            # the same verdicts as normalizing per screen).
            label = normalize_label(obs.get("sku_identifier"))
            if (
                reject_patterns
                and label
                and any(p.search(label) for p in reject_patterns)
            ):
                _count("identity_rejected")
                continue
            if require_patterns is not None and not (
                label and any(p.search(label) for p in require_patterns)
            ):
                _count("variant_unmatched")
                continue
        if extra_require:
            extra = obs.get("extra")
            extra_map = extra if isinstance(extra, dict) else {}
            if any(
                key in extra_map and extra_map[key] != value
                for key, value in extra_require.items()
            ):
                _count("extra_require_mismatch")
                continue
        rows.append(obs)
    return rows


# ------------------------------------------------ print resolution (3.6)


def lowest_eligible_print(
    rows: Sequence[Dict[str, Any]],
    *,
    obs_date: str,
    fx_records: Dict[str, Dict[str, Any]],
    fx_max_staleness_days: int = DEFAULT_FX_MAX_STALENESS_DAYS,
) -> Optional[Dict[str, Any]]:
    """The member's default print: lowest eligible per-GPU price over its
    PRE-SCREENED rows -- daily_source_observation's candidate rules
    verbatim (USD verbatim; EUR converts via the ECB record for <=
    obs_date BEFORE the minimum, R2; non-USD non-EUR rows are NEVER
    candidates -- only the EUR conversion is defined; FX outages collect loudly), applied
    over a sku SET instead of one sku. The chosen block additionally pins
    the winning row's stored ``sku`` (a broad member's print must say
    which instrument won) and ``machine_id`` when present (the calc-lane
    jump screen's same-machine delta reads the REFERENCE ARTIFACT, not a
    raw snapshot, so identity continuity must ride the chosen block)."""
    candidates: List[Dict[str, Any]] = []
    fx_errors: List[str] = []
    for obs in rows:
        usd = obs.get("price_usd_gpu_hr")
        fx_block: Dict[str, Any] = {}
        if usd is None:
            native = obs.get("price_native_per_gpu_hr")
            currency = obs.get("currency")
            if native is None or currency in (None, "USD", "UNKNOWN"):
                continue  # no honest way to price this print
            if currency != "EUR":
                continue  # only EUR conversion is defined (R2); extend by mint
            try:
                usd, fx_block = eur_to_usd(
                    float(native),
                    fx_records,
                    obs_date,
                    max_staleness_days=fx_max_staleness_days,
                )
            except FxUnavailableError as exc:
                fx_errors.append(str(exc))
                continue
        candidate = {
            "usd_per_gpu_hr": round(float(usd), 6),
            "sku": obs.get("sku"),
            "tier": obs.get("tier"),
            "gpu_count_basis": obs.get("gpu_count_basis"),
            "raw_value": obs.get("raw_value"),
            "raw_unit": obs.get("raw_unit"),
            "currency": obs.get("currency", "USD"),
            "native_per_gpu_hr": obs.get("price_native_per_gpu_hr"),
            "notes": obs.get("notes", ""),
            **fx_block,
        }
        if obs.get("machine_id") is not None:
            candidate["machine_id"] = obs.get("machine_id")
        candidates.append(candidate)
    if not candidates:
        if fx_errors:
            return {"fx_unavailable": True, "fx_errors": fx_errors}
        return None
    chosen = min(candidates, key=lambda c: c["usd_per_gpu_hr"])
    chosen["n_eligible_prints"] = len(candidates)
    if fx_errors:
        # Mixed-currency partial outage: still prices from USD prints, but
        # the dropped candidates must be VISIBLE (composite's rule).
        chosen["fx_errors_partial"] = fx_errors
    return chosen


def resolve_member_print(
    rows: Sequence[Dict[str, Any]],
    *,
    source_entry: Optional[Dict[str, Any]],
    statistic: Optional[str],
    statistic_params: Dict[str, Dict[str, Any]],
    obs_date: str,
    fx_records: Dict[str, Dict[str, Any]],
    fx_max_staleness_days: int = DEFAULT_FX_MAX_STALENESS_DAYS,
) -> Optional[Dict[str, Any]]:
    """Design section 3 step 6: the member's named statistic when
    configured, else the lowest-eligible rule with EUR FX. ``rows`` must
    already be the member's screened set (member_eligible_rows)."""
    if statistic is not None:
        return PANEL_STATISTIC_FNS[statistic](
            rows,
            source_entry=source_entry,
            statistic=statistic,
            params=statistic_params.get(statistic) or {},
        )
    return lowest_eligible_print(
        rows,
        obs_date=obs_date,
        fx_records=fx_records,
        fx_max_staleness_days=fx_max_staleness_days,
    )


# --------------------------------------------- jump screen at calc (3.5)


def _print_native_terms(chosen: Dict[str, Any]) -> Tuple[float, str]:
    """(price, currency) a print compares in for the jump screen: native
    terms when recorded (lowest-eligible prints), else the USD statistic
    value -- statistic chosen blocks have NO native key and are USD by
    construction."""
    native = chosen.get("native_per_gpu_hr")
    if isinstance(native, (int, float)) and not isinstance(native, bool):
        return float(native), str(chosen.get("currency", "USD") or "USD")
    return float(chosen["usd_per_gpu_hr"]), "USD"


def _jump_lowest_row(
    rows: Sequence[Dict[str, Any]], currency: str
) -> Optional[Dict[str, Any]]:
    """Lowest eligible native print among ``rows`` in ``currency`` --
    gpu_index.index.screens.lowest_eligible semantics MINUS the single-sku test
    (named divergence: these rows are already the member's panel-eligible
    set, with the sku-set/tier/implausible/identity/variant screens
    applied upstream, where the capture screen re-derived eligibility per
    row). Priceable currencies only, the screens.py rule."""
    candidates = [
        row
        for row in rows
        if row.get("currency", "USD") == currency
        and row.get("currency", "USD") in _PRICEABLE_CURRENCIES
        and isinstance(row.get("price_native_per_gpu_hr"), (int, float))
        and not isinstance(row.get("price_native_per_gpu_hr"), bool)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r["price_native_per_gpu_hr"]))


def _jump_same_machine_pct(
    rows: Sequence[Dict[str, Any]], reference_print: Dict[str, Any]
) -> Optional[float]:
    """gpu_index.index.screens._same_machine_pct semantics over the member's
    eligible rows (same named divergence as _jump_lowest_row): today's
    price of the machine the REFERENCE print came from, in the reference
    currency."""
    machine = reference_print.get("machine_id")
    if machine is None:
        return None
    ref_native, ref_currency = _print_native_terms(reference_print)
    if ref_native <= 0:
        return None
    today = [
        row
        for row in rows
        if row.get("machine_id") == machine
        and row.get("currency", "USD") == ref_currency
        and isinstance(row.get("price_native_per_gpu_hr"), (int, float))
        and not isinstance(row.get("price_native_per_gpu_hr"), bool)
    ]
    if not today:
        return None
    best = min(today, key=lambda r: float(r["price_native_per_gpu_hr"]))
    return (float(best["price_native_per_gpu_hr"]) / ref_native - 1.0) * 100.0


def jump_reference_prints(
    artifact: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """{source_id: chosen} from a published panel artifact -- the
    calc-lane reference book. Only status "ok" sources serve (capture
    parity: a quarantined/held-out/excluded print was flagged in ITS
    artifact and must not become the next observation's baseline; a
    member with no ok reference simply has no ratio next time, so a
    genuine uncorroborated repricing costs exactly one observation)."""
    out: Dict[str, Dict[str, Any]] = {}
    for source in (artifact or {}).get("sources") or []:
        if (
            isinstance(source, dict)
            and source.get("status") == "ok"
            and isinstance(source.get("chosen"), dict)
        ):
            out[str(source["source_id"])] = source["chosen"]
    return out


def apply_panel_jump_screen(
    current: Dict[str, Dict[str, Any]],
    reference_prints: Optional[Dict[str, Dict[str, Any]]],
    *,
    jump_params: Dict[str, Any],
    reference_label: Optional[str] = None,
) -> Dict[str, Any]:
    """The L5 jump screen AT CALC (design section 3 step 5): same
    corroboration and starvation semantics as gpu_index.index.screens
    .apply_jump_screen, with three deliberate frame changes -- (a) the
    verdict is ARTIFACT DATA ONLY, never a mutation of the immutable
    record; (b) the reference is a prior published observation's per-member
    chosen prints (jump_reference_prints), walked back by the CALLER up to
    jump_reference_max_lookback scheduled observations; (c) membership and
    eligibility come from the panel config's screens, applied upstream.

    ``current``: {sid: {"chosen": would-be print or None, "rows": the
    member's eligible rows}} for every ok-status, non-excluded member.
    ``reference_prints`` None = no usable reference: report-only, no
    verdicts (fail-open, the capture rule).

    Returns {reference, deltas, quarantined, quarantine_skipped}; the
    caller holds each quarantined member out of THIS observation (status
    ``uncorroborated_jump``) -- the next observation re-evaluates against
    a new reference, so a genuine uncorroborated repricing costs exactly
    one observation.
    """
    quarantine_pct = float(jump_params["quarantine_pct"])
    corroborate_pct = float(jump_params["corroborate_pct"])
    min_corroborators = int(jump_params["min_corroborators"])
    if not isinstance(reference_prints, dict):
        reference_prints = None
    deltas: List[Dict[str, Any]] = []
    moves: Dict[str, float] = {}
    for sid in sorted(current):
        info = current[sid]
        chosen = info.get("chosen")
        rows = info.get("rows") or []
        entry: Dict[str, Any] = {
            "source_id": sid,
            "book_pct": None,
            "same_machine_pct": None,
            "note": "",
        }
        reference = (reference_prints or {}).get(sid)
        if chosen is None:
            entry["note"] = "no eligible print today"
        elif not isinstance(reference, dict):
            entry["note"] = "no reference print"
        else:
            ref_native, ref_currency = _print_native_terms(reference)
            if ref_native <= 0:
                entry["note"] = "non-positive reference print"
            else:
                today_native, today_currency = _print_native_terms(chosen)
                comparable = True
                if today_currency != ref_currency:
                    # Compare in the REFERENCE print's currency (the
                    # capture rule): a mixed-currency book re-selects its
                    # same-currency print; a statistic print is USD-only
                    # and cannot re-select.
                    row = (
                        None
                        if "statistic" in chosen
                        else _jump_lowest_row(rows, ref_currency)
                    )
                    if row is None:
                        entry["note"] = "currency changed -- not comparable"
                        comparable = False
                    else:
                        today_native = float(row["price_native_per_gpu_hr"])
                if comparable:
                    pct = (today_native / ref_native - 1.0) * 100.0
                    entry["book_pct"] = round(pct, 2)
                    moves[sid] = pct
                    same = _jump_same_machine_pct(rows, reference)
                    if same is not None:
                        entry["same_machine_pct"] = round(same, 2)
        deltas.append(entry)

    # Corroboration starvation guard, verbatim semantics from
    # gpu_index.index.screens: with fewer comparable members than
    # min_corroborators + 1 a genuine market-wide move could never gather
    # corroborators -- stand down loudly instead of quarantining wholesale.
    would_fire = [
        sid for sid, pct in moves.items() if abs(pct) >= quarantine_pct
    ]
    quarantine_skipped = None
    if would_fire and len(moves) < min_corroborators + 1:
        quarantine_skipped = (
            f"only {len(moves)} comparable member(s) this observation -- "
            f"corroboration needs {min_corroborators + 1}+ to be decidable; "
            f"jump quarantine skipped for {sorted(would_fire)} (fail-open)"
        )

    quarantined: List[Dict[str, Any]] = []
    if quarantine_skipped is None:
        for sid in sorted(moves):
            pct = moves[sid]
            if abs(pct) < quarantine_pct:
                continue
            corroborators = sum(
                1
                for other, other_pct in moves.items()
                if other != sid and abs(other_pct) >= corroborate_pct
            )
            if corroborators >= min_corroborators:
                continue
            quarantined.append(
                {
                    "source_id": sid,
                    "book_pct": round(pct, 2),
                    "corroborators": corroborators,
                }
            )

    return {
        "reference": reference_label,
        "deltas": deltas,
        "quarantined": quarantined,
        "quarantine_skipped": quarantine_skipped,
    }


# -------------------------------------------------------- the observation


def _exclusion_reason(
    exclusions: Sequence[Dict[str, Any]],
    obs_date: str,
    obs_hour: int,
    source_id: str,
) -> Optional[str]:
    """Section 3 item 9: a date-only exclusion holds out every observation
    of that date; an hour-scoped one holds out exactly this observation
    (the scope rule is exclusion_applies -- one home with the CLI's pin
    check). The loader guarantees one scope per (date, source_id)."""
    for entry in exclusions:
        if entry["source_id"] != source_id:
            continue
        if exclusion_applies(entry, obs_date, obs_hour):
            return entry["reason"]
    return None


def compute_observation(
    *,
    config: Dict[str, Any],
    obs_stamp: int,
    snapshot: Optional[Dict[str, Any]],
    fx_records: Dict[str, Dict[str, Any]],
    window_history: Dict[str, List[float]],
    window_currencies: Optional[Dict[str, str]] = None,
    pending_currencies: Optional[Dict[str, Dict[str, Any]]] = None,
    weight_state: Optional[Dict[str, Any]] = None,
    reference_prints: Optional[Dict[str, Dict[str, Any]]] = None,
    reference_label: Optional[str] = None,
    schedule: Optional[Any] = None,
    calc_params: Optional[Dict[str, Any]] = None,
    compiled_screens: Optional[Dict[str, Any]] = None,
    record_quarantined: Optional[str] = None,
) -> Dict[str, Any]:
    """One scheduled observation's full artifact payload + state updates.

    The compute_day contract re-minted per observation. State dicts
    (window_history / window_currencies / pending_currencies /
    weight_state) are mutated and MUST be the same objects across a
    replay, exactly like compute_day; ``weight_state`` is REQUIRED (panel
    configs always set dynamic_weights, rule A1) and is advanced with
    this observation's own trusted prints AFTER the vector is computed
    (the no-slot_prints replay rule -- the artifact's sources[] block IS
    the print record). ``snapshot`` None publishes the explicit
    observation_missed artifact (design section 2: never skipped, never
    interpolated). ``reference_prints``/``reference_label`` are the jump
    screen's reference book (jump_reference_prints over the reference
    artifact) -- the walk-back across <= jump_screen.reference_max_lookback
    scheduled observations is the CALLER's job (it needs the store).
    ``schedule`` may be passed to amortize grid construction; when built
    here it comes from the same config keys, so the two paths cannot
    diverge. ``calc_params``/``compiled_screens`` (hot-loop invariant,
    review perf stage) likewise let a replay loop resolve
    panel_calc_params / compile_screens ONCE per run instead of once per
    observation: when passed, calc_params must carry the config's own
    methodology_id (checked loudly -- an amortized path pricing under a
    different lane's law must never run), and compiled_screens must be
    compile_screens(calc_params) computed by the caller (it requires
    calc_params for exactly that reason). Derived here from the same
    config otherwise, so the two paths cannot diverge.

    ``record_quarantined`` (adversarial review F6): the config's
    record-exclusion reason for this stamp, when one applies. The CALLER
    checks record_exclusions BEFORE reading the record and passes
    snapshot=None with the reason -- the whole point is that the stored
    object must never be parsed. The artifact publishes as an explicit
    recorded fact: index null, observation_missed FALSE (the record may
    hold bytes; they are quarantined, not missing), record_quarantined =
    the reason. State advances exactly as a missed observation (no
    prints; the weight block computes over an empty eligible set).

    Fail-closed guards: an UNSCHEDULED stamp raises (publishing off-grid
    would corrupt every attendance denominator and cutoff downstream);
    a quarantined stamp with a snapshot passed anyway raises (the engine
    must never price a record the config quarantined); fx_lane "none"
    forces the FX record set EMPTY here -- a USD-only lane must never
    convert, whatever the caller passed (the USD-only rule).
    """
    if calc_params is None:
        if compiled_screens is not None:
            raise ValueError(
                "compiled_screens requires calc_params -- screens compiled "
                "from one resolution must never run under another"
            )
        params = panel_calc_params(config)
    else:
        expected_id = str(config["calc"]["methodology_id"])
        if calc_params.get("methodology_id") != expected_id:
            raise ValueError(
                f"precomputed calc_params carry methodology_id "
                f"{calc_params.get('methodology_id')!r} but the config is "
                f"{expected_id!r} -- the amortized and derived paths must "
                f"be the same law"
            )
        params = calc_params
    if schedule is None:
        from gpu_index.index.panel_config import panel_schedule

        schedule = panel_schedule(config)
    obs_stamp = int(obs_stamp)
    if not schedule.is_scheduled(obs_stamp):
        raise ValueError(
            f"stamp {obs_stamp} ({stamp_to_hour_iso(obs_stamp)}) is not a "
            f"scheduled observation of this panel's era grid"
        )
    if record_quarantined is not None and snapshot is not None:
        raise ValueError(
            f"stamp {stamp_to_hour_iso(obs_stamp)} is record-quarantined "
            f"({record_quarantined!r}) but a snapshot was passed -- the "
            f"engine must never price a record the config quarantined"
        )
    obs_date, obs_hour = stamp_to_date_hour(obs_stamp)
    stamp_iso = stamp_to_hour_iso(obs_stamp)
    record_entry = record_source_for(params["record_sources"], obs_date)

    filter_terms = params["filter_terms"]
    recorded_terms = filter_terms == "recorded_currency"
    if recorded_terms and window_currencies is None:
        raise ValueError(
            "filter_terms 'recorded_currency' requires window_currencies "
            "(source_id -> currency) -- without it a multi-observation "
            "replay cannot detect recorded-currency changes (rule D2)"
        )
    if weight_state is None:
        raise ValueError(
            "panel lanes require weight_state (prices/vectors/mode) -- "
            "rule A1: weights recompute at every observation, and a "
            "replay without shared state cannot reconstruct the series"
        )
    pending = pending_currencies if pending_currencies is not None else {}
    sigma_floor = params["filter_sigma_floor"]
    median_votes = params["composite_statistic"] == MEDIAN_STDDEV_VOTES
    if params["fx_lane"] == "none":
        fx_records = {}

    screens = (
        compiled_screens if compiled_screens is not None else compile_screens(params)
    )
    member_ids = [m["source_id"] for m in params["members"]]

    source_entries = {
        s["source_id"]: s for s in (snapshot or {}).get("sources", [])
    }

    # Resolve pass: every member's would-be print BEFORE any weight or
    # verdict (the compute_day ordering -- the eligible set and the jump
    # screen both need completed resolution).
    resolutions: Dict[str, Dict[str, Any]] = {}
    for source_id in member_ids:
        entry = source_entries.get(source_id)
        resolved: Dict[str, Any] = {"entry": entry}
        excluded_reason = _exclusion_reason(
            params["manual_exclusions"], obs_date, obs_hour, source_id
        )
        if excluded_reason is not None:
            resolved["excluded_reason"] = excluded_reason
        else:
            member_screen = screens["members"][source_id]
            screen_counts: Dict[str, int] = {}
            rows = member_eligible_rows(
                entry,
                skus=member_screen["skus"],
                reject_patterns=screens["reject"],
                require_patterns=member_screen["require"],
                eligible_tiers=params["eligible_tiers"],
                extra_require=member_screen["extra_require"],
                screen_counts=screen_counts,
            )
            resolved["rows"] = rows
            resolved["screen_counts"] = screen_counts
            if screen_counts.get("extra_require_mismatch"):
                resolved["extra_require_mismatches"] = screen_counts[
                    "extra_require_mismatch"
                ]
            chosen = resolve_member_print(
                rows,
                source_entry=entry,
                statistic=member_screen["statistic"],
                statistic_params=params["statistic_params"],
                obs_date=obs_date,
                fx_records=fx_records,
                fx_max_staleness_days=params["fx_max_staleness_days"],
            )
            resolved["chosen"] = chosen
            if (
                chosen is not None
                and not chosen.get("fx_unavailable")
                and not chosen.get("held_out")
            ):
                resolved["observation"] = filter_observation(
                    chosen, filter_terms=filter_terms
                )
        resolutions[source_id] = resolved

    # Jump screen (section 3 step 5): would-be prints of ok-status,
    # non-excluded members vs the reference book.
    current: Dict[str, Dict[str, Any]] = {}
    for source_id in member_ids:
        resolved = resolutions[source_id]
        if "excluded_reason" in resolved:
            continue
        if (resolved.get("entry") or {}).get("status") != "ok":
            continue
        chosen = resolved.get("chosen")
        printable = (
            chosen
            if chosen
            and not chosen.get("fx_unavailable")
            and not chosen.get("held_out")
            else None
        )
        current[source_id] = {
            "chosen": printable,
            "rows": resolved.get("rows") or [],
        }
    jump_block = apply_panel_jump_screen(
        current,
        reference_prints,
        jump_params=params["jump_screen"],
        reference_label=reference_label,
    )
    quarantined = {q["source_id"] for q in jump_block["quarantined"]}

    # The eligible set (who printed a TRUSTED value this observation) --
    # quarantined members are OUT: at capture their rows never became
    # prints, and the calc-lane verdict must have the same reach.
    eligible = [
        source_id
        for source_id in member_ids
        if resolutions[source_id].get("observation") is not None
        and source_id not in quarantined
    ]

    dynamic_params = params["dynamic_weights"]
    weight_block = compute_panel_weights(
        weight_state,
        obs_stamp=obs_stamp,
        eligible=eligible,
        dw_params=dynamic_params,
        fallback_weights=dynamic_params["fallback_weights"],
        schedule=schedule,
    )
    weight_of: Dict[str, Optional[float]] = {
        source_id: weight_block["weights"].get(source_id)
        for source_id in member_ids
    }

    sources_block: List[Dict[str, Any]] = []
    passing: List[Tuple[str, float, float]] = []
    vote_stddevs: Dict[str, float] = {}
    prints_today: Dict[str, float] = {}
    filter_inputs: Dict[str, Optional[Tuple[float, str]]] = {}
    trusted_prints: Dict[str, Dict[str, Any]] = {}

    for source_id in member_ids:
        resolved = resolutions[source_id]
        entry = resolved.get("entry")
        detail: Dict[str, Any] = {
            "source_id": source_id,
            "weight": weight_of.get(source_id),
        }
        if resolved.get("extra_require_mismatches"):
            # Screened-out rows stay visible (the flag-never-delete
            # doctrine): a community-cloud print never leaves the book
            # silently.
            detail["extra_require_mismatches"] = resolved[
                "extra_require_mismatches"
            ]
        excluded_reason = resolved.get("excluded_reason")
        if excluded_reason is not None:
            detail["status"] = "manually_excluded"
            detail["excluded_reason"] = excluded_reason
            sources_block.append(detail)
            continue
        chosen = resolved.get("chosen")
        if chosen is not None and chosen.get("held_out"):
            # Statistic hold-out (section 4): the gate/floor verdict with
            # its counts is the day's record for this seat.
            detail["status"] = "held_out"
            detail["held_out"] = chosen["held_out"]
            sources_block.append(detail)
            continue
        if chosen is None or chosen.get("fx_unavailable"):
            detail["status"] = (
                "fx_unavailable"
                if chosen and chosen.get("fx_unavailable")
                else (entry or {}).get("status", "missing")
            )
            if chosen and chosen.get("fx_errors"):
                detail["fx_errors"] = chosen["fx_errors"]
            if (entry or {}).get("status") == "ok":
                # Dead-seat visibility (adversarial review F1): an
                # ok-status seat with NO print must say WHY on the
                # artifact. eligible_rows pins how many rows survived the
                # screens; a non-empty screen_counts block proves rows
                # matched the seat's skus and were eliminated BY SCREEN
                # (a mis-tokened variant rule shows up here forever),
                # distinguishable from no-rows-at-all (eligible_rows 0,
                # no screen_counts).
                detail["eligible_rows"] = len(resolved.get("rows") or [])
                if resolved.get("screen_counts"):
                    detail["screen_counts"] = dict(resolved["screen_counts"])
            sources_block.append(detail)
            continue
        if source_id in quarantined:
            # L5 at calc: held out for THIS observation only; the would-be
            # print stays in the artifact as evidence, but it enters
            # neither the filter window nor the weight series (capture
            # parity: a flagged row never became a print).
            detail["status"] = QUARANTINE_REASON
            detail["chosen"] = chosen
            sources_block.append(detail)
            continue
        price = chosen["usd_per_gpu_hr"]
        observation = resolved.get("observation")
        # The vote window (calc_v4) -- always the PRE-advance history; the
        # untrusted/mismatch paths leave it None (no vote is cast there).
        vote_tail: Optional[List[float]] = None
        if observation is None:
            # Rule D1, fail-closed: no trustworthy value exists in
            # window terms. Held out; the window is preserved.
            verdict: Dict[str, Any] = {
                "accepted": False,
                "unfiltered": False,
                "untrusted_currency": True,
                "currency_label": chosen.get("currency"),
                "n_history": len(window_history.get(source_id) or []),
            }
        else:
            filter_price, filter_currency = observation
            if window_incompatible(
                window_history, window_currencies, source_id, filter_currency
            ):
                queued = _pending_after(
                    pending, source_id, filter_price, filter_currency
                )
                prior_currency = window_currencies.get(source_id)
                if len(queued) >= CURRENCY_CONFIRM_DAYS:
                    if median_votes:
                        vote_tail = list(queued[:-1])
                    verdict = {
                        "accepted": True,
                        "unfiltered": True,
                        "currency_confirmed": True,
                        "currency": filter_currency,
                        "window_currency": prior_currency,
                        "filter_price": round(filter_price, 6),
                        "n_history": CURRENCY_CONFIRM_DAYS,
                    }
                else:
                    verdict = {
                        "accepted": False,
                        "unfiltered": False,
                        "currency_mismatch": True,
                        "currency": filter_currency,
                        "window_currency": prior_currency,
                        "filter_price": round(filter_price, 6),
                        "pending_count": len(queued),
                        "confirm_after": CURRENCY_CONFIRM_DAYS,
                        "n_history": len(window_history.get(source_id) or []),
                    }
            else:
                if median_votes:
                    vote_tail = list(
                        (window_history.get(source_id) or [])[
                            -params["filter_window"] :
                        ]
                    )
                verdict = evaluate_filter(
                    window_history.get(source_id, []),
                    filter_price,
                    window=params["filter_window"],
                    sigma=params["filter_sigma"],
                    warmup=params["filter_warmup"],
                    sigma_floor=sigma_floor,
                    currency=filter_currency if recorded_terms else None,
                )
        detail.update({"status": "ok", "chosen": chosen, "filter": verdict})
        prints_today[source_id] = price
        filter_inputs[source_id] = observation
        if observation is not None:
            # The weight series holds every real TRUSTED print -- accepted
            # AND sigma-fenced (the fence holds a print out of the INDEX,
            # never out of the weight series; R-winsor bounds it).
            trusted_prints[source_id] = series_print(price, observation)
        if verdict["accepted"]:
            passing.append((source_id, float(weight_of[source_id]), price))
            if median_votes:
                # The compute_day fx-factor rule verbatim: a non-USD vote
                # converts at the print's own rate, else the implied rate,
                # else the publish dies loudly (no parity by default).
                if observation is not None and observation[1] != "USD":
                    fx_rate = chosen.get("fx_rate")
                    native = chosen.get("native_per_gpu_hr")
                    if (
                        isinstance(fx_rate, (int, float))
                        and not isinstance(fx_rate, bool)
                        and math.isfinite(float(fx_rate))
                        and float(fx_rate) > 0
                    ):
                        fx_factor = float(fx_rate)
                    elif (
                        isinstance(native, (int, float))
                        and not isinstance(native, bool)
                        and math.isfinite(float(native))
                        and float(native) > 0
                        and math.isfinite(float(price))
                        and float(price) > 0
                    ):
                        fx_factor = float(price) / float(native)
                    else:
                        raise ValueError(
                            f"no trustworthy fx rate for {source_id}'s "
                            f"vote: fx_rate={fx_rate!r} native={native!r} "
                            f"usd={price!r}"
                        )
                else:
                    fx_factor = 1.0
                vote = vote_stddev(
                    vote_tail or [],
                    sigma_floor=sigma_floor,
                    fx_factor=fx_factor,
                )
                detail["vote"] = vote
                vote_stddevs[source_id] = vote["conf_usd_gpu_hr"]
        sources_block.append(detail)

    # R3 warm-up manual-verify flag (compute_day rule, mechanically
    # unchanged at hourly cadence -- design section 3 step 7 keeps the
    # warm-up machinery as-is).
    if prints_today:
        panel_mean = statistics.mean(prints_today.values())
        for detail in sources_block:
            verdict = detail.get("filter")
            if not verdict or not verdict.get("unfiltered"):
                continue
            price = detail["chosen"]["usd_per_gpu_hr"]
            if (
                panel_mean > 0
                and abs(price - panel_mean) / panel_mean * 100.0
                > params["manual_verify_pct"]
            ):
                verdict["manual_verify"] = True

    # Per-source outlier-window advance -- ONE state machine
    # (gpu_index.index.composite's),
    # per observation instead of per day. Quarantined members are ABSENT
    # here on purpose: their print never existed (capture parity), so the
    # window AND any pending currency streak are preserved untouched.
    for source_id, observation in filter_inputs.items():
        advance_window(
            window_history,
            window_currencies,
            pending,
            source_id,
            observation,
        )

    # Claim floor -> explicit dark artifact (design section 3 step 8).
    min_claim = params["min_sources_to_claim"]
    if len(passing) >= min_claim:
        composite = (
            median_stddev_composite(passing, vote_stddevs)
            if median_votes
            else weighted_composite(passing)
        )
    else:
        composite = None

    # Availability-verified weight share: the share of the weight that
    # actually priced THIS observation belonging to availability-verified
    # members -- a disclosure aggregate, never a screen. Added HERE and
    # not in median_stddev_composite because that helper also prices the
    # FROZEN daily series, whose artifacts must never grow the field.
    # Computed from the UNROUNDED passing weights (adversarial review:
    # summing the published 6dp renormalized weights can print 1.000002
    # with enough verified passers), so the share is bounded by 1.0 and
    # matches the sum of published rounded weights to within rounding.
    # Always a float on the wire (0.0, never integer 0). The list rides
    # calc_params (params, not live config), so published-stamp
    # recompute is byte-deterministic and a retune is a versioned change
    # the D2 fence owns. Dark hours carry no index and no share.
    if composite is not None:
        verified = set(params.get("availability_verified_sources") or [])
        passing_total = sum(weight for _, weight, _ in passing)
        composite["availability_verified_weight_share"] = round(
            sum(
                weight
                for source_id, weight, _ in passing
                if source_id in verified
            )
            / passing_total,
            6,
        )

    # Advance the observation-mode weight state with this observation's
    # pinned facts -- AFTER the vector was computed (nothing observed at t
    # enters t's weights), the same order replay applies.
    advance_panel_weight_state(
        weight_state,
        obs_stamp=obs_stamp,
        prints=trusted_prints,
        vector=weight_block["weights"],
        mode=weight_block["mode"],
        dw_params=dynamic_params,
    )

    return {
        "schema_version": PANEL_SCHEMA_VERSION,
        "kind": PANEL_COMPOSITE_KIND,
        "panel_id": config["panel_id"],
        "methodology_id": params["methodology_id"],
        # The resolved params travel with every artifact (the D2 pin);
        # embedded_calc_params is the ONE conversion the CLI compare
        # shares.
        "calc_params": embedded_calc_params(params),
        # Fixed-width observation key: lexicographic == chronological, so
        # the store's no-regress pointer and append-only composite keys
        # work unchanged (design section 1).
        "date": stamp_iso,
        "observation_date": obs_date,
        "observation_hour_utc": obs_hour,
        # A quarantined stamp is NOT missed: the record may hold bytes;
        # they are quarantined by config (F6), a recorded fact of its own.
        "observation_missed": snapshot is None and record_quarantined is None,
        "record_quarantined": record_quarantined,
        "panel_dark": composite is None,
        "index": composite,
        "sources": sources_block,
        "jump_screen": jump_block,
        "record_kind": record_entry["kind"],
        "snapshot_run_id": (snapshot or {}).get("run_id"),
        "snapshot_late_fill": (snapshot or {}).get("late_fill"),
        "weight_calc": weight_block,
    }
