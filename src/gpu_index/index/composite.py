# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Pure daily-composite calculation for the index baskets.

ONE calculator, parameterized by basket config: the B300 lane
(config/index_basket.json) and the B200 lane (config/index_basket_b200.json)
run the same functions. Basket-specific behavior comes ONLY from the config
— constituent role/weights, methodology id, per-source statistics, claim
minimum, FX lane — and every lane-specific calc parameter is a CONDITIONAL
calc_params key so the B300 lane's embedded params (and therefore its
artifact bytes) are unchanged by knobs it never set.

Every function here is a pure function of its inputs — zero I/O, zero
clock reads — so the whole index series is a deterministic replay of
(stored snapshots, stored FX records, config). Rules implemented (see
METHODOLOGY.md; the R*/D* labels below are pinned rule names):

  - R1 lowest NON-INTERRUPTIBLE per-GPU print wins per source (spot/
    preemptible excluded; committed tiers eligible — Latitude's monthly
    commit-equivalent beats its hourly);
  - R2 FX at calc time via the ECB reference rate, audit block embedded;
  - R3 filter warm-up: inert until 10 accepted observations (prints flagged
    ``unfiltered``); early prints >15% from the basket mean flagged
    ``manual_verify``;
  - R4 eager compute: day D publishes at the first run where D's canonical
    16:00Z snapshot exists, or nearest-slot promotion once the canonical
    window has closed; published composites are never revised;
  - the per-source outlier fence (METHODOLOGY.md §6.4): per-source mu±kσ over
    the trailing 20 observations; a failing print is held out same-day but
    still enters the window (genuine repricings cost exactly one day).
    calc_v3: under ``filter_terms: recorded_currency`` the window and the
    mu/sigma/deviation test run in the source's RECORDED currency — an FX
    move must never hold out a constituent whose native price did not move
    (observed live: a ~1% EURUSD move ejecting an unchanged EUR print) —
    and the band floors at ``filter_sigma_floor`` recorded-currency units:
    band = filter_sigma * max(sigma, filter_sigma_floor). The floor
    SUPERSEDES the retired sigma_zero accept-iff-deviation==0 special case
    (the ``sigma_zero`` flag remains, informational only). Percent-form
    floor (for NEW mints; mutually exclusive with the absolute key):
    ``filter_sigma_floor_pct`` floors the band at pct/100 of the
    trailing-window mean instead — one knob that scales across SKUs.
    Floor split: that key is FENCE-ONLY; the separate
    ``vote_sigma_floor_pct`` floors a vote's stddev at pct/100 of the
    print's own filter-terms price (only the legacy ABSOLUTE key still
    governs both sigmas, frozen semantics). Currency
    anomalies fail CLOSED (rule D1): a print whose filter input is
    untrusted (non-USD label with no native price, or an UNKNOWN/malformed
    label) is held out with the window preserved and never enters it; a
    TRUSTED print in a different currency than its window is held out
    (window preserved) until CURRENCY_CONFIRM_DAYS consecutive
    same-new-currency prints confirm a genuine change — only then is the
    old window discarded and the new-currency window seeded from the
    pending prints (still warm-up). Never a crash, never a mixed-currency
    window, and every such day publishes loudly (CLI WARNING + non-zero
    exit);
  - the legacy composite: weighted mean over passing constituents, weights
    renormalized to sum 1.0; zero passers → an explicit basket_dark record.
    calc_v4: under ``composite_statistic: median_ci_votes`` (a PINNED wire
    value; a later terminology change renamed only code and docs) the day
    instead prices as the median-of-votes aggregate (METHODOLOGY.md §7.1) —
    each passing source votes its weight at price and price +/- its
    standard deviation (its own trailing-window sigma in filter terms,
    floored at ``filter_sigma_floor`` — a frozen list price must not
    masquerade as conviction — FX-converted at the print's own rate), the
    index is the weighted median of all votes, and the published aggregate
    dispersion (the legacy ``confidence_usd_gpu_hr`` wire key) is the
    larger distance from the index to the 25th/75th weighted vote
    percentiles. The weighted mean stays in the artifact as a diagnostic.
    Screens, warm-up, R3, claim floors, fallback pool: all unchanged.

B200 lane differences (all pinned in its config):

  - NO warm-up publish gate: the series publishes from the FIRST canonical
    daily observation; sources without enough trailing history pass in the
    same warm-up/unfiltered status B300 uses (status "ok",
    filter.unfiltered=true, counted in sources_used_count);
  - claim minimum via calc.min_sources_to_claim (5-of-9 for B200): fewer
    passers than the floor publishes an explicit basket_dark artifact with
    index null — the same dark-day shape as zero passers;
  - per-source statistics via calc.source_statistics (the order-book rule,
    METHODOLOGY.md §6.2: vast prices as the volume-weighted median of
    rentable on-demand per-GPU asks across verified US/CA hosts, never the
    lowest print);
  - calc.fx_lane "none" for USD-only baskets: no ECB machinery; a non-USD
    print is held out fail-closed.
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import date, timedelta
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gpu_index.index.fx import (
    DEFAULT_FX_MAX_STALENESS_DAYS,
    FxUnavailableError,
    eur_to_usd,
)
from gpu_index.index.weights import (
    DEFAULT_TARGET_VARIANCE_FLOOR,
    advance_weight_state,
    compute_dynamic_weights,
    series_print,
)

# Mirrors gpu_index.index.config.DEFAULT_BASKET_ROLE (config lazy-imports this
# module from _validate, so the two constants are kept separate and pinned
# in lockstep by test rather than by import).
DEFAULT_BASKET_ROLE = "b300_basket"


def constituents(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The basket's constituent sources — THE one selection predicate.
    calc_params (fallback_weights) and compute_day (the weight domain)
    must never disagree about who is a constituent: compute_dynamic_weights
    indexes fallback_weights by compute_day's eligible sids, so a fork
    between two copies of this predicate is a KeyError at publish time."""
    basket_role = config.get("basket_role", DEFAULT_BASKET_ROLE)
    return [s for s in config["sources"] if s["role"] == basket_role]

CALC_METHODOLOGY_ID = "annex_a_v0_2_calc_v1"
COMPOSITE_SCHEMA_VERSION = 1
COMPOSITE_KIND = "index_basket_composite"

DEFAULT_INTERRUPTIBLE_TIERS = ("spot", "preemptible")
DEFAULT_FILTER_WINDOW = 20
DEFAULT_FILTER_SIGMA = 2.5
DEFAULT_FILTER_WARMUP = 10
DEFAULT_MANUAL_VERIFY_PCT = 15.0
DEFAULT_FALLBACK_POOL_SKU = "B200"
# calc_v3 knobs. Defaults preserve the legacy (calc_v1/v2)
# behavior exactly — the RULING lives in the configs, which set
# filter_terms "recorded_currency" and filter_sigma_floor 0.05 under their
# minted methodology ids.
DEFAULT_FILTER_SIGMA_FLOOR = 0.0
# Percent-form floor (superseding the absolute floor for NEW mints):
# filter_sigma_floor_pct is a PERCENT — the fence floors at pct/100 of the
# trailing-window mean — so the rule scales across SKUs whose prices differ
# by an order of magnitude. FENCE-ONLY since the floor split: the separate
# vote_sigma_floor_pct floors a vote's stddev at pct/100 of the print's own
# filter-terms price (only the legacy ABSOLUTE key still governs both
# sigmas, frozen semantics). Mutually exclusive with the absolute key at
# load; frozen series replay under whichever key their artifacts embed.
# This default is consumed ONLY by the panel loader's unconditional params
# slot (a config setting neither key); the daily basket's calc_params stays
# presence-gated (absent stays absent).
DEFAULT_FILTER_SIGMA_FLOOR_PCT = 3.0
DEFAULT_FILTER_TERMS = "usd"
VALID_FILTER_TERMS = ("usd", "recorded_currency")
# calc_v4 composite statistic. The default preserves the legacy
# weighted-mean composite exactly — the RULE lives in the
# configs, which set "median_ci_votes" under their minted methodology ids
# (annex_a_v0_2_calc_v4 / annex_a2_v0_3_calc_v3).
DEFAULT_COMPOSITE_STATISTIC = "weighted_mean"
MEDIAN_STDDEV_VOTES = "median_ci_votes"
VALID_COMPOSITE_STATISTICS = (DEFAULT_COMPOSITE_STATISTIC, MEDIAN_STDDEV_VOTES)
# Rule D1: a recorded-currency change is confirmed —
# and the incompatible old window replaced — only after this many
# CONSECUTIVE trusted prints in the same new currency. Verdicts embed the
# value (confirm_after) so old artifacts render era-correctly; changing it
# is a methodology change and requires a mint.
CURRENCY_CONFIRM_DAYS = 3
# Well-formed currency labels are ISO-4217-shaped (three ASCII letters,
# matched after strip+uppercase). "UNKNOWN"/""/None and anything else are
# NOT well-formed — those prints' filter inputs are untrusted (D1).
_CURRENCY_LABEL_RE = re.compile(r"[A-Z]{3}\Z")


def calc_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """The single resolution of calc parameters — the CLI and compute_day
    must consume THIS, never re-type defaults (a drifted default would let
    the exists-gate and the written payload disagree on methodology_id).

    Changing any value here without minting a NEW methodology_id silently
    recomputes historical filter decisions under the same series id — the
    params-pin test in test_index_composite.py fires on exactly that.
    """
    calc = config.get("calc") or {}
    params = {
        "methodology_id": calc.get("methodology_id", CALC_METHODOLOGY_ID),
        "target_sku": config.get("target_sku", "B300"),
        "interruptible_tiers": tuple(
            calc.get("interruptible_tiers", DEFAULT_INTERRUPTIBLE_TIERS)
        ),
        "filter_window": int(calc.get("filter_window", DEFAULT_FILTER_WINDOW)),
        "filter_sigma": float(calc.get("filter_sigma", DEFAULT_FILTER_SIGMA)),
        "filter_warmup": int(calc.get("filter_warmup", DEFAULT_FILTER_WARMUP)),
        "manual_verify_pct": float(
            calc.get("manual_verify_pct", DEFAULT_MANUAL_VERIFY_PCT)
        ),
        "promote_tie_break": calc.get("promote_tie_break", "later"),
        "fx_max_staleness_days": int(
            calc.get("fx_max_staleness_days", DEFAULT_FX_MAX_STALENESS_DAYS)
        ),
        "drift_scan_days": int(calc.get("drift_scan_days", 14)),
        "fallback_pool_sku": calc.get(
            "fallback_pool_sku", DEFAULT_FALLBACK_POOL_SKU
        ),
        "fallback_pool_sources": list(
            calc.get(
                "fallback_pool_sources",
                ["nebius", "e2e", "shadeform"],
            )
        ),
        # calc_v2: hand-audited (date, source) exclusions for
        # data-integrity incidents where the recorded print is known-wrong
        # by rule and the true print was never captured (exclusion is the
        # only honest replay — substitution would fabricate unrecorded
        # data). Sorted for deterministic artifact bytes; embedded in every
        # artifact via calc_params like every other parameter. Adding an
        # exclusion for an already-published day of the SAME methodology
        # requires a NEW methodology_id — published days pin.
        "manual_exclusions": sorted(
            (
                {
                    "date": str(e["date"]),
                    "source_id": str(e["source_id"]),
                    "reason": str(e["reason"]),
                }
                for e in calc.get("manual_exclusions", [])
            ),
            key=lambda e: (e["date"], e["source_id"]),
        ),
    }
    # Lane knobs — present ONLY when the config sets them. These
    # must never grow defaults here: calc_params is embedded verbatim in
    # every artifact, so an unconditional key would change the B300 lane's
    # artifact BYTES mid-series without a methodology mint.
    if "min_sources_to_claim" in calc:
        # B200 lane rule: fewer passers than this floor
        # publishes an explicit basket_dark day (index null).
        params["min_sources_to_claim"] = int(calc["min_sources_to_claim"])
    if "source_statistics" in calc:
        # source_id -> statistic id (SOURCE_STATISTIC_FNS). Sorted so the
        # embedded params are independent of config key order.
        params["source_statistics"] = {
            str(sid): str(stat)
            for sid, stat in sorted((calc["source_statistics"] or {}).items())
        }
    if "fx_lane" in calc:
        # "none" = USD-only basket: the CLI skips the ECB
        # lane entirely; any non-USD print is held out fail-closed.
        params["fx_lane"] = str(calc["fx_lane"])
    # calc_v3 filter knobs — conditional like the lane knobs above, so a
    # legacy-methodology config keeps its exact artifact bytes (an
    # unconditional key would fork a frozen series' embedded params without
    # a mint). Both live configs set them as of annex_a_v0_2_calc_v3 /
    # annex_a2_v0_3_calc_v2.
    if "filter_terms" in calc:
        # "recorded_currency" (calc_v3, R-native): the trailing window and
        # test run on each source's NATIVE print, so FX movement cannot
        # hold out a constituent whose recorded price never moved. USD
        # sources are numerically unchanged; cross-source math (the
        # composite, R3 manual_verify) stays in USD.
        params["filter_terms"] = str(calc["filter_terms"])
    if "filter_sigma_floor" in calc:
        # calc_v3 (R-floor): band = filter_sigma * max(sigma,
        # filter_sigma_floor), in recorded-currency units — supersedes the
        # retired sigma_zero accept-iff-deviation==0 special case.
        params["filter_sigma_floor"] = float(calc["filter_sigma_floor"])
    if "filter_sigma_floor_pct" in calc:
        # Percent-form floor: the fence band floors at pct/100 of the
        # trailing-window mean; a vote's stddev floors at pct/100 of the
        # print's own filter-terms price. Mutually exclusive with the
        # absolute key (enforced at load), conditional like every lane
        # knob: frozen series keep their artifact bytes.
        params["filter_sigma_floor_pct"] = float(calc["filter_sigma_floor_pct"])
    if "vote_sigma_floor_pct" in calc:
        # Floor split: filter_sigma_floor_pct is FENCE-ONLY; this key
        # floors the median-vote band at pct/100 of the print's OWN
        # filter-terms price. Load-validated to require median_ci_votes
        # and to refuse the absolute floor alongside (the absolute key's
        # frozen semantics govern both sigmas). Conditional like every
        # lane knob: frozen series keep their artifact bytes — today's
        # daily configs are frozen absolute, so this is vocabulary +
        # future-mint parity with the panel engine.
        params["vote_sigma_floor_pct"] = float(calc["vote_sigma_floor_pct"])
    if "composite_statistic" in calc:
        # calc_v4: "median_ci_votes" (pinned wire value; a later
        # terminology change renamed only code and docs) supersedes the
        # legacy weighted mean with the median-of-votes
        # aggregate — every passing source votes its weight at price and
        # price +/- its standard deviation (its trailing-window sigma, floored at
        # filter_sigma_floor); the index is the weighted median of all the
        # votes. Conditional like every lane knob: frozen series must keep
        # their exact artifact bytes.
        params["composite_statistic"] = str(calc["composite_statistic"])
    if "iqm_alpha" in calc:
        # The composite prices the day at the interquantile mean — the
        # mean of the weighted vote band [1/2 - alpha, 1/2 + alpha] —
        # instead of the point median (alpha 0 IS the median branch,
        # bit-for-bit; 1/2 is the full weighted mean of the votes).
        # Load-validated to [0, 0.5] and to require the median_ci_votes
        # statistic. Conditional like every lane knob: frozen series keep
        # their exact artifact bytes.
        params["iqm_alpha"] = float(calc["iqm_alpha"])
    if "dynamic_weights" in calc:
        # calc_v5 mint (dynamic predictive weighting): the day's weight
        # vector is DERIVED — predictiveness
        # scores from the published series through day t-1, softmax
        # allocation with a floor, per-source risk caps, and a global cap
        # — with the config's opening weights as the warm-up fallback
        # until every eligible source has a defined score (the weights
        # module is the single home of the semantics). EVERYTHING that shapes a
        # weight rides here — knobs, per-source risk caps, AND the
        # fallback vector itself (sorted for deterministic bytes) — so the
        # D2 refuse-to-extend fence covers a weight-methodology edit
        # exactly like any other param: before this mint, sources[].weight
        # sat OUTSIDE calc_params and could drift a published series
        # silently.
        dw = calc["dynamic_weights"] or {}
        params["dynamic_weights"] = {
            "scheme": str(dw["scheme"]),
            # R-slots: horizons in HOURS on the
            # capture slot grid ({6, 24, 48}; 1h joins when capture walks
            # to hourly snapshots). The grid itself is embedded so a
            # capture-cadence change is a MINT, never a silent shift of the
            # R-cutoff boundary or the sample anchors.
            "lookback_horizons_hours": [
                int(x) for x in dw["lookback_horizons_hours"]
            ],
            "forward_horizons_hours": [
                int(x) for x in dw["forward_horizons_hours"]
            ],
            "slot_hours_utc": sorted(
                int(h) for h in config["capture_slots_utc"]
            ),
            "history_days": int(dw["history_days"]),
            "half_life_days": float(dw["half_life_days"]),
            "ridge_lambda": float(dw["ridge_lambda"]),
            "gamma": float(dw["gamma"]),
            "weight_min": float(dw["weight_min"]),
            "weight_max": float(dw["weight_max"]),
            # R-insample: q = max(0, weighted
            # in-sample R^2) of the single fit — min_train_samples is the
            # only sample gate (supersedes the earlier walk-forward
            # out-of-sample gate; no OOS evaluation points exist).
            "min_train_samples": int(dw["min_train_samples"]),
            "target_variance_floor": float(
                dw.get(
                    "target_variance_floor", DEFAULT_TARGET_VARIANCE_FLOOR
                )
            ),
            # R-quorum: the eligible-count leg of the switch gate. Default
            # 1 = the pure-schema posture; both live configs pin their
            # capture claim floor (5).
            "switch_min_eligible": int(dw.get("switch_min_eligible", 1)),
            # R-winsor: |log return| cap on every feature/LOO leg. null =
            # unclamped (pure schema); both live configs pin 0.5.
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
            "fallback_weights": {
                s["source_id"]: float(s["weight"])
                for s in sorted(
                    constituents(config), key=lambda s: s["source_id"]
                )
            },
        }
    return params


# --------------------------------------------------------------- slot choice


def select_slot_snapshot(
    day_snapshots: Dict[int, Dict[str, Any]],
    *,
    canonical_hour: int,
    window_closed: bool,
    tie_break: str = "later",
) -> Optional[Tuple[Dict[str, Any], Optional[int]]]:
    """(snapshot, substituted_from_hour|None) per rule R4, or None when the
    day is not yet computable (canonical missing, window still open)."""
    if canonical_hour in day_snapshots:
        return day_snapshots[canonical_hour], None
    if not window_closed or not day_snapshots:
        return None
    hours = sorted(
        day_snapshots,
        key=lambda h: (
            abs(h - canonical_hour),
            -h if tie_break == "later" else h,
        ),
    )
    chosen = hours[0]
    return day_snapshots[chosen], chosen


# ------------------------------------------------------- observation choice


def daily_source_observation(
    source_entry: Dict[str, Any],
    *,
    sku: str,
    day: str,
    fx_records: Dict[str, Dict[str, Any]],
    interruptible_tiers: Sequence[str],
    fx_max_staleness_days: int = 7,
) -> Optional[Dict[str, Any]]:
    """The source's daily print: lowest eligible per-GPU price for ``sku``.

    Eligible = matching sku, non-interruptible tier (R1), plausible, and
    carrying a usable price. Non-USD prints convert via the ECB record for
    <=day BEFORE the minimum is taken (mixed-currency sources compare in
    USD). Returns None when the source has no eligible print (including
    when FX is unavailable — held out loudly by the caller).
    """
    if source_entry.get("status") != "ok":
        return None
    candidates: List[Dict[str, Any]] = []
    fx_errors: List[str] = []
    for obs in source_entry.get("observations") or []:
        if obs.get("sku") != sku:
            continue
        if obs.get("tier") in interruptible_tiers:
            continue
        if obs.get("implausible"):
            continue
        usd = obs.get("price_usd_gpu_hr")
        fx_block: Dict[str, Any] = {}
        if usd is None:
            native = obs.get("price_native_per_gpu_hr")
            currency = obs.get("currency")
            if native is None or currency in (None, "USD", "UNKNOWN"):
                continue  # no honest way to price this print
            if currency != "EUR":
                continue  # only EUR conversion is ruled (R2); extend by ruling
            try:
                usd, fx_block = eur_to_usd(
                    float(native),
                    fx_records,
                    day,
                    max_staleness_days=fx_max_staleness_days,
                )
            except FxUnavailableError as exc:
                fx_errors.append(str(exc))
                continue
        candidates.append(
            {
                "usd_per_gpu_hr": round(float(usd), 6),
                "tier": obs.get("tier"),
                "gpu_count_basis": obs.get("gpu_count_basis"),
                "raw_value": obs.get("raw_value"),
                "raw_unit": obs.get("raw_unit"),
                "currency": obs.get("currency", "USD"),
                "native_per_gpu_hr": obs.get("price_native_per_gpu_hr"),
                "notes": obs.get("notes", ""),
                **fx_block,
            }
        )
    if not candidates:
        if fx_errors:
            return {"fx_unavailable": True, "fx_errors": fx_errors}
        return None
    chosen = min(candidates, key=lambda c: c["usd_per_gpu_hr"])
    chosen["n_eligible_prints"] = len(candidates)
    if fx_errors:
        # A mixed-currency source under an FX outage still prices from its
        # USD prints, but the dropped candidates must be VISIBLE — the
        # minimum may be silently too high that day.
        chosen["fx_errors_partial"] = fx_errors
    return chosen


# ----------------------------------------------- per-source statistics (§5)


def us_ca_verified_host(verification: Any, region: Any) -> bool:
    """The order-book population screen (METHODOLOGY.md §6.2): a VERIFIED host in a US/CA
    region (country = the token after the region string's last comma —
    vast geolocations read 'Oregon, US', ', CA', 'Taiwan, TW').

    ONE home for the screen, shared by the calc (vast_vwm_verified_us_ca)
    and the vast capture's population recording (the sources module) — if
    the two screens could drift, the capture could record a population
    NARROWER than the statistic's claim: exactly the truncation defect a
    shared home exists to prevent.
    """
    if verification != "verified":
        return False
    country = str(region or "").rsplit(",", 1)[-1].strip().upper()
    return country in ("US", "CA")


def _weighted_median(pairs: Sequence[Tuple[float, float]]) -> float:
    """Weighted median over (value, weight>0) pairs, generalizing
    ``statistics.median``: the value at cumulative weight W/2, averaging the
    two straddling values when the half-point lands exactly on a boundary
    (so all-weights-1 reproduces statistics.median bit-for-bit)."""
    ordered = sorted(pairs)
    half = sum(w for _, w in ordered) / 2.0
    cum = 0.0
    for i, (value, weight) in enumerate(ordered):
        cum += weight
        if cum > half:
            return value
        if cum == half:
            return (value + ordered[i + 1][0]) / 2.0
    raise ValueError("weighted median of an empty/zero-weight book")


def vast_vwm_verified_us_ca(
    source_entry: Dict[str, Any],
    *,
    statistic: str,
    sku: str,
    interruptible_tiers: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """The vast order-book statistic (METHODOLOGY.md §6.2) — specified, not
    inherited: the volume-weighted median of rentable on-demand asks across
    VERIFIED US/CA hosts, per-GPU, from the captured deduped book (cheapest
    offer per machine). The capture records the FULL verified-US/CA
    population (cheapest-5 legacy rows + population rows + book_stats), and
    this statistic consumes every eligible recorded row — the recording
    must match the statistic's claim. A truncated capture is handled as a
    calc.manual_exclusions incident.

    Pinned interpretation choices: the raw book publishes instance totals
    ("dph_total") but the index unit requires per-GPU terms, so the
    statistic runs over the capture's normalized price_usd_gpu_hr
    (= dph_total / stated GPU count); "volume" = the offer's rentable GPU
    count (gpu_count_basis); count-4 marketplace slices are eligible
    (4-GPU slices of genuine 8-GPU servers are ordinary listings) exactly
    as B300's calculator prices vast slices. "deverified" is NOT
    "verified". USD-only lane: a priceless/non-USD row never enters.
    """
    if source_entry.get("status") != "ok":
        return None
    book: List[Tuple[float, float]] = []
    for obs in source_entry.get("observations") or []:
        if obs.get("sku") != sku:
            continue
        if obs.get("tier") in interruptible_tiers:
            continue
        if obs.get("implausible"):
            continue
        usd = obs.get("price_usd_gpu_hr")
        if usd is None:
            continue
        if not us_ca_verified_host(obs.get("verification"), obs.get("region")):
            continue
        book.append((float(usd), obs.get("gpu_count_basis") or 1))
    if not book:
        return None
    return {
        "usd_per_gpu_hr": round(_weighted_median(book), 6),
        "statistic": statistic,
        "currency": "USD",
        "n_eligible_prints": len(book),
        "gpu_volume": sum(w for _, w in book),
    }


# Registry: calc.source_statistics values must name one of these. Changing
# what an id COMPUTES without minting a new methodology_id is the same
# crime as editing any other calc param — the id rides calc_params into
# every artifact's bytes.
SOURCE_STATISTIC_FNS = {
    "vast_vwm_verified_us_ca": vast_vwm_verified_us_ca,
}


def resolve_daily_print(
    source_entry: Optional[Dict[str, Any]],
    *,
    source_id: str,
    params: Dict[str, Any],
    sku: str,
    day: str,
    fx_records: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The source's daily print under THIS basket's methodology: the
    configured per-source statistic when calc.source_statistics names one,
    else the default lowest-eligible-print rule (R1). Used by BOTH
    compute_day and the drift detector — a drift scan pricing a statistic
    source by the min rule would warn forever on every published day."""
    if source_entry is None:
        return None
    stat_id = (params.get("source_statistics") or {}).get(source_id)
    if stat_id is not None:
        return SOURCE_STATISTIC_FNS[stat_id](
            source_entry,
            statistic=stat_id,
            sku=sku,
            interruptible_tiers=params["interruptible_tiers"],
        )
    return daily_source_observation(
        source_entry,
        sku=sku,
        day=day,
        fx_records=fx_records,
        interruptible_tiers=params["interruptible_tiers"],
        fx_max_staleness_days=params["fx_max_staleness_days"],
    )


def resolve_slot_prints(
    day_snapshots: Dict[int, Dict[str, Any]],
    *,
    config: Dict[str, Any],
    day: str,
    fx_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{slot_hour_str: {source_id: series print}} for one CLOSED day's raw
    snapshots — the weight lane's slot-grid feed (rule R-slots).

    Each slot's per-source print resolves with the SAME machinery as the
    daily print — resolve_daily_print (R1 lowest-eligible / the configured
    per-source statistic), FX at the day's ECB record, and the
    filter_observation trust rules (untrusted-currency prints never enter;
    fx_unavailable and failed sources are gaps) — and manual_exclusions
    for (day, source) hold their slots out exactly as they hold the daily
    print out (a known-wrong recording must not feed the weight lane
    either). The daily sigma-fence is NOT applied at slot level (it is a
    daily-print adjudication; R-winsor bounds what any slot print can do
    to the estimator). Keys are strings so the block embeds in artifact
    JSON verbatim (weight_calc.slot_prints pins it for replay).
    """
    params = calc_params(config)
    filter_terms = params.get("filter_terms", DEFAULT_FILTER_TERMS)
    excluded = {
        e["source_id"]
        for e in params["manual_exclusions"]
        if e["date"] == day
    }
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for hour in sorted(day_snapshots):
        snapshot = day_snapshots[hour]
        entries = {
            s["source_id"]: s for s in (snapshot or {}).get("sources", [])
        }
        slot: Dict[str, Dict[str, Any]] = {}
        for source in constituents(config):
            source_id = source["source_id"]
            if source_id in excluded:
                continue
            chosen = resolve_daily_print(
                entries.get(source_id),
                source_id=source_id,
                params=params,
                sku=params["target_sku"],
                day=day,
                fx_records=fx_records,
            )
            if chosen is None or chosen.get("fx_unavailable"):
                continue
            observation = filter_observation(chosen, filter_terms=filter_terms)
            if observation is None:
                continue
            slot[source_id] = series_print(
                chosen["usd_per_gpu_hr"], observation
            )
        if slot:
            out[str(int(hour))] = slot
    return out


# ----------------------------------------------------------------- filter


def filter_observation(
    chosen: Dict[str, Any], *, filter_terms: str = DEFAULT_FILTER_TERMS
) -> Optional[Tuple[float, str]]:
    """The (price, currency) the §6.4 filter operates on for a print,
    or None when the print's filter input is UNTRUSTED (ruling D1).

    Under "recorded_currency" (calc_v3, R-native) an FX-converted
    print filters in its NATIVE terms — on 2026-08-20 scaleway was held out
    purely because EURUSD moved ~1% against a sigma-of-1.6-USD-cents
    window while its native price sat unchanged at 7.50 EUR. USD prints
    are numerically identical in both modes.

    Fail-closed rules (rule D1): native terms apply
    ONLY when the recorded currency is a well-formed non-USD label (three
    ASCII letters after strip+uppercase; "UNKNOWN"/""/malformed are not)
    AND a numeric native price is present. currency None/"USD" filters in
    USD terms. Anything else — a non-USD label with the native price
    missing, or an unrecognizable label — returns None: the print has NO
    trustworthy filter input and the caller holds it out with the window
    preserved. ONE home for the rule: compute_day and the CLI's
    replay-from-published window advance must never disagree about what
    entered a window.
    """
    usd = float(chosen["usd_per_gpu_hr"])
    if filter_terms != "recorded_currency":
        return usd, "USD"
    raw = chosen.get("currency")
    if raw is None:
        return usd, "USD"
    label = str(raw).strip().upper()
    if label == "USD":
        return usd, "USD"
    native = chosen.get("native_per_gpu_hr")
    # "UNKNOWN" and "" fail the three-letter shape; a well-formed non-USD
    # label additionally needs a numeric native price to be trustworthy.
    if (
        _CURRENCY_LABEL_RE.fullmatch(label)
        and isinstance(native, (int, float))
        and not isinstance(native, bool)
    ):
        return float(native), label
    return None  # untrusted: no honest value exists in window terms


def window_incompatible(
    window_history: Dict[str, List[float]],
    window_currencies: Optional[Dict[str, str]],
    source_id: str,
    currency: str,
) -> bool:
    """THE one predicate (ruling D2) for a currency-incompatible window: a
    non-empty trailing window recorded in a DIFFERENT currency than
    today's trusted print. Used by both the verdict path and the
    window-advance path so the two can never disagree."""
    if window_currencies is None:
        return False
    prior = window_currencies.get(source_id)
    return bool(
        prior is not None
        and prior != currency
        and window_history.get(source_id)
    )


def _pending_after(
    pending_currencies: Optional[Dict[str, Dict[str, Any]]],
    source_id: str,
    price: float,
    currency: str,
) -> List[float]:
    """The consecutive same-new-currency print streak INCLUDING today's
    print (ruling D1's confirmation counter). A different pending currency
    restarts the streak — one encoding, used by both the verdict path and
    the advance path."""
    pend = (pending_currencies or {}).get(source_id)
    if pend and pend["currency"] == currency:
        return list(pend["prints"]) + [price]
    return [price]


def advance_window(
    window_history: Dict[str, List[float]],
    window_currencies: Optional[Dict[str, str]],
    pending_currencies: Optional[Dict[str, Dict[str, Any]]],
    source_id: str,
    observation: Optional[Tuple[float, str]],
) -> None:
    """Advance one source's filter state with today's print (§6.4 rule:
    accepted AND held-out prints both enter their window, so the filter
    adapts) under the D1 fail-closed currency rules:

      - observation None (UNTRUSTED print): nothing enters — no
        trustworthy value exists in window terms (a documented divergence
        from the every-real-print rule) — and any pending
        currency-change streak is broken; the window is PRESERVED.
      - trusted print in the window's currency: appended (window rule);
        any pending streak is cleared (the source came back).
      - trusted print in a DIFFERENT currency: does NOT enter the old
        window (mixed-currency windows must never exist); it joins the
        pending streak, and the CURRENCY_CONFIRM_DAYS-th consecutive
        same-new-currency print confirms the change — the old window is
        discarded and the new-currency window is seeded from the pending
        prints (still warm-up).

    ``window_currencies``/``pending_currencies`` must be the same dicts
    across days for multi-day replays (the CLI); the state is
    reconstructable from published artifacts alone, so replays are
    deterministic.
    """
    if observation is None:
        if pending_currencies is not None:
            pending_currencies.pop(source_id, None)
        return
    price, currency = observation
    if window_incompatible(window_history, window_currencies, source_id, currency):
        queued = _pending_after(pending_currencies, source_id, price, currency)
        if len(queued) >= CURRENCY_CONFIRM_DAYS:
            window_history[source_id] = queued
            window_currencies[source_id] = currency
            if pending_currencies is not None:
                pending_currencies.pop(source_id, None)
        elif pending_currencies is not None:
            pending_currencies[source_id] = {
                "currency": currency,
                "prints": queued,
            }
        return
    if pending_currencies is not None:
        pending_currencies.pop(source_id, None)
    if window_currencies is not None:
        window_currencies[source_id] = currency
    window_history.setdefault(source_id, []).append(price)


def evaluate_filter(
    window_history: Sequence[float],
    price: float,
    *,
    window: int = DEFAULT_FILTER_WINDOW,
    sigma: float = DEFAULT_FILTER_SIGMA,
    warmup: int = DEFAULT_FILTER_WARMUP,
    sigma_floor: float = DEFAULT_FILTER_SIGMA_FLOOR,
    sigma_floor_pct: Optional[float] = None,
    currency: Optional[str] = None,
) -> Dict[str, Any]:
    """The §6.4 fence verdict for one print against ITS OWN trailing window.

    ``window_history`` holds every REAL prior print — accepted AND held-out
    (§6.4 rule: a confirmed repricing enters the window so the filter
    adapts). Do NOT "fix" this by excluding held-out prints; that breaks
    the adaptation rule. σ is the POPULATION standard deviation (pstdev) —
    a pinned calc_v1 methodology choice.

    calc_v3 (R-floor): band = sigma * max(σ, ``sigma_floor``), in
    recorded-currency units, so many days of an identical print can no
    longer arm hair-trigger rejection. This SUPERSEDES the retired
    sigma_zero accept-iff-deviation==0 special case; the ``sigma_zero``
    flag remains, informational only. (At sigma_floor 0 the retired rule
    and ``deviation <= 0`` are the same test, so legacy-methodology
    replays are bit-identical.) ``currency`` names the filter's operating
    currency: when given (the recorded-currency mode) the verdict records
    it. The explicit fence bounds band/lo/hi (legibility rule) are
    recorded whenever the fence is non-legacy — ``currency`` given OR
    the resolved floor > 0 (the floor changes the band even in USD terms);
    at the legacy defaults (floor 0, currency None) both are left out so
    legacy-methodology verdicts keep their published byte shape.

    Percent-form floor: ``sigma_floor_pct`` given (mutually exclusive with
    ``sigma_floor``, enforced at config load) floors the band at pct/100
    of MU — the trailing-window mean, the corridor's own center, so the
    fence stays symmetric around mu and the floor cannot be moved by the
    print under judgment. The floor scale then rides the instrument's
    price level instead of a per-SKU absolute rule. Resolution happens
    HERE and nowhere else: mu exists only inside this function, and the
    floor only binds post-warm-up, where mu is always defined. Percent
    mode fails CLOSED on mu <= 0 (a non-positive window mean is garbage
    upstream; mu*(pct/100) would silently disarm the floor exactly when
    the window is degenerate), and always records the fence bounds
    band/lo/hi — the verdict's byte shape is keyed on the CONFIG (the pct
    key's presence), never on the data. Legacy mode (absolute floor / no
    pct key) is byte-identical to before the percent rule.
    """
    extra = {} if currency is None else {"currency": currency}
    if len(window_history) < warmup:
        return {
            "accepted": True,
            "unfiltered": True,
            "n_history": len(window_history),
            **extra,
        }
    tail = list(window_history[-window:])
    mu = float(statistics.mean(tail))
    sd = float(statistics.pstdev(tail))
    if sigma_floor_pct is not None and mu <= 0:
        raise ValueError(
            f"evaluate_filter: sigma_floor_pct requires a positive "
            f"trailing-window mean, got mu={mu!r} — a non-positive mean "
            "is garbage upstream and would disarm the percent floor"
        )
    floor = (
        mu * (sigma_floor_pct / 100.0)
        if sigma_floor_pct is not None
        else sigma_floor
    )
    band = sigma * max(sd, floor)
    deviation = abs(price - mu)
    verdict = {
        "accepted": deviation <= band,
        "unfiltered": False,
        "mu": round(mu, 6),
        "sigma": round(sd, 6),  # raw sigma, always — the floor shows in band
        "deviation": round(deviation, 6),
        "n_history": len(tail),
        **extra,
    }
    if currency is not None or sigma_floor_pct is not None or floor > 0:
        verdict["band"] = round(band, 6)
        verdict["lo"] = round(mu - band, 6)
        verdict["hi"] = round(mu + band, 6)
    if sd == 0.0:
        verdict["sigma_zero"] = True
    return verdict


# --------------------------------------------------------------- composite


def _weighted_quantile(
    pairs: Sequence[Tuple[float, float]], q: Fraction
) -> float:
    """Value at cumulative weight q*W over (value, weight>0) pairs, with the
    SAME boundary convention as ``_weighted_median``: averaging the two
    straddling values when q*W lands exactly on a boundary.

    Weights accumulate as EXACT decimal fractions — ``Fraction(str(w))``
    is the weight as WRITTEN in the config, not its binary float
    approximation — so a boundary fires exactly when exact decimal
    arithmetic says it does. This is load-bearing: configured weights are decimals like
    0.15, whose float sums land a hair off the exact boundary, and which
    side of a boundary the index takes can move it whole cents — float
    dust must never price a day. At all-weights-1 (and any
    exactly-representable weights) q=1/2 reproduces ``_weighted_median``
    bit-for-bit — pinned by test. Kept a SEPARATE function: that helper
    prices the frozen B200 vast statistic and a composite-statistic lane
    must never touch its bytes."""
    if not 0 < q < 1:
        raise ValueError(f"quantile q must be in (0, 1), got {q}")
    if any(w <= 0 for _, w in pairs):
        raise ValueError("weighted quantile requires positive weights")
    ordered = sorted(pairs)
    target = q * sum(
        (Fraction(str(w)) for _, w in ordered), start=Fraction(0)
    )
    cum = Fraction(0)
    for i, (value, weight) in enumerate(ordered):
        cum += Fraction(str(weight))
        if cum > target:
            return value
        if cum == target:
            return (value + ordered[i + 1][0]) / 2.0
    raise ValueError("weighted quantile of an empty/zero-weight book")


def _interquantile_mean(
    pairs: Sequence[Tuple[float, float]], alpha: Fraction
) -> Fraction:
    """The mean of the weighted quantile band [1/2 - alpha, 1/2 + alpha]
    over (value, weight>0) pairs — the interquantile (symmetrically
    trimmed) mean. Each vote contributes its value in proportion to how
    much of its cumulative-weight span lies inside the band (fractional at
    the band edges), so the aggregate moves continuously with every vote
    in the central band, while votes wholly outside it still have no
    direct effect — the median's robustness with the mean's
    responsiveness, dialed by alpha. alpha -> 0 recovers the weighted
    median and alpha = 1/2 the weighted mean of the votes; the caller
    takes the alpha == 0 branch through ``_weighted_quantile`` so the
    degenerate case stays bit-for-bit the frozen median path.

    Everything here is EXACT rational arithmetic: weights via the
    ``Fraction(str(w))`` doctrine (float dust must never price a day), and
    vote VALUES via their shortest-repr decimals — the SAME rational the
    TS mirror derives from ``String(number)``, so both sides agree to the
    last digit. Returns the exact Fraction; the caller quantizes once."""
    if not 0 < alpha <= Fraction(1, 2):
        raise ValueError(f"iqm alpha must be in (0, 1/2], got {alpha}")
    if any(w <= 0 for _, w in pairs):
        raise ValueError("interquantile mean requires positive weights")
    ordered = sorted(pairs)
    total = sum(
        (Fraction(str(w)) for _, w in ordered), start=Fraction(0)
    )
    if total == 0:
        raise ValueError("interquantile mean of an empty/zero-weight book")
    lo = (Fraction(1, 2) - alpha) * total
    hi = (Fraction(1, 2) + alpha) * total
    acc = Fraction(0)
    cum = Fraction(0)
    for value, weight in ordered:
        prev = cum
        cum += Fraction(str(weight))
        overlap = min(cum, hi) - max(prev, lo)
        if overlap > 0:
            acc += Fraction(str(value)) * overlap
    return acc / (hi - lo)


def vote_stddev(
    vote_tail: Sequence[float],
    *,
    sigma_floor: float,
    fx_factor: float,
    sigma_floor_pct: Optional[float] = None,
    filter_price: Optional[float] = None,
) -> Dict[str, Any]:
    """calc_v4: one passing source's vote standard deviation (this band
    was formerly called the vote's "confidence interval"; the artifact
    keys conf_usd_gpu_hr / confidence_usd_gpu_hr are PINNED wire format
    from the frozen series and keep their names).

    sigma is the POPULATION stdev of the CALLER-SUPPLIED ``vote_tail`` —
    which history that is is the caller's (mode-dependent) contract: the
    outlier-fence window on legacy/daily lanes, the source's trailing
    dynamic-weights history under calc.vote_sigma_source "dw_history"
    (panel lanes). In every mode the tail is the PRE-advance history (a
    print never self-reports its own dispersion) in the filter's operating
    currency. The floor is ``filter_sigma_floor`` in those same units (the
    calc_v3 floor): a frozen list price has sigma 0, and without the floor
    its three coincident votes would claim infinite conviction the source
    never earned — staleness is not conviction. The floored value converts
    to USD at the print's own fx rate (cross-source votes must be
    common-currency). Fewer than two window entries
    (day-one, early warm-up) means sigma 0 → the floor IS the interval.

    Percent-form floor: ``sigma_floor_pct`` given (mutually exclusive with
    the absolute key at config load) floors the vote's stddev at pct/100
    of ``filter_price`` — the print's OWN price in the filter's operating
    currency, i.e. the value the vote is cast at. Floor split: the value
    callers pass here is the VOTE floor ``vote_sigma_floor_pct`` — the
    fence's ``filter_sigma_floor_pct`` is fence-only and never reaches
    this function; only the legacy ABSOLUTE ``sigma_floor`` still governs
    both sigmas (frozen semantics). The print anchors here, NOT the window
    mean the fence uses: a day-one vote has an empty tail (no mean
    exists), and "a source cannot claim conviction tighter than pct of its
    own quote" is the rule's unit. ``filter_price`` is required in percent
    mode and must be > 0: a non-positive print would resolve a floor of 0
    and mint a conf-0 vote — infinite conviction off a garbage price — so
    it fails loudly instead (the floor>0 invariant the median_ci_votes
    gate is built on).
    """
    if sigma_floor_pct is not None:
        if filter_price is None:
            raise ValueError(
                "vote_stddev: sigma_floor_pct requires filter_price "
                "(the percent floor anchors on the print being voted)"
            )
        if filter_price <= 0:
            raise ValueError(
                "vote_stddev: sigma_floor_pct requires filter_price > 0 "
                f"(got {filter_price!r}) — a non-positive print would "
                "disarm the floor and vote with conviction it never earned"
            )
        floor_native = filter_price * (sigma_floor_pct / 100.0)
    else:
        floor_native = sigma_floor
    sigma = float(statistics.pstdev(vote_tail)) if len(vote_tail) >= 2 else 0.0
    stddev_native = max(sigma, floor_native)
    return {
        "sigma": round(sigma, 6),
        "sigma_floored": sigma < floor_native,
        "conf_usd_gpu_hr": round(stddev_native * fx_factor, 6),
    }


def median_stddev_composite(
    passing: Sequence[Tuple[str, float, float]],
    vote_stddevs: Dict[str, float],
    *,
    iqm_alpha: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """calc_v4, superseding the legacy weighted mean: the median-of-votes
    aggregate (adapted from Pyth Network's price-aggregation design). Each
    passing source casts its weight at three values — its price
    and its price +/- its standard deviation — and the index is the
    weighted median of all the votes: a tight-interval source concentrates
    its weight, a wide-interval source spreads it, and no source can drag
    the index past what the vote quantiles allow, however confident it
    claims to be. The published aggregate dispersion (legacy
    confidence_usd_gpu_hr wire key) is the larger of the
    distances from the index to the 25th/75th weighted vote percentiles
    (widens exactly when sources disagree). Votes use each source's ROUNDED
    published standard deviation (the day's ``vote`` blocks), so the aggregate is
    recomputable from the artifact alone. The prior weighted mean stays in
    the artifact as a diagnostic, never the headline.

    ``iqm_alpha`` (the calc.iqm_alpha knob): nonzero prices the
    day at the interquantile mean of the SAME ballot — the mean of the
    weighted vote band [1/2 - alpha, 1/2 + alpha] — instead of the point
    median (see ``_interquantile_mean``). 0 (every mint through calc_v6 /
    a2 calc_v5; the default when the config elides the knob) takes the
    frozen median branch untouched, bit-for-bit. When nonzero, the block
    echoes the alpha and keeps the point median as a diagnostic
    (``vote_median_usd_gpu_hr``), so a tuned day remains recomputable and
    comparable from the artifact alone; the published dispersion anchors
    at the PUBLISHED (rounded) value. TWO rounding regimes coexist here,
    deliberately, and a mirror must route them per key: the HEADLINE's
    exact band-mean rational is quantized once (half-even at 6dp) and
    never passes through a raw float, while the vote_median diagnostic
    keeps the frozen path's float round() — it exists to print exactly
    what the alpha-0 branch would have published for the same ballot
    (pinned by test at an exact-boundary midpoint where the two regimes
    genuinely differ: 1.0000005 prints 1.0 as a band mean but 1.000001 as
    the frozen median)."""
    if not passing:
        return None
    votes: List[Tuple[float, float]] = []
    for source_id, weight, price in passing:
        stddev = vote_stddevs[source_id]
        # The DERIVED vote values are guarded too (the review): a
        # finite price and finite stddev can still overflow price ± stddev
        # to an infinity, which the sort would seat at the ladder's edge
        # and price a silently plausible day around.
        if not (
            math.isfinite(price)
            and math.isfinite(stddev)
            and stddev >= 0
            and math.isfinite(price - stddev)
            and math.isfinite(price + stddev)
        ):
            # Fail CLOSED (the D1 posture): published composites are never
            # revised, so a poisoned vote must kill the day's publish
            # loudly rather than price it — NaN compares False both ways,
            # so sorted() scatters it and the cum walk counts its weight
            # wherever it happened to land: a silently plausible wrong
            # index, worse than the loud NaN the retired mean produced.
            raise ValueError(
                f"non-finite vote for {source_id}: "
                f"price={price!r} stddev={stddev!r}"
            )
        votes.append((price - stddev, weight))
        votes.append((price, weight))
        votes.append((price + stddev, weight))
    median = _weighted_quantile(votes, Fraction(1, 2))
    if iqm_alpha:
        # Fraction(str(...)) reads the alpha as WRITTEN in the config —
        # the same doctrine as the weights, and the same rational the TS
        # mirror derives — then the exact mean is quantized at the
        # artifact's 6dp grain (round() on a Fraction is half-even, like
        # the float round the median branch feeds the writer). The block
        # writer below re-rounds this float; that second round is a
        # proven no-op (a 6dp decimal's nearest double re-rounds to
        # itself at 6dp), so the exact rational is still quantized
        # effectively once.
        value = float(
            round(_interquantile_mean(votes, Fraction(str(iqm_alpha))), 6)
        )
    else:
        value = median
    p25 = _weighted_quantile(votes, Fraction(1, 4))
    p75 = _weighted_quantile(votes, Fraction(3, 4))
    # The retired weighted-mean statistic rides as a diagnostic — DERIVED from
    # the same frozen helper that priced v1-v3, so the diagnostic and the
    # frozen series' math can never drift.
    base = weighted_composite(passing)
    block = {
        "value_usd_gpu_hr": round(value, 6),
        "statistic": MEDIAN_STDDEV_VOTES,
        "confidence_usd_gpu_hr": round(max(value - p25, p75 - value), 6),
        "vote_p25_usd_gpu_hr": round(p25, 6),
        "vote_p75_usd_gpu_hr": round(p75, 6),
        "weighted_mean_usd_gpu_hr": base["value_usd_gpu_hr"],
        "unweighted_mean_usd_gpu_hr": base["unweighted_mean_usd_gpu_hr"],
        "renormalized_weights": base["renormalized_weights"],
        "sources_used_count": base["sources_used_count"],
    }
    if iqm_alpha:
        # Conditional like the knob itself: an alpha-0 lane's artifact
        # bytes must not grow keys. The echoed alpha + point median make
        # a tuned day self-describing (mirror recomputes need the alpha
        # from the DAY, not a live config that may since have re-minted).
        block["iqm_alpha"] = iqm_alpha
        block["vote_median_usd_gpu_hr"] = round(median, 6)
    return block


def weighted_composite(
    passing: Sequence[Tuple[str, float, float]],
) -> Optional[Dict[str, Any]]:
    """The legacy composite: (source_id, weight, usd) → weighted mean with weights
    renormalized to sum 1.0 over the passers; None when nothing passed."""
    if not passing:
        return None
    total_weight = sum(w for _, w, _ in passing)
    value = sum(w * price for _, w, price in passing) / total_weight
    return {
        "value_usd_gpu_hr": round(value, 6),
        "unweighted_mean_usd_gpu_hr": round(
            statistics.mean(price for _, _, price in passing), 6
        ),
        "renormalized_weights": {
            sid: round(w / total_weight, 6) for sid, w, _ in passing
        },
        "sources_used_count": len(passing),
    }


def compute_day(
    *,
    config: Dict[str, Any],
    day: str,
    snapshot: Optional[Dict[str, Any]],
    substituted_from: Optional[int],
    window_history: Dict[str, List[float]],
    fx_records: Dict[str, Dict[str, Any]],
    window_currencies: Optional[Dict[str, str]] = None,
    pending_currencies: Optional[Dict[str, Dict[str, Any]]] = None,
    weight_state: Optional[Dict[str, Any]] = None,
    prior_slot_prints: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """One day's composite payload + the filter-window updates.

    ``window_history`` maps source_id → that source's trailing print series
    (mutated: today's real prints are appended whether accepted or held out
    — the adapt-to-repricing rule; fx-unavailable/missing sources
    contribute nothing). Under calc_v3 window prints are in the source's
    RECORDED currency; ``window_currencies`` (source_id → currency, mutated
    alongside) is how a currency change is detected and is REQUIRED in
    recorded-currency mode (rule D2: a replay without it cannot detect
    currency changes — pass one dict across days). ``pending_currencies``
    carries the D1 consecutive-new-currency confirmation streaks the same
    way; None (single-day callers) degrades to a per-call dict, which is
    fine for one day and wrong across days — the CLI passes both.

    ``weight_state`` ({"prices", "vectors", "mode"}) and
    ``prior_slot_prints`` (the just-closed prior day's per-slot trusted
    prints from resolve_slot_prints, {} at genesis) are REQUIRED when the
    config sets calc.dynamic_weights (the calc_v5 mint). R-slots: the
    prior day's slots are appended to the series BEFORE the vector is
    computed (they sit at or below the R-cutoff boundary — yesterday's
    last slot) and pinned in weight_calc.slot_prints, so weights for day t
    are a pure function of the published series through t-1's last slot
    plus today's eligible set, and the CLI's replay rebuilds the exact
    state from artifacts alone. Legacy configs ignore both — zero extra
    work, zero new keys.
    """
    params = calc_params(config)
    filter_terms = params.get("filter_terms", DEFAULT_FILTER_TERMS)
    recorded_terms = filter_terms == "recorded_currency"
    if recorded_terms and window_currencies is None:
        raise ValueError(
            "filter_terms 'recorded_currency' requires window_currencies "
            "(source_id -> currency) — without it a multi-day replay "
            "cannot detect recorded-currency changes (rule D2)"
        )
    pending = pending_currencies if pending_currencies is not None else {}
    if "filter_sigma_floor_pct" in params and "filter_sigma_floor" in params:
        # Load validation refuses the pair, but params can also arrive
        # from an artifact-embedded/unvalidated dict — with both keys the
        # binding floor is ambiguous, and no published artifact carries
        # both, so this raise is unreachable on every replay.
        raise ValueError(
            "params carry BOTH filter_sigma_floor and "
            "filter_sigma_floor_pct — one floor semantics per mint; "
            "the binding floor would be ambiguous"
        )
    if (
        "vote_sigma_floor_pct" in params
        and "filter_sigma_floor_pct" not in params
    ):
        # Floor split: the vote floor is a percent-regime key. Alongside
        # the ABSOLUTE filter_sigma_floor — whose frozen semantics govern
        # BOTH sigmas — the binding vote floor would be ambiguous; no
        # published artifact carries that pair, so this raise is
        # unreachable on every replay.
        raise ValueError(
            "params carry vote_sigma_floor_pct without "
            "filter_sigma_floor_pct — the vote floor is a percent-regime "
            "key and the absolute filter_sigma_floor keeps its frozen "
            "both-sigmas semantics; the binding vote floor would be "
            "ambiguous"
        )
    sigma_floor = float(
        params.get("filter_sigma_floor", DEFAULT_FILTER_SIGMA_FLOOR)
    )
    # Percent-form floor. None ≠ 0.0 here: None selects the legacy
    # absolute-floor path; 0.0 is a real percent rule. Load validation
    # guarantees the two keys never coexist. FENCE-ONLY since the floor
    # split — the vote floor resolves below.
    sigma_floor_pct = (
        float(params["filter_sigma_floor_pct"])
        if "filter_sigma_floor_pct" in params
        else None
    )
    # calc_v4: which cross-source statistic prices the day. The
    # legacy path must stay byte-identical, so EVERYTHING vote-related below
    # is gated on this flag — a frozen-series replay never computes (or
    # records) a vote.
    median_votes = (
        params.get("composite_statistic", DEFAULT_COMPOSITE_STATISTIC)
        == MEDIAN_STDDEV_VOTES
    )
    # Vote floor split (founder ruling 2026-08-27, mirrored from the panel
    # engine): a percent-regime median-votes params set MUST carry the
    # vote floor — silently falling back to the fence floor would price
    # votes under a rule the params never recorded (no published daily
    # series is percent-regime, so no artifact-embedded params
    # legitimately lack the key). The legacy absolute regime is untouched:
    # filter_sigma_floor feeds both sigmas verbatim.
    if (
        median_votes
        and sigma_floor_pct is not None
        and "vote_sigma_floor_pct" not in params
    ):
        raise ValueError(
            "percent-regime median_ci_votes params missing "
            "vote_sigma_floor_pct — the vote floor is its own knob "
            "(ruling 2026-08-27) and never silently falls back to the "
            "fence floor filter_sigma_floor_pct"
        )
    vote_sigma_floor_pct = (
        float(params["vote_sigma_floor_pct"])
        if median_votes and sigma_floor_pct is not None
        else None
    )
    vote_stddevs: Dict[str, float] = {}
    # One constituent role per basket lane — THE shared predicate (see
    # constituents()); calc_params' fallback_weights and this weight
    # domain must never diverge.
    weights = {s["source_id"]: s.get("weight") for s in constituents(config)}
    exclusions = {
        (e["date"], e["source_id"]): e["reason"]
        for e in params["manual_exclusions"]
    }
    sources_block: List[Dict[str, Any]] = []
    passing: List[Tuple[str, float, float]] = []
    prints_today: Dict[str, float] = {}  # USD — the R3 cross-source mean
    # source_id -> today's filter input: (price, currency) in filter terms,
    # or None for an untrusted print (D1 — nothing may enter the window).
    filter_inputs: Dict[str, Optional[Tuple[float, str]]] = {}

    source_entries = {
        s["source_id"]: s for s in (snapshot or {}).get("sources", [])
    }

    # Resolve pass: every constituent's daily print BEFORE any weight is
    # assigned. Under calc_v5 (dynamic_weights) the day's weight vector
    # depends on which sources printed a TRUSTED value today — the
    # eligible set — so resolution must complete first; for legacy configs
    # the reorder is behavior-identical (resolution is per-source pure and
    # every cross-source computation already runs after the loop).
    resolutions: Dict[str, Dict[str, Any]] = {}
    for source_id in weights:
        entry = source_entries.get(source_id)
        resolved: Dict[str, Any] = {"entry": entry}
        excluded_reason = exclusions.get((day, source_id))
        if excluded_reason is not None:
            resolved["excluded_reason"] = excluded_reason
        else:
            chosen = resolve_daily_print(
                entry,
                source_id=source_id,
                params=params,
                sku=params["target_sku"],
                day=day,
                fx_records=fx_records,
            )
            resolved["chosen"] = chosen
            if chosen is not None and not chosen.get("fx_unavailable"):
                # The one filter_observation call per print — reused by the
                # verdict path below AND by the eligibility test, so the
                # weight domain and the filter can never disagree about
                # which prints are trusted.
                resolved["observation"] = filter_observation(
                    chosen, filter_terms=filter_terms
                )
        resolutions[source_id] = resolved

    # calc_v5 (dynamic_weights): the day's weight vector is DERIVED — the
    # config's opening weights survive only as the warm-up fallback and
    # the per-source risk caps. The vector is computed from the weight
    # state (the series through day t-1) plus today's eligible set, rounded
    # once, and THAT rounded vector is pinned in the artifact, consumed by
    # the composite, and carried in state (the votes-use-rounded-stddev
    # pattern). Legacy configs take the config weights untouched.
    dynamic_params = params.get("dynamic_weights")
    weight_block: Optional[Dict[str, Any]] = None
    if dynamic_params is not None:
        if weight_state is None:
            raise ValueError(
                "calc.dynamic_weights requires weight_state (prices/"
                "vectors/mode) — without it a multi-day replay cannot "
                "reconstruct the weight series (the window_currencies "
                "rule, applied to weights)"
            )
        if prior_slot_prints is None:
            raise ValueError(
                "calc.dynamic_weights requires prior_slot_prints (the "
                "just-closed prior day's per-slot trusted prints, {} at "
                "genesis) — the slot-grid series (R-slots) advances from "
                "them and the artifact pins them for replay"
            )
        # R-slots: append the just-closed prior day's slot prints BEFORE
        # computing weights — every stamp is <= the R-cutoff boundary
        # (yesterday's last slot), so the ordering preserves 'nothing
        # captured today moves today's weights' exactly.
        prior_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        slot_prints_block = {
            "date": prior_day,
            "slots": {
                str(hour): dict(sorted(by_source.items()))
                for hour, by_source in sorted(
                    prior_slot_prints.items(), key=lambda kv: int(kv[0])
                )
            },
        }
        advance_weight_state(
            weight_state,
            day=prior_day,
            weight_block={"slot_prints": slot_prints_block},
        )
        eligible = [
            sid
            for sid in weights
            if resolutions[sid].get("observation") is not None
        ]
        weight_block = compute_dynamic_weights(
            weight_state,
            day=day,
            eligible=eligible,
            dw_params=dynamic_params,
            fallback_weights=dynamic_params["fallback_weights"],
        )
        # The pinned slot prints ride the artifact so the CLI's
        # replay-from-published rebuilds the exact series without ever
        # re-deriving from raw (which can legitimately grow later).
        weight_block["slot_prints"] = slot_prints_block
        weight_of: Dict[str, Optional[float]] = {
            sid: weight_block["weights"].get(sid) for sid in weights
        }
    else:
        weight_of = dict(weights)

    for source_id in weights:
        resolved = resolutions[source_id]
        entry = resolved["entry"]
        weight = weight_of.get(source_id)
        detail: Dict[str, Any] = {"source_id": source_id, "weight": weight}
        excluded_reason = resolved.get("excluded_reason")
        if excluded_reason is not None:
            # Manually excluded (calc_v2): no chosen print, no window entry
            # — the day's record still names the source and says WHY.
            detail["status"] = "manually_excluded"
            detail["excluded_reason"] = excluded_reason
            sources_block.append(detail)
            continue
        chosen = resolved.get("chosen")
        if chosen is None or chosen.get("fx_unavailable"):
            detail["status"] = (
                "fx_unavailable"
                if chosen and chosen.get("fx_unavailable")
                else (entry or {}).get("status", "missing")
            )
            if chosen and chosen.get("fx_errors"):
                detail["fx_errors"] = chosen["fx_errors"]
            sources_block.append(detail)
            continue
        price = chosen["usd_per_gpu_hr"]
        observation = resolved.get("observation")
        # calc_v4: the window today's vote CI derives from — computed ONLY
        # under the median statistic (legacy replays do zero vote work).
        # On the fence path it is captured before the verdict exists and
        # consumed only on acceptance; the untrusted/currency-mismatch
        # paths leave it None (no vote is cast there). Always the
        # PRE-advance history: a print never self-reports its dispersion.
        vote_tail: Optional[List[float]] = None
        if observation is None:
            # Rule D1, fail-closed: a non-USD label with no native price,
            # or an UNKNOWN/malformed label — there is NO trustworthy value
            # in window terms. Held out; the window is preserved and this
            # print does NOT enter it (a documented divergence from the
            # every-real-print rule); the CLI warns and exits 1.
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
                # Rule D1: a trusted print in a different currency than
                # its window. Held out, window preserved — until the
                # CURRENCY_CONFIRM_DAYS-th CONSECUTIVE same-new-currency
                # print confirms a genuine currency change, which discards
                # the old window and seeds the new one from the pending
                # prints (still warm-up). The pending streak reconstructs
                # from published artifacts alone, so replays and restarts
                # are deterministic.
                queued = _pending_after(
                    pending, source_id, filter_price, filter_currency
                )
                prior_currency = window_currencies.get(source_id)
                if len(queued) >= CURRENCY_CONFIRM_DAYS:
                    # calc_v4: the old-currency window is dead here; the
                    # only trustworthy same-currency history is the pending
                    # streak MINUS today's print (pre-advance rule).
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
                # calc_v4: exactly the slice evaluate_filter judges against
                # (warm-up windows are shorter than filter_window, so the
                # slice is the whole history).
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
                    sigma_floor_pct=sigma_floor_pct,
                    currency=filter_currency if recorded_terms else None,
                )
        detail.update({"status": "ok", "chosen": chosen, "filter": verdict})
        prints_today[source_id] = price
        filter_inputs[source_id] = observation
        if verdict["accepted"]:
            passing.append((source_id, float(weight), price))
            if median_votes:
                # calc_v4: an accepted print casts votes. Its CI lives in
                # the filter's operating currency; USD conversion uses the
                # print's own fx rate (present whenever the print was
                # FX-converted), else the rate the recorded USD/native
                # pair implies. No trustworthy rate = no vote = no publish
                # (the D1 fail-closed posture): a non-USD CI must never be
                # priced at parity by silent default.
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
                    # The VOTE floor (ruling 2026-08-27), never the
                    # fence's: None on legacy absolute lanes (sigma_floor
                    # governs both, frozen semantics).
                    sigma_floor_pct=vote_sigma_floor_pct,
                    filter_price=filter_price,
                    fx_factor=fx_factor,
                )
                detail["vote"] = vote
                vote_stddevs[source_id] = vote["conf_usd_gpu_hr"]
        sources_block.append(detail)

    # Warm-up manual-verify flag needs today's cross-source mean (R3).
    if prints_today:
        basket_mean = statistics.mean(prints_today.values())
        for detail in sources_block:
            verdict = detail.get("filter")
            if not verdict or not verdict.get("unfiltered"):
                continue
            price = detail["chosen"]["usd_per_gpu_hr"]
            if (
                basket_mean > 0
                and abs(price - basket_mean) / basket_mean * 100.0
                > params["manual_verify_pct"]
            ):
                verdict["manual_verify"] = True

    # Window rule (METHODOLOGY.md §6.4): every real print enters its source's window — accepted or
    # held out — so genuine repricings adapt within days. calc_v3 (D1):
    # window prints are in the filter's terms (the recorded currency);
    # untrusted prints and unconfirmed cross-currency prints never enter
    # (windows are single-currency by construction), and the confirmation
    # streak lives in `pending`. One state machine, shared with the CLI's
    # replay-from-published advance.
    for source_id, observation in filter_inputs.items():
        advance_window(
            window_history,
            window_currencies,
            pending,
            source_id,
            observation,
        )

    # Claim minimum (B200 lane: 5-of-9). Below the
    # floor the day still publishes — as an explicit basket_dark artifact
    # with index null, the SAME shape as a zero-passers dark day. Configs
    # without the knob keep the original zero-passers rule (floor 1).
    min_claim = int(params.get("min_sources_to_claim", 1))
    if len(passing) >= min_claim:
        composite = (
            median_stddev_composite(
                passing,
                vote_stddevs,
                iqm_alpha=float(params.get("iqm_alpha", 0.0)),
            )
            if median_votes
            else weighted_composite(passing)
        )
    else:
        composite = None

    # Fallback pool: simple mean of the pool's
    # proxy-sku prints (B200 for the B300 lane, B300 for the B200 lane).
    # Always the lowest-eligible-print rule — per-source statistics are a
    # constituent-print methodology, not a pool one.
    pool_block: List[Dict[str, Any]] = []
    pool_prices: List[float] = []
    for source_id in params["fallback_pool_sources"]:
        entry = source_entries.get(source_id)
        excluded_reason = exclusions.get((day, source_id))
        if excluded_reason is not None:
            pool_block.append(
                {
                    "source_id": source_id,
                    "status": "manually_excluded",
                    "excluded_reason": excluded_reason,
                }
            )
            continue
        chosen = (
            daily_source_observation(
                entry,
                sku=params["fallback_pool_sku"],
                day=day,
                fx_records=fx_records,
                interruptible_tiers=params["interruptible_tiers"],
                fx_max_staleness_days=params["fx_max_staleness_days"],
            )
            if entry
            else None
        )
        if chosen and not chosen.get("fx_unavailable"):
            pool_block.append({"source_id": source_id, "chosen": chosen})
            pool_prices.append(chosen["usd_per_gpu_hr"])
        else:
            pool_block.append(
                {
                    "source_id": source_id,
                    "status": (entry or {}).get("status", "missing"),
                }
            )

    payload = {
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "kind": COMPOSITE_KIND,
        "basket_id": config["basket_id"],
        "methodology_id": params["methodology_id"],
        # The resolved params travel with every artifact: a param edit that
        # forgot to mint a new methodology_id is visible in the series.
        "calc_params": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
        },
        "date": day,
        "basket_dark": composite is None,
        "index": composite,
        "sources": sources_block,
        "fallback_pool": {
            "sources": pool_block,
            "mean_usd_gpu_hr": round(statistics.mean(pool_prices), 6)
            if pool_prices
            else None,
        },
        "snapshot_run_id": (snapshot or {}).get("run_id"),
        "snapshot_late_fill": (snapshot or {}).get("late_fill"),
        "substituted_from_slot": substituted_from,
        "day_missed": snapshot is None,
    }
    if dynamic_params is not None:
        # calc_v5: the day's full weight audit trail — mode, the pinned
        # rounded vector, the pinned prior-day slot prints, per-source
        # q/Q/n_samples, cap flags — rides the artifact so the softmax ->
        # floor -> cap chain AND the slot series are recomputable from the
        # artifact alone, and the CLI's replay advance rebuilds the weight
        # state without re-deriving anything from raw.
        payload["weight_calc"] = weight_block
        # Advance the vector/mode latch with the SAME full block the
        # replay-from-published path consumes (the slot prints were already
        # appended pre-computation; re-advancing them is an idempotent
        # overwrite of identical values, keeping the two paths one code
        # path).
        advance_weight_state(weight_state, day=day, weight_block=weight_block)
    return payload
