# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Load + validate a basket lane config (config/index_basket*.json).

Every methodology parameter (slots, canonical hour, timeouts, prefix) is
config, not code — a parameter change is a config edit under a new
methodology_id, never a code change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "index_basket.json"

# One constituent role per basket lane (<chip>_basket) plus fallback-pool
# roles (<chip>_pool). A config names its own constituent role via
# `basket_role`; the default keeps the original B300 lane untouched.
VALID_ROLES = {"b300_basket", "b200_pool", "b200_basket", "b300_pool"}
DEFAULT_BASKET_ROLE = "b300_basket"


class BasketConfigError(RuntimeError):
    """config/index_basket.json is missing or malformed."""


def load_basket_config(path: Optional[Path] = None) -> Dict[str, Any]:
    # Precedence: explicit argument > env override > repo default. An exported
    # BASKET_CONFIG_PATH must never silently eat a --config flag.
    cfg_path = Path(
        path or os.environ.get("BASKET_CONFIG_PATH") or DEFAULT_CONFIG_PATH
    )
    if not cfg_path.exists():
        raise BasketConfigError(f"basket config missing: {cfg_path}")
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise BasketConfigError(f"basket config unparseable: {cfg_path}: {exc}") from exc
    _validate(cfg)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    for field in ("basket_id", "bucket_prefix", "capture_slots_utc", "sources"):
        if not cfg.get(field):
            raise BasketConfigError(f"basket config missing {field!r}")
    prefix = str(cfg["bucket_prefix"])
    if (
        not prefix.startswith("index/")
        or "\\" in prefix
        or any(seg in ("", ".", "..") for seg in prefix.split("/"))
    ):
        # The lanes may share one bucket; this makes "never writes
        # outside the index/ keyspace" mechanical rather than reviewed.
        # Dot segments are rejected because an HTTP path-style gateway may
        # normalize 'index/../<elsewhere>' straight out of the keyspace.
        raise BasketConfigError(
            f"bucket_prefix must live under 'index/' with clean path "
            f"segments: {prefix!r}"
        )
    slots = cfg["capture_slots_utc"]
    if not isinstance(slots, list) or not all(
        isinstance(h, int) and 0 <= h <= 23 for h in slots
    ):
        raise BasketConfigError(f"capture_slots_utc must be UTC hours 0-23: {slots!r}")
    if not 2 <= len(set(slots)) <= 4:
        # The capture cadence is 2-4 slots per day. A different cadence is a
        # deliberate methodology change, not a config typo — force the edit
        # here.
        raise BasketConfigError(
            f"capture_slots_utc must hold 2-4 distinct slots: {slots!r}"
        )
    canonical = cfg.get("canonical_slot_utc")
    if canonical is not None and canonical not in slots:
        raise BasketConfigError(
            f"canonical_slot_utc {canonical!r} is not one of capture_slots_utc {slots!r}"
        )
    basket_role = cfg.get("basket_role", DEFAULT_BASKET_ROLE)
    if basket_role not in VALID_ROLES:
        raise BasketConfigError(
            f"basket_role must be one of {sorted(VALID_ROLES)}, got {basket_role!r}"
        )
    seen = set()
    for src in cfg["sources"]:
        sid = src.get("source_id")
        if not sid:
            raise BasketConfigError(f"source without source_id: {src!r}")
        if sid in seen:
            raise BasketConfigError(f"duplicate source_id {sid!r}")
        seen.add(sid)
        if src.get("role") not in VALID_ROLES:
            raise BasketConfigError(
                f"source {sid!r} role {src.get('role')!r} not in {sorted(VALID_ROLES)}"
            )
        weight = src.get("weight")
        if src["role"] == basket_role and not (
            isinstance(weight, (int, float)) and weight > 0
        ):
            raise BasketConfigError(f"basket source {sid!r} needs a positive weight")
    basket_weights = [
        float(s["weight"]) for s in cfg["sources"] if s["role"] == basket_role
    ]
    if abs(sum(basket_weights) - 1.0) > 1e-9:
        raise BasketConfigError(
            f"{basket_role} weights must sum to 1.0, got {sum(basket_weights)}"
        )
    min_claim = cfg.get("min_basket_sources_to_claim", 1)
    if not (isinstance(min_claim, int) and 1 <= min_claim <= len(basket_weights)):
        raise BasketConfigError(
            f"min_basket_sources_to_claim must be an int in 1..{len(basket_weights)}, "
            f"got {min_claim!r}"
        )
    calc = cfg.get("calc") or {}
    for field, lo in (("filter_window", 2), ("filter_warmup", 1)):
        value = calc.get(field)
        if value is not None and not (isinstance(value, int) and value >= lo):
            raise BasketConfigError(f"calc.{field} must be an int >= {lo}, got {value!r}")
    sigma = calc.get("filter_sigma")
    if sigma is not None and not (
        isinstance(sigma, (int, float)) and sigma > 0
    ):
        raise BasketConfigError(f"calc.filter_sigma must be > 0, got {sigma!r}")
    # calc_v3 filter knobs. Both are calc params embedded in
    # every artifact — a validator typo must never quietly default one
    # into an existing series, so absent stays absent.
    sigma_floor = calc.get("filter_sigma_floor")
    if sigma_floor is not None and not (
        isinstance(sigma_floor, (int, float))
        and not isinstance(sigma_floor, bool)
        and sigma_floor >= 0
    ):
        raise BasketConfigError(
            f"calc.filter_sigma_floor must be a number >= 0, got {sigma_floor!r}"
        )
    filter_terms = calc.get("filter_terms")
    if filter_terms is not None:
        # Lazy import for the same reason as SOURCE_STATISTIC_FNS below:
        # composite is the single home of filter-terms semantics.
        from gpu_index.index.composite import VALID_FILTER_TERMS

        if filter_terms not in VALID_FILTER_TERMS:
            raise BasketConfigError(
                f"calc.filter_terms must be one of {sorted(VALID_FILTER_TERMS)}, "
                f"got {filter_terms!r}"
            )
    if "composite_statistic" in calc:
        # calc_v4: composite is the single home of the
        # cross-source statistic semantics, same lazy-import rule.
        # Validated on key PRESENCE, exactly the predicate calc_params
        # embeds on — a `null` here would otherwise slip through and pin
        # the string "None" into the series' artifact bytes forever.
        from gpu_index.index.composite import (
            MEDIAN_STDDEV_VOTES,
            VALID_COMPOSITE_STATISTICS,
        )

        composite_statistic = calc["composite_statistic"]
        if composite_statistic not in VALID_COMPOSITE_STATISTICS:
            raise BasketConfigError(
                "calc.composite_statistic must be one of "
                f"{sorted(VALID_COMPOSITE_STATISTICS)}, "
                f"got {composite_statistic!r}"
            )
        if composite_statistic == MEDIAN_STDDEV_VOTES:
            # The floor is what stops a frozen list price (sigma 0) from
            # voting with conviction it never earned — the ruling the
            # median-of-votes mint is built on. A lane config that copies
            # the statistic without the floor must fail at load, not
            # quietly pin floor-less params as that series' law.
            if not (
                isinstance(sigma_floor, (int, float))
                and not isinstance(sigma_floor, bool)
                and sigma_floor > 0
            ):
                raise BasketConfigError(
                    "calc.composite_statistic 'median_ci_votes' requires "
                    f"calc.filter_sigma_floor > 0, got {sigma_floor!r}"
                )
    if "dynamic_weights" in calc:
        # calc_v5 mint: dynamic predictive weighting. Every knob below is a
        # calc param embedded in every artifact — same absence discipline
        # as the calc_v3/v4 knobs — and a malformed value must fail at
        # LOAD, because the calc lane would otherwise pin it into a series'
        # bytes. The weights module is the single home of the scheme
        # semantics (lazy import, the SOURCE_STATISTIC_FNS rule).
        from gpu_index.index.weights import VALID_WEIGHT_SCHEMES

        dw = calc["dynamic_weights"]
        if not isinstance(dw, dict):
            raise BasketConfigError("calc.dynamic_weights must be an object")
        if dw.get("scheme") not in VALID_WEIGHT_SCHEMES:
            raise BasketConfigError(
                f"calc.dynamic_weights.scheme must be one of "
                f"{sorted(VALID_WEIGHT_SCHEMES)}, got {dw.get('scheme')!r}"
            )
        # R-slots: horizons live on the capture slot grid, in HOURS. The
        # grid must be UNIFORM (every consecutive gap equal, including the
        # wrap past midnight) or hour arithmetic has no single spacing, and
        # every horizon must be a multiple of that spacing >= one slot —
        # a horizon the grid cannot express would produce zero samples
        # forever, the silently-inert-scheme failure this validator exists
        # to refuse. (1h horizons become expressible when capture itself
        # walks to hourly snapshots — a cadence renegotiation.)
        sorted_slots = sorted(slots)
        gaps = [
            b - a for a, b in zip(sorted_slots, sorted_slots[1:])
        ] + [24 - sorted_slots[-1] + sorted_slots[0]]
        if len(set(gaps)) != 1:
            raise BasketConfigError(
                f"calc.dynamic_weights requires UNIFORM capture slot "
                f"spacing, got slots {sorted_slots!r} (gaps {gaps!r})"
            )
        slot_spacing = gaps[0]
        for field in ("lookback_horizons_hours", "forward_horizons_hours"):
            horizons = dw.get(field)
            if (
                not isinstance(horizons, list)
                or not horizons
                or not all(
                    isinstance(h, int) and not isinstance(h, bool) and h >= 1
                    for h in horizons
                )
                or sorted(set(horizons)) != horizons
            ):
                # Strictly-increasing pins the feature/score ORDER in the
                # embedded params — two orderings of the same set would be
                # two different byte streams for one methodology.
                raise BasketConfigError(
                    f"calc.dynamic_weights.{field} must be a strictly "
                    f"increasing list of ints >= 1, got {horizons!r}"
                )
            for horizon in horizons:
                if horizon % slot_spacing != 0 or horizon < slot_spacing:
                    raise BasketConfigError(
                        f"calc.dynamic_weights.{field} entry {horizon} is "
                        f"not expressible on the {slot_spacing}h capture "
                        f"slot grid (must be a multiple >= {slot_spacing})"
                    )
        for field, lo in (
            ("history_days", 1),
            ("min_train_samples", 1),
        ):
            value = dw.get(field)
            if not (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= lo
            ):
                raise BasketConfigError(
                    f"calc.dynamic_weights.{field} must be an int >= {lo}, "
                    f"got {value!r}"
                )
        # Finiteness is part of well-formed here:
        # json.loads accepts the Infinity literal, Infinity
        # passes bare comparisons, and gamma=Infinity would then survive
        # every artifact pin and NaN the first dynamic-mode publish weeks
        # after the merge — the validator's whole contract is fail at
        # LOAD. (NaN already fails: every comparison is False.)
        import math as _math

        for field, lo_exclusive in (
            ("half_life_days", True),
            ("ridge_lambda", False),
            ("gamma", False),
        ):
            value = dw.get(field)
            ok = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and _math.isfinite(value)
                and (value > 0 if lo_exclusive else value >= 0)
            )
            if not ok:
                raise BasketConfigError(
                    f"calc.dynamic_weights.{field} must be a finite number "
                    f"{'>' if lo_exclusive else '>='} 0, got {value!r}"
                )
        floor = dw.get("target_variance_floor")
        if floor is not None and not (
            isinstance(floor, (int, float))
            and not isinstance(floor, bool)
            and _math.isfinite(floor)
            and floor > 0
        ):
            # > 0, not >= 0: floor 0 admits a genuinely zero-variance
            # target into the R^2 denominator (the runtime guards it too,
            # but a floor that can never fire is a malformed config), and
            # Infinity makes every q permanently undefined — the silently
            # inert scheme the min_span check below exists to refuse.
            raise BasketConfigError(
                f"calc.dynamic_weights.target_variance_floor must be a "
                f"finite number > 0, got {floor!r}"
            )
        clamp = dw.get("max_abs_log_return")
        if clamp is not None and not (
            isinstance(clamp, (int, float))
            and not isinstance(clamp, bool)
            and _math.isfinite(clamp)
            and clamp > 0
        ):
            raise BasketConfigError(
                f"calc.dynamic_weights.max_abs_log_return must be a finite "
                f"number > 0, got {clamp!r}"
            )
        quorum = dw.get("switch_min_eligible")
        if quorum is not None and not (
            isinstance(quorum, int)
            and not isinstance(quorum, bool)
            and 1 <= quorum <= len(basket_weights)
        ):
            raise BasketConfigError(
                f"calc.dynamic_weights.switch_min_eligible must be an int "
                f"in 1..{len(basket_weights)}, got {quorum!r}"
            )
        w_min = dw.get("weight_min")
        w_max = dw.get("weight_max")
        for field, value in (("weight_min", w_min), ("weight_max", w_max)):
            if not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0 < value <= 1
            ):
                raise BasketConfigError(
                    f"calc.dynamic_weights.{field} must be a number in "
                    f"(0, 1], got {value!r}"
                )
        if w_min > w_max:
            raise BasketConfigError(
                f"calc.dynamic_weights.weight_min ({w_min}) must not exceed "
                f"weight_max ({w_max})"
            )
        if len(basket_weights) * w_min >= 1.0:
            # The floor must never consume the whole simplex over the FULL
            # constituent set (the eligible set only shrinks) — at >= 1 the
            # softmax spread is dead and every day is a degenerate publish.
            raise BasketConfigError(
                f"calc.dynamic_weights.weight_min {w_min} is infeasible for "
                f"{len(basket_weights)} constituents (N*w_min must be < 1)"
            )
        for weight in basket_weights:
            # Fallback mode's byte-parity promise (index math identical to
            # the frozen fixed-weight series) holds because the pinned
            # rounded-6dp vector EQUALS the raw config weight. A >6dp
            # weight would silently fork fallback values from the frozen
            # series at the rounding step — refuse it at load.
            if round(float(weight), 6) != float(weight):
                raise BasketConfigError(
                    f"dynamic_weights requires constituent weights exact "
                    f"at 6 decimal places (the pinned-vector precision), "
                    f"got {weight!r}"
                )
        # A defined score needs max(lookback) + max(forward) hours of
        # series plus min_train sample anchors (one per slot) inside the
        # history window; a window too short to EVER define one is a
        # silently inert scheme (the same failure class as a non-canonical
        # exclusion date): permanent fallback that reads as warm-up forever.
        min_span_hours = (
            max(dw["lookback_horizons_hours"])
            + max(dw["forward_horizons_hours"])
            + dw["min_train_samples"] * slot_spacing
            + slot_spacing
        )
        if dw["history_days"] * 24 < min_span_hours:
            raise BasketConfigError(
                f"calc.dynamic_weights.history_days {dw['history_days']} can "
                f"never define a score (needs >= {min_span_hours} hours for "
                f"the configured horizons and sample gates)"
            )
        caps = dw.get("source_weight_caps")
        if caps is not None:
            if not isinstance(caps, dict):
                raise BasketConfigError(
                    "calc.dynamic_weights.source_weight_caps must be an object"
                )
            constituent_ids = {
                s["source_id"]
                for s in cfg["sources"]
                if s["role"] == basket_role
            }
            for sid, cap in caps.items():
                if sid not in constituent_ids:
                    raise BasketConfigError(
                        f"source_weight_caps source_id {sid!r} is not a "
                        f"{basket_role} constituent"
                    )
                if not (
                    isinstance(cap, (int, float))
                    and not isinstance(cap, bool)
                    and w_min <= cap <= 1
                ):
                    # A cap below the floor is unsatisfiable arithmetic —
                    # the allocator would flag every day infeasible.
                    raise BasketConfigError(
                        f"source_weight_caps[{sid!r}] must be a number in "
                        f"[weight_min, 1], got {cap!r}"
                    )
    screens = cfg.get("capture_screens")
    if screens is not None and not isinstance(screens, dict):
        raise BasketConfigError(
            f"capture_screens must be an object, got {type(screens).__name__}"
        )
    screens = screens or {}
    for field in ("jump_quarantine_pct", "jump_corroborate_pct"):
        value = screens.get(field)
        if value is not None and not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            raise BasketConfigError(
                f"capture_screens.{field} must be a positive number, got {value!r}"
            )
    quarantine_pct = screens.get("jump_quarantine_pct")
    corroborate_pct = screens.get("jump_corroborate_pct")
    if (
        isinstance(quarantine_pct, (int, float))
        and isinstance(corroborate_pct, (int, float))
        and corroborate_pct > quarantine_pct
    ):
        # Movers between the two thresholds would fire quarantine but never
        # corroborate each other — a genuine market-wide move in that band
        # would quarantine wholesale.
        raise BasketConfigError(
            "capture_screens.jump_corroborate_pct must not exceed "
            f"jump_quarantine_pct ({corroborate_pct!r} > {quarantine_pct!r})"
        )
    min_corr = screens.get("jump_min_corroborators")
    if min_corr is not None and not (
        isinstance(min_corr, int) and not isinstance(min_corr, bool) and min_corr >= 1
    ):
        raise BasketConfigError(
            f"capture_screens.jump_min_corroborators must be an int >= 1, "
            f"got {min_corr!r}"
        )
    exclusions = calc.get("manual_exclusions")
    if exclusions is not None:
        if not isinstance(exclusions, list):
            raise BasketConfigError("calc.manual_exclusions must be a list")
        known_ids = {s.get("source_id") for s in cfg.get("sources", [])}
        from datetime import date as _date

        seen_pairs = set()
        for entry in exclusions:
            if not isinstance(entry, dict):
                raise BasketConfigError(
                    f"calc.manual_exclusions entries must be objects, got {entry!r}"
                )
            try:
                parsed = _date.fromisoformat(str(entry.get("date")))
            except (TypeError, ValueError) as exc:
                raise BasketConfigError(
                    f"manual_exclusions date must be YYYY-MM-DD, got {entry.get('date')!r}"
                ) from exc
            if parsed.isoformat() != str(entry.get("date")):
                # fromisoformat also accepts compact ('20260815') and
                # ISO-week ('2026-W33-2') forms, but compute_day matches by
                # EXACT string against date.isoformat() — a non-canonical
                # form would validate and then never fire (a silently inert
                # incident exclusion).
                raise BasketConfigError(
                    f"manual_exclusions date must be canonical YYYY-MM-DD, "
                    f"got {entry.get('date')!r}"
                )
            if entry.get("source_id") not in known_ids:
                raise BasketConfigError(
                    f"manual_exclusions source_id {entry.get('source_id')!r} "
                    "is not a configured source"
                )
            if not str(entry.get("reason") or "").strip():
                # An exclusion without a recorded reason is unauditable.
                raise BasketConfigError(
                    f"manual_exclusions entry for {entry.get('date')!r}/"
                    f"{entry.get('source_id')!r} needs a non-empty reason"
                )
            pair = (str(entry["date"]), str(entry["source_id"]))
            if pair in seen_pairs:
                raise BasketConfigError(
                    f"duplicate manual_exclusions entry for {pair!r}"
                )
            seen_pairs.add(pair)
    # Lane knobs (all optional; absent on the B300 lane — their
    # PRESENCE changes calc_params bytes, so a validator typo here must
    # never quietly default one into an existing series).
    min_claim_calc = calc.get("min_sources_to_claim")
    if min_claim_calc is not None and not (
        isinstance(min_claim_calc, int)
        and not isinstance(min_claim_calc, bool)
        and 1 <= min_claim_calc <= len(basket_weights)
    ):
        raise BasketConfigError(
            f"calc.min_sources_to_claim must be an int in 1..{len(basket_weights)}, "
            f"got {min_claim_calc!r}"
        )
    if min_claim_calc is not None and min_claim_calc != min_claim:
        # One floor, two knobs: the top-level min_basket_sources_to_claim
        # drives the CAPTURE coverage line, calc.min_sources_to_claim
        # drives the composite's dark-day floor. They describe the same
        # methodology threshold — silently diverging values would make the
        # capture report a coverage the calc doesn't honor.
        raise BasketConfigError(
            f"calc.min_sources_to_claim ({min_claim_calc}) must equal "
            f"min_basket_sources_to_claim ({min_claim}) — one claim floor"
        )
    fx_lane = calc.get("fx_lane")
    if fx_lane is not None and fx_lane not in ("ecb", "none"):
        raise BasketConfigError(
            f"calc.fx_lane must be 'ecb' or 'none', got {fx_lane!r}"
        )
    stats = calc.get("source_statistics")
    if stats is not None:
        if not isinstance(stats, dict):
            raise BasketConfigError("calc.source_statistics must be an object")
        # Lazy import: composite is the single home of the statistic
        # registry (an id names a computation whose meaning is pinned by
        # methodology_id), and config must not import it at module load.
        from gpu_index.index.composite import SOURCE_STATISTIC_FNS

        for sid, stat in stats.items():
            if sid not in seen:
                raise BasketConfigError(
                    f"source_statistics source_id {sid!r} is not a configured source"
                )
            if stat not in SOURCE_STATISTIC_FNS:
                raise BasketConfigError(
                    f"source_statistics[{sid!r}] names unknown statistic {stat!r} "
                    f"(known: {sorted(SOURCE_STATISTIC_FNS)})"
                )
    parity_id = cfg.get("fallback_parity_methodology_id")
    if parity_id is not None and not (
        isinstance(parity_id, str) and parity_id.strip()
    ):
        # The fallback-parity tripwire's frozen-series pointer (top-level
        # on purpose: it is OPERATIONAL — a comparison target — never a
        # calc param, so it must not ride calc_params into artifact bytes).
        raise BasketConfigError(
            f"fallback_parity_methodology_id must be a non-empty string, "
            f"got {parity_id!r}"
        )
    genesis = cfg.get("genesis_date")
    if genesis is not None:
        try:
            from datetime import date as _date

            parsed_genesis = _date.fromisoformat(genesis)
        except (TypeError, ValueError) as exc:
            raise BasketConfigError(
                f"genesis_date must be YYYY-MM-DD, got {genesis!r}"
            ) from exc
        if parsed_genesis.isoformat() != str(genesis):
            raise BasketConfigError(
                f"genesis_date must be canonical YYYY-MM-DD, got {genesis!r}"
            )


def sources_by_id(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {s["source_id"]: s for s in cfg["sources"]}
