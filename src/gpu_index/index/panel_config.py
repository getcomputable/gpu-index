# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Load + validate the hourly PANEL lane configs (METHODOLOGY.md).

A panel config is a NEW shape -- it is NOT a basket capture config
(gpu_index.index.config) and must never load through load_basket_config: panels
capture nothing (they are selections over a stored record), so the whole
capture vocabulary (capture_slots_utc / canonical_slot_utc / roles /
capture_screens) is absent, and the panel-only vocabulary (record_sources,
slot_grids, members with sku sets and variant rules, calc.eligible_tiers,
calc.jump_screen,
calc.dynamic_weights.attendance_floor, top-level drift_scan_observations)
is REQUIRED where the engine cannot run without it. Same doctrine as
gpu_index.index.config otherwise: every parameter is config so a counterparty
negotiation never becomes a code change, a malformed value must refuse at
LOAD (the calc lane would otherwise pin it into a published series'
bytes), and semantics live in ONE home (gpu_index.index.panel holds the statistic
registry and screen machinery; gpu_index.index.weights holds the weight scheme and
attendance semantics; gpu_index.index.panel_schedule holds the era-grid math) --
this validator lazy-imports each home exactly as gpu_index.index.config does.

Shape (see gpu_index.index.panel.panel_calc_params for what rides the artifact):

  panel_id            lane identity string (artifact field).
  bucket_prefix       the lane's keyspace. index/-contained with clean
                      segments (the basket rule), and fenced against the
                      known lane keyspaces: never equal to or nested with
                      index/raw_observatory (the READ-ONLY record a panel
                      consumes -- a lane writing there could shadow the
                      record), and never STRICTLY nested with any other
                      known lane prefix. Exact EQUALITY with a non-record
                      lane prefix is allowed on purpose: the migrated
                      B300/B200 hourly lanes live under their basket
                      lane's prefix (design section 1 table; section 9 FX
                      reuse), which is collision-safe because composites
                      key per methodology_id.
  genesis_date        canonical YYYY-MM-DD; the lane's first scheduled
                      observation day.
  drift_scan_observations
                      REQUIRED int >= 1 -- how many trailing published
                      observations the warn-only record drift scan
                      re-resolves. TOP-LEVEL on purpose (amended into the
                      mints before any observation published): it is an
                      operational knob like the daily lanes'
                      fallback_parity_methodology_id, NOT methodology, so
                      it must never ride calc_params/artifact bytes -- the
                      D2 fence would otherwise refuse to extend a series
                      over a pure ops retune. The retired calc-level
                      location is REFUSED loudly (the silently-inert-key
                      rule).
  record_exclusions   OPTIONAL [{date, hour, reason}] (amended into the
                      mints before any observation published; adversarial
                      review F6): quarantines ONE scheduled (date, hour)'s
                      record object -- the escape hatch for a poisoned/
                      unparseable stored snapshot that would otherwise
                      crash every firing forever. The observation
                      publishes an explicit record_quarantined artifact
                      (index null, observation_missed false) WITHOUT ever
                      reading the record. Series-shaping, so it rides
                      calc_params and pins per published observation like
                      manual_exclusions. Hours must sit on the era grid;
                      duplicates and empty reasons refuse.
  record_sources      ORDERED [{kind: basket|observatory, prefix,
                      from_date, to_date?}]. Coverage must be contiguous
                      and non-overlapping from genesis: the first
                      from_date IS genesis, each next from_date is the
                      prior to_date + 1 day, every non-final entry is
                      closed (to_date), and the FINAL entry is open-ended
                      (no to_date) -- a closed final record would make
                      some future scheduled observation unreadable, a
                      fail-at-load condition, never a runtime surprise.
  slot_grids          the era-scoped scheduled grid ([{from_date,
                      slot_hours_utc}]); validated by constructing
                      gpu_index.index.panel_schedule.PanelSchedule (one home for
                      the grid rules: uniform spacing incl. the midnight
                      wrap, first era at genesis, strictly increasing
                      eras).
  reject_tokens       optional panel-level identity screen: boundary-aware
                      tokens (gpu_index.observatory.catalog normalization) rejected
                      on sku_identifier. Broad panels omit it.
  members             [{source_id, weight, skus, statistic?, variant?,
                      extra_require?, notes?}]. Weights positive, exact at 6dp (the
                      pinned-vector precision), sum 1.0. skus
                      is the member's non-empty stored-sku selection set.
                      statistic must name a PANEL statistic
                      (gpu_index.index.panel.PANEL_STATISTIC_FNS). variant is
                      {mode: "label", require_tokens: [...]} or {mode:
                      "declared", evidence: "..."} -- declared mode
                      REQUIRES the provider-surface evidence string
                      (design section 8: quote the probe evidence), and
                      must not also carry require_tokens (two variant
                      rules on one seat is a config contradiction).
                      extra_require is an optional {key: value} object of
                      non-empty strings matched EXACTLY against a row's
                      structured extra dict (e.g. the runpod Secure-only
                      screen); a row without the key
                      PASSES -- the one deliberate fail-open, documented
                      at gpu_index.index.panel.member_eligible_rows.
  calc                methodology_id (a single clean key segment
                      [A-Za-z0-9._-]+ -- it becomes the composites
                      keyspace segment -- and, under a shared daily-lane
                      prefix, never one of FROZEN_METHODOLOGY_IDS: a
                      typo'd id would move a frozen series' latest.json
                      pointer), min_sources_to_claim (REQUIRED --
                      the claim floor is lane law, design section 1
                      table), eligible_tiers (REQUIRED non-empty tier
                      ALLOW-LIST containing "on-demand" -- methodology
                      section 5 "on-demand only", the hourly-mint tier
                      reconciliation; the retired exclusion-list key
                      interruptible_tiers is REFUSED outright -- the
                      frozen DAILY basket lanes keep it, panels never
                      had published bytes under it), filter knobs (window/
                      sigma/warmup/sigma_floor/terms/composite_statistic;
                      median_ci_votes requires sigma_floor > 0, the
                      basket rule), manual_verify_pct?, fx_lane (REQUIRED
                      ecb|none), fx_max_staleness_days?,
                      manual_exclusions ([{date, source_id, reason,
                      hour?}] -- hour scopes ONE observation, absent hour
                      holds out the whole date; design section 3 item 9),
                      jump_screen (REQUIRED {quarantine_pct,
                      corroborate_pct <= quarantine_pct,
                      min_corroborators, reference_max_lookback} -- the
                      L5-at-calc thresholds are methodology, minted into
                      calc_params), statistic_params? ({statistic_id:
                      {param: int}} overriding gpu_index.index.panel's chosen-prior
                      defaults; an entry for a statistic no member names
                      is inert config and refuses), dynamic_weights
                      (REQUIRED -- rule A1: panel weights recompute at
                      every observation, a panel without the block has no
                      weight law; includes the A2 attendance_floor,
                      validated by gpu_index.index.weights.validate_attendance_floor).
                      calc.drift_scan_observations is a RETIRED location,
                      refused loudly (see the top-level key above).

Unknown keys REFUSE at every documented level (top level, calc, member,
dynamic_weights, jump_screen, exclusion entries; adversarial review F12,
amended pre-publish): a key the engine does not read is silently inert
config -- a typo'd 'rejected_tokens' would ship a panel with NO identity
screen while its author reads a live one.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpu_index.index.panel_schedule import (
    PanelSchedule,
    PanelScheduleError,
    date_hour_to_stamp,
)
from gpu_index.observatory.catalog import normalize_label

# The lane keyspaces this repo already writes or reads under index/.
# index/raw_observatory is the READ-ONLY record (its own fence lives at
# gpu_index.observatory.config.RESERVED_LANE_PREFIXES for the capture side; the
# basket pair here is pinned in lockstep with that constant by test, the
# DEFAULT_BASKET_ROLE convention). The four H-series prefixes are the
# design section 1 lane table.
RAW_OBSERVATORY_PREFIX = "index/raw_observatory"
KNOWN_LANE_PREFIXES = (
    RAW_OBSERVATORY_PREFIX,
    "index/b300_basket",
    "index/b200_basket",
    "index/h100_sxm",
    "index/h200_sxm",
    "index/h100_broad",
    "index/h200_broad",
)

VALID_RECORD_KINDS = ("basket", "observatory")
VALID_VARIANT_MODES = ("label", "declared")

# The FROZEN daily series' methodology ids (security review, amended
# pre-publish). A panel lane sharing a daily lane's prefix (the migrated
# B300/B200 hourly lanes) writes composites under
# <prefix>/composites/<methodology_id>/ -- the SAME keyspace scheme the
# frozen daily series live in. Composite keys collide only when the
# methodology_id ALSO collides, so a typo'd panel id naming a frozen
# daily id (say annex_a_v0_2_calc_v6 instead of _v7) would publish
# hour-keyed artifacts into the frozen series' keyspace and MOVE ITS
# latest.json pointer ('2026-08-23T04' > '2026-08-23' lexicographically)
# -- silently repointing a frozen contractual series. Pinned here as a
# module constant and refused at load; grow it when a daily series mints.
FROZEN_METHODOLOGY_IDS = frozenset(
    [f"annex_a_v0_2_calc_v{i}" for i in range(1, 7)]
    + [f"annex_a2_v0_3_calc_v{i}" for i in range(1, 6)]
)

# The prefixes those frozen series live under (subset of
# KNOWN_LANE_PREFIXES): the frozen-id fence applies exactly there.
DAILY_LANE_PREFIXES = ("index/b300_basket", "index/b200_basket")

# Tokens must normalize to the catalog's token alphabet -- same rule as
# gpu_index.observatory.catalog (punctuation belongs in label prep, not the token).
_TOKEN_CHARS_RE = re.compile(r"^[A-Z0-9 ]+$")

# methodology_id becomes a bucket key SEGMENT
# (<prefix>/composites/<methodology_id>/...): one clean segment only, or
# a hostile/typo'd id ('a/b', '..', a trailing newline) could address
# keys outside its own composites keyspace.
_METHODOLOGY_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


class PanelConfigError(RuntimeError):
    """A panel config file is missing or malformed."""


def load_panel_config(path: Path) -> Dict[str, Any]:
    """Load + validate one panel config. No env-var fallback on purpose:
    every panel lane is invoked with an explicit --config path (six lanes
    share one entry point; a silently inherited env default could run the
    wrong lane's law)."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise PanelConfigError(f"panel config missing: {cfg_path}")
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise PanelConfigError(
            f"panel config unparseable: {cfg_path}: {exc}"
        ) from exc
    validate_panel_config(cfg)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def panel_schedule(config: Dict[str, Any]) -> PanelSchedule:
    """The lane's era grid -- built from the SAME two keys whether given
    the config or its embedded calc_params (both carry genesis_date +
    slot_grids, so replay-from-artifact rebuilds the identical grid)."""
    try:
        return PanelSchedule(
            genesis_date=config["genesis_date"],
            slot_grids=config["slot_grids"],
        )
    except PanelScheduleError as exc:
        raise PanelConfigError(f"slot_grids invalid: {exc}") from exc


# ------------------------------------------------------------- validation

# Documented schema allowlists (adversarial review F12, unknown-key
# rejection): a key the engine does not read would otherwise validate and
# silently do nothing -- the exact silently-inert-config class this loader
# refuses everywhere else (a typo'd 'rejected_tokens' would ship a panel
# with NO identity screen). Each allowlist is the docstring's schema plus
# audit-prose keys (description/notes/doc pointers) plus, for calc, the
# two RETIRED keys whose dedicated refusals must keep their own messages.
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "panel_id",
        "description",
        "methodology_doc",
        "design_doc",
        "bucket_prefix",
        "genesis_date",
        "drift_scan_observations",
        "record_sources",
        "slot_grids",
        "reject_tokens",
        "record_exclusions",
        "members",
        "calc",
        "_config_path",  # the loader's own bookkeeping key
    }
)
_CALC_KEYS = frozenset(
    {
        "methodology_id",
        "description",
        "min_sources_to_claim",
        "eligible_tiers",
        "filter_window",
        "filter_sigma",
        "filter_warmup",
        "filter_sigma_floor",
        "filter_terms",
        "composite_statistic",
        "manual_verify_pct",
        "fx_lane",
        "fx_max_staleness_days",
        "manual_exclusions",
        "jump_screen",
        "statistic_params",
        "dynamic_weights",
        # Retired locations -- recognized so their DEDICATED refusals
        # (naming the successor) fire instead of the generic unknown-key
        # message.
        "interruptible_tiers",
        "drift_scan_observations",
    }
)
_MEMBER_KEYS = frozenset(
    {
        "source_id",
        "display_name",
        "weight",
        "skus",
        "statistic",
        "variant",
        "extra_require",
        "notes",
    }
)
_DYNAMIC_WEIGHTS_KEYS = frozenset(
    {
        "scheme",
        "lookback_horizons_hours",
        "forward_horizons_hours",
        "history_days",
        "half_life_days",
        "ridge_lambda",
        "gamma",
        "weight_min",
        "weight_max",
        "min_train_samples",
        "target_variance_floor",
        "switch_min_eligible",
        "max_abs_log_return",
        "source_weight_caps",
        "attendance_floor",
    }
)
_JUMP_SCREEN_KEYS = frozenset(
    {
        "quarantine_pct",
        "corroborate_pct",
        "min_corroborators",
        "reference_max_lookback",
    }
)
_MANUAL_EXCLUSION_KEYS = frozenset({"date", "source_id", "reason", "hour"})
_RECORD_EXCLUSION_KEYS = frozenset({"date", "hour", "reason"})


def _refuse_unknown_keys(obj: Dict[str, Any], allowed: Any, label: str) -> None:
    unknown = sorted(str(k) for k in obj if k not in allowed)
    if unknown:
        raise PanelConfigError(
            f"{label} carries unrecognized key(s) {unknown} -- an unread "
            f"key is silently inert config (typo?); the documented schema "
            f"admits {sorted(allowed)}"
        )


def _canonical_date(value: Any, label: str) -> str:
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise PanelConfigError(
            f"{label} must be YYYY-MM-DD, got {value!r}"
        ) from exc
    if parsed.isoformat() != str(value):
        # Exact-string matching downstream (the basket manual_exclusions
        # lesson): a non-canonical form validates and then never fires.
        raise PanelConfigError(
            f"{label} must be canonical YYYY-MM-DD, got {value!r}"
        )
    return str(value)


def _clean_index_prefix(prefix: Any, label: str) -> str:
    text = str(prefix or "")
    if (
        not text.startswith("index/")
        or "\\" in text
        or any(seg in ("", ".", "..") for seg in text.split("/"))
    ):
        # The basket rule (gpu_index.index.config): the lanes share the curve
        # bucket, so "never touches keys outside index/" stays mechanical.
        raise PanelConfigError(
            f"{label} must live under 'index/' with clean path segments: "
            f"{prefix!r}"
        )
    return text


def _nested(a: str, b: str) -> bool:
    return a.startswith(b + "/") or b.startswith(a + "/")


def _validate_token_list(raw: Any, label: str) -> List[str]:
    if not isinstance(raw, list) or not raw:
        raise PanelConfigError(f"{label} must be a non-empty list of tokens")
    tokens: List[str] = []
    for token in raw:
        norm = normalize_label(token)
        if not norm or not _TOKEN_CHARS_RE.match(norm):
            raise PanelConfigError(
                f"{label} token {token!r} must normalize to [A-Z0-9 ]+ "
                f"(the gpu_index.observatory.catalog token rule)"
            )
        tokens.append(str(token))
    if len({normalize_label(t) for t in tokens}) != len(tokens):
        raise PanelConfigError(f"{label} contains duplicate tokens: {raw!r}")
    return tokens


def _require_number(
    value: Any, label: str, *, lo: float, lo_exclusive: bool = True
) -> float:
    ok = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > lo if lo_exclusive else value >= lo)
    )
    if not ok:
        raise PanelConfigError(
            f"{label} must be a finite number "
            f"{'>' if lo_exclusive else '>='} {lo}, got {value!r}"
        )
    return float(value)


def _require_int(value: Any, label: str, *, lo: int, hi: Optional[int] = None) -> int:
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= lo
    if ok and hi is not None:
        ok = value <= hi
    if not ok:
        bound = f"in {lo}..{hi}" if hi is not None else f">= {lo}"
        raise PanelConfigError(f"{label} must be an int {bound}, got {value!r}")
    return int(value)


def validate_panel_config(cfg: Dict[str, Any]) -> None:
    """Validate an in-memory panel config dict (load_panel_config wraps
    this; tests validate their fixtures through it so an engine test can
    never run a config the loader would refuse)."""
    if not isinstance(cfg, dict):
        raise PanelConfigError(f"panel config must be an object, got {cfg!r}")
    _refuse_unknown_keys(cfg, _TOP_LEVEL_KEYS, "panel config")
    for field in (
        "panel_id",
        "bucket_prefix",
        "genesis_date",
        "record_sources",
        "slot_grids",
        "members",
        "calc",
    ):
        if not cfg.get(field):
            raise PanelConfigError(f"panel config missing {field!r}")
    if not isinstance(cfg["panel_id"], str) or not cfg["panel_id"].strip():
        raise PanelConfigError("panel_id must be a non-empty string")

    prefix = _clean_index_prefix(cfg["bucket_prefix"], "bucket_prefix")
    if prefix == RAW_OBSERVATORY_PREFIX or _nested(prefix, RAW_OBSERVATORY_PREFIX):
        # The record a panel READS must never be a keyspace a panel lane
        # WRITES -- a composite or fx object landing under the observatory
        # prefix would shadow the primary record.
        raise PanelConfigError(
            f"bucket_prefix {prefix!r} collides with the read-only record "
            f"keyspace {RAW_OBSERVATORY_PREFIX!r}"
        )
    # The methodology-registry control plane is likewise never a
    # lane keyspace: a doc naming it could not stomp the control keys
    # (distinct key shapes) but would pollute the registry namespace with
    # composites/ and fx/ subtrees.
    _registry_prefix = "index/methodology_registry"
    if prefix == _registry_prefix or _nested(prefix, _registry_prefix):
        raise PanelConfigError(
            f"bucket_prefix {prefix!r} collides with the methodology "
            f"registry keyspace {_registry_prefix!r}"
        )
    for known in KNOWN_LANE_PREFIXES:
        if known == RAW_OBSERVATORY_PREFIX:
            continue
        if _nested(prefix, known):
            # Strict nesting only: EQUALITY with a non-record lane prefix
            # is sanctioned (migrated B300/B200 lanes share their basket
            # prefix; a panel config naming its own listed prefix is the
            # normal case) -- composites key per methodology_id, so equal
            # prefixes cannot collide keys, but a NESTED prefix would let
            # one lane's LIST see another's objects.
            raise PanelConfigError(
                f"bucket_prefix {prefix!r} nests with the known lane "
                f"keyspace {known!r}"
            )

    genesis = _canonical_date(cfg["genesis_date"], "genesis_date")

    # TOP-LEVEL operational knob (docstring): required, never in calc.
    _require_int(
        cfg.get("drift_scan_observations"), "drift_scan_observations", lo=1
    )

    _validate_record_sources(cfg["record_sources"], genesis)
    schedule = panel_schedule(cfg)  # one home for every slot_grids rule
    _validate_record_exclusions(cfg.get("record_exclusions"), schedule)

    reject_tokens = cfg.get("reject_tokens")
    if reject_tokens is not None:
        _validate_token_list(reject_tokens, "reject_tokens")

    member_ids = _validate_members(cfg["members"])
    _validate_calc(
        cfg["calc"],
        cfg["members"],
        member_ids,
        cfg["slot_grids"],
        schedule,
        prefix,
    )


def _validate_record_sources(entries: Any, genesis: str) -> None:
    if not isinstance(entries, list) or not entries:
        raise PanelConfigError("record_sources must be a non-empty list")
    prev_to: Optional[str] = None
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PanelConfigError(
                f"record_sources[{i}] must be an object, got {entry!r}"
            )
        if entry.get("kind") not in VALID_RECORD_KINDS:
            raise PanelConfigError(
                f"record_sources[{i}].kind must be one of "
                f"{sorted(VALID_RECORD_KINDS)}, got {entry.get('kind')!r}"
            )
        _clean_index_prefix(entry.get("prefix"), f"record_sources[{i}].prefix")
        from_date = _canonical_date(
            entry.get("from_date"), f"record_sources[{i}].from_date"
        )
        if i == 0:
            if from_date != genesis:
                raise PanelConfigError(
                    f"record_sources[0].from_date {from_date!r} must equal "
                    f"genesis_date {genesis!r} -- coverage starts at genesis"
                )
        else:
            expected = (
                date.fromisoformat(prev_to) + timedelta(days=1)
            ).isoformat()
            if from_date != expected:
                raise PanelConfigError(
                    f"record_sources[{i}].from_date {from_date!r} must be "
                    f"the day after the prior entry's to_date ({expected!r}) "
                    f"-- coverage must be contiguous and non-overlapping"
                )
        last = i == len(entries) - 1
        if last:
            if "to_date" in entry:
                # A closed final record makes some future scheduled
                # observation unreadable -- refuse at load, never surprise
                # at runtime.
                raise PanelConfigError(
                    f"record_sources[{i}] is the final entry and must be "
                    f"open-ended (no to_date), got {entry.get('to_date')!r}"
                )
        else:
            to_date = _canonical_date(
                entry.get("to_date"), f"record_sources[{i}].to_date"
            )
            if to_date < from_date:
                raise PanelConfigError(
                    f"record_sources[{i}].to_date {to_date!r} precedes its "
                    f"from_date {from_date!r}"
                )
            prev_to = to_date


def _validate_record_exclusions(entries: Any, schedule: PanelSchedule) -> None:
    """Top-level record_exclusions (adversarial review F6, amended into
    the mints before any observation published): [{date, hour, reason}]
    quarantining ONE scheduled (date, hour)'s record object -- the escape
    hatch for a poisoned/unparseable stored snapshot that would otherwise
    crash every firing forever (publish-in-order blocks the lane behind
    it; earliest-key-wins means a later good snapshot can never shadow
    it). Optional; every entry needs a canonical date, an hour ON the
    era grid (an off-grid entry would quarantine nothing -- the silently
    inert incident-record class), a non-empty reason, and no duplicates."""
    if entries is None:
        return
    if not isinstance(entries, list):
        raise PanelConfigError("record_exclusions must be a list")
    seen: set = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PanelConfigError(
                f"record_exclusions entries must be objects: {entry!r}"
            )
        _refuse_unknown_keys(
            entry, _RECORD_EXCLUSION_KEYS, "record_exclusions entry"
        )
        day = _canonical_date(entry.get("date"), "record_exclusions date")
        hour = _require_int(
            entry.get("hour"),
            f"record_exclusions hour for {day!r}",
            lo=0,
            hi=23,
        )
        if not str(entry.get("reason") or "").strip():
            raise PanelConfigError(
                f"record_exclusions entry for {day!r} hour {hour} needs a "
                f"non-empty reason"
            )
        if not schedule.is_scheduled(date_hour_to_stamp(day, hour)):
            raise PanelConfigError(
                f"record_exclusions hour {hour} on {day!r} is not a "
                f"scheduled observation hour of that date's era grid -- "
                f"the quarantine would silently cover nothing"
            )
        pair = (day, hour)
        if pair in seen:
            raise PanelConfigError(
                f"duplicate record_exclusions entry for {pair!r}"
            )
        seen.add(pair)


def _validate_members(members: Any) -> List[str]:
    if not isinstance(members, list) or not members:
        raise PanelConfigError("members must be a non-empty list")
    # Lazy import: gpu_index.index.panel is the single home of the panel statistic
    # registry (the SOURCE_STATISTIC_FNS rule).
    from gpu_index.index.panel import PANEL_STATISTIC_FNS

    seen: set = set()
    weights: List[float] = []
    for member in members:
        if not isinstance(member, dict):
            raise PanelConfigError(f"members entries must be objects: {member!r}")
        _refuse_unknown_keys(member, _MEMBER_KEYS, "member entry")
        sid = member.get("source_id")
        if not sid or not isinstance(sid, str):
            raise PanelConfigError(f"member without source_id: {member!r}")
        if sid in seen:
            raise PanelConfigError(f"duplicate member source_id {sid!r}")
        seen.add(sid)
        weight = member.get("weight")
        _require_number(weight, f"member {sid!r} weight", lo=0)
        if round(float(weight), 6) != float(weight):
            # The pinned-vector precision (gpu_index.index.config's dynamic_weights
            # rule): a >6dp weight forks the fallback vector at rounding.
            raise PanelConfigError(
                f"member {sid!r} weight must be exact at 6 decimal places, "
                f"got {weight!r}"
            )
        weights.append(float(weight))
        skus = member.get("skus")
        if (
            not isinstance(skus, list)
            or not skus
            or not all(isinstance(s, str) and s.strip() for s in skus)
        ):
            raise PanelConfigError(
                f"member {sid!r} skus must be a non-empty list of sku "
                f"strings, got {skus!r}"
            )
        if len(set(skus)) != len(skus):
            raise PanelConfigError(f"member {sid!r} skus contains duplicates")
        statistic = member.get("statistic")
        if statistic is not None and statistic not in PANEL_STATISTIC_FNS:
            raise PanelConfigError(
                f"member {sid!r} names unknown panel statistic "
                f"{statistic!r} (known: {sorted(PANEL_STATISTIC_FNS)})"
            )
        variant = member.get("variant")
        if variant is not None:
            if not isinstance(variant, dict):
                raise PanelConfigError(
                    f"member {sid!r} variant must be an object"
                )
            mode = variant.get("mode")
            if mode not in VALID_VARIANT_MODES:
                raise PanelConfigError(
                    f"member {sid!r} variant.mode must be one of "
                    f"{sorted(VALID_VARIANT_MODES)}, got {mode!r}"
                )
            if mode == "label":
                _validate_token_list(
                    variant.get("require_tokens"),
                    f"member {sid!r} variant.require_tokens",
                )
            else:
                evidence = variant.get("evidence")
                if not isinstance(evidence, str) or not evidence.strip():
                    # Design section 8: declared mode exists only where the
                    # provider's OWN surface confirms a single variant --
                    # the quote is the seat's license.
                    raise PanelConfigError(
                        f"member {sid!r} variant mode 'declared' requires "
                        f"a non-empty evidence string"
                    )
                if "require_tokens" in variant:
                    raise PanelConfigError(
                        f"member {sid!r} variant mode 'declared' must not "
                        f"also carry require_tokens (one rule per seat)"
                    )
        extra_require = member.get("extra_require")
        if extra_require is not None:
            if not isinstance(extra_require, dict) or not extra_require:
                # An empty object would validate and screen nothing --
                # the silently-inert-config class, refused like the rest.
                raise PanelConfigError(
                    f"member {sid!r} extra_require must be a non-empty "
                    f"object of {{extra key: required value}}, got "
                    f"{extra_require!r}"
                )
            for key, value in extra_require.items():
                if (
                    not isinstance(key, str)
                    or not key.strip()
                    or not isinstance(value, str)
                    or not value.strip()
                ):
                    raise PanelConfigError(
                        f"member {sid!r} extra_require entries must map "
                        f"non-empty string keys to non-empty string "
                        f"values, got {key!r}: {value!r}"
                    )
    if abs(sum(weights) - 1.0) > 1e-9:
        raise PanelConfigError(
            f"member weights must sum to 1.0, got {sum(weights)}"
        )
    return sorted(seen)


def _validate_calc(
    calc: Any,
    members: List[Dict[str, Any]],
    member_ids: List[str],
    slot_grids: List[Dict[str, Any]],
    schedule: PanelSchedule,
    prefix: str,
) -> None:
    if not isinstance(calc, dict):
        raise PanelConfigError("calc must be an object")
    _refuse_unknown_keys(calc, _CALC_KEYS, "calc")
    # One home for the filter/composite-statistic vocabulary (lazy import,
    # the gpu_index.index.config rule).
    from gpu_index.index.composite import (
        MEDIAN_STDDEV_VOTES,
        VALID_COMPOSITE_STATISTICS,
        VALID_FILTER_TERMS,
    )
    from gpu_index.index.panel import PANEL_STATISTIC_PARAM_DEFAULTS

    methodology_id = calc.get("methodology_id")
    if not isinstance(methodology_id, str) or not methodology_id.strip():
        raise PanelConfigError("calc.methodology_id must be a non-empty string")
    if (
        not _METHODOLOGY_ID_RE.fullmatch(methodology_id)
        or methodology_id in (".", "..")
    ):
        # The id becomes a bucket key segment (security review): one
        # clean [A-Za-z0-9._-]+ segment, never a dot-segment -- 'a/b' or
        # '..' would address keys outside the lane's composites keyspace.
        raise PanelConfigError(
            f"calc.methodology_id must be a single clean key segment "
            f"([A-Za-z0-9._-]+, not '.'/'..'), got {methodology_id!r}"
        )
    if prefix in DAILY_LANE_PREFIXES and methodology_id in FROZEN_METHODOLOGY_IDS:
        # See FROZEN_METHODOLOGY_IDS: on a shared daily prefix, a panel
        # naming a frozen daily id would publish hour-keyed artifacts
        # into the frozen series' keyspace and move its latest.json
        # pointer -- refuse at load, before any bucket access exists.
        raise PanelConfigError(
            f"calc.methodology_id {methodology_id!r} names a FROZEN daily "
            f"series under the shared prefix {prefix!r} -- a panel lane "
            f"must mint its own id, never write into a frozen keyspace"
        )

    _require_int(
        calc.get("min_sources_to_claim"),
        "calc.min_sources_to_claim",
        lo=1,
        hi=len(member_ids),
    )

    if "interruptible_tiers" in calc:
        # The exclusion-list key is RETIRED on panels (methodology
        # section 5: on-demand only). Silently ignoring it would be worse
        # than refusing it -- a config author would read a live screen
        # where none runs -- so refuse loudly and name the successor. The
        # frozen DAILY basket lanes (gpu_index.index.config / gpu_index.index.composite)
        # keep interruptible_tiers untouched: their published bytes
        # depend on it.
        raise PanelConfigError(
            "calc.interruptible_tiers is not a panel key: panel tier "
            "eligibility is the ALLOW-LIST calc.eligible_tiers "
            "(methodology section 5 'on-demand only', the hourly-mint "
            "tier reconciliation) -- declare eligible_tiers ['on-demand'] "
            "instead"
        )
    tiers = calc.get("eligible_tiers")
    if (
        not isinstance(tiers, list)
        or not tiers
        or not all(isinstance(t, str) and t.strip() for t in tiers)
    ):
        raise PanelConfigError(
            f"calc.eligible_tiers must be a non-empty list of tier "
            f"strings (required -- the allow-list is lane law), "
            f"got {tiers!r}"
        )
    if len(set(tiers)) != len(tiers):
        raise PanelConfigError(
            f"calc.eligible_tiers contains duplicates: {tiers!r}"
        )
    if "on-demand" not in tiers:
        # methodology section 5: on-demand IS the underlying; an
        # allow-list without it defines a panel over nothing it prices.
        raise PanelConfigError(
            f"calc.eligible_tiers must contain 'on-demand' (methodology "
            f"section 5: the on-demand rate is the underlying), got "
            f"{tiers!r}"
        )

    for field, lo in (("filter_window", 2), ("filter_warmup", 1)):
        if calc.get(field) is not None:
            _require_int(calc[field], f"calc.{field}", lo=lo)
    if calc.get("filter_sigma") is not None:
        _require_number(calc["filter_sigma"], "calc.filter_sigma", lo=0)
    sigma_floor = calc.get("filter_sigma_floor")
    if sigma_floor is not None:
        _require_number(
            sigma_floor, "calc.filter_sigma_floor", lo=0, lo_exclusive=False
        )
    filter_terms = calc.get("filter_terms")
    if filter_terms is not None and filter_terms not in VALID_FILTER_TERMS:
        raise PanelConfigError(
            f"calc.filter_terms must be one of {sorted(VALID_FILTER_TERMS)}, "
            f"got {filter_terms!r}"
        )
    if "composite_statistic" in calc:
        composite_statistic = calc["composite_statistic"]
        if composite_statistic not in VALID_COMPOSITE_STATISTICS:
            raise PanelConfigError(
                f"calc.composite_statistic must be one of "
                f"{sorted(VALID_COMPOSITE_STATISTICS)}, "
                f"got {composite_statistic!r}"
            )
        if composite_statistic == MEDIAN_STDDEV_VOTES and not (
            isinstance(sigma_floor, (int, float))
            and not isinstance(sigma_floor, bool)
            and sigma_floor > 0
        ):
            # The basket rule verbatim: a frozen list price (sigma 0) must
            # never vote with conviction it never earned.
            raise PanelConfigError(
                "calc.composite_statistic 'median_ci_votes' requires "
                f"calc.filter_sigma_floor > 0, got {sigma_floor!r}"
            )
    if calc.get("manual_verify_pct") is not None:
        _require_number(
            calc["manual_verify_pct"], "calc.manual_verify_pct", lo=0
        )

    fx_lane = calc.get("fx_lane")
    if fx_lane not in ("ecb", "none"):
        # REQUIRED for panels (no default): whether a lane converts is lane
        # law, and a silently defaulted lane could price EUR seats it never
        # ruled on (or hold out seats it meant to convert).
        raise PanelConfigError(
            f"calc.fx_lane must be 'ecb' or 'none' (required), got {fx_lane!r}"
        )
    if calc.get("fx_max_staleness_days") is not None:
        _require_int(
            calc["fx_max_staleness_days"], "calc.fx_max_staleness_days", lo=1
        )

    _validate_manual_exclusions(
        calc.get("manual_exclusions"), member_ids, schedule
    )
    _validate_jump_screen(calc.get("jump_screen"))
    _validate_statistic_params(
        calc.get("statistic_params"), members, PANEL_STATISTIC_PARAM_DEFAULTS
    )
    _validate_dynamic_weights(
        calc.get("dynamic_weights"), members, member_ids, slot_grids
    )
    if "drift_scan_observations" in calc:
        # Retired location (amended into the mints before any observation
        # published): the drift-scan bound is an OPERATIONAL knob and must
        # never ride calc_params/artifact bytes -- silently ignoring the
        # key here would leave a config author reading a live bound where
        # none runs (the interruptible_tiers doctrine), so refuse loudly
        # and name the successor.
        raise PanelConfigError(
            "calc.drift_scan_observations is not a calc key: the drift-scan "
            "bound is operational (it must not ride calc_params/artifact "
            "bytes) -- declare TOP-LEVEL drift_scan_observations instead"
        )


def _validate_manual_exclusions(
    exclusions: Any, member_ids: List[str], schedule: PanelSchedule
) -> None:
    if exclusions is None:
        return
    if not isinstance(exclusions, list):
        raise PanelConfigError("calc.manual_exclusions must be a list")
    seen_triples: set = set()
    day_scoped: set = set()
    hour_scoped: set = set()
    for entry in exclusions:
        if not isinstance(entry, dict):
            raise PanelConfigError(
                f"calc.manual_exclusions entries must be objects: {entry!r}"
            )
        _refuse_unknown_keys(
            entry, _MANUAL_EXCLUSION_KEYS, "manual_exclusions entry"
        )
        day = _canonical_date(entry.get("date"), "manual_exclusions date")
        sid = entry.get("source_id")
        if sid not in member_ids:
            raise PanelConfigError(
                f"manual_exclusions source_id {sid!r} is not a configured "
                f"member"
            )
        if not str(entry.get("reason") or "").strip():
            raise PanelConfigError(
                f"manual_exclusions entry for {day!r}/{sid!r} needs a "
                f"non-empty reason"
            )
        hour: Optional[int] = None
        if "hour" in entry:
            hour = _require_int(
                entry["hour"],
                f"manual_exclusions hour for {day!r}/{sid!r}",
                lo=0,
                hi=23,
            )
            if not schedule.is_scheduled(date_hour_to_stamp(day, hour)):
                # Same doctrine as the daily lanes' canonical-date rule: an
                # exclusion that validates but can never match an observation
                # is a silently inert incident record. An off-grid hour (or a
                # pre-genesis date) holds out nothing -- refuse at load.
                raise PanelConfigError(
                    f"manual_exclusions hour {hour} for {day!r}/{sid!r} is "
                    f"not a scheduled observation hour of that date's era "
                    f"grid -- the exclusion would silently hold out nothing"
                )
        triple = (day, str(sid), hour)
        if triple in seen_triples:
            raise PanelConfigError(
                f"duplicate manual_exclusions entry for {triple!r}"
            )
        seen_triples.add(triple)
        pair = (day, str(sid))
        if hour is None:
            day_scoped.add(pair)
        else:
            hour_scoped.add(pair)
    overlap = day_scoped & hour_scoped
    if overlap:
        # A date-level hold-out already covers every hour: an additional
        # hour-scoped entry for the same (date, source) is either redundant
        # or a mis-scoped edit -- both are config errors, refuse at load.
        raise PanelConfigError(
            f"manual_exclusions mixes date-level and hour-level entries for "
            f"{sorted(overlap)!r} -- one scope per (date, source_id)"
        )


def _validate_jump_screen(screen: Any) -> None:
    if not isinstance(screen, dict):
        # REQUIRED: L5-at-calc thresholds are methodology (they ride
        # calc_params into artifact bytes) -- no code default may pin them.
        raise PanelConfigError(
            "calc.jump_screen is required ({quarantine_pct, "
            "corroborate_pct, min_corroborators, reference_max_lookback})"
        )
    _refuse_unknown_keys(screen, _JUMP_SCREEN_KEYS, "calc.jump_screen")
    quarantine = _require_number(
        screen.get("quarantine_pct"), "calc.jump_screen.quarantine_pct", lo=0
    )
    corroborate = _require_number(
        screen.get("corroborate_pct"), "calc.jump_screen.corroborate_pct", lo=0
    )
    if corroborate > quarantine:
        # Movers between the two thresholds would fire quarantine but never
        # corroborate each other (the basket capture_screens rule).
        raise PanelConfigError(
            "calc.jump_screen.corroborate_pct must not exceed "
            f"quarantine_pct ({corroborate!r} > {quarantine!r})"
        )
    _require_int(
        screen.get("min_corroborators"),
        "calc.jump_screen.min_corroborators",
        lo=1,
    )
    _require_int(
        screen.get("reference_max_lookback"),
        "calc.jump_screen.reference_max_lookback",
        lo=1,
    )


def _validate_statistic_params(
    raw: Any,
    members: List[Dict[str, Any]],
    defaults: Dict[str, Dict[str, int]],
) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise PanelConfigError("calc.statistic_params must be an object")
    named = {m.get("statistic") for m in members if m.get("statistic")}
    for stat_id, params in raw.items():
        if stat_id not in defaults:
            raise PanelConfigError(
                f"calc.statistic_params names unknown statistic {stat_id!r} "
                f"(known: {sorted(defaults)})"
            )
        if stat_id not in named:
            # Params for a statistic no member names would validate and
            # then never fire -- the silently-inert-config class.
            raise PanelConfigError(
                f"calc.statistic_params entry {stat_id!r} is not named by "
                f"any member's statistic"
            )
        if not isinstance(params, dict):
            raise PanelConfigError(
                f"calc.statistic_params[{stat_id!r}] must be an object"
            )
        for name, value in params.items():
            if name not in defaults[stat_id]:
                raise PanelConfigError(
                    f"calc.statistic_params[{stat_id!r}] unknown param "
                    f"{name!r} (known: {sorted(defaults[stat_id])})"
                )
            _require_int(
                value, f"calc.statistic_params[{stat_id!r}].{name}", lo=1
            )


def _validate_dynamic_weights(
    dw: Any,
    members: List[Dict[str, Any]],
    member_ids: List[str],
    slot_grids: List[Dict[str, Any]],
) -> None:
    if not isinstance(dw, dict):
        # REQUIRED: rule A1 -- panel weights recompute at every
        # observation; a panel without the block has no weight law.
        raise PanelConfigError("calc.dynamic_weights is required for panels")
    _refuse_unknown_keys(dw, _DYNAMIC_WEIGHTS_KEYS, "calc.dynamic_weights")
    from gpu_index.index.weights import VALID_WEIGHT_SCHEMES, validate_attendance_floor

    if dw.get("scheme") not in VALID_WEIGHT_SCHEMES:
        raise PanelConfigError(
            f"calc.dynamic_weights.scheme must be one of "
            f"{sorted(VALID_WEIGHT_SCHEMES)}, got {dw.get('scheme')!r}"
        )
    # Horizon expressibility is checked against the FINAL era's spacing --
    # the open-ended era the lane lives in forever. A horizon a CLOSED
    # earlier era cannot express simply realizes zero samples there (its
    # endpoints never fall on that era's stamps), which is bounded and
    # honest; a horizon the final era cannot express would be the
    # silently-inert-scheme failure the basket validator refuses.
    final_slots = sorted(int(h) for h in slot_grids[-1]["slot_hours_utc"])
    gaps = [b - a for a, b in zip(final_slots, final_slots[1:])] + [
        24 - final_slots[-1] + final_slots[0]
    ]
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
            raise PanelConfigError(
                f"calc.dynamic_weights.{field} must be a strictly "
                f"increasing list of ints >= 1, got {horizons!r}"
            )
        for horizon in horizons:
            if horizon % slot_spacing != 0 or horizon < slot_spacing:
                raise PanelConfigError(
                    f"calc.dynamic_weights.{field} entry {horizon} is not "
                    f"expressible on the final era's {slot_spacing}h slot "
                    f"grid (must be a multiple >= {slot_spacing})"
                )
    for field in ("history_days", "min_train_samples"):
        _require_int(dw.get(field), f"calc.dynamic_weights.{field}", lo=1)
    for field, lo_exclusive in (
        ("half_life_days", True),
        ("ridge_lambda", False),
        ("gamma", False),
    ):
        _require_number(
            dw.get(field),
            f"calc.dynamic_weights.{field}",
            lo=0,
            lo_exclusive=lo_exclusive,
        )
    if dw.get("target_variance_floor") is not None:
        _require_number(
            dw["target_variance_floor"],
            "calc.dynamic_weights.target_variance_floor",
            lo=0,
        )
    if dw.get("max_abs_log_return") is not None:
        _require_number(
            dw["max_abs_log_return"],
            "calc.dynamic_weights.max_abs_log_return",
            lo=0,
        )
    if dw.get("switch_min_eligible") is not None:
        _require_int(
            dw["switch_min_eligible"],
            "calc.dynamic_weights.switch_min_eligible",
            lo=1,
            hi=len(member_ids),
        )
    w_min = dw.get("weight_min")
    w_max = dw.get("weight_max")
    for field, value in (("weight_min", w_min), ("weight_max", w_max)):
        ok = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 < value <= 1
        )
        if not ok:
            raise PanelConfigError(
                f"calc.dynamic_weights.{field} must be a number in (0, 1], "
                f"got {value!r}"
            )
    if w_min > w_max:
        raise PanelConfigError(
            f"calc.dynamic_weights.weight_min ({w_min}) must not exceed "
            f"weight_max ({w_max})"
        )
    if len(member_ids) * w_min >= 1.0:
        raise PanelConfigError(
            f"calc.dynamic_weights.weight_min {w_min} is infeasible for "
            f"{len(member_ids)} members (N*w_min must be < 1)"
        )
    # A2: the attendance floor is REQUIRED for panels -- gpu_index.index.weights is
    # the one home of the rule and its bounds.
    try:
        validate_attendance_floor(dw)
    except ValueError as exc:
        raise PanelConfigError(str(exc)) from exc
    min_span_hours = (
        max(dw["lookback_horizons_hours"])
        + max(dw["forward_horizons_hours"])
        + dw["min_train_samples"] * slot_spacing
        + slot_spacing
    )
    if dw["history_days"] * 24 < min_span_hours:
        raise PanelConfigError(
            f"calc.dynamic_weights.history_days {dw['history_days']} can "
            f"never define a score (needs >= {min_span_hours} hours for the "
            f"configured horizons and sample gates)"
        )
    caps = dw.get("source_weight_caps")
    if caps is not None:
        if not isinstance(caps, dict):
            raise PanelConfigError(
                "calc.dynamic_weights.source_weight_caps must be an object"
            )
        for sid, cap in caps.items():
            if sid not in member_ids:
                raise PanelConfigError(
                    f"source_weight_caps source_id {sid!r} is not a "
                    f"configured member"
                )
            ok = (
                isinstance(cap, (int, float))
                and not isinstance(cap, bool)
                and w_min <= cap <= 1
            )
            if not ok:
                raise PanelConfigError(
                    f"source_weight_caps[{sid!r}] must be a number in "
                    f"[weight_min, 1], got {cap!r}"
                )
