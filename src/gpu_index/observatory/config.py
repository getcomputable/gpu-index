# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Load + validate config/raw_observatory.json.

Every operational parameter (slots, canonical hour, timeouts, prefix, claim
floor, per-source options) is config so growing the lane — more sources,
more chips, a different cadence — is a config edit, never a code change
(config is the operational contract).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.observatory.catalog import SkuCatalogError, load_sku_catalog

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "raw_observatory.json"

# Source-type disclosure vocabulary (who is speaking when a price prints):
#   direct_principal  — the operator's own list price for its own capacity
#   direct_partnered  — an operator fronting partner-owned capacity
#   marketplace       — a venue where third-party hosts post asks
#   reseller          — re-publishes other clouds' capacity under its brand
#   aggregator        — collects other providers' prices; leads, not prints
#   hyperscaler       — the big-cloud tier (list prices, enterprise heavy)
VALID_SOURCE_TYPES = {
    "direct_principal",
    "direct_partnered",
    "marketplace",
    "reseller",
    "aggregator",
    "hyperscaler",
}

# The observatory may share a bucket's index/ keyspace with the other
# lanes but must never collide with another lane's prefix: the basket
# lanes AND the hourly panel-index lanes (which READ this lane's record
# and WRITE composites under their own prefixes -- an observatory config
# typo'd onto one of them could shadow a contractual series' keys). Grow
# this tuple when a new lane lands; it is pinned in set-equality lockstep
# with gpu_index.index.panel_config.KNOWN_LANE_PREFIXES minus the
# observatory's own prefix (test_panel_engine), so neither side can drift
# alone.
RESERVED_LANE_PREFIXES = (
    "index/b300_basket",
    "index/b200_basket",
    "index/h100_sxm",
    "index/h200_sxm",
    "index/h100_broad",
    "index/h200_broad",
)


class ObservatoryConfigError(RuntimeError):
    """config/raw_observatory.json is missing or malformed."""


def load_observatory_config(path: Optional[Path] = None) -> Dict[str, Any]:
    # Precedence: explicit argument > env override > repo default. An
    # exported RAW_OBSERVATORY_CONFIG_PATH must never silently eat a
    # --config flag.
    cfg_path = Path(
        path
        or os.environ.get("RAW_OBSERVATORY_CONFIG_PATH")
        or DEFAULT_CONFIG_PATH
    )
    if not cfg_path.exists():
        raise ObservatoryConfigError(f"observatory config missing: {cfg_path}")
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise ObservatoryConfigError(
            f"observatory config unparseable: {cfg_path}: {exc}"
        ) from exc
    _validate(cfg)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_catalog_path(cfg: Dict[str, Any]) -> Path:
    return _REPO_ROOT / str(cfg["sku_catalog_path"])


def capture_grid(cfg: Dict[str, Any]) -> Tuple[List[int], Optional[int], bool]:
    """The ONE normalization of a capture config's slot vocabulary:
    (slot marks as minute-of-day, canonical mark as minute-of-day or
    None, minute_tokens). Hour-vocabulary configs (capture_slots_utc /
    canonical_slot_utc) normalize h -> h*60 and keep the legacy
    slot<HH>- key tokens; minute-vocabulary configs
    (capture_slot_minutes_utc / canonical_slot_minute_utc — 15-min
    cadence design 2026-08-27) are minute-of-day already and write
    slot<HHMM>- tokens for every mark. _validate guarantees exactly one
    vocabulary is present."""
    if "capture_slot_minutes_utc" in cfg:
        slots = [int(m) for m in cfg["capture_slot_minutes_utc"]]
        canonical = cfg.get("canonical_slot_minute_utc")
        return slots, (None if canonical is None else int(canonical)), True
    slots = [int(h) * 60 for h in cfg["capture_slots_utc"]]
    canonical = cfg.get("canonical_slot_utc")
    return slots, (None if canonical is None else int(canonical) * 60), False


def _validate(cfg: Dict[str, Any]) -> None:
    for field in (
        "lane_id",
        "bucket_prefix",
        "sku_catalog_path",
        "sources",
    ):
        if not cfg.get(field):
            raise ObservatoryConfigError(f"observatory config missing {field!r}")
    if ("capture_slots_utc" in cfg) == ("capture_slot_minutes_utc" in cfg):
        # Exactly one vocabulary: both (or neither) is a config
        # contradiction, and a silently-ignored second grid would be the
        # silently-inert-key failure class.
        raise ObservatoryConfigError(
            "observatory config must declare exactly one of "
            "capture_slots_utc / capture_slot_minutes_utc"
        )
    if "capture_slots_utc" in cfg and "canonical_slot_minute_utc" in cfg:
        raise ObservatoryConfigError(
            "canonical_slot_minute_utc requires capture_slot_minutes_utc"
        )
    if "capture_slot_minutes_utc" in cfg and "canonical_slot_utc" in cfg:
        raise ObservatoryConfigError(
            "canonical_slot_utc requires capture_slots_utc"
        )

    prefix = str(cfg["bucket_prefix"])
    if (
        not prefix.startswith("index/")
        or "\\" in prefix
        or any(seg in ("", ".", "..") for seg in prefix.split("/"))
    ):
        # Same mechanical fence as the basket lanes: dot segments could be
        # gateway-normalized straight out of the index/ keyspace.
        raise ObservatoryConfigError(
            f"bucket_prefix must live under 'index/' with clean path "
            f"segments: {prefix!r}"
        )
    for reserved in RESERVED_LANE_PREFIXES:
        if (
            prefix == reserved
            or prefix.startswith(reserved + "/")
            or reserved.startswith(prefix + "/")
        ):
            # Nesting either way would let one lane's LIST/PUT see the
            # other's objects — the b200/b300 keyspace separation is an
            # invariant this lane must never be able to break by
            # config typo.
            raise ObservatoryConfigError(
                f"bucket_prefix {prefix!r} collides with the reserved basket "
                f"lane keyspace {reserved!r}"
            )

    if "capture_slots_utc" in cfg:
        slots = cfg["capture_slots_utc"]
        if not isinstance(slots, list) or not all(
            isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23
            for h in slots
        ):
            raise ObservatoryConfigError(
                f"capture_slots_utc must be UTC hours 0-23: {slots!r}"
            )
        if not 1 <= len(set(slots)) <= 24 or len(set(slots)) != len(slots):
            # Hourly cadence: the fence widened from the original 1-8 to
            # the full 24 — the vendor-politeness math was re-done at
            # hourly (each capture sends 1-2 requests per vendor;
            # vast/computepulse keep per-call spacing; ~0.5 req/min
            # worst-case average per vendor). Duplicate hours are still a
            # typo, and the fence still exists so a sub-hourly ambition
            # has to move to the MINUTE vocabulary deliberately, not just
            # grow this list.
            raise ObservatoryConfigError(
                f"capture_slots_utc must hold 1-24 distinct slots: {slots!r}"
            )
        canonical = cfg.get("canonical_slot_utc")
        if canonical is not None and (
            not isinstance(canonical, int)
            or isinstance(canonical, bool)
            or canonical not in slots
        ):
            raise ObservatoryConfigError(
                f"canonical_slot_utc {canonical!r} is not one of "
                f"capture_slots_utc {slots!r}"
            )
    else:
        slots = cfg["capture_slot_minutes_utc"]
        if not isinstance(slots, list) or not all(
            isinstance(m, int) and not isinstance(m, bool) and 0 <= m <= 1439
            for m in slots
        ):
            raise ObservatoryConfigError(
                f"capture_slot_minutes_utc must be UTC minutes-of-day "
                f"0-1439: {slots!r}"
            )
        if not 1 <= len(set(slots)) <= 96 or len(set(slots)) != len(slots):
            # 15-minute cadence: the deliberate-edit fence moves with the
            # vocabulary — 96 marks (every quarter hour) is the re-done
            # vendor-politeness ceiling (captures measured ~83s wall; each
            # sends 1-2 requests per vendor, vast keeps per-call spacing);
            # anything DENSER than 15 minutes must again edit code
            # deliberately, not just grow this list.
            raise ObservatoryConfigError(
                f"capture_slot_minutes_utc must hold 1-96 distinct marks: "
                f"{slots!r}"
            )
        canonical = cfg.get("canonical_slot_minute_utc")
        if canonical is not None and (
            not isinstance(canonical, int)
            or isinstance(canonical, bool)
            or canonical not in slots
        ):
            raise ObservatoryConfigError(
                f"canonical_slot_minute_utc {canonical!r} is not one of "
                f"capture_slot_minutes_utc {slots!r}"
            )
    if not isinstance(cfg["lane_id"], str):
        raise ObservatoryConfigError(f"lane_id must be a string: {cfg['lane_id']!r}")

    for field in (
        "per_source_timeout_seconds",
        "per_source_deadline_seconds",
        "capture_budget_seconds",
    ):
        value = cfg.get(field)
        if value is not None and not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            raise ObservatoryConfigError(
                f"{field} must be a positive number, got {value!r}"
            )

    sources = cfg["sources"]
    if not isinstance(sources, list):
        raise ObservatoryConfigError("sources must be a list")
    seen = set()
    for src in sources:
        if not isinstance(src, dict):
            raise ObservatoryConfigError(
                f"sources entries must be objects, got {src!r}"
            )
        sid = src.get("source_id")
        if not sid or not isinstance(sid, str):
            raise ObservatoryConfigError(f"source without source_id: {src!r}")
        if sid in seen:
            raise ObservatoryConfigError(f"duplicate source_id {sid!r}")
        seen.add(sid)
        if not src.get("display_name"):
            raise ObservatoryConfigError(f"source {sid!r} needs display_name")
        if src.get("source_type") not in VALID_SOURCE_TYPES:
            raise ObservatoryConfigError(
                f"source {sid!r} source_type {src.get('source_type')!r} not in "
                f"{sorted(VALID_SOURCE_TYPES)}"
            )
        if not isinstance(src.get("first_party"), bool):
            # Who-is-speaking disclosure is per-observation audit data; an
            # unset flag would silently default and mislabel a reseller.
            raise ObservatoryConfigError(
                f"source {sid!r} needs first_party true/false"
            )
        options = src.get("options")
        if options is not None and not isinstance(options, dict):
            raise ObservatoryConfigError(
                f"source {sid!r} options must be an object, got "
                f"{type(options).__name__}"
            )

    min_claim = cfg.get("min_sources_to_claim", 1)
    if not (
        isinstance(min_claim, int)
        and not isinstance(min_claim, bool)
        and 1 <= min_claim <= len(sources)
    ):
        raise ObservatoryConfigError(
            f"min_sources_to_claim must be an int in 1..{len(sources)}, "
            f"got {min_claim!r}"
        )

    # The catalog is half the schema — a capture with a broken catalog would
    # record every print unmapped, so fail at config load, not mid-capture.
    catalog_path = _REPO_ROOT / str(cfg["sku_catalog_path"])
    try:
        load_sku_catalog(catalog_path)
    except SkuCatalogError as exc:
        raise ObservatoryConfigError(
            f"sku_catalog_path {cfg['sku_catalog_path']!r} failed to load: {exc}"
        ) from exc


def sources_by_id(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {s["source_id"]: s for s in cfg["sources"]}
