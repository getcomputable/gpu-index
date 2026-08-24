# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Build the observatory capture snapshot — per-source raw prints, nothing else.

Same primary-artifact rule as the basket lanes: the
per-source time series IS the record. This document stores every source's
own prints (raw value as published + our per-GPU normalization + audit
fields) and derives nothing cross-source at all — no composite, no basket,
not even basis pairs. The only additions over the raw results are honest
rollups (which skus printed, which identifiers failed to normalize) so a
reader can see coverage without parsing every observation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

from gpu_index.common.slots import rfc3339
from gpu_index.observatory.catalog import match_sku, plausible_band
from gpu_index.observatory.config import sources_by_id

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "raw_price_observatory_capture"

# Identity/audit fields collectors may set at the top level of an
# observation (the basket-proven set). Everything else source-native
# belongs in the schema-free ``extra`` dict.
_PASSTHROUGH_KEYS = (
    "offer_id",
    "machine_id",
    "host_id",
    "verification",
    "memory_gb_label",
    "book_scope",
    "extra",
)


def normalize_observation(
    obs: Dict[str, Any], catalog: Dict[str, Any]
) -> Dict[str, Any]:
    """Fill defaults, derive the canonical sku, flag plausibility.

    Never drops or rewrites a raw field. Currency honesty:
    ``price_usd_gpu_hr`` is populated ONLY for USD-listed prints; a non-USD
    price stays native with a null USD field — FX is a consumer's decision,
    never a capture-time one.

    The sku is derived HERE, from the catalog and the provider's own
    sku_identifier — collectors never claim one. ``sku_match`` records how:
    'catalog' (an entry matched) or 'unmapped' (recorded honestly as
    sku null; the identifier surfaces in the snapshot's
    unmapped_identifiers rollup).
    """
    currency = obs.get("currency", "USD")
    native = obs.get("price_native_per_gpu_hr")
    usd = obs.get("price_usd_gpu_hr")
    if native is None:
        native = usd
    if usd is None and currency == "USD":
        usd = native

    identifier = obs.get("sku_identifier")
    entry = match_sku(catalog, identifier)

    out: Dict[str, Any] = {
        "sku": entry["sku"] if entry else None,
        "sku_match": "catalog" if entry else "unmapped",
        "vendor": entry.get("vendor") if entry else None,
        "sku_identifier": identifier,
        "price_usd_gpu_hr": usd,
        "price_native_per_gpu_hr": native,
        "currency": currency,
        "raw_value": obs.get("raw_value"),
        "raw_unit": obs.get("raw_unit", "usd_per_gpu_hr"),
        "gpu_count_basis": obs.get("gpu_count_basis", 1),
        "tier": obs.get("tier", "on-demand"),
        "region": obs.get("region", "?"),
        "notes": obs.get("notes", ""),
    }
    # The plausibility band is USD-DENOMINATED (catalog key
    # plausible_usd_gpu_hr), so it screens USD prints only: comparing an
    # INR/BRL native figure against a USD band flagged every correct print
    # from those surfaces — standing noise that would bury a real
    # implausible USD print. Non-USD prints are recorded unscreened
    # (documented behavior); a missing/invalid price is always flagged.
    price_ok = isinstance(native, (int, float)) and not isinstance(native, bool)
    if not price_ok:
        out["implausible"] = True
    elif currency == "USD":
        lo, hi = plausible_band(catalog, out["sku"])
        out["implausible"] = not (lo <= float(native) <= hi)
    else:
        out["implausible"] = False
    for key in _PASSTHROUGH_KEYS:
        if key in obs:
            # Shallow-copy the schema-free dict so nothing downstream can
            # mutate a collector-held reference (or vice versa).
            out[key] = dict(obs[key]) if key == "extra" else obs[key]
    return out


def build_capture_snapshot(
    *,
    config: Dict[str, Any],
    catalog: Dict[str, Any],
    source_results: Sequence[Dict[str, Any]],
    captured_at: datetime,
    run_id: str,
    slot_date: date,
    slot_hour_utc: int,
    canonical: bool,
    capturer: Dict[str, Any],
    previous_day_empty: Optional[bool] = None,
    late_fill: bool = False,
) -> Dict[str, Any]:
    """Assemble the immutable capture document.

    ``source_results``: one entry per configured source (the runner
    guarantees this — unimplemented/failed sources appear with status
    'unimplemented'/'error' so a missing feed is a visible hole, never a
    silently shorter list).
    """
    by_id = sources_by_id(config)
    sources: List[Dict[str, Any]] = []
    for res in source_results:
        sid = res["source_id"]
        cfg_entry = by_id.get(sid, {})
        observations = [
            normalize_observation(o, catalog)
            for o in (res.get("observations") or [])
        ]
        # Who-is-speaking disclosure comes from the VALIDATED config, never
        # from the collector's self-description — a collector defaulting
        # first_party=True must not be able to relabel a reseller. The
        # collector's value is only used when the config entry is silent
        # (minimal test configs).
        first_party = cfg_entry.get("first_party")
        if first_party is None:
            first_party = res.get("first_party_observation")
        entry = {
            "source_id": sid,
            "display_name": cfg_entry.get("display_name", sid),
            "source_type": cfg_entry.get("source_type"),
            "status": res.get("status", "error"),
            "method": res.get("method"),
            "url": res.get("url"),
            "first_party_observation": first_party,
            "fetched_at": res.get("fetched_at"),
            "elapsed_seconds": res.get("elapsed_seconds"),
            "observations": observations,
            # A partially-failed source (one chip's fetch died) must be
            # distinguishable from "the feed genuinely had no offers".
            "partial_errors": res.get("partial_errors"),
            "error": res.get("error"),
        }
        if res.get("book_stats") is not None:
            entry["book_stats"] = res["book_stats"]
        sources.append(entry)

    all_obs = [o for s in sources for o in s["observations"]]
    skus_observed = sorted({o["sku"] for o in all_obs if o["sku"] is not None})
    unmapped = sorted(
        {
            str(o["sku_identifier"])
            for o in all_obs
            if o["sku_match"] == "unmapped" and o.get("sku_identifier")
        }
    )

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "lane_id": config["lane_id"],
        "sku_catalog_path": config.get("sku_catalog_path"),
        "captured_at": rfc3339(captured_at),
        "capture_date": slot_date.isoformat(),
        "slot_hour_utc": int(slot_hour_utc),
        "canonical_slot": bool(canonical),
        # canonical_slot marks WHICH slot this is; late_fill marks whether
        # it was recorded after the mark hour (up to one window late).
        # Readers enforce their own staleness rule on the pair +
        # captured_at.
        "late_fill": bool(late_fill),
        "run_id": run_id,
        "capturer": capturer,
        "sources": sources,
        "sources_ok": sorted(
            s["source_id"] for s in sources if s["status"] == "ok"
        ),
        "sources_failed": sorted(
            s["source_id"]
            for s in sources
            if s["status"] not in ("ok", "unimplemented")
        ),
        "sources_unimplemented": sorted(
            s["source_id"] for s in sources if s["status"] == "unimplemented"
        ),
        "observation_count": len(all_obs),
        "skus_observed": skus_observed,
        # The grow-the-catalog worklist: labels that printed but mapped to
        # no catalog entry. Deliberately unique-sorted strings, not counts —
        # this is a visibility rollup, never an input to anything.
        "unmapped_identifiers": unmapped,
        "previous_day_empty": previous_day_empty,
    }
