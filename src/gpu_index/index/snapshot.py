# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Build the capture snapshot document — per-source prints, no composite.

The per-source time series IS the primary artifact. This
document stores every source's own prints (raw value as published + our
per-GPU normalization + audit fields) and derives nothing beyond per-source
basis pairs. No basket/index value appears anywhere in the payload — the
composite is phase 2, computed by replaying these snapshots.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

from gpu_index.index.config import sources_by_id
from gpu_index.common.slots import rfc3339

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "index_basket_capture"

# Plausibility band for single-GPU on-demand $/GPU-hr; out-of-band prints
# are still STORED (raw is raw) but flagged.
PLAUSIBLE_USD_GPU_HR = (0.5, 40.0)

# GB200-lookalike screen (METHODOLOGY.md §5, product identity) — the
# highest-consequence screen in the B200
# lane: Grace-coupled/rack-scale parts are marketed by real providers under
# the string "B200" (Oracle: "BM.GPU.GB200.4 (4x Nvidia B200 189GB NVL72)";
# Crusoe: "NVIDIA GB200 (186GB NVL72)") at 2-4x the underlying. A B200
# print carrying any of these signals is QUARANTINED pending manual
# confirmation, never admitted. Tokens are matched against the STRUCTURED
# sku_identifier field only — free-prose notes legitimately contain words
# like "NVLink". GPU counts 36/72 are rack-scale domains; count 4 alone is
# NOT screened (4-GPU marketplace slices of HGX boxes are legitimate).
GB200_LOOKALIKE_TOKENS = ("NVL", "GB200", "GB300", "GRACE", "B200A")
GB200_MEMORY_LABELS_GB = (186, 189)
GB200_GPU_COUNTS = (36, 72)


def gb200_lookalike_reason(obs: Dict[str, Any]) -> Optional[str]:
    if obs.get("sku") != "B200":
        return None
    identifier = str(obs.get("sku_identifier") or "").upper()
    for token in GB200_LOOKALIKE_TOKENS:
        if token in identifier:
            return f"token {token} in sku_identifier {identifier!r}"
    memory = obs.get("memory_gb_label")
    if (
        isinstance(memory, (int, float))
        and not isinstance(memory, bool)
        and int(memory) in GB200_MEMORY_LABELS_GB
    ):
        return f"memory label {int(memory)}GB is the GB200 packaging convention"
    if obs.get("gpu_count_basis") in GB200_GPU_COUNTS:
        return f"gpu count {obs['gpu_count_basis']} is a rack-scale NVLink domain"
    return None


def normalize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Fill defaults + plausibility flag; never drops or rewrites raw fields.

    Currency honesty: ``price_usd_gpu_hr`` is populated ONLY for USD-listed
    prints. A non-USD list price (Scaleway bills EUR) stays in
    ``price_native_per_gpu_hr`` + ``currency`` — FX conversion is a
    calc-time decision, never a collector's.
    """
    currency = obs.get("currency", "USD")
    native = obs.get("price_native_per_gpu_hr")
    usd = obs.get("price_usd_gpu_hr")
    if native is None:
        native = usd
    if usd is None and currency == "USD":
        usd = native
    out = {
        "sku": obs.get("sku"),
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
    lo, hi = PLAUSIBLE_USD_GPU_HR
    out["implausible"] = not (
        isinstance(native, (int, float)) and lo <= float(native) <= hi
    )
    # L2 identity continuity: marketplace collectors attach
    # source-native identity so tomorrow's print is attributable to the
    # same machine — the same-machine-vs-book delta in the capture screens
    # needs these to survive normalization. The provider SKU identifier +
    # stated memory label are audit fields —
    # what makes a GB200 substitution detectable after the fact.
    for key in (
        "offer_id",
        "machine_id",
        "host_id",
        "verification",
        "sku_identifier",
        "memory_gb_label",
        # Marks vast population rows recorded beyond the legacy
        # cheapest-5 (the statistic's full verified-US/CA book).
        "book_scope",
    ):
        if key in obs:
            out[key] = obs[key]
    gb200_reason = gb200_lookalike_reason(out)
    if gb200_reason is not None:
        out["implausible"] = True
        out["quarantined"] = "gb200_lookalike"
        out["notes"] = (
            f"{out['notes']} [GB200 QUARANTINE (product-identity screen): "
            f"{gb200_reason} — held pending manual confirmation]"
        ).strip()
    return out


def _od_median(observations: Sequence[Dict[str, Any]], sku: str) -> Optional[float]:
    vals = [
        float(o["price_usd_gpu_hr"])
        for o in observations
        if o.get("sku") == sku
        and o.get("tier") == "on-demand"
        and not o.get("implausible")
        and o.get("price_usd_gpu_hr") is not None
    ]
    return round(float(statistics.median(vals)), 4) if vals else None


def derive_basis_pairs(sources: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-source B300:B200 pairs where a source printed both chips this run.

    Derived-but-per-source: strengthens the basis-ratio sample
    without computing anything cross-source. Raw observations stay untouched
    in the source entries.
    """
    pairs: List[Dict[str, Any]] = []
    for src in sources:
        if src.get("status") != "ok":
            continue
        obs = src.get("observations") or []
        b200 = _od_median(obs, "B200")
        b300 = _od_median(obs, "B300")
        if b200 and b300 and b200 > 0:
            pairs.append(
                {
                    "source_id": src["source_id"],
                    "b200_usd_gpu_hr": b200,
                    "b300_usd_gpu_hr": b300,
                    "ratio_b300_b200": round(b300 / b200, 6),
                }
            )
    return pairs


def build_capture_snapshot(
    *,
    config: Dict[str, Any],
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
    for result in source_results:
        sid = result["source_id"]
        cfg_entry = by_id.get(sid, {})
        observations = [
            normalize_observation(o) for o in (result.get("observations") or [])
        ]
        entry = {
            "source_id": sid,
            "display_name": cfg_entry.get("display_name", sid),
            "role": cfg_entry.get("role"),
            "weight": cfg_entry.get("weight"),
            "source_type": cfg_entry.get("source_type"),
            "status": result.get("status", "error"),
            "method": result.get("method"),
            "url": result.get("url"),
            "first_party_observation": result.get("first_party_observation"),
            "fetched_at": result.get("fetched_at"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "observations": observations,
            # A partially-failed source (vast: one sku's fetch died) must
            # be distinguishable from "the feed genuinely had no offers".
            "partial_errors": result.get("partial_errors"),
            "error": result.get("error"),
        }
        if result.get("book_stats") is not None:
            # vast: per-sku book accounting so a truncated recording can
            # never again be invisible.
            entry["book_stats"] = result["book_stats"]
        sources.append(entry)

    ok = sorted(s["source_id"] for s in sources if s["status"] == "ok")
    failed = sorted(
        s["source_id"] for s in sources if s["status"] not in ("ok", "unimplemented")
    )
    unimplemented = sorted(
        s["source_id"] for s in sources if s["status"] == "unimplemented"
    )
    # First-class coverage read: which of the basket CONSTITUENTS printed,
    # separate from pool sources — the number the claim threshold and any
    # future calc report against.
    basket_role = config.get("basket_role", "b300_basket")
    # Coverage counts CONSTITUENT PRINTS, not collector liveness: a dual-sku
    # collector that came back ok with zero target-sku observations (e.g. a
    # priceless row skipped) must not claim basket coverage. Configs without
    # a target_sku (older tests) keep the status-only semantics.
    target_sku = config.get("target_sku")
    basket_ok = sorted(
        s["source_id"]
        for s in sources
        if s.get("role") == basket_role
        and s["status"] == "ok"
        and (
            target_sku is None
            or any(o.get("sku") == target_sku for o in s["observations"])
        )
    )

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "basket_id": config["basket_id"],
        "methodology_doc": config.get("methodology_doc"),
        "captured_at": rfc3339(captured_at),
        "capture_date": slot_date.isoformat(),
        "slot_hour_utc": int(slot_hour_utc),
        "canonical_slot": bool(canonical),
        # canonical_slot marks WHICH slot this is; late_fill marks whether it
        # was recorded after the mark hour (up to one window late). Readers
        # enforce their own staleness rule on the pair + captured_at.
        "late_fill": bool(late_fill),
        "run_id": run_id,
        "capturer": capturer,
        "sources": sources,
        "basis_pairs": derive_basis_pairs(sources),
        "sources_ok": ok,
        "sources_failed": failed,
        "sources_unimplemented": unimplemented,
        "basket_sources_ok": basket_ok,
        "previous_day_empty": previous_day_empty,
    }
