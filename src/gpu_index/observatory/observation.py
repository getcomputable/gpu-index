# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Shared helpers for observatory collectors — the collector contract.

Contract per collector module (gpu_index/observatory/sources/<source_id>.py):

  - defines SOURCE_ID (== its module name) and
    collect(timeout=..., options=None) -> the result dict built by
    ``result`` below;
  - records EVERY GPU rental price row its surface publishes that the
    recipe can extract with an identity pin — all chips, all vendors, all
    tiers (labeled) — never just one target sku. Rows the recipe cannot
    pin safely are SKIPPED (and, where countable, noted in partial_errors)
    rather than guessed;
  - every observation carries the RAW value as published plus our per-GPU
    normalization and an explicit currency — a non-USD list price is
    recorded natively (price_usd_gpu_hr stays None), never mislabeled;
  - sets sku_identifier to the provider's own structured label; the
    FRAMEWORK derives the canonical sku from the catalog — collectors
    never claim a sku themselves, so one normalization rule governs every
    source and can be re-derived from raw later;
  - raises on failure INCLUDING parsed-nothing: a page that silently
    changed shape must surface as an error in the snapshot, never as a
    healthy source with zero prints;
  - TLS verification is never weakened, and collectors never aggregate
    across sources or write anywhere.

Transport comes from gpu_index.common.http (fetch) — the hardened stack (https-only
redirects, body caps, certifi bundle) is shared lane machinery, not basket
policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gpu_index.common.slots import rfc3339

DEFAULT_TIMEOUT = 30.0


def observation(
    *,
    sku_identifier: str,
    price_per_gpu_hr: float,
    currency: str = "USD",
    raw_value: str,
    raw_unit: str = "usd_per_gpu_hr",
    gpu_count_basis: int = 1,
    tier: str = "on-demand",
    region: str = "?",
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One raw print. ``extra`` is the sanctioned home for source-native
    metadata (availability flags, verification levels, machine identity…) —
    schema-free by design so adding a source never forces a schema bump."""
    price = round(float(price_per_gpu_hr), 4)
    out: Dict[str, Any] = {
        "sku_identifier": str(sku_identifier),
        "price_usd_gpu_hr": price if currency == "USD" else None,
        "price_native_per_gpu_hr": price,
        "currency": currency,
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "gpu_count_basis": gpu_count_basis,
        "tier": tier,
        "region": region,
        "notes": notes,
    }
    if extra:
        out["extra"] = extra
    return out


def result(
    source_id: str,
    *,
    method: str,
    url: str,
    observations: List[Dict[str, Any]],
    first_party: bool = True,
    partial_errors: Optional[List[str]] = None,
    book_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not observations:
        raise RuntimeError(
            f"{source_id}: parsed zero GPU price observations — page/API "
            "shape changed or listings pulled; refusing to record an empty "
            "print"
        )
    out: Dict[str, Any] = {
        "source_id": source_id,
        "method": method,
        "url": url,
        "first_party_observation": first_party,
        "fetched_at": rfc3339(),
        "observations": observations,
    }
    if partial_errors:
        out["partial_errors"] = partial_errors
    if book_stats:
        out["book_stats"] = book_stats
    return out
