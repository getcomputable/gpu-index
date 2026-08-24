# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Voltage Park -- unauthenticated dashboard locations API, both network tiers.

Surface: GET https://cloud-api.voltagepark.com/api/v1/bare-metal/locations
(the SPA dashboard's logged-out variant; no auth, no cookies). Verified live
2026-08-22: results[] holds one location row keyed by specs_per_node.gpu_model
('h100-sxm5-80gb', 8x per node) with TWO published per-GPU hourly prices --
gpu_price_ethernet ("1.99") and gpu_price_infiniband ("2.49") -- as decimal
strings. Both are networking variants of the same ON-DEMAND product (ethernet
vs 3200Gbps InfiniBand), NOT an on-demand/reserved split (long-term reserve is
contact-only and never published), so tier stays "on-demand" for both and the
voltagepark-native ``network`` label lives in extra -- same vocabulary rule as
the runpod exemplar's secure/community clouds.

Basis and currency facts (why the fields mean what we record):

  - prices are per GPU per HOUR: the dashboard app multiplies by gpuCount to
    build a total (per-GPU proven), and og:description prints "$1.99/GPU/hour"
    (per-hour + USD proven). NEVER multiply by specs_per_node.gpu_count -- the
    raw figure is already the per-GPU rate (gpu_count_basis=1).
  - the payload carries no currency field; USD is pinned from the provider's
    own dollar prints (pricing FAQ "starting at $1.99", og:description above)
    which exactly match the ethernet figure. The InfiniBand figure has NO
    static human-readable confirmation -- the marketing card says "Contact for
    pricing" -- but it renders through the same dashboard USD price path (zod
    schema + deploy-flow minPrice), noted per-observation in extra.
  - prices arrive as JSON strings today but the app schema accepts
    number-or-string, so parsing goes through Decimal(str(x)) and tolerates a
    type flip without loosening the >0 pin.

Fail-closed identity pins (every one earned by a near-miss):

  - top level must carry a 'results' list and 'total_result_count' int, the
    page must be complete (has_next falsy AND total == len(results)) -- one
    page today (total_result_count=1); if pagination ever activates this
    collector raises rather than silently under-reporting;
  - per row, specs_per_node.gpu_model must be a non-empty string and BOTH
    price fields must Decimal-parse > 0; any missing/renamed field raises with
    zero rows recorded. gpu_count_ethernet/gpu_count_infiniband are numeric
    LOOKALIKES for the price fields (and sat at 0 while prices stayed live at
    probe time), so availability never gates a price and the pin is on the
    exact price field names;
  - any UNKNOWN row key containing 'gpu_price' raises: a new price column
    (a spot tier, a reserved price going public) is a published price this
    recipe never saw -- same refuse-to-under-report rule as pagination;
  - churn risk is real (Lightning AI merger announced, TensorDock acquired,
    numeric pricing already pulled from the marketing site) -- these pins are
    the tripwire.

Do NOT scrape the marketing /pricing page (a decoy: cards say "Contact for
pricing"; $1.99 lives only in meta tags/FAQ prose), and do NOT use
/api/v1/bare-metal/detailed-locations or /api/v1/billing/hourly-rate (both
account-scoped).
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "voltagepark"

URL = "https://cloud-api.voltagepark.com/api/v1/bare-metal/locations"

# (price field, count field, network) -- network is voltagepark-native
# vocabulary and lives in extra; tier stays in the lane-wide vocabulary
# ("on-demand" for both -- see module docstring).
_NETWORK_SURFACES = (
    ("gpu_price_ethernet", "gpu_count_ethernet", "ethernet"),
    ("gpu_price_infiniband", "gpu_count_infiniband", "infiniband"),
)

_KNOWN_PRICE_FIELDS = frozenset(field for field, _, _ in _NETWORK_SURFACES)


def _pin_price(row_label: str, field: str, value: Any) -> Decimal:
    """The exact-name price pin: missing/renamed/unparseable/non-positive
    raises (zero rows) -- a silent skip would drop a whole tier while the
    source looked healthy, and the count fields are numeric lookalikes."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RuntimeError(
            f"voltagepark: {row_label}: {field} is {value!r} -- price field "
            "missing or reshaped (count fields are numeric lookalikes; the "
            "pin is on the exact price field name); refusing to guess"
        )
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise RuntimeError(
            f"voltagepark: {row_label}: {field} {value!r} does not "
            "Decimal-parse -- price field reshaped; refusing to guess"
        ) from exc
    if not price.is_finite() or price <= 0:
        raise RuntimeError(
            f"voltagepark: {row_label}: {field} {value!r} is not a positive "
            "price -- not a real print; refusing to record"
        )
    return price


def parse_voltagepark(body: str) -> List[Dict[str, Any]]:
    payload = json.loads(body)
    results = payload.get("results") if isinstance(payload, dict) else None
    total = payload.get("total_result_count") if isinstance(payload, dict) else None
    if (
        not isinstance(results, list)
        or isinstance(total, bool)
        or not isinstance(total, int)
    ):
        raise RuntimeError(
            "voltagepark: top level reshaped -- expected a 'results' list and "
            "a 'total_result_count' int; refusing to guess at the new shape"
        )
    if payload.get("has_next"):
        raise RuntimeError(
            "voltagepark: has_next is truthy -- the single-page assumption "
            "broke; add pagination rather than silently under-reporting"
        )
    if total != len(results):
        raise RuntimeError(
            f"voltagepark: total_result_count {total} != {len(results)} rows "
            "in the page -- rows exist this fetch never saw; refusing to "
            "under-report"
        )
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"voltagepark: results[{idx}] is not an object -- row shape "
                "changed; refusing to guess"
            )
        row_label = f"row {row.get('id') or idx}"
        unknown_price_keys = sorted(
            key
            for key in row
            if "gpu_price" in key and key not in _KNOWN_PRICE_FIELDS
        )
        if unknown_price_keys:
            raise RuntimeError(
                f"voltagepark: {row_label} carries unknown price field(s) "
                f"{unknown_price_keys} -- a published price this recipe "
                "never saw (new tier?); refusing to under-report"
            )
        specs = row.get("specs_per_node")
        if not isinstance(specs, dict):
            raise RuntimeError(
                f"voltagepark: {row_label} lost specs_per_node -- identity "
                "anchor gone; refusing to guess"
            )
        gpu_model = specs.get("gpu_model")
        if not isinstance(gpu_model, str) or not gpu_model.strip():
            raise RuntimeError(
                f"voltagepark: {row_label} has no specs_per_node.gpu_model "
                "string -- no structured label to pin identity to; refusing "
                "to guess"
            )
        gpu_model = gpu_model.strip()
        node_gpus = specs.get("gpu_count")
        node_note = (
            f" ({node_gpus}x per node)"
            if isinstance(node_gpus, int) and not isinstance(node_gpus, bool)
            else ""
        )
        for price_field, count_field, network in _NETWORK_SURFACES:
            price = _pin_price(row_label, price_field, row.get(price_field))
            rows.append(
                observation(
                    sku_identifier=gpu_model,
                    price_per_gpu_hr=float(price),
                    # raw figure exactly as published (a decimal string
                    # today; str() keeps a number flip byte-honest too).
                    raw_value=str(row.get(price_field)),
                    gpu_count_basis=1,
                    tier="on-demand",
                    # Location rows carry only an opaque UUID -- no honest
                    # human region name to claim; the id rides in extra.
                    region="?",
                    notes=f"{network} networking{node_note}",
                    extra={
                        "network": network,
                        "location_id": row.get("id"),
                        # Availability metadata, NEVER a price gate -- both
                        # counts were 0 at probe time while prices stayed
                        # published.
                        "available_gpu_count": row.get(count_field),
                        "node_gpu_count": node_gpus,
                        "cpu_model": specs.get("cpu_model"),
                        "ram_gb": specs.get("ram_gb"),
                        "storage_gb": specs.get("storage_gb"),
                        # Currency provenance (no currency field in payload):
                        # ethernet figure matches the provider's own "$1.99/
                        # GPU/hour" prints; infiniband is dashboard/API-only.
                        "usd_basis": (
                            "provider dollar prints"
                            if network == "ethernet"
                            else "dashboard price path only (marketing says "
                            "contact for pricing)"
                        ),
                    },
                )
            )
    return rows


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(
        URL, headers={"accept": "application/json"}, timeout=timeout
    )
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=parse_voltagepark(body),
    )
