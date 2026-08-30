# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Hyperbolic -- public on-demand rental-options API.

Price surface: GET
https://api.hyperbolic.ai/v2/alpha/on-demand/rental-options. The route
answers anonymously, and Hyperbolic's logged-out GPU page serves the same
payload. Anonymous access re-verified live 2026-08-29.

The response is a bare list whose membership churns with inventory. Each
``costPerHourCents`` value is the WHOLE-option hourly total, so the per-GPU
USD price is cents / ``gpuCount`` / 100. A missing ``totalAvailable`` means
unknown, never zero, and availability never gates a published price.

Fail closed on a reshaped envelope, invalid price/count/identity fields, or
any new ``*PerHourCents`` key. The API also exposes Hyperbolic's supplier
cost, which is deliberately never copied into the published observatory
record; see the omission beside observation construction.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "hyperbolic"

URL = "https://api.hyperbolic.ai/v2/alpha/on-demand/rental-options"

_KNOWN_PRICE_FIELDS = frozenset(
    {"costPerHourCents", "providerCostPerHourCents"}
)


def _pin_positive_price(row_label: str, value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RuntimeError(
            f"hyperbolic: {row_label}: costPerHourCents has invalid type "
            f"{type(value).__name__} -- "
            "whole-option price missing or reshaped; refusing to guess"
        )
    try:
        cents = Decimal(str(value))
    except InvalidOperation as exc:
        raise RuntimeError(
            f"hyperbolic: {row_label}: costPerHourCents does not "
            "Decimal-parse; refusing to guess"
        ) from exc
    if not cents.is_finite() or cents <= 0:
        raise RuntimeError(
            f"hyperbolic: {row_label}: costPerHourCents is not a "
            "positive finite price; refusing to record"
        )
    return cents


def _pin_text(row_label: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"hyperbolic: {row_label}: {field} is not a non-empty string "
            f"(got {type(value).__name__}); refusing to guess"
        )
    return value.strip()


def parse_hyperbolic(body: str) -> List[Dict[str, Any]]:
    payload = json.loads(body)
    if not isinstance(payload, list):
        raise RuntimeError(
            "hyperbolic: top level is not a list -- rental-options response "
            "reshaped; refusing to guess at the new envelope"
        )

    observations: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"hyperbolic: row {idx} is not an object -- option shape "
                "changed; refusing to guess"
            )
        row_label = f"row {idx}"

        unknown_price_fields = sorted(
            key
            for key in row
            if key.endswith("PerHourCents") and key not in _KNOWN_PRICE_FIELDS
        )
        if unknown_price_fields:
            raise RuntimeError(
                f"hyperbolic: {row_label} carries "
                f"{len(unknown_price_fields)} unknown price field(s) "
                "matching *PerHourCents -- a published price this recipe "
                "never saw (new tier?); refusing to under-report"
            )

        raw_cost = row.get("costPerHourCents")
        cents = _pin_positive_price(row_label, raw_cost)

        gpu_count = row.get("gpuCount")
        if (
            isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or gpu_count < 1
        ):
            raise RuntimeError(
                f"hyperbolic: {row_label}: gpuCount must be int >= 1 for "
                "whole-option normalization (got "
                f"{type(gpu_count).__name__}); refusing to default to 1"
            )

        gpu_type = _pin_text(row_label, "gpuType", row.get("gpuType"))
        form_factor = _pin_text(
            row_label, "gpuFormFactor", row.get("gpuFormFactor")
        )
        region = _pin_text(row_label, "region", row.get("region"))
        connection_type = _pin_text(
            row_label, "connectionType", row.get("connectionType")
        )
        machine_type = _pin_text(
            row_label, "machineType", row.get("machineType")
        )

        ethernet_variant = row.get("ethernetVariant")
        if ethernet_variant is not None:
            ethernet_variant = _pin_text(
                row_label, "ethernetVariant", ethernet_variant
            )

        raw_available_gpu_count = row.get("totalAvailable")
        available_gpu_count = (
            raw_available_gpu_count
            if not isinstance(raw_available_gpu_count, bool)
            and isinstance(raw_available_gpu_count, int)
            and raw_available_gpu_count >= 0
            else None
        )

        enabled = row.get("enabled")
        if not isinstance(enabled, bool):
            raise RuntimeError(
                f"hyperbolic: {row_label}: enabled must be bool (got "
                f"{type(enabled).__name__}); refusing to guess option state"
            )

        price_per_gpu_hr = float(
            cents / Decimal(gpu_count) / Decimal(100)
        )
        published_price_per_gpu_hr = round(price_per_gpu_hr, 4)
        if (
            not math.isfinite(published_price_per_gpu_hr)
            or published_price_per_gpu_hr <= 0
        ):
            raise RuntimeError(
                f"hyperbolic: {row_label}: costPerHourCents falls outside "
                "the published positive finite per-GPU price range; "
                "refusing to record"
            )

        observations.append(
            observation(
                sku_identifier=f"{gpu_type.upper()} {form_factor.upper()}",
                price_per_gpu_hr=published_price_per_gpu_hr,
                raw_value=str(raw_cost),
                raw_unit="cents_per_instance_hr",
                gpu_count_basis=gpu_count,
                tier="on-demand",
                region=region,
                extra={
                    "enabled": enabled,
                    "gpu_form_factor": form_factor,
                    "connection_type": connection_type,
                    "ethernet_variant": ethernet_variant,
                    "machine_type": machine_type,
                    # Supplier-cost fields are excluded from the published
                    # record by policy: omit providerCostPerHourCents and
                    # never copy nested raw objects into observations or
                    # book_stats. Only the customer-facing total belongs
                    # in this public record.
                    # Missing or malformed availability degrades to unknown
                    # here because availability must never gate a price.
                    "available_gpu_count": available_gpu_count,
                },
            )
        )
    return observations


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, headers={"accept": "application/json"}, timeout=timeout)
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=parse_hyperbolic(body),
    )
