# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Runcrate -- public GPU pricing JSON feed, on-demand price per GPU-hour.

Price surface: GET https://www.runcrate.ai/api/gpu-pricing. The route answers
anonymously and is the same feed behind Runcrate's own pricing page. Verified
live 2026-11-08.

The body is ``{"gpus": [...], "fetched_at": ...}``. Each row is one GPU type
carrying ``min_price_hr`` (the live on-demand price per GPU-hour, USD),
``max_price_hr`` / ``avg_price_hr`` (spread across the live offers),
``available_options`` (how many offers are live) and a ``name`` label. The
price is already per GPU, so gpu_count_basis stays 1.

Identity pin: an explicit allowlist of the feed's own ``gpu_type`` ids whose
``name`` the catalog resolves unambiguously. The feed does not say SXM versus
PCIe, so H100 is NOT recorded (the catalog splits H100 from H100_PCIE and
guessing either would misprint the other). V100 is not recorded either (the
feed lists two memory sizes under separate ids and the catalog merges them).
H200 NVL is its own feed id and normalizes to H200_NVL; it never lands on
H200. A100 rows keep the memory size in the label ('A100 80GB', 'A100 40GB'),
which is how the catalog already treats every provider. Types outside the
allowlist are skipped silently by rule, not counted.

Edge shapes, all fail-closed:

  - ``available_options`` below 1 means no live offer, and the feed can keep
    a stale price for such a type: skipped and counted in partial_errors,
    never recorded;
  - a missing, retyped or non-positive ``min_price_hr`` (string, bool,
    NaN/inf, zero, negative) or a retyped ``available_options`` is a schema
    anomaly: skipped and counted in partial_errors;
  - a top-level shape change raises; a feed that yields zero rows raises in
    the result builder.

Currency: the feed publishes USD per GPU-hour; USD is pinned by the field
name (``*_price_hr``) and the pricing page, not inferred from a symbol.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "runcrate"

URL = "https://www.runcrate.ai/api/gpu-pricing"

# Feed gpu_type ids this recipe will record. Everything else is skipped.
_PINNED_TYPES = frozenset(
    {
        "B300",
        "B200",
        "H200",
        "H200_nvl",
        "A100_80G",
        "A100",
        "L40S",
        "L40",
        "RTX5090",
        "RTX4090",
        "RTX6000Ada",
        "RTXPro6000",
        "A6000",
        "A40",
        "RTX3090",
        "L4",
        "A10",
    }
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_runcrate(body: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the feed body -> (observations, partial_errors)."""
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("gpus"), list):
        raise RuntimeError(
            "runcrate: top level is not an object with a 'gpus' list -- feed "
            "reshaped; refusing to guess at the new envelope"
        )
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for idx, g in enumerate(payload["gpus"]):
        if not isinstance(g, dict):
            errors.append(f"row {idx} is not an object -- skipped")
            continue
        type_id = g.get("gpu_type")
        if not isinstance(type_id, str) or not type_id.strip():
            errors.append(f"row {idx} has no gpu_type id -- unattributable, skipped")
            continue
        if type_id not in _PINNED_TYPES:
            continue  # outside the identity pin: skipped silently by rule
        label = g.get("name")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"{type_id}: name is not a non-empty string -- skipped")
            continue
        label = label.strip()
        price = g.get("min_price_hr")
        if not _is_number(price) or not math.isfinite(price) or price <= 0:
            errors.append(
                f"{type_id}: min_price_hr is not a usable price "
                f"({str(price)[:40]!r}) -- skipped"
            )
            continue
        options = g.get("available_options")
        if (
            not _is_number(options)
            or not math.isfinite(options)
            or options != int(options)
        ):
            errors.append(
                f"{type_id}: available_options retyped ({str(options)[:40]!r}) "
                "-- skipped"
            )
            continue
        if options < 1:
            errors.append(
                f"{type_id}: no live offers (available_options={int(options)}) "
                "-- listed price may be stale, skipped"
            )
            continue
        vram = g.get("vram")
        obs = observation(
            sku_identifier=label,
            price_per_gpu_hr=float(price),
            raw_value=str(price),
            tier="on-demand",
            region="global",
            notes=f"{label} on-demand"
            + (f" {vram}" if isinstance(vram, str) and vram.strip() else ""),
            extra={
                "gpu_type_id": type_id,
                "available_options": int(options),
            },
        )
        rows.append(obs)
    return rows, errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, headers={"accept": "application/json"}, timeout=timeout)
    observations, partial_errors = parse_runcrate(body)
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
