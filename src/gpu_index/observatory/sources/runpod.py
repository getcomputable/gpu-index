# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""RunPod -- GraphQL gpuTypes, every listed GPU, all four price surfaces.

THE EXEMPLAR observatory collector (house style for the rest of the
package): parse is a pure function over the response body, collect is the
thin fetch wrapper, and the module records EVERYTHING its surface publishes
with tiers labeled -- the basket lane's runpod collector deliberately
requests securePrice only (the basket excludes Community Cloud); this one requests
all four price fields precisely because the observatory's job is the whole
surface. Separate module, separate request -- the basket recipe is untouched.

Fields verified live 2026-08-22: gpuTypes carries per-GPU-hour USD prices
{securePrice, communityPrice, secureSpotPrice, communitySpotPrice}; a zero/
null price means that surface doesn't offer the chip (skipped silently by
rule, not a $0 print). A price value of any OTHER shape -- string, bool,
NaN/inf, negative -- is a schema anomaly: skipped and counted in
partial_errors, never conflated with "not offered" and never recorded. The
``id`` string (e.g. 'NVIDIA B300 SXM6 AC') is the fuller structured label
and becomes sku_identifier; displayName rides in notes.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "runpod"

URL = "https://api.runpod.io/graphql"
_QUERY = {
    "query": (
        "{gpuTypes{id displayName memoryInGb "
        "securePrice communityPrice secureSpotPrice communitySpotPrice}}"
    )
}

# (field, tier, cloud) -- cloud is runpod-native vocabulary and lives in
# extra; tier stays in the lane-wide vocabulary.
_PRICE_SURFACES = (
    ("securePrice", "on-demand", "secure"),
    ("communityPrice", "on-demand", "community"),
    ("secureSpotPrice", "spot", "secure"),
    ("communitySpotPrice", "spot", "community"),
)


def parse_runpod(body: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the GraphQL body -> (observations, partial_errors).

    Fail-closed: null/zero prices are the API's documented "not offered on
    that surface" and skip silently; a price of any other unusable shape is
    counted in partial_errors so a partial schema change (one surface's
    field re-typed) surfaces in the snapshot instead of silently thinning
    the record. A whole-body shape change parses zero rows and the result
    builder raises.
    """
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for g in json.loads(body)["data"]["gpuTypes"]:
        identifier = str(g.get("id") or "").strip()
        display = str(g.get("displayName") or identifier)
        if not identifier:
            # A row we cannot attribute to a structured label is a row we
            # cannot honestly record.
            errors.append(
                f"row without id ({display[:60]!r}): unattributable, skipped"
            )
            continue
        mem = g.get("memoryInGb")
        mem_ok = isinstance(mem, (int, float)) and not isinstance(mem, bool)
        for field, tier, cloud in _PRICE_SURFACES:
            price = g.get(field)
            if price is None:
                continue  # null = not offered on that surface
            if (
                not isinstance(price, (int, float))
                or isinstance(price, bool)
                or not math.isfinite(price)
                or price < 0
            ):
                errors.append(
                    f"{identifier[:60]}: {field} is not a usable price "
                    f"({str(price)[:40]!r}) -- skipped"
                )
                continue
            if price == 0:
                continue  # zero = not offered on that surface
            obs = observation(
                sku_identifier=identifier,
                price_per_gpu_hr=float(price),
                raw_value=str(price),
                tier=tier,
                region="global",
                notes=f"{display} {cloud} cloud"
                + (f" {mem:g}GB" if mem_ok else ""),
                extra={"cloud": cloud, "display_name": display},
            )
            if mem_ok:
                obs["memory_gb_label"] = mem
            rows.append(obs)
    return rows, errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(
        URL,
        data=json.dumps(_QUERY).encode(),
        headers={"content-type": "application/json"},
        timeout=timeout,
    )
    observations, partial_errors = parse_runpod(body)
    return result(
        SOURCE_ID,
        method="graphql",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
