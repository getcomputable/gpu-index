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

Availability upgrade (availability accrual, verified live 2026-08-25): the SAME
single POST now also asks for chip-level stock probes and the per-
datacenter availability matrix -- GraphQL lets every new field ride in the
one request the collector already sends, so posture and load are unchanged.
The stock subtrees are FAIL-OPEN by ruling: a missing/misshapen subtree
appends to partial_errors and the price record still lands (price surfaces
remain the fail-closed core). New per-observation extra fields:

  - stock_status / stock_status_8x / stock_status_secure -- stockStatus
    from three aliased lowestPrice probes in the same request: unfiltered
    gpuCount:1, gpuCount:8 (can-you-actually-get-an-8x-box; verified live
    that 8x nulls out when no 8-GPU box is rentable), and gpuCount:1 +
    secureCloud:true. Values are OPAQUE provider strings recorded verbatim
    (observed vocabulary "Low"/"Medium"/null; "High" documented) -- no enum
    validation, so a new tier records instead of tripping. null is
    overloaded upstream: it appears both for chips with no public quote at
    all (MI300X live 2026-08-25) and for in-catalog chips at unattainable
    sizes (B200 at gpuCount:8), so null means "no public quote at this
    size/filter", NEVER "delisted". secureCloud:false is deliberately NOT
    probed: live it returns nulls even when community stock exists (input
    semantics broken/ignored on the false branch), so per-community stock
    cannot be derived.
  - max_gpu_count / offered_secure / offered_community -- the GpuType
    scalars maxGpuCount / secureCloud / communityCloud, verbatim; a retyped
    value records as None and is counted, never guessed.
  - dc_availability -- {dc_id: {available, stock_status, listed}} folded
    from dataCenters.gpuAvailability, region resolution the price rows
    (region 'global') lack. Join key is EXACT string equality gpuTypeId ==
    gpuTypes.id (verified live: 'NVIDIA B300 SXM6 AC' appears verbatim on
    both sides). The matrix is SPARSE -- each DC lists only chips it
    physically carries, so a chip absent from every DC records None (no
    signal, not zero stock); unlisted (listed:false) DCs appear in the live
    response and keep their flag. The matrix rides in extra rather than
    minting per-DC rows because gpuAvailability carries no price and
    observation() is price-centric.
  - dc_available_count -- scalar sum of available==true across LISTED DCs,
    so downstream reads a number without unpacking the map; None when the
    chip has no matrix row (or the whole block failed), which must never be
    conflated with 0.

graphql-spec.runpod.io claims Bearer auth is required for every query; live
probes contradict it (everything above answers unauthenticated), so RunPod
could legitimately close the gap any day -- by design that degrades to
partial_errors on the stock subtrees while the price core keeps printing.
The schema's numeric inventory fields (rentedCount, totalCount,
availableGpuCounts, ...) are null on every unauthenticated combination
probed; keyed collection would be a maintainer decision and is not built.
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
        "securePrice communityPrice secureSpotPrice communitySpotPrice "
        "secureCloud communityCloud maxGpuCount "
        "lowestPrice(input:{gpuCount:1})"
        "{stockStatus minimumBidPrice uninterruptablePrice} "
        "lp8: lowestPrice(input:{gpuCount:8}){stockStatus} "
        "lp1s: lowestPrice(input:{gpuCount:1,secureCloud:true}){stockStatus}} "
        "dataCenters{id name listed "
        "gpuAvailability{gpuTypeId available stockStatus}}}"
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

# (response alias, extra key) -- the three lowestPrice stock probes. GraphQL
# always echoes a requested field (null when it has no answer), so an ABSENT
# alias is a reshape and is counted; a null one is the documented "no public
# quote at this size/filter" and records None silently.
_STOCK_PROBES = (
    ("lowestPrice", "stock_status"),
    ("lp8", "stock_status_8x"),
    ("lp1s", "stock_status_secure"),
)


def _fold_datacenters(
    payload: Dict[str, Any], errors: List[str]
) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """dataCenters -> {gpuTypeId: {dc_id: {available, stock_status, listed}}}.

    Fail-open by ruling: a missing/misshapen block returns None with ONE
    loud partial_error -- the price rows must never dark on a stock-subtree
    reshape. Malformed individual entries are counted and skipped so one
    bad DC cannot erase the rest of the matrix.
    """
    dcs = payload.get("dataCenters")
    if not isinstance(dcs, list):
        errors.append(
            "dataCenters block missing/misshapen "
            f"({type(dcs).__name__}) -- dc availability unrecorded this "
            "capture; price rows unaffected"
        )
        return None
    dc_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dc in dcs:
        dc_id = str(dc.get("id") or "").strip() if isinstance(dc, dict) else ""
        if not dc_id:
            errors.append("dataCenters entry without id -- skipped")
            continue
        avail = dc.get("gpuAvailability")
        if avail is None:
            continue  # a DC advertising no per-chip rows is absence, not zero
        if not isinstance(avail, list):
            errors.append(
                f"{dc_id}: gpuAvailability misshapen "
                f"({type(avail).__name__}) -- skipped"
            )
            continue
        listed = dc.get("listed")
        for entry in avail:
            gpu_id = (
                str(entry.get("gpuTypeId") or "").strip()
                if isinstance(entry, dict)
                else ""
            )
            if not gpu_id:
                errors.append(
                    f"{dc_id}: gpuAvailability entry without gpuTypeId "
                    "-- skipped"
                )
                continue
            # available/stockStatus recorded verbatim (opaque by ruling);
            # dc_available_count guards with `is True` so a retyped flag
            # can never inflate the scalar.
            dc_map.setdefault(gpu_id, {})[dc_id] = {
                "available": entry.get("available"),
                "stock_status": entry.get("stockStatus"),
                "listed": listed,
            }
    return dc_map


def _stock_extras(
    g: Dict[str, Any],
    identifier: str,
    dc_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    errors: List[str],
) -> Dict[str, Any]:
    """Availability extras for one GpuType row -- fail-open, price-neutral.

    stockStatus values are opaque verbatim strings (no enum validation);
    null records None SILENTLY because it is the API's documented "no
    public quote at this size/filter". A missing probe alias or a retyped
    subtree/scalar is an anomaly: recorded as None and counted in
    partial_errors, and the price surfaces still print.
    """
    extra: Dict[str, Any] = {}
    for alias, key in _STOCK_PROBES:
        status = None
        if alias not in g:
            errors.append(
                f"{identifier[:60]}: {alias} probe missing from response "
                "-- stock unrecorded; price unaffected"
            )
        else:
            lp = g.get(alias)
            if lp is None:
                pass  # null probe = no public quote at this size/filter
            elif isinstance(lp, dict):
                status = lp.get("stockStatus")
                if status is not None and not isinstance(status, str):
                    errors.append(
                        f"{identifier[:60]}: {alias}.stockStatus retyped "
                        f"({str(status)[:40]!r}) -- recorded as None"
                    )
                    status = None
            else:
                errors.append(
                    f"{identifier[:60]}: {alias} misshapen "
                    f"({type(lp).__name__}) -- stock unrecorded; "
                    "price unaffected"
                )
        extra[key] = status
    max_count = g.get("maxGpuCount")
    if max_count is not None and (
        not isinstance(max_count, int) or isinstance(max_count, bool)
    ):
        errors.append(
            f"{identifier[:60]}: maxGpuCount retyped "
            f"({str(max_count)[:40]!r}) -- recorded as None"
        )
        max_count = None
    extra["max_gpu_count"] = max_count
    for field, key in (
        ("secureCloud", "offered_secure"),
        ("communityCloud", "offered_community"),
    ):
        val = g.get(field)
        if val is not None and not isinstance(val, bool):
            errors.append(
                f"{identifier[:60]}: {field} retyped "
                f"({str(val)[:40]!r}) -- recorded as None"
            )
            val = None
        extra[key] = val
    dc_rows = dc_map.get(identifier) if dc_map is not None else None
    extra["dc_availability"] = dc_rows
    # None (no matrix row / block failed) must never read as 0 -- absence
    # of a chip from the sparse matrix is not a signal.
    extra["dc_available_count"] = (
        sum(
            1
            for row in dc_rows.values()
            if row["available"] is True and row["listed"] is True
        )
        if dc_rows is not None
        else None
    )
    return extra


def parse_runpod(body: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the GraphQL body -> (observations, partial_errors).

    Fail-closed: null/zero prices are the API's documented "not offered on
    that surface" and skip silently; a price of any other unusable shape is
    counted in partial_errors so a partial schema change (one surface's
    field re-typed) surfaces in the snapshot instead of silently thinning
    the record. A whole-body shape change parses zero rows and the result
    builder raises. The availability subtrees (lowestPrice probes,
    dataCenters matrix) are the one sanctioned fail-open exception: their
    anomalies land in partial_errors and NEVER cost a price row.
    """
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    payload = json.loads(body)["data"]
    gpu_types = payload["gpuTypes"]
    dc_map = _fold_datacenters(payload, errors)
    for g in gpu_types:
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
        # Chip-level availability extras, built once per GPU row and shared
        # by all four price surfaces of that chip.
        stock_extra = _stock_extras(g, identifier, dc_map, errors)
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
                extra={"cloud": cloud, "display_name": display, **stock_extra},
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
