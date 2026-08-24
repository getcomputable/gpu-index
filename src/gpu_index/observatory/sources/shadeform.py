# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Shadeform -- homepage JSON blobs, every instance type, all clouds.

Shadeform is a RESELLER republishing other clouds' capacity, so
first_party=False and extra["cloud"] (which upstream cloud each row
re-publishes) is the load-bearing disclosure on every observation. The
basket lane's collector (gpu_index/index/sources.py parse_shadeform_b200) proved
the recipe but records the B200 pool only; the observatory records EVERY
blob row. Separate module, separate parse -- the basket recipe is untouched.

Shape verified live 2026-08-22: the homepage embeds one escaped JSON array
of instance-type blobs, each with "cloud" as its FIRST key and
"deployment_type" as its LAST (the blob regex is that ordering pin -- a
reshape matches nothing and the parsed-zero fence trips). Per blob:
``hourly_price`` is integer CENTS per INSTANCE hour (normalized per GPU by
``num_gpus``), ``gpu_type`` is the structured part label (sku_identifier),
``deployment_type`` is a form factor {vm, baremetal} -- both are on-demand
rentals, so tier maps to on-demand and the raw value rides in extra; an
unknown value is a row we refuse to tier-guess. ``availability`` regions
carry per-region stock flags (recorded verbatim in extra). CPU-only
instances (gpu_type "CPU", num_gpus 0) are out of scope for a GPU price
lane and are skipped silently by rule, not as failures.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "shadeform"

# Apex host first; www serves the same document (basket-proven fallback).
URLS = ("https://shadeform.com/", "https://www.shadeform.com/")

# Identity pin: "cloud" first key, "deployment_type" last key -- the blob
# boundary marker verified live (276/276 blobs, 2026-08-22).
_BLOB_RE = re.compile(r'\{"cloud":".*?"deployment_type":"[a-z_]+"\}')

# deployment_type is packaging (VM vs bare metal), not a rental tier: both
# live values are plain on-demand hourly rentals. The raw value is recorded
# in extra; anything outside this map is skipped, never tier-guessed.
_TIER_BY_DEPLOYMENT_TYPE = {"vm": "on-demand", "baremetal": "on-demand"}


def _region_rollup(availability: List[Dict[str, Any]]) -> str:
    """US / non-US / mixed from the row's own region display names.

    The full per-region stock detail is recorded verbatim in extra; this is
    only the coarse top-level field."""
    us_flags = [
        str(a.get("display_name") or "").startswith("US") for a in availability
    ]
    if not us_flags:
        return "?"
    if all(us_flags):
        return "US"
    if any(us_flags):
        return "mixed"
    return "non-US"


def parse_shadeform(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the homepage body -> (observations, partial_errors).

    Fail-closed: rows that fail an identity pin (unknown deployment_type,
    non-integer-cents price, unusable num_gpus, missing attribution) are
    skipped and counted in partial_errors; if blobs matched but ZERO rows
    survived the pins, the surface's semantics changed and we raise rather
    than return a healthy-looking empty print.
    """
    txt = html.replace('\\"', '"')
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen = set()
    blob_count = 0
    for m in _BLOB_RE.finditer(txt):
        blob = m.group(0)
        if blob in seen:
            # The same blob rendered twice (e.g. escaped payload duplicated
            # by the framework) must not double-count an offer.
            continue
        seen.add(blob)
        blob_count += 1
        try:
            rec = json.loads(blob)
        except json.JSONDecodeError:
            errors.append(f"unparseable listing blob: {blob[:80]}")
            continue
        gpu_type = str(rec.get("gpu_type") or "").strip()
        if gpu_type == "CPU":
            continue  # CPU-only instance: out of scope by rule, not an error
        cloud = str(rec.get("cloud") or "").strip()
        instance = str(rec.get("shade_instance_type") or "?").strip()
        label = f"{cloud or '?'}/{instance}"
        if not gpu_type or not cloud:
            errors.append(
                f"{label}: row without gpu_type/cloud attribution -- skipped"
            )
            continue
        deployment_type = rec.get("deployment_type")
        tier = _TIER_BY_DEPLOYMENT_TYPE.get(deployment_type)
        if tier is None:
            errors.append(
                f"{label}: unknown deployment_type {deployment_type!r} -- "
                "tier not guessable, skipped"
            )
            continue
        n = rec.get("num_gpus")
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            errors.append(
                f"{label}: num_gpus {n!r} unusable for per-GPU "
                "normalization -- skipped"
            )
            continue
        cents = rec.get("hourly_price")
        if not isinstance(cents, int) or isinstance(cents, bool) or cents <= 0:
            errors.append(
                f"{label}: hourly_price {cents!r} is not positive integer "
                "cents -- skipped"
            )
            continue
        availability = [
            a for a in rec.get("availability") or [] if isinstance(a, dict)
        ]
        config = rec.get("configuration") or {}
        vram = config.get("vram_per_gpu_in_gb")
        vram_ok = isinstance(vram, (int, float)) and not isinstance(vram, bool)
        obs = observation(
            sku_identifier=gpu_type,
            price_per_gpu_hr=cents / 100.0 / n,
            raw_value=str(cents),
            raw_unit="cents_per_instance_hr",
            gpu_count_basis=n,
            tier=tier,
            region=_region_rollup(availability),
            notes=f"{instance} via {cloud}"
            + (f" {vram:g}GB/GPU" if vram_ok else ""),
            extra={
                "cloud": cloud,
                "deployment_type": deployment_type,
                "shade_instance_type": instance,
                "cloud_instance_type": rec.get("cloud_instance_type"),
                "gpu_manufacturer": config.get("gpu_manufacturer"),
                "interconnect": rec.get("interconnect"),
                "nvlink": rec.get("nvlink"),
                "availability": availability,
            },
        )
        if vram_ok:
            obs["memory_gb_label"] = vram
        rows.append(obs)
    if blob_count and not rows:
        raise RuntimeError(
            f"{SOURCE_ID}: {blob_count} listing blobs matched but zero "
            f"passed the identity pins (sample reasons: {errors[:3]}) -- "
            "page semantics changed; refusing to guess"
        )
    return rows, errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    html: Optional[str] = None
    for url in URLS:
        try:
            html = fetch(url, timeout=timeout)
            break
        except Exception as exc:  # noqa: BLE001 -- try the www fallback
            last_err = exc
    if html is None:
        raise RuntimeError(
            f"{SOURCE_ID} unreachable on both hosts: {last_err}"
        )
    observations, partial_errors = parse_shadeform(html)
    return result(
        SOURCE_ID,
        method="html-json-blobs",
        url=URLS[0],
        observations=observations,
        first_party=False,  # reseller republishing other clouds' capacity
        partial_errors=partial_errors or None,
    )
