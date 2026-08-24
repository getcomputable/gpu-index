# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Scaleway -- public unauthenticated instance-catalog API, every GPU type.

Wide-net variant of the basket lane's B300-only scaleway collector
(gpu_index/index/sources.py parse_scaleway -- proven prior art; separate module,
separate request, the basket recipe is untouched). The observatory records
EVERY catalog entry carrying gpu >= 1: ``gpu_info.gpu_name`` is the
provider's structured part label and becomes sku_identifier; hourly_price
is EUR per NODE, normalized per-GPU by the catalog's own ``gpu`` count.
EUR is recorded natively (price_usd_gpu_hr stays None) -- FX is a
consumer's decision, never a capture-time one.

Surface facts verified live 2026-08-22 (fr-par-2, 136 catalog entries,
17 GPU rows across 6 GPU types: B300-SXM, H100-PCIe, H100-SXM, L4, L40S,
P100):

  - monthly_price is EXACTLY hourly_price * 730 on every GPU row -- a
    derived display figure, NOT a separate commitment tier, so it rides in
    extra for audit and is never recorded as a second observation;
  - zone coverage: fr-par-2 is a strict superset of the other probed
    zones' GPU offerings at identical prices (fr-par-1: L4/P100 subset;
    nl-ams-1: none; pl-waw-2: H100-PCIe/L4/L40S subset), so ONE zone keeps
    the fetch count at 1-2 requests without losing a GPU type;
  - mig_profile is null on every current row. A non-null mig_profile
    means a FRACTIONAL-GPU slice, where dividing the node price by the
    ``gpu`` count would be dishonest -- such rows are skipped and counted
    in partial_errors, never guessed.

Fail-closed identity pins: a reshaped ``servers`` mapping, a
non-numeric/non-finite hourly_price, or a non-numeric ``gpu`` count raises
with a specific message (a silent skip would zero the feed while looking
healthy -- and NaN parses as a float, so finiteness is part of "is a
number"); a numeric-but-fractional ``gpu`` count is skipped loudly (same
dishonest-division rule as mig_profile); an instance name repeating
across pages raises too -- pagination shifted mid-fetch, so rows may also
have been dropped unseen.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "scaleway"

ZONE = "fr-par-2"
URL = f"https://api.scaleway.com/instance/v1/zones/{ZONE}/products/servers"
PER_PAGE = 100  # the API's per_page max
# Catalog is ~136 entries today; 3 full pages (=300) means it more than
# doubled -- the fetch stops there but says so loudly rather than silently
# truncating (same cap discipline as the basket collector).
MAX_PAGES = 3

_GIB = 1024 ** 3


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_scaleway(
    page_bodies: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse over the fetched page bodies.

    Returns (observations, skipped_notes); skipped_notes feed
    partial_errors so an unpinnable row is a visible hole, never a guess.
    """
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    seen_names: set = set()
    for page_no, body in enumerate(page_bodies, start=1):
        payload = json.loads(body)
        servers = payload.get("servers") if isinstance(payload, dict) else None
        if not isinstance(servers, dict):
            raise RuntimeError(
                f"scaleway: page {page_no} reshaped -- 'servers' is no "
                "longer a name->spec mapping; refusing to guess at the new "
                "shape"
            )
        for name, spec in sorted(servers.items()):
            if name in seen_names:
                raise RuntimeError(
                    f"scaleway: instance type {name!r} appeared on two "
                    "pages -- pagination shifted mid-fetch, rows may also "
                    "have been dropped; refusing to double-print"
                )
            seen_names.add(name)
            if not isinstance(spec, dict):
                raise RuntimeError(
                    f"scaleway: spec for {name!r} is not a mapping -- "
                    "catalog entry shape changed"
                )
            gpu = spec.get("gpu")
            if gpu is None:
                continue  # no gpu field -- CPU-only catalog shape
            if not _is_number(gpu):
                raise RuntimeError(
                    f"scaleway: gpu count on {name!r} is no longer a plain "
                    f"number ({gpu!r}) -- count field reshaped; refusing to "
                    "guess which rows are GPU rows"
                )
            if int(gpu) != gpu:
                skipped.append(
                    f"{name}: fractional gpu count {gpu!r} -- skipped "
                    "(per-GPU normalization by a non-integral count would "
                    "be dishonest)"
                )
                continue
            count = int(gpu)
            if count < 1:
                continue  # CPU-only instance type
            gpu_info = spec.get("gpu_info") or {}
            gpu_name = str(gpu_info.get("gpu_name") or "").strip()
            if not gpu_name:
                skipped.append(
                    f"{name}: gpu row without gpu_info.gpu_name -- skipped, "
                    "no structured label to pin identity to"
                )
                continue
            if spec.get("mig_profile") is not None:
                skipped.append(
                    f"{name}: mig_profile={spec['mig_profile']!r} -- "
                    "fractional-GPU slice skipped (per-GPU normalization "
                    "by the node gpu count would be dishonest)"
                )
                continue
            hourly = spec.get("hourly_price")
            if hourly is None:
                skipped.append(
                    f"{name}: gpu row without hourly_price -- skipped "
                    "(unpriced catalog entry, not a $0 print)"
                )
                continue
            if not _is_number(hourly) or not math.isfinite(hourly):
                raise RuntimeError(
                    f"scaleway: hourly_price on {name!r} is no longer a "
                    f"plain finite number ({hourly!r}) -- price field "
                    "reshaped; refusing to guess"
                )
            if float(hourly) <= 0:
                skipped.append(
                    f"{name}: hourly_price {hourly} <= 0 -- skipped, not a "
                    "real print"
                )
                continue
            mem_bytes = gpu_info.get("gpu_memory")
            obs = observation(
                sku_identifier=gpu_name,
                price_per_gpu_hr=float(hourly) / count,
                currency="EUR",
                raw_value=str(hourly),
                raw_unit="eur_per_node_hr",
                gpu_count_basis=count,
                tier="on-demand",
                region=ZONE,
                notes=f"{name} node {hourly} EUR/hr",
                extra={
                    "instance_type": name,
                    "gpu_manufacturer": gpu_info.get("gpu_manufacturer"),
                    "gpu_memory_bytes": mem_bytes,
                    "monthly_price_eur_node": spec.get("monthly_price"),
                    "end_of_service": spec.get("end_of_service"),
                },
            )
            if _is_number(mem_bytes) and mem_bytes > 0 and mem_bytes % _GIB == 0:
                obs["memory_gb_label"] = int(mem_bytes) // _GIB
            rows.append(obs)
    return rows, skipped


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    page_bodies: List[str] = []
    full_pages = 0
    for page in range(1, MAX_PAGES + 1):
        body = fetch(
            f"{URL}?per_page={PER_PAGE}&page={page}", timeout=timeout
        )
        payload = json.loads(body)
        servers = (payload.get("servers") if isinstance(payload, dict) else None) or {}
        if not servers:
            break
        page_bodies.append(body)
        if len(servers) < PER_PAGE:
            break  # short page == last page; don't burn a request proving it
        full_pages += 1
    observations, skipped = parse_scaleway(page_bodies)
    partial_errors = list(skipped)
    if full_pages == MAX_PAGES:
        partial_errors.append(
            f"catalog paging stopped at {MAX_PAGES} full pages -- later "
            "pages unfetched; raise MAX_PAGES"
        )
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
