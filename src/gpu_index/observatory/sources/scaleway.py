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

Availability sibling (availability accrual, verified live 2026-08-25): the same
public unauthenticated Instance API publishes a per-instance-type stock
enum at ``.../products/servers/availability`` -- keys are the exact
catalog instance-type names, sole field per entry is ``availability``
with values from the scaleway-sdk-go ServerTypesAvailability enum
(available | scarce | shortage; no numeric stock exists unauthenticated).
Live fr-par-2 2026-08-25: page 1 = 100 entries, page 2 = 36 disjoint
entries, page 3 = empty; all 17 GPU catalog rows present (B300-SXM-*
shortage, H100-2-80G shortage, H100-SXM-* available...). collect()
fetches it with the same paginated loop and joins it in parse onto
extra["instance_type"], recording extra["availability"] VERBATIM.
Availability is per-zone AND per-size (fr-par-1's L4 ladder differs from
fr-par-2's), so only the fr-par-2 map is joined onto fr-par-2
observations -- other zones would be separate observations, out of scope.

The availability parse is fail-closed BY PLAN RULING (deliberate for a
sibling endpoint of the same API -- if it reshapes, the catalog is not
to be trusted either): a payload whose ``servers`` is not a mapping of
name -> {"availability": <str>} raises, as does a name repeating
across pages (same pagination tripwire as the catalog). The fail-soft
parts are per-row: a GPU catalog row absent from the availability map
records availability None with a loud note (proven possible -- fr-par-1
omits RENDER/P100 rows; never guess), and a value outside the known
enum is still recorded verbatim with an enum-drift note (tripwire
without darkening). The enum is an ordinal signal for consumers
(available > scarce > shortage); no doc glosses whether "shortage"
hard-blocks creation, so it is never interpreted at capture time.
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
AVAIL_URL = (
    f"https://api.scaleway.com/instance/v1/zones/{ZONE}/products/servers"
    "/availability"
)
PER_PAGE = 100  # the API's per_page max
# Catalog is ~136 entries today; 3 full pages (=300) means it more than
# doubled -- the fetch stops there but says so loudly rather than silently
# truncating (same cap discipline as the basket collector). The
# availability map (~140 entries live 2026-08-25) sits just above one
# page, so the same loud-stop discipline covers it too.
MAX_PAGES = 3

# The scaleway-sdk-go ServerTypesAvailability enum -- the full published
# vocabulary as of 2026-08-25. A value outside it is still recorded
# verbatim but noted in partial_errors (enum-drift tripwire without
# darkening); the semantics are an undocumented ordinal
# (available > scarce > shortage), never interpreted at capture time.
KNOWN_AVAILABILITY = frozenset({"available", "scarce", "shortage"})

_GIB = 1024 ** 3


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_availability(page_bodies: Sequence[str]) -> Dict[str, str]:
    """Pure parse of the availability sibling's page bodies into a flat
    instance-type -> verbatim-state map.

    Fail-closed by plan ruling: any reshape raises (same API host and
    versioned surface as the catalog -- a reshaped sibling means the
    catalog parse is not to be trusted either). An EMPTY map does not
    raise HERE (pure parser); collect() refuses an empty total map --
    the zone publishes ~140 entries, so a vanished map is a reshape,
    while a single catalog row missing from a populated map stays the
    per-row None + loud note case, never a guess.
    """
    out: Dict[str, str] = {}
    for page_no, body in enumerate(page_bodies, start=1):
        payload = json.loads(body)
        servers = payload.get("servers") if isinstance(payload, dict) else None
        if not isinstance(servers, dict):
            raise RuntimeError(
                f"scaleway: availability page {page_no} reshaped -- "
                "'servers' is no longer a name->state mapping; refusing to "
                "guess at the new shape"
            )
        for name, entry in sorted(servers.items()):
            if name in out:
                raise RuntimeError(
                    f"scaleway: instance type {name!r} appeared on two "
                    "availability pages -- pagination shifted mid-fetch, "
                    "rows may also have been dropped; refusing to join a "
                    "torn map"
                )
            state = entry.get("availability") if isinstance(entry, dict) else None
            if not isinstance(state, str):
                raise RuntimeError(
                    f"scaleway: availability entry for {name!r} is no "
                    "longer a mapping with a string 'availability' field "
                    "-- surface reshaped; refusing to guess"
                )
            out[name] = state
    return out


def parse_scaleway(
    page_bodies: Sequence[str],
    availability: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse over the fetched page bodies, joining the availability
    map (from ``parse_availability``, same zone only) onto each GPU row.

    Returns (observations, skipped_notes); skipped_notes feed
    partial_errors so an unpinnable row is a visible hole, never a guess.

    ``availability`` defaults to None (treated as an empty map) so the
    pre-availability call shape stays signature-compatible for external
    callers; every GPU row then records availability None WITH the loud
    join-miss note, never silently. collect() always passes the real
    fetched map (upstream declares the parameter required; the default
    is this repo's api-compat adaptation, behavior identical on the
    collect path).
    """
    if availability is None:
        availability = {}
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
            state = availability.get(name)
            if state is None:
                skipped.append(
                    f"{name}: no entry in the {ZONE} availability map -- "
                    "availability recorded as unknown (the map is proven "
                    "able to omit catalog rows; never guess)"
                )
            elif state not in KNOWN_AVAILABILITY:
                skipped.append(
                    f"{name}: availability {state!r} outside the known "
                    "available/scarce/shortage enum -- recorded verbatim "
                    "(ServerTypesAvailability drifted; ordinal reading "
                    "needs a human)"
                )
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
                    # Verbatim per-zone stock enum (None == not in the
                    # map); ordinal for consumers, never interpreted here.
                    "availability": state,
                },
            )
            if _is_number(mem_bytes) and mem_bytes > 0 and mem_bytes % _GIB == 0:
                obs["memory_gb_label"] = int(mem_bytes) // _GIB
            rows.append(obs)
    return rows, skipped


def _fetch_pages(base_url: str, timeout: float) -> Tuple[List[str], int]:
    """The shared paginated fetch loop (catalog + availability sibling)."""
    page_bodies: List[str] = []
    full_pages = 0
    for page in range(1, MAX_PAGES + 1):
        body = fetch(
            f"{base_url}?per_page={PER_PAGE}&page={page}", timeout=timeout
        )
        payload = json.loads(body)
        servers = payload.get("servers") if isinstance(payload, dict) else None
        if not isinstance(servers, dict):
            # A missing/renamed/retyped 'servers' envelope is a RESHAPE,
            # not the end of pagination -- keep the body so the fail-closed
            # parsers raise on it, instead of the page silently vanishing
            # into an empty (healthy-looking) map.
            page_bodies.append(body)
            break
        if not servers:
            break  # well-formed empty page == end of the book
        page_bodies.append(body)
        if len(servers) < PER_PAGE:
            break  # short page == last page; don't burn a request proving it
        full_pages += 1
    return page_bodies, full_pages


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    page_bodies, full_pages = _fetch_pages(URL, timeout)
    avail_bodies, avail_full_pages = _fetch_pages(AVAIL_URL, timeout)
    availability = parse_availability(avail_bodies)
    if not availability:
        # The zone publishes ~140 availability entries (live 2026-08-25);
        # an empty TOTAL map is an emptied or relocated surface, not a
        # stocked-out zone -- fail closed per the availability ruling
        # (the per-row map-omission notes cover single missing entries,
        # never a vanished map).
        raise RuntimeError(
            f"scaleway: the {ZONE} availability map came back empty -- "
            "surface emptied or relocated; refusing to record every GPU "
            "row as availability-unknown while looking healthy"
        )
    observations, skipped = parse_scaleway(page_bodies, availability)
    partial_errors = list(skipped)
    if full_pages == MAX_PAGES:
        partial_errors.append(
            f"catalog paging stopped at {MAX_PAGES} full pages -- later "
            "pages unfetched; raise MAX_PAGES"
        )
    if avail_full_pages == MAX_PAGES:
        partial_errors.append(
            f"availability paging stopped at {MAX_PAGES} full pages -- "
            "later pages unfetched; raise MAX_PAGES"
        )
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
