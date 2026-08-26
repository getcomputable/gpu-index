# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""GPU.ai -- public pricing API, every offer row, both capacity classes.

Surface verified live 2026-08-22: GET https://api.gpu.ai/v1/pricing
?limit=200 is public no-auth JSON (the OpenAPI contract at /v1/openapi.json
declares security [] for this operation) with keyset cursor pagination via
next_cursor. Rows are offer-level -- gpu_type x gpu_count x region x
capacity_class x disk config, many rows per chip -- and are recorded as
labeled rows, never pre-aggregated. The spec also lists a demo server
(api.demo.gpu.ai, used by its own code samples); this module is pinned to
the production host -- demo prices are not observations of anything.

Basis: price_per_hour is the hourly price for the WHOLE listed
configuration (schema-documented: "covers all gpu_count GPUs"), so the
per-GPU normalization divides by gpu_count while raw_value carries the
published per-instance figure verbatim. B300 lists ONLY as an 8x config
today -- skipping the division would record $50.62/GPU instead of $6.33.

Currency: the Pricing schema publishes no currency field. USD is IMPLIED,
not declared -- the site renders $, and the billing endpoints in the same
spec are amount_usd_cents / usdc-usdt -- so rows record as USD with the
implication noted per row; if a row ever grows a currency key that is not
"USD", the capture refuses rather than mislabel a native price.

Identity pins (fail-closed). This is a first-party structured API with a
declared schema, so a violating ROW means the surface reshaped and the
WHOLE capture is refused -- unlike heterogeneous scraped surfaces there is
no honest row-skip here:

  - envelope: a JSON object with data (list) and a PRESENT next_cursor
    (non-empty string or null) -- a vanished next_cursor key would
    silently truncate the book to one page;
  - every row: gpu_type (non-empty str), gpu_count (int >= 1),
    price_per_hour (number > 0), tier in {on_demand, spot}, region
    (non-empty str), available (int >= 0 -- load-bearing now that
    sold-out rows are recorded; 314/314 live rows pass on 2026-08-25),
    offering_id (non-empty str). offering_id is absent
    from the published Pricing schema but present on 100% of live rows
    (272/272 on 2026-08-22) and is the stable per-offer identity
    (customer-identical offers are pre-merged server-side with
    availability summed) -- without it rows are unattributable across
    captures, so it is pinned deliberately;
  - capacity_class in {secure, community, ""} AND it must agree with its
    required boolean twin `community` (the schema's own words: "the
    string twin of community"); a transiently-empty capacity_class
    (documented for cache entries written before the field existed)
    falls back to the boolean -- schema-warranted, not a guess -- with
    the fallback noted in extra. secure and community are recorded as
    separate labeled observations (the runpod pattern);
  - the tier enum includes "spot" but the live book is all on_demand
    today -- the pin accepts spot appearing;
  - pagination: at most MAX_PAGES pages (10 x 200 rows, far above the
    314-row live book; hitting the fence RAISES -- a runaway cursor or an
    exploded book both deserve a loud error, never a silent truncation);
    a cursor that fails to advance raises; rows replayed across page
    boundaries (a keyset book can shift between fetches) are deduped by
    offering_id and counted in partial_errors.

The book is fetched with include_unavailable=true: sold-out offers ARE
observations of the book, distinguished by extra.available == 0 (the
default response omits exactly the available==0 rows -- verified live
2026-08-25: 314 rows vs 249 default, difference 65 = the zero-available
count). Sold-out rows keep their full shape and stable offering_id (the
8x b300 held a0a48d83d5f6 across its stockout), so per-offer availability
time series survive stockouts instead of gapping, and a fully stocked-out
chip stays distinguishable from a delisted one. Two consequences:

  - a sold-out row's price_per_hour is a LISTED price, not currently
    transactable -- safe in this capture-only observatory, but any future
    consumer computing an offered-price statistic from gpuai rows must
    fence on extra.available > 0 first;
  - include_unavailable is UNDOCUMENTED (the public reference documents
    no /pricing query params); it is honored live but if the server ever
    stopped honoring it we would get the default book back -- zero
    available==0 rows, shape-identical, unpinnable. book_stats records
    rows_zero_available per capture so that failure mode is scannable:
    many consecutive captures at 0 while chips like b300 vanish from the
    book entirely is the tripwire.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List, Optional, Set

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "gpuai"

# Production host ONLY -- the OpenAPI spec also lists api.demo.gpu.ai.
URL_BASE = "https://api.gpu.ai/v1/pricing"
# The API's own page clamp (openapi: limit maximum 200, default 50).
API_MAX_LIMIT = 200
# Runaway fence: 10 x 200 = 2000 rows >> the 314-row live book
# (include_unavailable mode, 2026-08-25).
MAX_PAGES = 10

# API tier enum -> lane tier vocabulary, fail-closed.
_TIER_BY_API = {"on_demand": "on-demand", "spot": "spot"}
_CAPACITY_CLASSES = ("secure", "community", "")


def _refuse(reason: str) -> RuntimeError:
    return RuntimeError(
        f"gpuai: {reason} -- API shape changed; refusing the whole capture "
        "rather than guessing (see module docstring)"
    )


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def row_observation(row: Any) -> Dict[str, Any]:
    """One pricing row -> observation dict (pure; raises on pin violation)."""
    if not isinstance(row, dict):
        raise _refuse("pricing row is not an object")
    gpu_type = row.get("gpu_type")
    if not isinstance(gpu_type, str) or not gpu_type.strip():
        raise _refuse("row without a gpu_type label")
    gpu_count = row.get("gpu_count")
    if (
        not isinstance(gpu_count, int)
        or isinstance(gpu_count, bool)
        or gpu_count < 1
    ):
        raise _refuse(f"{gpu_type}: gpu_count is not a positive integer")
    price = row.get("price_per_hour")
    if not _number(price) or price <= 0:
        raise _refuse(f"{gpu_type}: price_per_hour is not a positive number")
    tier = _TIER_BY_API.get(row.get("tier"))
    if tier is None:
        raise _refuse(
            f"{gpu_type}: tier {row.get('tier')!r} outside the pinned enum"
        )
    region = row.get("region")
    if not isinstance(region, str) or not region.strip():
        raise _refuse(f"{gpu_type}: row without a region")
    offering_id = row.get("offering_id")
    if not isinstance(offering_id, str) or not offering_id.strip():
        raise _refuse(
            f"{gpu_type}: row without an offering_id -- the stable "
            "per-offer identity this time series keys on"
        )
    available = row.get("available")
    if (
        not isinstance(available, int)
        or isinstance(available, bool)
        or available < 0
    ):
        raise _refuse(
            f"{gpu_type}: available is not a non-negative integer -- "
            "load-bearing since sold-out rows (available==0) became part "
            "of the recorded book"
        )
    community = row.get("community")
    if not isinstance(community, bool):
        raise _refuse(f"{gpu_type}: community flag is not a boolean")
    capacity_class = row.get("capacity_class")
    if capacity_class not in _CAPACITY_CLASSES:
        raise _refuse(
            f"{gpu_type}: capacity_class {capacity_class!r} outside the "
            "pinned enum"
        )
    if capacity_class and (capacity_class == "community") != community:
        raise _refuse(
            f"{gpu_type}: capacity_class {capacity_class!r} disagrees with "
            f"its boolean twin community={community!r} -- the capacity "
            "label cannot be trusted"
        )
    if "currency" in row and row.get("currency") != "USD":
        raise _refuse(
            f"{gpu_type}: row grew a currency field "
            f"({row.get('currency')!r}) -- the implied-USD warrant no "
            "longer holds"
        )

    cls = capacity_class or ("community" if community else "secure")
    extra: Dict[str, Any] = {
        "capacity_class": cls,
        "available": available,
        "instant_boot": row.get("instant_boot"),
        "deployment_type": row.get("deployment_type"),
    }
    if not capacity_class:
        # Documented transient (pre-field cache entries): the class above
        # came from the required boolean twin, not the string field.
        extra["capacity_class_source"] = "community_flag_fallback"
    obs = observation(
        sku_identifier=gpu_type,
        price_per_gpu_hr=float(price) / gpu_count,
        raw_value=str(price),
        raw_unit="usd_per_instance_hr",
        gpu_count_basis=gpu_count,
        tier=tier,
        region=region,
        notes=(
            f"{gpu_type} {cls} capacity, {gpu_count}x offer; USD implied "
            "(API publishes no currency field)"
        ),
        extra=extra,
    )
    obs["offer_id"] = offering_id
    return obs


def parse_pricing_page(body: str) -> Dict[str, Any]:
    """Validate one /pricing page and convert every row (pure).

    Returns {"observations", "next_cursor", "raw_row_count"}. Raises on
    ANY pin violation -- the module docstring says why this surface is
    refused whole instead of row-skipped.
    """
    doc = json.loads(body)
    if not isinstance(doc, dict):
        raise _refuse("pricing response is not a JSON object")
    data = doc.get("data")
    if not isinstance(data, list):
        raise _refuse("pricing response lost its data list")
    if "next_cursor" not in doc:
        raise _refuse(
            "pricing response lost its next_cursor key -- pagination "
            "semantics changed and the book would silently truncate"
        )
    cursor = doc["next_cursor"]
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise _refuse("next_cursor is neither null nor a non-empty string")
    return {
        "observations": [row_observation(row) for row in data],
        "next_cursor": cursor,
        "raw_row_count": len(data),
    }


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seen_offers: Set[str] = set()
    dups = 0
    raw_rows = 0
    cursor: Optional[str] = None
    pages = 0
    while True:
        # include_unavailable brings the sold-out (available==0) rows into
        # the book -- see the module docstring for the posture and the
        # silent-revert tripwire.
        url = f"{URL_BASE}?limit={API_MAX_LIMIT}&include_unavailable=true"
        if cursor is not None:
            url += "&cursor=" + urllib.parse.quote(cursor, safe="")
        page = parse_pricing_page(fetch(url, timeout=timeout))
        pages += 1
        raw_rows += page["raw_row_count"]
        for obs in page["observations"]:
            oid = obs["offer_id"]
            if oid in seen_offers:
                dups += 1
                continue
            seen_offers.add(oid)
            rows.append(obs)
        next_cursor = page["next_cursor"]
        if next_cursor is None:
            break
        if next_cursor == cursor:
            raise RuntimeError(
                "gpuai: next_cursor did not advance between pages -- "
                "pagination shape changed; refusing to loop"
            )
        if pages >= MAX_PAGES:
            raise RuntimeError(
                f"gpuai: book still paginating after {MAX_PAGES} pages "
                f"({raw_rows} rows fetched vs a 314-row live book on "
                "2026-08-25) -- runaway cursor or an exploded book; "
                "refusing to record a silently truncated capture"
            )
        cursor = next_cursor
    partial = None
    if dups:
        partial = [
            f"deduplicated {dups} offering_id replay(s) across page "
            "fetches (keyset book shifted between pages)"
        ]
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL_BASE,
        observations=rows,
        partial_errors=partial,
        book_stats={
            "pages_fetched": pages,
            "raw_rows": raw_rows,
            "rows_recorded": len(rows),
            # Stockout breadth per capture -- ALSO the include_unavailable
            # silent-revert tripwire (see module docstring): a long run of
            # zeros here while chips vanish from the book means the
            # undocumented param stopped being honored.
            "rows_zero_available": sum(
                1 for r in rows if r["extra"]["available"] == 0
            ),
        },
    )
