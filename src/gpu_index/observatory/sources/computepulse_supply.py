# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Compute Pulse supply -- public capacity-offer marketplace book.

AGGREGATOR-PUBLISHED CLASSIFIEDS, NOT PRINTS: every row on /api/supply is a
SELLER-DECLARED capacity offer that compute-pulse.com republishes -- the
surface's own meta.notice says verbatim "Listings are discovery records,
not guaranteed quotes. Confirm identity, availability, price, and terms
before transacting.", and sellers can publish as "Anonymous". The rows are
recorded anyway because the availability window IS the payload (this is
the sibling of the computepulse listings collector, added for the
availability-accrual lane): each offer states available_now,
availability_state, available_from/available_until, gpu_count,
min_bookable_gpus, min_duration_hours, interconnect -- all recorded
VERBATIM in ``extra`` so a consumer can screen on the full declaration;
nothing here is ever a first-party price (first_party False).

Surface verified live 2026-08-25: GET /api/supply?intent=selling&
include_ended=false&limit=200 (public, no auth; openapi 3.1.0 at
/openapi.json declares no security scheme). The whole selling book fit one
page (page.total 25, next_offset null; 28 with include_ended=true), but
the pagination guard stays anyway. The listings sort enum does NOT apply
here (sort=freshest drew a live HTTP 400 on 2026-08-25 -- supply's enum is
fresh/price/capacity/soonest), so sort is left unset (server default
"fresh").

Identity and honesty pins (each exists because the live surface could
drift into silently lying):

  - visibility: meta.visibility must equal the openapi const
    "published_public_fields_only" -- the envelope's own statement that
    these are the published public fields; anything else means the shape
    (or the publication contract) changed, so the parse raises.
  - intent: the query filters intent=selling and every recorded row's own
    intent must be "capacity_offer" (openapi enum: capacity_offer |
    buyer_requirement). A buyer_requirement row slipping through would be
    a BID, not an ask -- recording its price as a supply-side offer would
    invert the sign, so mismatched rows are skipped + counted.
  - currency: price_per_gpu_hour is only read as USD when the row's own
    price_currency says "USD" (the openapi enum is ["USD", null] and its
    price description says USD, but the per-row field is the structured
    warrant). A priced row with a non-USD ISO code records NATIVELY
    (price_usd_gpu_hr stays None); a priced row with null/junk currency is
    skipped + counted, never assumed USD.
  - unpriced offers: price null means "not stated" (openapi) and REAL rows
    carry it (an Anonymous 8x H100 InfiniBand offer, live 2026-08-25).
    They can never be observation rows (no $0 prices, ever) -- their
    capacity signal is kept in book_stats (offers_total / offers_priced /
    offers_unpriced / offers_available_now) plus a partial_errors note, so
    the missing price never silently drops the declared supply.
  - tier: this surface has NO billing_type -- tier is DERIVED by ruling:
    min_duration_hours >= 720 -> monthly-commit, else (smaller, null, or
    unparseable) on-demand. The verbatim min_duration_hours and the
    derivation rule ride in extra on every row so the derivation is
    auditable and re-derivable from raw. Live 2026-08-25: 730/2016/8736
    -hour floors -> monthly-commit; 168/672/null -> on-demand.
  - availability_state: openapi enum is now|soon|ended but only "now" has
    ever been observed live (25/25 rows, 2026-08-25) -- recorded VERBATIM
    with no enum fence (fail-open by plan: an unobserved "soon" appearing
    is signal, not an error).
  - basis: price_per_gpu_hour is the seller's ALREADY-PER-GPU figure --
    raw_value carries it verbatim (raw_unit says per_gpu_hr, so raw ==
    price directly) and gpu_count_basis carries the offer's stated
    gpu_count. A priced row without a positive integer gpu_count is
    skipped, never given a guessed basis.
  - pagination: page.next_offset must be null or strictly advance past the
    current offset (no loops); offers are deduped across pages by slug
    (openapi: "Stable public listing slug") so a book shifting between
    fetches does not double-record an offer.

Rows are LEADS with provenance: extra carries company (can be
"Anonymous"), slug (the stable public key under /supply/<slug>), and the
verbatim availability window. The parse fails closed on envelope pins;
result() covers the parsed-nothing case (an all-unpriced book would raise
there -- a capture with zero price observations must never look healthy).
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

# One request per capture today (25-offer book, 200-row page clamp); the
# spacing only matters if the book ever grows past a page. Same politeness
# rationale as the computepulse listings sibling.
REQUEST_SPACING_SECONDS = 0.5

SOURCE_ID = "computepulse_supply"

URL_BASE = "https://compute-pulse.com/api/supply"
# The API's own page clamp (openapi: limit maximum 200).
API_MAX_LIMIT = 200
# 5 pages x 200 = 1000 offers before truncation -- 40x the live book
# (25 offers, 2026-08-25); if this cap ever bites, the truncation note
# says so loudly.
MAX_PAGES = 5
# openapi const for meta.visibility -- the envelope's publication contract.
VISIBILITY_PIN = "published_public_fields_only"
# Tier derivation by ruling (no billing_type exists on this surface):
# a stated minimum term of 720h (30 days) or more is a monthly commitment,
# anything smaller/unstated prices as on-demand.
MONTHLY_COMMIT_MIN_HOURS = 720
TIER_DERIVATION = (
    "derived: min_duration_hours >= 720 -> monthly-commit, else on-demand "
    "(no billing_type on this surface)"
)

_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def offer_observation(
    row: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """One capacity offer -> (observation, None) or (None, skip_category).

    Skip categories are countable pin names, surfaced in partial_errors --
    a row is never guessed into the record. "unpriced" is special-cased by
    the caller into book_stats counts (a real book state, not a defect).
    """
    if not isinstance(row, dict):
        return None, "malformed_row"
    intent = row.get("intent")
    if intent != "capacity_offer":
        # The query filters intent=selling; a non-offer row means the
        # server-side filter stopped discriminating. A buyer_requirement
        # price is a bid, not an ask -- never recordable as supply.
        return None, "non_offer_intent"
    identifier = str(row.get("gpu_model") or "").strip()
    if not identifier:
        return None, "unlabeled"
    price = row.get("price_per_gpu_hour")
    if not _positive_number(price):
        # Null means "not stated" (openapi) and zero is not a price; the
        # caller keeps the capacity signal in book_stats counts.
        return None, "unpriced"
    gpu_count = row.get("gpu_count")
    if (
        not isinstance(gpu_count, int)
        or isinstance(gpu_count, bool)
        or gpu_count < 1
    ):
        return None, "bad_gpu_count"
    currency_raw = row.get("price_currency")
    code = str(currency_raw).strip().upper() if currency_raw is not None else ""
    if code == "USD":
        currency = "USD"
        raw_unit = "usd_per_gpu_hr_seller_declared"
    elif _CURRENCY_CODE_RE.match(code):
        # openapi drift (the enum is ["USD", null]) but the row still
        # states a recognizable currency: record natively, never as USD.
        currency = code
        raw_unit = f"{code.lower()}_per_gpu_hr_seller_declared"
    else:
        # Priced but no structured currency warrant (null or junk):
        # refusing to assume USD beats mislabeling a price.
        return None, "unpinned_currency"

    mdh = row.get("min_duration_hours")
    monthly = (
        isinstance(mdh, (int, float))
        and not isinstance(mdh, bool)
        and mdh >= MONTHLY_COMMIT_MIN_HOURS
    )
    tier = "monthly-commit" if monthly else "on-demand"

    company = str(row.get("company") or "").strip()
    extra: Dict[str, Any] = {
        # The availability window IS the payload -- everything verbatim,
        # plus the provenance a consumer needs to screen a classified.
        "company": row.get("company"),
        "slug": row.get("slug"),
        "intent": intent,
        "available_now": row.get("available_now"),
        "availability_state": row.get("availability_state"),
        "available_from": row.get("available_from"),
        "available_until": row.get("available_until"),
        "gpus_per_node": row.get("gpus_per_node"),
        "min_bookable_gpus": row.get("min_bookable_gpus"),
        "min_duration_hours": mdh,
        "interconnect": row.get("interconnect"),
        "tier_derivation": TIER_DERIVATION,
    }
    notes = (
        f"{company or 'unattributed'} capacity offer via compute-pulse "
        f"supply marketplace, {gpu_count}x listing (seller-declared "
        "classified, not a quote)"
    )
    return (
        observation(
            sku_identifier=identifier,
            price_per_gpu_hr=float(price),
            currency=currency,
            raw_value=str(price),
            raw_unit=raw_unit,
            gpu_count_basis=gpu_count,
            tier=tier,
            region=str(row.get("region") or "?"),
            notes=notes,
            extra=extra,
        ),
        None,
    )


def parse_supply_page(body: str) -> Dict[str, Any]:
    """Validate one /api/supply page and apply every pin (pure).

    Returns {"observations", "rows", "skips", "unpriced", "available_now",
    "raw_row_count", "total", "next_offset"} -- the count keys are
    PAGE-LOCAL; "rows" carries per-row (slug, available_now, skip, obs)
    records so the caller can dedup every book-level rollup across pages.
    Raises on envelope-level pin failures (shape, visibility const,
    pagination shape) -- those poison the whole page, not single rows.
    """
    doc = json.loads(body)
    if not isinstance(doc, dict):
        raise RuntimeError(
            "computepulse_supply: supply response is not a JSON object "
            "-- API shape changed"
        )
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError(
            "computepulse_supply: supply response lost its meta block "
            "-- API shape changed"
        )
    visibility = meta.get("visibility")
    if visibility != VISIBILITY_PIN:
        raise RuntimeError(
            f"computepulse_supply: meta.visibility is {visibility!r}, not "
            f"{VISIBILITY_PIN!r} -- the publication contract changed; "
            "refusing to record"
        )
    data = doc.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            "computepulse_supply: supply response lost its data list "
            "-- API shape changed"
        )
    page = doc.get("page")
    if not isinstance(page, dict):
        raise RuntimeError(
            "computepulse_supply: supply response lost its page block "
            "-- API shape changed"
        )
    total = page.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise RuntimeError(
            "computepulse_supply: page.total is not a non-negative "
            "integer -- pagination shape changed"
        )
    next_offset = page.get("next_offset")
    if next_offset is not None and (
        not isinstance(next_offset, int)
        or isinstance(next_offset, bool)
        or next_offset < 0
    ):
        raise RuntimeError(
            "computepulse_supply: page.next_offset is neither null nor a "
            "non-negative integer -- pagination shape changed"
        )
    observations: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    skips: Dict[str, int] = {}
    unpriced = 0
    available_now = 0
    for row in data:
        raw_slug = row.get("slug") if isinstance(row, dict) else None
        available = isinstance(row, dict) and row.get("available_now") is True
        if available:
            # Counted over ALL rows (priced or not): the declared-available
            # capacity signal must survive a missing price.
            available_now += 1
        obs, skip = offer_observation(row)
        if obs is not None:
            observations.append(obs)
        elif skip == "unpriced":
            unpriced += 1
        else:
            skips[skip] = skips.get(skip, 0) + 1
        # Per-row record for the caller's CROSS-PAGE dedup: the page-local
        # counts above are honest for one page, but a book shifting between
        # fetches replays offers across pages and every book-level rollup
        # (not just observations) must count a replayed offer once.
        rows.append(
            {
                "slug": (
                    raw_slug
                    if isinstance(raw_slug, str) and raw_slug
                    else None
                ),
                "available_now": available,
                "skip": skip,
                "obs": obs,
            }
        )
    return {
        "observations": observations,
        "rows": rows,
        "skips": skips,
        "unpriced": unpriced,
        "available_now": available_now,
        "raw_row_count": len(data),
        "total": total,
        "next_offset": next_offset,
    }


def _fetch_book(
    timeout: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]:
    """The whole selling book: up to MAX_PAGES pages (one covers it live),
    deduped across pages by stable slug. Returns (observations, stats,
    skips)."""
    observations: List[Dict[str, Any]] = []
    seen_slugs: Set[str] = set()
    skips: Dict[str, int] = {}
    dups = 0
    unpriced = 0
    available_now = 0
    raw_rows = 0
    offset = 0
    pages_fetched = 0
    total = 0
    truncated = False
    while True:
        if pages_fetched > 0:
            time.sleep(REQUEST_SPACING_SECONDS)
        # NO sort param: supply's sort enum differs from listings'
        # (sort=freshest drew a live 400 on 2026-08-25); default "fresh".
        query = urllib.parse.urlencode(
            {
                "intent": "selling",
                "include_ended": "false",
                "limit": API_MAX_LIMIT,
                "offset": offset,
            }
        )
        page = parse_supply_page(fetch(f"{URL_BASE}?{query}", timeout=timeout))
        pages_fetched += 1
        raw_rows += page["raw_row_count"]
        total = page["total"]
        for rec in page["rows"]:
            slug = rec["slug"]
            if slug is not None:
                if slug in seen_slugs:
                    # Replayed across pages (keyset book shifted between
                    # fetches): counted ONCE in every rollup --
                    # observations AND the available/unpriced/skip counts.
                    dups += 1
                    continue
                seen_slugs.add(slug)
            if rec["available_now"]:
                available_now += 1
            if rec["obs"] is not None:
                observations.append(rec["obs"])
            elif rec["skip"] == "unpriced":
                unpriced += 1
            elif rec["skip"] is not None:
                skips[rec["skip"]] = skips.get(rec["skip"], 0) + 1
        next_offset = page["next_offset"]
        if next_offset is None:
            break
        if next_offset <= offset:
            raise RuntimeError(
                f"computepulse_supply: next_offset {next_offset} does not "
                f"advance past offset {offset} -- pagination shape changed; "
                "refusing to loop"
            )
        if pages_fetched >= MAX_PAGES:
            truncated = True
            break
        offset = next_offset
    if dups:
        skips["duplicate_slug"] = dups
    stats = {
        "offers_total": total,
        "pages_fetched": pages_fetched,
        "raw_rows": raw_rows,
        "offers_priced": len(observations),
        "offers_unpriced": unpriced,
        "offers_available_now": available_now,
        "fetch_truncated": truncated,
    }
    return observations, stats, skips


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    del options  # single fixed public book -- nothing to configure
    observations, stats, skips = _fetch_book(timeout)
    partial: List[str] = []
    print(
        f"  computepulse_supply: {stats['raw_rows']} offers over "
        f"{stats['pages_fetched']} page(s), recording "
        f"{stats['offers_priced']} priced of {stats['offers_total']} "
        f"({stats['offers_unpriced']} unpriced, "
        f"{stats['offers_available_now']} declared available now)"
    )
    if skips:
        partial.append(
            "skipped "
            + ", ".join(f"{count} {cat}" for cat, count in sorted(skips.items()))
            + " offer(s) (fail-closed pins; see module docstring)"
        )
    if stats["offers_unpriced"]:
        partial.append(
            f"{stats['offers_unpriced']} unpriced offer(s) counted in "
            "book_stats only (price null = not stated; never a $0 row -- "
            "the declared capacity survives in offers_unpriced/"
            "offers_available_now)"
        )
    if stats["fetch_truncated"]:
        partial.append(
            f"book wider than the {MAX_PAGES}-page cap -- recorded first "
            f"{stats['offers_priced']} priced of {stats['offers_total']} "
            "offers (server default sort)"
        )
    if not observations and stats["raw_rows"] > stats["offers_unpriced"]:
        # Priced rows existed but ALL flunked the pins: the supply API
        # shape or its intent filter changed; refusing to guess. (An
        # all-unpriced or empty book falls through to result(), which
        # raises its parsed-nothing refusal.)
        raise RuntimeError(
            f"computepulse_supply: {stats['raw_rows']} offers fetched but "
            "ZERO survived the identity and honesty pins -- the supply API "
            "shape changed; refusing to guess"
        )
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL_BASE,
        observations=observations,
        first_party=False,  # seller-declared classifieds republished by the aggregator
        partial_errors=partial or None,
        book_stats=stats,
    )
