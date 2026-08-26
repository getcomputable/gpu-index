# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Compute Pulse -- public listings API, per-chip aggregator lead books.

AGGREGATOR, NOT A PROVIDER: every row is compute-pulse.com's observation of
some OTHER provider's price -- leads, not prints (first_party False). The
rows are recorded anyway because the surface is wide (hundreds of listings
per chip across providers we do not collect directly), but every row
carries its full trust metadata (provider, source_kind, source_url,
human_verified, verification_level, available) in ``extra`` so a consumer
can screen on provenance; nothing here is ever a first-party price.

Surface verified live 2026-08-22: GET /api/listings?gpu=<slug> (public,
no auth; openapi at /openapi.json; page clamp 200 rows). The /gpu/<model>
HTML pages truncate to ~12 rows and are never used. Slugs and the page
limit come from config (options["gpus"], options["limit_per_gpu"]) --
this module never invents a chip list. Requests are sort=price_asc and
pagination is capped at MAX_PAGES_PER_GPU pages per slug, so a book wider
than the cap records its cheapest window with a loud partial_errors note
(page.total keeps the full book size honest in book_stats).

Identity and honesty pins (each exists because the live surface could
drift into silently lying):

  - row attribution: a recorded row's own links.gpu URL must end with
    "/gpu/<queried slug>" -- the aggregator's structured claim that the row
    belongs to the queried chip. Mismatch = skipped + counted; a non-empty
    fetch where ZERO rows survive raises (the gpu filter stopped
    discriminating).
  - currency: meta.price_unit must equal the openapi const "USD per GPU
    hour" before price_per_gpu_hour is ever read as USD; a changed unit
    raises. A row carrying a native currency + price_native is recorded
    NATIVELY (price_usd_gpu_hr stays None; the aggregator's own USD
    conversion rides in extra.aggregator_usd_per_gpu_hr) -- FX embedded by
    the aggregator must never masquerade as a USD list price. A non-ISO
    currency string records as "UNKNOWN", never assumed USD.
  - liveness: meta.catalog_status must be "live_or_cached"; the API's
    documented "seed_fallback" state (canned bootstrap data) raises --
    seeded rows are not observations of anything.
  - pagination: page.next_offset must be null or strictly advance past the
    current offset (no loops); listing ids are deduped across pages so a
    book shifting under sort=price_asc between fetches does not double-record
    a row. Ids are only stable WITHIN one aggregator book generation (the
    catalog refreshes about hourly and reassigns ids wholesale -- verified
    live 2026-08-22), so the dedup guards the seconds between page fetches,
    never day-over-day identity.
  - basis: price_per_gpu_hour is the aggregator's ALREADY-PER-GPU figure --
    raw_value carries it verbatim (raw_unit says per_gpu_hr, so raw ==
    price directly; basis multiplication applies to per-instance raws
    only) and gpu_count_basis carries the row's stated gpu_count, the
    instance size the aggregator normalized by. A row without a positive
    integer gpu_count is skipped, never given a guessed basis.
  - tiers: billing_type maps fail-closed (on_demand/spot/reserved/monthly
    only); an unmapped label is skipped + counted, never guessed into a
    tier.

Whole-book availability count (added 2026-08-25): after each slug's price
pages, one extra GET with the server-side filter ``available=true&limit=1``
records page.total as book_stats[slug]["available_total"] -- the exact
count of available-flagged listings across the WHOLE book, past the 2-page
truncation cap (verified live 2026-08-25: nvidia-b300 total 3 under
available=true; openapi documents the ``available`` boolean query param
and the nullable per-row field). The probe reuses parse_listings_page, so
the catalog_status/price_unit/pagination pins hold for the count too. It
is FAIL-OPEN BY RULING: a probe failure (or a probe body flunking the
envelope pins) records available_total null plus a partial_errors note and
must never fail the slug -- availability capture may not gate or alter
price collection. The count is book-generation-scoped like everything else
on this API: compare available_total/listings_total within one capture,
never diff across captures.

Per-slug failures land in partial_errors; the source raises only when no
slug produced rows and at least one genuinely failed (result() itself
covers the all-empty case). A slug with zero listings is a countable note,
not a failure (amd-mi325x's book is genuinely empty as of 2026-08-22).
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

# Up to (MAX_PAGES_PER_GPU + 1) * len(options.gpus) calls per capture (the
# +1 is the per-slug available_total probe); space them out of politeness
# to an unauthenticated public API (vast drew a live 429 from a rapid-fire
# burst on 2026-08-22 — don't find out this API's limit the same way).
# ~15s added for the current 10-slug config, trivially inside the
# per-source deadline.
REQUEST_SPACING_SECONDS = 0.5

SOURCE_ID = "computepulse"

URL_BASE = "https://compute-pulse.com/api/listings"
# 2 pages x limit<=200 = at most 400 cheapest rows per slug; the H100/A100
# books already exceed this (1000+ listings) so the overflow note is a
# steady-state fact, not a rare warning.
MAX_PAGES_PER_GPU = 2
# The API's own page clamp (openapi: limit maximum 200). options above it
# would 400, options below it just fetch smaller pages.
API_MAX_LIMIT = 200
# openapi const for meta.price_unit -- the ONLY warrant for reading
# price_per_gpu_hour as USD.
PRICE_UNIT_PIN = "USD per GPU hour"
LIVE_CATALOG_STATUS = "live_or_cached"

# billing_type -> lane tier vocabulary, fail-closed (the openapi billing
# enum also names "custom" -- bespoke terms, not a comparable hourly print;
# it and anything else unrecognized is skipped + counted).
TIER_BY_BILLING = {
    "on_demand": "on-demand",
    "spot": "spot",
    "reserved": "reserved",
    "monthly": "monthly-commit",
}

_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def row_observation(
    row: Any, slug: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """One listing row -> (observation, None) or (None, skip_category).

    Skip categories are countable pin names, surfaced per slug in
    partial_errors -- a row is never guessed into the record.
    """
    if not isinstance(row, dict):
        return None, "malformed_row"
    links = row.get("links")
    gpu_link = str(links.get("gpu") or "") if isinstance(links, dict) else ""
    if not gpu_link.endswith(f"/gpu/{slug}"):
        # The aggregator's own attribution of the row to a chip page. If
        # this stops matching the queried slug, the server-side gpu filter
        # no longer discriminates for this row.
        return None, "identity_mismatch"
    identifier = str(row.get("gpu_model") or "").strip()
    if not identifier:
        return None, "unlabeled"
    provider = str(row.get("provider") or "").strip()
    if not provider:
        # An aggregator row without provider attribution is an
        # unattributable lead -- worthless to a consumer screening on
        # provenance.
        return None, "unattributed"
    gpu_count = row.get("gpu_count")
    if (
        not isinstance(gpu_count, int)
        or isinstance(gpu_count, bool)
        or gpu_count < 1
    ):
        return None, "bad_gpu_count"
    tier = TIER_BY_BILLING.get(row.get("billing_type"))
    if tier is None:
        return None, "unmapped_billing"

    listing_id = row.get("id")
    extra: Dict[str, Any] = {
        # Trust metadata -- these rows are LEADS, NOT PRINTS; consumers
        # must be able to screen on all of it.
        "provider": provider,
        "source_kind": row.get("source_kind"),
        "source_url": row.get("source_url"),
        "human_verified": row.get("human_verified"),
        "verification_level": row.get("verification_level"),
        "available": row.get("available"),
        # listing_id is only stable within one aggregator book generation
        # (ids are reassigned on the ~hourly catalog refresh -- verified
        # live 2026-08-22), so it keys the in-run cross-page dedup, never
        # day-over-day identity; the aggregator's own observation time
        # bounds how stale the lead already was.
        "listing_id": str(listing_id) if listing_id is not None else None,
        "aggregator_fetched_at": row.get("fetched_at"),
        "source_updated_at": row.get("source_updated_at"),
    }

    usd = row.get("price_per_gpu_hour")
    native = row.get("price_native")
    currency_raw = row.get("currency")
    code = str(currency_raw).strip().upper() if currency_raw is not None else ""

    if code and code != "USD" and _positive_number(native):
        # Native-currency listing: record the provider's own quoted amount
        # natively; the aggregator's USD conversion is ITS number (FX at
        # aggregation time), kept visible in extra but never recorded as a
        # USD list price.
        if _CURRENCY_CODE_RE.match(code):
            currency = code
            raw_unit = f"{code.lower()}_per_gpu_hr_aggregator_stated"
        else:
            currency = "UNKNOWN"
            raw_unit = "native_per_gpu_hr_aggregator_stated"
            extra["currency_raw"] = currency_raw
        if _positive_number(usd):
            extra["aggregator_usd_per_gpu_hr"] = usd
        price = float(native)
        raw_value = str(native)
    elif _positive_number(usd):
        # USD warranted by the meta.price_unit pin (checked per page before
        # any row is parsed), never assumed.
        currency = "USD"
        price = float(usd)
        raw_value = str(usd)
        raw_unit = "usd_per_gpu_hr_aggregator_stated"
        if code and code != "USD":
            # Currency stated but no usable native amount: the USD figure
            # is still the aggregator's genuine statement; the provider's
            # currency stays visible so the FX embedding is auditable.
            extra["provider_currency"] = currency_raw
    else:
        return None, "unpriced"

    variant = row.get("gpu_variant")
    notes = (
        f"{provider} {tier} lead via compute-pulse aggregator, "
        f"{gpu_count}x listing"
    )
    if isinstance(variant, str) and variant.strip():
        extra["gpu_variant"] = variant
        notes += f" ({variant})"
    obs = observation(
        sku_identifier=identifier,
        price_per_gpu_hr=price,
        currency=currency,
        raw_value=raw_value,
        raw_unit=raw_unit,
        gpu_count_basis=gpu_count,
        tier=tier,
        region=str(row.get("region") or "?"),
        notes=notes,
        extra=extra,
    )
    vram = row.get("vram_gb")
    if _positive_number(vram):
        obs["memory_gb_label"] = vram
    return obs, None


def parse_listings_page(body: str, slug: str) -> Dict[str, Any]:
    """Validate one /api/listings page and apply every pin (pure).

    Returns {"observations", "skips", "raw_row_count", "total",
    "next_offset"}. Raises on envelope-level pin failures (shape, price
    unit, seed fallback, pagination shape) -- those poison the whole page,
    not single rows.
    """
    doc = json.loads(body)
    if not isinstance(doc, dict):
        raise RuntimeError(
            f"computepulse {slug}: listings response is not a JSON object "
            "-- API shape changed"
        )
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"computepulse {slug}: listings response lost its meta block "
            "-- API shape changed"
        )
    status = meta.get("catalog_status")
    if status != LIVE_CATALOG_STATUS:
        raise RuntimeError(
            f"computepulse {slug}: catalog_status is {status!r}, not "
            f"{LIVE_CATALOG_STATUS!r} -- seeded/unknown fallback data is "
            "not a live observation; refusing to record"
        )
    price_unit = meta.get("price_unit")
    if price_unit != PRICE_UNIT_PIN:
        raise RuntimeError(
            f"computepulse {slug}: meta.price_unit is {price_unit!r}, not "
            f"{PRICE_UNIT_PIN!r} -- refusing to interpret "
            "price_per_gpu_hour as USD"
        )
    data = doc.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            f"computepulse {slug}: listings response lost its data list "
            "-- API shape changed"
        )
    page = doc.get("page")
    if not isinstance(page, dict):
        raise RuntimeError(
            f"computepulse {slug}: listings response lost its page block "
            "-- API shape changed"
        )
    total = page.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise RuntimeError(
            f"computepulse {slug}: page.total is not a non-negative "
            "integer -- pagination shape changed"
        )
    next_offset = page.get("next_offset")
    if next_offset is not None and (
        not isinstance(next_offset, int)
        or isinstance(next_offset, bool)
        or next_offset < 1
    ):
        raise RuntimeError(
            f"computepulse {slug}: page.next_offset is neither null nor a "
            "positive integer -- pagination shape changed"
        )
    observations: List[Dict[str, Any]] = []
    skips: Dict[str, int] = {}
    for row in data:
        obs, skip = row_observation(row, slug)
        if obs is not None:
            observations.append(obs)
        else:
            skips[skip] = skips.get(skip, 0) + 1
    return {
        "observations": observations,
        "skips": skips,
        "raw_row_count": len(data),
        "total": total,
        "next_offset": next_offset,
    }


def dedup_new_observations(
    observations: List[Dict[str, Any]], seen_ids: Set[str]
) -> Tuple[List[Dict[str, Any]], int]:
    """Cross-page dedup by listing id (pure; mutates seen_ids). Ids are
    stable within one book generation but reassigned on the aggregator's
    ~hourly refresh, and sort=price_asc pages are fetched seconds apart --
    so this guards the common replay (a book shifting a row across the
    page boundary), while a refresh landing exactly between pages could
    still slip a duplicate or cost one lead. Rows without an id are kept
    (losing a real lead is worse than a rare duplicate)."""
    new: List[Dict[str, Any]] = []
    dups = 0
    for obs in observations:
        lid = (obs.get("extra") or {}).get("listing_id")
        if lid is not None:
            if lid in seen_ids:
                dups += 1
                continue
            seen_ids.add(lid)
        new.append(obs)
    return new, dups


def _validated_options(
    options: Optional[Dict[str, Any]],
) -> Tuple[List[str], int]:
    """Fail closed: this collector never invents a chip list -- the
    queried slugs are operational config (config/raw_observatory.json)."""
    if not isinstance(options, dict):
        raise RuntimeError(
            "computepulse: options missing -- config/raw_observatory.json "
            "must supply options.gpus and options.limit_per_gpu; refusing "
            "to guess a slug list"
        )
    slugs = options.get("gpus")
    if (
        not isinstance(slugs, list)
        or not slugs
        or not all(isinstance(s, str) and s.strip() for s in slugs)
    ):
        raise RuntimeError(
            "computepulse: options.gpus must be a non-empty list of "
            "compute-pulse gpu slug strings"
        )
    if len(set(slugs)) != len(slugs):
        raise RuntimeError(
            "computepulse: options.gpus contains duplicates -- a "
            "duplicated slug would double-record its book"
        )
    limit = options.get("limit_per_gpu")
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= API_MAX_LIMIT
    ):
        raise RuntimeError(
            "computepulse: options.limit_per_gpu must be an integer in "
            f"1..{API_MAX_LIMIT} (the API's own page clamp)"
        )
    return list(slugs), limit


def _fetch_available_total(slug: str, timeout: float) -> int:
    """Whole-book available-flagged count for one slug: page.total under the
    server-side ``available=true`` filter (limit=1 -- only the envelope is
    wanted; the single row is discarded). Runs through parse_listings_page
    so the count inherits the catalog_status/price_unit/pagination pins.
    Raises on any failure; the CALLER converts that into a fail-open
    partial_errors note -- availability must never gate price collection."""
    query = urllib.parse.urlencode({"gpu": slug, "available": "true", "limit": 1})
    page = parse_listings_page(fetch(f"{URL_BASE}?{query}", timeout=timeout), slug)
    return page["total"]


def _fetch_slug_book(
    slug: str, limit: int, timeout: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int], Optional[str]]:
    """One slug's recorded book: up to MAX_PAGES_PER_GPU price_asc pages,
    deduped by listing id, plus the whole-book available_total probe.
    Returns (observations, stats, skips, avail_error) -- avail_error is the
    fail-open partial_errors note when the probe failed, else None."""
    observations: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    skips: Dict[str, int] = {}
    dups = 0
    raw_rows = 0
    offset = 0
    pages_fetched = 0
    total = 0
    truncated = False
    while True:
        if pages_fetched > 0:
            time.sleep(REQUEST_SPACING_SECONDS)
        query = urllib.parse.urlencode(
            {"gpu": slug, "sort": "price_asc", "limit": limit, "offset": offset}
        )
        page = parse_listings_page(
            fetch(f"{URL_BASE}?{query}", timeout=timeout), slug
        )
        pages_fetched += 1
        raw_rows += page["raw_row_count"]
        total = page["total"]
        for cat, count in page["skips"].items():
            skips[cat] = skips.get(cat, 0) + count
        new_obs, page_dups = dedup_new_observations(
            page["observations"], seen_ids
        )
        dups += page_dups
        observations.extend(new_obs)
        next_offset = page["next_offset"]
        if next_offset is None:
            break
        if next_offset <= offset:
            raise RuntimeError(
                f"computepulse {slug}: next_offset {next_offset} does not "
                f"advance past offset {offset} -- pagination shape changed; "
                "refusing to loop"
            )
        if pages_fetched >= MAX_PAGES_PER_GPU:
            truncated = True
            break
        offset = next_offset
    if dups:
        skips["duplicate_listing_id"] = dups
    # Whole-book availability count -- one extra spaced GET per slug,
    # FAIL-OPEN: a broken probe records null + a note, never a slug failure
    # (availability capture may not dark the price lane -- ruling).
    avail_error: Optional[str] = None
    time.sleep(REQUEST_SPACING_SECONDS)
    try:
        available_total: Optional[int] = _fetch_available_total(slug, timeout)
    except Exception as exc:  # noqa: BLE001 -- fail-open by ruling; note carries the cause
        available_total = None
        avail_error = (
            f"{slug}: available_total probe failed "
            f"({type(exc).__name__}: {exc}) -- whole-book availability "
            "count missing this capture; price rows unaffected"
        )
    stats = {
        "listings_total": total,
        "available_total": available_total,
        "pages_fetched": pages_fetched,
        "raw_rows": raw_rows,
        "rows_recorded": len(observations),
        "fetch_truncated": truncated,
    }
    return observations, stats, skips, avail_error


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    slugs, limit = _validated_options(options)
    rows: List[Dict[str, Any]] = []
    partial: List[str] = []
    failures: List[str] = []
    book_stats: Dict[str, Dict[str, Any]] = {}
    for i, slug in enumerate(slugs):
        try:
            if i > 0:
                time.sleep(REQUEST_SPACING_SECONDS)
            slug_rows, stats, skips, avail_error = _fetch_slug_book(
                slug, limit, timeout
            )
        except Exception as exc:  # noqa: BLE001 -- one slug's feed must not hide the rest
            failures.append(f"{slug}: {type(exc).__name__}: {exc}")
            continue
        book_stats[slug] = stats
        if avail_error is not None:
            # Fail-open by ruling: the availability probe's failure is a
            # note, never a slug failure.
            partial.append(avail_error)
        # Per-slug one-liner (config-derived strings and counts only --
        # remote strings are never printed raw from this module).
        avail = stats["available_total"]
        print(
            f"  computepulse {slug}: {stats['raw_rows']} rows over "
            f"{stats['pages_fetched']} page(s), recording "
            f"{stats['rows_recorded']} of {stats['listings_total']} listings"
            + ("" if avail is None else f" ({avail} book-wide available)")
        )
        if skips:
            partial.append(
                f"{slug}: skipped "
                + ", ".join(
                    f"{count} {cat}" for cat, count in sorted(skips.items())
                )
                + " row(s) (fail-closed pins; see module docstring)"
            )
        if stats["raw_rows"] == 0:
            # NOT an error: an empty aggregator book is a real state
            # (amd-mi325x as of 2026-08-22) -- but an unknown/typo slug
            # looks identical, so the note names both readings.
            partial.append(
                f"{slug}: zero listings in the aggregator catalog (empty "
                "book or unknown slug, not a failure)"
            )
            continue
        if not slug_rows:
            failures.append(
                f"{slug}: {stats['raw_rows']} listings fetched but ZERO "
                "survived the identity and honesty pins -- the listings "
                "API shape or its gpu filter changed; refusing to guess"
            )
            continue
        rows.extend(slug_rows)
        if stats["fetch_truncated"]:
            partial.append(
                f"{slug}: book wider than the {MAX_PAGES_PER_GPU}-page cap "
                f"-- recorded cheapest {stats['rows_recorded']} of "
                f"{stats['listings_total']} listings (sort price_asc)"
            )
    if not rows and failures:
        # Every slug either failed or printed nothing, and at least one
        # genuinely failed: surface the real causes instead of the generic
        # parsed-nothing message.
        raise RuntimeError(
            f"computepulse: no slug produced observations and "
            f"{len(failures)} slug(s) failed: " + "; ".join(failures)
        )
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL_BASE,
        observations=rows,
        first_party=False,  # aggregator republishing providers' prices -- leads, not prints
        partial_errors=(failures + partial) or None,
        book_stats=book_stats or None,
    )
