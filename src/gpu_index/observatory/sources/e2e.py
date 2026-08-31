# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""E2E Networks - USD pricing table + JSON-LD offer catalog, every GPU row.

One fetch of the pricing page (?region=us), two surfaces recorded:

  - the server-rendered USD table (Item/vRAM/vCPUs/RAM,GB/Hourly/Monthly/
    Annually): EVERY GPU row, three tiers labeled - hourly on-demand,
    monthly-commit, and the annual-commitment column recorded as
    'reserved' (the closest lane-wide label). Monthly/annual prints keep
    the raw figure verbatim and normalize to per-GPU-hr with the same
    hours conventions the basket lane's e2e cross-check proved
    (~730 h/month, 8760 h/year);
  - the page's JSON-LD OfferCatalog: the same GPUs' INR hourly list
    prices, recorded natively (currency honesty - price_usd_gpu_hr stays
    null; 671 INR/hr B200 vs $6.99 is the ~96 INR/USD list-to-list gap
    between E2E's two price lists, not an error). CPU instance offers in
    the same catalog are not GPU rentals and are excluded by the explicit
    category == 'GPU Cloud Instance' fence.

Identity pins (the basket lane's single-B200 pins, generalized per row;
each exists because a lookalike is really on the page):

  - the pricing table is selected by its EXACT header row - zero or
    multiple matching tables raise. The page's flight payload carries
    escaped copies of the nav GPU labels AND the whole offer catalog, so
    cell soup outside the one real <table> / the real ld+json <script>
    tags is never scanned;
  - per row, the vRAM/vCPUs/RAM integer spec triple must sit between the
    label and the price cells, and every COMPUTABLE same-currency pair of
    the monthly/hourly (~730x), annual/monthly (~10.5x) and annual/hourly
    ratios must land in band - a row failing a computable cross-check
    goes to partial_errors and is skipped whole, never guessed (column
    reorder protection). A row with a single priced cell (contact-us
    siblings) records that cell on the header+spec pins alone, with the
    unpriced cells counted in partial_errors;
  - price cells parse only from an explicit currency symbol ($ or the
    rupee sign, recorded as USD/INR respectively); any other cell text is
    skipped into partial_errors. JSON-LD offers record priceCurrency
    verbatim - a missing one records as UNKNOWN, never assumed USD;
  - the per-GPU basis is the page's own footnote ('Prices are per
    GPU-hour', right under the table); the table also has a 1x/2x/4x/8x
    multiplier control, and a multiplied default view would pass the
    header pin and every ratio band (uniform scaling), so the footnote
    disappearing raises rather than risking an 8x per-GPU print.

Verified live 2026-08-22: 10 GPU rows in the USD table (B200 / H200 /
H100 / RTX PRO 6000 / A100 80GB / A100 40GB / A40 / L40S / A30 / L4), all
three columns $-priced; the JSON-LD catalog quotes the same 10 parts in
INR per hour. The two same-label A100 rows are told apart by the vRAM
spec cell (memory_gb_label 80 vs 40).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "e2e"

URL = "https://www.e2enetworks.com/pricing?region=us"

_LD_SCRIPT_RE = re.compile(
    r"<script[^>]*application/ld\+json[^>]*>(.*?)</script>", re.S
)
_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_INT_RE = re.compile(r"\d+")

# Exact header of the USD pricing table - the table-selection pin AND the
# column-order pin in one: prices are only ever read from the single table
# whose columns are exactly these, in this order.
USD_TABLE_HEADER = (
    "Item",
    "vRAM",
    "vCPUs",
    "RAM, GB",
    "Hourly/On-Demand",
    "Monthly",
    "Annually",
)

# U+20B9 is the Indian rupee sign - E2E's INR surfaces print it. Kept as
# an escape: source files stay ASCII-only.
_MONEY_RE = re.compile("^([$\u20b9])\\s*([0-9][0-9,]*(?:\\.[0-9]+)?)$")
_CURRENCY_BY_SYMBOL = {"$": "USD", "\u20b9": "INR"}

HOURS_PER_MONTH = 730.0  # 8760/12 - the ratio the basket lane cross-checks
HOURS_PER_YEAR = 8760.0

# Dimensionless price-ratio bands. monthly/hourly inherits the basket
# lane's proven band; the annual bands bound E2E's observed annual
# discounts (annual/monthly 9.6-11.5 live 2026-08-22) with headroom.
MONTHLY_OVER_HOURLY_BAND = (400.0, 1100.0)
ANNUAL_OVER_MONTHLY_BAND = (6.0, 14.0)
ANNUAL_OVER_HOURLY_BAND = (3000.0, 13000.0)

# (row cell index, tier label, hours divisor, raw-unit suffix, note)
_TIER_COLUMNS = (
    (4, "on-demand", 1.0, "per_gpu_hr", "hourly on-demand cell"),
    (5, "monthly-commit", HOURS_PER_MONTH, "per_gpu_month",
     "monthly cell normalized at 730 h/month"),
    (6, "reserved", HOURS_PER_YEAR, "per_gpu_year",
     "annual-commitment cell normalized at 8760 h/year"),
)

REGION = "IN (India DCs)"

# The page's own per-GPU basis statement - the footnote right under the
# table. A 1x/2x/4x/8x multiplier control sits above the SAME table; a
# server-side default flip to a multiplied view would pass the header pin
# and every ratio band (uniform scaling), so the per-GPU claim in
# raw_unit/price_per_gpu_hr is fenced on the page still saying it.
PER_GPU_FOOTNOTE = "Prices are per GPU-hour"

_VRAM_RE = re.compile(r"(\d+)\s*GB\s+VRAM", re.I)


def _strip(fragment: str) -> str:
    return _TAG_RE.sub("", fragment.replace("<!-- -->", "")).strip()


def _parse_money(cell: str) -> Optional[Tuple[str, float]]:
    m = _MONEY_RE.match(cell)
    if not m:
        return None
    return _CURRENCY_BY_SYMBOL[m.group(1)], float(m.group(2).replace(",", ""))


def _find_pricing_table_rows(html: str) -> List[List[str]]:
    """Data rows (as stripped td-cell lists) of the ONE table whose header
    row is exactly USD_TABLE_HEADER. Zero or multiple matches raise -
    with escaped table lookalikes in the flight payload, guessing between
    candidates is how the wrong cell becomes a price."""
    matches: List[List[str]] = []
    for table in _TABLE_RE.findall(html):
        trs = _TR_RE.findall(table)
        if not trs:
            continue
        header = tuple(_strip(c) for c in _CELL_RE.findall(trs[0]))
        if header == USD_TABLE_HEADER:
            matches.append(trs[1:])
    if len(matches) != 1:
        raise RuntimeError(
            f"e2e: expected exactly one pricing table with header "
            f"{list(USD_TABLE_HEADER)}, found {len(matches)} - page "
            "reshaped, refusing to guess which cells are prices"
        )
    return [[_strip(c) for c in _CELL_RE.findall(tr)] for tr in matches[0]]


def _ratio_in_band(
    numer: Optional[Tuple[str, float]],
    denom: Optional[Tuple[str, float]],
    band: Tuple[float, float],
) -> Optional[bool]:
    """None when the pair isn't computable (missing cell or mixed
    currency); the cross-check suite only judges computable pairs."""
    if numer is None or denom is None or numer[0] != denom[0]:
        return None
    if denom[1] <= 0:
        return False
    return band[0] <= numer[1] / denom[1] <= band[1]


def _row_observations(
    cells: List[str], partial_errors: List[str]
) -> List[Dict[str, Any]]:
    label = cells[0] if cells else ""
    if len(cells) != 7:
        partial_errors.append(
            f"e2e: table row {label!r} has {len(cells)} cells, expected 7 "
            "- skipped"
        )
        return []
    specs = cells[1:4]
    if (
        not label
        or _parse_money(label) is not None
        or _INT_RE.fullmatch(label)
        or not all(_INT_RE.fullmatch(c) for c in specs)
    ):
        partial_errors.append(
            f"e2e: table row {label!r} failed the label + vRAM/vCPUs/RAM "
            f"spec-triple pin (specs {specs!r}) - skipped"
        )
        return []
    vram, vcpus, ram = (int(c) for c in specs)
    prices = {idx: _parse_money(cells[idx]) for idx, *_ in _TIER_COLUMNS}
    checks = (
        ("monthly/hourly", _ratio_in_band(prices[5], prices[4], MONTHLY_OVER_HOURLY_BAND)),
        ("annual/monthly", _ratio_in_band(prices[6], prices[5], ANNUAL_OVER_MONTHLY_BAND)),
        ("annual/hourly", _ratio_in_band(prices[6], prices[4], ANNUAL_OVER_HOURLY_BAND)),
    )
    failed = [name for name, ok in checks if ok is False]
    if failed:
        partial_errors.append(
            f"e2e: table row {label!r} failed cross-check(s) {failed} on "
            f"price cells {cells[4:7]!r} - possible column reorder, whole "
            "row skipped"
        )
        return []
    rows: List[Dict[str, Any]] = []
    for idx, tier, divisor, unit_suffix, note in _TIER_COLUMNS:
        parsed = prices[idx]
        if parsed is None:
            partial_errors.append(
                f"e2e: table row {label!r} {tier} cell {cells[idx]!r} is "
                "not a priced cell - that tier skipped"
            )
            continue
        currency, value = parsed
        obs = observation(
            sku_identifier=label,
            price_per_gpu_hr=value / divisor,
            currency=currency,
            raw_value=cells[idx],
            raw_unit=f"{currency.lower()}_{unit_suffix}",
            tier=tier,
            region=REGION,
            notes=(
                f"{vram}GB vRAM / {vcpus} vCPU / {ram}GB RAM row of the "
                f"USD pricing table; {note}"
            ),
            extra={"surface": "usd_table", "vcpus": vcpus, "ram_gb": ram},
        )
        obs["memory_gb_label"] = vram
        rows.append(obs)
    return rows


def _iter_offers(node: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(node, dict):
        if node.get("@type") == "Offer":
            yield node
            return
        for value in node.values():
            yield from _iter_offers(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_offers(value)


def _jsonld_observations(
    html: str, partial_errors: List[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    blocks = _LD_SCRIPT_RE.findall(html)
    for blob in blocks:
        try:
            data = json.loads(blob)
        except ValueError:
            partial_errors.append(
                "e2e: unparseable application/ld+json block - skipped"
            )
            continue
        for offer in _iter_offers(data):
            item = offer.get("itemOffered") or {}
            category = offer.get("category") or item.get("category")
            if category != "GPU Cloud Instance":
                continue  # CPU instances etc. - not GPU rentals
            name = str(offer.get("name") or item.get("name") or "").strip()
            spec = offer.get("priceSpecification") or {}
            price = offer.get("price", spec.get("price"))
            unit = spec.get("unitText")
            currency = str(
                offer.get("priceCurrency")
                or spec.get("priceCurrency")
                or "UNKNOWN"
            )
            if not name:
                partial_errors.append(
                    "e2e: unnamed GPU offer in ld+json - skipped"
                )
                continue
            if unit != "per hour":
                partial_errors.append(
                    f"e2e: ld+json GPU offer {name!r} unitText {unit!r} != "
                    "'per hour' - cannot claim an hourly print, skipped"
                )
                continue
            if (
                not isinstance(price, (int, float))
                or isinstance(price, bool)
                or price <= 0
            ):
                partial_errors.append(
                    f"e2e: ld+json GPU offer {name!r} has no positive "
                    "numeric price - skipped"
                )
                continue
            desc = str(item.get("description") or "")
            extra: Dict[str, Any] = {"surface": "jsonld_offer_catalog"}
            if offer.get("url"):
                extra["url"] = offer["url"]
            if offer.get("availability"):
                extra["availability"] = offer["availability"]
            obs = observation(
                sku_identifier=name,
                price_per_gpu_hr=float(price),
                currency=currency,
                raw_value=str(price),
                raw_unit=f"{currency.lower()}_per_gpu_hr",
                tier="on-demand",
                region=REGION,
                notes="hourly list price from the page's JSON-LD offer "
                "catalog" + (f"; {desc}" if desc else ""),
                extra=extra,
            )
            vram_m = _VRAM_RE.search(desc)
            if vram_m:
                obs["memory_gb_label"] = int(vram_m.group(1))
            rows.append(obs)
    if not rows:
        partial_errors.append(
            f"e2e: JSON-LD offer surface yielded zero GPU offers "
            f"({len(blocks)} ld+json block(s) present) - secondary surface "
            "dark or reshaped"
        )
    return rows


def parse_e2e(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse over the fetched page: (observations, partial_errors).

    The USD table is the primary surface - header pin present but zero
    pinned rows raises (whole-surface reshape must never demote silently
    to the JSON-LD list alone). The JSON-LD surface going dark is a
    partial_error, not a failure."""
    partial_errors: List[str] = []
    table_rows: List[Dict[str, Any]] = []
    for cells in _find_pricing_table_rows(html):
        table_rows.extend(_row_observations(cells, partial_errors))
    if PER_GPU_FOOTNOTE not in html:
        raise RuntimeError(
            f"e2e: per-GPU basis footnote {PER_GPU_FOOTNOTE!r} is gone - "
            "with the 1x/2x/4x/8x multiplier control on this table, the "
            "prices can no longer be claimed per-GPU; refusing to record"
        )
    if not table_rows:
        raise RuntimeError(
            "e2e: pricing table matched the header pin but yielded zero "
            f"pinned GPU rows (row errors: {partial_errors!r}) - page "
            "reshaped"
        )
    return table_rows + _jsonld_observations(html, partial_errors), partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_e2e(html)
    return result(
        SOURCE_ID,
        method="html-table+jsonld",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
