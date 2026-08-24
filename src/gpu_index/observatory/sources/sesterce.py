# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Sesterce -- www.sesterce.com homepage, the #pricing live-offers table.

One fetch of the homepage (Next.js ISR, server-regenerated ~every 5 min and
self-labeled 'CACHED 5M'; the page-embedded render timestamp is recorded in
extra so a reader can see the staleness). The id="pricing" section carries
Sesterce's only public price surface: a terminal-styled table of live
on-demand offers from the cloud.sesterce.com console -- six NVIDIA SKUs
live 2026-08-22 (B300/B200/H200/H100/A100/L40S). cloud.sesterce.com/pricing
itself is a 404 (console client route) and docs.sesterce.com publishes no
pricing API, so this table is the record; reserved/private-cloud pricing is
quote-only and never prints here.

FAIL-CLOSED PINS, in order inside the section slice:

  1. the id="pricing" section anchor, required unique page-wide;
  2. the literal table-footer marker 'LIVE OFFERS . CLOUD.SESTERCE.COM .
     USD . ON-DEMAND' (middle-dot separated, sits right after </tbody>) --
     the currency AND tier pin: a flip to EUR (Sesterce is a French
     operator) or a dropped ON-DEMAND must dark the source, never mislabel
     a print;
  3. the byte-exact <thead> (Model | From / GPU / hr | Available now |
     Listed GPUs | Live regions | Action) -- the unit-basis/column-order
     pin: prices are USD per GPU per hour, so gpu_count_basis stays 1 and
     price*basis reproduces the raw cell;
  4. exactly one <tbody> in the section, between thead and marker; rows
     parse from it ONLY.

Row honesty: the price column is a 'From / GPU / hr' FLOOR -- the lowest
across configs/regions, not a specific config (flagged in notes and
extra.price_basis); H100 and A100 blend 'SXM / PCIe' form factors under one
figure, so the bbg-vendor qualifier line rides in notes/extra and is never
split into per-form-factor prints. The availability columns (Available now
/ Listed GPUs / Live regions) are separate published stats recorded in
extra -- B300/B200 print 'Limited' with 0 live regions yet carry real
prices; availability is never a price qualifier. A price cell that bears
digits but misses the exact $D.DD shape raises (a currency/format change
is never guessed into USD); a row with no price cell at all is skipped and
counted in partial_errors.

Lookalike surfaces that must NEVER print (each verified live 2026-08-22):
the bbg-ticker marquee above the table duplicates every row TWICE (12
items, bare '$8.48' text); the RSC flight payload at the bottom of the page
carries the whole table a third time with RSC dollar-escaped prices
('$$8.48') that a naive page-wide price regex would catch; and the previous
section's static 'Sesterce CLI' marketing terminal shows a 4-chip capacity
strip whose figures conflict with the live table (stale marketing copy, no
currency/tier pin) -- deliberately skipped and counted in partial_errors.
Only rows from the pinned tbody are recorded.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "sesterce"

URL = "https://www.sesterce.com/"
# One global floor price per chip; region coverage is a published stat
# (extra.live_regions/total_regions), not a price axis.
REGION = "global"

_ANCHOR = 'id="pricing"'
# Segments are separated by the middle dot U+00B7 (escaped: source stays
# ASCII-only). Deliberately excludes the trailing '<middot> CACHED 5M' --
# a changed cache window is not a pricing-identity change.
_MARKER = (
    "LIVE OFFERS \u00b7 CLOUD.SESTERCE.COM \u00b7 USD \u00b7 ON-DEMAND"
)
_THEAD = (
    '<thead><tr><th scope="col">Model</th>'
    '<th scope="col">From / GPU / hr</th>'
    '<th scope="col">Available now</th>'
    '<th scope="col">Listed GPUs</th>'
    '<th scope="col">Live regions</th>'
    '<th scope="col" class="right"><span class="sr-only">Action</span></th>'
    "</tr></thead>"
)

_MODEL_RE = re.compile(r'<div class="bbg-model">([^<]+)</div>')
_VENDOR_RE = re.compile(r'<div class="bbg-vendor">([^<]+)</div>')
_PRICE_SPAN_RE = re.compile(r'<span class="bbg-price">([^<]*)</span>')
_PRICE_RE = re.compile(r"^\$(\d+\.\d{2})$")
_AVAIL_NOW_RE = re.compile(r'<span class="bbg-(flat|up|down)">([^<]*)</span>')
_LISTED_RE = re.compile(r'<td class="bbg-best">([^<]*)</td>')
_REGIONS_RE = re.compile(r'class="bbg-avail">(\d+)/(\d+) regions')
_GPUS_INT_RE = re.compile(r"^(\d+) GPUs$")
_RENDERED_AT_RE = re.compile(r'<span class="bloomberg__time">([^<]*)</span>')

# The static 'Sesterce CLI' marketing terminal's capacity strip (previous
# section) -- countable so its skip is visible in partial_errors and a
# count change (new lookalike surface) is visible in the record.
_CLI_STRIP_ATTR = 'class="scloud-capacity__price"'


def _require_once(hay: str, needle: str, what: str, where: str) -> int:
    count = hay.count(needle)
    if count != 1:
        raise RuntimeError(
            f"sesterce: {what} found {count}x in {where} (need exactly 1) "
            "-- page reshaped or a lookalike surface appeared; refusing to "
            "guess"
        )
    return hay.index(needle)


def _pricing_tbody(html: str) -> Tuple[str, str]:
    """(section, tbody inner) -- every identity pin enforced fail-closed."""
    start = _require_once(html, _ANCHOR, "#pricing section anchor", "page")
    end = html.find("</section>", start)
    if end < 0:
        raise RuntimeError(
            "sesterce: no </section> after the #pricing anchor -- section "
            "markup reshaped; refusing to guess the slice bounds"
        )
    section = html[start:end]
    i_marker = _require_once(
        section,
        _MARKER,
        "'LIVE OFFERS/USD/ON-DEMAND' currency+tier marker",
        "the #pricing section",
    )
    i_thead = _require_once(
        section, _THEAD, "pinned table header", "the #pricing section"
    )
    i_open = _require_once(
        section, "<tbody>", "<tbody> open tag", "the #pricing section"
    )
    i_close = _require_once(
        section, "</tbody>", "</tbody> close tag", "the #pricing section"
    )
    # Live layout 2026-08-22: thead -> tbody -> marker (the marker is the
    # table's FOOTER caption, sitting right after </tbody>).
    if not (i_thead < i_open < i_close < i_marker):
        raise RuntimeError(
            "sesterce: thead/tbody/marker out of expected order in the "
            "#pricing section -- refusing to guess which table is the live "
            "offers table"
        )
    return section, section[i_open + len("<tbody>") : i_close]


def parse_sesterce(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the homepage -> (observations, partial_errors)."""
    section, tbody = _pricing_tbody(html)
    rendered = _RENDERED_AT_RE.search(section)
    rendered_at = rendered.group(1) if rendered else None

    partial_errors: List[str] = []
    strip_chips = html.count(_CLI_STRIP_ATTR)
    if strip_chips:
        partial_errors.append(
            f"cli-demo capacity strip: {strip_chips} price chips outside "
            "the pinned #pricing live-offers table skipped -- static "
            "marketing widget (no currency/tier pin; figures conflict with "
            "the live table)"
        )

    rows: List[Dict[str, Any]] = []
    chunks = [c for c in tbody.split("<tr>") if c.strip()]
    for idx, chunk in enumerate(chunks):
        models = _MODEL_RE.findall(chunk)
        if len(models) > 1:
            raise RuntimeError(
                f"sesterce: table row {idx} carries {len(models)} model "
                "labels -- row markup reshaped (merged rows would "
                "misattribute prices); refusing to extract"
            )
        if not models:
            partial_errors.append(
                f"table row {idx}: no bbg-model label -- skipped, not "
                "guessed"
            )
            continue
        model = models[0].strip()
        price_cells = _PRICE_SPAN_RE.findall(chunk)
        if len(price_cells) > 1:
            raise RuntimeError(
                f"sesterce: row {model!r} carries {len(price_cells)} price "
                "cells -- row markup reshaped; refusing to pick one"
            )
        if not price_cells:
            partial_errors.append(
                f"row {model!r}: no price cell -- unpriced row skipped"
            )
            continue
        raw_price = price_cells[0].strip()
        match = _PRICE_RE.match(raw_price)
        if not match:
            raise RuntimeError(
                f"sesterce: row {model!r} price cell {raw_price!r} misses "
                "the exact $D.DD pin -- currency or format changed; "
                "refusing to guess (the USD marker pins today's currency, "
                "the cell must agree)"
            )
        price = float(match.group(1))

        vendor_match = _VENDOR_RE.search(chunk)
        vendor_line = vendor_match.group(1).strip() if vendor_match else None
        extra: Dict[str, Any] = {
            "price_basis": "from_floor",
            "console": "cloud.sesterce.com",
        }
        if vendor_line:
            extra["vendor_line"] = vendor_line
        avail = _AVAIL_NOW_RE.search(chunk)
        if avail:
            extra["availability_trend"] = avail.group(1)
            extra["available_now"] = avail.group(2).strip()
        listed = _LISTED_RE.search(chunk)
        if listed:
            listed_label = listed.group(1).strip()
            extra["listed_gpus_label"] = listed_label
            listed_int = _GPUS_INT_RE.match(listed_label)
            if listed_int:
                extra["listed_gpus"] = int(listed_int.group(1))
        regions = _REGIONS_RE.search(chunk)
        if regions:
            extra["live_regions"] = int(regions.group(1))
            extra["total_regions"] = int(regions.group(2))
        if rendered_at:
            extra["page_rendered_at"] = rendered_at

        rows.append(
            observation(
                sku_identifier=model,
                price_per_gpu_hr=price,
                raw_value=raw_price,
                tier="on-demand",
                region=REGION,
                notes=(
                    (f"{vendor_line}; " if vendor_line else "")
                    + "'From / GPU / hr' floor -- lowest across "
                    "configs/regions, not a specific config (USD on-demand "
                    "pinned by the section marker)"
                ),
                extra=extra,
            )
        )
    if not rows:
        raise RuntimeError(
            "sesterce: pinned #pricing table present but zero pinnable "
            f"rows ({len(chunks)} row chunks) -- table reshaped or "
            "listings pulled; refusing to record a silent hole"
        )
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_sesterce(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
