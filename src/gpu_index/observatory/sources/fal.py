# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""fal -- fal.ai/pricing, Serverless & Compute Pricing GPU table, both columns.

One fetch of the pricing page (server-rendered Next.js/Vercel HTML,
verified live 2026-08-22). The document carries THREE <table> blocks in
near-identical markup with NO table ids: the GPU rental table plus two
model-API tables pricing per OUTPUT unit ($0.05 per second of video,
$0.03 per image -- not GPU rental at all). Only the header-cell tuple
discriminates, so table selection is pinned to the exact header
('GPU', 'VRAM', 'List Price', 'As low as') and requires EXACTLY ONE
match: zero means the page reshaped (or moved to client-side rendering,
which would empty the server HTML -- fails closed by construction), two
means a lookalike appeared and attribution would be ambiguous either way.

Two price columns per row -- the load-bearing honesty split:

  - 'List Price' -> the bookable serverless rate (tier "serverless");
  - 'As low as'  -> a contact-sales marketing floor, recorded under its
    own tier label "serverless_as_low_as". The page's own prose quotes it
    ('as low as $1.89/hr for H100'); it must NEVER print as the list rate
    or every downstream reader sees the floor as the market price.

Row fence (fail closed, per row): exactly 4 cells, cell[1] matching
digits+'GB', cells 2 and 3 matching the exact '$D.DD/h' shape, and no two
rows claiming the same GPU label. This five-row first-party rate card has
no legitimate row variants today, so ANY non-conforming row (an unpriced/
contact-us cell, a currency change, an added column) refuses the whole
capture rather than being skipped -- on a surface this small a deviation
is a reshape signal, and skipping would silently drop a chip's history.

Other traps this module is shaped around:

  - DOUBLE BYTES: every price string appears twice in the document (the
    HTML table + a Next.js RSC flight duplicate with row keys GPU-B300,
    GPU-B200, ...) -- parsing is scoped to the one pinned <table>; nothing
    ever regex-counts the whole page;
  - H100 and H200 share the same $4.50/h list price live -- rows are keyed
    by label, never deduped by price;
  - bare 'H100' (80GB) does not say SXM/PCIe -- the catalog policy for
    unqualified H100 labels applies downstream; the collector records the
    label exactly as published plus the row's own VRAM cell;
  - prices are USD ($ glyph, US company) per SINGLE GPU per hour (each
    row lists one GPU and its VRAM; the page prose quotes $/hr per GPU),
    so gpu_count_basis=1 and price*basis reproduces the raw cell by
    construction. fal bills per-second granularity in practice but
    publishes /h rates -- raw_value keeps the published figure exactly.
"""

from __future__ import annotations

import html as html_lib
import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "fal"

URL = "https://fal.ai/pricing"

# One global rate card -- the table publishes no region column.
_REGION = "unspecified"

# The ONLY discriminator between the GPU rental table and the two
# model-API lookalikes (header Model | Unit | Price | Output per $1) --
# the surrounding markup is class-identical and carries no table ids.
_EXPECTED_HEADER = ("GPU", "VRAM", "List Price", "As low as")

_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_VRAM_RE = re.compile(r"^(\d+)GB$")
_PRICE_RE = re.compile(r"^\$(\d+(?:\.\d{1,2})?)/h$")


def _text(cell_html: str) -> str:
    """Tag-stripped, entity-unescaped, whitespace-collapsed cell text."""
    return _WS_RE.sub(" ", html_lib.unescape(_TAG_RE.sub(" ", cell_html))).strip()


def _table_rows(table_html: str) -> List[List[str]]:
    return [
        [_text(c) for c in _CELL_RE.findall(row)]
        for row in _ROW_RE.findall(table_html)
    ]


def _pinned_table_rows(page: str) -> List[List[str]]:
    """The GPU table's data rows, selected by the exact header tuple.

    Fail-closed identity pin: exactly one table on the page may carry the
    pinned header. The two model-API tables are excluded here -- their
    header tuple differs; nothing else about their markup does.
    """
    matches = [
        rows
        for tbl in _TABLE_RE.findall(page)
        if (rows := _table_rows(tbl)) and tuple(rows[0]) == _EXPECTED_HEADER
    ]
    if not matches:
        raise RuntimeError(
            f"fal: no table with the pinned header {_EXPECTED_HEADER!r} -- "
            "page reshaped or moved to client-side rendering (the model-API "
            "tables price per output unit, not GPU rental, and are excluded "
            "by this pin); refusing to extract"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"fal: {len(matches)} tables match the pinned header "
            f"{_EXPECTED_HEADER!r} (need exactly 1) -- a duplicate/lookalike "
            "render would double-print every row; refusing to pick one"
        )
    return matches[0][1:]


def _fenced_row(cells: List[str]) -> Tuple[str, int, str, str]:
    """(label, vram_gb, list_cell, as_low_as_cell), fail-closed.

    Any non-conforming row refuses the whole capture: this five-row rate
    card has no legitimate row variants, so a deviation (contact-us cell,
    currency change, added column) is a reshape signal, never a skip.
    """
    if len(cells) != 4:
        raise RuntimeError(
            f"fal: GPU table row {cells!r} has {len(cells)} cells, expected "
            "4 -- table reshaped; refusing to extract"
        )
    label, vram_text, list_text, low_text = cells
    if not label:
        raise RuntimeError(
            "fal: GPU table row with an empty label cell -- a price we "
            "cannot attribute to a chip; refusing to extract"
        )
    vram_match = _VRAM_RE.match(vram_text)
    if not vram_match:
        raise RuntimeError(
            f"fal: row {label!r}: VRAM cell {vram_text!r} misses the "
            "digits+GB pin -- column order/format changed; refusing to "
            "attribute the price cells"
        )
    for column, text in (("List Price", list_text), ("As low as", low_text)):
        if not _PRICE_RE.match(text):
            raise RuntimeError(
                f"fal: row {label!r}, {column} column: cell {text!r} misses "
                "the exact $D.DD/h pin -- unpriced/contact-us row or a "
                "currency/format change; refusing to guess (fail-closed row "
                "fence: on this five-row rate card any deviation is a "
                "reshape signal)"
            )
    return label, int(vram_match.group(1)), list_text, low_text


def parse_fal_pricing(page: str) -> List[Dict[str, Any]]:
    """Pure parse of the pricing page body -> observations."""
    data_rows = _pinned_table_rows(page)
    if not data_rows:
        raise RuntimeError(
            "fal: pinned GPU table present but holds zero data rows -- "
            "listings pulled or table emptied; refusing to record a silent "
            "hole"
        )
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for cells in data_rows:
        label, vram_gb, list_text, low_text = _fenced_row(cells)
        if label in seen:
            raise RuntimeError(
                f"fal: GPU label {label!r} appears in more than one table "
                "row -- ambiguous identity (rows are keyed by label, never "
                "deduped by price); refusing to extract"
            )
        seen.add(label)
        for tier, text, notes, extra in (
            (
                "serverless",
                list_text,
                (
                    f"Serverless & Compute Pricing list rate, {vram_gb}GB "
                    "VRAM, USD per GPU per hour (fal bills per-second "
                    "granularity; the published rate is /h)"
                ),
                {"column": "List Price", "vram_label": cells[1]},
            ),
            (
                "serverless_as_low_as",
                low_text,
                (
                    "'As low as' contact-sales floor -- a marketing "
                    "minimum, not the bookable list rate (that is this "
                    "row's tier=serverless print)"
                ),
                {
                    "column": "As low as",
                    "pricing_basis": "as_low_as_floor",
                    "vram_label": cells[1],
                },
            ),
        ):
            obs = observation(
                sku_identifier=label,
                price_per_gpu_hr=float(_PRICE_RE.match(text).group(1)),
                raw_value=text,
                tier=tier,
                region=_REGION,
                notes=notes,
                extra=extra,
            )
            obs["memory_gb_label"] = vram_gb
            rows.append(obs)
    return rows


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    return result(
        SOURCE_ID, method="html", url=URL, observations=parse_fal_pricing(body)
    )
