# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""TensorPool -- tensorpool.dev/pricing comparison grid, TensorPool column ONLY.

One fetch of the server-rendered pricing page (~31KB, no auth, verified
live 2026-08-22). The page renders the same price table TWICE: mobile
per-provider cards first, then the desktop comparison grid (Category |
GPU Type | Lambda Labs | TensorPool | Traditional Clouds). The grid is the
record; the mobile TensorPool card is re-parsed as an EQUALITY TRIPWIRE
only -- recording both would double-print every row, and the card lacks
the grid's /hr suffix so the two renderings need different regexes.

THE trap: every chip label appears three times per grid row -- the Lambda
Labs and 'Traditional Clouds (GCP, AWS, etc.)' columns carry the same
labels with different prices (H100 SXM: $3.29 Lambda / $1.99 TensorPool /
$14 Traditional), and B200 SXM is $4.99 in BOTH the Lambda and TensorPool
columns, so a mis-pinned parser can pass a B200 spot-check by accident.
Column identity is therefore double-pinned, fail-closed:

  - header cells 0-3 must read exactly Category | GPU Type | Lambda Labs |
    TensorPool, and cell 4 must start with 'Traditional Clouds';
  - the bg-[#B8E7F5] highlight class (TensorPool's own you-are-here paint)
    must sit on EXACTLY the TensorPool-column cell of the header and of
    every data row. The class appearing on any other grid cell, or missing
    from a TensorPool cell, means it no longer discriminates the column
    and the parse raises -- a Tailwind restyle that drops the arbitrary-
    value class correctly fails closed; do not relax this pin without
    re-verifying column identity by eye.

Competitor cells are NEVER recorded: the Lambda column is another
provider's price and the Traditional Clouds column is TensorPool's own
marketing estimate of GCP/AWS -- neither is a TensorPool offer.

Row semantics: prices are per GPU per hour (the mobile cards head the same
figures 'GPU COSTS / HR', and the Lambda column's $4.99 B200 / $3.29 H100
match Lambda's real per-GPU rates), so gpu_count_basis=1 and price*basis
== raw by construction. The grid's only Category label is ON-DEMAND
(reservations and committed discounts are contact-us, unpriced): the
first data row must carry exactly that label and every later category
cell must be empty, so a future second tier section raises for
re-verification instead of printing under the wrong tier. The CPU row is
priced but not a GPU and is fenced by label; the storage section
($100/TB/mo shared, $50/TB/mo object) lives outside the grid slice. A
TensorPool cell reading N/A is skipped + counted in partial_errors; any
other non-price cell text raises (a currency/format change is never
guessed into USD -- the bare $ is recorded as USD only while the exact
$D[.DD]/hr pin holds on a US provider's page).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "tensorpool"

URL = "https://tensorpool.dev/pricing"

# Tailwind column template of the desktop comparison grid -- the locating
# anchor, required EXACTLY once (zero = page reshaped, two = a lookalike
# grid appeared and attribution would be ambiguous either way).
_GRID_ANCHOR = "grid-cols-[200px_200px_1fr_1fr_1fr]"
# One half of the column-identity pin (see module docstring).
_HIGHLIGHT = "bg-[#B8E7F5]"

_N_COLS = 5
_TP_COL = 3  # 0-based: Category, GPU Type, Lambda Labs, TensorPool, Trad.
_EXPECTED_HEADERS = ("Category", "GPU Type", "Lambda Labs", "TensorPool")
_TRADITIONAL_PREFIX = "Traditional Clouds"
_ON_DEMAND_LABEL = "ON-DEMAND"
_NOT_OFFERED = "N/A"
_CPU_LABEL = "CPU"  # priced in the grid but not a GPU -- fenced by label

# Mobile-card tripwire slice: the page's second rendering of the same rows.
_CARD_START = ">TensorPool</h3>"
_CARD_END = "Lambda Labs</h3>"

_DIV_RE = re.compile(r"<div\b[^>]*>|</div>")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Grid price cells carry the /hr suffix; the mobile-card rows do not.
_GRID_PRICE_RE = re.compile(r"\$(\d+(?:\.\d+)?)/hr")
_CARD_ROW_RE = re.compile(
    r">([A-Z0-9 ]+?) - <strong>(\$\d+(?:\.\d+)?|N/A)</strong>"
)

_REGION = "unspecified"


def _text(fragment: str) -> str:
    """Tag-stripped, whitespace-collapsed text."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", fragment)).strip()


def _grid_slice(html: str) -> str:
    """The comparison grid's balanced-div slice, fail-closed on the anchor."""
    count = html.count(_GRID_ANCHOR)
    if count != 1:
        raise RuntimeError(
            f"tensorpool: grid anchor {_GRID_ANCHOR!r} appears {count}x "
            "(need exactly 1) -- page reshaped or a lookalike grid "
            "appeared; refusing to locate the comparison table"
        )
    start = html.rfind("<div", 0, html.index(_GRID_ANCHOR))
    if start < 0:
        raise RuntimeError(
            "tensorpool: no enclosing <div> before the grid anchor -- "
            "markup reshaped; refusing to extract"
        )
    depth = 0
    for match in _DIV_RE.finditer(html, start):
        depth += -1 if match.group(0) == "</div>" else 1
        if depth == 0:
            return html[start : match.end()]
    raise RuntimeError(
        "tensorpool: the grid container div never closes -- markup "
        "reshaped; refusing to extract"
    )


def _grid_cells(grid: str) -> List[Tuple[str, str]]:
    """(opening tag, tag-stripped text) per DIRECT child div, in DOM order."""
    cells: List[Tuple[str, str]] = []
    depth = 0
    cell_start: Optional[int] = None
    for match in _DIV_RE.finditer(grid):
        if match.group(0) == "</div>":
            depth -= 1
            if depth == 1 and cell_start is not None:
                cell_html = grid[cell_start : match.start()]
                open_end = cell_html.index(">")
                cells.append(
                    (cell_html[:open_end], _text(cell_html[open_end + 1 :]))
                )
                cell_start = None
        else:
            depth += 1
            if depth == 2 and cell_start is None:
                cell_start = match.start()
    return cells


def _check_shape(cells: List[Tuple[str, str]]) -> None:
    """Header-order + highlight-discrimination pins over the whole grid."""
    if len(cells) < 2 * _N_COLS or len(cells) % _N_COLS:
        raise RuntimeError(
            f"tensorpool: grid holds {len(cells)} cells -- not a header row "
            f"plus data rows of {_N_COLS}; refusing to group cells into rows"
        )
    header = tuple(text for _, text in cells[:_N_COLS])
    if header[:4] != _EXPECTED_HEADERS or not header[4].startswith(
        _TRADITIONAL_PREFIX
    ):
        raise RuntimeError(
            f"tensorpool: grid header {header!r} != pinned "
            f"{_EXPECTED_HEADERS!r} + {_TRADITIONAL_PREFIX!r}... -- column "
            "order/labels reshaped; refusing to attribute any price to "
            "TensorPool (competitor columns carry the same chip labels)"
        )
    for idx, (attrs, text) in enumerate(cells):
        is_tp_col = idx % _N_COLS == _TP_COL
        if (_HIGHLIGHT in attrs) != is_tp_col:
            side = (
                "missing from TensorPool-column"
                if is_tp_col
                else "leaked onto non-TensorPool"
            )
            raise RuntimeError(
                f"tensorpool: highlight class {_HIGHLIGHT!r} {side} grid "
                f"cell {idx} ({text[:40]!r}) -- the class no longer "
                "discriminates the TensorPool column; refusing to extract"
            )


def _parse_grid_rows(
    cells: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str, Optional[float]]], List[str]]:
    """-> ([(label, raw cell text, price-or-None)], partial_errors).

    Covers EVERY data row including CPU and unpriced ones, so the mobile-
    card tripwire can compare complete renderings; the GPU/price fences
    are applied by the caller.
    """
    entries: List[Tuple[str, str, Optional[float]]] = []
    errors: List[str] = []
    seen: set = set()
    data = cells[_N_COLS:]
    for row_idx in range(len(data) // _N_COLS):
        row = data[row_idx * _N_COLS : (row_idx + 1) * _N_COLS]
        category = row[0][1]
        if row_idx == 0:
            if category != _ON_DEMAND_LABEL:
                raise RuntimeError(
                    f"tensorpool: first grid Category label {category!r} != "
                    f"{_ON_DEMAND_LABEL!r} -- tier attribution unverified; "
                    "refusing to label rows on-demand"
                )
        elif category:
            raise RuntimeError(
                f"tensorpool: unexpected Category label {category!r} on "
                f"grid row {row_idx + 1} -- a second tier section appeared; "
                "refusing to extract until its rows can be labeled honestly"
            )
        label = row[1][1]
        if not label:
            raise RuntimeError(
                f"tensorpool: grid row {row_idx + 1} has an empty GPU Type "
                "cell -- row identity unpinnable; refusing to extract"
            )
        if label in seen:
            raise RuntimeError(
                f"tensorpool: GPU Type {label!r} appears twice in the grid "
                "-- row identity ambiguous; refusing to extract"
            )
        seen.add(label)
        cell_text = row[_TP_COL][1]
        if cell_text == _NOT_OFFERED:
            entries.append((label, cell_text, None))
            if label != _CPU_LABEL:
                errors.append(
                    f"row {label!r}: TensorPool cell is 'N/A' -- unpriced, "
                    "skipped"
                )
            continue
        match = _GRID_PRICE_RE.fullmatch(cell_text)
        if not match:
            raise RuntimeError(
                f"tensorpool: row {label!r} TensorPool cell {cell_text!r} "
                "misses the exact $D[.DD]/hr pin -- currency or format "
                "changed; refusing to guess"
            )
        entries.append((label, cell_text, float(match.group(1))))
    return entries, errors


def _card_map(html: str) -> Dict[str, Optional[float]]:
    """label -> price-or-None from the mobile TensorPool card rendering."""
    for pin in (_CARD_START, _CARD_END):
        count = html.count(pin)
        if count != 1:
            raise RuntimeError(
                f"tensorpool: mobile-card pin {pin!r} appears {count}x "
                "(need exactly 1) -- card rendering reshaped; the "
                "grid/card cross-check can no longer run, refusing to "
                "record unverified rows"
            )
    start = html.index(_CARD_START)
    end = html.index(_CARD_END)
    if end <= start:
        raise RuntimeError(
            "tensorpool: mobile cards reordered (Lambda Labs card before "
            "TensorPool) -- the slice would read another provider's card; "
            "refusing to extract"
        )
    out: Dict[str, Optional[float]] = {}
    for label, value in _CARD_ROW_RE.findall(html[start:end]):
        label = label.strip()
        if label in out:
            raise RuntimeError(
                f"tensorpool: label {label!r} repeats inside the mobile "
                "TensorPool card -- ambiguous; refusing to extract"
            )
        out[label] = None if value == _NOT_OFFERED else float(value[1:])
    if not out:
        raise RuntimeError(
            "tensorpool: mobile TensorPool card slice held zero rows -- "
            "card markup reshaped; the grid/card cross-check can no longer "
            "run, refusing to record unverified rows"
        )
    return out


def parse_tensorpool_pricing(
    html: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the pricing page -> (observations, partial_errors)."""
    cells = _grid_cells(_grid_slice(html))
    _check_shape(cells)
    entries, partial_errors = _parse_grid_rows(cells)
    grid_prices = {label: price for label, _, price in entries}
    card_prices = _card_map(html)
    if card_prices != grid_prices:
        raise RuntimeError(
            f"tensorpool: grid TensorPool column {grid_prices!r} != "
            f"mobile-card rendering {card_prices!r} -- the page's two "
            "renderings of the same table disagree; refusing to record "
            "either"
        )
    rows: List[Dict[str, Any]] = []
    for label, cell_text, price in entries:
        if label == _CPU_LABEL or price is None:
            continue
        rows.append(
            observation(
                sku_identifier=label,
                price_per_gpu_hr=price,
                raw_value=cell_text,
                tier="on-demand",
                region=_REGION,
                notes=(
                    "TensorPool column of the /pricing comparison grid, "
                    "per GPU per hour (GPU COSTS / HR); Lambda Labs and "
                    "Traditional Clouds competitor columns excluded; "
                    "mobile-card rendering cross-checked equal"
                ),
                extra={
                    "column": "TensorPool",
                    "grid_category": _ON_DEMAND_LABEL,
                },
            )
        )
    if not rows:
        raise RuntimeError(
            "tensorpool: zero GPU price observations from the grid -- "
            + ("; ".join(partial_errors) or "no row-level reasons recorded")
        )
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_tensorpool_pricing(html)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
