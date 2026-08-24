# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Together AI -- together.ai/pricing, GPU Clusters on-demand + reserved tenors.

One fetch of the pricing page. The gpu-clusters -> sandbox section slice is
LOAD-BEARING (basket-proven, gpu_index/index/sources.py parse_together): the
Dedicated Inference section carries an HGX B200 row that is structurally
near-identical INCLUDING the is-right-border price cell ($8.99 live
2026-08-22), so no row/cell pin can discriminate page-wide -- only the
slice does. This observatory collector widens the basket's single-cell
extraction to the whole sliced table: every hardware row, the ON-Demand
column, and the four reserved tenor columns (tier="reserved", tenor in
extra/notes), each pinned fail-closed.

Surface shape (verified live 2026-08-22): the slice holds TWO tables.
First a hidden (<div class="hide">) Hardware/Hourly table whose prices are
byte-identical to the visible ON-Demand column -- recording it would
double-print every rate, so parsing starts at the visible table's unique
heading anchor ('On-demand hourly rates and reserved capacity'), which
sits AFTER the hidden table, and every parsed row must additionally carry
is-right-border on its column-1 cell (hidden-table rows do not). Then the
visible table: header row 1 = Hardware | ON-Demand | Reserved(colspan=4),
header row 2 = 'Pay as you go' + the four tenor labels. Tenor attribution
rides on column ORDER, so the full caption sequence is pinned exactly and
any drift raises; the basket's byte-identical header-order pin (column 1 =
ON-Demand = is-right-border) is kept on top of that.

Cell honesty:

  - exact '$D.DD' after whitespace collapse (the live 31-90d H100 cell is
    ' $3.45') -> a print; prices are per GPU per hour by the pinned page
    caption, so gpu_count_basis=1 and price*basis == raw by construction;
  - em-dash cell -> chip not offered on that column (GB200/GB300 NVL72 and
    HGX B300 live); skipped silently, never a $0 print;
  - 'Contact us' (or any other digit-free) cell -> priced on request;
    skipped + counted in partial_errors, including the colspan=4 cells
    that span all four tenor columns in the unpriced rows;
  - any other digit-bearing text raises (a currency/format change is never
    guessed into USD); a PRICED cell spanning >1 tenor column is skipped +
    counted (there is no honest single-tenor attribution for it); a row
    whose cells cover != 4 tenor columns raises.

sku_identifier is the provider's own hardware label with tags stripped
('NVIDIA HGX B200', 'NVIDIA GB200 NVL72', ...). Source stays ASCII-only:
the em-dash is escape-coded.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "together"

URL = "https://www.together.ai/pricing"

_SLICE_START = 'id="gpu-clusters" class="section-anchor'
_SLICE_END = 'id="sandbox" class="section-anchor'

# The visible combined table's heading -- unique in the slice (the hidden
# duplicate table before it is headed plain 'On-demand').
_TABLE_ANCHOR = "On-demand hourly rates and reserved capacity"

# Every pin must occur EXACTLY ONCE in the slice: zero means the layout
# reshaped, two means a lookalike table appeared and attribution would be
# ambiguous either way.
_IDENTITY_PINS = (
    _TABLE_ANCHOR,
    ">ON-Demand<",
    "All prices are per GPU per hour",
    # Header-ORDER pin, byte-identical to the basket collector's (verified
    # exactly-one occurrence live 2026-08-16 and 2026-08-22): binds column
    # 1 = ON-Demand = is-right-border, so a column reorder that keeps the
    # border class on the first cell fails loud instead of silently
    # recording a reserved tenor as on-demand.
    'caption-m">Hardware</p></div></div></th><th class="is-right-border">'
    '<div class="pricing_inline"><div class="opacity-70">'
    '<p class="caption-m">ON-Demand</p>',
)

_TENOR_LABELS = ("7-30 days", "31-90 days", "91-180 days", "181+ days")
# The full caption sequence of the two-row header, in document order.
_EXPECTED_CAPTIONS = (
    "Hardware",
    "ON-Demand",
    "Reserved",
    "Pay as you go",
) + _TENOR_LABELS

_THEAD_RE = re.compile(r"<thead>(.*?)</thead>", re.DOTALL)
_CAPTION_RE = re.compile(r'<p class="caption-m">([^<]*)</p>')
_ROW_RE = re.compile(r'<tr fs-list-element="item">(.*?)</tr>', re.DOTALL)
_CELL_RE = re.compile(r"<td([^>]*)>(.*?)</td>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PRICE_RE = re.compile(r"^\$(\d+\.\d{2})$")
_HAS_DIGIT_RE = re.compile(r"\d")
_COLSPAN_RE = re.compile(r'colspan="(\d+)"')

_RIGHT_BORDER = "is-right-border"
_EM_DASH = "\u2014"

_REGION = "unspecified"


def _text(cell_html: str) -> str:
    """Tag-stripped, whitespace-collapsed cell text."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", cell_html)).strip()


def _sliced_table(html: str) -> str:
    """The visible combined table's scope, fail-closed on every pin."""
    start = html.find(_SLICE_START)
    end = html.find(_SLICE_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "together: gpu-clusters/sandbox section anchors missing or "
            "reordered -- refusing to scan (the Dedicated Inference section "
            "carries a structurally identical HGX B200 row, is-right-border "
            "price cell included; only the section slice discriminates)"
        )
    section = html[start:end]
    for pin in _IDENTITY_PINS:
        count = section.count(pin)
        if count != 1:
            raise RuntimeError(
                f"together: identity pin {pin[:60]!r} found {count}x in the "
                "gpu-clusters slice (need exactly 1) -- layout changed or a "
                "lookalike table appeared; refusing to extract"
            )
    scope = section[section.index(_TABLE_ANCHOR) :]
    table_end = scope.find("</table>")
    if table_end < 0:
        raise RuntimeError(
            "together: no </table> after the combined-table heading -- "
            "table markup reshaped; refusing to extract"
        )
    return scope[:table_end]


def _check_header(table: str) -> None:
    theads = _THEAD_RE.findall(table)
    if len(theads) != 1:
        raise RuntimeError(
            f"together: expected exactly one <thead> in the combined table, "
            f"found {len(theads)} -- table markup reshaped"
        )
    captions = tuple(c.strip() for c in _CAPTION_RE.findall(theads[0]))
    if captions != _EXPECTED_CAPTIONS:
        raise RuntimeError(
            f"together: header captions {captions!r} != pinned "
            f"{_EXPECTED_CAPTIONS!r} -- column order/labels reshaped; "
            "refusing to attribute prices to tiers"
        )


def _classify_cell(text: str, where: str) -> Tuple[str, Any]:
    """('price', float) | ('dash', None) silent | ('unpriced', text) counted.

    Digit-bearing text that misses the exact $D.DD pin raises -- a currency
    or format change must never be guessed into USD.
    """
    match = _PRICE_RE.match(text)
    if match:
        return "price", float(match.group(1))
    if text == _EM_DASH:
        return "dash", None
    if _HAS_DIGIT_RE.search(text):
        raise RuntimeError(
            f"together: {where}: cell {text!r} looks priced but misses the "
            "exact $D.DD pin -- currency or format changed; refusing to "
            "guess"
        )
    return "unpriced", text


def parse_together_pricing(
    html: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the pricing page -> (observations, partial_errors)."""
    table = _sliced_table(html)
    _check_header(table)
    partial_errors: List[str] = []
    rows: List[Dict[str, Any]] = []
    items = _ROW_RE.findall(table)
    if not items:
        raise RuntimeError(
            "together: zero hardware rows in the combined table -- page "
            "reshaped or listings pulled"
        )
    for row_html in items:
        cells = _CELL_RE.findall(row_html)
        if len(cells) < 2:
            raise RuntimeError(
                f"together: hardware row with fewer than 2 cells "
                f"({_text(row_html)[:60]!r}) -- table markup reshaped"
            )
        hw_attrs, hw_html = cells[0]
        hardware = _text(hw_html)
        if not hardware or _RIGHT_BORDER in hw_attrs:
            raise RuntimeError(
                f"together: row label cell empty or border-marked "
                f"({_text(row_html)[:60]!r}) -- column binding broke; "
                "refusing to extract"
            )
        od_attrs, od_html = cells[1]
        if _RIGHT_BORDER not in od_attrs:
            raise RuntimeError(
                f"together: row {hardware!r}: column-1 price cell lost its "
                "is-right-border marker -- ON-Demand column binding broke "
                "(hidden-table or reordered row?); refusing to extract"
            )
        od_text = _text(od_html)
        where = f"row {hardware!r}, ON-Demand column"
        kind, value = _classify_cell(od_text, where)
        if kind == "price":
            rows.append(
                observation(
                    sku_identifier=hardware,
                    price_per_gpu_hr=value,
                    raw_value=od_text,
                    tier="on-demand",
                    region=_REGION,
                    notes=(
                        "GPU Clusters on-demand rate, per GPU per hour "
                        "(pinned page caption); Dedicated Inference "
                        "lookalike row excluded by section slice"
                    ),
                    extra={"column": "ON-Demand (Pay as you go)"},
                )
            )
        elif kind == "unpriced":
            partial_errors.append(f"{where}: unpriced cell {value!r} -- skipped")
        col = 0
        for attrs, cell_html in cells[2:]:
            span_match = _COLSPAN_RE.search(attrs)
            span = int(span_match.group(1)) if span_match else 1
            covered = _TENOR_LABELS[col : col + span]
            col += span
            if col > len(_TENOR_LABELS):
                break  # over-coverage -- the exactness check below raises
            text = _text(cell_html)
            where = f"row {hardware!r}, reserved {'/'.join(covered)}"
            kind, value = _classify_cell(text, where)
            if kind == "price":
                if span != 1:
                    partial_errors.append(
                        f"{where}: priced cell spans {span} tenor columns "
                        "-- no single-tenor attribution; skipped"
                    )
                    continue
                tenor = covered[0]
                rows.append(
                    observation(
                        sku_identifier=hardware,
                        price_per_gpu_hr=value,
                        raw_value=text,
                        tier="reserved",
                        region=_REGION,
                        notes=(
                            f"GPU Clusters reserved capacity, {tenor} "
                            "tenor, per GPU per hour (pinned page caption)"
                        ),
                        extra={"column": "Reserved", "tenor": tenor},
                    )
                )
            elif kind == "unpriced":
                partial_errors.append(
                    f"{where}: unpriced cell {value!r} -- skipped"
                )
        if col != len(_TENOR_LABELS):
            raise RuntimeError(
                f"together: row {hardware!r}: tenor cells cover {col} "
                f"columns, expected {len(_TENOR_LABELS)} -- table reshaped; "
                "refusing to extract"
            )
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_together_pricing(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
