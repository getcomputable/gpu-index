# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Massed Compute -- /pricing Next.js RSC flight rows, every GPU VM row.

Widened from the basket lane's parse_massedcompute (gpu_index/index/sources.py,
B200/B300 only): the key regex here matches EVERY row key (H-series,
A-series, L-series, RTX workstation parts...) while keeping all of the
basket parser's fences -- plus one more, because the widened net actually
hit the failure the basket lane only feared (details on the pin below).

Surface shape (verified live 2026-08-22): row keys look like
'\\"B300 SXM6-x 8-0\\"' / '\\"H100 (80GB)-mobile-x 1-0\\"' -- every row is
rendered twice (mobile card + desktop table). A literal dollar sign is
RSC-escaped as '$$', so prices print as '$$52.80'. The published figure is
a NODE price for the stated GPU quantity (the row's own Qty); per-GPU
normalization divides by that stated count. The page is a single
on-demand VM price table (columns Qty/vCPU/RAM/Storage/Price) -- one tier,
one price column, USD. No public JSON API exists (/api/pricing 307s to
signin).

Identity pins (every one earned):
  - a row's price window is bounded at the NEXT row key of ANY sku
    (basket-proven: a priceless row must never steal a neighbor's price);
  - AND at the end of the current RSC chunk-definition line (literal
    backslash-n in the stream). Measured live 2026-08-22: some rows defer
    price cells to later-pushed chunks ('7d:...$$23.36'), and those chunk
    definitions land inside OTHER rows' key-to-key windows -- the desktop
    'H100 (80GB) x8' row publishes 'Request' (no price), yet its naive
    window contained the deferred $$23.36 belonging to 'H100 NVL (94GB)
    NVLink x8': a silent wrong print (2.92 vs the true 2.73 per GPU).
    Chunk definitions always start on a new backslash-n-separated line,
    so fencing at the line end keeps attribution exact;
  - exactly ONE price in the fenced window records. Zero (contact-us
    'Request' rows, RSC-deferred price cells) skips; more than one
    (promo/strikethrough ambiguity) skips -- never guess which is real;
  - the mobile/desktop copies of a row must agree: if both survive their
    fences with DIFFERENT prices, the row is skipped and noted in
    partial_errors rather than double-printed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "massedcompute"

URL = "https://vm.massedcompute.com/pricing"

# ONE regex plays both roles (row identity AND window bounding) so the two
# can never diverge: a key shape only the bounding regex could see would
# stop windows correctly but hide rows, and vice versa.
_KEY_RE = re.compile(
    r'\\"(?P<label>[^"\\]{1,60}?)(?P<mobile>-mobile)?-x\s*(?P<count>\d+)-\d+\\"'
)
_PRICE_RE = re.compile(r"\$\$([\d,]+(?:\.\d+)?)")
# Key -> its own price sit ~600-1100 chars apart (basket measurement
# 2026-08-10, re-confirmed 2026-08-22); 4000 gives slack without reaching
# into a neighboring row when the other two fences are somehow absent.
_PRICE_WINDOW = 4000
# Literal backslash-n: ends an RSC chunk-definition line in the flight
# stream. Deferred price chunks ('7d:[...$$23.36...]') always start past
# one, so this fence is what keeps them out of foreign rows' windows.
_CHUNK_LINE_END = "\\n"


def parse_massedcompute(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the fetched page: (observations, partial_errors).

    Raises with a specific message when the page reshapes under either pin
    (no row keys at all / keys but no pinnable price) -- result() would
    catch the empty list too, but the specific message beats the generic.
    """
    keys = list(_KEY_RE.finditer(html))
    if not keys:
        raise RuntimeError(
            "massedcompute: zero RSC row keys on /pricing -- the "
            "'<label>-x <count>-<idx>' key shape changed; refusing to guess"
        )
    # (label, count) -> {node_price: {"raw": str, "variants": set}}
    by_row: Dict[Tuple[str, int], Dict[float, Dict[str, Any]]] = {}
    order: List[Tuple[str, int]] = []
    saw_multi: Set[Tuple[str, int]] = set()
    for i, m in enumerate(keys):
        label = m.group("label").strip()
        count = int(m.group("count"))
        if not label or not count:
            continue
        row = (label, count)
        if row not in by_row:
            by_row[row] = {}
            order.append(row)
        end = m.end() + _PRICE_WINDOW
        if i + 1 < len(keys):
            end = min(end, keys[i + 1].start())
        line_end = html.find(_CHUNK_LINE_END, m.end())
        if line_end != -1:
            end = min(end, line_end)
        prices = _PRICE_RE.findall(html[m.end() : end])
        if len(prices) != 1:
            # Zero: contact-us 'Request' row or an RSC-deferred price cell
            # (the twin page copy usually carries it inline). Two or more:
            # ambiguous -- never guess which figure is real.
            if len(prices) > 1:
                saw_multi.add(row)
            continue
        value = float(prices[0].replace(",", ""))
        slot = by_row[row].setdefault(
            value, {"raw": prices[0], "variants": set()}
        )
        slot["variants"].add("mobile" if m.group("mobile") else "desktop")

    rows: List[Dict[str, Any]] = []
    unpriced: List[str] = []
    ambiguous: List[str] = []
    conflicting: List[str] = []
    for label, count in order:
        found = by_row[(label, count)]
        if not found:
            (ambiguous if (label, count) in saw_multi else unpriced).append(
                f"{label} x{count}"
            )
            continue
        if len(found) > 1:
            conflicting.append(f"{label} x{count}: {sorted(found)}")
            continue
        value, slot = next(iter(found.items()))
        rows.append(
            observation(
                sku_identifier=label,
                price_per_gpu_hr=value / count,
                raw_value=slot["raw"],
                raw_unit="usd_per_node_hr",
                gpu_count_basis=count,
                tier="on-demand",
                region="unspecified",
                notes=(
                    f"{label} x{count} node ${slot['raw']}/hr "
                    "(on-demand VM tier)"
                ),
                extra={"row_variants": sorted(slot["variants"])},
            )
        )
    if not rows:
        raise RuntimeError(
            f"massedcompute: {len(order)} row keys but not one pinnable "
            "price -- the '$$' price escaping or the cell layout changed; "
            "refusing to guess"
        )
    partial_errors: List[str] = []
    if unpriced:
        partial_errors.append(
            "unpriced rows skipped (contact-us 'Request' or RSC-deferred "
            "price cell in both page copies): " + ", ".join(unpriced)
        )
    if ambiguous:
        partial_errors.append(
            "ambiguous rows skipped (multiple prices inside one row "
            "window): " + ", ".join(ambiguous)
        )
    if conflicting:
        partial_errors.append(
            "rows with CONFLICTING prices across page copies skipped "
            "(never guess which is real): " + "; ".join(conflicting)
        )
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    rows, partial_errors = parse_massedcompute(html)
    return result(
        SOURCE_ID,
        method="html-regex",
        url=URL,
        observations=rows,
        partial_errors=partial_errors or None,
    )
