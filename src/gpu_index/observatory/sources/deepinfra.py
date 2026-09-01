# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Deep Infra -- deepinfra.com/pricing, Custom LLMs dedicated-GPU table.

One fetch of the pricing page (server-rendered MUI markup, no auth). The
page is dominated by ~30 serverless per-token model tables plus two
lookalike 'Tier' tables whose Price column holds multiplier/threshold
strings ('1x base price'), so the identity pin is the exact header cell
sequence GPU | Memory | Price -- verified live 2026-08-22 to discriminate
against every other table on the page (34 total). If the pinned table ever
renders more than once (most other tables on this page appear twice via
tab-panel duplication), all header-matching copies must carry identical
rows or the parse aborts -- duplicated-but-diverging tables mean
attribution would be a guess either way.

Product semantics (load-bearing for the tier label): this is the 'Custom
LLMs' dedicated-GPU price -- container hosting for deploying your own
model, pay-for-uptime with minute-granularity billing and weekly invoicing
-- not a bare VM rental. There is no committed/reserved surface: tier is
"on-demand" and the product identity rides in notes/extra. B200/B300
CLUSTERS are contact-sales only, with no public price
rows, so nothing is recorded (or skipped-counted) for them.

Cell honesty, per data row of the pinned table (exactly 3 cells or raise):

  - price cell must fullmatch '$D[.DD] / GPU-hour' after tag-strip +
    unescape + whitespace collapse (live cells are '<b>$3.69</b> /
    GPU-hour') -> a USD print, per single GPU per hour by the cell's own
    denomination, so gpu_count_basis=1 and price*basis == raw by
    construction;
  - a digit-free price cell (a future 'Contact us') -> skipped + counted
    in partial_errors, never a guessed print;
  - any other digit-bearing text raises -- a currency or format change is
    never guessed into USD;
  - duplicate GPU labels raise: B200 vs B300 differ only by memory, so a
    label-mangling regression must fail loud rather than double-print one
    chip. The Memory cell is recorded (memory_gb_label + extra) as the
    cheap drift witness but is not itself a fence.

NEVER grep this page for chip names outside the pinned table: an i18n JSON
blob elsewhere in the markup carries the same labels ('gpu_b200':'B200')
and an unpriced '${{price}} / hour' template.
"""

from __future__ import annotations

import html as html_mod
import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "deepinfra"

URL = "https://deepinfra.com/pricing"

# The identity pin: every <th> of the table, cleaned, in order.
_EXPECTED_HEADER = ("GPU", "Memory", "Price")

_TABLE_RE = re.compile(r"<table.*?</table>", re.DOTALL)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_TH_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.DOTALL)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# The page prints '$' only; the literal '$' in this pin is what lets the
# print claim USD -- any other currency mark lands in the raises-branch.
_PRICE_RE = re.compile(r"\$(\d+(?:\.\d+)?) / GPU-hour")
_MEMORY_GB_RE = re.compile(r"(\d+)\s*GB")
_HAS_DIGIT_RE = re.compile(r"\d")

_REGION = "unspecified"


def _text(cell_html: str) -> str:
    """Tag-stripped, entity-unescaped, whitespace-collapsed cell text."""
    return _WS_RE.sub(" ", html_mod.unescape(_TAG_RE.sub(" ", cell_html))).strip()


def _pinned_row_sets(page: str) -> List[Tuple[Tuple[str, str, str], ...]]:
    """Data-row triples of every header-matching table, fail-closed."""
    row_sets: List[Tuple[Tuple[str, str, str], ...]] = []
    for table in _TABLE_RE.findall(page):
        header = tuple(_text(c) for c in _TH_RE.findall(table))
        if header != _EXPECTED_HEADER:
            continue
        triples: List[Tuple[str, str, str]] = []
        for row_html in _ROW_RE.findall(table):
            ths = _TH_RE.findall(row_html)
            tds = _TD_RE.findall(row_html)
            if ths and not tds:
                continue  # the (already header-pinned) <thead> row
            if ths or len(tds) != 3:
                raise RuntimeError(
                    f"deepinfra: pinned GPU table row does not carry exactly "
                    f"3 <td> cells ({_text(row_html)[:60]!r}) -- table markup "
                    "reshaped; refusing to extract"
                )
            triples.append(tuple(_text(c) for c in tds))
        if not triples:
            raise RuntimeError(
                "deepinfra: pinned GPU table has zero data rows -- listings "
                "pulled or markup reshaped; refusing to record"
            )
        row_sets.append(tuple(triples))
    return row_sets


def parse_deepinfra_pricing(
    page: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the pricing page -> (observations, partial_errors)."""
    row_sets = _pinned_row_sets(page)
    if not row_sets:
        raise RuntimeError(
            "deepinfra: no table with header GPU|Memory|Price on the page "
            "-- the Custom LLMs dedicated-GPU table moved or the page "
            "reshaped; refusing to scan lookalike tables (Tier-multiplier "
            "and per-token Model tables share the page)"
        )
    if len(row_sets) > 1 and any(rs != row_sets[0] for rs in row_sets[1:]):
        raise RuntimeError(
            f"deepinfra: {len(row_sets)} tables match the GPU|Memory|Price "
            "header pin with DIFFERING rows -- attribution would be a "
            "guess; refusing to extract"
        )

    rows: List[Dict[str, Any]] = []
    partial_errors: List[str] = []
    seen_labels: set = set()
    for label, memory_text, price_text in row_sets[0]:
        if not label:
            raise RuntimeError(
                "deepinfra: pinned GPU table row with an empty GPU label -- "
                "refusing to extract"
            )
        if label in seen_labels:
            raise RuntimeError(
                f"deepinfra: GPU label {label!r} appears twice in the pinned "
                "table -- labels mangled (B200 vs B300 differ only by "
                "memory); refusing to extract"
            )
        seen_labels.add(label)
        match = _PRICE_RE.fullmatch(price_text)
        if not match:
            if _HAS_DIGIT_RE.search(price_text):
                raise RuntimeError(
                    f"deepinfra: row {label!r}: price cell {price_text!r} "
                    "looks priced but misses the exact '$D.DD / GPU-hour' "
                    "pin -- currency or format changed; refusing to guess"
                )
            partial_errors.append(
                f"row {label!r}: unpriced cell {price_text!r} -- skipped"
            )
            continue
        mem_match = _MEMORY_GB_RE.fullmatch(memory_text)
        obs = observation(
            sku_identifier=label,
            price_per_gpu_hr=float(match.group(1)),
            raw_value=price_text,
            tier="on-demand",
            region=_REGION,
            notes=(
                f"Custom LLMs dedicated GPU deployment, {memory_text}, "
                "pay-for-uptime (minute-granularity billing, invoiced "
                "weekly) -- container hosting for your own model, not a "
                "bare VM"
            ),
            extra={
                "product": "custom_llm_dedicated_deployment",
                "memory_label": memory_text,
            },
        )
        if mem_match:
            obs["memory_gb_label"] = int(mem_match.group(1))
        rows.append(obs)
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_deepinfra_pricing(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
