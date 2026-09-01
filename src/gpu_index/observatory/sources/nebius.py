# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Nebius - server-rendered escaped-JSON price table, every GPU row, both tiers.

Prior art: the basket lane's parse_nebius (gpu_index/index/sources.py) pins the
FULL header row and refuses rows carrying a third price column, because a
substring pin would silently record the wrong tier if a column were inserted.
This observatory collector keeps that fail-closed discipline but parses the
table STRUCTURALLY: the escaped-JSON blob is unescaped, the one GPU table is
located by an exact-header pin ('"table":{"content":[' + the exact 5-column
header), and the whole row array is json.loads'd - so a reshaped table breaks
loudly instead of a regex quietly matching less. The basket recipe is
untouched; separate module, separate parse.

Surface verified live 2026-08-22: nebius.com/prices carries exactly ONE GPU
table ("NVIDIA GPU Instances"; the pin is asserted unique so a second table
with the same header raises for deliberate review). Columns are
[Item, vCPUs, RAM, Preemptible GPU-hour, On-demand GPU-hour]; prices are
already per GPU-hour (vCPU/RAM are per-GPU resource shares, not node
counts), so gpu_count_basis stays 1. Edge shapes handled:

  - GB300/GB200 NVL72 rows print '--' / '[Contact us](...)' - digit-free
    cells are published-unpriced, skipped and counted in partial_errors;
  - L40S rows print 'from $X.XX' floor prices - recorded with the qualifier
    preserved in raw_value/extra, never silently flattened to a firm quote;
  - a price cell that carries digits but not the pinned '$' shape RAISES
    (currency/figure no longer attributable - never assumed USD);
  - the page's other tables (CPU-only 'Price per hour', Storage, Other
    services) have different headers and never match the pin. The separate
    AI Studio surface (/prices-ai-studio) is per-token inference pricing,
    not GPU rental - deliberately out of scope.

Currency: USD is pinned on the literal '$' in each accepted price cell.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "nebius"

URL = "https://nebius.com/prices"

# The FULL header row, exact - any added/renamed/reordered column kills the
# pin and raises rather than guessing which price is which tier.
_HEADER_ROW = (
    '["Item","vCPUs","RAM, GB","Preemptible, GPU-hour","On-demand, GPU-hour"]'
)
_TABLE_PIN = '"table":{"content":[' + _HEADER_ROW
# The content array closes with ']]' immediately before customColumnWidth.
_TABLE_END = ']],"customColumnWidth"'

_N_COLUMNS = 5
# (column index, tier) - order is guaranteed by the exact header pin above.
_PRICE_COLUMNS = ((4, "on-demand"), (3, "preemptible"))

# Accepted price cells: '$7.85' or 'from $1.82' (provider floor price),
# nothing else. The '$' is the currency pin; thousands separators must be
# well-formed groups of three - a misplaced comma raises (digits present,
# no match) rather than silently concatenating to a wrong figure.
_PRICE_CELL_RE = re.compile(r"^(from )?\$(\d+(?:,\d{3})*(?:\.\d+)?)$")
_DIGIT_RE = re.compile(r"\d")


def parse_nebius(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the fetched page: (observations, skipped-row notes)."""
    plain = html.replace("\\", "")
    first = plain.find(_TABLE_PIN)
    if first == -1:
        raise RuntimeError(
            "nebius: GPU price-table pin (exact 5-column header) not found "
            "- page reshaped or a price column added/renamed; refusing to "
            "guess which column is which tier"
        )
    if plain.find(_TABLE_PIN, first + 1) != -1:
        raise RuntimeError(
            "nebius: GPU price-table pin matched more than one table - "
            "rows are no longer attributable to a single table; refusing "
            "to guess"
        )
    start = first + len('"table":{"content":')
    end = plain.find(_TABLE_END, first)
    if end == -1:
        raise RuntimeError(
            "nebius: pinned GPU table no longer terminates as expected "
            "(']],\"customColumnWidth\"' missing) - page reshaped"
        )
    try:
        table = json.loads(plain[start : end + 2])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"nebius: pinned GPU table no longer parses as a JSON row "
            f"array: {exc}"
        ) from exc

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for row in table[1:]:  # table[0] is the pinned header row
        if (
            not isinstance(row, list)
            or len(row) != _N_COLUMNS
            or not all(isinstance(cell, str) for cell in row)
        ):
            raise RuntimeError(
                "nebius: GPU table row is not "
                f"{_N_COLUMNS} string cells - table reshaped: {row!r:.200}"
            )
        item = row[0].strip()
        vcpus = row[1].strip()
        ram = row[2].strip()
        if not item:
            raise RuntimeError(
                "nebius: GPU table row with an empty Item label - a print "
                "we cannot attribute is a print we cannot record"
            )
        printed_any = False
        for col, tier in _PRICE_COLUMNS:
            cell = row[col].strip()
            m = _PRICE_CELL_RE.match(cell)
            if m is None:
                if _DIGIT_RE.search(cell):
                    # Digits outside the pinned '$x.xx' shape: the figure /
                    # currency is no longer attributable - never guess USD.
                    raise RuntimeError(
                        f"nebius: unrecognized price cell for {item!r} "
                        f"({tier}): {cell!r:.100}"
                    )
                continue  # digit-free = published-unpriced (dashes/Contact us)
            price = float(m.group(2).replace(",", ""))
            if price <= 0:
                skipped.append(
                    f"zero-price cell skipped ({tier}): {item} = {cell}"
                )
                continue
            qualifier = "from" if m.group(1) else None
            extra: Dict[str, Any] = {
                "vcpus_per_gpu": vcpus,
                "ram_gb_per_gpu": ram,
            }
            if qualifier:
                extra["price_qualifier"] = qualifier
            rows.append(
                observation(
                    sku_identifier=item,
                    price_per_gpu_hr=price,
                    raw_value=cell,
                    tier=tier,
                    # The page publishes one price list with no region
                    # qualifier anywhere - record that honestly rather
                    # than claiming the prices are global.
                    region="unspecified",
                    notes=f"{item} ({vcpus} vCPU / {ram} GB per-GPU share)"
                    + (" - provider 'from' floor price" if qualifier else ""),
                    extra=extra,
                )
            )
            printed_any = True
        if not printed_any:
            skipped.append(
                f"unpriced row skipped (no numeric price published): {item}"
            )
    return rows, skipped


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    observations, skipped = parse_nebius(html)
    return result(
        SOURCE_ID,
        method="html-embedded-json",
        url=URL,
        observations=observations,
        partial_errors=skipped or None,
    )
