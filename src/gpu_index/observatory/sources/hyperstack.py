# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Hyperstack -- gpu-pricing page, all three price tables, tiers labeled.

Widens the basket lane's B200/B300-only hyperstack recipe (parse_hyperstack
in gpu_index.index.sources -- separate module, separate fetch; the basket recipe is
untouched) to every GPU on the page. Same HTML->pipe-text transform. The
prior art's exactly-3-numeric-spec-fields fence is KEPT for the on-demand
rows -- it is what excludes rows that lack the per-GPU spec columns; do not
loosen it. The two spec-less tables get their own fences: reservation rows
are pinned by their trailing 'Reserve here' link cell, spot rows by the
section slice between the spot column header and the CPU-pricing header.

Page shape verified live 2026-08-22 (each table rendered exactly once):

  'On-Demand GPU Pricing'  GPU Model | VRAM (GB) | Max pCPUs per GPU |
                           Max RAM (GB) per GPU | Pricing Per Hour
                                                      -> tier on-demand
  'Reservation Pricing'    GPU Model | Starting from | Reserve
                                                      -> tier reserved
  'Spot VM Pricing'        GPU Model | Spot VM        -> tier spot

Prices are USD per GPU per hour (the spec columns are explicit per-GPU
shares), so gpu_count_basis stays 1 and price*basis reproduces the raw
cell. The reservation figure is a 'Starting from' floor, flagged in
notes/extra. Nav and footer carry priceless lookalike labels (GB200 NVL72,
GB300 NVL72, HGX B200...) that must never print -- every fence requires a
price cell inside its own section slice. Section anchors are required
UNIQUE and in page order; a missing or duplicated anchor raises rather
than guessing (a second render of a table would otherwise double-print
every row; a renamed header would silently lose a tier's history).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "hyperstack"

URL = "https://www.hyperstack.cloud/gpu-pricing"
# The page publishes one global price list (no per-region columns);
# Hyperstack's fleet is Europe/Canada-heavy -- same label the basket lane
# records.
REGION = "EU-heavy"

_TAGS_RE = re.compile(r"<[^>]+>")
_GAPS_RE = re.compile(r"(?:\||\s|&nbsp;|\xa0)+")

# Section anchors, required unique and in this order. The on-demand anchor
# embeds the FULL column header (nebius-style guard): a column added,
# renamed or reordered raises rather than misattributing which number is
# the price.
_OD_HEADER = (
    "On-Demand|GPU|Pricing|GPU|Model|VRAM|(GB)|Max|pCPUs|per|GPU|"
    "Max|RAM|(GB)|per|GPU|Pricing|Per|Hour"
)
_RESERVED_HEADER = "Reservation|Pricing|GPU|Model|Starting|from|Reserve"
_SPOT_HEADER = "Spot|VM|Pricing"
_SPOT_COLUMNS = "GPU|Model|Spot|VM"
# The CPU table is not parsed (no GPU rows) but its header is load-bearing:
# it is the end fence of the spot slice.
_CPU_HEADER = "On-Demand|CPU|Pricing"

# A GPU label: 1-6 pipe-separated tokens after the literal 'NVIDIA|' vendor
# cell. The (?!NVIDIA\|) guard stops a label from swallowing the next row's
# vendor cell, so an unpriced/malformed row is skipped (and counted via the
# row-start tally) instead of being glued onto its neighbour's price.
# The row-start tally alone cannot see a priced row whose vendor cell is
# not 'NVIDIA' (a new vendor appearing, or a vendor-cell rebrand on a mixed
# table) -- the price-cell tally below counts every '$' cell in the slice
# against the pinned rows so such a row skips LOUDLY, never silently.
_LABEL = r"((?:(?!NVIDIA\|)[A-Za-z0-9-]+\|){1,6}?)"
_PRICE = r"\$([\d,]+(?:\.\d+)?)"
_PRICE_RE = re.compile(_PRICE)
_OD_ROW_RE = re.compile(
    r"NVIDIA\|" + _LABEL + r"([\d.]+)\|([\d.]+)\|([\d.]+)\|" + _PRICE
)
_RESERVED_ROW_RE = re.compile(
    r"NVIDIA\|" + _LABEL + _PRICE + r"\|Reserve\|here"
)
# Spot rows have no spec columns and no trailing link -- the fence is the
# section slice itself.
_SPOT_ROW_RE = re.compile(r"NVIDIA\|" + _LABEL + _PRICE)

# Marketing banner seen live 2026-08-22: 'NVIDIA B300s are coming to
# Hyperstack -- On-Demand in August...'. While it is up, the priced B300
# on-demand row may be forward-listed; flagged in notes, never dropped.
_B300_BANNER = "NVIDIA|B300s|are|coming"


def _pipe_text(html: str) -> str:
    return _GAPS_RE.sub("|", _TAGS_RE.sub("|", html))


def _anchor_index(txt: str, anchor: str, what: str) -> int:
    first = txt.find(anchor)
    if first < 0:
        raise RuntimeError(
            f"hyperstack: {what} anchor {anchor!r} missing -- page reshaped; "
            "refusing to guess section bounds"
        )
    if txt.find(anchor, first + 1) >= 0:
        raise RuntimeError(
            f"hyperstack: {what} anchor {anchor!r} appears more than once -- "
            "a duplicate table render would double-print rows; refusing to "
            "pick one"
        )
    return first


def _number(text: str) -> Any:
    value = float(text)
    return int(value) if value.is_integer() else value


def parse_hyperstack(
    html: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse: (observations, partial_errors) from the page body."""
    txt = _pipe_text(html)
    i_od = _anchor_index(txt, _OD_HEADER, "on-demand table header")
    i_rv = _anchor_index(txt, _RESERVED_HEADER, "reservation table header")
    i_sp = _anchor_index(txt, _SPOT_HEADER, "spot table header")
    i_sc = _anchor_index(txt, _SPOT_COLUMNS, "spot column header")
    i_cpu = _anchor_index(txt, _CPU_HEADER, "CPU table header")
    if not (i_od < i_rv < i_sp < i_sc < i_cpu):
        raise RuntimeError(
            "hyperstack: pricing sections out of expected page order -- "
            "refusing to guess which table carries which tier"
        )
    banner = _B300_BANNER in txt

    sections = (
        ("on-demand", txt[i_od + len(_OD_HEADER):i_rv], _OD_ROW_RE),
        ("reserved", txt[i_rv + len(_RESERVED_HEADER):i_sp], _RESERVED_ROW_RE),
        ("spot", txt[i_sc + len(_SPOT_COLUMNS):i_cpu], _SPOT_ROW_RE),
    )
    rows: List[Dict[str, Any]] = []
    partial_errors: List[str] = []
    for tier, section, row_re in sections:
        matches = list(row_re.finditer(section))
        row_starts = section.count("NVIDIA|")
        price_cells = len(_PRICE_RE.findall(section))
        if not matches:
            raise RuntimeError(
                f"hyperstack: {tier} table header present but zero pinnable "
                f"rows ({row_starts} row starts) -- table reshaped or "
                "listings pulled; refusing to record a silent hole"
            )
        if len(matches) < row_starts:
            partial_errors.append(
                f"{tier} table: {row_starts - len(matches)} of {row_starts} "
                "rows had no pinnable price cell -- skipped, not guessed"
            )
        if price_cells > len(matches):
            partial_errors.append(
                f"{tier} table: {price_cells - len(matches)} of "
                f"{price_cells} price cells matched no pinnable NVIDIA row "
                "-- new vendor or reshaped row skipped, not guessed"
            )
        for m in matches:
            label = m.group(1).strip("|").replace("|", " ")
            identifier = f"NVIDIA {label}"
            price_str = m.groups()[-1]
            price = float(price_str.replace(",", ""))
            if tier == "on-demand":
                vram = _number(m.group(2))
                pcpus = _number(m.group(3))
                ram = _number(m.group(4))
                notes = (
                    f"{vram:g}GB VRAM, max {pcpus:g} pCPU / {ram:g}GB RAM "
                    "per GPU"
                )
                if banner and "B300" in label:
                    notes += (
                        " (page banner still says B300s are coming - "
                        "possibly forward-listed)"
                    )
                obs = observation(
                    sku_identifier=identifier,
                    price_per_gpu_hr=price,
                    raw_value=f"${price_str}",
                    tier=tier,
                    region=REGION,
                    notes=notes,
                    extra={
                        "table": "on_demand_gpu_pricing",
                        "vram_gb": vram,
                        "max_pcpus_per_gpu": pcpus,
                        "max_ram_gb_per_gpu": ram,
                    },
                )
                obs["memory_gb_label"] = vram
            elif tier == "reserved":
                obs = observation(
                    sku_identifier=identifier,
                    price_per_gpu_hr=price,
                    raw_value=f"${price_str}",
                    tier=tier,
                    region=REGION,
                    notes=(
                        "reservation 'Starting from' rate -- a floor, not a "
                        "fixed print"
                    ),
                    extra={
                        "table": "reservation_pricing",
                        "pricing_basis": "starting_from",
                    },
                )
            else:
                obs = observation(
                    sku_identifier=identifier,
                    price_per_gpu_hr=price,
                    raw_value=f"${price_str}",
                    tier=tier,
                    region=REGION,
                    notes="Spot VM rate",
                    extra={"table": "spot_vm_pricing"},
                )
            rows.append(obs)
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_hyperstack(html)
    return result(
        SOURCE_ID,
        method="html-regex",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
