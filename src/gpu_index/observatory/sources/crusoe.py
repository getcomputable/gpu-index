# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Crusoe Cloud -- www.crusoe.ai/cloud/pricing, GPU instances table.

One fetch of the server-rendered Webflow pricing page, pinned to the exact
www URL (the bare-domain and trailing-slash variants 301 to it). Parsing is
fenced to the substring between the literal section headings 'GPU instances
pricing' and 'CPU instances pricing' -- the fence is LOAD-BEARING because
the page reuses the SAME row container class and the SAME price-cell class
('pricing-rich w-richtext') on three lookalike surfaces that must never
print here: the CPU instance rows ($0.04/vCPU-hr), the Serverless Inference
table ($/M-token prices), and the Self-Serve Deployments rows, which carry
IDENTICAL chip headings ('NVIDIA H100', 'NVIDIA H200') priced $5.50/$6.00
bare for a different managed product.

Surface shape (verified live 2026-08-22): rows split on the container
'<div role="listitem" class="prixing-item w-dyn-item">' -- 'prixing' is the
site's real class spelling (a Webflow typo); pinning the corrected spelling
matches nothing. Per row: one h4 heading (chip label), exactly two spec
tags (memory, form factor -- the form-factor tag may be empty, e.g. L40S),
and exactly two 'pricing-rich' cells whose ORDER carries the tier: column 1
= On-demand, column 2 = Current spot, bound by the pinned three-caption
header sequence (GPU model | On-demand | Current spot). The second column
is SPOT, not reserved -- reserved capacity is contact-sales prose elsewhere
on the page.

Cell honesty:

  - a cell prints ONLY if the tag-stripped text of its ENTIRE body (every
    paragraph, not just the first) matches the exact '$D.DD/GPU-hr' pin --
    that suffix appears solely in the real GPU-rental cells page-wide and
    encodes the per-GPU-hour basis, so gpu_count_basis=1 and
    price*basis == raw by construction; USD by the pinned '$' (the page
    prices in dollars only); a basis qualifier appended anywhere in the
    cell ('per 8-GPU node') breaks the pin and raises;
  - a digit-free cell (the '/contact-sales' anchor rows: GB200, B200,
    MI355X today, and every Current-spot cell) is priced-on-request --
    skipped + counted in partial_errors, never synthesized; if Crusoe ever
    publishes numbers there, the same pin admits them automatically;
  - any other digit-bearing text raises (a currency/format/basis change is
    never guessed into a print).

sku_identifier joins the heading with the row's non-empty spec tags
('NVIDIA A100 80GB SXM' vs 'NVIDIA A100 80GB PCIe') because the two A100
rows share a bare heading -- the form-factor tag is part of row identity.
A capture with fewer than 4 priced on-demand rows raises (6 live
2026-08-22) instead of returning a thin success.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "crusoe"

URL = "https://www.crusoe.ai/cloud/pricing"

_SECTION_START = "GPU instances pricing"
_SECTION_END = "CPU instances pricing"

# Header-ORDER pin: binds cell 1 = On-demand and cell 2 = Current spot.
# Must occur exactly once page-wide AND inside the GPU section -- a column
# relabel/reorder (or a second lookalike header) fails loud instead of
# silently recording spot as on-demand.
_HEADER_PIN = (
    '<div class="pricing-heading is-d">GPU model</div>'
    '<div class="pricing-heading is-d">On-demand</div>'
    '<div class="pricing-heading is-d">Current spot</div>'
)

# The site's real row-container class -- 'prixing' is Crusoe's own typo.
_ROW_SPLIT = '<div role="listitem" class="prixing-item w-dyn-item">'

_HEADING_RE = re.compile(r'<h4 class="pricing-item-heading">([^<]+)</h4>')
_TAG_RE = re.compile(r'<div class="pricing-tag[^"]*">([^<]*)</div>')
# Captures the WHOLE cell body (to the first close-div) so the honesty pin
# below sees every word in the cell -- a first-<p>-only capture would let a
# basis qualifier in a trailing paragraph ('per 8-GPU node') vanish while
# the price still printed. The page's Storage table already splits price
# cells across two <p>s, so multi-paragraph cells are a live Webflow
# pattern, not a hypothetical. A nested <div> inside a cell would truncate
# this capture, so the row loop raises on any '<div' in the captured body.
_CELL_RE = re.compile(
    r'<div[^>]*class="pricing-rich w-richtext"[^>]*>(.*?)</div>', re.DOTALL
)
_MARKUP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# The identity/basis pin: '$D.DD/GPU-hr' exactly. Bare '$5.50' (Self-Serve
# Deployments), '$0.04/vCPU-hr' (CPU rows) and the $/M-token inference cells
# all miss it.
_PRICE_RE = re.compile(r"^\$(\d+\.\d{2})/GPU-hr$")
_HAS_DIGIT_RE = re.compile(r"\d")
_VENDOR_RE = re.compile(r"^(NVIDIA|AMD)\s")
_MEMORY_TAG_RE = re.compile(r"^(\d+)GB$")

_COLUMNS = ("On-demand", "Current spot")
_TIERS = ("on-demand", "spot")

_MIN_PRICED_ON_DEMAND = 4

_REGION = "unspecified"


def _text(cell_html: str) -> str:
    """Tag-stripped, whitespace-collapsed cell text."""
    return _WS_RE.sub(" ", _MARKUP_RE.sub(" ", cell_html)).strip()


def _gpu_section(html: str) -> str:
    """The GPU-instances slice, fail-closed on every identity pin."""
    for pin in (_SECTION_START, _SECTION_END, _HEADER_PIN):
        count = html.count(pin)
        if count != 1:
            raise RuntimeError(
                f"crusoe: identity pin {pin[:60]!r} found {count}x page-wide "
                "(need exactly 1) -- page restructured or a lookalike "
                "section appeared; refusing to extract"
            )
    start = html.index(_SECTION_START)
    end = html.index(_SECTION_END)
    if end <= start:
        raise RuntimeError(
            "crusoe: 'CPU instances pricing' precedes 'GPU instances "
            "pricing' -- sections reordered; refusing to extract"
        )
    section = html[start:end]
    if _HEADER_PIN not in section:
        raise RuntimeError(
            "crusoe: the GPU model|On-demand|Current spot header sits "
            "outside the GPU-instances section -- layout reshaped; refusing "
            "to attribute tiers by cell order"
        )
    return section


def _classify_cell(text: str, where: str) -> Tuple[str, Any]:
    """('price', float) | ('unpriced', text) counted by the caller.

    Digit-bearing text that misses the exact '$D.DD/GPU-hr' pin raises --
    a currency, format, or basis change must never be guessed into a print.
    """
    match = _PRICE_RE.match(text)
    if match:
        return "price", float(match.group(1))
    if _HAS_DIGIT_RE.search(text):
        raise RuntimeError(
            f"crusoe: {where}: cell {text!r} looks priced but misses the "
            "exact $D.DD/GPU-hr pin -- currency/format/basis changed; "
            "refusing to guess"
        )
    return "unpriced", text


def parse_crusoe_pricing(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the pricing page -> (observations, partial_errors)."""
    section = _gpu_section(html)
    parts = section.split(_ROW_SPLIT)
    if len(parts) < 2:
        raise RuntimeError(
            "crusoe: zero GPU rows in the GPU-instances section -- row "
            "container class changed or listings pulled"
        )
    partial_errors: List[str] = []
    rows: List[Dict[str, Any]] = []
    priced_on_demand = 0
    for row_html in parts[1:]:
        headings = _HEADING_RE.findall(row_html)
        if len(headings) != 1:
            raise RuntimeError(
                f"crusoe: GPU row with {len(headings)} chip headings "
                f"({_text(row_html)[:60]!r}) -- row markup reshaped; "
                "refusing to extract"
            )
        heading = headings[0].strip()
        if not _VENDOR_RE.match(heading):
            partial_errors.append(
                f"row {heading!r}: heading missing the NVIDIA/AMD vendor "
                "prefix -- cannot pin identity; skipped"
            )
            continue
        tags = [_text(t) for t in _TAG_RE.findall(row_html)]
        if len(tags) != 2:
            raise RuntimeError(
                f"crusoe: row {heading!r}: expected exactly 2 spec tags "
                f"(memory, form factor), found {len(tags)} -- row identity "
                "(A100 SXM vs PCIe) rides on the tags; refusing to extract"
            )
        memory_tag, form_tag = tags
        identifier = " ".join(t for t in (heading, memory_tag, form_tag) if t)
        cells = _CELL_RE.findall(row_html)
        if len(cells) != 2:
            raise RuntimeError(
                f"crusoe: row {identifier!r}: expected exactly 2 price "
                f"cells (On-demand, Current spot), found {len(cells)} -- "
                "tier attribution rides on cell order; refusing to extract"
            )
        mem_match = _MEMORY_TAG_RE.match(memory_tag)
        for cell_html, column, tier in zip(cells, _COLUMNS, _TIERS):
            where = f"row {identifier!r}, {column} column"
            if "<div" in cell_html:
                raise RuntimeError(
                    f"crusoe: {where}: nested <div> inside a price cell -- "
                    "cell markup reshaped (capture would truncate); "
                    "refusing to extract"
                )
            text = _text(cell_html)
            kind, value = _classify_cell(text, where)
            if kind != "price":
                partial_errors.append(
                    f"{where}: unpriced cell {value!r} -- skipped"
                )
                continue
            if tier == "on-demand":
                priced_on_demand += 1
            obs = observation(
                sku_identifier=identifier,
                price_per_gpu_hr=value,
                raw_value=text,
                tier=tier,
                region=_REGION,
                notes=(
                    f"GPU instances table, {column} column, per GPU per "
                    "hour by the pinned /GPU-hr suffix; Self-Serve "
                    "Deployments lookalike rows excluded by section fence"
                ),
                extra={
                    "column": column,
                    "memory_tag": memory_tag,
                    "form_factor_tag": form_tag or None,
                },
            )
            if mem_match:
                obs["memory_gb_label"] = int(mem_match.group(1))
            rows.append(obs)
    if priced_on_demand < _MIN_PRICED_ON_DEMAND:
        raise RuntimeError(
            f"crusoe: only {priced_on_demand} priced on-demand GPU rows "
            f"(expect >= {_MIN_PRICED_ON_DEMAND}; 6 live 2026-08-22) -- "
            "page reshaped or listings pulled; refusing to record a thin "
            "capture"
        )
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_crusoe_pricing(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
