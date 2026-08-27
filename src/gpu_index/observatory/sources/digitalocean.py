# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""DigitalOcean -- docs GPU Droplet pricing page, On-Demand + Spot tables.

One fetch of the server-rendered docs page (Hugo, prices in the raw bytes,
no JS needed). The docs page is DELIBERATELY the source over the marketing
page (www.digitalocean.com/pricing/gpu-droplets): the marketing page
interleaves '12 Month Reserved Price*' teasers ($3.26 H100, $7.94 B300)
with on-demand figures in a JS-heavy blob, while the docs page separates
tiers into cleanly anchored sections. The collector sends the project
User-Agent defined in gpu_index.common.http.

Page shape verified live 2026-08-22 (11 on-demand + 8 spot rows):

  - tier attribution comes ONLY from the section heading anchors, pinned
    fail-closed: for each of (on-demand-gpu-droplet-pricing -> on-demand,
    spot-gpu-droplet-pricing -> spot) the literal '<h2 id="...">' tag must
    appear exactly once, the slice runs to the NEXT '<h[2-4] id=' heading
    (refusing an unterminated slice), and the slice must hold exactly one
    <table> whose <th> cells are exactly GPU|Price. The third table
    (contract-gpu-droplet-pricing) publishes 'Per contract after contacting
    sales' instead of numbers and is never sliced; the same page's CPU
    pricing section and bandwidth prose ($0.01/GiB) are excluded the same
    way. The spot section is a public preview whose prices 'may change
    daily based on available idle GPU capacity' (page's own caveat, fine
    for slotted capture) -- the widely quoted $11.19 B300 is THIS tier;
    on-demand B300 is contract-only with no published number and must
    never be recorded;
  - row fences: a data row must carry exactly 2 <td> (else the label->
    price binding is suspect and the parse raises); a label cell that does
    not start 'NVIDIA '/'AMD ' is skipped + counted (a new vendor prefix
    is recorded as a visible partial error, never guessed); a price cell
    must fullmatch '$D[.DD] per hour' -- digit-free text (a contact-sales
    cell migrating in) is skipped + counted, digit-bearing text that
    misses the pin raises (a currency/format change is never guessed into
    USD). A section whose table yields zero observations raises even when
    every row was individually skippable;
  - prices are whole-droplet hourly rates in USD (the $ literal is part of
    the price pin); per-GPU normalization divides by the GPU count the
    page itself states in the label -- '(8x)' rows are 8-GPU droplets,
    unsuffixed rows are single-GPU. The multiplier is parsed generically
    ('(Nx)' -> N, label-final) so a new droplet size records its stated
    basis; a '(Nx)' token that is NOT label-final (footnote marker or
    suffix drift) makes the basis unbindable and the row is skipped +
    counted rather than defaulted to a 1-GPU basis. Every
    (8x) price is exactly 8x its 1x row today but that is NOT enforced --
    the page publishes both figures independently (spot repricing could
    diverge them) and each is recorded as printed;
  - label gotchas, all kept verbatim in sku_identifier (the catalog
    normalizes downstream): 'NVIDIA L40s' has a lowercase s; 'NVIDIA B300,
    air-cooled' and ', liquid-cooled' are distinct rows at the same price
    (both recorded, never deduped); 'NVIDIA RTX 4000'/'NVIDIA RTX 6000'
    omit 'Ada' although DO's marketing page names them RTX 4000 Ada /
    RTX 6000 Ada -- the bare labels normalize to the catalog's
    generation-AMBIGUOUS RTX_6000/RTX_4000 bucket skus (a deliberate
    catalog rule: never guess a generation from a bare label), resolved
    in config, never patched here;
  - per-tier chip sets differ by design (MI350X/MI355X spot-only, MI300X/
    MI325X on-demand-only today) -- rows are recorded per section, never
    joined across tables.

No pagination; a single static page per capture.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "digitalocean"

URL = "https://docs.digitalocean.com/products/droplets/details/pricing/"

# (heading anchor id, lane tier label, tier note) -- the section heading is
# the ONLY tier attribution on this page. The contract section's anchor is
# deliberately absent: its rows are unpriced by design.
_SECTIONS = (
    (
        "on-demand-gpu-droplet-pricing",
        "on-demand",
        "on-demand GPU droplet hourly rate",
    ),
    (
        "spot-gpu-droplet-pricing",
        "spot",
        "spot GPU droplet rate (public preview; may change daily on idle "
        "capacity, page's own caveat)",
    ),
)

_NEXT_HEADING_RE = re.compile(r'<h[2-4] id="')
_TH_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.DOTALL)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_EXPECTED_HEADER = ("GPU", "Price")
# Vendor-prefixed part labels only -- fences out prose/junk landing in the
# label column (and makes a column swap fail visibly row by row).
_LABEL_RE = re.compile(r"^(?:NVIDIA|AMD)\s")
# The full published price cell, exactly: literal $, dollars[.cents],
# literal ' per hour'. The $ literal IS the currency pin.
_PRICE_RE = re.compile(r"\$([0-9]+(?:\.[0-9]{2})?) per hour")
_HAS_DIGIT_RE = re.compile(r"[0-9]")
# The page's own GPU-count statement: '... (8x)' droplets carry 8 GPUs.
_MULTI_GPU_RE = re.compile(r"\(([0-9]+)x\)$")
# Any '(Nx)' token anywhere in the label -- fences the end-anchored pin
# above: if the page states a count but it is no longer label-final (a
# footnote marker or suffix drifted in), the basis is unbindable and the
# row must be skipped loudly, never defaulted to 1 (which would record the
# whole-droplet price as a per-GPU price, 8x too high).
_ANY_MULTI_GPU_RE = re.compile(r"\([0-9]+x\)")

_REGION = "unspecified"


def _text(cell_html: str) -> str:
    """Tag-stripped, whitespace-collapsed cell text."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", cell_html)).strip()


def _section_table(html: str, anchor_id: str) -> str:
    """The section's single pricing table, fail-closed on every pin."""
    pin = f'<h2 id="{anchor_id}">'
    count = html.count(pin)
    if count != 1:
        raise RuntimeError(
            f"digitalocean: heading pin {pin!r} appears {count}x (need "
            "exactly 1) -- page reshaped or section renamed; refusing to "
            "guess section bounds"
        )
    start = html.index(pin) + len(pin)
    nxt = _NEXT_HEADING_RE.search(html, start)
    if not nxt:
        raise RuntimeError(
            f"digitalocean: no following heading after {anchor_id!r} -- an "
            "unterminated slice could swallow the unpriced contract table "
            "below; refusing to extract"
        )
    section = html[start : nxt.start()]
    n_tables = section.count("<table")
    if n_tables != 1:
        raise RuntimeError(
            f"digitalocean: section {anchor_id!r} holds {n_tables} tables "
            "(need exactly 1) -- layout changed or a lookalike table "
            "appeared; refusing to extract"
        )
    table = section.split("<table", 1)[1]
    end = table.find("</table>")
    if end < 0:
        raise RuntimeError(
            f"digitalocean: unterminated <table> in section {anchor_id!r} "
            "-- table markup reshaped; refusing to extract"
        )
    table = table[:end]
    header = tuple(_text(th) for th in _TH_RE.findall(table))
    if header != _EXPECTED_HEADER:
        raise RuntimeError(
            f"digitalocean: section {anchor_id!r} header cells {header!r} "
            f"!= pinned {_EXPECTED_HEADER!r} -- column order/labels "
            "reshaped; refusing to attribute prices"
        )
    return table


def parse_digitalocean(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the docs page -> (observations, partial_errors)."""
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for anchor_id, tier, tier_note in _SECTIONS:
        table = _section_table(html, anchor_id)
        data_rows = [tr for tr in _TR_RE.findall(table) if "<td" in tr]
        if not data_rows:
            raise RuntimeError(
                f"digitalocean: section {anchor_id!r} table has zero data "
                "rows -- page reshaped or listings pulled"
            )
        recorded_before = len(rows)
        for row_html in data_rows:
            cells = _TD_RE.findall(row_html)
            if len(cells) != 2:
                raise RuntimeError(
                    f"digitalocean: section {anchor_id!r} row with "
                    f"{len(cells)} cells (need exactly 2: GPU|Price) -- "
                    "label->price binding suspect; refusing to extract "
                    f"({_text(row_html)[:60]!r})"
                )
            label = _text(cells[0])
            price_text = _text(cells[1])
            where = f"{anchor_id}: row {label!r}"
            if not _LABEL_RE.match(label):
                errors.append(
                    f"{where}: label lacks the NVIDIA/AMD vendor prefix -- "
                    "unpinnable part, row skipped"
                )
                continue
            price_match = _PRICE_RE.fullmatch(price_text)
            if not price_match:
                if _HAS_DIGIT_RE.search(price_text):
                    raise RuntimeError(
                        f"digitalocean: {where}: price cell {price_text!r} "
                        "looks priced but misses the exact '$D.DD per "
                        "hour' pin -- currency or unit changed; refusing "
                        "to guess"
                    )
                errors.append(
                    f"{where}: unpriced cell {price_text!r} -- skipped"
                )
                continue
            droplet_price = float(price_match.group(1))
            multi = _MULTI_GPU_RE.search(label)
            if multi is None and _ANY_MULTI_GPU_RE.search(label):
                errors.append(
                    f"{where}: '(Nx)' GPU-count token is not label-final "
                    "-- the page's stated basis is unbindable, row "
                    "skipped (never defaulted to 1)"
                )
                continue
            gpu_count = int(multi.group(1)) if multi else 1
            if gpu_count < 1:
                errors.append(
                    f"{where}: stated GPU count {gpu_count} unusable -- "
                    "per-GPU basis unknown, row skipped"
                )
                continue
            rows.append(
                observation(
                    sku_identifier=label,
                    price_per_gpu_hr=droplet_price / gpu_count,
                    raw_value=price_text,
                    raw_unit="usd_per_instance_hr",
                    gpu_count_basis=gpu_count,
                    tier=tier,
                    region=_REGION,
                    notes=(
                        f"{label} {gpu_count}-GPU droplet {tier_note}; "
                        "whole-droplet rate divided by the page's own "
                        "count statement"
                    ),
                    extra={"section_anchor": anchor_id},
                )
            )
        if len(rows) == recorded_before:
            # Row-level skips are legitimate one by one, but a whole
            # section printing nothing means the fences stopped fitting --
            # fail closed with the per-row reasons.
            raise RuntimeError(
                f"digitalocean: section {anchor_id!r} produced zero "
                f"observations across {len(data_rows)} rows -- "
                f"{'; '.join(errors) or 'no row-level reasons recorded'}"
            )
    return rows, errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_digitalocean(html)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
