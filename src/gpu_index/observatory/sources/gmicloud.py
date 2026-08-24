# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""GMI Cloud -- /en/pricing server-rendered GPU cards, list-floor prices.

Page shape verified live 2026-08-22 (~101KB, single GET, no auth, no
pagination; transport note: the default python-urllib UA is 403'd by this
host while the framework's explicit UA (gpu_index.common.http) passes --
the shared fetch
already sends it). Five 'NVIDIA <chip>' h3 cards each carry a 'from $X.XX'
span immediately followed by a '/GPU-hour' unit span; that three-anchor
adjacency (NVIDIA h3 + 'from $' figure + unit span) is the ONLY price pin
because '$2.00' appears 9x in the bytes (meta description, og/twitter
tags, RSC flight duplicates) -- free-grepping dollars would multiply-record
H100.

Fail-closed identity pins:
  - every '/GPU-hour' span on the page must land inside exactly one
    NVIDIA-h3 card segment, at most one per card -- a unit span the parse
    cannot attribute to a chip heading means the card grid reshaped, so it
    raises rather than misattributes a price;
  - a card whose price slot is adjacent to the unit span but is not a
    'from $X.XX' figure is the page's own unpriced encoding (the GB300
    card prints 'Pre order' with a dangling '/GPU-hour' span) -> skipped
    and counted in partial_errors, never guessed; a unit span with NO
    adjacent price slot at all is a reshape -> raise.

Semantics:
  - prices are marketing 'from' floors, one per chip; the page's own FAQ
    states on-demand and committed pricing differ, so tier is recorded as
    'from_floor' (closest honest label), NEVER as on-demand;
  - the unit is the page's own '/GPU-hour' span -> per-GPU basis 1; the
    '$' figure on the en-US page of this US provider is USD;
  - sku_identifier is the verbatim h3 heading text -- the ' GPU' suffix is
    inconsistent ('NVIDIA H100 GPU' vs 'NVIDIA H200'), the catalog
    normalizes. GB200/GB300 are Grace-Blackwell superchips, first-class
    catalog skus, never B200/B300 (boundary-aware matching guarantees it);
  - JSON-LD on the page carries NO per-GPU offers (Organization/WebPage/
    Service/FAQPage only) -- the HTML cards are the only structured price
    surface. Bare /pricing 30x-redirects to a locale path; the /en/ URL is
    pinned so a default-locale change can never swap the page under us.
    The per-card availability badge (the div immediately before the h3:
    'AVAILABLE NOW' / 'Limited Availability') rides in extra.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "gmicloud"

URL = "https://www.gmicloud.ai/en/pricing"

_UNIT_TOKEN = "/GPU-hour"
_CARD_PREFIX = "NVIDIA "
# Plain-text h3 headings only ([^<]+): a card heading that grows nested
# tags stops matching, its unit span becomes unattributable, and the
# global attribution check below raises -- never a silent drop.
_H3_RE = re.compile(r"<h3[^>]*>([^<]+)</h3>")
# The ONLY price pin: a 'from $X.XX' span immediately followed by the
# '/GPU-hour' unit span. The 'from ' literal is load-bearing -- it is what
# justifies the from_floor tier label; if the page drops it the row is
# skipped (loose slot below), never relabeled.
_PRICE_PAIR_RE = re.compile(
    r"<span[^>]*>(from \$(\d[\d,]*\.\d{2}))</span>\s*"
    r"<span[^>]*>/GPU-hour</span>"
)
# Loose adjacency (any text in the price slot) -- used only to tell an
# unpriced card ('Pre order') from a reshaped one.
_PRICE_SLOT_RE = re.compile(
    r"<span[^>]*>([^<]*)</span>\s*<span[^>]*>/GPU-hour</span>"
)
# Availability badge: the absolutely-positioned div that sits INSIDE the
# card immediately BEFORE its h3. Metadata only (extra), so a miss is a
# missing badge, never a failed row.
_BADGE_RE = re.compile(r'<div class="absolute[^"]*">([^<]*)</div>\s*$')


def parse_gmicloud(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse: (observations, partial_errors) from the page body."""
    h3s = list(_H3_RE.finditer(html))
    cards = [
        (i, m) for i, m in enumerate(h3s)
        if m.group(1).strip().startswith(_CARD_PREFIX)
    ]
    if not cards:
        raise RuntimeError(
            "gmicloud: no 'NVIDIA <chip>' h3 card headings on the page -- "
            "card grid reshaped or pulled; refusing to extract"
        )

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    attributed_units = 0
    for i, m in cards:
        label = m.group(1).strip()
        seg_end = h3s[i + 1].start() if i + 1 < len(h3s) else len(html)
        seg = html[m.end():seg_end]
        units = seg.count(_UNIT_TOKEN)
        attributed_units += units
        if units == 0:
            errors.append(
                f"{label}: heading carries no '{_UNIT_TOKEN}' span -- not a "
                "price card, skipped"
            )
            continue
        if units > 1:
            raise RuntimeError(
                f"gmicloud: card {label!r} segment holds {units} "
                f"'{_UNIT_TOKEN}' spans -- prices can no longer be "
                "attributed to one chip heading; refusing to extract"
            )
        priced = _PRICE_PAIR_RE.findall(seg)
        if not priced:
            slot = _PRICE_SLOT_RE.findall(seg)
            if len(slot) == 1:
                errors.append(
                    f"{label}: price slot {slot[0].strip()!r} is not a "
                    "'from $X.XX' figure -- unpriced/pre-order card "
                    "skipped, never guessed"
                )
                continue
            raise RuntimeError(
                f"gmicloud: card {label!r} has a '{_UNIT_TOKEN}' span with "
                "no adjacent price slot -- card markup reshaped; refusing "
                "to guess which figure belongs to it"
            )
        raw_text, figure = priced[0]
        price = float(figure.replace(",", ""))
        if price <= 0:
            errors.append(
                f"{label}: printed floor {raw_text!r} is not a positive "
                "price -- skipped, not a $0 print"
            )
            continue

        prefix_start = h3s[i - 1].end() if i > 0 else 0
        badge = _BADGE_RE.search(html[prefix_start:m.start()])
        badge_text = badge.group(1).strip() if badge else ""
        extra: Dict[str, Any] = (
            {"availability_badge": badge_text} if badge_text else {}
        )
        rows.append(
            observation(
                sku_identifier=label,
                price_per_gpu_hr=price,
                currency="USD",
                raw_value=raw_text,
                raw_unit="usd_per_gpu_hr",
                gpu_count_basis=1,
                tier="from_floor",
                region="global",
                notes=(
                    f"{label} card {raw_text}{_UNIT_TOKEN} marketing list "
                    "floor (page FAQ: on-demand and committed rates "
                    "differ, so deliberately not labeled on-demand"
                    + (f"; badge {badge_text!r}" if badge_text else "")
                    + ")"
                ),
                extra=extra,
            )
        )

    total_units = html.count(_UNIT_TOKEN)
    if attributed_units != total_units:
        raise RuntimeError(
            f"gmicloud: {total_units - attributed_units} of {total_units} "
            f"'{_UNIT_TOKEN}' price spans sit outside NVIDIA card segments "
            "-- priced rows the parse cannot attribute to a chip; refusing "
            "to extract"
        )
    if not rows:
        raise RuntimeError(
            f"gmicloud: {len(cards)} NVIDIA card headings but zero priced "
            "observations -- "
            + ("; ".join(errors) or "no card-level reasons recorded")
        )
    return rows, errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    observations, partial_errors = parse_gmicloud(html)
    return result(
        SOURCE_ID,
        method="html-regex",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
    )
