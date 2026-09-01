# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Hot Aisle -- hotaisle.xyz/pricing, MI300X VM + bare-metal plan cards.

One fetch of the server-rendered (Astro) pricing page; one of the few AMD
principals with a public per-GPU-hour list price. The parse is anchored on
the three plan cards (kicker VM/VM/Bare metal), each carrying the full
identity chain in order: tier kicker -> plan <h2> (Small/Medium/Large) ->
gpu-count span + chip span -> a '$D.DD/GPU/hr' print. All captures are
required per card and every card that opens with a kicker must parse --
a matched kicker with an unreadable body is a reshape and raises.

Shape verified live 2026-08-22:

  - prices are published PER GPU PER HOUR ('$2.99/GPU/hr'): the literal
    '/GPU/hr' adjacent to the figure pins the basis (gpu_count_basis=1,
    price*basis == raw by construction) and the '$' pins USD;
  - kicker 'VM' cards are the self-serve on-demand VM rate (hero copy:
    new-customer rate, billed by the minute); kicker 'Bare metal' is a
    ONE-MONTH-MINIMUM committed rate -- recorded tier='monthly-commit',
    never on-demand, and the 'one-month minimum' phrase is pinned in the
    card so a dropped commit term fails loud instead of mislabeling;
  - the hero section above the cards repeats the same two prices under
    kickers 'Virtual machines' / '8x bare metal' -- lookalikes excluded by
    the exact-kicker pin (recording them would double-print every rate),
    and the meta/og/twitter description tags repeat '$2.99' again, which is
    why a card-anchored parse is mandatory over any dollar sweep;
  - the chip span prints 'MI300x' (lowercase x) -- upcased before use;
    the Medium card's count span is '2x &amp; 4x' (HTML-escaped ampersand);
  - an optional prose row: existing customers 'grandfathered at
    $1.99/GPU/hr' -- existing-customers-only, not purchasable, recorded
    tier='legacy' with the chip pinned via the adjacent
    why-we-raised-our-mi300x-price blog link; prose absent = no row, prose
    present but unpinnable = skipped + counted, never guessed;
  - MI355X appears in nav/JSON-LD/footer with NO published price (its page
    is a capacity-raise story) and the JSON-LD OfferCatalog carries Offers
    without price fields -- neither is a price surface; no rows are ever
    invented for them.

Transport note: the collector sends the project User-Agent defined in
gpu_index.common.http; no per-source headers are set.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "hotaisle"

URL = "https://hotaisle.xyz/pricing"

# Exact-kicker pin: the hero lookalikes ('Virtual machines',
# '8x bare metal') and any future card must not match by accident -- the
# '">' prefix and '</p>' suffix bind the whole kicker text.
_KICKER_RE = re.compile(r'uppercase tracking-wide">(VM|Bare metal)</p>')
_PLAN_RE = re.compile(r"<h2[^>]*>([A-Za-z ]+)</h2>")
# Count span + chip span, adjacent, in order; the chip capture doubles as
# the chip-shape pin (MI + 3 digits + X/x, nothing else).
_COUNT_CHIP_RE = re.compile(
    r"<span>(\d+x(?:\s*&amp;\s*\d+x)*)</span><span[^>]*>(MI\d{3}[Xx])</span>"
)
# '$' pins USD; the adjacent '/GPU/hr' pins the per-GPU-hour basis.
_PRICE_RE = re.compile(r"\$(\d+\.\d{2})/GPU/hr")
_GRANDFATHERED_WORD = "grandfathered at"
_GRANDFATHERED_RE = re.compile(
    r"grandfathered at.{0,120}?<strong[^>]*>(\$(\d+\.\d{2})/GPU/hr)</strong>",
    re.DOTALL,
)

# Live cards parse within ~600 chars of the kicker; 1500 tolerates copy
# growth without reaching the next card (~2500 chars away), and windows are
# additionally bounded by the next kicker.
_CARD_WINDOW = 1500
# The price sits ~100-330 chars after the chip span live.
_PRICE_PROXIMITY = 400
# The chip attribution for the grandfathered prose (the mi300x blog slug)
# sits ~120 chars after the price live.
_LEGACY_CHIP_PROXIMITY = 400

_MIN_CARDS = 3  # fewer matched cards than today's three = page reshaped

_TIER_BY_KICKER = {"VM": "on-demand", "Bare metal": "monthly-commit"}
_COMMIT_PHRASE = "one-month minimum"

_REGION = "unspecified"


def _parse_card(kicker: str, window: str) -> Dict[str, Any]:
    """One plan card -> one observation; any missing capture raises."""
    plan_m = _PLAN_RE.search(window)
    cc_m = _COUNT_CHIP_RE.search(window)
    if not plan_m or not cc_m or plan_m.start() >= cc_m.start():
        raise RuntimeError(
            f"hotaisle: {kicker!r} card missing or reordered plan/count/chip "
            "captures -- card markup reshaped; refusing to extract"
        )
    plan = plan_m.group(1).strip()
    count_label = cc_m.group(1).replace("&amp;", "&")
    count_label = re.sub(r"\s+", " ", count_label)
    chip = cc_m.group(2).upper()
    price_zone = window[cc_m.end() : cc_m.end() + _PRICE_PROXIMITY]
    prices = list(_PRICE_RE.finditer(price_zone))
    if len(prices) != 1:
        raise RuntimeError(
            f"hotaisle: {kicker!r}/{plan!r} card has {len(prices)} "
            f"'$D.DD/GPU/hr' prints within {_PRICE_PROXIMITY} chars of the "
            "chip span (need exactly 1) -- price/basis/currency pin broke; "
            "refusing to guess"
        )
    tier = _TIER_BY_KICKER[kicker]
    if tier == "monthly-commit" and _COMMIT_PHRASE not in window.lower():
        raise RuntimeError(
            f"hotaisle: bare-metal card {plan!r} lost its "
            f"{_COMMIT_PHRASE!r} phrase -- the commit term is what makes "
            "the monthly-commit label honest; refusing to attribute a tier"
        )
    note = (
        f"{plan} bare-metal plan, {count_label} {chip} per node, "
        "one-month minimum commitment; price published per GPU per hour"
        if tier == "monthly-commit"
        else f"{plan} VM plan, {count_label} {chip} node sizes, self-serve "
        "VM rate published per GPU per hour"
    )
    return observation(
        sku_identifier=chip,
        price_per_gpu_hr=float(prices[0].group(1)),
        raw_value=prices[0].group(0),
        tier=tier,
        region=_REGION,
        notes=note,
        extra={
            "plan": plan,
            "card_kicker": kicker,
            "node_gpus_label": count_label,
            "chip_label_as_published": cc_m.group(2),
        },
    )


def _legacy_rows(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The optional grandfathered prose rate -- pinned or skipped+counted."""
    rows: List[Dict[str, Any]] = []
    partial_errors: List[str] = []
    matches = list(_GRANDFATHERED_RE.finditer(html))
    n_prose = html.count(_GRANDFATHERED_WORD)
    if n_prose > len(matches):
        partial_errors.append(
            f"{n_prose - len(matches)} 'grandfathered at' prose mention(s) "
            "did not match the pinned $D.DD/GPU/hr pattern -- skipped, "
            "never guessed"
        )
    for m in matches:
        tail = html[m.end() : m.end() + _LEGACY_CHIP_PROXIMITY]
        if "mi300x" not in tail.lower():
            partial_errors.append(
                "grandfathered rate found but the adjacent mi300x blog-link "
                "chip pin is gone -- no honest chip attribution; skipped"
            )
            continue
        rows.append(
            observation(
                sku_identifier="MI300X",
                price_per_gpu_hr=float(m.group(2)),
                raw_value=m.group(1),
                tier="legacy",
                region=_REGION,
                notes=(
                    "grandfathered rate for existing customers with running "
                    "compute -- not purchasable by new customers; price "
                    "published per GPU per hour"
                ),
                extra={
                    "availability": "existing customers only",
                    "chip_pin": "why-we-raised-our-mi300x-price blog link",
                },
            )
        )
    return rows, partial_errors


def parse_hotaisle(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of the pricing page -> (observations, partial_errors)."""
    kickers = list(_KICKER_RE.finditer(html))
    if len(kickers) < _MIN_CARDS:
        raise RuntimeError(
            f"hotaisle: matched {len(kickers)} plan-card kickers, expected "
            f">= {_MIN_CARDS} -- pricing grid reshaped (Tailwind anchor "
            "churn?) or cards pulled; refusing to extract"
        )
    rows: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str, str], str] = {}
    for i, m in enumerate(kickers):
        end = m.end() + _CARD_WINDOW
        if i + 1 < len(kickers):
            end = min(end, kickers[i + 1].start())
        obs = _parse_card(m.group(1), html[m.end() : end])
        key = (obs["extra"]["plan"], obs["sku_identifier"], obs["tier"])
        if key in seen:
            raise RuntimeError(
                f"hotaisle: duplicate card for plan/chip/tier {key!r} "
                f"(prices {seen[key]} vs {obs['raw_value']}) -- ambiguous "
                "attribution; refusing to double-print"
            )
        seen[key] = obs["raw_value"]
        rows.append(obs)
    legacy, partial_errors = _legacy_rows(html)
    rows.extend(legacy)
    return rows, partial_errors


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    rows, partial_errors = parse_hotaisle(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=rows,
        partial_errors=partial_errors or None,
    )
