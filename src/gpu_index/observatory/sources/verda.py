# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Verda (formerly DataCrunch) -- JSON-LD Offer graph on /pricing, all tiers.

The basket lane's verda collector (gpu_index.index.sources.parse_verda) keeps only
B200/B300 on-demand/spot instance rows; this observatory module records the
WHOLE offer graph -- every chip, plus the serverless and instant-cluster
rows the basket excludes -- with tiers labeled honestly. Separate module,
separate parse: the basket recipe is untouched.

Surface verified live 2026-08-22: one <script type="application/ld+json">
element whose @graph carries ~160 top-level Offer nodes named
'<N>x <MODEL> <MEM> <tier phrase>' (e.g. '104x B300 SXM6 268GB instant
cluster'). Identity pins, each earned on the live page:

  - name-suffix <-> description double-read: every tier phrase pairs with
    one fixed description (' spot' <-> 'Hourly spot instance price', ...).
    A row where the two disagree means the tier vocabulary moved and EVERY
    row is suspect, so the parse RAISES rather than recording a
    plausible-but-wrong tier into permanent history.
  - the leading '<N>x ' count is the provider's own GPU count and the only
    honest per-GPU divisor: serverless and instant-cluster figures are
    N-GPU totals (104x B300 at 780.0 is 104 x the 7.50 1x rate, verified).
    A GPU row without a parseable count cannot be normalized and is
    skipped into partial_errors -- never assumed to be 1x.
  - blank model labels (Verda's unnamed CPU-instance rows render as
    name ' on-demand') pin to no part and are skipped, counted.
  - a top-level price that disagrees with the offer's own
    priceSpecification is ambiguous (which figure is real?) and the row is
    skipped into partial_errors (they agree on every live row today).
  - currency is read per offer blob (top level and priceSpecification,
    which agree on every live row); a row where the two DISAGREE is
    ambiguous in the same way prices are -- the page has a EUR toggle, and
    a EUR figure recorded as USD would be wrong forever -- so it is
    skipped into partial_errors. Absent/blank on both reads records as
    'UNKNOWN', never assumed USD.
  - storage offers ('NVMe storage', description 'Monthly storage price per
    GiB') are not GPU rentals and are excluded by their own description.

The full offer name is the sku_identifier -- the catalog's boundary-aware
matcher keeps 'GB300 SXM6 288GB' out of B300 and 'B200 CC' (confidential
compute) inside B200.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "verda"

URL = "https://verda.com/pricing"

_JSONLD_RE = re.compile(
    r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
    re.DOTALL,
)
_COUNT_RE = re.compile(r"^(\d+)x\s+")
_CC_RE = re.compile(r"\bCC\b")
_MEMORY_RE = re.compile(r"\b(\d+)GB\b")
_PRICE_STR_RE = re.compile(r"^\d+(\.\d+)?$")

# (name suffix, exact published description, tier, product). The suffix
# classifies; the description independently confirms (see module
# docstring). ' serverless spot' and ' serverless continuous' must sit
# above ' spot' / the generic rows since their names end in overlapping
# phrases; matching walks this tuple in order.
_TIER_ROWS = (
    (
        " serverless continuous",
        "Hourly serverless container price",
        "serverless",
        "serverless",
    ),
    (
        " serverless spot",
        "Hourly serverless spot container price",
        "serverless",
        "serverless",
    ),
    (
        " instant cluster",
        "Hourly instant cluster price",
        "on-demand",
        "instant-cluster",
    ),
    (" on-demand", "Hourly on-demand instance price", "on-demand", "instance"),
    (" spot", "Hourly spot instance price", "spot", "instance"),
)

_STORAGE_DESCRIPTION = "Monthly storage price per GiB"

# What one row's raw figure denominates: a 1x row is per GPU already; a
# multi-GPU figure covers the whole rentable unit, named by product.
_UNIT_NOUN = {
    "instance": "instance",
    "serverless": "container",
    "instant-cluster": "cluster",
}


def _is_offer(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    node_type = node.get("@type")
    if isinstance(node_type, list):
        return "Offer" in node_type
    return node_type == "Offer"


def _offer_nodes(html: str) -> List[Dict[str, Any]]:
    blocks = _JSONLD_RE.findall(html)
    if not blocks:
        raise RuntimeError(
            "verda: no application/ld+json block on /pricing -- page reshaped"
        )
    offers: List[Dict[str, Any]] = []
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"verda: JSON-LD block unparseable ({exc}) -- page reshaped"
            ) from exc
        graph = data.get("@graph") if isinstance(data, dict) else data
        if not isinstance(graph, list):
            continue
        offers.extend(node for node in graph if _is_offer(node))
    if not offers:
        raise RuntimeError(
            "verda: JSON-LD parsed but holds zero Offer nodes -- graph "
            "reshaped or offers moved off the top level; refusing to guess"
        )
    return offers


def _price_of(node: Dict[str, Any]) -> Optional[float]:
    """The node's 'price' as a positive float, else None. Accepts the
    numeric JSON the page publishes today plus a plain-decimal string (the
    other schema.org-legal spelling) -- anything else is not a price."""
    price = node.get("price")
    if isinstance(price, bool):
        return None
    if isinstance(price, (int, float)):
        return float(price) if price > 0 else None
    if isinstance(price, str) and _PRICE_STR_RE.match(price):
        return float(price) if float(price) > 0 else None
    return None


def _currency_of(offer: Dict[str, Any]) -> Optional[str]:
    """The offer's currency label, or None when the top level and the
    priceSpecification carry DIFFERENT non-blank labels (ambiguous: which
    one prices the figure?). Absent/blank everywhere is 'UNKNOWN' -- never
    assumed USD."""
    spec = offer.get("priceSpecification")
    spec = spec if isinstance(spec, dict) else {}
    labels = []
    for node in (offer, spec):
        value = node.get("priceCurrency")
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())
    if not labels:
        return "UNKNOWN"
    if len(set(labels)) > 1:
        return None
    return labels[0]


def parse_verda(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(observations, partial_errors) from the /pricing JSON-LD. Pure over
    the body; raises when an identity pin stops discriminating."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    n_unnamed = 0
    uncounted: List[str] = []
    unpriced: List[str] = []
    ambiguous: List[str] = []
    mixed_currency: List[str] = []
    unclassified: List[str] = []

    for offer in _offer_nodes(html):
        name = str(offer.get("name") or "")
        desc = str(offer.get("description") or "")
        if desc == _STORAGE_DESCRIPTION:
            continue  # storage products, not GPU rentals -- out of scope
        for suffix, want_desc, tier, product in _TIER_ROWS:
            if name.endswith(suffix):
                break
        else:
            unclassified.append(name or "<blank>")
            continue
        if desc != want_desc:
            raise RuntimeError(
                f"verda: offer {name!r} carries description {desc!r} "
                f"(expected {want_desc!r} for the {suffix.strip()!r} "
                "suffix) -- tier labels reshaped; refusing to guess which "
                "side is honest"
            )
        label = name[: -len(suffix)]
        count_match = _COUNT_RE.match(label)
        model = _COUNT_RE.sub("", label, count=1).strip()
        if not model:
            # Verda's unnamed (CPU-instance) rows publish a blank model --
            # a price with no part attached cannot be recorded honestly.
            n_unnamed += 1
            continue
        if not count_match:
            uncounted.append(name)
            continue
        count = int(count_match.group(1))
        price = _price_of(offer)
        if price is None or count < 1:
            unpriced.append(name)
            continue
        spec = offer.get("priceSpecification")
        if isinstance(spec, dict) and "price" in spec:
            spec_price = _price_of(spec)
            if spec_price is None or abs(spec_price - price) > 1e-9:
                ambiguous.append(
                    f"{name}: price {offer.get('price')!r} != "
                    f"priceSpecification.price {spec.get('price')!r}"
                )
                continue
        currency = _currency_of(offer)
        if currency is None:
            spec_currency = (
                spec.get("priceCurrency") if isinstance(spec, dict) else None
            )
            mixed_currency.append(
                f"{name}: priceCurrency {offer.get('priceCurrency')!r} != "
                f"priceSpecification.priceCurrency {spec_currency!r}"
            )
            continue
        raw_price = offer.get("price")
        raw_value = (
            raw_price if isinstance(raw_price, str) else str(raw_price)
        )
        key = (name, raw_value, currency)
        if key in seen:
            # The page lists two 1x H100 hardware variants (different
            # CPU/RAM shapes) whose offer name AND price are identical --
            # one print per distinct (name, price, currency), not two.
            continue
        seen.add(key)

        prefix = (
            currency.lower()
            if len(currency) == 3 and currency.isalpha()
            else "unknown"
        )
        raw_unit = (
            f"{prefix}_per_gpu_hr"
            if count == 1
            else f"{prefix}_per_{_UNIT_NOUN[product]}_hr"
        )
        extra: Dict[str, Any] = {"product": product, "gpu_model": model}
        if product == "serverless":
            extra["serverless_billing"] = (
                "spot" if suffix == " serverless spot" else "continuous"
            )
        if _CC_RE.search(model):
            extra["confidential_compute"] = True
        obs = observation(
            sku_identifier=name,
            price_per_gpu_hr=price / count,
            currency=currency,
            raw_value=raw_value,
            raw_unit=raw_unit,
            gpu_count_basis=count,
            tier=tier,
            region="EU (Nordic DCs)",
            notes=desc,
            extra=extra,
        )
        memory_match = _MEMORY_RE.search(model)
        if memory_match:
            obs["memory_gb_label"] = int(memory_match.group(1))
        rows.append(obs)

    partial: List[str] = []
    if n_unnamed:
        partial.append(
            f"skipped {n_unnamed} offer rows with a blank model label "
            "(Verda's unnamed CPU-instance rows) -- no part to pin"
        )
    if uncounted:
        partial.append(
            f"skipped {len(uncounted)} GPU offer rows without a leading "
            "'Nx ' GPU count (cannot normalize per-GPU): "
            + "; ".join(sorted(set(uncounted))[:5])
        )
    if unpriced:
        partial.append(
            f"skipped {len(unpriced)} offer rows without a positive "
            "numeric price: " + "; ".join(sorted(set(unpriced))[:5])
        )
    if ambiguous:
        partial.append(
            f"skipped {len(ambiguous)} offer rows with disagreeing price "
            "fields: " + "; ".join(sorted(ambiguous)[:5])
        )
    if mixed_currency:
        partial.append(
            f"skipped {len(mixed_currency)} offer rows with disagreeing "
            "currency labels: " + "; ".join(sorted(mixed_currency)[:5])
        )
    if unclassified:
        partial.append(
            f"skipped {len(unclassified)} offer rows with an unrecognized "
            "tier phrase: " + "; ".join(sorted(set(unclassified))[:5])
        )
    return rows, partial


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    html = fetch(URL, timeout=timeout)
    rows, partial = parse_verda(html)
    return result(
        SOURCE_ID,
        method="jsonld",
        url=URL,
        observations=rows,
        partial_errors=partial or None,
    )
