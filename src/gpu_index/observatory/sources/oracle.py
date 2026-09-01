# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Oracle Cloud Infrastructure (OCI) -- public cetools price-list API.

The apexapps.oracle.com cetools /products/ JSON is the exact backend that
hydrates oracle.com/cloud/compute/pricing (the human page's price cells are
empty divs keyed by data-partnumber, so the page itself cannot serve as an
independent scrape target). One GET with currencyCode=USD returns the whole
price list (~627 items, ~189KB; WITHOUT the currency filter it balloons to
a ~2.8MB multi-currency payload). partNumber and currencyCode are the ONLY
query params the server honors -- name=/serviceCategory= are silently
ignored -- so ALL row fencing is client-side.

Surface facts verified live 2026-08-22 (top level {lastUpdated, items},
29 rows with metricName == 'GPU Per Hour'):

  - the exact-metric fence alone excludes the per-NODE VMware
    BM.GPU.A10.64 commit tiers ('Node Per Hour', $16/$13/$11 per node, NOT
    per GPU) and the Roving Edge / Cloud@Customer per-day possession rows;
  - LOOKALIKE SOFTWARE ROWS: 'OCI - NVIDIA AI Enterprise - H100' ($2.50)
    and 7 siblings share the same serviceCategory 'Compute - GPU' AND the
    same 'GPU Per Hour' metric but are per-GPU-hour LICENSE add-ons, not
    rentals -- fenced by known partNumbers plus displayName marker;
  - Compute Cloud@Customer rows are on-prem appliance pricing, and two of
    them share an IDENTICAL displayName ('... Compute - GPU.L40S',
    B110965 $3.50 vs B111455 $2.90) -- only partNumber discriminates, so
    recording them under the displayName identifier would be ambiguous;
    excluded and counted in book_stats, never guessed;
  - displayName formatting is inconsistent ('OCI- Compute',
    'Compute  - GPU - A10') -- identity anchors on partNumber, never on
    regexing a chip out of the name;
  - PAY_AS_YOU_GO is the only price model published (no committed GPU
    tiers on this surface): an on-demand list price, already per GPU per
    hour (shape BM.GPU.H100.8 bills 8x this rate; cross-checked against
    Oracle's own H200-GA blog quoting $10/GPU/hr for BM.GPU.H100.8);
  - top-level lastUpdated is catalog-WIDE, not per-row; it rides in
    book_stats as the staleness tripwire.

Fail-closed identity pins: for every known rental partNumber the collector
asserts the row's displayName still carries the expected chip token -- if
Oracle ever remapped part numbers, the catalog would otherwise derive a
WRONG sku from the drifted name, so drift raises instead of recording
misattributed prints. A reshaped top level, a multi-currency or non-USD
localization (the requested currency filter broke), an unrecognized
serviceCategory (how a NEW on-prem product line would sneak in), an
ambiguous double PAY_AS_YOU_GO price, or a non-numeric value each raise or
skip-with-note per the contract. Parts whose displayName carries no chip
token at all (hardware-generation labels X7/V2/E3/E4: B88517, B88518,
B89734, B92740, B93544) are recorded honestly under the provider's own
label and normalize to unmapped -- the ambiguity is Oracle's, and a null
sku beats a guessed one. New GPU-Per-Hour partNumbers outside the map are
recorded honestly the same way.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "oracle"

URL = (
    "https://apexapps.oracle.com/pls/apex/cetools/api/v1/products/"
    "?currencyCode=USD"
)

GPU_METRIC = "GPU Per Hour"  # exact string; 'Node Per Hour' etc. are out
_PAYG = "PAY_AS_YOU_GO"

# Public-cloud GPU rentals live in exactly these two categories today
# (H100T sits in '- Other'). Anything else with a 'GPU Per Hour' metric is
# skipped with a loud note until a human vets it -- this is how a new
# on-prem product line (the next Cloud@Customer) fails closed instead of
# printing as a public rental.
_PUBLIC_CATEGORIES = ("Compute - GPU", "Compute - GPU - Other")

# NVIDIA AI Enterprise rows are per-GPU-hour SOFTWARE licenses sharing the
# rental rows' category and metric. Fence on the known parts AND the
# displayName marker so a newly added license row is still caught.
_LICENSE_MARKER = "NVIDIA AI Enterprise"
_LICENSE_PARTS = frozenset(
    ("B111824", "B111825", "B111826", "B111827",
     "B111828", "B111829", "B111830", "B111831")
)

# Compute Cloud@Customer = on-prem appliance pricing (two rows share an
# identical displayName; only partNumber discriminates them). Fence on the
# category, the known parts, AND the displayName marker.
_ON_PREM_CATEGORY = "Compute Cloud@Customer"
_ON_PREM_MARKER = "Cloud@Customer"
_ON_PREM_PARTS = frozenset(("B110965", "B111454", "B111455"))

# partNumber -> chip token its displayName must still carry (checked with
# the same uppercase/separator/boundary discipline the sku catalog uses).
# partNumber is the stable anchor; this map is the tripwire that the
# anchor<->name binding has not drifted under us. Parts whose names carry
# no chip token (X7/V2/E3/E4 generation labels) are deliberately absent --
# they normalize to unmapped, so a name drift cannot misattribute them.
_PART_EXPECTED_TOKEN = {
    "B98415": "H100",
    "B109480": "H100T",  # distinct variant -- never folds into H100
    "B110519": "H200",
    "B110978": "B200",
    "B112237": "B300",
    "B110979": "GB200",
    "B112140": "GB300",
    "B109485": "MI300X",
    "B111758": "MI355X",
    "B109479": "L40S",
    "B95907": "A100",
    "B95909": "A10",
    "B112613": "RTX PRO 6000",
}

_SEPARATORS_RE = re.compile(r"[-_/]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _has_token(label: str, token: str) -> bool:
    """Boundary-aware token check mirroring observatory.catalog's label
    prep, so this tripwire agrees with what the catalog would match."""
    text = _WHITESPACE_RE.sub(" ", _SEPARATORS_RE.sub(" ", label.upper()))
    return (
        re.search(r"(?<![A-Z0-9])" + re.escape(token) + r"(?![A-Z0-9])", text)
        is not None
    )


def parse_oracle(
    body: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Pure parse over the fetched body.

    Returns (observations, skipped_notes, book_stats); skipped_notes feed
    partial_errors so an unpinnable row is a visible hole, never a guess,
    and book_stats accounts for the deliberate policy fences (license and
    on-prem rows) plus the catalog-wide lastUpdated staleness tripwire.
    """
    payload = json.loads(body)
    if (
        not isinstance(payload, dict)
        or "lastUpdated" not in payload
        or not isinstance(payload.get("items"), list)
    ):
        raise RuntimeError(
            "oracle: response is no longer {lastUpdated, items: [...]} -- "
            "cetools API reshaped; refusing to guess at the new shape"
        )
    items = payload["items"]

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    n_gpu_metric = 0
    n_license = 0
    n_on_prem = 0
    for item in items:
        if not isinstance(item, dict) or item.get("metricName") != GPU_METRIC:
            continue  # OCPU/ECPU/Node/per-day metrics: not per-GPU-hour rows
        n_gpu_metric += 1
        part = str(item.get("partNumber") or "").strip()
        display = str(item.get("displayName") or "").strip()
        if not part or not display:
            skipped.append(
                f"GPU-metric row without partNumber/displayName "
                f"(partNumber={part!r}) -- skipped, no identity to pin"
            )
            continue
        if part in _LICENSE_PARTS or _LICENSE_MARKER in display:
            n_license += 1  # software license add-on, not a rental
            continue
        category = str(item.get("serviceCategory") or "")
        if (
            part in _ON_PREM_PARTS
            or category == _ON_PREM_CATEGORY
            or _ON_PREM_MARKER in display
        ):
            n_on_prem += 1  # on-prem appliance pricing, not public cloud
            continue
        if category not in _PUBLIC_CATEGORIES:
            skipped.append(
                f"{part} ({display}): unrecognized serviceCategory "
                f"{category!r} -- skipped until vetted (public rentals are "
                f"{list(_PUBLIC_CATEGORIES)}; a new on-prem product line "
                "must not print as one)"
            )
            continue
        expected = _PART_EXPECTED_TOKEN.get(part)
        if expected is not None and not _has_token(display, expected):
            raise RuntimeError(
                f"oracle: part {part} displayName {display!r} no longer "
                f"carries the expected {expected!r} token -- the "
                "partNumber<->name binding drifted and the catalog would "
                "derive a wrong sku; refusing to record misattributed prints"
            )
        locs = item.get("currencyCodeLocalizations")
        if locs is None or locs == []:
            skipped.append(
                f"{part} ({display}): no currency localization -- unpriced "
                "row skipped, not a $0 print"
            )
            continue
        if (
            not isinstance(locs, list)
            or len(locs) != 1
            or not isinstance(locs[0], dict)
        ):
            raise RuntimeError(
                f"oracle: part {part} carries {locs!r} localizations -- the "
                "currencyCode=USD server-side filter broke (multi-currency "
                "payload) or the field reshaped; refusing to pick a currency"
            )
        currency = locs[0].get("currencyCode")
        if currency != "USD":
            raise RuntimeError(
                f"oracle: requested USD but part {part} localized as "
                f"{currency!r} -- the currency filter broke; refusing to "
                "record under an assumed currency"
            )
        prices = locs[0].get("prices")
        if not isinstance(prices, list):
            raise RuntimeError(
                f"oracle: prices on part {part} is no longer a list "
                f"({prices!r}) -- price field reshaped; refusing to guess"
            )
        payg = [
            p for p in prices if isinstance(p, dict) and p.get("model") == _PAYG
        ]
        other_models = sorted(
            {
                str(p.get("model"))
                for p in prices
                if isinstance(p, dict) and p.get("model") != _PAYG
            }
        )
        if other_models:
            skipped.append(
                f"{part} ({display}): unrecognized price model(s) "
                f"{', '.join(other_models)} alongside {_PAYG} -- a new tier "
                "published; recorded the known tier only, teach the "
                "collector the new one"
            )
        if not payg:
            skipped.append(
                f"{part} ({display}): no {_PAYG} price -- unpriced row "
                "skipped, not a $0 print"
            )
            continue
        if len(payg) > 1:
            raise RuntimeError(
                f"oracle: part {part} publishes {len(payg)} {_PAYG} prices "
                "-- ambiguous; refusing to pick one"
            )
        value = payg[0].get("value")
        if not _is_number(value):
            raise RuntimeError(
                f"oracle: {_PAYG} value on part {part} is not a plain "
                f"number ({value!r}) -- price field reshaped; refusing to "
                "guess"
            )
        if float(value) <= 0:
            skipped.append(
                f"{part} ({display}): {_PAYG} value {value} <= 0 -- "
                "skipped, not a real print"
            )
            continue
        rows.append(
            observation(
                sku_identifier=display,
                price_per_gpu_hr=float(value),
                raw_value=str(value),
                tier="on-demand",  # PAY_AS_YOU_GO list price
                region="global",  # OCI list prices are region-uniform
                notes=f"partNumber {part} {_PAYG} list price",
                extra={
                    "part_number": part,
                    "service_category": category,
                },
            )
        )
    return rows, skipped, {
        "catalog_last_updated": payload.get("lastUpdated"),
        "items_total": len(items),
        "gpu_metric_rows": n_gpu_metric,
        "recorded": len(rows),
        "software_license_rows_excluded": n_license,
        "on_prem_appliance_rows_excluded": n_on_prem,
    }


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, skipped, book_stats = parse_oracle(body)
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=observations,
        partial_errors=skipped or None,
        book_stats=book_stats,
    )
