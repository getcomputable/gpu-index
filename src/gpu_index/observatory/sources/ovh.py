# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""OVHcloud -- public no-auth order-catalog JSON, both billing subsidiaries.

Two GETs per capture (fetch count fixed by the configured subsidiary list,
capped at MAX_SUBSIDIARIES): the FR book on api.ovh.com (bills EUR; the
full GPU lineup incl. H100/H200/A100) and the SEPARATE US legal entity's
book on api.us.ovhcloud.com (bills USD; L4/L40S/V100S only -- a genuinely
smaller lineup, not a parse failure). The two entities publish different
lineups AND different prices, so every observation is pinned to its book:
``locale.currencyCode`` and ``locale.subsidiary`` from the same response
must equal the per-endpoint expectation or the whole book RAISES -- prices
must never be recorded under the wrong currency/entity. (A third,
world-English price book exists behind www.ovhcloud.com/en/ -- deliberately
not fetched; cross-checks must be same-subsidiary.)

Row fence (identity pin, fail-closed): only addons with
``blobs.commercial.brick == "gpu"`` are GPU *instance rental* rows. The
brick pin -- not gpu-blob presence -- is load-bearing: ``ai-*`` managed-
service addons (AI Deploy/Training under brick "ai-serving") carry a full
``blobs.technical.gpu`` spec but are priced PER MINUTE, and would poison
the record as ~60x-cheap hourly prints if the fence were "has a gpu blob".
Within the brick, a row needs ``blobs.technical.gpu.model`` (the provider's
structured part label -> sku_identifier) and ``.gpu.number`` (the per-GPU
divisor); rows missing either are skipped into partial_errors, never
guessed.

Price semantics verified live 2026-08-22:

  - ``pricings[].price`` is an integer in 1e-8 currency units (280000000 =
    2.80). TRIPWIRE per row: price/1e8 must agree with the row's own
    ``formattedPrice`` display string to the half-cent -- the display is
    rounded to 2 decimals (win-t2-45 US: 155819000 -> 1.55819 shown as
    "$1.56 USD"), so the check is a tolerance, not equality. A row whose
    two figures disagree (scale convention broken) is failed, never
    rescaled. Prices are ex-VAT (``tax`` is a separate field), matching
    OVH's own "HT" display;
  - tier comes from the planCode billing-mode suffix, NOT from
    intervalUnit: US rows publish intervalUnit "none"/interval 0 on BOTH
    hourly-consumption AND some monthly rows (win-l4-*.monthly.postpaid),
    so intervalUnit is unreliable in both directions. ``*.consumption`` =
    hourly on-demand; ``*.monthly.postpaid`` = the monthly rate, recorded
    as monthly-commit and normalized at 730 h/month (the lane-wide
    convention); any other suffix is skipped+noted, never guessed;
  - ``win-*`` flavors are the same hardware at Windows-license-included
    prices (t2-45 US 0.88 vs win-t2-45 1.56/hr) -- recorded, labeled via
    the structured ``blobs.technical.os.family`` field (notes +
    extra.windows_license_included), never mixed silently with Linux rows;
  - ``t1-le-*``/``t2-le-*`` are legacy flavor variants of t1/t2 (usually
    identical prices; US monthly diverges) -- all recorded as separate
    plan_code rows, flagged extra.legacy_le_variant for dedupe-aware
    consumers;
  - some rows are tagged "coming_soon" with a published price (A10 and
    Quadro-RTX5000 on FR today) -- recorded with blobs.tags in extra so
    consumers can screen;
  - "Quadro-RTX5000" is the Turing-era Quadro card, NOT an RTX 5000
    Ada/5090 -- the model string rides through verbatim and the catalog
    decides (today it maps to the Turing RTX_5000 entry, which sits below
    RTX_5000_ADA exactly to keep this label off the Ada part).

Transport trap: the FR body is ~8.14 MB -- 97% of gpu_index.common.http's
MAX_RESPONSE_BYTES (8 MiB). Catalog growth of ~3% will make the fetch
refuse the body and fail this source loudly (fail-closed, never truncated);
book_stats records body_bytes per book and a partial_error fires above 85%
of the cap so the drift is visible in every capture before it breaks.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import MAX_RESPONSE_BYTES, fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "ovh"

# One entry per OVH billing subsidiary (separate legal entities, separate
# hosts, separate currencies). ovhSubsidiary=US is INVALID on api.ovh.com
# (400) -- the US book only exists on the us.ovhcloud.com host. Config
# options.subsidiaries overrides this list verbatim.
DEFAULT_SUBSIDIARIES = (
    {
        "endpoint": (
            "https://api.ovh.com/1.0/order/catalog/public/cloud"
            "?ovhSubsidiary=FR"
        ),
        "expected_currency": "EUR",
    },
    {
        "endpoint": (
            "https://api.us.ovhcloud.com/1.0/order/catalog/public/cloud"
            "?ovhSubsidiary=US"
        ),
        "expected_currency": "USD",
    },
)

# Fetch-count fence: each subsidiary is one ~2-8 MB GET; more than 4 books
# is a deliberate renegotiation of the lane's fetch budget, not a config
# tweak.
MAX_SUBSIDIARIES = 4

PRICE_UNITS_PER_CURRENCY = 10 ** 8  # pricings[].price integer scale
HOURS_PER_MONTH = 730.0  # lane-wide monthly normalization convention

_CONSUMPTION_SUFFIX = ".consumption"
_MONTHLY_SUFFIX = ".monthly.postpaid"

# Early warning well before the transport cap actually bites (see module
# docstring -- FR is at 97% today).
_BODY_CAP_WARN_FRACTION = 0.85

# formattedPrice must be ONE number (<=2 decimals, "." or "," separator)
# wrapped in non-digit symbol text ("2.80 EUR-sign", "$1.56 USD"). A
# thousands-separated or otherwise reshaped display fails the match and the
# row is skipped+noted -- the 1e-8 cross-check is mandatory, not optional.
_FORMATTED_NUM_RE = re.compile(
    r"^[^0-9]*([0-9]+(?:[.,][0-9]{1,2})?)[^0-9]*$"
)
_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")

# Half-cent tolerance: formattedPrice is the cent-rounded display of the
# exact 1e-8 integer. Any wider and a scale drift could hide; any narrower
# and every legitimately sub-cent price (0.9776 -> "0.98") would fail.
_DISPLAY_TOLERANCE = 0.005 + 1e-9


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_ovh(
    body: str, *, expected_currency: str, expected_subsidiary: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of one subsidiary's catalog body.

    Returns (observations, skipped_notes); skipped_notes feed
    partial_errors so an unpinnable row is a visible hole, never a guess.
    Book-level identity problems (wrong currency/subsidiary, reshaped
    addons, zero GPU rows, duplicate planCodes) RAISE -- recording under
    the wrong book is worse than recording nothing.
    """
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"ovh: {expected_subsidiary} catalog payload is not a JSON "
            "object -- endpoint reshaped"
        )
    locale = payload.get("locale")
    if not isinstance(locale, dict):
        raise RuntimeError(
            f"ovh: {expected_subsidiary} catalog has no locale object -- "
            "cannot pin the book's currency; refusing to guess"
        )
    currency = locale.get("currencyCode")
    subsidiary = locale.get("subsidiary")
    if currency != expected_currency or subsidiary != expected_subsidiary:
        raise RuntimeError(
            f"ovh: expected the {expected_subsidiary}/{expected_currency} "
            f"book but the response says subsidiary={subsidiary!r} "
            f"currency={currency!r} -- refusing to record prices under the "
            "wrong billing entity"
        )
    addons = payload.get("addons")
    if not isinstance(addons, list):
        raise RuntimeError(
            f"ovh: {expected_subsidiary} catalog 'addons' is no longer a "
            "list -- shape changed; refusing to guess"
        )
    gpu_addons = [
        a
        for a in addons
        if isinstance(a, dict)
        and ((a.get("blobs") or {}).get("commercial") or {}).get("brick")
        == "gpu"
    ]
    if not gpu_addons:
        raise RuntimeError(
            f"ovh: {expected_subsidiary} book has ZERO brick=='gpu' addons "
            f"(of {len(addons)} total) -- GPU lineup pulled or brick "
            "vocabulary changed; refusing to record an empty book silently"
        )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    seen_plan_codes: set = set()
    region = f"{expected_subsidiary} subsidiary"
    for addon in sorted(gpu_addons, key=lambda a: str(a.get("planCode"))):
        plan_code = str(addon.get("planCode") or "").strip()
        if not plan_code:
            skipped.append(
                f"{expected_subsidiary}: gpu addon without planCode -- "
                "skipped, no flavor identity to pin"
            )
            continue
        if plan_code in seen_plan_codes:
            raise RuntimeError(
                f"ovh: planCode {plan_code!r} appears twice in the "
                f"{expected_subsidiary} book -- catalog reshaped, "
                "refusing to double-print"
            )
        seen_plan_codes.add(plan_code)

        blobs = addon.get("blobs") or {}
        technical = blobs.get("technical") or {}
        gpu = technical.get("gpu") or {}
        model = str(gpu.get("model") or "").strip()
        number = gpu.get("number")
        if not model:
            skipped.append(
                f"{expected_subsidiary}/{plan_code}: gpu addon without "
                "blobs.technical.gpu.model -- skipped, no structured part "
                "label to pin identity to"
            )
            continue
        if not _is_number(number) or number < 1 or int(number) != number:
            skipped.append(
                f"{expected_subsidiary}/{plan_code}: gpu.number {number!r} "
                "is not a whole count >= 1 -- per-GPU normalization "
                "impossible, skipped"
            )
            continue
        count = int(number)

        if plan_code.endswith(_CONSUMPTION_SUFFIX):
            base = plan_code[: -len(_CONSUMPTION_SUFFIX)]
            tier = "on-demand"
            hours = 1.0
            unit_suffix = "per_node_hr"
            period_note = "/hr"
        elif plan_code.endswith(_MONTHLY_SUFFIX):
            base = plan_code[: -len(_MONTHLY_SUFFIX)]
            tier = "monthly-commit"
            hours = HOURS_PER_MONTH
            unit_suffix = "per_node_month"
            period_note = "/month (normalized at 730h/month)"
        else:
            skipped.append(
                f"{expected_subsidiary}/{plan_code}: unknown billing-mode "
                "suffix (neither .consumption nor .monthly.postpaid) -- "
                "skipped, refusing to guess the period"
            )
            continue

        pricings = addon.get("pricings")
        if not isinstance(pricings, list) or len(pricings) != 1:
            n = len(pricings) if isinstance(pricings, list) else "no"
            skipped.append(
                f"{expected_subsidiary}/{plan_code}: {n} pricing phases "
                "(expected exactly 1) -- skipped, refusing to choose "
                "between phases"
            )
            continue
        pricing = pricings[0] if isinstance(pricings[0], dict) else {}
        price_units = pricing.get("price")
        if (
            not isinstance(price_units, int)
            or isinstance(price_units, bool)
            or price_units <= 0
        ):
            skipped.append(
                f"{expected_subsidiary}/{plan_code}: price {price_units!r} "
                "is not a positive integer of 1e-8 currency units -- "
                "unpriced or rescaled row skipped"
            )
            continue
        formatted = str(pricing.get("formattedPrice") or "")
        shown_m = _FORMATTED_NUM_RE.match(formatted)
        if not shown_m:
            skipped.append(
                f"{expected_subsidiary}/{plan_code}: formattedPrice "
                f"{formatted!r} unparseable -- the mandatory 1e-8 scale "
                "cross-check cannot run, row skipped"
            )
            continue
        node_price = price_units / PRICE_UNITS_PER_CURRENCY
        shown = float(shown_m.group(1).replace(",", "."))
        if abs(node_price - shown) > _DISPLAY_TOLERANCE:
            skipped.append(
                f"{expected_subsidiary}/{plan_code}: price {price_units} "
                f"(/1e8 = {node_price:g}) disagrees with its own "
                f"formattedPrice {formatted!r} -- scale convention broken, "
                "row failed"
            )
            continue

        os_family = (technical.get("os") or {}).get("family")
        windows = os_family == "windows"
        tags = blobs.get("tags")
        extra: Dict[str, Any] = {
            "plan_code": plan_code,
            "subsidiary": expected_subsidiary,
            "os_family": os_family,
            "tags": tags if isinstance(tags, list) else None,
            "formatted_price": formatted,
        }
        if windows:
            extra["windows_license_included"] = True
        if "-le-" in base:
            extra["legacy_le_variant"] = True
        mem_iface = (gpu.get("memory") or {}).get("interface")
        if mem_iface:
            extra["gpu_memory_interface"] = mem_iface

        notes = (
            f"{base} node {node_price:g} {expected_currency}"
            f"{period_note} ex-VAT"
        )
        if windows:
            notes += ", Windows license included"

        obs = observation(
            sku_identifier=model,
            price_per_gpu_hr=node_price / hours / count,
            currency=expected_currency,
            raw_value=str(price_units),
            raw_unit=f"{expected_currency.lower()}_1e-8_{unit_suffix}",
            gpu_count_basis=count,
            tier=tier,
            region=region,
            notes=notes,
            extra=extra,
        )
        mem = (gpu.get("memory") or {}).get("size")
        if _is_number(mem) and mem > 0 and int(mem) == mem:
            obs["memory_gb_label"] = int(mem)
        rows.append(obs)
    return rows, skipped


def _subsidiary_specs(
    options: Optional[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """Validated (endpoint, expected_currency, expected_subsidiary) specs.

    The expected subsidiary is read from the endpoint's own ovhSubsidiary
    query parameter, so the request and the response-locale pin can never
    be configured apart.
    """
    subs = (options or {}).get("subsidiaries", DEFAULT_SUBSIDIARIES)
    if not isinstance(subs, (list, tuple)) or not subs:
        raise ValueError(
            "ovh: options.subsidiaries must be a non-empty list of "
            "{endpoint, expected_currency} objects"
        )
    if len(subs) > MAX_SUBSIDIARIES:
        raise ValueError(
            f"ovh: {len(subs)} subsidiaries configured -- more than "
            f"{MAX_SUBSIDIARIES} multi-MB catalog fetches per capture is a "
            "fetch-budget renegotiation, not a config tweak"
        )
    specs: List[Tuple[str, str, str]] = []
    seen_subsidiaries: set = set()
    for entry in subs:
        if not isinstance(entry, dict):
            raise ValueError(
                f"ovh: subsidiary entry is not an object: {entry!r}"
            )
        endpoint = entry.get("endpoint")
        currency = entry.get("expected_currency")
        if not isinstance(endpoint, str) or not endpoint.startswith(
            "https://"
        ):
            raise ValueError(
                f"ovh: subsidiary endpoint must be an https URL, got "
                f"{endpoint!r}"
            )
        if not isinstance(currency, str) or not _CURRENCY_CODE_RE.match(
            currency
        ):
            raise ValueError(
                f"ovh: expected_currency must be a 3-letter code, got "
                f"{currency!r}"
            )
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(endpoint).query)
        sub_values = query.get("ovhSubsidiary", [])
        if len(sub_values) != 1 or not sub_values[0]:
            raise ValueError(
                f"ovh: endpoint {endpoint!r} must carry exactly one "
                "ovhSubsidiary query parameter -- it is the book identity "
                "the response locale is pinned against"
            )
        subsidiary = sub_values[0]
        if subsidiary in seen_subsidiaries:
            raise ValueError(
                f"ovh: subsidiary {subsidiary!r} configured twice -- "
                "refusing to double-print a book"
            )
        seen_subsidiaries.add(subsidiary)
        specs.append((endpoint, currency, subsidiary))
    return specs


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    specs = _subsidiary_specs(options)
    observations: List[Dict[str, Any]] = []
    partial_errors: List[str] = []
    books: List[Dict[str, Any]] = []
    for endpoint, currency, subsidiary in specs:
        body = fetch(endpoint, timeout=timeout)
        body_bytes = len(body.encode("utf-8"))
        if body_bytes > _BODY_CAP_WARN_FRACTION * MAX_RESPONSE_BYTES:
            partial_errors.append(
                f"{subsidiary}: catalog body {body_bytes} bytes is "
                f"{100.0 * body_bytes / MAX_RESPONSE_BYTES:.0f}% of the "
                f"{MAX_RESPONSE_BYTES}-byte transport cap -- the fetch "
                "will start failing outright if the catalog keeps growing"
            )
        rows, skipped = parse_ovh(
            body,
            expected_currency=currency,
            expected_subsidiary=subsidiary,
        )
        observations.extend(rows)
        partial_errors.extend(skipped)
        books.append(
            {
                "subsidiary": subsidiary,
                "endpoint": endpoint,
                "currency": currency,
                "observations": len(rows),
                "body_bytes": body_bytes,
            }
        )
    return result(
        SOURCE_ID,
        method="api-json",
        url=specs[0][0],
        observations=observations,
        partial_errors=partial_errors or None,
        book_stats={"subsidiaries": books},
    )
