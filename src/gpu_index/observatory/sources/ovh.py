# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""OVHcloud -- public no-auth order-catalog JSON, both billing subsidiaries.

Four GETs per capture: two cloud price catalogs (fetch count fixed by the
configured subsidiary list, capped at MAX_SUBSIDIARIES) plus two dedicated-
server availability books (fixed, see below -- a deliberate renegotiation
of the original two-GET budget for the availability-accrual
lane). The cloud catalogs: the FR
book on api.ovh.com (bills EUR; the full GPU lineup incl. H100/H200/A100)
and the SEPARATE US legal entity's book on api.us.ovhcloud.com (bills USD;
L4/L40S/V100S only -- a genuinely smaller lineup, not a parse failure).
The two entities publish different
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

Baremetal GPU availability (availability accrual, live evidence 2026-08-25): two more
GETs per capture read the dedicated-server availability books -- the EU
entity's on api.ovh.com (8,973 entries, 169 with a "gpu" hardware part,
4.33 MB) and the US entity's on api.us.ovhcloud.com (15,129 entries, 319
gpu, 6.34 MB -- 76% of the transport cap, so both books ride the same
_BODY_CAP_WARN_FRACTION tripwire as the FR catalog). Fetched UNFILTERED:
the server-side ?gpu= filter is verified working but drift-blind (a
filtered fetch can never surface a NEW gpu part name), so we filter
client-side on a truthy "gpu" key. Two landing channels:

  - book_stats["dedicated_gpu_availability"][EU|US] holds the verbatim
    GPU rows (fqn, planCode, gpu, datacenters[{datacenter, availability}])
    so nothing is dropped when a part has no cloud twin (V100S, RX6700XT
    today);
  - cloud observations whose blobs.technical.gpu.model matches a parsed
    baremetal part model (L4 and L40S today; gpu-{N}x{vendor}-{model}
    [-{mem}g] -> gpu_count + label, tesla-v100s -> V100S) get
    extra.dedicated_gpu_availability -- per-part {gpu_part, gpu_count,
    plan_codes, per_datacenter: {dc: {state: config count}}}, EU-vs-US
    deduped by fqn+datacenter BEFORE the rollup (the US book carries
    -eu/-ca plan variants) -- plus an explicit
    extra.dedicated_availability_product_line = "baremetal": this signal
    covers BAREMETAL GPU only; the Public Cloud GPU flavors priced above
    have NO availability surface at all (OVH staff confirmed none exists
    even authenticated), so baremetal stock must never be read as
    cloud-instance stock.

Availability-state semantics are inferred from OVH's order funnel, NOT
documented in the response: 1H-low/1H-high ~ deliverable within ~1h at
low/high stock depth; NNNH lead-time buckets (72H, 240H, 720H, 1440H all
live 2026-08-25); plus unavailable/comingSoon/unknown. "unknown" dominates
the eu-west-par-a/b/c local-zone rows (145 of 522 EU GPU dc-states) --
recorded verbatim, the consumer decides whether unknown means unavailable.
A string outside this vocabulary is noted in partial_errors (once per
distinct string per book) and still recorded, never failed. Empty
datacenters lists are REAL (all 94 23scalegpu0*-v1-eu US L4 configs live
2026-08-25): recorded verbatim, zero rollup contribution, no note. The
whole availability channel is fail-open per the lane ruling -- fetch or
parse failure of either book lands in partial_errors and drops that book
only, price rows are never gated -- while WITHIN the parse the book-level
fences stay fail-closed (non-list payload, zero gpu rows, duplicate fqn
all RAISE into that book's partial_error).

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
# tweak. (The two fixed dedicated-availability GETs below sit outside this
# fence -- adding them WAS the availability-accrual renegotiation.)
MAX_SUBSIDIARIES = 4

# One fixed extra GET per billing entity: the dedicated-server (baremetal)
# availability book. EU lives on api.ovh.com (the same host that serves
# the FR cloud catalog -- one book for the whole EU entity), US on the
# separate US legal entity's host. Deliberately NOT configurable and NOT
# filtered server-side (?gpu= works but is drift-blind -- it can never
# surface a new gpu part name); the "gpu" key is filtered client-side.
DEDICATED_AVAILABILITY_BOOKS = (
    (
        "EU",
        "https://api.ovh.com/v1/dedicated/server/datacenter/availabilities",
    ),
    (
        "US",
        "https://api.us.ovhcloud.com/1.0/dedicated/server/datacenter"
        "/availabilities",
    ),
)

# Availability vocabulary seen live 2026-08-25 (semantics inferred, see
# module docstring). Anything outside this set AND outside the NNNH
# lead-time family is noted in partial_errors (once per distinct string
# per book) and still recorded verbatim -- new vocabulary must surface,
# never fail the parse.
KNOWN_AVAILABILITY_STATES = frozenset(
    {"1H-low", "1H-high", "unavailable", "comingSoon", "unknown"}
)
# 72H on GPU rows; 240H/720H/1440H live on non-GPU US rows 2026-08-25.
_LEAD_TIME_STATE_RE = re.compile(r"^[0-9]+H$")

# gpu hardware-part label: gpu-{N}x{vendor}-{model}[-{mem}g]. Parsed into
# (gpu_count, model label) -- the label is the join key against the cloud
# book's blobs.technical.gpu.model (l4 -> L4, l40s-48g -> L40S,
# tesla-v100s -> V100S, radeon rx6700xt-12g -> RX6700XT); the part string
# itself always rides verbatim. A new vendor token fails the parse into a
# partial_error note (visible, non-fatal) rather than a guessed label.
_GPU_PART_RE = re.compile(r"^gpu-([0-9]+)x(nvidia|radeon)-([a-z0-9-]+)$")
_GPU_PART_MEM_SUFFIX_RE = re.compile(r"-[0-9]+g$")
_GPU_PART_TESLA_PREFIX = "tesla-"

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


def _parse_gpu_part(part: str) -> Optional[Tuple[int, str]]:
    """(gpu_count, model label) from a gpu hardware-part string, or None.

    None means "keep the row verbatim in book_stats but do not join it to
    cloud observations" -- an unrecognized part shape must never guess a
    model label.
    """
    m = _GPU_PART_RE.match(part)
    if not m:
        return None
    count = int(m.group(1))
    if count < 1:
        return None
    tail = _GPU_PART_MEM_SUFFIX_RE.sub("", m.group(3))
    if tail.startswith(_GPU_PART_TESLA_PREFIX):
        tail = tail[len(_GPU_PART_TESLA_PREFIX):]
    if not tail:
        return None
    return count, tail.upper()


def parse_dedicated_availabilities(
    body: str, *, book: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Pure parse of one dedicated-availability book down to its GPU rows.

    Returns (gpu_rows, notes): gpu_rows carry exactly the four recorded
    fields (fqn, planCode, gpu, datacenters[{datacenter, availability}])
    verbatim; notes feed partial_errors. Availability is a metadata
    channel -- collect() catches anything raised here so a reshape can
    never dark the price lane -- but WITHIN the parse the book-level
    fences are fail-closed: a payload that is not a list, has zero
    gpu-part rows, or repeats an fqn RAISES rather than recording a
    silently-empty or double-counted availability book.
    """
    payload = json.loads(body)
    if not isinstance(payload, list):
        raise RuntimeError(
            f"ovh: dedicated {book} availability payload is not a JSON "
            "list of config entries -- endpoint reshaped"
        )
    rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    seen_fqns: set = set()
    unknown_states: Dict[str, int] = {}
    for entry in payload:
        # Non-GPU configs carry no "gpu" key at all (live 2026-08-25:
        # 8,804 of 8,973 EU entries) -- out of scope, not failures. A
        # falsy value would be equally join-less and is skipped the same
        # silent way; only a truthy NON-STRING part is a reshape note.
        if not isinstance(entry, dict) or not entry.get("gpu"):
            continue
        gpu = entry["gpu"]
        if not isinstance(gpu, str):
            notes.append(
                f"dedicated {book}: entry {entry.get('fqn')!r} carries a "
                f"non-string gpu part {gpu!r} -- row skipped, no part "
                "identity to record"
            )
            continue
        fqn = entry.get("fqn")
        plan_code = entry.get("planCode")
        if (
            not isinstance(fqn, str)
            or not fqn
            or not isinstance(plan_code, str)
            or not plan_code
        ):
            notes.append(
                f"dedicated {book}: gpu row (part {gpu!r}) without string "
                "fqn/planCode -- row skipped, no config identity to pin"
            )
            continue
        if fqn in seen_fqns:
            raise RuntimeError(
                f"ovh: dedicated {book} fqn {fqn!r} appears twice -- book "
                "reshaped, refusing to double-count configs"
            )
        seen_fqns.add(fqn)
        dcs = entry.get("datacenters")
        if not isinstance(dcs, list):
            notes.append(
                f"dedicated {book}/{fqn}: datacenters is not a list -- "
                "row skipped, refusing to guess states"
            )
            continue
        dc_rows: List[Dict[str, str]] = []
        reshaped = False
        for dc in dcs:
            name = dc.get("datacenter") if isinstance(dc, dict) else None
            state = dc.get("availability") if isinstance(dc, dict) else None
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(state, str)
                or not state
            ):
                reshaped = True
                break
            if state not in KNOWN_AVAILABILITY_STATES and not (
                _LEAD_TIME_STATE_RE.match(state)
            ):
                unknown_states[state] = unknown_states.get(state, 0) + 1
            dc_rows.append({"datacenter": name, "availability": state})
        if reshaped:
            notes.append(
                f"dedicated {book}/{fqn}: datacenters entries reshaped "
                "(expected {datacenter, availability} string pairs) -- "
                "row skipped, refusing to guess states"
            )
            continue
        # Empty datacenters lists are REAL (94 US -v1-eu L4 configs live
        # 2026-08-25) -- recorded verbatim, zero rollup contribution.
        rows.append(
            {
                "fqn": fqn,
                "planCode": plan_code,
                "gpu": gpu,
                "datacenters": dc_rows,
            }
        )
    if not rows:
        raise RuntimeError(
            f"ovh: dedicated {book} availability book has ZERO gpu-part "
            f"rows (of {len(payload)} entries) -- GPU baremetal lineup "
            "pulled or 'gpu' vocabulary changed; refusing to record an "
            "empty availability book silently"
        )
    for state in sorted(unknown_states):
        notes.append(
            f"dedicated {book}: availability state {state!r} "
            f"(x{unknown_states[state]}) is outside the known vocabulary "
            "-- recorded verbatim; extend the vocabulary deliberately"
        )
    return rows, notes


def dedicated_gpu_summaries(
    books: List[Tuple[str, List[Dict[str, Any]]]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Rollup of parsed GPU rows: model label -> per-part summaries.

    Dedupes EU-vs-US overlap by fqn+datacenter BEFORE the rollup (zero
    overlapping fqns live 2026-08-25, but the US book carries -eu/-ca plan
    variants so the fence stays; a deduped pair whose states disagree is
    noted, first-seen kept). per_datacenter maps each datacenter to
    {state: config count} -- counts, not one state, because sibling
    configs of one part genuinely disagree (23scalegpu01-v2 ram-192g
    gra="72H" vs ram-384g gra="1H-low" live) and picking a single state
    would be a semantic ranking that belongs to the consumer.
    """
    seen: Dict[Tuple[str, str], str] = {}
    notes: List[str] = []
    parsed_parts: Dict[str, Optional[Tuple[int, str]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    plan_codes: Dict[str, set] = {}
    for book, rows in books:
        for row in rows:
            part = row["gpu"]
            if part not in parsed_parts:
                parsed_parts[part] = _parse_gpu_part(part)
                if parsed_parts[part] is None:
                    notes.append(
                        f"dedicated {book}: gpu part {part!r} does not "
                        "parse as gpu-{N}x{vendor}-{model} -- kept "
                        "verbatim in book_stats, excluded from the "
                        "cloud-observation join"
                    )
            parsed = parsed_parts[part]
            if parsed is None:
                continue
            count, model = parsed
            summary = summaries.setdefault(
                part,
                {
                    "gpu_part": part,
                    "gpu_count": count,
                    "model": model,
                    "per_datacenter": {},
                },
            )
            plan_codes.setdefault(part, set()).add(row["planCode"])
            for dc in row["datacenters"]:
                key = (row["fqn"], dc["datacenter"])
                state = dc["availability"]
                if key in seen:
                    if seen[key] != state:
                        notes.append(
                            f"dedicated {book}/{row['fqn']}/"
                            f"{dc['datacenter']}: duplicate "
                            f"fqn+datacenter across books disagrees "
                            f"({seen[key]!r} vs {state!r}) -- first-seen "
                            "kept in the rollup"
                        )
                    continue
                seen[key] = state
                per_dc = summary["per_datacenter"].setdefault(
                    dc["datacenter"], {}
                )
                per_dc[state] = per_dc.get(state, 0) + 1
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for part in sorted(summaries):
        summary = summaries[part]
        summary["plan_codes"] = sorted(plan_codes[part])
        summary["per_datacenter"] = {
            dc: dict(sorted(states.items()))
            for dc, states in sorted(summary["per_datacenter"].items())
        }
        by_model.setdefault(summary.pop("model"), []).append(summary)
    return by_model, notes


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

    # Dedicated-server (baremetal) GPU availability -- metadata channel,
    # fail-open per book: a fetch or parse failure here becomes a
    # partial_error and drops THAT book's availability only; the price
    # observations above are already parsed and are never gated.
    dedicated_rows: Dict[str, List[Dict[str, Any]]] = {}
    for book, url in DEDICATED_AVAILABILITY_BOOKS:
        try:
            avail_body = fetch(url, timeout=timeout)
        except Exception as exc:
            partial_errors.append(
                f"dedicated {book}: availability fetch failed ({exc}) -- "
                "baremetal availability dropped this capture; price rows "
                "unaffected"
            )
            continue
        avail_bytes = len(avail_body.encode("utf-8"))
        if avail_bytes > _BODY_CAP_WARN_FRACTION * MAX_RESPONSE_BYTES:
            partial_errors.append(
                f"dedicated {book}: availability body {avail_bytes} bytes "
                f"is {100.0 * avail_bytes / MAX_RESPONSE_BYTES:.0f}% of "
                f"the {MAX_RESPONSE_BYTES}-byte transport cap -- the "
                "fetch will start failing outright if the book keeps "
                "growing"
            )
        try:
            gpu_rows, notes = parse_dedicated_availabilities(
                avail_body, book=book
            )
        except Exception as exc:
            partial_errors.append(
                f"dedicated {book}: availability parse failed ({exc}) -- "
                "baremetal availability dropped this capture; price rows "
                "unaffected"
            )
            continue
        partial_errors.extend(notes)
        dedicated_rows[book] = gpu_rows
    if dedicated_rows:
        by_model, join_notes = dedicated_gpu_summaries(
            [
                (book, dedicated_rows[book])
                for book, _ in DEDICATED_AVAILABILITY_BOOKS
                if book in dedicated_rows
            ]
        )
        partial_errors.extend(join_notes)
        for obs in observations:
            summaries = by_model.get(obs["sku_identifier"].strip().upper())
            if summaries:
                extra = obs.setdefault("extra", {})
                extra["dedicated_gpu_availability"] = summaries
                extra["dedicated_availability_product_line"] = "baremetal"

    stats: Dict[str, Any] = {"subsidiaries": books}
    if dedicated_rows:
        stats["dedicated_gpu_availability"] = dedicated_rows
    return result(
        SOURCE_ID,
        method="api-json",
        url=specs[0][0],
        observations=observations,
        partial_errors=partial_errors or None,
        book_stats=stats,
    )
