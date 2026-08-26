# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Sesterce console -- cloud.sesterce.com/compute, the embedded live-offers
book (sibling surface to sesterce.py's homepage floor table).

One plain GET of the console's public marketplace page (no auth, no JS, no
special headers; cache-control no-store, rendered fresh per request on
Vercel -- unlike the homepage's ~5-min ISR, so the two surfaces can
legitimately disagree within minutes). The page server-renders the FULL
/gpu-cloud/instances/offers response -- the same data the key-gated API
documented at docs.sesterce.com would return -- into its Next.js RSC flight
payload (verified live 2026-08-25: 82 offers across 21 gpuNames, 253
per-offer region entries of which 110 available=true). Each offer is a
rentable config (chip x gpuCount x deploymentType) carrying an
INSTANCE-TOTAL hourlyPrice and a per-region availability array of explicit
booleans -- the richest availability grain Sesterce publishes, recorded
verbatim in extra; the homepage's Available now / Live regions cells are
derived marketing stats over this book.

FLIGHT-PAYLOAD PINS (a Next.js internal, NOT a contract -- every pin fails
closed):

  1. collect every self.__next_f.push([1,"..."]) chunk string IN ORDER and
     join BEFORE scanning -- the payload streams across ~20 script chunks
     and CAN split mid-JSON (live 2026-08-25 the offers array happens to
     sit inside one 74KB chunk; never rely on that); zero chunks raises;
  2. unescape the joined text as ONE JSON string literal (Next emits
     JSON.stringify-escaped chunk strings; an escape json.loads refuses is
     a reshape and raises);
  3. require exactly one '"filteredOffers":[' in the whole flight text --
     the RSC reference strings ('$1b:1:props:filteredOffers:N', 51x live
     2026-08-25) elsewhere in the payload do NOT match the key pin;
  4. balanced-bracket slice the array (string-aware scan; an unbalanced
     array raises) and json.loads it; a '$undefined' inside the slice
     raises (Next's non-JSON sentinel -- present elsewhere in the payload,
     absent from the array today);
  5. per-offer load-bearing fields are type-pinned fail-closed (gpuName /
     gpuCount / hourlyPrice / instanceId / deploymentType / the
     availability list and its per-entry 'available' booleans) -- a
     reshaped offer raises, never guesses.

Row honesty: sku_identifier is the RAW gpuName ('A100_80G', 'RTXPro6000'
stay verbatim -- the framework derives canonical skus, collectors never
claim them). hourlyPrice is INSTANCE-TOTAL, so per-GPU = hourlyPrice /
gpuCount with gpu_count_basis carrying the divisor and raw_value the
untouched instance figure (float artifacts like 30.008000000000003 kept).
Currency: the book publishes no currency field; USD is pinned only by the
console UI's own dollar prints and the sibling homepage's USD footer
marker -- a silent upstream flip would be invisible here, which is why the
homepage source keeps its byte-exact currency pin. The homepage 'From /
GPU / hr' floor does NOT reconcile against this book live 2026-08-25
(homepage H100 $1.83 vs console H100 min $2.189/GPU) -- floor provenance
unproven, so the two surfaces are recorded independently and never
reconciled at collect time. 'filteredOffers' implies server-applied
default filters: fetched with no query params it carried the full book and
the adjacent filter-state object showed availability filtering off, but a
console default-filter change would silently narrow the list -- book_stats
records the offer count per capture so a step-change is visible.

Rows that never print: gpuCount<1 offers (the CPU vm tiers cpu_small/
medium/large, gpuCount 0 live 2026-08-25) have no honest per-GPU divisor
-- skipped and counted, never guessed to a basis; a zero/negative
hourlyPrice is never a $0 print -- skipped and counted. Standing tripwire:
chips the homepage advertises that have ZERO offers in this book (B200 and
B300 live 2026-08-25) are noted in partial_errors each capture -- the book
is the truth for rentable capacity, the homepage row is marketing.

Availability semantics, partially unproven (2026-08-25): per-region
'available' booleans presumably mean deployable-now (false entries DO
appear on live offers, e.g. RTXPro6000x8 TYO4), but sold-out behavior is
unobserved -- 0 offers are all-false and 0 have empty availability arrays
today, so whether a fully-dark offer stays listed or silently delists is
unknown; treat book-count drops as delisting, not error. countryCode can
be null (baremetal regions like TYO4/AMS); region slugs are source-local
('TYO4', 'chicago-usa-4') and recorded verbatim; cloud.name values
('AZ_02', 'AZ_22') are anonymized upstream-provider tags -- Sesterce is
itself an aggregator -- recorded but never interpreted.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "sesterce_console"

URL = "https://cloud.sesterce.com/compute"
# One instance-total price per offer; per-region truth (explicit booleans)
# rides verbatim in extra.availability, never as a price axis.
REGION = "global"

# Every streamed flight chunk, in document order. Each push argument is an
# independently valid JSON-escaped string literal, but the UNDERLYING
# payload text can split mid-JSON across chunks -- join, then scan.
_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)')
_OFFERS_KEY = '"filteredOffers":['
# Chips the homepage advertises with zero offers in the console book, live
# 2026-08-25 -- their continued absence is noted per capture (tripwire,
# never a failure); their appearance in the book simply stops the note.
_HOMEPAGE_ONLY_CHIPS = ("B200", "B300")


def _flight_payload(html: str) -> str:
    """Join every streamed chunk in order and unescape once, fail-closed."""
    chunks = _CHUNK_RE.findall(html)
    if not chunks:
        raise RuntimeError(
            "sesterce_console: zero self.__next_f.push flight chunks on "
            "the page -- Next.js payload shape changed (framework upgrade "
            "or client-only rendering); refusing to guess"
        )
    try:
        return json.loads('"' + "".join(chunks) + '"')
    except ValueError as exc:
        raise RuntimeError(
            "sesterce_console: joined flight chunks are not one valid "
            f"JSON string literal ({exc}) -- chunk escaping changed; "
            "refusing to guess"
        ) from exc


def _offers_slice(flight: str) -> str:
    """The balanced '"filteredOffers":[...]' array text, fail-closed."""
    count = flight.count(_OFFERS_KEY)
    if count != 1:
        raise RuntimeError(
            f"sesterce_console: {_OFFERS_KEY!r} found {count}x in the "
            "flight payload (need exactly 1) -- payload reshaped or a "
            "lookalike prop appeared; refusing to guess which array is "
            "the offers book"
        )
    start = flight.index(_OFFERS_KEY) + len(_OFFERS_KEY) - 1
    depth = 0
    in_str = False
    escaped = False
    end = -1
    for i in range(start, len(flight)):
        char = flight[i]
        if in_str:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise RuntimeError(
            "sesterce_console: the filteredOffers array never closes -- "
            "flight payload truncated or reshaped; refusing to parse a "
            "partial book"
        )
    piece = flight[start:end]
    if "$undefined" in piece:
        raise RuntimeError(
            "sesterce_console: RSC '$undefined' sentinel inside the "
            "filteredOffers slice -- the array is no longer plain JSON; "
            "refusing to guess"
        )
    return piece


def _pin_str(offer: Dict[str, Any], key: str, where: str) -> str:
    value = offer.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"sesterce_console: {where}: {key} is {value!r}, not a "
            "non-empty string -- offer shape changed; refusing to guess"
        )
    return value.strip()


def _pin_availability(
    offer: Dict[str, Any], where: str
) -> List[Dict[str, Any]]:
    entries = offer.get("availability")
    if not isinstance(entries, list):
        raise RuntimeError(
            f"sesterce_console: {where}: availability is "
            f"{type(entries).__name__}, not a list -- the per-region "
            "boolean grain vanished; refusing to guess"
        )
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("available"), bool
        ):
            raise RuntimeError(
                f"sesterce_console: {where}: availability entry {entry!r} "
                "misses the explicit boolean 'available' pin -- entry "
                "shape changed (a non-boolean would silently miscount "
                "regions_available); refusing to guess"
            )
    return entries


def parse_console_book(
    html: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Pure parse of the /compute page -> (observations, partial_errors,
    book_stats)."""
    piece = _offers_slice(_flight_payload(html))
    try:
        offers = json.loads(piece)
    except ValueError as exc:
        raise RuntimeError(
            "sesterce_console: pinned filteredOffers slice is not valid "
            f"JSON ({exc}) -- payload reshaped; refusing to guess"
        ) from exc
    if not isinstance(offers, list) or not offers:
        raise RuntimeError(
            "sesterce_console: pinned filteredOffers slice is "
            f"{'empty' if isinstance(offers, list) else 'not a list'} -- "
            "book pulled or reshaped; refusing to record a silent hole"
        )

    observations: List[Dict[str, Any]] = []
    skips: Dict[str, int] = {}
    region_entries = 0
    region_entries_available = 0
    gpu_names: set = set()
    for idx, offer in enumerate(offers):
        if not isinstance(offer, dict):
            raise RuntimeError(
                f"sesterce_console: offer {idx} is "
                f"{type(offer).__name__}, not an object -- array shape "
                "changed; refusing to guess"
            )
        where = f"offer {idx}"
        gpu_name = _pin_str(offer, "gpuName", where)
        gpu_names.add(gpu_name)
        where = f"offer {idx} ({gpu_name})"
        instance_id = _pin_str(offer, "instanceId", where)
        deployment = _pin_str(offer, "deploymentType", where)
        gpu_count = offer.get("gpuCount")
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int):
            raise RuntimeError(
                f"sesterce_console: {where}: gpuCount is {gpu_count!r}, "
                "not an integer -- per-GPU normalization basis lost; "
                "refusing to guess"
            )
        hourly = offer.get("hourlyPrice")
        if (
            isinstance(hourly, bool)
            or not isinstance(hourly, (int, float))
            or not math.isfinite(hourly)
        ):
            raise RuntimeError(
                f"sesterce_console: {where}: hourlyPrice is {hourly!r}, "
                "not a finite number -- price shape changed; refusing to "
                "guess"
            )
        availability = _pin_availability(offer, where)
        regions_listed = len(availability)
        regions_available = sum(
            1 for entry in availability if entry["available"] is True
        )
        region_entries += regions_listed
        region_entries_available += regions_available

        if gpu_count < 1:
            # The CPU vm tiers (cpu_small/medium/large, gpuCount 0 live
            # 2026-08-25): no honest per-GPU divisor -- never guessed to 1.
            skips["non_gpu_zero_count"] = (
                skips.get("non_gpu_zero_count", 0) + 1
            )
            continue
        if hourly <= 0:
            # A zero-priced offer is never a $0 print.
            skips["zero_price"] = skips.get("zero_price", 0) + 1
            continue

        config = offer.get("configuration")
        if config is not None and not isinstance(config, dict):
            raise RuntimeError(
                f"sesterce_console: {where}: configuration is "
                f"{type(config).__name__}, not an object -- offer shape "
                "changed; refusing to guess"
            )
        cloud = offer.get("cloud")
        if cloud is not None and not isinstance(cloud, dict):
            raise RuntimeError(
                f"sesterce_console: {where}: cloud is "
                f"{type(cloud).__name__}, not an object -- offer shape "
                "changed; refusing to guess"
            )
        config = config or {}
        extra: Dict[str, Any] = {
            "instance_id": instance_id,
            "deployment_type": deployment,
            "gpu_count": gpu_count,
            "hourly_price_instance": hourly,
            # Verbatim [{name, region, countryCode, available}] -- the
            # richest published availability grain; slugs source-local.
            "availability": availability,
            "regions_listed": regions_listed,
            "regions_available": regions_available,
            "spot_offer_available": offer.get("spotOfferAvailable"),
            "hourly_spot_price": offer.get("hourlySpotPrice"),
            "nvlink": offer.get("nvlink"),
            "interconnect": config.get("interconnect"),
            "vram_gb": config.get("vRamGB"),
            "cloud_name": (cloud or {}).get("name"),
        }
        observations.append(
            observation(
                sku_identifier=gpu_name,
                price_per_gpu_hr=hourly / gpu_count,
                raw_value=str(hourly),
                raw_unit="usd_per_instance_hr",
                gpu_count_basis=gpu_count,
                tier="on-demand",
                region=REGION,
                notes=(
                    f"console live-offers book: {instance_id} "
                    f"({deployment}), instance-total hourly rate for "
                    f"{gpu_count} GPU(s); per-region availability "
                    "booleans verbatim in extra"
                ),
                extra=extra,
            )
        )
    if not observations:
        raise RuntimeError(
            f"sesterce_console: {len(offers)} offers in the pinned book "
            "but ZERO survived the value pins ("
            + ", ".join(
                f"{count} {cat}" for cat, count in sorted(skips.items())
            )
            + ") -- book semantics changed; refusing to guess"
        )

    partial_errors: List[str] = []
    if skips:
        partial_errors.append(
            "skipped "
            + ", ".join(
                f"{count} {cat}" for cat, count in sorted(skips.items())
            )
            + " offer(s) (fail-closed pins; see module docstring)"
        )
    absent = [c for c in _HOMEPAGE_ONLY_CHIPS if c not in gpu_names]
    if absent:
        partial_errors.append(
            "homepage-advertised chip(s) with zero offers in the console "
            "book: " + ", ".join(absent) + " -- the book is the truth for "
            "rentable capacity, the homepage row is marketing"
        )

    stats = {
        "offers_in_book": len(offers),
        "offers_recorded": len(observations),
        "offers_skipped": sum(skips.values()),
        # Whole-book label census (includes skipped rows) so a delisting
        # or a default-filter narrowing shows as a step-change on record.
        "gpu_names": sorted(gpu_names),
        "region_entries": region_entries,
        "region_entries_available": region_entries_available,
    }
    return observations, partial_errors, stats


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    body = fetch(URL, timeout=timeout)
    observations, partial_errors, stats = parse_console_book(body)
    return result(
        SOURCE_ID,
        method="html",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
        book_stats=stats,
    )
