# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Voltage Park -- unauthenticated dashboard locations API, both network tiers.

Price surface: GET https://cloud-api.voltagepark.com/api/v1/bare-metal/locations
(the SPA dashboard's logged-out variant; no auth, no cookies). Verified live
2026-08-22, re-verified byte-identical 2026-08-25: results[] holds one
location row keyed by specs_per_node.gpu_model ('h100-sxm5-80gb', 8x per
node) with TWO published per-GPU hourly prices -- gpu_price_ethernet
("1.99") and gpu_price_infiniband ("2.49") -- as decimal strings. Both are networking variants of the same ON-DEMAND product (ethernet
vs 3200Gbps InfiniBand), NOT an on-demand/reserved split (long-term reserve is
contact-only and never published), so tier stays "on-demand" for both and the
voltagepark-native ``network`` label lives in extra -- same vocabulary rule as
the runpod exemplar's secure/community clouds.

Basis and currency facts (why the fields mean what we record):

  - prices are per GPU per HOUR: the dashboard app multiplies by gpuCount to
    build a total (per-GPU proven), and og:description prints "$1.99/GPU/hour"
    (per-hour + USD proven). NEVER multiply by specs_per_node.gpu_count -- the
    raw figure is already the per-GPU rate (gpu_count_basis=1).
  - the payload carries no currency field; USD is pinned from the provider's
    own dollar prints (pricing FAQ "starting at $1.99", og:description above)
    which exactly match the ethernet figure. The InfiniBand figure has NO
    static human-readable confirmation -- the marketing card says "Contact for
    pricing" -- but it renders through the same dashboard USD price path (zod
    schema + deploy-flow minPrice), noted per-observation in extra.
  - prices arrive as JSON strings today but the app schema accepts
    number-or-string, so parsing goes through Decimal(str(x)) and tolerates a
    type flip without loosening the >0 pin.

Fail-closed identity pins (every one earned by a near-miss):

  - top level must carry a 'results' list and 'total_result_count' int, the
    page must be complete (has_next falsy AND total == len(results)) -- one
    page today (total_result_count=1); if pagination ever activates this
    collector raises rather than silently under-reporting;
  - per row, specs_per_node.gpu_model must be a non-empty string and BOTH
    price fields must Decimal-parse > 0; any missing/renamed field raises with
    zero rows recorded. gpu_count_ethernet/gpu_count_infiniband are numeric
    LOOKALIKES for the price fields (and sat at 0 while prices stayed live at
    probe time), so availability never gates a price and the pin is on the
    exact price field names;
    (the counts are no longer MERE lookalikes -- since 2026-08-25 they are
    one leg of the corroborated 4-surface availability signal below -- but
    the price pin stays on the exact price field names regardless);
  - any UNKNOWN row key containing 'gpu_price' raises: a new price column
    (a spot tier, a reserved price going public) is a published price this
    recipe never saw -- same refuse-to-under-report rule as pagination;
  - churn risk is real (Lightning AI merger announced, TensorDock acquired,
    numeric pricing already pulled from the marketing site) -- these pins are
    the tripwire.

Availability capture (availability accrual, added 2026-08-25) -- three legs:

  - Rung 1: each bare-metal row also carries two reserved-capacity fields,
    reserved_gpu_count_infiniband_cluster_one/_two (nullable ints per the
    live openapi.json; both null 2026-08-25). Recorded verbatim into extra
    next to available_gpu_count -- nulls as-is, and like every availability
    field here they NEVER gate a price.
  - Rung 2: two more unauthenticated GETs per collect() run land verbatim
    in book_stats["vm_instant"]: /api/v1/instant-deploy-presets/ (4 H100 VM
    presets 2026-08-25, each with location_ids_with_availability -- the
    location UUIDs with stock RIGHT NOW -- plus per-INSTANCE hourly rates:
    "15.120000" is the 8-GPU preset, NEVER a per-GPU price) and
    /api/v1/virtual-machines/instant/locations (the spec's durable,
    non-deprecated replacement: rows carry available_presets[].available_vms
    integers; empty page 2026-08-25). The presets endpoint is DEPRECATED in
    the spec -- churn expected; prefer available_vms long-term.
  - Semantics are CLAIMED availability until a nonzero print lands: every
    field on every surface read 0/empty at both probes (2026-08-22 and
    2026-08-25) while prices stayed published. Four surfaces reading empty
    SIMULTANEOUSLY (gpu_count_*=0, all location_ids_with_availability=[],
    instant locations results=[], legacy /locations/ results=[]) corroborate
    "sold out" over junk fields, but anonymous callers being served zeroed
    availability is not excluded (the spec marks instant/locations as
    bearer-required yet it 200s anonymously). First nonzero observation is
    the semantics proof.

The VM surfaces do NOT emit observations (2026-08-25 ruling): voltagepark
is a seated source in the live H100 panels, which read raw-observatory
observations -- new 1x-H100 preset price rows (~$1.89/GPU-hr) would change
the seat's lowest-eligible print, a live-index composition change out of
scope for availability accrual. Preset prices ride as book_stats metadata
only. Both VM fetches are FAIL-OPEN (partial_errors) so a VM-surface
failure or reshape can never dark the proven bare-metal price lane; the
parse fences within each VM surface stay fail-closed and surface through
the same partial_errors channel.

Do NOT scrape the marketing /pricing page (it carries no numeric prices;
cards say "Contact for pricing"; $1.99 lives only in meta tags/FAQ
prose), and do NOT use
/api/v1/bare-metal/detailed-locations or /api/v1/billing/hourly-rate (both
account-scoped).
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "voltagepark"

URL = "https://cloud-api.voltagepark.com/api/v1/bare-metal/locations"

# VM availability surfaces (book_stats metadata only -- see docstring
# ruling; NEVER promoted to observations without a maintainer decision).
PRESETS_URL = "https://cloud-api.voltagepark.com/api/v1/instant-deploy-presets/"
VM_LOCATIONS_URL = (
    "https://cloud-api.voltagepark.com/api/v1/virtual-machines/instant/locations"
)

# One label, stamped on the whole vm_instant block: nonzero has never been
# observed on any surface (see docstring), so nothing here is proven stock.
_CLAIMED_AVAILABILITY = (
    "claimed-availability -- every field on every surface read 0/empty at "
    "both probes (2026-08-22, 2026-08-25) while prices stayed published; "
    "the first nonzero print is the semantics proof"
)

# (price field, count field, network) -- network is voltagepark-native
# vocabulary and lives in extra; tier stays in the lane-wide vocabulary
# ("on-demand" for both -- see module docstring).
_NETWORK_SURFACES = (
    ("gpu_price_ethernet", "gpu_count_ethernet", "ethernet"),
    ("gpu_price_infiniband", "gpu_count_infiniband", "infiniband"),
)

_KNOWN_PRICE_FIELDS = frozenset(field for field, _, _ in _NETWORK_SURFACES)


def _pin_price(row_label: str, field: str, value: Any) -> Decimal:
    """The exact-name price pin: missing/renamed/unparseable/non-positive
    raises (zero rows) -- a silent skip would drop a whole tier while the
    source looked healthy, and the count fields are numeric lookalikes."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RuntimeError(
            f"voltagepark: {row_label}: {field} is {value!r} -- price field "
            "missing or reshaped (count fields are numeric lookalikes; the "
            "pin is on the exact price field name); refusing to guess"
        )
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise RuntimeError(
            f"voltagepark: {row_label}: {field} {value!r} does not "
            "Decimal-parse -- price field reshaped; refusing to guess"
        ) from exc
    if not price.is_finite() or price <= 0:
        raise RuntimeError(
            f"voltagepark: {row_label}: {field} {value!r} is not a positive "
            "price -- not a real print; refusing to record"
        )
    return price


def parse_voltagepark(body: str) -> List[Dict[str, Any]]:
    payload = json.loads(body)
    results = payload.get("results") if isinstance(payload, dict) else None
    total = payload.get("total_result_count") if isinstance(payload, dict) else None
    if (
        not isinstance(results, list)
        or isinstance(total, bool)
        or not isinstance(total, int)
    ):
        raise RuntimeError(
            "voltagepark: top level reshaped -- expected a 'results' list and "
            "a 'total_result_count' int; refusing to guess at the new shape"
        )
    if payload.get("has_next"):
        raise RuntimeError(
            "voltagepark: has_next is truthy -- the single-page assumption "
            "broke; add pagination rather than silently under-reporting"
        )
    if total != len(results):
        raise RuntimeError(
            f"voltagepark: total_result_count {total} != {len(results)} rows "
            "in the page -- rows exist this fetch never saw; refusing to "
            "under-report"
        )
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"voltagepark: results[{idx}] is not an object -- row shape "
                "changed; refusing to guess"
            )
        row_label = f"row {row.get('id') or idx}"
        unknown_price_keys = sorted(
            key
            for key in row
            if "gpu_price" in key and key not in _KNOWN_PRICE_FIELDS
        )
        if unknown_price_keys:
            raise RuntimeError(
                f"voltagepark: {row_label} carries unknown price field(s) "
                f"{unknown_price_keys} -- a published price this recipe "
                "never saw (new tier?); refusing to under-report"
            )
        specs = row.get("specs_per_node")
        if not isinstance(specs, dict):
            raise RuntimeError(
                f"voltagepark: {row_label} lost specs_per_node -- identity "
                "anchor gone; refusing to guess"
            )
        gpu_model = specs.get("gpu_model")
        if not isinstance(gpu_model, str) or not gpu_model.strip():
            raise RuntimeError(
                f"voltagepark: {row_label} has no specs_per_node.gpu_model "
                "string -- no structured label to pin identity to; refusing "
                "to guess"
            )
        gpu_model = gpu_model.strip()
        node_gpus = specs.get("gpu_count")
        node_note = (
            f" ({node_gpus}x per node)"
            if isinstance(node_gpus, int) and not isinstance(node_gpus, bool)
            else ""
        )
        for price_field, count_field, network in _NETWORK_SURFACES:
            price = _pin_price(row_label, price_field, row.get(price_field))
            rows.append(
                observation(
                    sku_identifier=gpu_model,
                    price_per_gpu_hr=float(price),
                    # raw figure exactly as published (a decimal string
                    # today; str() keeps a number flip byte-honest too).
                    raw_value=str(row.get(price_field)),
                    gpu_count_basis=1,
                    tier="on-demand",
                    # Location rows carry only an opaque UUID -- no honest
                    # human region name to claim; the id rides in extra.
                    region="?",
                    notes=f"{network} networking{node_note}",
                    extra={
                        "network": network,
                        "location_id": row.get("id"),
                        # Availability metadata, NEVER a price gate -- both
                        # counts were 0 at probe time while prices stayed
                        # published. The reserved cluster counts are nullable
                        # ints per the live openapi.json (null 2026-08-25):
                        # nulls recorded as-is, same rule.
                        "available_gpu_count": row.get(count_field),
                        "reserved_gpu_count_infiniband_cluster_one": row.get(
                            "reserved_gpu_count_infiniband_cluster_one"
                        ),
                        "reserved_gpu_count_infiniband_cluster_two": row.get(
                            "reserved_gpu_count_infiniband_cluster_two"
                        ),
                        "node_gpu_count": node_gpus,
                        "cpu_model": specs.get("cpu_model"),
                        "ram_gb": specs.get("ram_gb"),
                        "storage_gb": specs.get("storage_gb"),
                        # Currency provenance (no currency field in payload):
                        # ethernet figure matches the provider's own "$1.99/
                        # GPU/hour" prints; infiniband is dashboard/API-only.
                        "usd_basis": (
                            "provider dollar prints"
                            if network == "ethernet"
                            else "dashboard price path only (marketing says "
                            "contact for pricing)"
                        ),
                    },
                )
            )
    return rows


def parse_vm_presets(body: str) -> Dict[str, Any]:
    """Verbatim capture of /instant-deploy-presets/ for book_stats. Fences
    fail-closed WITHIN this parse (collect() catches them into
    partial_errors); the only derived figure is available_location_count
    per preset -- everything else rides exactly as published."""
    payload = json.loads(body)
    if not isinstance(payload, list):
        raise RuntimeError(
            "voltagepark: instant-deploy-presets top level is not a list -- "
            "surface reshaped (it is spec-deprecated; the durable "
            "replacement is virtual-machines/instant/locations); refusing "
            "to guess"
        )
    counts: Dict[str, int] = {}
    for idx, preset in enumerate(payload):
        if not isinstance(preset, dict):
            raise RuntimeError(
                f"voltagepark: presets[{idx}] is not an object -- preset "
                "shape changed; refusing to guess"
            )
        preset_id = preset.get("id")
        if not isinstance(preset_id, str) or not preset_id.strip():
            raise RuntimeError(
                f"voltagepark: presets[{idx}] has no string id -- no key "
                "to pin the per-preset counts to; refusing to guess"
            )
        if preset_id in counts:
            raise RuntimeError(
                f"voltagepark: duplicate preset id {preset_id!r} -- the "
                "per-preset count map would silently collapse rows; "
                "refusing to under-report"
            )
        locs = preset.get("location_ids_with_availability")
        if not isinstance(locs, list) or any(
            not isinstance(loc, str) for loc in locs
        ):
            raise RuntimeError(
                f"voltagepark: preset {preset_id}: "
                f"location_ids_with_availability is {locs!r} -- expected a "
                "list of location UUID strings (the availability signal and "
                "join key); refusing to guess"
            )
        counts[preset_id] = len(locs)
    return {
        "presets_url": PRESETS_URL,
        "presets_deprecated_in_spec": True,
        # "15.120000" is the WHOLE 8-GPU instance (1-GPU preset is
        # "1.890000") -- never read these as price_per_gpu_hr.
        "compute_rate_hourly_basis": "per-instance",
        # Verbatim at the richest published grain -- resources.gpus maps
        # model -> {"count": N}, rates arrive as decimal strings.
        "presets": payload,
        "preset_available_location_counts": counts,
    }


def parse_vm_locations(body: str) -> Dict[str, Any]:
    """Verbatim capture of /virtual-machines/instant/locations (the
    non-deprecated VM availability surface; rows carry
    available_presets[].available_vms per the live openapi.json). Same
    envelope pins as the bare-metal page; rows themselves ride verbatim --
    the accrual wants the first nonzero print recorded, not fenced away."""
    payload = json.loads(body)
    results = payload.get("results") if isinstance(payload, dict) else None
    total = payload.get("total_result_count") if isinstance(payload, dict) else None
    if (
        not isinstance(results, list)
        or isinstance(total, bool)
        or not isinstance(total, int)
    ):
        raise RuntimeError(
            "voltagepark: instant locations top level reshaped -- expected "
            "a 'results' list and a 'total_result_count' int; refusing to "
            "guess at the new shape"
        )
    if payload.get("has_next"):
        raise RuntimeError(
            "voltagepark: instant locations has_next is truthy -- the "
            "single-page assumption broke; add pagination rather than "
            "silently under-reporting"
        )
    if total != len(results):
        raise RuntimeError(
            f"voltagepark: instant locations total_result_count {total} != "
            f"{len(results)} rows in the page -- rows exist this fetch "
            "never saw; refusing to under-report"
        )
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"voltagepark: instant locations results[{idx}] is not an "
                "object -- row shape changed; refusing to guess"
            )
    return {
        "locations_url": VM_LOCATIONS_URL,
        "locations": results,
        "locations_total_result_count": total,
    }


# (surface label, url, parser) -- both fetched fail-open in collect(); the
# label prefixes the partial_error so a firing fence names its surface.
_VM_SURFACES = (
    ("presets", PRESETS_URL, parse_vm_presets),
    ("locations", VM_LOCATIONS_URL, parse_vm_locations),
)


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    # Price lane first, fail-closed, exactly as before -- nothing below may
    # run if the proven surface darked.
    body = fetch(
        URL, headers={"accept": "application/json"}, timeout=timeout
    )
    observations = parse_voltagepark(body)
    # VM availability lane, FAIL-OPEN per surface: a fetch failure or a
    # fail-closed parse fence becomes one partial_error and the price rows
    # above still record (docstring ruling).
    partial_errors: List[str] = []
    vm_instant: Dict[str, Any] = {}
    for label, url, parser in _VM_SURFACES:
        try:
            vm_instant.update(
                parser(
                    fetch(
                        url,
                        headers={"accept": "application/json"},
                        timeout=timeout,
                    )
                )
            )
        except Exception as exc:  # fail-open: availability never darks price
            partial_errors.append(
                f"vm_instant {label} not recorded (fail-open, price lane "
                f"unaffected): {exc}"
            )
    if vm_instant:
        vm_instant["semantics"] = _CLAIMED_AVAILABILITY
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=observations,
        partial_errors=partial_errors or None,
        book_stats={"vm_instant": vm_instant} if vm_instant else None,
    )
