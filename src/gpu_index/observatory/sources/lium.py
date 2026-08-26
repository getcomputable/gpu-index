# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Lium (Bittensor subnet 51, ex-Celium) -- /api/executors, the whole book.

Public no-auth JSON endpoint returning the ENTIRE live machine book in one
response (86 rows on 2026-08-22, no pagination) -- one fetch per capture.
Like vast, the book is per-MACHINE, not per-sku: heavy duplication (39x
RTX 5090 today) is the real shape of the marketplace and every row is
recorded, deduped by the row's own machine uuid (``id``).

Surface facts verified live 2026-08-22:

  - ``machine_name`` is the provider's structured part label (e.g.
    'NVIDIA B300 SXM6 AC', 'NVIDIA RTX PRO 6000 Blackwell Server Edition')
    and becomes sku_identifier verbatim -- the book carries true lookalike
    pairs (Server vs Workstation Edition, H200 vs H200 NVL, H100 80GB HBM3
    vs H100 PCIe) that must never be collapsed at capture time;
  - ``price_per_gpu`` is ALREADY per-GPU, USD per hour. Basis proven by
    the UI's '$/GPU/hr' vs 'total $/hr' toggle (whole-machine total =
    gpu_count x price_per_gpu; a rendered '$128.00/hr' is 16 x $8.0003) --
    so gpu_count_basis stays 1 and price*basis reproduces the raw figure;
    the machine size rides in extra.gpu_count. Currency: rows carry no
    currency field, but every rendered price on lium.io is '$' and the
    recon cross-checked three API rows against homepage '$/GPU' cards --
    the USD label is that warrant, not an assumption of convenience.
    Miner-set prices carry float residue (8.000300000000001); raw_value
    records it verbatim and price is never part of any identity key;
  - ``tier`` is lium's own vocabulary {'secure', 'spot'}, mapped
    fail-closed to lane tiers (secure -> on-demand); specs.is_spot
    duplicates it and is used as a belt-and-braces cross-check;
  - ``pending_price_per_hour`` + ``price_change_effective_date`` are a
    SCHEDULED FUTURE price change -- despite the per_hour name they are
    never the current price. When set they ride in extra, clearly labeled
    pending; the recorded price is price_per_gpu only.

Fail-closed identity pins (the transport raises on non-2xx, and an HTML
error page or moved endpoint fails the JSON-list pin loudly):

  - the body must parse to a NON-EMPTY JSON list -- an empty book or a
    reshaped envelope aborts the capture with zero rows claimed;
  - EVERY element must be an object carrying machine_name (non-empty
    str), price_per_gpu (real number), gpu_count (int), tier (non-empty
    str) and id (non-empty str) -- any missing key or wrong type raises
    with the row index and field, never a silent skip that could zero the
    feed while looking healthy.

Value-level oddities on single rows skip + count instead (one weird row
must not dark the source): unknown tier label, tier/is_spot contradiction,
non-finite price (json.loads accepts NaN/Infinity literals and both slip
past a <= 0 comparison -- same guard as runpod/scaleway), non-positive
price (not a $0 print), gpu_count < 1, duplicate machine id.
A book where EVERY row was skipped raises -- the pins stopped
discriminating, which is a shape change, not a thin market.

AVAILABILITY ACCRUAL (live evidence 2026-08-25) -- three rungs,
all fail-open: an availability reshape must never dark the price lane.

  - rung 1: ``available_gpu_count`` (already passed through) is now PINNED
    as an int (bool-excluded) with 0 <= v <= gpu_count when present; a
    violating value records None and counts under
    book_stats["availability_skips"]["bad_available_gpu_count"] -- the row
    itself still prints. Live 2026-08-25: all 81 rows carried a valid
    count, none None;
  - rung 2: one extra GET of /api/machines (public per the OpenAPI spec)
    per capture -- CHIP-LEVEL occupancy + fleet size for ~91 models
    (2026-08-25; 73 of them with ZERO fleet -- an absence signal the book
    cannot show, counted in book_stats["models_zero_capacity"]). Join key:
    executors.machine_name == machines.name, exact verbatim string match
    (live-verified identical for every book label incl. 'NVIDIA RTX PRO
    6000 Blackwell Server Edition'); each observation gains
    extra.model_rental_rate + extra.model_total_gpu_count (None + counted
    when absent/invalid). SEMANTICS WARRANT (cross-footed 2026-08-25,
    caveat: rental_rate is NOT documented): per machine_name,
    total_gpu_count * (1 - rental_rate) matched sum(available_gpu_count)
    from /api/executors within +/-1 on all 14 book models -- so
    total_gpu_count is the FLEET size (rented + unrented) and rental_rate
    the fraction rented. rental_rate 0.0 with total_gpu_count 0 conflates
    'no fleet' with 'none rented' (A30/A40 live), which is why BOTH fields
    are recorded;
  - rung 3: one GET of /api/machines/capacity (public per spec) -- a
    CHIP x BUCKET-SIZE matrix (13 base models x config sizes {1, 8}:
    max_cap/unrented_count/hourly_rate) stored VERBATIM as
    book_stats["capacity_buckets"], a per-capture aggregate, NOT
    observations. Its ``base_model`` vocabulary ('H100', 'RTX PRO 6000')
    is COLLAPSED and cannot distinguish the true lookalike pairs above --
    it must NEVER be joined onto machine_name. The spec sources it from
    'GPU_ESTIMATES_KEY from Redis' and max_cap looks like a capped
    estimate cache (10 in every 1x bucket on 2026-08-25, 32/64 in 8x
    buckets): treat unrented_count as indicative, never a hard count.

FALLBACK ANCHOR (documented, not implemented): the homepage embeds the
identical rows in <script id="__NEXT_DATA__"> at
props.pageProps.dehydratedState.queries[queryKey==["/executors",null]]
.state.data -- if /api/executors (a Next.js proxy route) ever moves, that
is where the book lives. Auth-posture note (2026-08-25): openapi.json
marks GET /executors JwtAccessBearer while it serves no-auth today -- a
posture that could be locked down; /machines, /machines/capacity and
/executors/count|total-count|stats are explicitly public in the spec and
are the durable surfaces (count + total-count give a book-free occupancy
ratio fallback if /api/machines is ever fenced off).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "lium"

URL = "https://lium.io/api/executors"

# Availability surfaces (availability accrual) -- both explicitly public in
# lium.io/api/openapi.json; both fail-open (see module docstring).
MACHINES_URL = "https://lium.io/api/machines"
CAPACITY_URL = "https://lium.io/api/machines/capacity"

# lium's own tier vocabulary -> lane tiers, fail-closed (an unknown label
# is skipped + counted, never guessed into a tier).
TIER_BY_LIUM = {"secure": "on-demand", "spot": "spot"}

# The whole book is recorded while it stays small (86 machines on
# 2026-08-22; the transport's 8MB body cap ~= 900 rows is the hard fence).
# Past this soft fence the capture still records everything but says so
# loudly -- the escape hatch is a vast-style record_limit_per_gpu option.
BOOK_GROWTH_ALERT_ROWS = 300

# (field, type, allow-bool is never wanted) -- the structural pin below.
_REQUIRED_FIELDS: Tuple[Tuple[str, Any], ...] = (
    ("machine_name", str),
    ("price_per_gpu", (int, float)),
    ("gpu_count", int),
    ("tier", str),
    ("id", str),
)


def _pin_book(doc: Any) -> List[Dict[str, Any]]:
    """The structural identity pin: a non-empty list of machine objects,
    each carrying every field the recipe reads, correctly typed. Raises
    with a specific message on any drift -- see module docstring."""
    if not isinstance(doc, list):
        raise RuntimeError(
            "lium: /api/executors did not return a JSON list -- endpoint "
            "moved or reshaped (fallback book lives in the homepage "
            "__NEXT_DATA__); refusing to guess"
        )
    if not doc:
        raise RuntimeError(
            "lium: executor book is EMPTY -- either every listing was "
            "pulled or the endpoint changed; refusing to claim zero rows "
            "silently"
        )
    for idx, row in enumerate(doc):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"lium: book element {idx} is not an object -- API shape "
                "changed; refusing to guess"
            )
        for field, types in _REQUIRED_FIELDS:
            value = row.get(field)
            if (
                not isinstance(value, types)
                or isinstance(value, bool)
                or (types is str and not value.strip())
            ):
                raise RuntimeError(
                    f"lium: book element {idx} field {field!r} is missing, "
                    f"mistyped or empty ({value!r}) -- row shape changed; "
                    "aborting the capture rather than recording a book "
                    "whose pins no longer discriminate"
                )
    return doc


def _pin_available_gpu_count(
    row: Dict[str, Any],
) -> Tuple[Optional[int], Optional[str]]:
    """Rung-1 availability pin, fail-open PER ROW: available_gpu_count
    must be an int (bool-excluded) with 0 <= v <= gpu_count when present.
    Returns (value, None) or (None, note_category) -- the observation
    still prints either way; only the availability field degrades."""
    value = row.get("available_gpu_count")
    if value is None:
        return None, None
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= row["gpu_count"]
    ):
        return None, "bad_available_gpu_count"
    return value, None


def _row_observation(
    row: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """One structurally-pinned machine row -> (observation, None,
    availability_note) or (None, skip_category, None). Skip categories are
    countable pin names; availability notes count in book_stats without
    skipping the row (fail-open, module docstring rung 1)."""
    lium_tier = row["tier"]
    tier = TIER_BY_LIUM.get(lium_tier)
    if tier is None:
        return None, "unmapped_tier", None
    specs = row.get("specs")
    is_spot = specs.get("is_spot") if isinstance(specs, dict) else None
    if isinstance(is_spot, bool) and is_spot != (lium_tier == "spot"):
        # Belt and braces: specs.is_spot duplicates tier today; a
        # contradiction means the tier semantics drifted for this row.
        return None, "tier_spot_mismatch", None
    price = row["price_per_gpu"]
    if not math.isfinite(price):
        # json.loads accepts NaN/Infinity literals; both pass isinstance
        # ((int, float)) AND a <= 0 comparison, so without this pin a
        # miner-set NaN would record as a live USD print.
        return None, "non_finite_price", None
    if price <= 0:
        return None, "unpriced", None  # not a $0 print
    gpu_count = row["gpu_count"]
    if gpu_count < 1:
        return None, "bad_gpu_count", None

    available, avail_note = _pin_available_gpu_count(row)
    extra: Dict[str, Any] = {
        "lium_tier": lium_tier,
        "gpu_count": gpu_count,
        "available_gpu_count": available,
        "miner_hotkey": row.get("miner_hotkey"),
        "validator_hotkey": row.get("validator_hotkey"),
        "reliability_score": row.get("reliability_score"),
    }
    if (
        row.get("pending_price_per_hour") is not None
        or row.get("price_change_effective_date") is not None
    ):
        # A SCHEDULED FUTURE change -- recorded labeled-pending in extra,
        # NEVER as the current price (see module docstring trap).
        extra["pending_price_per_hour"] = row.get("pending_price_per_hour")
        extra["price_change_effective_date"] = row.get(
            "price_change_effective_date"
        )

    location = row.get("location")
    country = (
        str(location.get("country_code") or "").strip()
        if isinstance(location, dict)
        else ""
    )
    obs = observation(
        sku_identifier=row["machine_name"],
        price_per_gpu_hr=float(price),
        # USD warrant: see module docstring (every rendered lium price is
        # '$', cross-checked against these API rows) -- not an assumption.
        currency="USD",
        raw_value=str(price),  # verbatim, incl. miner-set float residue
        raw_unit="usd_per_gpu_hr",  # price_per_gpu is already per-GPU
        gpu_count_basis=1,
        tier=tier,
        region=country or "?",
        notes=f"lium {lium_tier} tier, {gpu_count}x machine",
        extra=extra,
    )
    # Identity continuity: tomorrow's print must be attributable to the
    # same machine (sanctioned top-level passthrough field).
    obs["machine_id"] = row["id"]
    return obs, None, avail_note


def parse_lium(
    body: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Pure parse over the fetched body: pin the book shape, then record
    every row that survives the value pins. Returns
    (observations, partial_notes, book_stats)."""
    book = _pin_book(json.loads(body))
    observations: List[Dict[str, Any]] = []
    skips: Dict[str, int] = {}
    avail_skips: Dict[str, int] = {}
    seen_ids: set = set()
    for row in book:
        if row["id"] in seen_ids:
            # ids are unique per machine today; a replay is dropped and
            # counted, never double-printed.
            skips["duplicate_id"] = skips.get("duplicate_id", 0) + 1
            continue
        seen_ids.add(row["id"])
        obs, skip, avail_note = _row_observation(row)
        if obs is not None:
            observations.append(obs)
            if avail_note is not None:
                avail_skips[avail_note] = avail_skips.get(avail_note, 0) + 1
        else:
            skips[skip] = skips.get(skip, 0) + 1
    if not observations:
        raise RuntimeError(
            f"lium: {len(book)} executors fetched but ZERO survived the "
            "value pins ("
            + ", ".join(f"{count} {cat}" for cat, count in sorted(skips.items()))
            + ") -- the tier/price semantics changed; refusing to guess"
        )
    notes: List[str] = []
    if skips:
        notes.append(
            "skipped "
            + ", ".join(f"{count} {cat}" for cat, count in sorted(skips.items()))
            + " row(s) (fail-closed pins; see module docstring)"
        )
    if len(book) > BOOK_GROWTH_ALERT_ROWS:
        notes.append(
            f"book grew past {BOOK_GROWTH_ALERT_ROWS} machines "
            f"({len(book)}) -- still recorded fully; consider a vast-style "
            "record_limit_per_gpu option"
        )
    tiers: Dict[str, int] = {}
    for obs in observations:
        lt = obs["extra"]["lium_tier"]
        tiers[lt] = tiers.get(lt, 0) + 1
    stats = {
        "machines_total": len(book),
        "rows_recorded": len(observations),
        "rows_skipped": sum(skips.values()),
        "distinct_labels": len({o["sku_identifier"] for o in observations}),
        "by_lium_tier": tiers,
        # Availability degradations never skip a row -- they count here
        # (module docstring rung 1), keyed by pin name.
        "availability_skips": avail_skips,
    }
    return observations, notes, stats


def parse_machines(
    body: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    """Pure parse of /api/machines (rung 2): fail-closed structural pin
    (non-empty JSON list of objects with name/rental_rate/total_gpu_count,
    correctly typed), fail-open value guards PER MODEL (a weird rate on
    one chip must not blank the whole map). Returns
    (by_name, value_skips, catalog_stats); the raise stays INSIDE the
    availability parse -- collect() turns it into a partial_error, never
    a dark price lane."""
    doc = json.loads(body)
    if not isinstance(doc, list) or not doc:
        raise RuntimeError(
            "lium: /api/machines did not return a non-empty JSON list -- "
            "availability surface moved or reshaped; refusing to guess"
        )
    by_name: Dict[str, Dict[str, Any]] = {}
    value_skips: Dict[str, int] = {}
    models_zero_capacity = 0
    for idx, row in enumerate(doc):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"lium: /api/machines element {idx} is not an object -- "
                "availability surface reshaped; refusing to guess"
            )
        name = row.get("name")
        rental_rate = row.get("rental_rate")
        total = row.get("total_gpu_count")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(rental_rate, (int, float))
            or isinstance(rental_rate, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
        ):
            raise RuntimeError(
                f"lium: /api/machines element {idx} is missing or mistypes "
                "name/rental_rate/total_gpu_count -- row shape changed; "
                "refusing a map whose pins no longer discriminate"
            )
        # Value guards, fail-open per model: out-of-range figures record
        # None + count instead of poisoning every joined observation.
        # (isfinite: json.loads accepts NaN/Infinity -- same trap as the
        # price pin above.)
        if not math.isfinite(rental_rate) or not 0 <= rental_rate <= 1:
            rental_rate = None
            value_skips["bad_model_rental_rate"] = (
                value_skips.get("bad_model_rental_rate", 0) + 1
            )
        if total < 0:
            total = None
            value_skips["bad_model_total_gpu_count"] = (
                value_skips.get("bad_model_total_gpu_count", 0) + 1
            )
        elif total == 0:
            # The absence signal: a model lium prices but has zero fleet
            # for -- the book alone can never show this.
            models_zero_capacity += 1
        by_name[name] = {
            "model_rental_rate": rental_rate,
            "model_total_gpu_count": total,
        }
    catalog_stats = {
        "models_in_catalog": len(doc),
        "models_zero_capacity": models_zero_capacity,
    }
    return by_name, value_skips, catalog_stats


def parse_capacity(body: str) -> List[Dict[str, Any]]:
    """Pure parse of /api/machines/capacity (rung 3): pin a non-empty JSON
    list of objects each carrying base_model (non-empty str) and buckets
    (object), then return the payload VERBATIM -- stored whole as
    book_stats["capacity_buckets"], NEVER joined onto machine_name (its
    base_model vocabulary is collapsed; module docstring)."""
    doc = json.loads(body)
    if not isinstance(doc, list) or not doc:
        raise RuntimeError(
            "lium: /api/machines/capacity did not return a non-empty JSON "
            "list -- capacity surface moved or reshaped; refusing to guess"
        )
    for idx, row in enumerate(doc):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("base_model"), str)
            or not row["base_model"].strip()
            or not isinstance(row.get("buckets"), dict)
        ):
            raise RuntimeError(
                f"lium: /api/machines/capacity element {idx} lacks the "
                "base_model/buckets shape -- surface reshaped; refusing to "
                "record an unrecognizable matrix"
            )
    return doc


def attach_model_stats(
    observations: List[Dict[str, Any]],
    by_name: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, int]:
    """Attach the /api/machines chip-level occupancy to every observation
    via the verbatim join key machine_name == name (exact string match,
    live-verified 2026-08-25). ``by_name`` None means the availability
    fetch/parse failed fail-open: the fields still land (as None) so the
    extra schema stays stable, without per-row counts -- the partial_error
    already says why. Returns countable notes for book_stats."""
    counts: Dict[str, int] = {}
    for obs in observations:
        entry = by_name.get(obs["sku_identifier"]) if by_name else None
        if by_name is not None and entry is None:
            counts["model_not_in_machines"] = (
                counts.get("model_not_in_machines", 0) + 1
            )
        extra = obs["extra"]
        extra["model_rental_rate"] = (
            entry["model_rental_rate"] if entry is not None else None
        )
        extra["model_total_gpu_count"] = (
            entry["model_total_gpu_count"] if entry is not None else None
        )
    return counts


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    # The price surface stays fail-closed exactly as before; everything
    # after this line is availability accrual and must never raise out.
    observations, notes, stats = parse_lium(fetch(URL, timeout=timeout))

    by_name = None
    try:
        by_name, value_skips, catalog_stats = parse_machines(
            fetch(MACHINES_URL, timeout=timeout)
        )
        for cat, count in value_skips.items():
            stats["availability_skips"][cat] = (
                stats["availability_skips"].get(cat, 0) + count
            )
        stats.update(catalog_stats)
    except Exception as exc:  # fail-open: availability never darks prices
        notes.append(
            "availability enrichment skipped -- /api/machines fetch/parse "
            f"failed (price rows unaffected): {exc}"
        )
    for cat, count in attach_model_stats(observations, by_name).items():
        stats["availability_skips"][cat] = (
            stats["availability_skips"].get(cat, 0) + count
        )

    try:
        stats["capacity_buckets"] = parse_capacity(
            fetch(CAPACITY_URL, timeout=timeout)
        )
    except Exception as exc:  # fail-open, same posture
        notes.append(
            "capacity buckets skipped -- /api/machines/capacity fetch/parse "
            f"failed (price rows unaffected): {exc}"
        )

    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=observations,
        partial_errors=notes or None,
        book_stats=stats,
    )
