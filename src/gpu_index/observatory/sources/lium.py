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

FALLBACK ANCHOR (documented, not implemented): the homepage embeds the
identical rows in <script id="__NEXT_DATA__"> at
props.pageProps.dehydratedState.queries[queryKey==["/executors",null]]
.state.data -- if /api/executors (a Next.js proxy route) ever moves, that
is where the book lives.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "lium"

URL = "https://lium.io/api/executors"

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


def _row_observation(
    row: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """One structurally-pinned machine row -> (observation, None) or
    (None, skip_category). Skip categories are countable pin names."""
    lium_tier = row["tier"]
    tier = TIER_BY_LIUM.get(lium_tier)
    if tier is None:
        return None, "unmapped_tier"
    specs = row.get("specs")
    is_spot = specs.get("is_spot") if isinstance(specs, dict) else None
    if isinstance(is_spot, bool) and is_spot != (lium_tier == "spot"):
        # Belt and braces: specs.is_spot duplicates tier today; a
        # contradiction means the tier semantics drifted for this row.
        return None, "tier_spot_mismatch"
    price = row["price_per_gpu"]
    if not math.isfinite(price):
        # json.loads accepts NaN/Infinity literals; both pass isinstance
        # ((int, float)) AND a <= 0 comparison, so without this pin a
        # miner-set NaN would record as a live USD print.
        return None, "non_finite_price"
    if price <= 0:
        return None, "unpriced"  # not a $0 print
    gpu_count = row["gpu_count"]
    if gpu_count < 1:
        return None, "bad_gpu_count"

    extra: Dict[str, Any] = {
        "lium_tier": lium_tier,
        "gpu_count": gpu_count,
        "available_gpu_count": row.get("available_gpu_count"),
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
    return obs, None


def parse_lium(
    body: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Pure parse over the fetched body: pin the book shape, then record
    every row that survives the value pins. Returns
    (observations, partial_notes, book_stats)."""
    book = _pin_book(json.loads(body))
    observations: List[Dict[str, Any]] = []
    skips: Dict[str, int] = {}
    seen_ids: set = set()
    for row in book:
        if row["id"] in seen_ids:
            # ids are unique per machine today; a replay is dropped and
            # counted, never double-printed.
            skips["duplicate_id"] = skips.get("duplicate_id", 0) + 1
            continue
        seen_ids.add(row["id"])
        obs, skip = _row_observation(row)
        if obs is not None:
            observations.append(obs)
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
    }
    return observations, notes, stats


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    observations, notes, stats = parse_lium(fetch(URL, timeout=timeout))
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL,
        observations=observations,
        partial_errors=notes or None,
        book_stats=stats,
    )
