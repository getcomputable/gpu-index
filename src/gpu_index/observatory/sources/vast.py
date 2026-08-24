# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Vast.ai -- bundles API, per-chip cheapest-machine books (wide net).

Observatory adaptation of the basket lane's hardened vast collector
(gpu_index.index.sources.collect_vast). The offer parse
(parse_vast_offers) is IMPORTED from gpu_index.index.sources on purpose: it carries
the L0 arithmetic tripwire (recorded price times its basis must reproduce
the offer's dph_total), the is_bid belt-and-braces, and the num_gpus 1..16
fence, all battle-tested live; a refactor there breaks this import loudly
at registry time, never silently.

Query shape (proven live in the basket lane, re-verified 2026-08-22):
  - one ASC-by-dph_total call per chip in options["gpu_names"], plus one
    DESC call ONLY when the ASC response hits FETCH_LIMIT. Ascending
    dph_total truncates the LARGEST instance totals first -- exactly the
    cheap-per-GPU multi-GPU boxes (the 08-13 burial class) -- so a
    full-limit ASC book is never trusted alone. At most 2 calls per chip.
  - NO verified filter (broken semantics against the live API: the real
    field is the STRING ``verification``; a verified:{eq:true} filter
    silently dropped the cheapest hosts) and no num_gpus preference. Fetch
    the thin book, rank locally, record verification as metadata.
  - FETCH_LIMIT must stay at or below the server's own page clamp
    (observed 2026-08-22: limit=400 returned 64 offers). If FETCH_LIMIT
    exceeded the clamp, the raw_count >= FETCH_LIMIT truncation signal
    below could NEVER fire and a truncated book would look complete.

Selection per chip: dedup by MACHINE keeping its cheapest per-GPU offer,
rank per-GPU ascending (never by instance total), record the cheapest
options["record_limit_per_gpu"] machines with identity fields
(offer_id/machine_id/host_id/verification; geolocation rides as region).
Identity pin: the server-side gpu_name eq filter is load-bearing, so an
offer whose OWN gpu_name does not equal the queried string is skipped and
counted in partial_errors, never recorded into the queried book.

A chip whose book is empty is a partial_errors note, not an error (vast
genuinely lists zero rentable MI300X machines as of 2026-08-22); the
source raises only when NO chip produced rows and at least one chip
actually failed (result() itself raises on the all-empty case).
"""

from __future__ import annotations

import json
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.http import fetch
from gpu_index.index.sources import parse_vast_offers
from gpu_index.observatory.observation import DEFAULT_TIMEOUT, observation, result

SOURCE_ID = "vast"

URL_BASE = "https://console.vast.ai/api/v0/bundles/"
# See module docstring: must stay <= the server's own page clamp (64
# observed 2026-08-22) or the truncation guard goes blind. 50 is the value
# the basket lane has run live since its hardening rewrite.
FETCH_LIMIT = 50
# The wide net fires up to 2*len(gpu_names) calls per capture (vs the
# basket lane's 2-4), and a rapid-fire burst drew a live 429 on 2026-08-22.
# Space the per-chip fetches so one capture stays inside vast's
# unauthenticated per-minute budget; ~30s measured for the current 21-chip
# config list (26 calls), well inside the configured 180s per-source
# deadline (config/raw_observatory.json per_source_deadline_seconds).
REQUEST_SPACING_SECONDS = 0.75


def _query_url(gpu_name: str, order: str = "asc") -> str:
    # Hardened query shape: no verified filter, no num_gpus preference,
    # whole thin book ordered by dph_total. ``order`` matters -- see the
    # module docstring on ASC truncation.
    q: Dict[str, Any] = {
        "gpu_name": {"eq": gpu_name},
        "rentable": {"eq": True},
        "type": "on-demand",
        "order": [["dph_total", order]],
        "limit": FETCH_LIMIT,
    }
    return URL_BASE + "?q=" + urllib.parse.quote(json.dumps(q))


def pin_candidates(
    candidates: List[Dict[str, Any]], gpu_name: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Identity pin: keep only offers the server itself attributes to the
    queried gpu_name (exact string equality against the offer's own
    gpu_name field). A mismatched or missing label means the eq filter no
    longer discriminates for that row -- skipped and counted, never guessed
    into the queried book."""
    pinned: List[Dict[str, Any]] = []
    skipped = 0
    for cand in candidates:
        if cand.get("gpu_name") == gpu_name:
            pinned.append(cand)
        else:
            skipped += 1
    return pinned, skipped


def merge_candidate_books(
    asc: List[Dict[str, Any]], desc: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """ASC head + DESC tail, deduped by offer id. An offer present in both
    windows must not be double-counted; offers with no id are kept (losing
    a real ask is worse than a rare duplicate candidate, and the
    machine-level dedup in rank_machines collapses duplicates of any
    identified machine anyway)."""
    seen = {c["offer_id"] for c in asc if c["offer_id"] is not None}
    merged = list(asc)
    for cand in desc:
        if cand["offer_id"] is None or cand["offer_id"] not in seen:
            merged.append(cand)
    return merged


def rank_machines(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Dedup by machine (keep its cheapest PER-GPU offer), rank per-GPU
    ascending. L2 identity lesson: rank by per-GPU price, never by
    instance total -- dph_total ordering buried $6.25/GPU 8x boxes beneath
    $10.94/GPU 1-2 GPU slices of one expensive host. One row per MACHINE so
    a single host listing every slice size cannot crowd the record."""
    best_per_machine: Dict[Any, Dict[str, Any]] = {}
    for idx, cand in enumerate(candidates):
        key = cand.get("machine_id")
        if key is None:
            # Never collapse offers we cannot attribute to a machine.
            key = ("unidentified", cand.get("offer_id"), idx)
        cur = best_per_machine.get(key)
        if cur is None or cand["per_gpu"] < cur["per_gpu"]:
            best_per_machine[key] = cand
    return sorted(best_per_machine.values(), key=lambda c: c["per_gpu"])


def observation_row(cand: Dict[str, Any]) -> Dict[str, Any]:
    row = observation(
        sku_identifier=cand["gpu_name"],
        price_per_gpu_hr=cand["per_gpu"],
        raw_value=str(cand["dph_total"]),
        raw_unit="usd_per_instance_hr",
        gpu_count_basis=cand["num_gpus"],
        tier="on-demand",
        region=cand["geolocation"],
        notes=(
            f"{cand['num_gpus']}x rentable on-demand ask (dph_total incl "
            f"default storage), cheapest offer of machine "
            f"{cand['machine_id']}"
        ),
    )
    # L2 identity continuity: tomorrow's print must be
    # attributable to the same machine. These are sanctioned top-level
    # passthrough fields in the snapshot schema.
    row.update(
        {
            "offer_id": cand["offer_id"],
            "machine_id": cand["machine_id"],
            "host_id": cand["host_id"],
            "verification": cand["verification"],
        }
    )
    return row


def select_chip_observations(
    candidates: List[Dict[str, Any]], record_limit: int
) -> List[Dict[str, Any]]:
    """The recorded rows for one chip's book: cheapest ``record_limit``
    machines, per-GPU ascending."""
    return [observation_row(c) for c in rank_machines(candidates)[:record_limit]]


def chip_book_stats(
    candidates: List[Dict[str, Any]],
    record_limit: int,
    *,
    fetch_truncated: bool = False,
    coverage_gap: bool = False,
) -> Dict[str, Any]:
    """Truncation must never be invisible: per-chip book
    accounting recorded alongside the rows, computed with the same
    dedup/rank helper the selection uses."""
    ranked = rank_machines(candidates)
    stats: Dict[str, Any] = {
        "candidate_offers": len(candidates),
        "machines_total": len(ranked),
        "rows_recorded": min(len(ranked), record_limit),
        "fetch_truncated": fetch_truncated,
        "coverage_gap": coverage_gap,
    }
    if ranked:
        stats["per_gpu_min"] = round(ranked[0]["per_gpu"], 4)
        stats["per_gpu_max"] = round(ranked[-1]["per_gpu"], 4)
    return stats


def _validated_options(
    options: Optional[Dict[str, Any]],
) -> Tuple[List[str], int]:
    """Fail closed: this collector never invents a chip list -- the queried
    names are operational config (config/raw_observatory.json)."""
    if not isinstance(options, dict):
        raise RuntimeError(
            "vast: options missing -- config/raw_observatory.json must "
            "supply options.gpu_names and options.record_limit_per_gpu; "
            "refusing to guess a chip list"
        )
    names = options.get("gpu_names")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n.strip() for n in names)
    ):
        raise RuntimeError(
            "vast: options.gpu_names must be a non-empty list of vast "
            "gpu_name strings"
        )
    if len(set(names)) != len(names):
        raise RuntimeError(
            "vast: options.gpu_names contains duplicates -- a duplicated "
            "name would double-record its book"
        )
    limit = options.get("record_limit_per_gpu")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise RuntimeError(
            "vast: options.record_limit_per_gpu must be a positive integer"
        )
    return list(names), limit


def _fetch_chip_book(
    gpu_name: str, timeout: float
) -> Tuple[int, List[Dict[str, Any]], bool, bool]:
    """One chip's candidate set: ASC head, plus the DESC tail iff the ASC
    response hit FETCH_LIMIT. Returns (raw_offer_count, candidates,
    fetch_truncated, coverage_gap)."""
    body = fetch(_query_url(gpu_name, "asc"), timeout=timeout)
    raw_count = len(json.loads(body).get("offers") or [])
    candidates = parse_vast_offers(body)
    fetch_truncated = raw_count >= FETCH_LIMIT
    coverage_gap = False
    if fetch_truncated:
        desc_body = fetch(_query_url(gpu_name, "desc"), timeout=timeout)
        desc_raw_count = len(json.loads(desc_body).get("offers") or [])
        candidates = merge_candidate_books(
            candidates, parse_vast_offers(desc_body)
        )
        raw_count += desc_raw_count
        # Book wider than both windows combined: mid-book offers may be
        # missing -- visible in partial_errors, never silent.
        coverage_gap = desc_raw_count >= FETCH_LIMIT
    return raw_count, candidates, fetch_truncated, coverage_gap


def collect(
    timeout: float = DEFAULT_TIMEOUT, options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    gpu_names, record_limit = _validated_options(options)
    rows: List[Dict[str, Any]] = []
    partial: List[str] = []
    failures: List[str] = []
    book_stats: Dict[str, Dict[str, Any]] = {}
    for idx, gpu_name in enumerate(gpu_names):
        if idx:
            time.sleep(REQUEST_SPACING_SECONDS)
        try:
            raw_count, candidates, fetch_truncated, coverage_gap = (
                _fetch_chip_book(gpu_name, timeout)
            )
        except Exception as exc:  # noqa: BLE001 -- one chip's feed must not hide the rest
            failures.append(f"{gpu_name}: {type(exc).__name__}: {exc}")
            continue
        if raw_count == 0:
            # NOT an error: vast genuinely has empty books for some chips
            # (MI300X as of 2026-08-22). Recorded so the note is countable.
            partial.append(
                f"{gpu_name}: zero rentable on-demand offers on the book "
                "(empty book, not a failure)"
            )
            book_stats[gpu_name] = chip_book_stats([], record_limit)
            continue
        pinned, skipped = pin_candidates(candidates, gpu_name)
        if skipped:
            partial.append(
                f"{gpu_name}: skipped {skipped} offer(s) whose own gpu_name "
                "did not match the queried name (identity pin)"
            )
        if not pinned:
            failures.append(
                f"{gpu_name}: {raw_count} offers fetched but ZERO survived "
                "the validity and gpu_name identity pins -- the bundles API "
                "shape or its gpu_name filter changed; refusing to guess"
            )
            continue
        chip_rows = select_chip_observations(pinned, record_limit)
        stats = chip_book_stats(
            pinned,
            record_limit,
            fetch_truncated=fetch_truncated,
            coverage_gap=coverage_gap,
        )
        rows.extend(chip_rows)
        book_stats[gpu_name] = stats
        if coverage_gap:
            partial.append(
                f"{gpu_name}: book wider than both fetch windows "
                f"(>= {2 * FETCH_LIMIT} offers) -- mid-book cheap-per-GPU "
                "offers may be missing from the candidate set"
            )
        # Per-chip one-liner (config-derived strings only -- remote strings
        # are never printed raw from this module; parse_vast_offers cleans
        # its own anomaly prints).
        print(
            f"  vast {gpu_name}: {stats['candidate_offers']} candidate "
            f"offers, {stats['machines_total']} machines, recording "
            f"{stats['rows_recorded']} (cheapest "
            f"${stats['per_gpu_min']:.4f}/gpu-hr)"
        )
    if not rows and failures:
        # Every chip either failed or printed nothing, and at least one
        # genuinely failed: surface the real causes instead of the generic
        # parsed-nothing message.
        raise RuntimeError(
            f"vast: no chip produced observations and {len(failures)} "
            "chip(s) failed: " + "; ".join(failures)
        )
    return result(
        SOURCE_ID,
        method="api-json",
        url=URL_BASE,
        observations=rows,
        partial_errors=(failures + partial) or None,
        book_stats=book_stats or None,
    )
