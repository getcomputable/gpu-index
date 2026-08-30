# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Capture-side data-quality screens.

Run AFTER collection and BEFORE upload, and may only ever FLAG — via the
same ``implausible`` machinery the plausibility band uses — never drop or
rewrite a raw price. The calc lane stays untouched: R1 already excludes
implausible prints, so no methodology change is implied by flagging here.

Two layers, both against the PREVIOUS stored snapshot (the pointer's):

  - Delta report (L2): per basket constituent, the move of its would-be
    daily print (book delta) and — where identity fields exist — the move
    of the machine the reference print came from (same-machine delta). A
    same-machine 0% beside a book +59% is an extraction/selection artifact,
    not a repricing; the pair is printed on every capture so the next bad
    print is diagnosable from the job log alone.
  - Jump quarantine (L5): a single source whose print moves >= the jump
    threshold while fewer than the required number of OTHER basket sources
    move >= the corroboration threshold is quarantined for THIS capture
    only (the next capture re-evaluates against the new reference, so a
    genuine uncorroborated repricing costs exactly one capture). The
    corroboration gate is the point: a real market-wide event (May-June
    B200, -30% across the board) passes; a single-source glitch does not.

Comparisons are in NATIVE currency terms per source — a same-source ratio
needs no FX — mirroring the drift detector's native-terms rule. Thresholds
live in config ``capture_screens`` (config, not code).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_JUMP_QUARANTINE_PCT = 25.0
DEFAULT_JUMP_CORROBORATE_PCT = 10.0
DEFAULT_JUMP_MIN_CORROBORATORS = 2

QUARANTINE_REASON = "uncorroborated_jump"

__all__ = [
    "QUARANTINE_REASON",
    "apply_jump_screen",
    "lowest_eligible",
    "screen_params",
]


def screen_params(config: Dict[str, Any]) -> Dict[str, Any]:
    screens = config.get("capture_screens") or {}
    return {
        "jump_quarantine_pct": float(
            screens.get("jump_quarantine_pct", DEFAULT_JUMP_QUARANTINE_PCT)
        ),
        "jump_corroborate_pct": float(
            screens.get("jump_corroborate_pct", DEFAULT_JUMP_CORROBORATE_PCT)
        ),
        "jump_min_corroborators": int(
            screens.get(
                "jump_min_corroborators", DEFAULT_JUMP_MIN_CORROBORATORS
            )
        ),
    }


# Currencies the calc lane can actually price (USD direct, EUR via the ECB
# record) — a print the composite would never use must not become the
# screen's reference or today's would-be print (e.g. a deliberately
# UNKNOWN-labeled currency). Public: the panel engine screens rows
# against the same fence (ARCHITECTURE.md).
PRICEABLE_CURRENCIES = ("USD", "EUR")


def _eligible(
    obs: Any, sku: str, interruptible: Sequence[str]
) -> bool:
    """Mirror of R1 eligibility at capture time, in native terms: matching
    sku, non-interruptible tier, priceable currency, not already flagged,
    carrying a price. Reference observations are remote-derived: a non-dict
    entry is simply ineligible, never a crash."""
    if not isinstance(obs, dict):
        return False
    return (
        obs.get("sku") == sku
        and obs.get("tier") not in interruptible
        and not obs.get("implausible")
        and obs.get("currency", "USD") in PRICEABLE_CURRENCIES
        and isinstance(obs.get("price_native_per_gpu_hr"), (int, float))
        and not isinstance(obs.get("price_native_per_gpu_hr"), bool)
    )


def lowest_eligible(
    source_entry: Optional[Dict[str, Any]],
    sku: str,
    interruptible: Sequence[str],
    *,
    currency: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The source's would-be daily print: lowest eligible native price.
    ``currency`` narrows to one currency — native-terms comparisons are only
    meaningful within a single currency, so the book delta selects today's
    print in the REFERENCE print's currency."""
    if not isinstance(source_entry, dict):
        return None
    candidates = [
        o
        for o in source_entry.get("observations") or []
        if _eligible(o, sku, interruptible)
        and (currency is None or o.get("currency", "USD") == currency)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda o: float(o["price_native_per_gpu_hr"]))


def _same_machine_pct(
    source_entry: Dict[str, Any],
    reference_print: Dict[str, Any],
    sku: str,
    interruptible: Sequence[str],
) -> Optional[float]:
    """Today's price of the machine the REFERENCE print came from — the
    identity-continuity half of the delta pair."""
    machine = reference_print.get("machine_id")
    if machine is None:
        return None
    today = [
        o
        for o in source_entry.get("observations") or []
        if o.get("machine_id") == machine and _eligible(o, sku, interruptible)
    ]
    if not today:
        return None
    best = min(today, key=lambda o: float(o["price_native_per_gpu_hr"]))
    if best.get("currency") != reference_print.get("currency"):
        return None
    ref_price = float(reference_print["price_native_per_gpu_hr"])
    if ref_price <= 0:
        return None
    return (float(best["price_native_per_gpu_hr"]) / ref_price - 1.0) * 100.0


def apply_jump_screen(
    payload: Dict[str, Any],
    reference: Optional[Dict[str, Any]],
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the delta report and apply the L5 quarantine in place.

    Returns {reference, deltas, quarantined}; ``payload`` observations of a
    quarantined source (target sku, non-interruptible) gain
    ``implausible=True`` + ``quarantined=QUARANTINE_REASON`` + a loud note.
    Fail-open by design: no reference snapshot -> report only, no flags.
    """
    params = screen_params(config)
    sku = config.get("target_sku", "B300")
    interruptible = tuple(
        (config.get("calc") or {}).get(
            "interruptible_tiers", ["spot", "preemptible"]
        )
    )
    # The reference is remote-derived (bucket JSON): any shape surprise
    # degrades to "no reference", never a crash — a broken reference must
    # not cost the capture.
    if not isinstance(reference, dict):
        reference = None
    ref_label = None
    ref_sources: Dict[str, Dict[str, Any]] = {}
    if reference:
        try:
            slot_txt = f"slot{int(reference.get('slot_hour_utc') or 0):02d}"
        except (TypeError, ValueError):
            slot_txt = "slot??"
        ref_label = f"{reference.get('capture_date', '?')} {slot_txt}"
        ref_sources = {
            s.get("source_id"): s
            for s in reference.get("sources") or []
            if isinstance(s, dict)
        }

    deltas: List[Dict[str, Any]] = []
    moves: Dict[str, float] = {}
    today_prints: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    basket_role = config.get("basket_role", "b300_basket")
    for src in payload.get("sources") or []:
        if src.get("role") != basket_role or src.get("status") != "ok":
            continue
        sid = src["source_id"]
        entry = {
            "source_id": sid,
            "book_pct": None,
            "same_machine_pct": None,
            "note": "",
        }
        today_any = lowest_eligible(src, sku, interruptible)
        ref = lowest_eligible(ref_sources.get(sid), sku, interruptible)
        if today_any is None:
            entry["note"] = "no eligible print today"
        elif ref is None:
            entry["note"] = "no reference print"
        elif float(ref["price_native_per_gpu_hr"]) <= 0:
            entry["note"] = "non-positive reference print"
        else:
            # Compare in the REFERENCE print's currency: a mixed-currency
            # source still gets a meaningful ratio from its same-currency
            # print instead of dropping out of the corroborator pool.
            today = lowest_eligible(
                src, sku, interruptible, currency=ref.get("currency", "USD")
            )
            if today is None:
                entry["note"] = "currency changed — not comparable"
            else:
                pct = (
                    float(today["price_native_per_gpu_hr"])
                    / float(ref["price_native_per_gpu_hr"])
                    - 1.0
                ) * 100.0
                entry["book_pct"] = round(pct, 2)
                moves[sid] = pct
                today_prints[sid] = (src, today)
                same = _same_machine_pct(src, ref, sku, interruptible)
                if same is not None:
                    entry["same_machine_pct"] = round(same, 2)
        deltas.append(entry)

    # Corroboration starvation guard: with fewer comparable sources than
    # min_corroborators + 1, a genuine market-wide move could never gather
    # enough corroborators and would be quarantined wholesale — fail open
    # (loudly) instead. A single-source glitch on a thin day slips this
    # capture; the delta report still shows it.
    would_fire = [
        sid
        for sid, pct in moves.items()
        if abs(pct) >= params["jump_quarantine_pct"]
    ]
    quarantine_skipped = None
    if would_fire and len(moves) < params["jump_min_corroborators"] + 1:
        quarantine_skipped = (
            f"only {len(moves)} comparable source(s) this capture — "
            f"corroboration needs {params['jump_min_corroborators'] + 1}+ "
            f"to be decidable; jump quarantine skipped for "
            f"{sorted(would_fire)} (fail-open)"
        )

    quarantined: List[Dict[str, Any]] = []
    if quarantine_skipped is None:
        for sid, pct in moves.items():
            if abs(pct) < params["jump_quarantine_pct"]:
                continue
            corroborators = sum(
                1
                for other, other_pct in moves.items()
                if other != sid
                and abs(other_pct) >= params["jump_corroborate_pct"]
            )
            if corroborators >= params["jump_min_corroborators"]:
                continue
            src, _ = today_prints[sid]
            for obs in src.get("observations") or []:
                if _eligible(obs, sku, interruptible):
                    obs["implausible"] = True
                    obs["quarantined"] = QUARANTINE_REASON
                    obs["notes"] = (
                        str(obs.get("notes") or "")
                        + f" [L5 QUARANTINE: book {pct:+.1f}% vs {ref_label}, "
                        f"{corroborators}/{params['jump_min_corroborators']} "
                        "corroborating movers — flagged for this capture only]"
                    ).strip()
            quarantined.append(
                {
                    "source_id": sid,
                    "book_pct": round(pct, 2),
                    "corroborators": corroborators,
                }
            )

    return {
        "reference": ref_label,
        "deltas": deltas,
        "quarantined": quarantined,
        "quarantine_skipped": quarantine_skipped,
    }
