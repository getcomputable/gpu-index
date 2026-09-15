# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Raw-only reproduction of the published weighting and index pipeline."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, Optional, Tuple

from gpu_index.index.panel import (
    attendance_events_for_stamp,
    median_stddev_composite,
)
from gpu_index.index.panel_schedule import obs_key_to_stamp, stamp_to_obs_key
from gpu_index.index.weights import (
    advance_panel_weight_state,
    compute_attendance_view,
    compute_panel_weights,
    new_weight_state,
)
from gpu_index.published.artifacts import PublishedRecordError
from gpu_index.published.verify import (
    MIN_DISCLOSURE_WINDOW_DAYS,
    _carried_vote_disclosed,
    _history_bound_days,
)

VERDICT_MATCH = "match"
VERDICT_MISMATCH = "mismatch"
FULL_HISTORY_BOUND_DAYS = MIN_DISCLOSURE_WINDOW_DAYS
# The launch-era methodology also covers observations before public launch.
_PUBLIC_LAUNCH = datetime.fromisoformat("2026-09-01T00:13:39Z")


class FullReproductionRefusal(PublishedRecordError):
    """The public record does not expose enough raw history to derive."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FullDivergence:
    quantity: str
    source_id: Optional[str]
    derived: Any
    published: Any


@dataclass(frozen=True)
class FullObservationCheck:
    sku: str
    observed_at: str
    verdict: str
    derived_value: Optional[float]
    published_value: Optional[float]
    derived_band: Optional[float]
    published_band: Optional[float]
    derived_weights: Dict[str, float]
    first_divergence: Optional[FullDivergence] = None
    methodology_id: str = ""
    version: int | None = None


@dataclass(frozen=True)
class FullReproduction:
    checks: Tuple[FullObservationCheck, ...]


class _ObservedSchedule:
    """The exact public observation lattice, exposed as an engine schedule."""

    def __init__(self, observations: Iterable[dict]) -> None:
        keys = {
            _stamp(observation): str(observation["observed_at"])[11:16]
            for observation in observations
        }
        self._stamps = tuple(sorted(keys))
        if not self._stamps:
            raise FullReproductionRefusal(
                "empty_history", "the public history contains no observations"
            )
        self._minute_keyed = any(label[3:] != "00" for label in keys.values())
        self.genesis_stamp = self._stamps[0]

    def scheduled_stamps(self, window_start: int, window_end: int) -> list[int]:
        lo = bisect_left(self._stamps, int(window_start))
        hi = bisect_left(self._stamps, int(window_end))
        return list(self._stamps[lo:hi])

    def prev_scheduled_stamp(self, stamp: int) -> Optional[int]:
        index = bisect_left(self._stamps, int(stamp)) - 1
        return None if index < 0 else self._stamps[index]

    def stamp_key(self, stamp: int) -> str:
        return stamp_to_obs_key(stamp, minute_keyed=self._minute_keyed)


def _stamp(observation: dict) -> int:
    observed_at = observation.get("observed_at")
    if not isinstance(observed_at, str) or len(observed_at) < 16:
        raise FullReproductionRefusal(
            "invalid_observation_time",
            f"an observation has no usable observed_at: {observed_at!r}",
        )
    return obs_key_to_stamp(observed_at[:16].replace(":", ""))


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _smoothing_armed(params: dict, *, observed_at: Any) -> bool:
    """COM-1582: ``calc_params.pre_smoothing_half_life_hours`` arms the
    smoothed seat law for the whole observation. Usability-validated only
    (the (0, 2] mint ceiling is lane law, not a reproduce precondition);
    absent means the pre-smoothing generations replay byte-identically."""
    if "pre_smoothing_half_life_hours" not in params:
        return False
    half_life = params["pre_smoothing_half_life_hours"]
    if not _is_number(half_life) or half_life <= 0:
        raise FullReproductionRefusal(
            "invalid_smoothing_params",
            f"{observed_at}: calc_params.pre_smoothing_half_life_hours "
            f"must be a finite number > 0, got {half_life!r}",
        )
    return True


def _cast_price(receipt: dict, *, observed_at: Any) -> float:
    """The disclosed cast price on a smoothing-armed lane's voting row.

    This reproduction never reruns the engine's EWMA: ``smoothed_vote_usd``
    is the EXACT price the engine aggregated (fresh rows: the seat's
    smoothed series advanced by this print; carried flavors: the frozen
    smoothed booked price), so the vote prices THAT number and the row's
    raw ``price`` stays evidence. A voting row without the disclosure (or
    with an unusable one) is a torn artifact: refuse loudly, never fall
    back to the raw print (the 2026-09-14T0100 incident posture)."""
    source_id = receipt.get("source_id")
    cast = receipt.get("smoothed_vote_usd")
    if cast is None:
        raise FullReproductionRefusal(
            "missing_cast_price",
            f"{observed_at} {source_id}: the seat votes on a "
            "smoothing-armed lane but discloses no smoothed_vote_usd -- "
            "the cast price is unknowable, refusing to reproduce (no "
            "raw-price fallback)",
        )
    if not _is_number(cast) or cast <= 0:
        raise FullReproductionRefusal(
            "unusable_cast_price",
            f"{observed_at} {source_id}: smoothed_vote_usd {cast!r} is "
            "not a finite positive number -- torn artifact, refusing to "
            "reproduce (no raw-price fallback)",
        )
    return float(cast)


def _dynamic_params(observation: dict) -> tuple[dict, Dict[str, float]]:
    params = observation.get("calc_params")
    if not isinstance(params, dict) or not isinstance(params.get("liveness"), dict):
        raise FullReproductionRefusal(
            "missing_liveness_params",
            f"{observation.get('observed_at')}: calc_params.liveness is required",
        )
    members = params.get("members")
    if not isinstance(members, list) or not members:
        raise FullReproductionRefusal(
            "missing_opening_weights",
            f"{observation.get('observed_at')}: calc_params.members is required",
        )
    fallback = {
        str(member["source_id"]): float(member["opening_weight"])
        for member in members
    }
    dynamic = dict(params["liveness"])
    dynamic["fallback_weights"] = fallback
    dynamic["source_weight_caps"] = {}
    return dynamic, fallback


def _project_classifier_row(receipt: dict, *, smoothing_armed: bool) -> dict:
    if "upstream_status" not in receipt:
        raise FullReproductionRefusal(
            "missing_upstream_status",
            f"receipt {receipt.get('source_id')!r} has no upstream_status",
        )
    upstream_status = receipt["upstream_status"]
    detail: Dict[str, Any] = {
        "source_id": receipt.get("source_id"),
        "status": upstream_status,
    }
    if upstream_status == "ok" and receipt.get("price") is not None:
        detail["chosen"] = {}
        if receipt.get("filter_verdict") == "not_evaluated":
            detail["filter"] = {"untrusted_currency": True}
        else:
            detail["filter"] = {}
        if smoothing_armed and _carried_vote_disclosed(receipt):
            # COM-1570 fence-reject carry: the row keeps the real
            # rejected print (status/price/verdict untouched), only the
            # VOTE was substituted from the carry book -- the classifier
            # must read the seat absent, exactly like the engine's own
            # carried_vote arm. Armed lanes only: pre-smoothing bytes
            # classify unchanged whatever a row happens to carry. The
            # corpus flattens the engine block; rebuild the engine shape
            # the ported classifier expects.
            detail["carried_vote"] = {"from": receipt["carried_vote_from"]}
            if isinstance(receipt.get("carry_basis"), str):
                detail["carried_vote"]["carry_basis"] = receipt["carry_basis"]
    if upstream_status == "carried":
        detail["carried"] = {"carry_basis": receipt.get("carry_basis")}
    return detail


def public_attendance_events(observation: dict) -> Dict[str, str]:
    """Port the engine classifier onto the public receipt vocabulary."""
    smoothing_armed = _smoothing_armed(
        observation.get("calc_params") or {},
        observed_at=observation.get("observed_at"),
    )
    rows = [
        _project_classifier_row(receipt, smoothing_armed=smoothing_armed)
        for receipt in observation["receipts"]
    ]
    reason = observation.get("reason")
    return attendance_events_for_stamp(
        rows,
        observation_missed=(
            bool(observation.get("observation_missed"))
            or reason == "observation_missed"
        ),
        record_quarantined=(
            observation.get("record_quarantined")
            or (reason if reason == "record_quarantined" else None)
        ),
    )


def _trusted_receipts(
    observation: dict, manual: set[str], *, smoothing_armed: bool = False
) -> Dict[str, dict]:
    trusted: Dict[str, dict] = {}
    for receipt in observation["receipts"]:
        source_id = str(receipt["source_id"])
        if receipt.get("price_disclosure") == "withheld":
            if receipt.get("upstream_status") in ("ok", "carried"):
                raise FullReproductionRefusal(
                    "withheld_price_history",
                    f"{observation['observed_at']} {source_id}: the raw price "
                    "history is withheld",
                )
            continue
        if smoothing_armed and _carried_vote_disclosed(receipt):
            # COM-1570 fence-reject carry: the real rejected print stays
            # on the row, but the engine deliberately never advances the
            # weight-series presence record with it (the seat must read
            # attendance-ABSENT) -- so this reproduction must not feed
            # the weight prints or the carry book from it either. A
            # plain fence reject (no disclosure) keeps today's rule: the
            # fence holds a print out of the INDEX, never out of the
            # presence record.
            continue
        if (
            receipt.get("upstream_status") == "ok"
            and receipt.get("carry_basis") is None
            and receipt.get("filter_verdict") in ("accepted", "rejected")
            and _is_number(receipt.get("price"))
            and source_id not in manual
        ):
            trusted[source_id] = receipt
    return trusted


def public_weight_print(receipt: dict, *, observed_at: str) -> dict:
    """Restore the engine's recorded-currency weight-series print."""
    source_id = str(receipt.get("source_id"))
    price = receipt.get("price")
    currency = receipt.get("currency")
    if not _is_number(price) or not isinstance(currency, str) or not currency:
        raise FullReproductionRefusal(
            "missing_weight_print_terms",
            f"{observed_at} {source_id}: price and recorded currency are "
            "required to derive the weight history",
        )
    fx_rate = receipt.get("fx_rate")
    if fx_rate is None:
        if currency != "USD":
            raise FullReproductionRefusal(
                "missing_weight_print_terms",
                f"{observed_at} {source_id}: a non-USD weight-history "
                "print requires its public FX rate",
            )
        native = float(price)
    elif _is_number(fx_rate) and float(fx_rate) > 0:
        native = round(float(price) / float(fx_rate), 6)
    else:
        raise FullReproductionRefusal(
            "missing_weight_print_terms",
            f"{observed_at} {source_id}: the public FX rate is not a "
            "positive finite number",
        )
    return {
        "usd": float(price),
        "native": native,
        "currency": currency,
    }


def _first_divergence(
    receipts: list[dict],
    block: dict,
    derived_weights: Dict[str, float],
    *,
    derived_value: Optional[float],
    published_value: Optional[float],
    derived_band: Optional[float],
    published_band: Optional[float],
) -> Optional[FullDivergence]:
    by_source = {str(receipt["source_id"]): receipt for receipt in receipts}
    comparisons = (
        ("attendance", "attendance_factor", "attendance_factor"),
        ("score", "Q", "liveness_score"),
    )
    for quantity, derived_key, published_key in comparisons:
        for source_id in sorted(by_source):
            derived = (block["sources"].get(source_id) or {}).get(derived_key)
            published = by_source[source_id].get(published_key)
            if derived != published:
                return FullDivergence(
                    quantity, source_id, derived, published
                )
    for source_id in sorted(by_source):
        derived = derived_weights.get(source_id)
        published = by_source[source_id].get("weight")
        if derived != published:
            return FullDivergence("weight", source_id, derived, published)
    derived = (derived_value, derived_band)
    published = (published_value, published_band)
    if derived != published:
        return FullDivergence("value", None, derived, published)
    return None


def read_full_history(
    reader: Any, *, sku: str, target_date: str, version: int | None = None
) -> list[dict]:
    """Read the contiguous public history required by the weighting engine.

    The public 90-day series identifies the observable record origin. If a
    day exists immediately before that rolling window, walk backward until
    the disclosure bound or the first unavailable day. A younger version
    starts at its public corpus origin, from the engine's empty genesis state.
    """
    # One version for the series, origin probe and every history day.
    version_args = {"version": version} if version is not None else {}
    series = reader.read_series("90d", sku=sku, **version_args)
    from_observed_at = (series or {}).get("meta", {}).get("from_observed_at")
    if not isinstance(from_observed_at, str) or len(from_observed_at) < 10:
        raise FullReproductionRefusal(
            "history_origin_unavailable",
            "the public 90d series does not disclose from_observed_at; "
            f"full reproduction needs the {FULL_HISTORY_BOUND_DAYS}-day "
            "history bound or a public corpus origin",
        )
    try:
        target = date.fromisoformat(target_date)
        series_start = date.fromisoformat(from_observed_at[:10])
    except ValueError as exc:
        raise FullReproductionRefusal(
            "invalid_history_date", f"the public history date is invalid: {exc}"
        ) from exc
    if series_start > target:
        raise FullReproductionRefusal(
            "target_not_observable",
            f"the public 90d series begins at {series_start}, after {target}",
        )

    target_day = reader.read_day(target_date, sku=sku, **version_args)
    target_rows = (target_day or {}).get("data", {}).get("observations", [])
    bound_days = max(
        (_history_bound_days(
            history_days=row["calc_params"]["liveness"]["history_days"],
            forward_horizons_hours=row["calc_params"]["liveness"]["forward_horizons_hours"],
        ) for row in target_rows if row.get("calc_params", {}).get("liveness")),
        default=FULL_HISTORY_BOUND_DAYS,
    )
    bound_start = target - timedelta(days=bound_days - 1)
    start = max(series_start, bound_start)
    cached_days = {target: target_day}
    # Walk back from an observable day: a young series may contain 91-99
    # days even though the full disclosure-bound day does not exist yet.
    while start > bound_start:
        previous_date = start - timedelta(days=1)
        previous = reader.read_day(previous_date.isoformat(), sku=sku, **version_args)
        if previous is None:
            break
        cached_days[previous_date] = previous
        start = previous_date
    bound_label = (
        f"{bound_days}-day history bound beginning {start.isoformat()}"
        if start == bound_start
        else f"public corpus origin {start.isoformat()}"
    )

    observations = []
    cursor = start
    while cursor <= target:
        day = cursor.isoformat()
        envelope = cached_days.get(cursor)
        if envelope is None:
            envelope = reader.read_day(day, sku=sku, **version_args)
        if envelope is None:
            raise FullReproductionRefusal(
                "insufficient_observable_history",
                f"full reproduction requires a contiguous {bound_label}; "
                f"published day {day} is unavailable",
            )
        rows = (envelope.get("data") or {}).get("observations")
        if not isinstance(rows, list):
            raise FullReproductionRefusal(
                "invalid_history_day",
                f"published day {day} has no observations array",
            )
        observations.extend(rows)
        cursor += timedelta(days=1)

    series_rows = (series.get("data") or {}).get("observations")
    if not isinstance(series_rows, list):
        raise FullReproductionRefusal(
            "insufficient_observable_history",
            "the public 90d series does not expose its observation lattice",
        )
    lower = max(series_start, start).isoformat()
    upper = target.isoformat()
    expected = sorted(
        str(row.get("observed_at"))
        for row in series_rows
        if isinstance(row, dict)
        and isinstance(row.get("observed_at"), str)
        and lower <= str(row["observed_at"])[:10] <= upper
    )
    actual = sorted(
        str(row.get("observed_at"))
        for row in observations
        if isinstance(row, dict)
        and isinstance(row.get("observed_at"), str)
        and lower <= str(row["observed_at"])[:10] <= upper
    )
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        detail = []
        if missing:
            detail.append(f"missing {missing[0]}")
        if extra:
            detail.append(f"unexpected {extra[0]}")
        raise FullReproductionRefusal(
            "insufficient_observable_history",
            "the public series/day observation lattice differs"
            + (f" ({'; '.join(detail)})" if detail else ""),
        )
    return observations


def reproduce_full_history(
    observations: Iterable[dict], *, target_date: str,
    comparison_rows: dict[str, dict] | None = None,
) -> FullReproduction:
    """Derive target-day weights, votes, IQM, and index from raw public rows.

    Optional comparison rows supply published outputs only; every derivation
    input and state transition still comes from the version history.
    """
    history = sorted(list(observations), key=_stamp)
    schedule = _ObservedSchedule(history)
    state = new_weight_state()
    carry_book: Dict[str, dict] = {}
    checks = []

    for observation in history:
        receipts = observation.get("receipts")
        if not isinstance(receipts, list):
            raise FullReproductionRefusal(
                "missing_receipts",
                f"{observation.get('observed_at')}: receipts are required",
            )
        events = public_attendance_events(observation)
        params = observation["calc_params"]
        smoothing_armed = _smoothing_armed(
            params, observed_at=observation["observed_at"]
        )
        dynamic, fallback = _dynamic_params(observation)
        observation_date = str(observation["observed_at"])[:10]
        manual = {
            str(row["source_id"])
            for row in params.get("manual_exclusions", [])
            if row.get("date") == observation_date
        }
        trusted = _trusted_receipts(
            observation, manual, smoothing_armed=smoothing_armed
        )
        obs_stamp = _stamp(observation)
        attendance = compute_attendance_view(
            state,
            sorted(fallback),
            obs_stamp=obs_stamp,
            schedule=schedule,
            dw_params=dynamic,
        )
        eligible = [
            source_id
            for source_id in sorted(trusted)
            if not attendance[source_id]["excluded"]
        ]
        carrying_ids = {
            str(receipt["source_id"])
            for receipt in receipts
            if receipt.get("upstream_status") == "carried"
            and receipt.get("carry_basis") == "no_price"
            and str(receipt["source_id"]) not in manual
            and not attendance[str(receipt["source_id"])]["excluded"]
        }
        if smoothing_armed:
            # COM-1570 fence-reject carried votes ride the SAME D4
            # fading-weight domain as the state-2 no_price carries: the
            # seat is absent from the presence record this stamp, but its
            # CURRENT (fading) weight prices its booked vote -- never the
            # booked weight (that is the state-3 machinery's rule).
            carrying_ids |= {
                str(receipt["source_id"])
                for receipt in receipts
                if _carried_vote_disclosed(receipt)
                and str(receipt["source_id"]) not in manual
                and not attendance[str(receipt["source_id"])]["excluded"]
            }
        carrying = sorted(carrying_ids)
        block = compute_panel_weights(
            state,
            obs_stamp=obs_stamp,
            eligible=eligible,
            dw_params=dynamic,
            fallback_weights=fallback,
            schedule=schedule,
            carrying=carrying,
            attendance_view=attendance,
        )

        derived_weights: Dict[str, float] = {}
        for receipt in receipts:
            source_id = str(receipt["source_id"])
            if receipt.get("upstream_status") == "carried":
                carried = carry_book.get(source_id)
                if carried is None:
                    raise FullReproductionRefusal(
                        "insufficient_carry_history",
                        f"{observation['observed_at']} {source_id}: the public "
                        "history has no prior accepted raw vote for this carry",
                    )
                weight = (
                    block["weights"].get(source_id)
                    if receipt.get("carry_basis") == "no_price"
                    else carried["weight"]
                )
            else:
                weight = block["weights"].get(source_id)
            if weight is not None:
                derived_weights[source_id] = float(weight)

        passing = []
        stddevs = {}
        carried_vote_count = 0
        for receipt in receipts:
            source_id = str(receipt["source_id"])
            fence_reject_carried = smoothing_armed and _carried_vote_disclosed(
                receipt
            )
            if (
                receipt.get("filter_verdict") != "accepted"
                and not fence_reject_carried
            ):
                continue
            upstream_status = receipt.get("upstream_status")
            carried_voter = False
            if fence_reject_carried and upstream_status == "ok":
                # COM-1570 fence-reject carry: the rejected verdict judged
                # the REAL print the row keeps as evidence; the engine
                # cast the seat's booked smoothed vote instead, at its
                # CURRENT fading weight (never the booked weight) with
                # the BOOKED vote dispersion.
                carried = carry_book.get(source_id)
                if carried is None:
                    raise FullReproductionRefusal(
                        "insufficient_carry_history",
                        f"{observation['observed_at']} {source_id}: the public "
                        "history has no prior accepted raw vote for this "
                        "fence-reject carry",
                    )
                weight = derived_weights.get(source_id)
                if weight is None:
                    continue
                price = _cast_price(
                    receipt, observed_at=observation["observed_at"]
                )
                sd = carried["sd"]
                carried_voter = True
            elif upstream_status == "carried":
                carried = carry_book.get(source_id)
                assert carried is not None
                weight = derived_weights.get(source_id)
                if weight is None:
                    continue
                # Smoothing-armed carried rows re-cast the FROZEN smoothed
                # state, disclosed on the row itself; pre-smoothing rows
                # re-cast the booked raw vote resolved from prior bytes.
                price = (
                    _cast_price(
                        receipt, observed_at=observation["observed_at"]
                    )
                    if smoothing_armed
                    else carried["price"]
                )
                sd = carried["sd"]
                carried_voter = True
            elif upstream_status == "ok":
                weight = block["weights"].get(source_id)
                if weight is None:
                    # A voteless fresh print (e.g. the K_A recovery
                    # print) never voted upstream and discloses no cast
                    # price -- the weight gate must run BEFORE any
                    # cast-price demand.
                    continue
                price = (
                    _cast_price(
                        receipt, observed_at=observation["observed_at"]
                    )
                    if smoothing_armed
                    else receipt.get("price")
                )
                sd = receipt.get("sd")
            else:
                continue
            if not _is_number(price):
                continue
            if not _is_number(sd):
                raise FullReproductionRefusal(
                    "missing_dispersion_history",
                    f"{observation['observed_at']} {source_id}: a contributing "
                    "raw dispersion value is unavailable",
                )
            passing.append((source_id, float(weight), float(price)))
            stddevs[source_id] = float(sd)
            if carried_voter:
                carried_vote_count += 1
        # The claim-floor law: carried votes (status-carried re-casts AND
        # fence-reject carried votes) may move the median but never keep
        # a dying panel lit -- the floor counts OBSERVED voters only on
        # the smoothing-armed generations; pre-smoothing bytes keep the
        # frozen count untouched.
        observed_count = (
            len(passing) - carried_vote_count
            if smoothing_armed
            else len(passing)
        )
        composite = (
            median_stddev_composite(
                passing,
                stddevs,
                iqm_alpha=float(params.get("iqm_alpha", 0.0)),
            )
            if observed_count >= int(params["min_sources_to_publish"])
            else None
        )
        derived_value = None if composite is None else composite["value_usd_gpu_hr"]
        derived_band = (
            None if composite is None else composite["confidence_usd_gpu_hr"]
        )
        if observation_date == target_date:
            comparison = (
                observation if comparison_rows is None
                else comparison_rows.get(observation["observed_at"], observation)
            )
            published_value = comparison.get("value_usd_gpu_hr")
            published_band = comparison.get("stability_band_usd_gpu_hr")
            divergence = _first_divergence(
                comparison["receipts"],
                block,
                derived_weights,
                derived_value=derived_value,
                published_value=published_value,
                derived_band=derived_band,
                published_band=published_band,
            )
            checks.append(
                FullObservationCheck(
                    sku=str(observation.get("sku", "")),
                    observed_at=str(observation["observed_at"]),
                    verdict=(
                        VERDICT_MATCH
                        if divergence is None
                        else VERDICT_MISMATCH
                    ),
                    derived_value=derived_value,
                    published_value=published_value,
                    derived_band=derived_band,
                    published_band=published_band,
                    derived_weights=derived_weights,
                    first_divergence=divergence,
                    methodology_id=str(observation.get("methodology_id", "")),
                )
            )

        for source_id, receipt in trusted.items():
            if receipt.get("filter_verdict") != "accepted":
                continue
            weight = block["weights"].get(source_id)
            sd = receipt.get("sd")
            if weight is None or not _is_number(sd):
                continue
            carry_book[source_id] = {
                "stamp": obs_stamp,
                "price": float(receipt["price"]),
                "sd": float(sd),
                "weight": float(weight),
            }

        prints = {
            source_id: public_weight_print(
                receipt, observed_at=str(observation["observed_at"])
            )
            for source_id, receipt in trusted.items()
        }
        advance_panel_weight_state(
            state,
            obs_stamp=obs_stamp,
            prints=prints,
            vector=block["weights"],
            mode=block["mode"],
            dw_params=dynamic,
            events=events,
        )

    if not checks:
        raise FullReproductionRefusal(
            "target_not_observable",
            f"the public history contains no observation for {target_date}",
        )
    return FullReproduction(checks=tuple(checks))


def reproduce_published_history(
    reader: Any, *, sku: str, target_date: str,
) -> FullReproduction:
    """Re-derive each as-published stamp within its effective version's history."""
    pointer = reader.version_pointer(sku)
    if pointer is None or "history_path" not in pointer:
        raise FullReproductionRefusal(
            "history_path_unavailable",
            f"latest.json does not advertise as-published history for {sku}",
        )
    envelope = reader.read_day(target_date, sku=sku)
    rows = (envelope or {}).get("data", {}).get("observations", [])
    if not rows:
        raise FullReproductionRefusal(
            "target_not_observable", f"no {sku} observations for {target_date}"
        )
    succession = sorted(
        pointer["succession"], key=lambda entry: datetime.fromisoformat(entry["effective_from"])
    )
    groups: dict[int, dict[str, dict]] = {}
    for row in rows:
        stamp = row["observed_at"]
        at = max(datetime.fromisoformat(stamp), _PUBLIC_LAUNCH)
        live = [entry for entry in succession
                if datetime.fromisoformat(entry["effective_from"]) <= at]
        if not live:
            raise FullReproductionRefusal(
                "version_unavailable", f"{stamp}: no effective version is advertised"
            )
        selected = live[-1]
        if row["sku"] != sku or row["methodology_id"] != selected["methodology_id"]:
            raise PublishedRecordError(
                f"{stamp}: as-published row disagrees with effective version "
                f"{selected['version']} methodology_id {selected['methodology_id']}"
            )
        targets = groups.setdefault(selected["version"], {})
        if stamp in targets:
            raise PublishedRecordError(f"{stamp}: repeated as-published observation")
        targets[stamp] = row

    checks = []
    for version, targets in groups.items():
        history = read_full_history(reader, sku=sku, target_date=target_date, version=version)
        run = reproduce_full_history(history, target_date=target_date, comparison_rows=targets)
        selected_checks = [check for check in run.checks if check.observed_at in targets]
        if sorted(check.observed_at for check in selected_checks) != sorted(targets):
            raise FullReproductionRefusal(
                "target_not_observable",
                f"version {version}: history does not contain every as-published stamp",
            )
        checks.extend(replace(check, version=version) for check in selected_checks)
    return FullReproduction(checks=tuple(sorted(checks, key=lambda check: check.observed_at)))
