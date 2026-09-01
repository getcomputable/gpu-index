# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Raw-only reproduction of the published weighting and index pipeline."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta
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

VERDICT_MATCH = "match"
VERDICT_MISMATCH = "mismatch"
FULL_HISTORY_BOUND_DAYS = 100


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


def _project_classifier_row(receipt: dict) -> dict:
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
    if upstream_status == "carried":
        detail["carried"] = {"carry_basis": receipt.get("carry_basis")}
    return detail


def public_attendance_events(observation: dict) -> Dict[str, str]:
    """Port the engine classifier onto the public receipt vocabulary."""
    rows = [_project_classifier_row(receipt) for receipt in observation["receipts"]]
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


def _trusted_receipts(observation: dict, manual: set[str]) -> Dict[str, dict]:
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


def read_full_history(reader: Any, *, sku: str, target_date: str) -> list[dict]:
    """Read the contiguous public history required by the weighting engine.

    The public 90-day series identifies the observable record origin. If a
    day exists immediately before that rolling window, the lane is older and
    the full 100-day disclosure bound is required. If it does not, the series
    begins at the lane's public corpus origin and replay starts from the
    engine's empty genesis state.
    """
    series = reader.read_series("90d", sku=sku)
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

    previous = reader.read_day((series_start - timedelta(days=1)).isoformat(), sku=sku)
    if previous is None:
        start = series_start
        bound_label = f"public corpus origin {series_start.isoformat()}"
    else:
        start = target - timedelta(days=FULL_HISTORY_BOUND_DAYS - 1)
        bound_label = (
            f"{FULL_HISTORY_BOUND_DAYS}-day history bound beginning "
            f"{start.isoformat()}"
        )

    observations = []
    cursor = start
    while cursor <= target:
        day = cursor.isoformat()
        envelope = previous if cursor == series_start - timedelta(days=1) else None
        if envelope is None:
            envelope = reader.read_day(day, sku=sku)
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
    lower = series_start.isoformat()
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
    observations: Iterable[dict], *, target_date: str
) -> FullReproduction:
    """Derive target-day weights, votes, IQM, and index from raw public rows."""
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
        dynamic, fallback = _dynamic_params(observation)
        observation_date = str(observation["observed_at"])[:10]
        manual = {
            str(row["source_id"])
            for row in params.get("manual_exclusions", [])
            if row.get("date") == observation_date
        }
        trusted = _trusted_receipts(observation, manual)
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
        carrying = sorted(
            str(receipt["source_id"])
            for receipt in receipts
            if receipt.get("upstream_status") == "carried"
            and receipt.get("carry_basis") == "no_price"
            and str(receipt["source_id"]) not in manual
            and not attendance[str(receipt["source_id"])]["excluded"]
        )
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
        for receipt in receipts:
            source_id = str(receipt["source_id"])
            if receipt.get("filter_verdict") != "accepted":
                continue
            upstream_status = receipt.get("upstream_status")
            if upstream_status == "carried":
                carried = carry_book.get(source_id)
                assert carried is not None
                price = carried["price"]
                sd = carried["sd"]
                weight = derived_weights.get(source_id)
            elif upstream_status == "ok":
                price = receipt.get("price")
                sd = receipt.get("sd")
                weight = block["weights"].get(source_id)
            else:
                continue
            if weight is None or not _is_number(price):
                continue
            if not _is_number(sd):
                raise FullReproductionRefusal(
                    "missing_dispersion_history",
                    f"{observation['observed_at']} {source_id}: a contributing "
                    "raw dispersion value is unavailable",
                )
            passing.append((source_id, float(weight), float(price)))
            stddevs[source_id] = float(sd)
        composite = (
            median_stddev_composite(
                passing,
                stddevs,
                iqm_alpha=float(params.get("iqm_alpha", 0.0)),
            )
            if len(passing) >= int(params["min_sources_to_publish"])
            else None
        )
        derived_value = None if composite is None else composite["value_usd_gpu_hr"]
        derived_band = (
            None if composite is None else composite["confidence_usd_gpu_hr"]
        )
        if observation_date == target_date:
            published_value = observation.get("value_usd_gpu_hr")
            published_band = observation.get("stability_band_usd_gpu_hr")
            divergence = _first_divergence(
                receipts,
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
