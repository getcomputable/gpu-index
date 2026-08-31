# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Recompute-and-match over published observations.

Each published ``gpu_price_index_observation`` carries per-provider
receipts: ``{price, sd, weight, status, filter_verdict, ...}`` plus the
disclosure flag ``price_disclosure`` ("published" | "withheld"). The
producer's aggregate is a pure function of exactly those receipt fields
(the panel engine prices from the ROUNDED published per-source values by
design, so the artifact alone recomputes it):

  - passing set = receipts with status "ok" AND filter_verdict
    "accepted" (the panel's own predicate; the publisher emits weight
    and sd non-null exactly for that contributing set);
  - each passing source votes its weight at price and price +/- sd
    (three votes), and the index is the declared interquantile mean of
    those votes (or the frozen-v1 weighted median when alpha is absent);
    the published stability band is the larger distance from the index
    to the 25th/75th weighted vote percentiles.

The vote/IQM math is IMPORTED from the panel engine
(``gpu_index.index.panel.median_stddev_composite``) — the same function
that priced the observation — never duplicated here, so this check can
only diverge from production if the inputs diverge.

Withheld sources: the publisher's disclosure pass nulls price+sd on a
receipt and marks it ``price_disclosure: "withheld"`` while the
published index value is
unchanged. A withheld CONTRIBUTING receipt therefore makes the exact
vote rebuild impossible and the observation degrades to
digest-verification only, saying which sources are withheld. A withheld
non-contributing receipt (rejected/excluded/never priced into the
composite) does not impair the recompute and full verification proceeds.

No-print observations (value null) are checked for consistency instead:
the passing set must be below ``calc_params.min_sources_to_publish``
(the same minimum-panel rule the panel applies), and — when every receipt is
disclosed — the published no-print reason must re-derive from the
receipts (the publisher derives it from the same receipt fields).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from gpu_index.common.jsondiff import field_diffs
from gpu_index.index.panel import median_stddev_composite
from gpu_index.published.artifacts import PublishedRecordError

VERDICT_MATCH = "match"
VERDICT_MISMATCH = "mismatch"
VERDICT_DEGRADED = "degraded"

_SUPPORTED_AGGREGATION = "median_stddev_votes"
_ERA3_IQM_ALPHA = 0.16666


class UnsupportedStatisticError(PublishedRecordError):
    """The observation declares aggregate math this verifier cannot run."""

# The published disclosure window must cover the weighting methodology's
# lookback, or the reproducibility claim for the WEIGHTS degrades.
# Derivation: liveness weights are fitted over a 90-day history of
# samples whose forward outcomes extend up to 2 days past each sample
# (METHODOLOGY.md section 8), so re-deriving one day's weight vector from
# published observations needs 90 + 2 = 92 days of published history
# before it; 100 leaves slack for edge effects at the window boundaries.
# Per-observation recompute-and-match is unaffected either way: it
# consumes only each observation's own receipts, which embed the
# liveness weights as published. What a shorter window costs is the
# ability to re-derive the weight vector itself from the public record.
MIN_DISCLOSURE_WINDOW_DAYS = 100  # 90d lookback + 2d forward + slack


def disclosure_window_warning(
    *, sku: str, verified_date: str, probe_date: str
) -> str:
    """Non-fatal warning text for a full-history check over a published
    window shorter than ``MIN_DISCLOSURE_WINDOW_DAYS``.

    The caller has probed the record source for the day file
    ``MIN_DISCLOSURE_WINDOW_DAYS - 1`` days before ``verified_date`` and
    found it unavailable, so the observable published history for the
    lane is shorter than the bound. The warning never fails a run.
    """
    return (
        f"{sku} {verified_date}: the published day file "
        f"{MIN_DISCLOSURE_WINDOW_DAYS - 1} days back ({probe_date}) is "
        "not observable from this record source, so the observable "
        "published history is shorter than the "
        f"{MIN_DISCLOSURE_WINDOW_DAYS}-day disclosure bound (90d "
        "weighting lookback + 2d forward outcomes + slack). The "
        "recompute-and-match above is unaffected: it consumes only each "
        "observation's own receipts, which embed the liveness weights "
        "as published. What the shorter window withholds is re-deriving "
        "the weight vector itself from the public record for this day"
    )

# Receipt-derived no-print reasons, exactly as the publisher derives them.
_RECEIPT_DERIVED_REASONS = (
    "no_eligible_sources",
    "all_sources_filtered",
    "insufficient_coverage",
)


@dataclass(frozen=True)
class ObservationCheck:
    """One observation's recompute-and-match outcome."""

    sku: str
    observed_at: str
    status: str  # published status: "ok" | "no_print"
    verdict: str  # VERDICT_MATCH | VERDICT_MISMATCH | VERDICT_DEGRADED
    published_value: Optional[float] = None
    published_band: Optional[float] = None
    recomputed_value: Optional[float] = None
    recomputed_band: Optional[float] = None
    withheld_sources: Tuple[str, ...] = ()
    messages: Tuple[str, ...] = field(default=())


def _receipt_field(receipt: dict, name: str, index: int) -> Any:
    if name not in receipt:
        raise PublishedRecordError(
            f"receipts[{index}] is missing the {name!r} field the "
            "recompute consumes"
        )
    return receipt[name]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def recompute_observation(observation: dict) -> ObservationCheck:
    """Recompute one published observation from its own receipts and
    match the published index value and stability band exactly."""
    sku = observation.get("sku")
    observed_at = observation.get("observed_at")
    status = observation.get("status")
    if not isinstance(sku, str) or not isinstance(observed_at, str):
        raise PublishedRecordError(
            "observation is missing its sku/observed_at identity"
        )
    if status not in ("ok", "no_print"):
        raise PublishedRecordError(
            f"observation {sku} {observed_at} has unknown status {status!r}"
        )
    if observation.get("kind") != "gpu_price_index_observation":
        raise PublishedRecordError(
            f"observation {sku} {observed_at} kind "
            f"{observation.get('kind')!r} is not gpu_price_index_observation"
        )
    calc_params = observation.get("calc_params")
    if not isinstance(calc_params, dict):
        raise PublishedRecordError(
            f"observation {sku} {observed_at} has no calc_params"
        )
    aggregation = calc_params.get("aggregation")
    if aggregation != _SUPPORTED_AGGREGATION:
        raise UnsupportedStatisticError(
            f"observation {sku} {observed_at} declares unsupported "
            f"calc_params.aggregation {aggregation!r}; this verifier "
            f"implements {_SUPPORTED_AGGREGATION!r}"
        )
    iqm_alpha = calc_params.get("iqm_alpha", 0.0)
    if not _finite_number(iqm_alpha) or not 0 <= iqm_alpha <= 0.5:
        raise PublishedRecordError(
            f"observation {sku} {observed_at} calc_params.iqm_alpha "
            f"must be a finite number in [0, 0.5], got {iqm_alpha!r}"
        )
    min_to_publish = calc_params.get("min_sources_to_publish")
    if not isinstance(min_to_publish, int) or min_to_publish < 1:
        raise PublishedRecordError(
            f"observation {sku} {observed_at} "
            f"calc_params.min_sources_to_publish is {min_to_publish!r}"
        )
    receipts = observation.get("receipts")
    if not isinstance(receipts, list):
        raise PublishedRecordError(
            f"observation {sku} {observed_at} has no receipts array"
        )

    passing: List[Tuple[str, float, float]] = []
    vote_stddevs: Dict[str, float] = {}
    withheld_contributing: List[str] = []
    any_withheld = False
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise PublishedRecordError(
                f"receipts[{index}] must be an object"
            )
        source_id = _receipt_field(receipt, "source_id", index)
        disclosure = _receipt_field(receipt, "price_disclosure", index)
        if disclosure not in ("published", "withheld"):
            raise PublishedRecordError(
                f"receipts[{index}] price_disclosure {disclosure!r} is "
                "neither 'published' nor 'withheld'"
            )
        if disclosure == "withheld":
            any_withheld = True
        contributing = (
            _receipt_field(receipt, "status", index) == "ok"
            and _receipt_field(receipt, "filter_verdict", index)
            == "accepted"
        )
        if not contributing:
            continue
        if disclosure == "withheld":
            withheld_contributing.append(source_id)
            continue
        price = _receipt_field(receipt, "price", index)
        sd = _receipt_field(receipt, "sd", index)
        weight = _receipt_field(receipt, "weight", index)
        if not (
            _finite_number(price)
            and _finite_number(sd)
            and _finite_number(weight)
        ):
            raise PublishedRecordError(
                f"contributing receipt {source_id} carries non-numeric "
                f"price/sd/weight: {price!r}/{sd!r}/{weight!r}"
            )
        # The exact tuple the panel engine fed the vote aggregate:
        # (source_id, float(weight), price) with the ROUNDED published
        # sd as the vote stddev (panel.py compute_observation).
        passing.append((source_id, float(weight), price))
        vote_stddevs[source_id] = sd

    if withheld_contributing:
        return ObservationCheck(
            sku=sku,
            observed_at=observed_at,
            status=status,
            verdict=VERDICT_DEGRADED,
            published_value=observation.get("value_usd_gpu_hr"),
            published_band=observation.get("stability_band_usd_gpu_hr"),
            withheld_sources=tuple(withheld_contributing),
            messages=(
                "degraded to digest-verification only: contributing "
                "source(s) "
                + ", ".join(withheld_contributing)
                + " have price_disclosure 'withheld' (price and sd "
                "nulled by the publisher's disclosure policy), so the "
                "exact vote rebuild is impossible for this observation",
            ),
        )

    # The minimum-panel rule, verbatim from the panel engine: a composite
    # exists iff the passing set reaches min_sources_to_publish.
    composite = (
        median_stddev_composite(
            passing,
            vote_stddevs,
            iqm_alpha=iqm_alpha,
        )
        if len(passing) >= min_to_publish
        else None
    )

    published_value = observation.get("value_usd_gpu_hr")
    published_band = observation.get("stability_band_usd_gpu_hr")

    if status == "no_print":
        messages: List[str] = []
        if published_value is not None or published_band is not None:
            messages.append(
                "no_print observation carries a non-null value/band"
            )
        if composite is not None:
            messages.append(
                f"published no_print but the receipts rebuild a composite "
                f"({len(passing)} passing sources >= "
                f"min_sources_to_publish {min_to_publish}): recomputed "
                f"value {composite['value_usd_gpu_hr']}"
            )
        reason = observation.get("reason")
        if (
            not messages
            and not any_withheld
            and reason in _RECEIPT_DERIVED_REASONS
        ):
            derived = _no_print_reason(receipts)
            if derived != reason:
                messages.append(
                    f"reason: published {reason!r} vs recomputed "
                    f"{derived!r} from the receipts"
                )
        return ObservationCheck(
            sku=sku,
            observed_at=observed_at,
            status=status,
            verdict=VERDICT_MISMATCH if messages else VERDICT_MATCH,
            published_value=published_value,
            published_band=published_band,
            messages=tuple(messages),
        )

    # status == "ok": the recompute must land exactly on the published
    # value and stability band (same rounding: the engine publishes both at
    # round(x, 6) and the receipts carry the exact same floats back).
    if composite is None:
        return ObservationCheck(
            sku=sku,
            observed_at=observed_at,
            status=status,
            verdict=VERDICT_MISMATCH,
            published_value=published_value,
            published_band=published_band,
            messages=(
                f"published status ok but only {len(passing)} passing "
                f"disclosed sources (< min_sources_to_publish "
                f"{min_to_publish}): no composite is recomputable",
            ),
        )
    recomputed_value = composite["value_usd_gpu_hr"]
    recomputed_band = composite["confidence_usd_gpu_hr"]
    diffs = field_diffs(
        {
            "value_usd_gpu_hr": published_value,
            "stability_band_usd_gpu_hr": published_band,
        },
        {
            "value_usd_gpu_hr": recomputed_value,
            "stability_band_usd_gpu_hr": recomputed_band,
        },
    )
    if diffs and "iqm_alpha" not in calc_params:
        # The official verdict above always follows the artifact's declared
        # params: absent alpha means the frozen-v1 default 0.0. This second
        # aggregate is diagnostic only. It recognizes the COM-1451 failure
        # shape without silently accepting the undisclosed era3 rule.
        era3_composite = median_stddev_composite(
            passing, vote_stddevs, iqm_alpha=_ERA3_IQM_ALPHA
        )
        if (
            era3_composite is not None
            and era3_composite["value_usd_gpu_hr"] == published_value
            and era3_composite["confidence_usd_gpu_hr"] == published_band
        ):
            diffs.append(
                "calc_params.iqm_alpha is absent, so the verifier used "
                "the frozen-v1 default 0.0; the published value and band "
                f"instead match iqm_alpha {_ERA3_IQM_ALPHA}, which the "
                "publisher must disclose"
            )
    return ObservationCheck(
        sku=sku,
        observed_at=observed_at,
        status=status,
        verdict=VERDICT_MISMATCH if diffs else VERDICT_MATCH,
        published_value=published_value,
        published_band=published_band,
        recomputed_value=recomputed_value,
        recomputed_band=recomputed_band,
        messages=tuple(diffs),
    )


def _no_print_reason(receipts: List[dict]) -> str:
    """The publisher's receipt-derived no-print reason, re-derivable
    only when fully disclosed."""
    priced = [r for r in receipts if r.get("price") is not None]
    if not priced:
        return "no_eligible_sources"
    if all(r.get("filter_verdict") == "rejected" for r in priced):
        return "all_sources_filtered"
    return "insufficient_coverage"


def select_observations(
    envelope: dict,
    *,
    sku: Optional[str] = None,
    stamp: Optional[str] = None,
) -> List[dict]:
    """Observations from a verified latest/day envelope, optionally
    filtered to one SKU and/or one YYYY-MM-DDTHH stamp."""
    data = envelope.get("data") or {}
    if data.get("kind") == "gpu_index_series":
        raise PublishedRecordError(
            "series artifacts carry aggregate rows without receipts; "
            "recompute-and-match runs on latest.json or a day file"
        )
    selected = []
    for observation in data.get("observations", []):
        if sku is not None and observation.get("sku") != sku:
            continue
        if stamp is not None and not str(
            observation.get("observed_at", "")
        ).startswith(stamp):
            continue
        selected.append(observation)
    return selected
