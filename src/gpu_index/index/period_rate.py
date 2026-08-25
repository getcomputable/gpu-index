# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Period rate, gap fill, and coverage report over a published hourly
panel series (METHODOLOGY.md section 6.1).

The hourly series is the index; a PERIOD RATE is a convention applied on
top of it by a referencing contract: the time-average of the hourly
index values within the period, with any missing hour carried forward
under the fill rule below. This module is the calculation agent's
canonical implementation of that convention plus the coverage record
that accompanies it. Three postures, fixed on purpose:

  - **Nothing here is ever stored or published back into the series.**
    Published hourly values are never revised; a missing hour's
    artifact stays an explicit observation_missed / panel_dark /
    record_quarantined publish. Fill values are DERIVED on demand from
    the published record and carry per-stamp provenance
    (observed | filled | dropped_genesis), so a fill is always
    distinguishable from a value computed from raw snapshots.
  - **Pure functions, zero I/O, zero clock reads** -- same posture as
    gpu_index.index.panel_schedule / gpu_index.index.weights. The CLI
    (scripts/compute_period_rate.py) owns bucket reads and now().
  - **The fill never feeds the weighting.** Scoring requires real
    observations at both ends of a return and drops any sample it
    cannot form; this module exists only because a contract demands a
    figure where scoring may return undefined.

Fill rule (methodology section 6.1). Every stamp in a gap of G
consecutive missing scheduled stamps takes the mean of the last
min(G, L) FILLED stamps immediately preceding the gap, where
L = FILL_LOOKBACK_HOURS = 72. "Filled" means carrying a published index
value -- an earlier gap's derived fill is NOT a filled stamp and never
enters a later gap's window (fills are not evidence). Because a gap
contains no filled stamps, every stamp of one gap takes the same value.
Where fewer than min(G, L) filled stamps exist the window uses what
exists; where NONE exists (a panel's genesis) the gap's stamps are
DROPPED from the average instead -- the only case that shrinks the
denominator.

**A gap is a property of the SERIES, not of the querying period**
(section 6.1 defines it as "a run of consecutive missing hours" and
contemplates one spanning a whole period). A missing run that straddles
a period boundary therefore keeps its FULL length G on both sides: the
run extends backward through pre-period missing stamps and forward past
the period's end to wherever the record resumes (bounded by
`frontier_stamp`, the exclusive extent of the known record). One
physical outage fills every one of its hours at ONE value, whichever
contract period asks. The BAND input `longest_gap` deliberately stays
the gap's IN-PERIOD extent -- the coverage bands measure how much of
THIS period is carried rather than observed, while the fill measures
the physical outage; each gap row publishes both (`length` in-period,
`run_length` physical, with `extends_before`/`extends_after`, and
`open_at_frontier` when the record has not yet resumed).

Units are SCHEDULED STAMPS, which on the hourly grid are exactly the
methodology's hours. On the migrated B300/B200 lanes' pre-cutover
4-slot era (4 observations/day through 2026-08-23) a stamp is a 6-hour
slot and the stamp-counted rule applies unchanged; no contract period
can reference that era (the period machinery postdates the hourly
cutover), so this is a display/investigation semantic only, recorded
here rather than silently redefined.

Coverage bands (methodology section 6.1, RECOMMENDED CONTRACT DEFAULTS
-- not index parameters; only FILL_LOOKBACK_HOURS is an index
parameter and changing it is a versioned change):

    settles        coverage >= 98%  and  longest gap <= 2% of period
    review         90% <= coverage < 98%  (gap still <= 2%)
    determination  coverage < 90%  or  longest gap > 2% of period

Band verdicts compare in EXACT INTEGER ARITHMETIC (filled * 100 vs
scheduled * 98, never float ratios): a period sitting exactly on a
threshold must land in the same band on every replay of every machine.

The coverage report publishes every period, passing or failing --
scheduled and filled counts, coverage, and every gap with its stamps
and per-stamp cause (missed / dark / quarantined / unpublished).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gpu_index.index.panel_schedule import PanelSchedule, stamp_to_hour_iso

__all__ = [
    "FILL_LOOKBACK_HOURS",
    "REVIEW_COVERAGE_PCT",
    "DETERMINATION_COVERAGE_PCT",
    "DETERMINATION_GAP_SHARE_PCT",
    "BAND_SETTLES",
    "BAND_REVIEW",
    "BAND_DETERMINATION",
    "CAUSE_MISSED",
    "CAUSE_DARK",
    "CAUSE_QUARANTINED",
    "CAUSE_UNPUBLISHED",
    "PeriodRateError",
    "classify_artifact",
    "coverage_band",
    "find_gaps",
    "period_report",
]


class PeriodRateError(ValueError):
    """A malformed input -- an artifact that violates its own published
    invariants, or a window nothing is scheduled in. Loud on purpose:
    a settlement convention must refuse before it guesses."""


# The ONE index parameter of section 6.1 (Appendix C.1). Counted in
# scheduled stamps == hours on the hourly grid; changing it is a
# versioned change.
FILL_LOOKBACK_HOURS = 72

# Recommended contract defaults (section 6.1 / C.1) -- thresholds a
# referencing contract may adopt or replace; the period rate computes
# identically either way. Integer percents so band tests stay exact.
REVIEW_COVERAGE_PCT = 98
DETERMINATION_COVERAGE_PCT = 90
DETERMINATION_GAP_SHARE_PCT = 2

BAND_SETTLES = "settles"
BAND_REVIEW = "review"
BAND_DETERMINATION = "determination"

# Why a scheduled stamp carries no index value. missed / unpublished
# are collection-side holes (no bytes; unpublished = no artifact in the
# record at all -- either never published or, on a mirror read, not yet
# ingested); dark / quarantined are published artifacts whose index is
# null (claim floor / F6 record quarantine).
CAUSE_MISSED = "missed"
CAUSE_DARK = "dark"
CAUSE_QUARANTINED = "quarantined"
CAUSE_UNPUBLISHED = "unpublished"

_CAUSES = (CAUSE_MISSED, CAUSE_DARK, CAUSE_QUARANTINED, CAUSE_UNPUBLISHED)

# Provenance of one stamp's contribution to the period average.
SOURCE_OBSERVED = "observed"
SOURCE_FILLED = "filled"
SOURCE_DROPPED_GENESIS = "dropped_genesis"


def classify_artifact(
    artifact: Optional[Mapping[str, Any]],
) -> Tuple[Optional[float], Optional[str]]:
    """One published panel artifact -> (index value, missing-cause).

    Exactly one of the pair is non-None. None artifact (no object in
    the record) -> (None, "unpublished"). Fail-closed on shape: an
    artifact claiming a value that is not a finite number, claiming
    observation_missed while carrying an index, or contradicting its
    own panel_dark <-> index-null invariant, is a record we do not
    understand -- refuse, never coerce (the isfinite doctrine: NaN or
    Infinity must not reach settlement arithmetic).
    """
    if artifact is None:
        return None, CAUSE_UNPUBLISHED
    index = artifact.get("index")
    missed = bool(artifact.get("observation_missed"))
    quarantined = artifact.get("record_quarantined")
    if bool(artifact.get("panel_dark")) != (index is None):
        raise PeriodRateError(
            f"artifact {artifact.get('date')!r} violates the panel_dark "
            f"<-> index-null invariant -- refusing a record we do not "
            f"understand"
        )
    if index is None:
        if quarantined is not None:
            return None, CAUSE_QUARANTINED
        if missed:
            return None, CAUSE_MISSED
        return None, CAUSE_DARK
    if missed or quarantined is not None:
        raise PeriodRateError(
            f"artifact {artifact.get('date')!r} carries an index while "
            f"claiming missed/quarantined -- refusing a record that "
            f"violates its own invariants"
        )
    value = index.get("value_usd_gpu_hr") if isinstance(index, Mapping) else None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise PeriodRateError(
            f"artifact {artifact.get('date')!r} index.value_usd_gpu_hr must "
            f"be a finite number, got {value!r}"
        )
    return float(value), None


def find_gaps(
    scheduled: Sequence[int],
    statuses: Mapping[int, Tuple[Optional[float], Optional[str]]],
) -> List[Dict[str, Any]]:
    """Runs of consecutive missing scheduled stamps, in stamp order.

    Each gap: {"start_stamp", "end_stamp" (inclusive), "length",
    "causes": {cause: count}}. A stamp absent from `statuses` is
    unpublished (see classify_artifact).
    """
    gaps: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for stamp in scheduled:
        value, cause = statuses.get(stamp, (None, CAUSE_UNPUBLISHED))
        if value is not None:
            current = None
            continue
        if cause not in _CAUSES:
            raise PeriodRateError(
                f"stamp {stamp_to_hour_iso(stamp)} carries no value and an "
                f"unknown cause {cause!r}"
            )
        if current is None:
            current = {
                "start_stamp": stamp,
                "end_stamp": stamp,
                "length": 0,
                "causes": {},
            }
            gaps.append(current)
        current["end_stamp"] = stamp
        current["length"] += 1
        current["causes"][cause] = current["causes"].get(cause, 0) + 1
    return gaps


def coverage_band(
    *, scheduled: int, filled: int, longest_gap: int
) -> str:
    """Band verdict in exact integer arithmetic (module docstring).

    scheduled == 0 is the caller's refusal case, not a band.
    """
    if scheduled <= 0:
        raise PeriodRateError("coverage band undefined over zero scheduled stamps")
    if (
        filled * 100 < scheduled * DETERMINATION_COVERAGE_PCT
        or longest_gap * 100 > scheduled * DETERMINATION_GAP_SHARE_PCT
    ):
        return BAND_DETERMINATION
    if filled * 100 < scheduled * REVIEW_COVERAGE_PCT:
        return BAND_REVIEW
    return BAND_SETTLES


def _fill_window(
    preceding_filled: Sequence[Tuple[int, float]], gap_length: int
) -> Optional[Dict[str, Any]]:
    """The fill for one gap: mean of the last min(G, L) filled stamps
    immediately preceding it. `preceding_filled` is every filled
    (stamp, value) strictly before the gap, ascending. None when no
    filled stamp precedes (genesis drop)."""
    if not preceding_filled:
        return None
    want = min(gap_length, FILL_LOOKBACK_HOURS)
    window = preceding_filled[-want:]
    mean = sum(v for _, v in window) / len(window)
    return {
        "value": mean,
        "window_stamps": len(window),
        "window_start": window[0][0],
        "window_end": window[-1][0],
    }


def period_report(
    *,
    schedule: PanelSchedule,
    start_stamp: int,
    end_stamp: int,
    statuses: Mapping[int, Tuple[Optional[float], Optional[str]]],
    methodology_id: str,
    panel_id: str,
    include_series: bool = False,
    clipped_at_stamp: Optional[int] = None,
    frontier_stamp: Optional[int] = None,
) -> Dict[str, Any]:
    """The coverage report + period rate for [start_stamp, end_stamp).

    `statuses` must cover (at least) every published stamp from far
    enough before start_stamp that fill windows resolve -- the CLI
    walks back until FILL_LOOKBACK_HOURS filled stamps precede the
    period or genesis is hit -- plus, when `frontier_stamp` reaches
    beyond the period, every published stamp in [end_stamp,
    frontier_stamp) so a tail gap can find where the record resumes.
    Half-open on purpose: two adjacent periods share no stamp.
    `clipped_at_stamp` records that the caller truncated a
    still-running period at the record frontier; `frontier_stamp`
    (exclusive; defaults to end_stamp) is the extent of the KNOWN
    record and bounds the forward run extension.
    """
    if end_stamp <= start_stamp:
        raise PeriodRateError(
            f"period end {stamp_to_hour_iso(end_stamp)} must be after start "
            f"{stamp_to_hour_iso(start_stamp)}"
        )
    frontier = end_stamp if frontier_stamp is None else int(frontier_stamp)
    if frontier < end_stamp:
        raise PeriodRateError(
            "frontier_stamp must not precede the period end -- the period "
            "must lie inside the known record"
        )
    scheduled = schedule.scheduled_stamps(start_stamp, end_stamp)
    if not scheduled:
        raise PeriodRateError(
            f"no scheduled stamps in [{stamp_to_hour_iso(start_stamp)}, "
            f"{stamp_to_hour_iso(end_stamp)}) -- period precedes the lane's "
            f"genesis or is narrower than its grid"
        )

    def is_filled(stamp: int) -> bool:
        return statuses.get(stamp, (None, None))[0] is not None

    # Context: every filled SCHEDULED stamp strictly before the period,
    # ascending. Off-grid keys are never evidence (a stamp the schedule
    # does not own cannot enter a fill window), and only the trailing
    # FILL_LOOKBACK_HOURS can ever enter one.
    context = [
        (stamp, statuses[stamp][0])
        for stamp in sorted(statuses)
        if stamp < start_stamp
        and statuses[stamp][0] is not None
        and schedule.is_scheduled(stamp)
    ][-FILL_LOOKBACK_HOURS:]

    gaps = find_gaps(scheduled, statuses)

    # Series-level run extension (module docstring): a gap touching the
    # period's first scheduled stamp continues backward through every
    # missing scheduled stamp before the period; one touching the last
    # continues forward until the record resumes or the frontier.
    for gap in gaps:
        gap["extends_before"] = 0
        gap["extends_after"] = 0
        gap["open_at_frontier"] = False
    if gaps and gaps[0]["start_stamp"] == scheduled[0]:
        run_before = 0
        cursor = schedule.prev_scheduled_stamp(start_stamp)
        newest_context = context[-1][0] if context else None
        while cursor is not None and not is_filled(cursor):
            if newest_context is not None and cursor <= newest_context:
                # Structural guard: the newest filled stamp bounds the
                # walk; nothing filled can sit inside the run.
                break
            run_before += 1
            cursor = schedule.prev_scheduled_stamp(cursor)
        gaps[0]["extends_before"] = run_before
    if gaps and gaps[-1]["end_stamp"] == scheduled[-1] and frontier > end_stamp:
        run_after = 0
        resumed = False
        for stamp in schedule.scheduled_stamps(end_stamp, frontier):
            if is_filled(stamp):
                resumed = True
                break
            run_after += 1
        gaps[-1]["extends_after"] = run_after
        gaps[-1]["open_at_frontier"] = not resumed
    elif gaps and gaps[-1]["end_stamp"] == scheduled[-1]:
        # No record is known beyond the period: the tail run is open by
        # definition unless the period end IS the frontier and nothing
        # follows -- record it as open so a settling reader sees it.
        gaps[-1]["open_at_frontier"] = True

    fills_by_start: Dict[int, Optional[Dict[str, Any]]] = {}
    for gap in gaps:
        preceding = [
            (stamp, statuses[stamp][0])
            for stamp in scheduled
            if stamp < gap["start_stamp"] and is_filled(stamp)
        ]
        run_length = gap["length"] + gap["extends_before"] + gap["extends_after"]
        gap["run_length"] = run_length
        fills_by_start[gap["start_stamp"]] = _fill_window(
            context + preceding, run_length
        )

    filled_count = 0
    cause_totals = {cause: 0 for cause in _CAUSES}
    series: List[Dict[str, Any]] = []
    entering: List[float] = []
    carried = 0
    dropped = 0
    observed_sum = 0.0
    gap_iter = iter(gaps)
    current_gap = next(gap_iter, None)
    for stamp in scheduled:
        value, cause = statuses.get(stamp, (None, CAUSE_UNPUBLISHED))
        if current_gap is not None and stamp > current_gap["end_stamp"]:
            current_gap = next(gap_iter, None)
        if value is not None:
            filled_count += 1
            observed_sum += value
            entering.append(value)
            entry: Dict[str, Any] = {
                "stamp": stamp_to_hour_iso(stamp),
                "source": SOURCE_OBSERVED,
                "value_usd_gpu_hr": round(value, 6),
            }
        else:
            cause_totals[cause] += 1
            fill = fills_by_start[current_gap["start_stamp"]]
            if fill is None:
                dropped += 1
                entry = {
                    "stamp": stamp_to_hour_iso(stamp),
                    "source": SOURCE_DROPPED_GENESIS,
                    "value_usd_gpu_hr": None,
                    "cause": cause,
                }
            else:
                carried += 1
                entering.append(fill["value"])
                entry = {
                    "stamp": stamp_to_hour_iso(stamp),
                    "source": SOURCE_FILLED,
                    "value_usd_gpu_hr": round(fill["value"], 6),
                    "cause": cause,
                }
        if include_series:
            series.append(entry)

    scheduled_count = len(scheduled)
    longest_gap = max((gap["length"] for gap in gaps), default=0)
    rate = (sum(entering) / len(entering)) if entering else None
    filled_only_mean = (observed_sum / filled_count) if filled_count else None

    report: Dict[str, Any] = {
        "kind": "index_period_report",
        "schema_version": 1,
        "methodology_id": methodology_id,
        "panel_id": panel_id,
        "period": {
            "start": stamp_to_hour_iso(start_stamp),
            "end": stamp_to_hour_iso(end_stamp),
            "half_open": True,
            "clipped_at": (
                stamp_to_hour_iso(clipped_at_stamp)
                if clipped_at_stamp is not None
                else None
            ),
        },
        "params": {
            # The index parameter (versioned change to alter).
            "fill_lookback_hours": FILL_LOOKBACK_HOURS,
            # Recommended contract defaults -- NOT index parameters; a
            # referencing contract may draw its own lines and the
            # period rate computes identically.
            "recommended_bands": {
                "review_coverage_pct": REVIEW_COVERAGE_PCT,
                "determination_coverage_pct": DETERMINATION_COVERAGE_PCT,
                "determination_gap_share_pct": DETERMINATION_GAP_SHARE_PCT,
            },
        },
        "coverage": {
            "scheduled": scheduled_count,
            "filled": filled_count,
            "coverage_ratio": round(filled_count / scheduled_count, 6),
            "longest_gap": longest_gap,
            "longest_gap_share": round(longest_gap / scheduled_count, 6),
            "causes": cause_totals,
            "gaps": [
                {
                    "start": stamp_to_hour_iso(gap["start_stamp"]),
                    "end": stamp_to_hour_iso(gap["end_stamp"]),
                    "length": gap["length"],
                    # The physical outage (module docstring): in-period
                    # length plus its extensions across the period
                    # boundaries. THIS sizes the fill window; `length`
                    # feeds the band.
                    "run_length": gap["run_length"],
                    "extends_before": gap["extends_before"],
                    "extends_after": gap["extends_after"],
                    "open_at_frontier": gap["open_at_frontier"],
                    "causes": dict(sorted(gap["causes"].items())),
                    "fill": (
                        None
                        if fills_by_start[gap["start_stamp"]] is None
                        else {
                            "value_usd_gpu_hr": round(
                                fills_by_start[gap["start_stamp"]]["value"], 6
                            ),
                            "window_stamps": fills_by_start[gap["start_stamp"]][
                                "window_stamps"
                            ],
                            "window_start": stamp_to_hour_iso(
                                fills_by_start[gap["start_stamp"]]["window_start"]
                            ),
                            "window_end": stamp_to_hour_iso(
                                fills_by_start[gap["start_stamp"]]["window_end"]
                            ),
                        }
                    ),
                }
                for gap in gaps
            ],
        },
        "band": coverage_band(
            scheduled=scheduled_count,
            filled=filled_count,
            longest_gap=longest_gap,
        ),
        "period_rate": {
            "value_usd_gpu_hr": round(rate, 6) if rate is not None else None,
            "stamps_observed": filled_count,
            "stamps_carried": carried,
            "stamps_dropped_genesis": dropped,
            "carried_share": (
                round(carried / len(entering), 6) if entering else None
            ),
        },
        "diagnostics": {
            # What "average only the filled hours" would have printed --
            # the alternative section 6.1 rejects; published so the
            # difference is inspectable, never the rate.
            "filled_only_mean_usd_gpu_hr": (
                round(filled_only_mean, 6) if filled_only_mean is not None else None
            ),
        },
    }
    if include_series:
        report["series"] = series
    return report
