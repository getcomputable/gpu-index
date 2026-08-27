# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for gpu_index.index.period_rate (METHODOLOGY.md sections
9.3-9.4 -- fill rule, coverage record, band verdicts, period rate).

The heart is the SHARED vector file tests/fixtures/period_rate_vectors.json:
hand-computed expectations for every rule in sections 9.3-9.4 (neighbour fill,
window-scales-with-gap, the 72-stamp cap, filled-hours-only windows that
skip earlier holes and never treat a derived fill as evidence, the
genesis drop, per-stamp cause attribution, exact-threshold band edges in
integer arithmetic, the era-boundary window, and a whole-period gap
resolving from before the period). A private downstream consumer keeps
its own copy of the same vectors, so the two implementations can only
drift by failing one of these suites -- change the file in both or
neither.

Beyond the vectors: classify_artifact's fail-closed shape rules (an
artifact violating its own invariants refuses, NaN/Infinity never reach
settlement arithmetic), report refusals, provenance bookkeeping in the
embedded series, and byte determinism of the JSON report.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from gpu_index.index.panel_schedule import PanelSchedule, hour_iso_to_stamp
from gpu_index.index.period_rate import (
    CAUSE_DARK,
    CAUSE_MISSED,
    CAUSE_QUARANTINED,
    CAUSE_UNPUBLISHED,
    FILL_LOOKBACK_HOURS,
    PeriodRateError,
    SOURCE_DROPPED_GENESIS,
    SOURCE_FILLED,
    SOURCE_OBSERVED,
    classify_artifact,
    coverage_band,
    find_gaps,
    period_report,
)

# Cross-copy tripwire: a private downstream consumer pins the same digest
# over its own copy, so either side editing its file alone fails its own
# suite and forces the paired update. Changing the vectors means updating
# the digest on both sides.
VECTORS_SHA256 = "cb3ba79aaeed6abfd133242b202b6cbbd1fdd0b79727ac691465bd8f2142f29d"

_VECTORS_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "period_rate_vectors.json"
)
VECTORS = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def test_vector_bytes_pinned_cross_repo():
    digest = hashlib.sha256(_VECTORS_PATH.read_bytes()).hexdigest()
    assert digest == VECTORS_SHA256


def _case(name):
    for case in VECTORS["cases"]:
        if case["name"] == name:
            return case
    raise AssertionError(f"vector case {name!r} missing")


def _build(case):
    schedule = PanelSchedule(**case["schedule"])
    statuses = {}
    for stamp_iso, value, cause in case["statuses"]:
        statuses[hour_iso_to_stamp(stamp_iso)] = (
            (float(value), None) if value is not None else (None, cause)
        )
    return schedule, statuses


def _report(case, **kwargs):
    schedule, statuses = _build(case)
    if "frontier" in case and "frontier_stamp" not in kwargs:
        kwargs["frontier_stamp"] = hour_iso_to_stamp(case["frontier"])
    return period_report(
        schedule=schedule,
        start_stamp=hour_iso_to_stamp(case["period"]["start"]),
        end_stamp=hour_iso_to_stamp(case["period"]["end"]),
        statuses=statuses,
        methodology_id="test_lane_v1",
        panel_id="TEST",
        **kwargs,
    )


def test_vector_constant_matches_module():
    assert VECTORS["fill_lookback_hours"] == FILL_LOOKBACK_HOURS


@pytest.mark.parametrize(
    "name", [case["name"] for case in VECTORS["cases"]]
)
def test_vectors(name):
    case = _case(name)
    report = _report(case, include_series=True)
    expect = case["expect"]

    assert report["band"] == expect["band"]
    for key, want in expect["coverage"].items():
        assert report["coverage"][key] == want, (name, key)
    for key, want in expect["period_rate"].items():
        assert report["period_rate"][key] == want, (name, key)

    if "gap_fills" in expect:
        got = [
            {
                "start": gap["start"],
                "value_usd_gpu_hr": (gap["fill"] or {}).get("value_usd_gpu_hr"),
                "window_stamps": (gap["fill"] or {}).get("window_stamps"),
                "window_start": (gap["fill"] or {}).get("window_start"),
                "window_end": (gap["fill"] or {}).get("window_end"),
            }
            for gap in report["coverage"]["gaps"]
        ]
        assert got == expect["gap_fills"], name
    if "gap_causes" in expect:
        by_start = {gap["start"]: gap["causes"] for gap in report["coverage"]["gaps"]}
        for want in expect["gap_causes"]:
            assert by_start[want["start"]] == want["causes"], name
    if "gap_runs" in expect:
        runs = {
            gap["start"]: {
                "start": gap["start"],
                "run_length": gap["run_length"],
                "extends_before": gap["extends_before"],
                "extends_after": gap["extends_after"],
                "open_at_frontier": gap["open_at_frontier"],
            }
            for gap in report["coverage"]["gaps"]
        }
        for want in expect["gap_runs"]:
            assert runs[want["start"]] == want, name
    if "filled_only_mean_usd_gpu_hr" in expect:
        assert (
            report["diagnostics"]["filled_only_mean_usd_gpu_hr"]
            == expect["filled_only_mean_usd_gpu_hr"]
        ), name

    # Series provenance bookkeeping must reconcile with the headline
    # counts -- a fill must NEVER present as an observed value.
    series = report["series"]
    counts = {SOURCE_OBSERVED: 0, SOURCE_FILLED: 0, SOURCE_DROPPED_GENESIS: 0}
    for entry in series:
        counts[entry["source"]] += 1
        if entry["source"] == SOURCE_OBSERVED:
            assert "cause" not in entry
        else:
            assert entry["cause"] in (
                CAUSE_MISSED, CAUSE_DARK, CAUSE_QUARANTINED, CAUSE_UNPUBLISHED
            )
        if entry["source"] == SOURCE_DROPPED_GENESIS:
            assert entry["value_usd_gpu_hr"] is None
    assert counts[SOURCE_OBSERVED] == report["period_rate"]["stamps_observed"]
    assert counts[SOURCE_FILLED] == report["period_rate"]["stamps_carried"]
    assert (
        counts[SOURCE_DROPPED_GENESIS]
        == report["period_rate"]["stamps_dropped_genesis"]
    )
    assert len(series) == report["coverage"]["scheduled"]


# ------------------------------------------------------- classify_artifact


def test_classify_unpublished():
    assert classify_artifact(None) == (None, CAUSE_UNPUBLISHED)


def test_classify_missed_dark_quarantined_value():
    assert classify_artifact(
        {
            "index": None,
            "observation_missed": True,
            "panel_dark": True,
            "record_quarantined": None,
        }
    ) == (None, CAUSE_MISSED)
    assert classify_artifact(
        {
            "index": None,
            "observation_missed": False,
            "panel_dark": True,
            "record_quarantined": None,
        }
    ) == (None, CAUSE_DARK)
    # Quarantine wins the label even though missed is False by invariant.
    assert classify_artifact(
        {
            "index": None,
            "observation_missed": False,
            "panel_dark": True,
            "record_quarantined": "F6",
        }
    ) == (None, CAUSE_QUARANTINED)
    value, cause = classify_artifact(
        {
            "index": {"value_usd_gpu_hr": 7.62},
            "observation_missed": False,
            "panel_dark": False,
            "record_quarantined": None,
        }
    )
    assert (value, cause) == (7.62, None)


def test_classify_refuses_index_on_missed():
    with pytest.raises(PeriodRateError, match="violates its own invariants"):
        classify_artifact(
            {
                "date": "2026-08-24T00",
                "index": {"value_usd_gpu_hr": 5.0},
                "observation_missed": True,
                "panel_dark": False,
                "record_quarantined": None,
            }
        )


@pytest.mark.parametrize(
    "artifact",
    [
        # panel_dark true while carrying an index value.
        {
            "date": "2026-08-24T00",
            "index": {"value_usd_gpu_hr": 5.0},
            "observation_missed": False,
            "panel_dark": True,
            "record_quarantined": None,
        },
        # index null while claiming not-dark (also catches a foreign
        # artifact missing the panel_dark field entirely).
        {
            "date": "2026-08-24T00",
            "index": None,
            "observation_missed": False,
            "record_quarantined": None,
        },
    ],
)
def test_classify_refuses_dark_index_contradiction(artifact):
    with pytest.raises(PeriodRateError, match="panel_dark"):
        classify_artifact(artifact)


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -float("inf"), "7.5", True, None]
)
def test_classify_refuses_non_finite_or_non_numeric(bad):
    with pytest.raises(PeriodRateError, match="finite number"):
        classify_artifact(
            {
                "date": "2026-08-24T00",
                "index": {"value_usd_gpu_hr": bad},
                "observation_missed": False,
                "panel_dark": False,
                "record_quarantined": None,
            }
        )


# ---------------------------------------------------------------- refusals


def test_band_refuses_zero_scheduled():
    with pytest.raises(PeriodRateError, match="zero scheduled"):
        coverage_band(scheduled=0, filled=0, longest_gap=0)


def test_report_refuses_empty_and_inverted_periods():
    case = _case("full_coverage_settles")
    schedule, statuses = _build(case)
    start = hour_iso_to_stamp("2026-08-23T06")
    with pytest.raises(PeriodRateError, match="must be after start"):
        period_report(
            schedule=schedule,
            start_stamp=start,
            end_stamp=start,
            statuses=statuses,
            methodology_id="m",
            panel_id="p",
        )
    # A period entirely before genesis has nothing scheduled in it.
    with pytest.raises(PeriodRateError, match="no scheduled stamps"):
        period_report(
            schedule=schedule,
            start_stamp=hour_iso_to_stamp("2026-08-20T00"),
            end_stamp=hour_iso_to_stamp("2026-08-21T00"),
            statuses={},
            methodology_id="m",
            panel_id="p",
        )


def test_find_gaps_refuses_unknown_cause():
    schedule, _ = _build(_case("full_coverage_settles"))
    scheduled = schedule.scheduled_stamps(
        hour_iso_to_stamp("2026-08-23T00"), hour_iso_to_stamp("2026-08-23T02")
    )
    with pytest.raises(PeriodRateError, match="unknown cause"):
        find_gaps(scheduled, {scheduled[0]: (None, "weird")})


# ------------------------------------------------------------- report shape


def test_params_echo_and_carried_share():
    report = _report(_case("cause_mix_in_one_gap"))
    assert report["params"]["fill_lookback_hours"] == FILL_LOOKBACK_HOURS
    assert report["params"]["recommended_bands"] == {
        "review_coverage_pct": 98,
        "determination_coverage_pct": 90,
        "determination_gap_share_pct": 2,
    }
    # 2 observed + 4 carried -> carried share 4/6.
    assert report["period_rate"]["carried_share"] == round(4 / 6, 6)
    assert report["period"]["clipped_at"] is None
    assert "series" not in report  # only with include_series


def test_clipped_at_passthrough():
    report = _report(
        _case("full_coverage_settles"),
        clipped_at_stamp=hour_iso_to_stamp("2026-08-23T06"),
    )
    assert report["period"]["clipped_at"] == "2026-08-23T06"


def test_report_is_byte_deterministic_and_input_pure():
    case = _case("window_skips_holes_fills_not_evidence")
    schedule, statuses = _build(case)
    before = dict(statuses)
    kwargs = dict(
        schedule=schedule,
        start_stamp=hour_iso_to_stamp(case["period"]["start"]),
        end_stamp=hour_iso_to_stamp(case["period"]["end"]),
        statuses=statuses,
        methodology_id="test_lane_v1",
        panel_id="TEST",
        include_series=True,
    )
    first = json.dumps(period_report(**kwargs), sort_keys=True)
    second = json.dumps(period_report(**kwargs), sort_keys=True)
    assert first == second
    assert statuses == before


def test_all_report_numbers_finite():
    for case in VECTORS["cases"]:
        report = _report(case)
        rate = report["period_rate"]["value_usd_gpu_hr"]
        if rate is not None:
            assert math.isfinite(rate)


def test_off_grid_context_keys_are_never_evidence():
    """A statuses key the schedule does not own (here hour 05 on a
    4-slot era) must never enter a fill window (adversarial review)."""
    schedule = PanelSchedule(
        genesis_date="2026-08-20",
        slot_grids=[{"from_date": "2026-08-20", "slot_hours_utc": [4, 10, 16, 22]}],
    )
    statuses = {
        hour_iso_to_stamp("2026-08-21T22"): (2.0, None),
        # Off-grid: hour 05 is not a scheduled slot of this era.
        hour_iso_to_stamp("2026-08-22T05"): (99.0, None),
        hour_iso_to_stamp("2026-08-22T10"): (4.0, None),
        hour_iso_to_stamp("2026-08-22T16"): (None, "missed"),
        hour_iso_to_stamp("2026-08-22T22"): (6.0, None),
    }
    report = period_report(
        schedule=schedule,
        start_stamp=hour_iso_to_stamp("2026-08-22T10"),
        end_stamp=hour_iso_to_stamp("2026-08-23T00"),
        statuses=statuses,
        methodology_id="m",
        panel_id="p",
    )
    gap = report["coverage"]["gaps"][0]
    # G=1 -> the single nearest preceding SCHEDULED filled stamp (T10),
    # never the off-grid 99.0.
    assert gap["fill"]["value_usd_gpu_hr"] == 4.0
    assert gap["fill"]["window_start"] == "2026-08-22T10"
