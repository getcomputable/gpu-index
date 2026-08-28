# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for the observation-mode (hourly panel) weight engine.

Pins, per the panel weighting rules (A1/A2):

  - PanelSchedule math: era-scoped grids, the 4-slot -> hourly era
    boundary, genesis clipping, prev-scheduled cutoffs, stamp
    conversions, and the fail-closed construction fences;
  - A2 attendance: per-era scheduled denominators across the boundary,
    the genesis zero-scheduled corner (ratio 1.0 -- nothing scheduled
    means nothing missed, and the switch must be HELD at genesis);
  - R-cutoff per observation: cutoff = the previous SCHEDULED stamp,
    era-aware, with nothing past the cutoff entering any score;
  - the A2 transition rule: an attendance-passer without a defined Q
    holds the switch, a sparse below-floor source does not, a thin
    eligible set never switches, and the latch is permanent with
    switched_on pinned as the full YYYY-MM-DDTHH stamp;
  - the perf path (features once per tau, all-N LOO exclusions from one
    full-sum-minus-own pass per endpoint pair) is BIT-IDENTICAL to a
    naive recompute-everything reimplementation;
  - pruning at advance provably changes no computable score, keeps the
    newest below-threshold vector as the resolution anchor, and the
    replay path (advance-only from pinned facts) rebuilds the exact
    live state and identical subsequent vectors.

Day-mode functions are untouched by construction -- the frozen daily
series' own suite (test_dynamic_weights.py) runs alongside this one.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from gpu_index.index.panel_schedule import (
    PanelSchedule,
    PanelScheduleError,
    date_hour_to_stamp,
    hour_iso_to_stamp,
    stamp_to_date_hour,
    stamp_to_hour_iso,
)
from gpu_index.index.weights import (
    PRUNE_MARGIN_MINUTES,
    advance_panel_weight_state,
    attendance,
    compute_panel_weights,
    in_sample_q,
    new_weight_state,
    predictive_scores_obs,
    series_print,
    source_return,
    validate_attendance_floor,
)

CORE = ("lead", "echo", "s1", "s2", "s3")

_SIGNS = [
    1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1,
    -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1,
]


def _boundary_schedule():
    """The migrated-lane shape: 4-slot basket era, then hourly."""
    return PanelSchedule(
        genesis_date="2026-08-10",
        slot_grids=[
            {"from_date": "2026-08-10", "slot_hours_utc": [4, 10, 16, 22]},
            {"from_date": "2026-08-24", "slot_hours_utc": list(range(24))},
        ],
    )


def _hourly_schedule(genesis="2026-08-01"):
    return PanelSchedule(
        genesis_date=genesis,
        slot_grids=[{"from_date": genesis, "slot_hours_utc": list(range(24))}],
    )


def _dw(**overrides):
    dw = {
        "scheme": "predictive_v1",
        "lookback_horizons_hours": [1, 2],
        "forward_horizons_hours": [1, 2],
        "history_days": 2,
        "half_life_days": 1,
        "ridge_lambda": 0.001,
        "gamma": 4.0,
        "weight_min": 0.025,
        "weight_max": 0.30,
        "min_train_samples": 3,
        "target_variance_floor": 1e-12,
        "switch_min_eligible": 3,
        "max_abs_log_return": 0.5,
        "source_weight_caps": {},
        "attendance_floor": 0.5,
    }
    dw.update(overrides)
    return dw


def _entry(price, currency="USD", usd=None):
    return {
        "usd": price if usd is None else usd,
        "native": price,
        "currency": currency,
    }


def _core_walk_state(schedule, obs_stamp, span_hours):
    """Prints at every scheduled stamp in [obs - span, obs): 'lead' walks
    the signs path, 'echo' replays lead one scheduled stamp late, s1-s3
    are frozen list prices -- lead's excess movement predicts the
    rest-of-basket by construction (the day-suite fixture, re-anchored
    to the schedule grid). One pinned vector at the window's first stamp
    (vector resolution is last-at-or-below tau)."""
    stamps = schedule.scheduled_stamps(obs_stamp - span_hours * 60, obs_stamp)
    state = new_weight_state()
    levels = [0.0]
    for i in range(1, len(stamps)):
        levels.append(levels[-1] + 0.02 * _SIGNS[i % len(_SIGNS)])
    for i, stamp in enumerate(stamps):
        row = {
            "lead": round(10.0 * math.exp(levels[i]), 6),
            "echo": round(10.0 * math.exp(levels[i - 1] if i else 0.0), 6),
            "s1": 10.0,
            "s2": 9.0,
            "s3": 11.0,
        }
        for sid, price in row.items():
            state["prices"].setdefault(sid, {})[stamp] = _entry(price)
    state["vectors"][stamps[0]] = {sid: 0.2 for sid in CORE}
    return state, stamps


def _add_flat_prints(state, sid, price, stamps):
    for stamp in stamps:
        state["prices"].setdefault(sid, {})[stamp] = _entry(price)


# ------------------------------------------------------------ schedule math


def test_stamp_conversions_round_trip():
    stamp = date_hour_to_stamp("2026-08-24", 5)
    assert stamp_to_date_hour(stamp) == ("2026-08-24", 5)
    assert stamp_to_hour_iso(stamp) == "2026-08-24T05"
    assert hour_iso_to_stamp("2026-08-24T05") == stamp
    with pytest.raises(PanelScheduleError):
        hour_iso_to_stamp("2026-08-24 05")


def test_prev_scheduled_stamp_is_era_aware_across_the_boundary():
    schedule = _boundary_schedule()
    first_hourly = date_hour_to_stamp("2026-08-24", 0)
    assert schedule.prev_scheduled_stamp(first_hourly) == date_hour_to_stamp(
        "2026-08-23", 22
    )
    assert schedule.prev_scheduled_stamp(first_hourly + 1) == first_hourly
    assert schedule.prev_scheduled_stamp(
        date_hour_to_stamp("2026-08-23", 22)
    ) == date_hour_to_stamp("2026-08-23", 16)
    # Genesis clip: nothing scheduled precedes the first observation.
    assert schedule.prev_scheduled_stamp(schedule.genesis_stamp) is None
    assert schedule.prev_scheduled_stamp(schedule.genesis_stamp - 100) is None
    # Within the basket era, a mid-gap stamp resolves to the prior slot.
    assert schedule.prev_scheduled_stamp(
        date_hour_to_stamp("2026-08-20", 12)
    ) == date_hour_to_stamp("2026-08-20", 10)


def test_is_scheduled_per_era():
    schedule = _boundary_schedule()
    assert schedule.is_scheduled(date_hour_to_stamp("2026-08-20", 4))
    assert not schedule.is_scheduled(date_hour_to_stamp("2026-08-20", 5))
    assert not schedule.is_scheduled(date_hour_to_stamp("2026-08-23", 23))
    assert schedule.is_scheduled(date_hour_to_stamp("2026-08-24", 23))
    # Pre-genesis is never scheduled, even at a slot hour.
    assert not schedule.is_scheduled(date_hour_to_stamp("2026-08-09", 16))


def test_scheduled_stamps_window_counts_per_era_and_clips_at_genesis():
    schedule = _boundary_schedule()
    window = schedule.scheduled_stamps(
        date_hour_to_stamp("2026-08-21", 0), date_hour_to_stamp("2026-08-25", 0)
    )
    assert len(window) == 3 * 4 + 24  # three 4-slot days + one hourly day
    assert window == sorted(window)
    assert window[0] == date_hour_to_stamp("2026-08-21", 4)
    assert window[-1] == date_hour_to_stamp("2026-08-24", 23)
    # Half-open: the window end stamp itself is excluded.
    assert date_hour_to_stamp("2026-08-25", 0) not in window
    # Genesis clip: a window reaching back before genesis holds nothing
    # pre-genesis, and a mid-day window start drops earlier slots.
    clipped = schedule.scheduled_stamps(
        schedule.genesis_stamp - 240, schedule.genesis_stamp + 1
    )
    assert clipped == [schedule.genesis_stamp]
    partial = schedule.scheduled_stamps(
        date_hour_to_stamp("2026-08-20", 12), date_hour_to_stamp("2026-08-21", 0)
    )
    assert partial == [
        date_hour_to_stamp("2026-08-20", 16),
        date_hour_to_stamp("2026-08-20", 22),
    ]


def test_iter_scheduled_is_inclusive_from_genesis():
    schedule = _boundary_schedule()
    head = list(schedule.iter_scheduled(schedule.genesis_stamp + 24 * 60))
    assert head[0] == schedule.genesis_stamp
    assert [stamp_to_hour_iso(s) for s in head] == [
        "2026-08-10T04",
        "2026-08-10T10",
        "2026-08-10T16",
        "2026-08-10T22",
        "2026-08-11T04",
    ]
    assert list(schedule.iter_scheduled(schedule.genesis_stamp)) == [
        schedule.genesis_stamp
    ]


@pytest.mark.parametrize(
    "grids",
    [
        [],  # no eras
        [{"from_date": "2026-08-10", "slot_hours_utc": [4, 10, 16]}],  # lumpy
        [{"from_date": "2026-08-10", "slot_hours_utc": []}],
        [{"from_date": "2026-08-10", "slot_hours_utc": [22, 4, 10, 16]}],
        [{"from_date": "2026-08-10", "slot_hours_utc": [4, 10, 10, 16]}],
        [{"from_date": "2026-08-10", "slot_hours_utc": [0, 24]}],
        [{"from_date": "2026-08-10", "slot_hours_utc": [True, 12]}],
        [{"from_date": "2026-08-11", "slot_hours_utc": [4, 16]}],  # not genesis
        [
            {"from_date": "2026-08-10", "slot_hours_utc": [4, 16]},
            {"from_date": "2026-08-10", "slot_hours_utc": [0, 12]},  # not increasing
        ],
        [
            {"from_date": "2026-08-10", "slot_hours_utc": [4, 16]},
            {"from_date": "2026-08-09", "slot_hours_utc": [0, 12]},
        ],
        [{"slot_hours_utc": [4, 16]}],  # from_date missing
        [{"from_date": "2026-08-10"}],  # slots missing
    ],
)
def test_schedule_construction_is_fail_closed(grids):
    with pytest.raises(PanelScheduleError):
        PanelSchedule(genesis_date="2026-08-10", slot_grids=grids)


def test_prev_scheduled_stamp_walks_into_the_prior_era():
    """Coverage stage (recipe 10): an hourly era followed by a SINGLE-SLOT
    [12] era loads, and prev_scheduled_stamp of the coarse era's FIRST
    stamp -- whose own era holds no earlier stamp at all -- walks into
    the prior era's last stamp instead of returning None or a phantom."""
    schedule = PanelSchedule(
        genesis_date="2026-08-10",
        slot_grids=[
            {"from_date": "2026-08-10", "slot_hours_utc": list(range(24))},
            {"from_date": "2026-08-20", "slot_hours_utc": [12]},
        ],
    )
    first_coarse = date_hour_to_stamp("2026-08-20", 12)
    assert schedule.is_scheduled(first_coarse)
    assert not schedule.is_scheduled(date_hour_to_stamp("2026-08-20", 11))
    # The coarse era holds nothing before 08-20T12: walk into the hourly
    # era's final stamp.
    assert schedule.prev_scheduled_stamp(first_coarse) == date_hour_to_stamp(
        "2026-08-19", 23
    )
    # Within the coarse era, day-over-day at the single slot.
    assert schedule.prev_scheduled_stamp(
        date_hour_to_stamp("2026-08-21", 12)
    ) == first_coarse
    # A window straddling the boundary counts stamps per era.
    window = schedule.scheduled_stamps(
        date_hour_to_stamp("2026-08-19", 0), date_hour_to_stamp("2026-08-22", 0)
    )
    assert len(window) == 24 + 2  # one hourly day + two single-slot days


def test_schedule_accepts_single_slot_and_bad_genesis_refused():
    single = PanelSchedule(
        genesis_date="2026-08-10",
        slot_grids=[{"from_date": "2026-08-10", "slot_hours_utc": [16]}],
    )
    assert single.genesis_stamp == date_hour_to_stamp("2026-08-10", 16)
    assert single.prev_scheduled_stamp(
        date_hour_to_stamp("2026-08-12", 16)
    ) == date_hour_to_stamp("2026-08-11", 16)
    with pytest.raises(PanelScheduleError):
        PanelSchedule(
            genesis_date="not-a-date",
            slot_grids=[{"from_date": "2026-08-10", "slot_hours_utc": [16]}],
        )


# --------------------------------------------------------------- attendance


def test_attendance_counts_per_era_scheduled_stamps_across_the_boundary():
    schedule = _boundary_schedule()
    obs = date_hour_to_stamp("2026-08-25", 0)  # window spans both eras
    prices = {"full": {}, "old": {}}
    for stamp in schedule.scheduled_stamps(obs - 96 * 60, obs):
        prices["full"][stamp] = _entry(5.0)
        if stamp < date_hour_to_stamp("2026-08-24", 0):
            prices["old"][stamp] = _entry(5.0)
    # A print at an UNSCHEDULED stamp neither helps nor hurts.
    prices["old"][date_hour_to_stamp("2026-08-23", 23)] = _entry(5.0)
    full = attendance(
        prices, "full", obs_stamp=obs, schedule=schedule, window_minutes=96 * 60
    )
    assert full == {"printed": 36, "scheduled": 36, "ratio": 1.0}
    old = attendance(
        prices, "old", obs_stamp=obs, schedule=schedule, window_minutes=96 * 60
    )
    assert old["printed"] == 12  # only the 4-slot-era stamps
    assert old["scheduled"] == 36  # 3 x 4 basket days + 24 hourly
    assert old["ratio"] == pytest.approx(12 / 36)
    ghost = attendance(
        prices, "ghost", obs_stamp=obs, schedule=schedule, window_minutes=96 * 60
    )
    assert ghost == {"printed": 0, "scheduled": 36, "ratio": 0.0}


def test_attendance_genesis_clip_and_zero_scheduled_window():
    schedule = _boundary_schedule()
    genesis = schedule.genesis_stamp
    # The genesis observation: zero scheduled stamps precede it, so
    # NOTHING was missed -- ratio 1.0 for everyone (the rule that keeps
    # the vacuous switch clause from firing at genesis).
    first = attendance(
        {}, "any", obs_stamp=genesis, schedule=schedule, window_minutes=96 * 60
    )
    assert first == {"printed": 0, "scheduled": 0, "ratio": 1.0}
    # One observation later the denominator is 1 and prints decide.
    second = date_hour_to_stamp("2026-08-10", 10)
    prices = {"there": {genesis: _entry(5.0)}}
    assert attendance(
        prices, "there", obs_stamp=second, schedule=schedule, window_minutes=96 * 60
    ) == {"printed": 1, "scheduled": 1, "ratio": 1.0}
    assert attendance(
        prices, "absent", obs_stamp=second, schedule=schedule, window_minutes=96 * 60
    ) == {"printed": 0, "scheduled": 1, "ratio": 0.0}


# ------------------------------------------------------- attendance_floor


@pytest.mark.parametrize(
    "bad",
    [0, 0.0, -0.5, 1.5, True, "0.5", float("inf"), float("nan"), None],
)
def test_attendance_floor_validator_rejects(bad):
    with pytest.raises(ValueError, match="attendance_floor"):
        validate_attendance_floor({"attendance_floor": bad})


def test_attendance_floor_validator_accepts_and_requires_the_key():
    assert validate_attendance_floor({"attendance_floor": 0.5}) == 0.5
    assert validate_attendance_floor({"attendance_floor": 1}) == 1.0
    with pytest.raises(ValueError, match="attendance_floor"):
        validate_attendance_floor({})


# ------------------------------------------------------------ cutoff rules


def test_scores_stop_exactly_at_the_previous_scheduled_stamp():
    """R-cutoff across the era boundary: for the FIRST hourly observation
    the cutoff is the last 4-slot stamp -- prints after it (the phantom
    23:00 hour, the observation itself) can never enter a score, while
    the cutoff stamp itself is a realized target endpoint."""
    schedule = _boundary_schedule()
    dw = _dw(
        lookback_horizons_hours=[6],
        forward_horizons_hours=[6],
        history_days=3,
    )
    obs = date_hour_to_stamp("2026-08-24", 0)
    cutoff = date_hour_to_stamp("2026-08-23", 22)
    state, _ = _core_walk_state(schedule, obs, span_hours=96)
    base = predictive_scores_obs(
        state, obs_stamp=obs, source_ids=list(CORE), dw_params=dw,
        schedule=schedule,
    )
    # Window [cutoff - 72h, cutoff - 6h] on the 4-slot grid: 12 anchors.
    assert base["lead"]["n_samples"]["6"] == 12
    assert base["lead"]["Q"] is not None
    plus = copy.deepcopy(state)
    for stamp in (date_hour_to_stamp("2026-08-23", 23), obs):
        for sid in CORE:
            plus["prices"][sid][stamp] = _entry(99.0)
    assert (
        predictive_scores_obs(
            plus, obs_stamp=obs, source_ids=list(CORE), dw_params=dw,
            schedule=schedule,
        )
        == base
    )
    minus = copy.deepcopy(state)
    for sid in CORE:
        del minus["prices"][sid][cutoff]
    trimmed = predictive_scores_obs(
        minus, obs_stamp=obs, source_ids=list(CORE), dw_params=dw,
        schedule=schedule,
    )
    assert trimmed["lead"]["n_samples"]["6"] == 11  # the tau = cutoff-6 target


def test_genesis_observation_scores_undefined_never_zero():
    schedule = _boundary_schedule()
    dw = _dw()
    state = new_weight_state()
    scores = predictive_scores_obs(
        state,
        obs_stamp=schedule.genesis_stamp,
        source_ids=["a", "b"],
        dw_params=dw,
        schedule=schedule,
    )
    assert scores == {
        sid: {
            "q": {"1": None, "2": None},
            "n_samples": {"1": 0, "2": 0},
            "Q": None,
        }
        for sid in ("a", "b")
    }


# --------------------------------------------------------- transition (A2)


def test_attendance_passer_without_q_holds_the_switch():
    """A2 leg (a): a regular attender mid warm-up (>= floor, no defined Q
    yet) HOLDS the switch -- the outage/warm-up protection the day-mode
    R-quorum-v2 deliberately gave up, restored bounded by the floor."""
    schedule = _hourly_schedule()
    obs = date_hour_to_stamp("2026-08-15", 12)
    dw = _dw(min_train_samples=40)
    state, _ = _core_walk_state(schedule, obs, span_hours=60)
    window = schedule.scheduled_stamps(obs - 48 * 60, obs)
    _add_flat_prints(state, "newbie", 8.0, window[-30:])  # 30/48 attendance
    block = compute_panel_weights(
        state,
        obs_stamp=obs,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    assert block["mode"] == "fallback"
    assert "switched_on" not in block
    assert "slot_prints" not in block
    newbie = block["sources"]["newbie"]
    assert newbie["attendance_printed"] == 30
    assert newbie["attendance_scheduled"] == 48
    assert newbie["attendance_ratio"] == 0.625  # a passer at floor 0.5
    assert newbie["Q"] is None  # ... without a defined Q: holds
    for sid in CORE:
        assert block["sources"][sid]["attendance_ratio"] == 1.0
        assert block["sources"][sid]["Q"] is not None
    # Fallback pins the config weights restricted to the eligible set.
    assert block["weights"] == {sid: 0.2 for sid in CORE}


def test_sparse_below_floor_source_does_not_hold_the_switch():
    """A2 leg (b): a sparse source below the attendance floor is scored
    (auditable Q: null, attendance pinned) but does NOT hold the switch."""
    schedule = _hourly_schedule()
    obs = date_hour_to_stamp("2026-08-15", 12)
    dw = _dw()
    state, _ = _core_walk_state(schedule, obs, span_hours=60)
    window = schedule.scheduled_stamps(obs - 48 * 60, obs)
    _add_flat_prints(state, "ghost", 7.0, window[::3])  # 16/48 attendance
    block = compute_panel_weights(
        state,
        obs_stamp=obs,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    assert block["mode"] == "dynamic"
    assert block["switched_on"] == stamp_to_hour_iso(obs)
    ghost = block["sources"]["ghost"]
    assert ghost["attendance_printed"] == 16
    assert ghost["attendance_scheduled"] == 48
    assert ghost["attendance_ratio"] == 0.333333333  # 9dp pin
    assert ghost["Q"] is None
    assert "ghost" not in block["weights"]  # scored, not eligible
    assert sum(block["weights"].values()) == pytest.approx(1.0, abs=1e-5)


def test_thin_eligible_set_never_switches_even_with_all_q_defined():
    """A2 leg (c): fewer than switch_min_eligible sources eligible at
    this observation holds the switch regardless of scores."""
    schedule = _hourly_schedule()
    obs = date_hour_to_stamp("2026-08-15", 12)
    dw = _dw(switch_min_eligible=3)
    state, _ = _core_walk_state(schedule, obs, span_hours=60)
    block = compute_panel_weights(
        state,
        obs_stamp=obs,
        eligible=["lead", "echo"],
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    assert all(
        block["sources"][sid]["Q"] is not None for sid in CORE
    )  # every passer scored and defined ...
    assert block["mode"] == "fallback"  # ... but eligible (2) < quorum (3)
    assert "switched_on" not in block
    assert block["weights"] == {"lead": 0.2, "echo": 0.2}


def test_switch_latch_is_permanent_and_stamps_switched_on():
    """A2 leg (d): the latch flips once, pins the full YYYY-MM-DDTHH
    stamp, and never releases -- a later thin observation stays dynamic."""
    schedule = _hourly_schedule()
    obs = date_hour_to_stamp("2026-08-15", 12)
    dw = _dw()
    state, _ = _core_walk_state(schedule, obs, span_hours=60)
    first = compute_panel_weights(
        state,
        obs_stamp=obs,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    assert first["mode"] == "dynamic"
    assert first["switched_on"] == "2026-08-15T12"
    prints = {sid: series_print(10.0, (10.0, "USD")) for sid in CORE}
    advance_panel_weight_state(
        state,
        obs_stamp=obs,
        prints=prints,
        vector=first["weights"],
        mode=first["mode"],
        dw_params=dw,
    )
    assert state["mode"] == "dynamic"
    second = compute_panel_weights(
        state,
        obs_stamp=obs + 60,
        eligible=["lead"],  # below quorum forever after: irrelevant
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    assert second["mode"] == "dynamic"
    assert "switched_on" not in second


def test_zero_attendance_passers_hold_the_switch():
    """A2 amendment (adversarial review F2, docs/adversarial-reviews.md):
    with EVERY candidate below
    the attendance floor the every-passer-has-Q clause is vacuously true
    -- the amended quorum requires a NON-EMPTY passer set, so a
    post-outage observation (eligible >= quorum, zero passers) must stay
    fallback instead of flipping the permanent latch on zero
    information."""
    schedule = _hourly_schedule()
    obs = date_hour_to_stamp("2026-08-15", 12)
    dw = _dw()
    window = schedule.scheduled_stamps(obs - 48 * 60, obs)
    state = new_weight_state()
    for sid in CORE:
        _add_flat_prints(state, sid, 9.0, window[::3])  # 16/48 < floor 0.5
    block = compute_panel_weights(
        state,
        obs_stamp=obs,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    assert len(CORE) >= dw["switch_min_eligible"]  # quorum leg satisfied
    for sid in CORE:
        assert block["sources"][sid]["attendance_ratio"] < 0.5  # no passers
    assert block["mode"] == "fallback"
    assert "switched_on" not in block
    assert block["weights"] == {sid: 0.2 for sid in CORE}


def test_attendance_floor_boundary_at_exactly_the_floor():
    """The boundary pin (amended design section 5): ratio exactly ==
    floor IS a passer -- 24 of 48 scheduled stamps passes at floor 0.5
    (and, mid warm-up with no defined Q, HOLDS the switch), while 23 of
    48 sits below the floor and does not."""
    schedule = _hourly_schedule()
    obs = date_hour_to_stamp("2026-08-15", 12)
    dw = _dw(min_train_samples=40)  # CORE define; a 24-print seat cannot
    window = schedule.scheduled_stamps(obs - 48 * 60, obs)
    assert len(window) == 48

    at_floor, _ = _core_walk_state(schedule, obs, span_hours=60)
    _add_flat_prints(at_floor, "edge", 8.0, window[:24])  # exactly 24/48
    block = compute_panel_weights(
        at_floor,
        obs_stamp=obs,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    edge = block["sources"]["edge"]
    assert edge["attendance_ratio"] == 0.5
    assert edge["Q"] is None
    for sid in CORE:
        assert block["sources"][sid]["Q"] is not None
    assert block["mode"] == "fallback"  # the boundary passer holds it

    below, _ = _core_walk_state(schedule, obs, span_hours=60)
    _add_flat_prints(below, "edge", 8.0, window[:23])  # 23/48: not a passer
    block = compute_panel_weights(
        below,
        obs_stamp=obs,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in CORE},
        schedule=schedule,
    )
    edge = block["sources"]["edge"]
    assert edge["attendance_ratio"] == round(23 / 48, 9)
    assert edge["Q"] is None
    assert block["mode"] == "dynamic"  # below-floor seats never hold it
    assert block["switched_on"] == stamp_to_hour_iso(obs)


def test_genesis_observation_cannot_fire_the_switch():
    """The genesis corner end-to-end: zero scheduled history makes every
    candidate a passer with ratio 1.0 and Q undefined, so the vacuous
    every-passer-defined clause cannot fire the permanent switch at the
    first observation."""
    schedule = _boundary_schedule()
    dw = _dw()
    eligible = ["a", "b", "c", "d", "e"]
    block = compute_panel_weights(
        new_weight_state(),
        obs_stamp=schedule.genesis_stamp,
        eligible=eligible,
        dw_params=dw,
        fallback_weights={sid: 0.2 for sid in eligible},
        schedule=schedule,
    )
    assert block["mode"] == "fallback"
    assert "switched_on" not in block
    for sid in eligible:
        source = block["sources"][sid]
        assert source["Q"] is None
        assert source["attendance_scheduled"] == 0
        assert source["attendance_ratio"] == 1.0
    assert block["weights"] == {sid: 0.2 for sid in eligible}


# ------------------------------------------------- perf-path equivalence


def _lcg(seed):
    x = seed
    while True:
        x = (1103515245 * x + 12345) % (1 << 31)
        yield x / float(1 << 31)


def _naive_pos(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _naive_clamp(value, max_abs):
    if max_abs is None:
        return value
    cap = float(max_abs)
    if value > cap:
        return cap
    if value < -cap:
        return -cap
    return value


def _naive_loo(prices, vector, exclude, t0, t1, max_abs):
    """The observation-mode LOO definition (both-endpoints total minus the
    scored source's own contribution), recomputed from scratch -- no
    caching, no sharing."""
    num = 0.0
    den = 0.0
    count = 0
    own = None
    for sid in sorted(vector):
        weight = vector[sid]
        if not _naive_pos(weight):
            continue
        series = prices.get(sid) or {}
        p0 = (series.get(t0) or {}).get("usd")
        p1 = (series.get(t1) or {}).get("usd")
        if not _naive_pos(p0) or not _naive_pos(p1):
            continue
        c_num = float(weight) * float(p1)
        c_den = float(weight) * float(p0)
        num += c_num
        den += c_den
        count += 1
        if sid == exclude:
            own = (c_num, c_den)
    if own is not None:
        num -= own[0]
        den -= own[1]
        count -= 1
    if count == 0 or num <= 0 or den <= 0:
        return None
    return _naive_clamp(math.log(num / den), max_abs)


def _naive_scores(state, obs_stamp, source_ids, dw, schedule):
    """Direct reimplementation of the observation-mode sampling: one full
    pass per (source, horizon, tau), rebuilding features per horizon and
    every LOO sum per call. The fast path must match this bit for bit."""
    cutoff = schedule.prev_scheduled_stamp(obs_stamp)
    prices = state.get("prices") or {}
    vectors = state.get("vectors") or {}
    lookbacks = [int(x) for x in dw["lookback_horizons_hours"]]
    horizons = [int(x) for x in dw["forward_horizons_hours"]]
    history_minutes = int(dw["history_days"]) * 1440
    max_abs = dw.get("max_abs_log_return")
    out = {}
    for sid in source_ids:
        series = prices.get(sid) or {}
        q_by_h = {}
        n_by_h = {}
        for h in horizons:
            samples = []
            for tau in sorted(series):
                if tau < cutoff - history_minutes or tau + h * 60 > cutoff:
                    continue
                vector_stamp = None
                for vs in sorted(vectors):  # linear scan on purpose
                    if vs <= tau:
                        vector_stamp = vs
                if vector_stamp is None:
                    continue
                vector = vectors[vector_stamp]
                if not vector:
                    continue
                target = _naive_loo(
                    prices, vector, sid, tau, tau + h * 60, max_abs
                )
                if target is None:
                    continue
                feats = []
                for lb in lookbacks:
                    own_r = source_return(
                        series, tau - lb * 60, tau, max_abs_log_return=max_abs
                    )
                    rest_r = _naive_loo(
                        prices, vector, sid, tau - lb * 60, tau, max_abs
                    )
                    if own_r is None or rest_r is None:
                        feats = None
                        break
                    feats.append(own_r - rest_r)
                if feats is None:
                    continue
                samples.append((tau, feats, target))
            q, n_samples = in_sample_q(
                samples,
                anchor=cutoff,
                ridge_lambda=float(dw["ridge_lambda"]),
                half_life=float(dw["half_life_days"]) * 1440.0,
                min_train_samples=int(dw["min_train_samples"]),
                target_variance_floor=float(dw["target_variance_floor"]),
            )
            q_by_h[str(h)] = q
            n_by_h[str(h)] = n_samples
        defined = all(q is not None for q in q_by_h.values())
        q_score = (
            sum(q for q in q_by_h.values() if q is not None) / len(q_by_h)
            if defined and q_by_h
            else None
        )
        out[sid] = {"q": q_by_h, "n_samples": n_by_h, "Q": q_score}
    return out


def test_fast_scoring_path_is_bit_identical_to_the_naive_formulation():
    """Design section 5 perf rules under proof: shared per-tau features and
    the cached full-sum-minus-own LOO pass must be NUMERICALLY IDENTICAL
    to recomputing everything from scratch, on gnarly data -- dropouts,
    a currency change (returns never FX-spliced), two pinned vectors, and
    a window spanning the 4-slot -> hourly era boundary."""
    schedule = _boundary_schedule()
    dw = _dw(
        lookback_horizons_hours=[6, 24],
        forward_horizons_hours=[6, 24],
        history_days=4,
    )
    obs = date_hour_to_stamp("2026-08-25", 6)
    rng = _lcg(20260823)
    source_ids = ["a", "b", "c", "d", "e", "f"]
    bases = [7.5, 9.0, 10.0, 11.0, 3.2, 5.0]
    stamps = schedule.scheduled_stamps(
        date_hour_to_stamp("2026-08-18", 0), obs
    )
    state = new_weight_state()
    flip = date_hour_to_stamp("2026-08-23", 0)
    for base, sid in zip(bases, source_ids):
        price = base
        for stamp in stamps:
            price *= math.exp((next(rng) - 0.5) * 0.06)
            if next(rng) < 0.2:
                continue  # dropout: gaps, never carry-forward
            if sid == "f" and stamp >= flip:
                entry = _entry(round(price, 6), "EUR", round(price * 1.08, 6))
            else:
                entry = _entry(round(price, 6))
            state["prices"].setdefault(sid, {})[stamp] = entry
    state["vectors"][stamps[0]] = {
        sid: round(1.0 / 6, 6) for sid in source_ids
    }
    state["vectors"][date_hour_to_stamp("2026-08-22", 16)] = {
        "a": 0.3, "b": 0.2, "c": 0.15, "d": 0.15, "e": 0.1, "f": 0.1,
    }
    fast = predictive_scores_obs(
        state,
        obs_stamp=obs,
        source_ids=source_ids,
        dw_params=dw,
        schedule=schedule,
    )
    naive = _naive_scores(state, obs, source_ids, dw, schedule)
    assert fast == naive  # exact float equality, key for key
    # The fixture genuinely exercises both paths.
    assert any(v["Q"] is not None for v in fast.values())
    assert any(v["Q"] is None for v in fast.values())
    assert any(
        n > 0 for v in fast.values() for n in v["n_samples"].values()
    )


# ----------------------------------------------------------------- pruning


def test_pruning_provably_changes_no_computable_score():
    schedule = _hourly_schedule("2026-06-01")
    dw = _dw()
    obs = date_hour_to_stamp("2026-07-20", 8)
    state, stamps = _core_walk_state(schedule, obs, span_hours=400)
    threshold = obs - (48 * 60 + 2 * 60 + PRUNE_MARGIN_MINUTES)
    assert stamps[0] < threshold  # deep history: pruning has work to do
    row = {"lead": 10.0, "echo": 10.0, "s1": 10.0, "s2": 9.0, "s3": 11.0}
    prints = {sid: _entry(price) for sid, price in row.items()}
    vector = {sid: 0.2 for sid in row}
    unpruned = copy.deepcopy(state)
    for sid, entry in prints.items():  # manual, prune-free advance
        unpruned["prices"][sid][obs] = dict(entry)
    unpruned["vectors"][obs] = dict(vector)
    pruned = copy.deepcopy(state)
    advance_panel_weight_state(
        pruned,
        obs_stamp=obs,
        prints=prints,
        vector=vector,
        mode="fallback",
        dw_params=dw,
    )
    assert min(min(s) for s in unpruned["prices"].values()) < threshold
    assert min(min(s) for s in pruned["prices"].values()) >= threshold
    # The only pre-threshold vector survives as the resolution anchor.
    assert stamps[0] in pruned["vectors"]
    next_obs = obs + 60
    kwargs = dict(dw_params=dw, schedule=schedule)
    scores_pruned = predictive_scores_obs(
        pruned, obs_stamp=next_obs, source_ids=sorted(row), **kwargs
    )
    scores_full = predictive_scores_obs(
        unpruned, obs_stamp=next_obs, source_ids=sorted(row), **kwargs
    )
    assert scores_pruned == scores_full
    assert any(v["Q"] is not None for v in scores_pruned.values())  # not vacuous
    block_pruned = compute_panel_weights(
        pruned,
        obs_stamp=next_obs,
        eligible=sorted(row),
        fallback_weights=vector,
        **kwargs,
    )
    block_full = compute_panel_weights(
        unpruned,
        obs_stamp=next_obs,
        eligible=sorted(row),
        fallback_weights=vector,
        **kwargs,
    )
    assert block_pruned == block_full


def test_pruning_refuses_a_lookback_wider_than_the_margin():
    dw = _dw(
        lookback_horizons_hours=[1, 200],
        forward_horizons_hours=[1, 2],
    )
    with pytest.raises(ValueError, match="PRUNE_MARGIN_MINUTES"):
        advance_panel_weight_state(
            new_weight_state(),
            obs_stamp=date_hour_to_stamp("2026-08-15", 12),
            prints={},
            vector={},
            mode="fallback",
            dw_params=dw,
        )


def test_prune_sweep_defers_until_the_threshold_advances_a_day():
    """Hot-loop invariant (review perf stage): the full prune sweep runs
    only when its threshold has advanced >= 24h since the last sweep --
    a state with no bookkeeping key (hand-built fixtures, first advance)
    prunes immediately and gains the key; hourly advances inside the 24h
    fence keep sub-threshold data (score-neutral: nothing below the
    strict threshold is ever consulted); the first advance past the
    fence sweeps everything up to ITS threshold."""
    schedule = _hourly_schedule("2026-06-01")
    dw = _dw()
    span = 48 * 60 + 2 * 60 + PRUNE_MARGIN_MINUTES  # the strict per-advance bound
    obs = date_hour_to_stamp("2026-07-20", 8)
    state, stamps = _core_walk_state(schedule, obs, span_hours=400)
    assert state["_prune_threshold"] is None  # new_weight_state seeds it
    row = {sid: _entry(10.0) for sid in CORE}

    def _advance(at):
        advance_panel_weight_state(
            state,
            obs_stamp=at,
            prints=row,
            vector={sid: 0.2 for sid in CORE},
            mode="fallback",
            dw_params=dw,
        )

    _advance(obs)  # no key recorded yet -> sweeps and records
    assert state["_prune_threshold"] == obs - span
    assert min(min(s) for s in state["prices"].values()) >= obs - span
    # Re-add a stamp just under the NEXT hour's threshold: hourly
    # advances inside the 24h fence must NOT sweep it away.
    stale = obs - span - 1
    state["prices"]["lead"][stale] = _entry(10.0)
    for hour in range(1, 24):
        _advance(obs + hour * 60)
    assert state["_prune_threshold"] == obs - span  # unchanged: deferred
    assert stale in state["prices"]["lead"]
    # 24h of threshold advance later the sweep fires and catches up.
    _advance(obs + 24 * 60)
    assert state["_prune_threshold"] == obs + 24 * 60 - span
    assert stale not in state["prices"]["lead"]
    assert min(min(s) for s in state["prices"].values()) >= obs + 24 * 60 - span


# ------------------------------------------------------------------ replay


def test_replay_from_pinned_facts_rebuilds_state_and_next_vector():
    """THE determinism contract, observation-mode: a fresh state advanced
    from each observation's pinned facts alone (prints + rounded vector +
    mode, through the JSON boundary every real replay crosses) equals the
    live-carried state float for float -- across the era boundary and the
    switch day -- and prices the next observation identically."""
    schedule = PanelSchedule(
        genesis_date="2026-08-22",
        slot_grids=[
            {"from_date": "2026-08-22", "slot_hours_utc": [4, 10, 16, 22]},
            {"from_date": "2026-08-24", "slot_hours_utc": list(range(24))},
        ],
    )
    dw = _dw(lookback_horizons_hours=[6], forward_horizons_hours=[6])
    fallback = {sid: 0.2 for sid in CORE}
    last = date_hour_to_stamp("2026-08-25", 12)
    stamps = list(schedule.iter_scheduled(last))
    levels = [0.0]
    for i in range(1, len(stamps) + 1):
        levels.append(levels[-1] + 0.02 * _SIGNS[i % len(_SIGNS)])

    def _row(i):
        return {
            "lead": round(10.0 * math.exp(levels[i]), 6),
            "echo": round(10.0 * math.exp(levels[i - 1] if i else 0.0), 6),
            "s1": 10.0,
            "s2": 9.0,
            "s3": 11.0,
        }

    live = new_weight_state()
    pinned = []
    blocks = []
    for i, stamp in enumerate(stamps):
        block = compute_panel_weights(
            live,
            obs_stamp=stamp,
            eligible=list(CORE),
            dw_params=dw,
            fallback_weights=fallback,
            schedule=schedule,
        )
        prints = {
            sid: series_print(price, (price, "USD"))
            for sid, price in _row(i).items()
        }
        advance_panel_weight_state(
            live,
            obs_stamp=stamp,
            prints=prints,
            vector=block["weights"],
            mode=block["mode"],
            dw_params=dw,
        )
        blocks.append(block)
        pinned.append(
            (
                stamp,
                json.loads(
                    json.dumps(
                        {
                            "prints": prints,
                            "weights": block["weights"],
                            "mode": block["mode"],
                        }
                    )
                ),
            )
        )
    # The drive crossed the switch exactly once and stamped it fully.
    switch_stamps = [b["switched_on"] for b in blocks if "switched_on" in b]
    assert len(switch_stamps) == 1
    assert len(switch_stamps[0]) == 13 and switch_stamps[0][10] == "T"
    modes = [b["mode"] for b in blocks]
    assert "fallback" in modes and modes[-1] == "dynamic"
    assert modes == sorted(modes, key=lambda m: m == "dynamic")  # monotonic
    replayed = new_weight_state()
    for stamp, facts in pinned:
        advance_panel_weight_state(
            replayed,
            obs_stamp=stamp,
            prints=facts["prints"],
            vector=facts["weights"],
            mode=facts["mode"],
            dw_params=dw,
        )
    assert replayed == live
    next_stamp = last + 60
    next_live = compute_panel_weights(
        live,
        obs_stamp=next_stamp,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights=fallback,
        schedule=schedule,
    )
    next_replayed = compute_panel_weights(
        replayed,
        obs_stamp=next_stamp,
        eligible=list(CORE),
        dw_params=dw,
        fallback_weights=fallback,
        schedule=schedule,
    )
    assert next_live == next_replayed
    assert next_live["mode"] == "dynamic"
