# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Attendance weighting (METHODOLOGY.md sections 8.6-8.7).

The three legs the methodology names, plus the contract that makes it
safe to ship into a live series:

  - **A_i**, the exponentially weighted attendance average over the
    90-day window: our-side failures drop from BOTH sums (a provider is
    never penalised for our outage), everything else absent counts 0
    (the ramp-in rule), a genesis/all-skip window scores 1.0;
  - **the hard cutoff**: past K_A wall-time hours without a usable
    price the seat is excluded until it prints again -- resolved by a
    backward walk in which our failures consume nothing, so an outage
    can never manufacture (or erase) an exclusion, and per-stamp era
    spacing keeps the verdict stable across a cadence change;
  - **allocation**: softmax(gamma*Q + eta*ln A) with ceilings scaled by
    A_i and floors collapsing with them, so a fading seat rides to
    near-zero instead of hitting a second cliff.

And the load-bearing safety property, pinned twice over: with the knobs
ABSENT the engine is byte-identical to the pre-attendance one (the
published series must not move because code landed), and at A_i = 1 for
every seat the armed allocator reduces BIT-EXACTLY to the legacy body
(so arming moves weights only where attendance actually differs).

The classifier is pinned as ONE function serving both the live path and
replay: every verdict is decided by published artifact bytes alone, or a
replayed series would fork from the series that published it.
"""

from __future__ import annotations

import math

import pytest

from gpu_index.index.panel import (
    ATTENDANCE_KNOWN_STATUSES,
    CARRY_BASIS_NO_PRICE,
    attendance_events_for_stamp,
    classify_attendance_source,
)
from gpu_index.index.panel_schedule import PanelSchedule
from gpu_index.index.weights import (
    EVENT_NO_PRICE,
    EVENT_SKIP,
    advance_panel_weight_state,
    allocate_weights,
    attendance_armed,
    attendance_minted,
    compute_attendance_view,
    compute_panel_weights,
    new_weight_state,
    series_print,
    validate_attendance_params,
)

HALF_LIFE_HOURS = 6.0
EXCLUSION_HOURS = 24.0


def _schedule(genesis="2026-08-01"):
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


def _minted(**overrides):
    knobs = {
        "attendance_half_life_hours": HALF_LIFE_HOURS,
        "attendance_eta": 0.0,
        "no_price_exclusion_hours": EXCLUSION_HOURS,
    }
    knobs.update(overrides)
    return _dw(**knobs)


def _armed(eta=0.5, **overrides):
    return _minted(attendance_eta=eta, **overrides)


def _state(prices=None, events=None):
    state = new_weight_state()
    for sid, stamps in (prices or {}).items():
        state["prices"][sid] = {
            int(t): series_print(1.0, (1.0, "USD")) for t in stamps
        }
    if events is not None:
        state["events"] = {
            sid: {int(t): code for t, code in series.items()}
            for sid, series in events.items()
        }
    return state


# ------------------------------------------------------------ the knobs


def test_the_triple_rides_together_or_not_at_all():
    assert validate_attendance_params(_dw()) is None
    partial = _dw(attendance_eta=0.5)
    with pytest.raises(ValueError, match="ride together"):
        validate_attendance_params(partial)


def test_bounds_are_wall_time_hours_inside_the_history_window():
    history_hours = 2 * 24
    ok = validate_attendance_params(
        _dw(
            attendance_half_life_hours=history_hours,
            attendance_eta=0,
            no_price_exclusion_hours=history_hours,
        )
    )
    assert ok == {
        "attendance_half_life_hours": float(history_hours),
        "attendance_eta": 0.0,
        "no_price_exclusion_hours": float(history_hours),
    }
    for bad in ({"attendance_half_life_hours": history_hours + 1},
                {"no_price_exclusion_hours": 0},
                {"attendance_eta": -1},
                {"attendance_half_life_hours": True}):
        params = _minted()
        params.update(bad)
        with pytest.raises(ValueError):
            validate_attendance_params(params)


def test_minted_and_armed_are_one_home_each():
    assert not attendance_minted(_dw())
    assert not attendance_armed(_dw())
    assert attendance_minted(_minted())
    assert not attendance_armed(_minted())  # eta 0: minted but dark
    assert attendance_armed(_armed())


# ------------------------------------------------- the section-3 classifier


def test_classifier_reads_only_published_bytes():
    # State 1 (implicit): a trusted print, including a sigma-fenced one --
    # the fence holds a print out of the INDEX, never out of the record.
    assert classify_attendance_source(
        {"status": "ok", "chosen": {"usd_per_gpu_hr": 2.0},
         "filter": {"accepted": True}}
    ) is None
    assert classify_attendance_source(
        {"status": "ok", "chosen": {"usd_per_gpu_hr": 2.0},
         "filter": {"accepted": False}}
    ) is None
    # State 2: read fine, nothing usable.
    assert classify_attendance_source({"status": "ok"}) == EVENT_NO_PRICE
    assert classify_attendance_source(
        {"status": "ok", "chosen": {"usd_per_gpu_hr": 2.0},
         "filter": {"untrusted_currency": True}}
    ) == EVENT_NO_PRICE
    assert classify_attendance_source({"status": "held_out"}) == EVENT_NO_PRICE
    assert classify_attendance_source(
        {"status": "uncorroborated_jump"}
    ) == EVENT_NO_PRICE
    # State 3: our failure.
    assert classify_attendance_source({"status": "error"}) == EVENT_SKIP
    assert classify_attendance_source({"status": "fx_unavailable"}) == EVENT_SKIP
    assert classify_attendance_source(
        {"status": "manually_excluded"}
    ) == EVENT_SKIP
    # No entry: implicit, never carried.
    assert classify_attendance_source({"status": "missing"}) is None


def test_carry_basis_is_the_replay_discriminator():
    """The ONE byte that tells an armed no-price carry (the provider had
    nothing) from a collection carry (we failed) on replay."""
    assert classify_attendance_source(
        {"status": "carried",
         "carried": {"carry_basis": CARRY_BASIS_NO_PRICE}}
    ) == EVENT_NO_PRICE
    assert classify_attendance_source(
        {"status": "carried", "carried": {"failure_kind": "fetch"}}
    ) == EVENT_SKIP


def test_an_unknown_status_fails_closed_to_skip():
    """A vocabulary a later binary invents freezes A_i rather than
    silently penalising the provider."""
    assert "some_future_status" not in ATTENDANCE_KNOWN_STATUSES
    assert classify_attendance_source(
        {"status": "some_future_status"}
    ) == EVENT_SKIP


def test_a_lane_wide_outage_is_skip_for_every_seat():
    """Precedence: our capture failed, so no provider is marked absent --
    the per-source rows all read 'missing' and would otherwise count 0."""
    rows = [{"source_id": "a", "status": "missing"},
            {"source_id": "b", "status": "missing"}]
    assert attendance_events_for_stamp(
        rows, observation_missed=True, record_quarantined=None
    ) == {"a": EVENT_SKIP, "b": EVENT_SKIP}
    assert attendance_events_for_stamp(
        rows, observation_missed=False, record_quarantined="poisoned"
    ) == {"a": EVENT_SKIP, "b": EVENT_SKIP}
    # Without the lane-wide flags the same rows are implicit no-entry.
    assert attendance_events_for_stamp(
        rows, observation_missed=False, record_quarantined=None
    ) == {}


# ------------------------------------------------------------ A_i itself


def test_a_present_provider_scores_one_through_our_outages():
    """The only reading under which our failures cost the provider
    nothing: skip stamps leave BOTH sums."""
    schedule = _schedule()
    obs = schedule.genesis_stamp + 48 * 60
    window = schedule.scheduled_stamps(obs - 2 * 1440, obs)
    present, outage = window[:-3], window[-3:]
    state = _state(
        prices={"always": present},
        events={"always": {t: EVENT_SKIP for t in outage}},
    )
    view = compute_attendance_view(
        state, ["always"], obs_stamp=obs, schedule=schedule,
        dw_params=_minted(),
    )
    assert view["always"]["factor"] == 1.0


def test_absence_fades_at_the_declared_half_life():
    """No prints at all inside the window: A_i = 0, and one print at
    exactly one half-life back weighs half of one at the newest stamp."""
    schedule = _schedule()
    obs = schedule.genesis_stamp + 48 * 60
    view = compute_attendance_view(
        _state(prices={"quiet": []}), ["quiet"], obs_stamp=obs,
        schedule=schedule, dw_params=_minted(),
    )
    assert view["quiet"]["factor"] == 0.0

    window = schedule.scheduled_stamps(obs - 2 * 1440, obs)
    decay_total = sum(
        2.0 ** (-(obs - s) / (HALF_LIFE_HOURS * 60.0)) for s in window
    )
    one_stamp = window[-1]
    view = compute_attendance_view(
        _state(prices={"one": [one_stamp]}), ["one"], obs_stamp=obs,
        schedule=schedule, dw_params=_minted(),
    )
    expected = (
        2.0 ** (-(obs - one_stamp) / (HALF_LIFE_HOURS * 60.0)) / decay_total
    )
    assert view["one"]["factor"] == pytest.approx(expected, rel=1e-12)


def test_a_genesis_window_scores_one_not_zero():
    """Nothing scheduled means nothing missed -- the existing ratio's
    convention, so a fresh lane never opens at A_i = 0."""
    schedule = _schedule()
    view = compute_attendance_view(
        _state(), ["new"], obs_stamp=schedule.genesis_stamp,
        schedule=schedule, dw_params=_minted(),
    )
    assert view["new"]["factor"] == 1.0


def test_the_shared_decay_table_matches_a_naive_per_source_recompute():
    """One decay table per observation is a perf shape, not a different
    number: bit-identical to computing w(s) per source."""
    schedule = _schedule()
    obs = schedule.genesis_stamp + 48 * 60
    window = schedule.scheduled_stamps(obs - 2 * 1440, obs)
    prices = {"a": window[::2], "b": window[1::3], "c": window[-5:]}
    events = {"b": {t: EVENT_SKIP for t in window[:4]}}
    state = _state(prices=prices, events=events)
    view = compute_attendance_view(
        state, sorted(prices), obs_stamp=obs, schedule=schedule,
        dw_params=_minted(),
    )
    for sid in prices:
        num = den = 0.0
        for s in window:
            if events.get(sid, {}).get(s) == EVENT_SKIP:
                continue
            w = 2.0 ** (-(obs - s) / (HALF_LIFE_HOURS * 60.0))
            den += w
            if s in set(prices[sid]):
                num += w
        naive = 1.0 if den == 0.0 else num / den
        assert view[sid]["factor"] == naive


# ------------------------------------------------------- the hard cutoff


def _quiet_state(schedule, obs, quiet_hours, *, skips=()):
    window = schedule.scheduled_stamps(obs - 2 * 1440, obs)
    quiet = [s for s in window if s >= obs - quiet_hours * 60]
    printed = [s for s in window if s not in set(quiet)]
    return _state(
        prices={"seat": printed},
        events={"seat": {
            t: (EVENT_SKIP if t in set(skips) else EVENT_NO_PRICE)
            for t in quiet
        }},
    ), quiet


def test_exclusion_fires_past_the_cutoff_and_not_before():
    schedule = _schedule()
    obs = schedule.genesis_stamp + 48 * 60
    for quiet_hours, expected in ((int(EXCLUSION_HOURS) - 1, False),
                                  (int(EXCLUSION_HOURS) + 1, True)):
        state, quiet = _quiet_state(schedule, obs, quiet_hours)
        view = compute_attendance_view(
            state, ["seat"], obs_stamp=obs, schedule=schedule,
            dw_params=_armed(),
        )
        assert view["seat"]["excluded"] is expected
        assert view["seat"]["streak"] == len(quiet)


def test_our_outage_can_never_manufacture_an_exclusion():
    """Skips consume nothing: a long our-side gap bracketed by a short
    provider absence is not K_A hours of provider absence."""
    schedule = _schedule()
    obs = schedule.genesis_stamp + 48 * 60
    window = schedule.scheduled_stamps(obs - 2 * 1440, obs)
    quiet = [s for s in window if s >= obs - 30 * 60]
    events = {t: EVENT_SKIP for t in quiet}
    # Only the two newest stamps are genuine provider absence.
    for t in quiet[-2:]:
        events[t] = EVENT_NO_PRICE
    state = _state(
        prices={"seat": [s for s in window if s not in set(quiet)]},
        events={"seat": events},
    )
    view = compute_attendance_view(
        state, ["seat"], obs_stamp=obs, schedule=schedule, dw_params=_armed(),
    )
    assert view["seat"]["excluded"] is False
    assert view["seat"]["streak"] == 2


def test_exclusion_is_armed_only_but_the_streak_always_publishes():
    """While dark the disclosure still rides every observation; only the
    verdict waits for eta."""
    schedule = _schedule()
    obs = schedule.genesis_stamp + 48 * 60
    state, quiet = _quiet_state(schedule, obs, int(EXCLUSION_HOURS) + 1)
    dark = compute_attendance_view(
        state, ["seat"], obs_stamp=obs, schedule=schedule, dw_params=_minted(),
    )
    armed = compute_attendance_view(
        state, ["seat"], obs_stamp=obs, schedule=schedule, dw_params=_armed(),
    )
    assert dark["seat"]["excluded"] is False
    assert armed["seat"]["excluded"] is True
    assert dark["seat"]["streak"] == armed["seat"]["streak"] == len(quiet)


def test_a_knob_less_lane_computes_no_view_at_all():
    """The structural skip: not a zero, not a default -- nothing."""
    schedule = _schedule()
    assert compute_attendance_view(
        _state(), ["seat"], obs_stamp=schedule.genesis_stamp + 60,
        schedule=schedule, dw_params=_dw(),
    ) is None


# --------------------------------------------------------- the allocation


def _scores(n=5):
    return {f"s{i}": 0.05 * i for i in range(n)}


def test_at_full_attendance_the_armed_allocator_is_bit_exact_legacy():
    """Arming must move weights only where attendance actually differs;
    every intermediate float has to reduce, not merely round the same."""
    scores = _scores()
    legacy, legacy_flags = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
    )
    armed, armed_flags = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
        attendance_factors={sid: 1.0 for sid in scores}, attendance_eta=0.5,
    )
    assert armed == legacy
    assert armed_flags == legacy_flags


def test_eta_zero_never_reaches_the_armed_body():
    scores = _scores()
    legacy, _ = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
    )
    dark, _ = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
        attendance_factors={sid: 0.1 for sid in scores}, attendance_eta=0.0,
    )
    assert dark == legacy


def test_a_fading_seat_rides_below_the_floor_instead_of_off_a_cliff():
    """The ceiling scales with A_i and the floor collapses with it, so a
    seat at 4% attendance is capped at 4% of the ceiling -- below w_min,
    which is the point (a second cliff at A ~ w_min/w_max is what the
    collapsing floor exists to avoid)."""
    scores = _scores()
    factors = {sid: 1.0 for sid in scores}
    factors["s0"] = 0.04
    weights, _ = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
        attendance_factors=factors, attendance_eta=0.5,
    )
    assert weights["s0"] == pytest.approx(0.30 * 0.04, rel=1e-12)
    assert weights["s0"] < 0.025
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-9)


def test_a_zero_attendance_seat_never_reaches_the_log():
    """ln(0) is never called; the seat keeps an explicit 0.0 row."""
    scores = _scores()
    factors = {sid: 1.0 for sid in scores}
    factors["s0"] = 0.0
    weights, _ = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
        attendance_factors=factors, attendance_eta=0.5,
    )
    assert weights["s0"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-9)


def test_ceilings_too_low_to_hold_the_mass_expand_proportionally():
    """Panel-wide low attendance pins the allocation at the ceilings and
    flags it, rather than publishing an infeasible vector."""
    scores = _scores()
    factors = {sid: 0.1 for sid in scores}  # cap mass 5 * 0.03 = 0.15
    weights, flags = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
        attendance_factors=factors, attendance_eta=0.5,
    )
    assert flags["degenerate_allocation"] == "cap_proportional"
    assert weights == {sid: pytest.approx(1 / 5) for sid in scores}


def test_every_seat_at_zero_attendance_publishes_uniform_not_a_divide():
    scores = _scores()
    weights, flags = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30, source_caps={},
        attendance_factors={sid: 0.0 for sid in scores}, attendance_eta=0.5,
    )
    assert flags["degenerate_allocation"] == "uniform"
    assert weights == {sid: pytest.approx(1 / 5) for sid in scores}


def test_the_tilt_is_exactly_eta_log_a():
    """The published formula, checked against a direct softmax rather
    than the implementation's own arithmetic."""
    # Attendance far enough above the fade that the scaled ceilings do
    # not bind -- this pins the TILT, not the ceiling rule.
    scores = {"a": 0.2, "b": 0.1}
    factors = {"a": 0.8, "b": 1.0}
    eta, gamma, w_min = 0.5, 4.0, 0.0
    weights, _ = allocate_weights(
        scores, gamma=gamma, weight_min=w_min, weight_max=1.0,
        source_caps={}, attendance_factors=factors, attendance_eta=eta,
    )
    raw = {
        sid: math.exp(gamma * scores[sid] + eta * math.log(factors[sid]))
        for sid in scores
    }
    total = sum(raw.values())
    for sid in scores:
        assert weights[sid] == pytest.approx(raw[sid] / total, rel=1e-12)


# ---------------------------------------------------- state and the block


def test_a_knob_less_lane_never_grows_the_events_key():
    """The dark contract at the state layer: no new bytes on a lane that
    was not minted, even when events are handed in."""
    state = new_weight_state()
    advance_panel_weight_state(
        state, obs_stamp=1440, prints={}, vector=None, mode="fallback",
        dw_params=_dw(), events={"a": EVENT_NO_PRICE},
    )
    assert "events" not in state


def test_a_minted_lane_records_and_prunes_events_with_prices():
    state = new_weight_state()
    advance_panel_weight_state(
        state, obs_stamp=1440, prints={}, vector=None, mode="fallback",
        dw_params=_minted(), events={"a": EVENT_NO_PRICE, "b": EVENT_SKIP},
    )
    assert state["events"] == {"a": {1440: EVENT_NO_PRICE},
                               "b": {1440: EVENT_SKIP}}


def test_an_unknown_event_code_raises_before_anything_is_written():
    """All-or-nothing: a torn advance is a state no replay produces."""
    state = new_weight_state()
    with pytest.raises(ValueError, match="unknown attendance event code"):
        advance_panel_weight_state(
            state, obs_stamp=1440, prints={"a": series_print(1.0, (1.0, "USD"))},
            vector={"a": 1.0}, mode="dynamic", dw_params=_minted(),
            events={"a": "wat"},
        )
    assert state["prices"] == {}
    assert state.get("mode", "fallback") == "fallback"


def test_the_publication_domain_widens_to_every_member_when_minted():
    """A seat absent longer than the history window still publishes its
    exclusion state -- otherwise the disclosure disappears exactly when
    it matters most."""
    schedule = _schedule()
    obs = schedule.genesis_stamp + 48 * 60
    state = _state(prices={"printer": schedule.scheduled_stamps(
        obs - 2 * 1440, obs)})
    fallback = {"printer": 0.5, "ghost": 0.5}
    block = compute_panel_weights(
        state, obs_stamp=obs, eligible=["printer"], dw_params=_minted(),
        fallback_weights=fallback, schedule=schedule,
    )
    assert set(block["sources"]) == {"printer", "ghost"}
    for sid in fallback:
        row = block["sources"][sid]
        assert set(row) >= {
            "attendance_factor", "no_price_streak", "no_price_excluded"
        }
    assert block["sources"]["ghost"]["attendance_factor"] == 0.0
    # ... and a knob-less lane keeps the scored set verbatim.
    bare = compute_panel_weights(
        state, obs_stamp=obs, eligible=["printer"], dw_params=_dw(),
        fallback_weights=fallback, schedule=schedule,
    )
    assert set(bare["sources"]) == {"printer"}
    assert "attendance_factor" not in bare["sources"]["printer"]


def test_carrying_seats_need_the_minted_knobs():
    schedule = _schedule()
    obs = schedule.genesis_stamp + 60
    with pytest.raises(ValueError, match="carrying seats require"):
        compute_panel_weights(
            _state(), obs_stamp=obs, eligible=[], dw_params=_dw(),
            fallback_weights={"a": 1.0}, schedule=schedule, carrying=["a"],
        )
