# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit + end-to-end tests for the calc_v5 dynamic weighting engine.

Pins: the allocation math (softmax -> floor -> per-source risk caps ->
iteratively redistributed global cap, uniform-with-flag on infeasible
bounds), the pure-Python ridge (weighted z-scoring, unpenalized intercept,
zero-variance features neutralized), the in-sample scoring gates
(R-insample: min_train / target-variance floor on the slot grid), the
return semantics (native same-currency source returns, USD fixed-weight
fixed-composition LOO returns, exact-date endpoints), the fallback-mode
parity invariant (index math byte-identical to the frozen fixed-weight
series), the permanent fallback -> dynamic switchover, and the replay
determinism contract (the CLI's advance-from-published rebuilds the exact
weight state the compute path carried).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from gpu_index.index.composite import calc_params, compute_day, resolve_slot_prints
from gpu_index.index.config import BasketConfigError, load_basket_config
from gpu_index.index.weights import (
    allocate_weights,
    build_samples,
    fit_ridge,
    loo_basket_return,
    predict_ridge,
    in_sample_q,
    solve_linear,
    source_return,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ws():
    return {"prices": {}, "vectors": {}, "mode": "fallback"}


# ------------------------------------------------------------- allocation


def test_allocate_all_equal_scores_is_uniform():
    weights, flags = allocate_weights(
        {s: 0.0 for s in "abcde"}, gamma=4.0, weight_min=0.025, weight_max=0.30
    )
    assert all(w == pytest.approx(0.2) for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert flags == {"degenerate_allocation": None, "capped": []}


def test_allocate_orders_by_score_and_respects_floor():
    weights, flags = allocate_weights(
        {"a": 0.30, "b": 0.15, "c": 0.0, "d": 0.0, "e": 0.0},
        gamma=4.0,
        weight_min=0.025,
        weight_max=0.30,
    )
    assert weights["a"] > weights["b"] > weights["c"]
    assert weights["c"] == weights["d"] == weights["e"]
    assert min(weights.values()) >= 0.025
    assert sum(weights.values()) == pytest.approx(1.0)
    assert flags["degenerate_allocation"] is None


def test_allocate_global_cap_redistributes_proportionally():
    """One dominant score rails into w_max; the excess flows to the
    uncapped proportionally to share, so equal-share losers stay equal."""
    weights, flags = allocate_weights(
        {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0},
        gamma=4.0,
        weight_min=0.025,
        weight_max=0.30,
    )
    assert weights["a"] == pytest.approx(0.30)
    assert flags["capped"] == ["a"]
    others = [weights[s] for s in "bcde"]
    assert all(w == pytest.approx(others[0]) for w in others)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_allocate_per_source_risk_cap_binds_below_global():
    """A thin-book source carrying its own lower per-source risk cap
    cannot ride predictiveness past that cap, however dominant its
    score."""
    weights, flags = allocate_weights(
        {"vast": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0},
        gamma=4.0,
        weight_min=0.025,
        weight_max=0.30,
        source_caps={"vast": 0.10},
    )
    assert weights["vast"] == pytest.approx(0.10)
    assert flags["capped"] == ["vast"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert max(weights.values()) <= 0.30 + 1e-9


def test_allocate_cascade_capping_terminates_and_sums_to_one():
    """Redistribution pushing a second source over its cap caps it on the
    next pass — the capped set grows monotonically, never thrashes."""
    weights, flags = allocate_weights(
        {"a": 1.0, "b": 0.9, "c": 0.0, "d": 0.0, "e": 0.0, "f": 0.0},
        gamma=8.0,
        weight_min=0.025,
        weight_max=0.30,
    )
    assert weights["a"] == pytest.approx(0.30)
    assert weights["b"] == pytest.approx(0.30)
    assert set(flags["capped"]) == {"a", "b"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert min(weights.values()) >= 0.025


def test_allocate_degenerate_bounds_flag_and_price_honestly():
    # Caps cannot hold the mass (n*w_max < 1, the small-N days B300's claim
    # floor of 1 makes real): CAP-PROPORTIONAL, preserving relative
    # haircuts — a capped source scales by the same factor as everyone.
    weights, flags = allocate_weights(
        {"a": 1.0, "b": 0.0}, gamma=4.0, weight_min=0.025, weight_max=0.30
    )
    assert weights == {"a": 0.5, "b": 0.5}  # equal caps -> equal shares
    assert flags["degenerate_allocation"] == "cap_proportional"
    assert "fallback_reason" in flags
    # Heterogeneous caps: the thin-book haircut survives proportionally
    # (vast at cap 0.10 gets 0.10/0.70, NOT uniform 1/3).
    weights, flags = allocate_weights(
        {"vast": 1.0, "b": 0.0, "c": 0.0},
        gamma=4.0,
        weight_min=0.025,
        weight_max=0.30,
        source_caps={"vast": 0.10},
    )
    assert flags["degenerate_allocation"] == "cap_proportional"
    assert weights["vast"] == pytest.approx(0.10 / 0.70)
    assert weights["b"] == pytest.approx(0.30 / 0.70)
    assert sum(weights.values()) == pytest.approx(1.0)
    # Floors alone overflow (n*w_min > 1): uniform.
    weights, flags = allocate_weights(
        {"a": 0.5, "b": 0.0, "c": 0.1}, gamma=4.0, weight_min=0.4, weight_max=1.0
    )
    assert weights == {sid: pytest.approx(1 / 3) for sid in "abc"}
    assert flags["degenerate_allocation"] == "uniform"
    # A per-source cap below the floor is unsatisfiable arithmetic: uniform
    # (runtime armor — the validator refuses such configs at load).
    weights, flags = allocate_weights(
        {"a": 0.5, "b": 0.0, "c": 0.1, "d": 0.2, "e": 0.0},
        gamma=4.0,
        weight_min=0.025,
        weight_max=0.30,
        source_caps={"a": 0.01},
    )
    assert flags["degenerate_allocation"] == "uniform"
    assert weights == {sid: pytest.approx(0.2) for sid in "abcde"}


def test_allocate_all_capped_at_exact_cap_mass_publishes_the_caps():
    """The exactly-1.0 corner: caps {0.3, 0.3, 0.3,
    0.1} sum to exactly 1 — the honest converged allocation IS the caps,
    never a uniform fallback (which would hand the 0.1-capped source 25%)."""
    weights, flags = allocate_weights(
        {"a": 1.0, "b": 0.9, "c": 0.8, "vast": 0.99},
        gamma=50.0,  # rail everyone over their cap
        weight_min=0.025,
        weight_max=0.30,
        source_caps={"vast": 0.10},
    )
    assert flags["degenerate_allocation"] is None
    assert weights == {
        "a": pytest.approx(0.30),
        "b": pytest.approx(0.30),
        "c": pytest.approx(0.30),
        "vast": pytest.approx(0.10),
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_allocate_is_insertion_order_independent():
    scores = {"a": 0.3, "b": 0.1, "c": 0.0, "d": 0.25, "e": 0.05}
    forward, _ = allocate_weights(
        scores, gamma=4.0, weight_min=0.025, weight_max=0.30
    )
    reversed_insertion = dict(reversed(list(scores.items())))
    backward, _ = allocate_weights(
        reversed_insertion, gamma=4.0, weight_min=0.025, weight_max=0.30
    )
    assert forward == backward


# ------------------------------------------------------------ ridge/solver


def test_solve_linear_known_system_and_singular_returns_none():
    assert solve_linear([[2.0, 0.0], [0.0, 4.0]], [2.0, 8.0]) == [1.0, 2.0]
    assert solve_linear([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0]) is None


def test_fit_ridge_recovers_exact_linear_relation_at_tiny_lambda():
    rows = [[0.1], [-0.1], [0.1], [-0.1], [0.1], [-0.1]]
    targets = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05]
    model = fit_ridge(rows, targets, [1.0] * 6, ridge_lambda=1e-9)
    assert predict_ridge(model, [0.1]) == pytest.approx(0.05, abs=1e-6)
    assert predict_ridge(model, [-0.1]) == pytest.approx(-0.05, abs=1e-6)


def test_fit_ridge_heavy_lambda_shrinks_toward_weighted_mean():
    rows = [[0.1], [-0.1], [0.1], [-0.1]]
    targets = [0.05, -0.05, 0.05, -0.05]
    model = fit_ridge(rows, targets, [1.0] * 4, ridge_lambda=1e9)
    # beta -> 0, intercept -> weighted mean (0 here).
    assert predict_ridge(model, [0.1]) == pytest.approx(0.0, abs=1e-6)


def test_fit_ridge_zero_variance_feature_is_neutralized():
    """Step-function list prices make all-zero excess-return features the
    ROUTINE case: the z-score guard must neutralize them (coefficient 0),
    never divide by zero or NaN a weight."""
    rows = [[0.0, 0.1], [0.0, -0.1], [0.0, 0.1], [0.0, -0.1]]
    targets = [0.05, -0.05, 0.05, -0.05]
    model = fit_ridge(rows, targets, [1.0] * 4, ridge_lambda=1e-9)
    assert model["betas"][0] == pytest.approx(0.0)
    prediction = predict_ridge(model, [123.0, 0.1])  # dead feature ignored
    assert prediction == pytest.approx(0.05, abs=1e-6)
    assert math.isfinite(prediction)


# -------------------------------------------------------- returns/samples


def _series(entries):
    """{ordinal: (usd, native, ccy)} -> the weight-state price shape."""
    return {
        t: {"usd": u, "native": n, "currency": c}
        for t, (u, n, c) in entries.items()
    }


def test_source_return_native_terms_exact_days_same_currency():
    series = _series(
        {
            10: (7.5, 7.5, "USD"),
            11: (7.8, 7.8, "USD"),
            13: (8.7, 7.5, "EUR"),
            14: (8.8, 7.6, "EUR"),
        }
    )
    assert source_return(series, 10, 11) == pytest.approx(math.log(7.8 / 7.5))
    assert source_return(series, 11, 12) is None  # gap: no LOCF, ever
    assert source_return(series, 11, 13) is None  # spans a currency change
    # Native terms: the EUR source's return ignores the USD conversion.
    assert source_return(series, 13, 14) == pytest.approx(math.log(7.6 / 7.5))


def test_loo_basket_return_fixed_weights_fixed_composition():
    prices = {
        "a": _series({10: (8.0, 8.0, "USD"), 11: (8.8, 8.8, "USD")}),
        "b": _series({10: (6.0, 6.0, "USD"), 11: (6.3, 6.3, "USD")}),
        "c": _series({10: (7.0, 7.0, "USD")}),  # absent at t1 -> excluded
        "x": _series({10: (5.0, 5.0, "USD"), 11: (9.0, 9.0, "USD")}),
    }
    vector = {"a": 0.3, "b": 0.2, "c": 0.4, "x": 0.1}
    value = loo_basket_return(prices, vector, exclude="x", t0=10, t1=11)
    # c missing at t1 drops it from BOTH endpoints; x is excluded; the
    # denominators cancel so no renormalization is needed.
    expected = math.log((0.3 * 8.8 + 0.2 * 6.3) / (0.3 * 8.0 + 0.2 * 6.0))
    assert value == pytest.approx(expected)
    # No common source across the interval -> undefined.
    assert loo_basket_return(prices, {"c": 1.0}, exclude="x", t0=10, t1=11) is None


def test_build_samples_slot_grid_cutoff_vector_and_endpoint_rules():
    """R-slots + R-cutoff: anchors are the source's own slot stamps; every
    endpoint (feature and target) must be realized by the cutoff — the
    prior day's LAST slot — and an anchor whose DAY has no pinned weight
    vector contributes nothing."""
    slots = (4, 10, 16, 22)
    stamps = [d * 24 + h for d in range(21) for h in slots]
    prices = {
        "i": _series({t: (10.0 + 0.001 * t, 10.0 + 0.001 * t, "USD") for t in stamps}),
        "j": _series({t: (5.0 + 0.001 * t, 5.0 + 0.001 * t, "USD") for t in stamps}),
    }
    vectors = {d: {"i": 0.5, "j": 0.5} for d in range(21)}
    cutoff = 19 * 24 + 22  # computing day 20: yesterday's last slot
    samples = build_samples(
        prices,
        vectors,
        source_id="i",
        cutoff_hour=cutoff,
        horizon_hours=6,
        lookbacks_hours=[6],
        history_hours=90 * 24,
    )
    taus = [t for t, _, _ in samples]
    # Last anchor: target [tau, tau+6] must land ON the cutoff at most —
    # day 19 slot16 (target realized at day 19 slot22 == cutoff).
    assert max(taus) == cutoff - 6
    # First anchor: needs a print 6h back — day 0 slot10 (stamp 10).
    assert min(taus) == 10
    # A day with no pinned vector contributes no anchors.
    del vectors[10]
    samples = build_samples(
        prices,
        vectors,
        source_id="i",
        cutoff_hour=cutoff,
        horizon_hours=6,
        lookbacks_hours=[6],
        history_hours=90 * 24,
    )
    day10 = [t for t, _, _ in samples if 10 * 24 <= t < 11 * 24]
    assert day10 == []


# -------------------------------------------------------------- scoring


def _linear_samples(count, slope=0.5, noise_signs=None):
    # anchors on a 6h slot grid; x alternates +/-0.1; y = slope * x exactly
    # unless noise_signs supplies an incoherent target pattern.
    samples = []
    for k in range(1, count + 1):
        x = 0.1 if k % 2 else -0.1
        if noise_signs is None:
            y = slope * x
        else:
            y = 0.05 * noise_signs[k % len(noise_signs)]
        samples.append((k * 6, [x], y))
    return samples


def test_in_sample_q_perfect_fit_scores_one():
    q, n = in_sample_q(
        _linear_samples(12),
        anchor=14 * 6,
        ridge_lambda=1e-9,
        half_life=30 * 24,
        min_train_samples=3,
        target_variance_floor=1e-12,
    )
    assert n == 12
    assert q == pytest.approx(1.0, abs=1e-9)


def test_in_sample_q_undefined_below_min_train():
    q, n = in_sample_q(
        _linear_samples(2),
        anchor=10 * 6,
        ridge_lambda=1e-9,
        half_life=30 * 24,
        min_train_samples=3,
        target_variance_floor=1e-12,
    )
    assert q is None and n == 2


def test_in_sample_q_undefined_on_frozen_target():
    """A constant target has no movement to explain — 0/0 must never mint
    a score (the step-function degenerate case)."""
    samples = [(k * 6, [0.1 if k % 2 else -0.1], 0.0) for k in range(1, 13)]
    q, n = in_sample_q(
        samples,
        anchor=14 * 6,
        ridge_lambda=1e-9,
        half_life=30 * 24,
        min_train_samples=3,
        target_variance_floor=1e-12,
    )
    assert q is None and n == 12


def test_in_sample_q_optimism_is_bounded_by_shrinkage():
    """R-insample's accepted trade, pinned: in-sample R^2 on incoherent
    noise is POSITIVE (under ridge with an unpenalized intercept it is
    mathematically >= 0 — the clip guards only float dust), and heavy
    shrinkage is the counterweight — lambda -> huge sends betas -> 0 and
    q -> ~0 (the intercept-only fit)."""
    noise = _linear_samples(14, noise_signs=[1, 1, -1, 1, -1, -1, 1])
    q_loose, n = in_sample_q(
        noise,
        anchor=16 * 6,
        ridge_lambda=1e-9,
        half_life=30 * 24,
        min_train_samples=3,
        target_variance_floor=1e-12,
    )
    assert n == 14
    assert q_loose is not None and 0.0 <= q_loose <= 1.0
    q_shrunk, _ = in_sample_q(
        noise,
        anchor=16 * 6,
        ridge_lambda=1e9,
        half_life=30 * 24,
        min_train_samples=3,
        target_variance_floor=1e-12,
    )
    assert q_shrunk == pytest.approx(0.0, abs=1e-6)
    assert q_shrunk <= q_loose


# --------------------------------------------------- validator (config.py)


def _dyn_calc_block(**overrides):
    block = {
        "scheme": "predictive_v1",
        "lookback_horizons_hours": [6, 24, 48],
        "forward_horizons_hours": [6, 24, 48],
        "history_days": 90,
        "half_life_days": 30,
        "ridge_lambda": 1.0,
        "gamma": 4.0,
        "weight_min": 0.025,
        "weight_max": 0.30,
        "min_train_samples": 10,
        "target_variance_floor": 1e-12,
        "source_weight_caps": {"vast": 0.10},
    }
    block.update(overrides)
    return block


def _write_config(tmp_path, dyn_block):
    cfg = json.loads(
        (REPO_ROOT / "config" / "index_basket.json").read_text()
    )
    cfg["calc"]["dynamic_weights"] = dyn_block
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.mark.parametrize(
    "mutation",
    [
        {"scheme": "made_up_v9"},
        {"lookback_horizons_hours": [24, 6]},  # order is artifact bytes
        {"lookback_horizons_hours": []},
        {"forward_horizons_hours": [6, 6, 24]},
        {"lookback_horizons_hours": [7]},  # not on the 6h slot grid
        {"forward_horizons_hours": [3]},  # below one slot spacing
        {"half_life_days": 0},
        {"ridge_lambda": -1},
        {"weight_min": 0.0},
        {"weight_min": 0.4, "weight_max": 0.3},
        {"weight_min": 0.2},  # 8 constituents * 0.2 >= 1: floor eats simplex
        {"min_train_samples": 0},
        {"history_days": 6},  # can never define a score: silently inert
        {"source_weight_caps": {"nobody": 0.1}},
        {"source_weight_caps": {"vast": 0.01}},  # cap below the floor
        {"source_weight_caps": {"e2e": 0.1}},  # pool source, not constituent
    ],
)
def test_validator_rejects_malformed_dynamic_weights(tmp_path, mutation):
    path = _write_config(tmp_path, _dyn_calc_block(**mutation))
    with pytest.raises(BasketConfigError):
        load_basket_config(path)


def test_validator_accepts_the_live_blocks():
    for name in ("index_basket.json", "index_basket_b200.json"):
        cfg = load_basket_config(REPO_ROOT / "config" / name)
        assert "dynamic_weights" in cfg["calc"]


# ------------------------------------------- compute_day integration + e2e


def _obs(sku, usd, tier="on-demand"):
    return {
        "sku": sku,
        "price_usd_gpu_hr": usd,
        "price_native_per_gpu_hr": usd,
        "currency": "USD",
        "tier": tier,
        "gpu_count_basis": 1,
        "raw_value": str(usd),
        "raw_unit": "usd_per_gpu_hr",
        "implausible": False,
        "notes": "",
    }


def _snapshot(run_id, prices):
    return {
        "run_id": run_id,
        "late_fill": False,
        "sources": [
            {
                "source_id": sid,
                "status": "ok",
                "observations": [_obs("B300", price)],
            }
            for sid, price in prices.items()
        ],
    }


_SIGNS = [
    1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1,
    -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1,
]


def _dyn_test_config(tmp_path, **dyn_overrides):
    """A 5-constituent dynamic lane on the slot grid: 'lead' moves, 'echo'
    replays lead's previous SLOT, s1-s3 are frozen list prices — lead's
    excess movement predicts the rest-of-basket by construction."""
    dyn = {
        "scheme": "predictive_v1",
        "lookback_horizons_hours": [6, 24],
        "forward_horizons_hours": [6, 24],
        "history_days": 30,
        "half_life_days": 10,
        "ridge_lambda": 0.001,
        "gamma": 4.0,
        "weight_min": 0.025,
        "weight_max": 0.30,
        "min_train_samples": 3,
        "target_variance_floor": 1e-12,
        "switch_min_eligible": 5,
        "max_abs_log_return": 0.5,
        "source_weight_caps": {},
    }
    dyn.update(dyn_overrides)
    parity_id = "test_dyn_v0"
    cfg = {
        "schema_version": 1,
        "basket_id": "test_dynamic_basket",
        "target_sku": "B300",
        "capture_slots_utc": [4, 10, 16, 22],
        "canonical_slot_utc": 16,
        "genesis_date": "2026-07-01",
        "bucket_prefix": "index/test_dyn",
        "fallback_parity_methodology_id": parity_id,
        "min_basket_sources_to_claim": 1,
        "calc": {
            "methodology_id": "test_dyn_v1",
            "interruptible_tiers": ["spot", "preemptible"],
            "filter_window": 20,
            "filter_sigma": 3.0,
            "filter_sigma_floor": 0.05,
            "filter_terms": "recorded_currency",
            "composite_statistic": "median_ci_votes",
            "filter_warmup": 10,
            "manual_verify_pct": 15,
            "promote_tie_break": "later",
            "fx_lane": "none",
            "fallback_pool_sku": "B200",
            "fallback_pool_sources": [],
            "dynamic_weights": dyn,
        },
        "sources": [
            {
                "source_id": sid,
                "display_name": sid,
                "role": "b300_basket",
                "weight": 0.2,
                "source_type": "direct_principal",
                "skus": ["B300"],
            }
            for sid in ("lead", "echo", "s1", "s2", "s3")
        ],
    }
    path = tmp_path / "dyn.json"
    path.write_text(json.dumps(cfg))
    return load_basket_config(path)


def _slot_levels(n_slots):
    g = [0.0]
    for i in range(1, n_slots + 1):
        g.append(g[-1] + 0.02 * _SIGNS[i % len(_SIGNS)])
    return g


_SLOT_HOURS = (4, 10, 16, 22)


def _slot_prices(day_index, slot_index, levels, skip=()):
    """One slot's synthetic basket: 'lead' walks the signs path, 'echo'
    replays lead ONE SLOT (6h) late, s1-s3 are frozen list prices — lead's
    excess movement predicts the rest-of-basket at every horizon by
    construction."""
    k = day_index * 4 + slot_index
    prices = {
        "lead": round(10.0 * math.exp(levels[k]), 6),
        "echo": round(10.0 * math.exp(levels[k - 1] if k else 0.0), 6),
        "s1": 10.0,
        "s2": 9.0,
        "s3": 11.0,
    }
    for sid in skip:
        prices.pop(sid, None)
    return prices


def _day_snapshots(day_index, levels, skip=()):
    return {
        hour: _snapshot(
            f"run-{day_index}-{hour:02d}",
            _slot_prices(day_index, slot_index, levels, skip),
        )
        for slot_index, hour in enumerate(_SLOT_HOURS)
    }


def _prior_prints(config, day_index, levels, skip_days=None):
    """The R-slots feed for computing day_index: the prior day's slots
    resolved exactly the way the CLI resolves them."""
    if day_index == 0:
        return {}
    genesis = date.fromisoformat(config["genesis_date"])
    prior_day = (genesis + timedelta(days=day_index - 1)).isoformat()
    skip = (skip_days or {}).get(day_index - 1, [])
    return resolve_slot_prints(
        _day_snapshots(day_index - 1, levels, skip),
        config=config,
        day=prior_day,
        fx_records={},
    )


def _drive(config, days, *, skip_days=None):
    """Run compute_day over the synthetic slot-grid lane, threading all
    state the way the CLI does; returns (payloads, weight_state)."""
    genesis = date.fromisoformat(config["genesis_date"])
    levels = _slot_levels(days * 4 + 8)
    window_history: dict = {}
    window_currencies: dict = {}
    pending: dict = {}
    weight_state = _ws()
    payloads = []
    for i in range(days):
        day = (genesis + timedelta(days=i)).isoformat()
        skip = (skip_days or {}).get(i, [])
        day_snaps = _day_snapshots(i, levels, skip)
        payloads.append(
            compute_day(
                config=config,
                day=day,
                snapshot=day_snaps[16],
                substituted_from=None,
                window_history=window_history,
                fx_records={},
                window_currencies=window_currencies,
                pending_currencies=pending,
                weight_state=weight_state,
                prior_slot_prints=_prior_prints(config, i, levels, skip_days),
            )
        )
    return payloads, weight_state


def test_compute_day_requires_weight_state_under_dynamic_config(tmp_path):
    config = _dyn_test_config(tmp_path)
    with pytest.raises(ValueError, match="weight_state"):
        compute_day(
            config=config,
            day="2026-07-01",
            snapshot=_snapshot("r0", {"lead": 10.0}),
            substituted_from=None,
            window_history={},
            fx_records={},
            window_currencies={},
        )


def test_fallback_mode_pins_config_weights_restricted_to_eligible(tmp_path):
    """Day one, no history: fallback mode. The pinned vector is the opening
    2.1 config weights RESTRICTED to the day's eligible sources — raw
    values, deliberately unnormalized, so the composite's own passer
    renormalization reproduces the frozen fixed-weight series byte for
    byte. A missing source carries weight None in its detail."""
    config = _dyn_test_config(tmp_path)
    payload = compute_day(
        config=config,
        day="2026-07-01",
        snapshot=_snapshot("r0", {"lead": 10.0, "echo": 10.0, "s1": 10.0, "s2": 9.0}),
        substituted_from=None,
        window_history={},
        fx_records={},
        window_currencies={},
        pending_currencies={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    weight_calc = payload["weight_calc"]
    assert weight_calc["mode"] == "fallback"
    assert weight_calc["weights"] == {
        "echo": 0.2, "lead": 0.2, "s1": 0.2, "s2": 0.2
    }
    by_id = {s["source_id"]: s for s in payload["sources"]}
    assert by_id["s3"]["weight"] is None
    assert by_id["lead"]["weight"] == 0.2
    # Parity invariant: the same snapshot under the SAME config minus
    # dynamic_weights (the frozen-series shape) prices identically.
    frozen = json.loads(json.dumps(config))
    del frozen["calc"]["dynamic_weights"]
    legacy = compute_day(
        config=frozen,
        day="2026-07-01",
        snapshot=_snapshot("r0", {"lead": 10.0, "echo": 10.0, "s1": 10.0, "s2": 9.0}),
        substituted_from=None,
        window_history={},
        fx_records={},
        window_currencies={},
        pending_currencies={},
    )
    assert payload["index"] == legacy["index"]
    assert "weight_calc" not in legacy


def test_switchover_is_loud_permanent_and_rewards_the_predictive_source(tmp_path):
    config = _dyn_test_config(tmp_path)
    payloads, weight_state = _drive(config, 20)
    modes = [p["weight_calc"]["mode"] for p in payloads]
    switch_days = [
        i
        for i, p in enumerate(payloads)
        if p["weight_calc"].get("switched_on")
    ]
    assert len(switch_days) == 1, "exactly one switch day, flagged"
    switch = switch_days[0]
    assert all(m == "fallback" for m in modes[:switch])
    assert all(m == "dynamic" for m in modes[switch:]), "the switch is permanent"
    assert weight_state["mode"] == "dynamic"
    # Pre-switch days pin the config weights; the estimator block still
    # records the (undefined) scores so warm-up is auditable.
    pre = payloads[max(switch - 1, 0)]["weight_calc"]
    assert pre["weights"] == {s: 0.2 for s in ("echo", "lead", "s1", "s2", "s3")}
    # Post-switch: lead's excess movement predicts the rest-of-basket by
    # construction, so lead's Q tops echo's (echo's target is the FUTURE
    # lead step — unpredictable) and lead carries the max weight.
    final = payloads[-1]["weight_calc"]
    assert final["sources"]["lead"]["Q"] is not None
    assert final["sources"]["lead"]["Q"] > (final["sources"]["echo"]["Q"] or 0.0)
    weights = final["weights"]
    assert max(weights, key=weights.get) == "lead"
    assert weights["lead"] > 0.2
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
    assert min(weights.values()) >= 0.025 - 1e-9
    assert max(weights.values()) <= 0.30 + 1e-9
    # The composite consumed the pinned dynamic weights: passers renormalize
    # from weight_calc.weights, not the config values.
    renorm = payloads[-1]["index"]["renormalized_weights"]
    assert renorm["lead"] == pytest.approx(
        weights["lead"] / sum(weights.values()), abs=1e-6
    )


def test_per_source_risk_cap_holds_in_the_full_engine(tmp_path):
    config = _dyn_test_config(tmp_path, source_weight_caps={"lead": 0.10})
    payloads, _ = _drive(config, 20)
    final = payloads[-1]["weight_calc"]
    assert final["mode"] == "dynamic"
    assert final["weights"]["lead"] == pytest.approx(0.10)
    assert "lead" in final["capped"]
    assert sum(final["weights"].values()) == pytest.approx(1.0, abs=1e-5)


def test_small_eligible_set_publishes_degenerate_cap_proportional(tmp_path):
    """After the switch, a day where only two sources print: n*w_max < 1 is
    arithmetically unsatisfiable — cap-proportional over eligible (equal
    caps here, so 0.5/0.5), flagged forever in the artifact, and the day
    still prices."""
    config = _dyn_test_config(tmp_path)
    payloads, _ = _drive(
        config, 20, skip_days={19: ["s1", "s2", "s3"]}
    )
    final = payloads[-1]["weight_calc"]
    assert final["mode"] == "dynamic"
    assert final["degenerate_allocation"] == "cap_proportional"
    assert final["weights"] == {"echo": 0.5, "lead": 0.5}
    assert payloads[-1]["basket_dark"] is False


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "compute_index_composite_dynw",
        REPO_ROOT / "scripts" / "compute_index_composite.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_from_published_rebuilds_the_exact_weight_state(tmp_path):
    """THE determinism contract: a fresh weight state advanced from the
    published artifacts alone (the CLI's replay path) equals the state the
    compute path carried — prices, pinned vectors, and the mode latch,
    float for float. If these ever diverge, later days' LOO baskets fork
    between a live run and a replay. The drive deliberately crosses the
    ugly terrain where forks hide: a per-source gap spell, a FULL dark day
    mid-series (empty weight vector — nothing may be stored for it), and
    the JSON serialization boundary every real replay crosses."""
    config = _dyn_test_config(tmp_path)
    payloads, driven_state = _drive(
        config,
        16,
        skip_days={
            5: ["s2"],
            6: ["s2"],
            10: ["lead", "echo", "s1", "s2", "s3"],  # full dark day
            11: ["echo"],
        },
    )
    dark = payloads[10]
    assert dark["basket_dark"] is True
    assert dark["weight_calc"]["weights"] == {}
    cli = _load_cli()
    replayed = _ws()
    for payload in payloads:
        # Round-trip through bytes: the real replay reads artifacts back
        # from the bucket, never in-memory dicts.
        cli.advance_weight_state_from_published(
            json.loads(json.dumps(payload)), replayed
        )
    assert replayed == driven_state
    # And the replayed state prices the NEXT day identically to a state
    # carried live (same vector, same scores, same pinned slot prints).
    genesis = date.fromisoformat(config["genesis_date"])
    levels = _slot_levels(17 * 4 + 8)
    day = (genesis + timedelta(days=16)).isoformat()
    prior = _prior_prints(config, 16, levels)
    kwargs = dict(
        config=config,
        day=day,
        snapshot=_day_snapshots(16, levels)[16],
        substituted_from=None,
        fx_records={},
        prior_slot_prints=prior,
    )
    live = compute_day(
        window_history={},
        window_currencies={},
        pending_currencies={},
        weight_state=driven_state,
        **kwargs,
    )
    again = compute_day(
        window_history={},
        window_currencies={},
        pending_currencies={},
        weight_state=replayed,
        **kwargs,
    )
    assert live["weight_calc"] == again["weight_calc"]


def test_dynamic_weights_ride_calc_params_under_the_pin(tmp_path):
    """The D2 fence must cover the weight methodology: every knob, the risk
    caps, AND the fallback vector are embedded — an edit to any of them
    (including a source's 2.1 weight, which sat OUTSIDE calc_params before
    this mint) now trips the refuse-to-extend check."""
    config = _dyn_test_config(tmp_path)
    params = calc_params(config)
    dw = params["dynamic_weights"]
    assert dw["fallback_weights"] == {
        "echo": 0.2, "lead": 0.2, "s1": 0.2, "s2": 0.2, "s3": 0.2
    }
    assert dw["scheme"] == "predictive_v1"
    mutated = json.loads(json.dumps(config))
    mutated["sources"][0]["weight"] = 0.25
    mutated["sources"][1]["weight"] = 0.15
    assert (
        calc_params(mutated)["dynamic_weights"]["fallback_weights"]
        != dw["fallback_weights"]
    )


# ----------------------------------------- review-hardening coverage (2026-08-23)


def test_source_return_winsorizes_at_the_pinned_cap():
    """R-winsor: an absurd trusted print's return clamps at the cap; a
    genuine repricing far below the cap is untouched."""
    series = _series(
        {
            10: (7.5, 7.5, "USD"),
            11: (7500.0, 7500.0, "USD"),  # crafted poison, ~log 6.9
            12: (8.3, 8.3, "USD"),
        }
    )
    assert source_return(series, 10, 11, max_abs_log_return=0.5) == 0.5
    assert source_return(series, 11, 12, max_abs_log_return=0.5) == -0.5
    genuine = source_return(series, 10, 12, max_abs_log_return=0.5)
    assert genuine == pytest.approx(math.log(8.3 / 7.5))
    # Unclamped when the param is absent (the pure-schema posture).
    assert source_return(series, 10, 11) == pytest.approx(math.log(1000.0))


def test_loo_return_winsorizes_the_poisoned_basket_leg():
    """One rival's absurd print inside the LOO sum is bounded before it can
    poison a source's features or targets."""
    prices = {
        "poison": _series({10: (7.0, 7.0, "USD"), 11: (7000.0, 7000.0, "USD")}),
        "b": _series({10: (6.0, 6.0, "USD"), 11: (6.1, 6.1, "USD")}),
        "i": _series({10: (8.0, 8.0, "USD"), 11: (8.0, 8.0, "USD")}),
    }
    vector = {"poison": 0.4, "b": 0.4, "i": 0.2}
    raw = loo_basket_return(prices, vector, exclude="i", t0=10, t1=11)
    clamped = loo_basket_return(
        prices, vector, exclude="i", t0=10, t1=11, max_abs_log_return=0.5
    )
    assert raw > 5.0  # unbounded poison
    assert clamped == 0.5


def test_sparse_source_no_longer_holds_the_switch(tmp_path):
    """R-quorum-v2 (supersedes the earlier
    all-recently-printed rule): the switch fires once enough eligible
    sources are scored — a sparse source that can never clear the sample
    gates does NOT block it, and it simply stays an auditable null."""
    config = _dyn_test_config(tmp_path, switch_min_eligible=4)
    # s3 prints ONLY on day 0 (its anchors there lack the 24h lookback, so
    # it can never earn a defined Q) and never prints again.
    payloads, weight_state = _drive(
        config, 12, skip_days={i: ["s3"] for i in range(1, 12)}
    )
    assert weight_state["mode"] == "dynamic"
    switch_days = [
        i for i, p in enumerate(payloads) if p["weight_calc"].get("switched_on")
    ]
    assert len(switch_days) == 1
    final = payloads[-1]["weight_calc"]
    assert final["mode"] == "dynamic"
    # The sparse source is still scored for the audit trail — permanently
    # null — and carries no weight while it is not eligible.
    assert final["sources"]["s3"]["Q"] is None
    assert "s3" not in final["weights"]


def test_switch_requires_the_eligible_quorum(tmp_path):
    """R-quorum leg (b): with switch_min_eligible above the day's eligible
    count, the latch never fires even when every scored source is defined
    — a thin day cannot latch the methodology."""
    config = _dyn_test_config(tmp_path, switch_min_eligible=5)
    payloads, weight_state = _drive(
        config, 20, skip_days={i: ["s3"] for i in range(20)}
    )
    # s3 NEVER prints: it is outside the quorum domain entirely; the other
    # four all earn defined Q — but eligible (4) < quorum (5), forever.
    final = payloads[-1]["weight_calc"]
    assert all(
        final["sources"][sid]["Q"] is not None
        for sid in ("lead", "echo", "s1", "s2")
    )
    assert weight_state["mode"] == "fallback"
    assert all(p["weight_calc"]["mode"] == "fallback" for p in payloads)


def test_post_switch_undefined_q_scores_zero_not_fallback(tmp_path):
    """R-undefined, post-switch leg: a source that becomes eligible only
    after the latch (no sample history) scores 0 — the day stays dynamic,
    the source floors at w_min-ish, and its Q publishes as an auditable
    null."""
    config = _dyn_test_config(tmp_path, switch_min_eligible=4)
    payloads, _ = _drive(config, 17, skip_days={i: ["s3"] for i in range(15)})
    # Day 16 is the first compute after s3 returned (day 15): s3 is
    # eligible, but its day-15 anchors lack the 24h lookback (day 14 is a
    # gap), so its Q is an auditable null and it scores 0.
    first_back = payloads[16]["weight_calc"]
    assert first_back["mode"] == "dynamic"
    assert first_back["sources"]["s3"]["Q"] is None
    assert first_back["weights"]["s3"] >= 0.025 - 1e-9
    assert first_back["weights"]["s3"] < 0.2  # scored 0, not fallback 0.2
    assert sum(first_back["weights"].values()) == pytest.approx(1.0, abs=1e-5)


def test_held_out_print_stays_in_weight_domain_and_series(tmp_path):
    """R-series/R-eligible on the slot grid: a fence-REJECTED daily print
    still carries a weight (eligibility is the trust test, not the fence),
    and an absurd-but-trusted slot print still enters the weight series via
    resolve_slot_prints (no slot-level sigma-fence; R-winsor is the bound)
    — with the replay path reproducing the exact state from the pinned
    artifact block."""
    config = _dyn_test_config(tmp_path)
    ws = _ws()
    payload = compute_day(
        config=config,
        day="2026-08-01",
        snapshot=_snapshot(
            "r",
            {"lead": 30.0, "echo": 10.0, "s1": 10.0, "s2": 9.0, "s3": 11.0},
        ),
        substituted_from=None,
        window_history={"lead": [10.0] * 12},
        fx_records={},
        window_currencies={"lead": "USD"},
        pending_currencies={},
        weight_state=ws,
        prior_slot_prints=resolve_slot_prints(
            {16: _snapshot("p", {"lead": 30.0, "echo": 10.0})},
            config=config,
            day="2026-07-31",
            fx_records={},
        ),
    )
    lead = next(s for s in payload["sources"] if s["source_id"] == "lead")
    assert lead["filter"]["accepted"] is False  # held out of the INDEX
    assert payload["weight_calc"]["weights"]["lead"] == 0.2  # still eligible
    # The absurd-but-trusted prior-day slot print entered the series.
    stamp = date.fromisoformat("2026-07-31").toordinal() * 24 + 16
    assert ws["prices"]["lead"][stamp]["native"] == 30.0
    # And the artifact pinned it for replay.
    pinned = payload["weight_calc"]["slot_prints"]
    assert pinned["date"] == "2026-07-31"
    assert pinned["slots"]["16"]["lead"]["native"] == 30.0
    cli = _load_cli()
    replayed = _ws()
    cli.advance_weight_state_from_published(
        json.loads(json.dumps(payload)), replayed
    )
    assert replayed == ws


def test_non_finite_weight_kills_the_publish_loudly(tmp_path, monkeypatch):
    """The poisoned-weight guard is live code, not decoration: a NaN
    escaping allocation raises with the day and source named, never prices
    a silently-wrong index."""
    import gpu_index.index.weights as weights_module

    config = _dyn_test_config(tmp_path)
    dw_params = calc_params(config)["dynamic_weights"]
    state = _ws()
    state["mode"] = "dynamic"
    monkeypatch.setattr(
        weights_module,
        "allocate_weights",
        lambda *a, **k: (
            {"lead": float("nan")},
            {"degenerate_allocation": None, "capped": []},
        ),
    )
    with pytest.raises(ValueError, match="lead"):
        weights_module.compute_dynamic_weights(
            state,
            day="2026-08-01",
            eligible=["lead"],
            dw_params=dw_params,
            fallback_weights=dw_params["fallback_weights"],
        )


def test_validator_requires_6dp_exact_weights_under_dynamic(tmp_path):
    """Fallback byte-parity is mechanical, not aspirational: a constituent
    weight that does not survive round(w, 6) would fork fallback-mode
    values from the frozen series at the pinned-vector rounding step —
    refused at load."""
    cfg = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    for src in cfg["sources"]:
        if src["source_id"] == "verda":
            src["weight"] = 0.1500000001
        if src["source_id"] == "nebius":
            src["weight"] = 0.1499999999
    path = tmp_path / "sixdp.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(BasketConfigError, match="6 decimal"):
        load_basket_config(path)


class _FakeS3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        import io as _io

        if Key not in self.objects:
            error = Exception("missing")
            error.response = {
                "Error": {"Code": "NoSuchKey"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            }
            raise error
        return {"Body": _io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


def test_cli_switch_day_warns_loudly_but_publishes_green(
    tmp_path, monkeypatch, capsys
):
    """The planned methodology transition end-to-end through main(): the
    switch day emits the SWITCHED ON warning, the firing exits 0 (loud,
    never red — unlike the currency anomalies), and every day publishes
    under the new keyspace with its mode pinned."""
    from datetime import datetime, timezone

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)
    config = _dyn_test_config(tmp_path)
    client = _FakeS3()
    genesis = date.fromisoformat(config["genesis_date"])
    _seed_dyn_days(client, config, 12)
    cli = _load_cli()

    class _StubBucket:
        bucket = "curves"

    monkeypatch.setattr(
        cli.BucketConfig, "from_env", staticmethod(lambda: _StubBucket())
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: client)
    monkeypatch.setattr(cli, "ensure_rates", lambda *a, **k: {})
    now = datetime(2026, 7, 13, 5, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(
        "sys.argv",
        ["compute", "--sync", "--config", config["_config_path"]],
    )
    exit_code = cli.main()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "SWITCHED ON" in out
    modes = []
    for i in range(12):
        day = (genesis + timedelta(days=i)).isoformat()
        stored = json.loads(
            client.objects[f"index/test_dyn/composites/test_dyn_v1/{day}.json"]
        )
        modes.append(stored["weight_calc"]["mode"])
    assert modes[0] == "fallback" and modes[-1] == "dynamic"
    assert sum(1 for i in range(12) if "switched_on" in json.loads(
        client.objects[
            f"index/test_dyn/composites/test_dyn_v1/"
            f"{(genesis + timedelta(days=i)).isoformat()}.json"
        ]
    )["weight_calc"]) == 1


def test_allocate_survives_share_underflow_at_extreme_gamma():
    """Red-team finding: at extreme-but-finite gamma every non-max share
    underflows to exactly 0.0; when the max source caps out, the
    redistribution pool is 0 — the excess must spread deterministically
    over the uncapped, never dark the publish with a ZeroDivisionError."""
    weights, flags = allocate_weights(
        {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0},
        gamma=800.0,
        weight_min=0.025,
        weight_max=0.30,
    )
    assert weights["a"] == pytest.approx(0.30)
    others = [weights[s] for s in "bcde"]
    assert all(w == pytest.approx(0.175) for w in others)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert flags["degenerate_allocation"] is None


def _seed_dyn_days(client, config, n_days):
    genesis = date.fromisoformat(config["genesis_date"])
    levels = _slot_levels(n_days * 4 + 8)
    for i in range(n_days):
        day = (genesis + timedelta(days=i)).isoformat()
        for hour, snap in _day_snapshots(i, levels).items():
            key = (
                f"index/test_dyn/snapshots/{day}/"
                f"slot{hour:02d}-20260701T{hour:02d}1000Z-{i:04d}.json"
            )
            client.objects[key] = json.dumps(snap).encode()


def _wire_dyn_cli(monkeypatch, client, now, config):
    from datetime import datetime as _dt  # noqa: F401

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)
    cli = _load_cli()

    class _StubBucket:
        bucket = "curves"

    monkeypatch.setattr(
        cli.BucketConfig, "from_env", staticmethod(lambda: _StubBucket())
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: client)
    monkeypatch.setattr(cli, "ensure_rates", lambda *a, **k: {})
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(
        "sys.argv", ["compute", "--sync", "--config", config["_config_path"]]
    )
    return cli


def test_fallback_parity_tripwire_warns_red_on_frozen_series_mismatch(
    tmp_path, monkeypatch, capsys
):
    """Mirror-drift armor (red-team finding): fallback mode CLAIMS index
    values byte-identical to the frozen series, but this series replays
    from the CURRENT raw store. A frozen-series artifact for the same day
    carrying a DIFFERENT value must warn loudly and redden the firing —
    while still publishing (the currency-anomaly posture: the frozen
    artifact stands, this series' replay-from-raw is its own record)."""
    from datetime import datetime, timezone

    config = _dyn_test_config(tmp_path)
    client = _FakeS3()
    _seed_dyn_days(client, config, 1)
    client.objects[
        "index/test_dyn/composites/test_dyn_v0/2026-07-01.json"
    ] = json.dumps(
        {"date": "2026-07-01", "index": {"value_usd_gpu_hr": 1.234567}}
    ).encode()
    now = datetime(2026, 7, 2, 5, 0, tzinfo=timezone.utc)
    cli = _wire_dyn_cli(monkeypatch, client, now, config)
    exit_code = cli.main()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FALLBACK PARITY MISMATCH" in out
    assert "test_dyn_v0" in out
    # Still published under the new keyspace — loud, never dark.
    assert (
        "index/test_dyn/composites/test_dyn_v1/2026-07-01.json"
        in client.objects
    )


def test_fallback_parity_tripwire_is_silent_on_matching_values(
    tmp_path, monkeypatch, capsys
):
    from datetime import datetime, timezone

    config = _dyn_test_config(tmp_path)
    # First run: publish day 1 with no frozen series present.
    client = _FakeS3()
    _seed_dyn_days(client, config, 1)
    now = datetime(2026, 7, 2, 5, 0, tzinfo=timezone.utc)
    cli = _wire_dyn_cli(monkeypatch, client, now, config)
    assert cli.main() == 0
    capsys.readouterr()
    published = json.loads(
        client.objects["index/test_dyn/composites/test_dyn_v1/2026-07-01.json"]
    )
    # Second run from scratch with the frozen series carrying the SAME
    # value: the tripwire stays silent and the firing stays green.
    client2 = _FakeS3()
    _seed_dyn_days(client2, config, 1)
    client2.objects[
        "index/test_dyn/composites/test_dyn_v0/2026-07-01.json"
    ] = json.dumps(
        {
            "date": "2026-07-01",
            "index": {
                "value_usd_gpu_hr": published["index"]["value_usd_gpu_hr"]
            },
        }
    ).encode()
    cli = _wire_dyn_cli(monkeypatch, client2, now, config)
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "FALLBACK PARITY" not in out


def test_report_surfaces_the_weighting_posture(tmp_path):
    """Red-team finding: the ops dashboard must show the weighting mode,
    the switch-pending blockers, and degenerate days — the artifact JSON
    and a one-time CLI line are not an ops surface."""
    from datetime import datetime, timezone

    from gpu_index.index.report import render_report

    config = _dyn_test_config(tmp_path)
    payload = compute_day(
        config=config,
        day="2026-07-01",
        snapshot=_snapshot(
            "r0", {"lead": 10.0, "echo": 10.0, "s1": 10.0, "s2": 9.0, "s3": 11.0}
        ),
        substituted_from=None,
        window_history={},
        fx_records={},
        window_currencies={},
        pending_currencies={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date={"2026-07-01": payload},
        now=datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc),
        basket_label="test basket",
    )
    assert "Weighting" in html
    assert "fallback" in html
    assert "switch pending" in html
