# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for the daily composite calc.

Pins: the pure calc rules (every recorded ruling), the ECB FX lane, the
snapshot reader's earliest-key rule, the composite store discipline, and
the CLI's replay/idempotency — including the two golden days under the
calc_v4 median-of-CI-votes statistic: the methodology worked example
($7.675 vote median; the retired $7.6405 weighted / $7.595 unweighted
means ride as diagnostics) and the real first live capture (2026-08-10,
8/8 constituents, EUR converted at 1.1555, vote median $7.475), plus the
same worked example frozen under explicit calc_v3 params ($7.6405
headline, legacy key set). The calc_v5 mint (dynamic_weights)
starts in FALLBACK mode — every golden index value
here is byte-identical to the calc_v4 series; what's new is the pinned
weight_calc audit block and the dynamic_weights calc_params entry.
"""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import date, datetime, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from gpu_index.index.composite import (
    _interquantile_mean,
    _weighted_median,
    _weighted_quantile,
    calc_params,
    compute_day,
    daily_source_observation,
    evaluate_filter,
    median_stddev_composite,
    select_slot_snapshot,
    vote_stddev,
    weighted_composite,
)
from gpu_index.index.config import BasketConfigError, load_basket_config
from gpu_index.index.fx import (
    FxUnavailableError,
    eur_to_usd,
    lookup_rate,
    parse_ecb_rates,
    persist_rates,
)
from gpu_index.common.store import (
    BucketPublishError,
    composite_exists,
    read_day_snapshots,
    upload_composite,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_basket_config(REPO_ROOT / "config" / "index_basket.json")


def _ws():
    """Fresh calc_v5 weight state — REQUIRED by compute_day whenever the
    config sets calc.dynamic_weights (the CLI threads its own across
    days); multi-day tests thread ONE of these like window_history."""
    return {"prices": {}, "vectors": {}, "mode": "fallback"}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    # GITHUB_ACTIONS flips warn() to '::warning::' (no uppercase WARNING) —
    # without the delenv the WARNING string asserts fail ON CI ITSELF.
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)


# ------------------------------------------------------------------ fakes


class _NoSuchKey(Exception):
    def __init__(self):
        super().__init__("missing")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.put_order = []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
        self.put_order.append(Key)

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


def _obs(sku, usd=None, native=None, currency="USD", tier="on-demand", basis=1, implausible=False):
    return {
        "sku": sku,
        "price_usd_gpu_hr": usd,
        "price_native_per_gpu_hr": native if native is not None else usd,
        "currency": currency,
        "tier": tier,
        "gpu_count_basis": basis,
        "raw_value": str(usd if usd is not None else native),
        "raw_unit": "usd_per_gpu_hr",
        "implausible": implausible,
        "notes": "",
    }


def _entry(source_id, observations, status="ok"):
    return {"source_id": source_id, "status": status, "observations": observations}


FX_2026_08_10 = {
    "2026-08-10": {
        "source": "ecb_reference_rate",
        "as_of": "2026-08-10",
        "rates": {"USD": 1.1555},
    }
}


# --------------------------------------------------------------------- fx

ECB_XML = """<?xml version="1.0" encoding="UTF-8"?>
<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01" xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
<gesmes:subject>Reference rates</gesmes:subject>
<Cube>
<Cube time='2026-08-08'><Cube currency='USD' rate='1.1500'/><Cube currency='JPY' rate='183.0'/></Cube>
<Cube time='2026-08-10'><Cube currency='USD' rate='1.1555'/></Cube>
</Cube>
</gesmes:Envelope>
"""


def test_parse_ecb_rates_and_zero_cube_failure():
    rates = parse_ecb_rates(ECB_XML)
    assert rates["2026-08-10"]["USD"] == 1.1555
    assert rates["2026-08-08"]["JPY"] == 183.0
    with pytest.raises(RuntimeError, match="zero dated"):
        parse_ecb_rates("<Envelope/>")


def test_fx_lookup_walks_back_over_non_publication_days():
    stored = {
        "2026-08-08": {"as_of": "2026-08-08", "rates": {"USD": 1.15}},
    }
    # Sunday 08-10: walk back to Friday 08-08, and SAY so in fx_as_of.
    rate, as_of = lookup_rate(stored, "2026-08-10", "USD")
    assert (rate, as_of) == (1.15, "2026-08-08")
    usd, block = eur_to_usd(7.5, stored, "2026-08-10")
    assert usd == pytest.approx(8.625)
    assert block == {
        "fx_rate": 1.15,
        "fx_source": "ecb_reference_rate",
        "fx_as_of": "2026-08-08",
    }


def test_fx_fails_closed_past_staleness_window():
    stored = {"2026-07-01": {"as_of": "2026-07-01", "rates": {"USD": 1.1}}}
    with pytest.raises(FxUnavailableError):
        lookup_rate(stored, "2026-08-10", "USD", max_staleness_days=7)


def test_persist_rates_is_append_only_first_write_wins():
    client = FakeS3()
    persist_rates(
        client, "curves", prefix="index/b300_basket", rates_by_day={"2026-08-10": {"USD": 1.1555}}
    )
    persist_rates(
        client, "curves", prefix="index/b300_basket", rates_by_day={"2026-08-10": {"USD": 9.99}}
    )
    stored = json.loads(client.objects["index/b300_basket/fx/ecb-2026-08-10.json"])
    assert stored["rates"]["USD"] == 1.1555  # a published rate never changes


# ------------------------------------------------------------- slot choice


def test_select_slot_canonical_wins_and_open_window_waits():
    snaps = {16: {"run_id": "a"}, 10: {"run_id": "b"}}
    assert select_slot_snapshot(snaps, canonical_hour=16, window_closed=False) == (
        {"run_id": "a"},
        None,
    )
    assert (
        select_slot_snapshot({10: {"run_id": "b"}}, canonical_hour=16, window_closed=False)
        is None
    )


def test_select_slot_promotes_nearest_tie_breaks_later():
    snaps = {10: {"run_id": "ten"}, 22: {"run_id": "twentytwo"}}
    payload, promoted = select_slot_snapshot(
        snaps, canonical_hour=16, window_closed=True
    )
    assert (payload["run_id"], promoted) == ("twentytwo", 22)
    payload, promoted = select_slot_snapshot(
        {4: {"run_id": "four"}}, canonical_hour=16, window_closed=True
    )
    assert (payload["run_id"], promoted) == ("four", 4)


# ------------------------------------------------------- observation choice


def test_daily_observation_excludes_interruptible_and_implausible():
    entry = _entry(
        "verda",
        [
            _obs("B300", usd=7.5),
            _obs("B300", usd=3.75, tier="spot"),
            _obs("B300", usd=4.30, tier="preemptible"),
            _obs("B300", usd=900.0, implausible=True),
            _obs("B200", usd=6.11),
        ],
    )
    chosen = daily_source_observation(
        entry, sku="B300", day="2026-08-10", fx_records={}, interruptible_tiers=("spot", "preemptible")
    )
    assert chosen["usd_per_gpu_hr"] == 7.5  # spot/preemptible/implausible never win
    assert chosen["n_eligible_prints"] == 1


def test_daily_observation_committed_tier_wins_when_cheaper():
    entry = _entry(
        "latitude",
        [_obs("B300", usd=16.0), _obs("B300", usd=8.0, tier="monthly-commit")],
    )
    chosen = daily_source_observation(
        entry, sku="B300", day="2026-08-10", fx_records={}, interruptible_tiers=("spot", "preemptible")
    )
    assert (chosen["usd_per_gpu_hr"], chosen["tier"]) == (8.0, "monthly-commit")


def test_daily_observation_converts_eur_before_taking_the_minimum():
    entry = _entry(
        "scaleway",
        [
            _obs("B300", native=7.50, usd=None, currency="EUR", basis=8),
            _obs("B300", native=8.52, usd=None, currency="EUR", basis=4),
        ],
    )
    chosen = daily_source_observation(
        entry,
        sku="B300",
        day="2026-08-10",
        fx_records=FX_2026_08_10,
        interruptible_tiers=(),
    )
    assert chosen["usd_per_gpu_hr"] == pytest.approx(7.50 * 1.1555)
    assert chosen["fx_rate"] == 1.1555
    assert chosen["fx_as_of"] == "2026-08-10"


def test_daily_observation_fx_missing_is_held_out_not_guessed():
    entry = _entry(
        "scaleway", [_obs("B300", native=7.50, usd=None, currency="EUR", basis=8)]
    )
    chosen = daily_source_observation(
        entry, sku="B300", day="2026-08-10", fx_records={}, interruptible_tiers=()
    )
    assert chosen["fx_unavailable"] is True


def test_daily_observation_unknown_currency_never_prices():
    entry = _entry(
        "verda", [_obs("B300", native=7.50, usd=None, currency="UNKNOWN")]
    )
    assert (
        daily_source_observation(
            entry, sku="B300", day="2026-08-10", fx_records=FX_2026_08_10, interruptible_tiers=()
        )
        is None
    )


# ----------------------------------------------------------------- filter


def test_filter_warmup_accepts_unfiltered():
    verdict = evaluate_filter([7.5] * 9, 20.0, warmup=10)
    assert verdict == {"accepted": True, "unfiltered": True, "n_history": 9}


def test_filter_accepts_within_band_and_holds_out_beyond():
    history = [7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 7.1, 7.3, 7.5, 7.7, 7.9]
    ok = evaluate_filter(history, 7.5, warmup=10)
    assert ok["accepted"] is True and not ok.get("unfiltered")
    held = evaluate_filter(history, 12.0, warmup=10)
    assert held["accepted"] is False
    assert held["deviation"] > 2.5 * held["sigma"]


def test_filter_sigma_zero_repricing_costs_days_then_adapts():
    """Step-function list prices at sigma_floor 0 (legacy default): a big
    reprice from a constant series is held out (flagged sigma_zero), enters
    the window per the §6.4 rule, and gets accepted once the window has
    absorbed enough of the new level. calc_v3 retired the sigma_zero
    accept-iff-deviation==0 SPECIAL CASE; at floor 0 the general band test
    (deviation <= 0) is the identical rule, so legacy series replay
    bit-identically — which is exactly what this test now proves."""
    history = [7.5] * 11
    day1 = evaluate_filter(history, 9.0, warmup=10)
    assert (day1["accepted"], day1["sigma_zero"]) == (False, True)
    history = history + [9.0]  # held-out prints still enter the window
    day2 = evaluate_filter(history, 9.0, warmup=10)
    assert day2["accepted"] is False  # window still dominated by 7.5
    history = history + [9.0]
    day3 = evaluate_filter(history, 9.0, warmup=10)
    assert day3["accepted"] is True  # adapted
    same = evaluate_filter([7.5] * 11, 7.5, warmup=10)
    assert (same["accepted"], same.get("sigma_zero")) == (True, True)
    # And the legacy verdict SHAPE is untouched (no currency, no fence
    # bounds) — a frozen-methodology replay must keep its published bytes.
    assert set(day1) == {
        "accepted", "unfiltered", "mu", "sigma", "deviation", "n_history",
        "sigma_zero",
    }


def test_filter_sigma_floor_supersedes_the_sigma_zero_rule():
    """R-floor (calc_v3): band = sigma * max(sd, sigma_floor). A constant
    window no longer arms hair-trigger rejection — a small reprice inside
    the floored band is ACCEPTED (the retired rule held it out), while raw
    sigma stays reported and sigma_zero stays informational."""
    small_reprice = evaluate_filter(
        [7.5] * 11, 7.55, warmup=10, sigma_floor=0.05, currency="EUR"
    )
    assert small_reprice["accepted"] is True  # dev 0.05 <= 2.5*0.05
    assert small_reprice["sigma"] == 0.0  # raw sigma, floor shows in band
    assert small_reprice["sigma_zero"] is True  # informational only now
    assert small_reprice["band"] == pytest.approx(0.125)
    assert (small_reprice["lo"], small_reprice["hi"]) == (7.375, 7.625)
    assert small_reprice["currency"] == "EUR"
    # Under the retired rule (floor 0) the same print was held out.
    legacy = evaluate_filter([7.5] * 11, 7.55, warmup=10)
    assert legacy["accepted"] is False


def test_filter_floor_golden_verdicts_from_2026_08_20():
    """The calc_v3 mint's two worked examples, encoded exactly.

    1. Ten identical 7.50 EUR native prints + one more identical print:
       ACCEPTED under the floor (band = 2.5 * 0.05 = 0.125 native,
       deviation 0) — the scaleway re-acceptance.
    2. A USD source at mu 6.052 / sigma 0.2134 printing 6.60 is still
       HELD OUT: deviation 0.548 > 2.5 * max(0.2134, 0.05) = 0.5335 —
       massedcompute's genuine 5.95 -> 6.60 repricing sits ~1.5 cents over
       even the floored band (and self-heals via the trailing window).
    """
    scaleway = evaluate_filter(
        [7.5] * 10, 7.5, warmup=10, sigma_floor=0.05, currency="EUR"
    )
    assert scaleway["accepted"] is True
    assert scaleway["deviation"] == 0.0
    assert scaleway["band"] == pytest.approx(2.5 * 0.05)
    # Synthetic 20-print window with EXACTLY the 2026-08-20 prod moments
    # (18 low prints + 2 high prints solve mean=6.052, pstdev=0.2134).
    window = [5.980866666666667] * 18 + [6.6922] * 2
    import statistics as _st

    assert round(_st.mean(window), 6) == 6.052
    assert round(_st.pstdev(window), 6) == 0.2134
    massed = evaluate_filter(
        window, 6.60, warmup=10, sigma_floor=0.05, currency="USD"
    )
    assert massed["accepted"] is False  # a genuine repricing still filters
    assert massed["mu"] == 6.052
    assert massed["sigma"] == 0.2134
    assert massed["deviation"] == 0.548
    assert massed["band"] == pytest.approx(0.5335)  # floor did NOT bind
    assert massed["currency"] == "USD"


def test_filter_floor_emits_fence_without_currency_key():
    """The fence bounds surface whenever the floor changed the band — even
    with no ``currency`` passed (USD-terms mode). The currency key itself
    stays absent: only recorded-currency verdicts name one."""
    verdict = evaluate_filter([7.5] * 11, 7.55, warmup=10, sigma_floor=0.05)
    assert verdict["accepted"] is True  # dev 0.05 <= 2.5 * 0.05
    assert verdict["band"] == pytest.approx(0.125)
    assert (verdict["lo"], verdict["hi"]) == (7.375, 7.625)
    assert "currency" not in verdict


# --------------------------------------------------------------- composite


def _worked_example_snapshot():
    prints = {
        "verda": 7.50,
        "nebius": 7.85,
        "hyperstack": 7.40,
        "scaleway": 8.54,  # an aggregator table treated this as a USD print
        "runpod": 7.39,
        "vast": 8.13,
        "massedcompute": 5.95,
        "latitude": 8.00,
    }
    return {
        "run_id": "worked-day-1",
        "late_fill": False,
        "sources": [_entry(sid, [_obs("B300", usd=p)]) for sid, p in prints.items()],
    }


def test_golden_worked_example():
    """Worked-example day-1 prints under calc_v4, hand-verified.

    All eight sources are day-one (empty windows), so every vote CI is the
    0.05 floor (this snapshot records scaleway as a USD print). 24 votes,
    total weight 3.0; cumulative weights are decimal-EXACT. The hand-walked
    ladder: the median target 1.5 lands exactly on verda's high vote 7.55
    (cum ...7.45→1.05, 7.45→1.20, 7.50→1.35, 7.55→1.50) so the index
    averages the straddling votes (7.55 + 7.80)/2 = 7.675; the p25 target
    0.75 is crossed inside hyperstack's mid vote 7.40 (cum 0.65→0.80) so
    p25 = 7.40; the p75 target 2.25 lands exactly on latitude's high vote
    8.05 (cum 2.15→2.25) so p75 averages with vast's low vote
    (8.05 + 8.08)/2 = 8.065. Confidence = max(7.675−7.40, 8.065−7.675) =
    0.39 — wide, because the day-one basket genuinely disagrees. The
    retired weighted mean (7.6405) and unweighted mean (7.595) ride as
    diagnostics. Under calc_v5 the day computes in dynamic-weights
    FALLBACK mode: every value above is unchanged, and the new
    weight_calc block pins the config 2.1 weights over the eligible
    set."""
    history: dict = {}
    payload = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=_worked_example_snapshot(),
        substituted_from=None,
        window_history=history,
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    index = payload["index"]
    assert index["value_usd_gpu_hr"] == pytest.approx(7.675, abs=1e-6)
    assert index["statistic"] == "median_ci_votes"
    assert index["vote_p25_usd_gpu_hr"] == pytest.approx(7.40, abs=1e-6)
    assert index["vote_p75_usd_gpu_hr"] == pytest.approx(8.065, abs=1e-6)
    assert index["confidence_usd_gpu_hr"] == pytest.approx(0.39, abs=1e-6)
    assert index["weighted_mean_usd_gpu_hr"] == pytest.approx(7.6405, abs=1e-6)
    assert index["unweighted_mean_usd_gpu_hr"] == pytest.approx(7.595, abs=1e-6)
    assert index["sources_used_count"] == 8
    assert payload["basket_dark"] is False
    # Initialization gap (methodology worked example): Massed at ~22% below the mean is
    # exactly the print the warm-up manual_verify flag exists for. Under
    # calc_v4 its day-one vote CI is the bare floor — the tight-warm-up
    # default recorded for ratification at the calc_v4 mint.
    by_id = {s["source_id"]: s for s in payload["sources"]}
    assert by_id["massedcompute"]["filter"]["manual_verify"] is True
    assert by_id["massedcompute"]["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.05,
    }
    assert all(
        s["filter"]["unfiltered"] for s in payload["sources"] if s.get("filter")
    )
    # Every day-1 print entered its source's window.
    assert all(len(v) == 1 for v in history.values()) and len(history) == 8
    # calc_v5 fallback mode: the pinned weight vector IS the config opening
    # 2.1 weights over the day's eligible set (all eight printed).
    assert payload["weight_calc"]["mode"] == "fallback"
    assert payload["weight_calc"]["weights"] == {
        "hyperstack": 0.15,
        "latitude": 0.1,
        "massedcompute": 0.1,
        "nebius": 0.15,
        "runpod": 0.1,
        "scaleway": 0.15,
        "vast": 0.1,
        "verda": 0.15,
    }


def test_golden_worked_example_frozen_v3():
    """The SAME prints under explicit calc_v3 params must keep the frozen
    series' headline (legacy weighted mean 7.6405) and its exact index
    key set — no statistic, no confidence, no vote blocks, and (since the
    calc_v5 mint) no weight_calc. A frozen replay that grew (or repriced)
    keys would fork published artifact bytes without a mint."""
    cfg = json.loads(json.dumps(CONFIG))
    cfg["calc"]["methodology_id"] = "annex_a_v0_2_calc_v3"
    del cfg["calc"]["composite_statistic"]
    del cfg["calc"]["dynamic_weights"]  # calc_v3 predates the v5 mint
    payload = compute_day(
        config=cfg,
        day="2026-08-10",
        snapshot=_worked_example_snapshot(),
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records={},
    )
    index = payload["index"]
    assert index["value_usd_gpu_hr"] == pytest.approx(7.6405, abs=1e-6)
    assert index["unweighted_mean_usd_gpu_hr"] == pytest.approx(7.595, abs=1e-6)
    assert set(index.keys()) == {
        "value_usd_gpu_hr",
        "unweighted_mean_usd_gpu_hr",
        "renormalized_weights",
        "sources_used_count",
    }
    assert "composite_statistic" not in payload["calc_params"]
    assert all("vote" not in s for s in payload["sources"])
    # Frozen configs do zero weight work: no params entry, no audit block
    # (and compute_day never demanded a weight_state above).
    assert "dynamic_weights" not in payload["calc_params"]
    assert "weight_calc" not in payload


def _live_first_capture_snapshot():
    return {
        "run_id": "20260810T211029Z-b128",
        "late_fill": True,
        "sources": [
            _entry("verda", [_obs("B300", usd=7.50), _obs("B300", usd=3.75, tier="spot"), _obs("B200", usd=6.11)]),
            _entry("nebius", [_obs("B300", usd=7.85), _obs("B300", usd=4.30, tier="preemptible"), _obs("B200", usd=7.15)]),
            _entry("hyperstack", [_obs("B300", usd=7.40), _obs("B200", usd=6.00)]),
            _entry(
                "scaleway",
                [
                    _obs("B300", native=9.48, usd=None, currency="EUR", basis=2),
                    _obs("B300", native=8.52, usd=None, currency="EUR", basis=4),
                    _obs("B300", native=7.50, usd=None, currency="EUR", basis=8),
                ],
            ),
            _entry("runpod", [_obs("B300", usd=7.39), _obs("B200", usd=6.79)]),
            _entry("vast", [_obs("B300", usd=6.876, basis=2), _obs("B300", usd=6.8755, basis=4), _obs("B200", usd=5.5013, basis=8)]),
            _entry("massedcompute", [_obs("B300", usd=5.95, basis=8)]),
            _entry("latitude", [_obs("B300", usd=16.0, basis=8), _obs("B300", usd=8.0, tier="monthly-commit", basis=8)]),
            _entry("e2e", [_obs("B200", usd=6.99)]),
            _entry("shadeform", [_obs("B200", usd=3.74, basis=8), _obs("B200", usd=6.99, basis=1)]),
        ],
    }


def test_golden_first_live_capture():
    """The real 2026-08-10 capture under calc_v4: EUR at the day's actual
    ECB rate (1.1555). Hand-verified ladder: every day-one CI is the 0.05
    floor, except scaleway whose floor converts at the print's own rate —
    0.05 EUR x 1.1555 = 0.057775 USD. The median target 1.5 lands exactly
    on the boundary between hyperstack's high vote and verda's low vote
    (both 7.45, cum 1.35→1.50) so the index averages with verda's mid vote:
    (7.45 + 7.50)/2 = 7.475. p25 = 7.35 (target 0.75 crossed inside
    hyperstack's low vote, cum 0.70→0.85); p75 target 2.25 lands exactly on
    nebius's high vote 7.90 (cum 2.10→2.25) → (7.90 + 7.95)/2 = 7.925.
    Confidence = max(7.475−7.35, 7.925−7.475) = 0.45. The retired weighted
    mean — 7.533988, the calc_v2/v3 series' published day-1 value — rides
    as a diagnostic. calc_v5 fallback-mode parity: all values unchanged;
    weight_calc pins the config 2.1 weights over the eligible eight."""
    payload = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=_live_first_capture_snapshot(),
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    index = payload["index"]
    scaleway = next(s for s in payload["sources"] if s["source_id"] == "scaleway")
    assert scaleway["chosen"]["usd_per_gpu_hr"] == pytest.approx(8.66625)
    assert scaleway["chosen"]["gpu_count_basis"] == 8  # cheapest config won
    assert scaleway["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.057775,
    }
    assert index["value_usd_gpu_hr"] == pytest.approx(7.475, abs=1e-6)
    assert index["vote_p25_usd_gpu_hr"] == pytest.approx(7.35, abs=1e-6)
    assert index["vote_p75_usd_gpu_hr"] == pytest.approx(7.925, abs=1e-6)
    assert index["confidence_usd_gpu_hr"] == pytest.approx(0.45, abs=1e-6)
    old_weighted_mean = 0.15 * (7.50 + 7.85 + 7.40 + 8.66625) + 0.10 * (
        7.39 + 6.8755 + 5.95 + 8.0
    )
    assert index["weighted_mean_usd_gpu_hr"] == pytest.approx(
        old_weighted_mean, abs=1e-6
    )
    assert index["sources_used_count"] == 8
    pool = payload["fallback_pool"]
    pool_by_id = {p["source_id"]: p for p in pool["sources"]}
    assert set(pool_by_id) == {"nebius", "e2e", "shadeform"}
    assert pool_by_id["nebius"]["chosen"]["usd_per_gpu_hr"] == 7.15
    assert pool_by_id["shadeform"]["chosen"]["usd_per_gpu_hr"] == 3.74
    assert pool["mean_usd_gpu_hr"] == pytest.approx((7.15 + 6.99 + 3.74) / 3)
    assert payload["snapshot_late_fill"] is True
    # calc_v5 fallback mode: config 2.1 weights over the day's eligible set.
    assert payload["weight_calc"]["mode"] == "fallback"
    assert payload["weight_calc"]["weights"] == {
        "hyperstack": 0.15,
        "latitude": 0.1,
        "massedcompute": 0.1,
        "nebius": 0.15,
        "runpod": 0.1,
        "scaleway": 0.15,
        "vast": 0.1,
        "verda": 0.15,
    }


def test_compute_day_percent_floor_end_to_end_with_eur_seat():
    """Ruling 2026-08-26 through the DAILY engine: compute_day resolves
    filter_sigma_floor_pct, the fence floors at 3% of the trailing-window
    MEAN, votes floor at 3% of each print's OWN filter-terms price
    (converted at the print's own rate for the EUR seat), and calc_params
    embeds the pct key only. The massedcompute contrast pins the two
    semantics apart end-to-end: a 6.15 print off a frozen 5.95 window is
    ACCEPTED at 3% (band 3.0 * 0.1785 = 0.5355 >= dev 0.20) but HELD OUT
    under the absolute 0.05 config (band 0.15 < 0.20) — and its vote conf
    is 0.1845 (3% of the 6.15 PRINT), not 0.1785 (3% of the 5.95 mean),
    pinning the vote's print-anchor."""
    cfg = json.loads(json.dumps(CONFIG))
    del cfg["calc"]["filter_sigma_floor"]
    cfg["calc"]["filter_sigma_floor_pct"] = 3.0
    cfg["calc"]["vote_sigma_floor_pct"] = 3.0  # floor split, both 3%
    cfg["calc"]["methodology_id"] = "annex_a_v0_2_calc_v_test_pct"
    snapshot = {
        "run_id": "pct-e2e",
        "late_fill": False,
        "sources": [
            _entry("verda", [_obs("B300", usd=7.50)]),
            _entry("nebius", [_obs("B300", usd=7.85)]),
            _entry("hyperstack", [_obs("B300", usd=7.40)]),
            _entry("runpod", [_obs("B300", usd=7.39)]),
            _entry(
                "scaleway",
                [_obs("B300", native=7.50, usd=None, currency="EUR")],
            ),
            _entry("massedcompute", [_obs("B300", usd=6.15)]),
        ],
    }

    def _run(config):
        return compute_day(
            config=config,
            day="2026-08-10",
            snapshot=json.loads(json.dumps(snapshot)),
            substituted_from=None,
            window_history={"massedcompute": [5.95] * 11},
            window_currencies={"massedcompute": "USD"},
            fx_records=FX_2026_08_10,
            weight_state=_ws(),
            prior_slot_prints={},
        )

    payload = _run(cfg)
    assert payload["calc_params"]["filter_sigma_floor_pct"] == 3.0
    assert payload["calc_params"]["vote_sigma_floor_pct"] == 3.0
    assert "filter_sigma_floor" not in payload["calc_params"]
    by_id = {s["source_id"]: s for s in payload["sources"]}
    # Fence: 3% of the window MEAN, symmetric around mu.
    massed = by_id["massedcompute"]["filter"]
    assert massed["accepted"] is True
    assert massed["band"] == pytest.approx(3.0 * 0.1785)
    assert (massed["lo"], massed["hi"]) == (
        pytest.approx(5.95 - 0.5355),
        pytest.approx(5.95 + 0.5355),
    )
    # Vote: 3% of the PRINT (6.15 -> 0.1845), not of the mean (0.1785).
    assert by_id["massedcompute"]["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": pytest.approx(0.1845),
    }
    # USD day-one seat: conf = 3% of its own quote.
    assert by_id["verda"]["vote"]["conf_usd_gpu_hr"] == pytest.approx(0.225)
    # EUR seat: 3% of the NATIVE print, converted at the print's own rate.
    # Pinned to the engine's exact double walk: 7.5 * (3.0/100.0) is
    # 0.22499999999999998, x 1.1555 = 0.2599874999... -> 0.259987 at the
    # 6dp wire round (NOT the decimal 0.2599875 -> 0.259988) — published
    # bytes follow IEEE doubles, and this pin keeps that deliberate.
    assert by_id["scaleway"]["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.259987,
    }
    # The SAME inputs under the absolute-floor config hold the massed
    # print out — the swap is a genuinely different rule, end to end.
    absolute = _run(CONFIG)
    massed_abs = {s["source_id"]: s for s in absolute["sources"]}[
        "massedcompute"
    ]
    assert massed_abs["filter"]["accepted"] is False
    assert massed_abs["filter"]["band"] == pytest.approx(0.15)


def test_weighted_composite_renormalizes_and_dark_day():
    result = weighted_composite([("a", 0.15, 8.0), ("b", 0.10, 6.0)])
    assert result["value_usd_gpu_hr"] == pytest.approx((0.15 * 8 + 0.10 * 6) / 0.25)
    assert sum(result["renormalized_weights"].values()) == pytest.approx(1.0)
    assert weighted_composite([]) is None

    dark = compute_day(
        config=CONFIG,
        day="2026-08-11",
        snapshot={"run_id": "x", "late_fill": False, "sources": []},
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    assert dark["basket_dark"] is True and dark["index"] is None

    missed = compute_day(
        config=CONFIG,
        day="2026-08-12",
        snapshot=None,
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    assert missed["day_missed"] is True and missed["basket_dark"] is True


# ---------------------------------------------- median of CI votes (calc_v4)


def test_weighted_quantile_matches_weighted_median_at_half():
    """The vote median must inherit the ONE established convention (the
    vast order-book statistic's): at q=1/2 with exactly-representable
    weights the two functions agree bit-for-bit — including the exact
    half-boundary averaging case."""
    cases = [
        [(5.0, 1.0)],
        [(5.0, 1.0), (7.0, 1.0)],
        [(3.0, 1.0), (5.0, 1.0), (9.0, 1.0)],
        [(4.0, 2.0), (8.0, 2.0)],  # exact boundary -> 6.0
        [(4.0, 1.0), (6.0, 4.0), (9.0, 1.0)],
        [(1.0, 3.0), (2.0, 1.0), (3.0, 2.0), (10.0, 2.0)],
    ]
    for pairs in cases:
        assert _weighted_quantile(pairs, Fraction(1, 2)) == _weighted_median(
            pairs
        )


def test_weighted_quantile_accumulates_exact_decimal_weights():
    """Decimal §2.1 weights land exactly on quantile boundaries in paper
    arithmetic; float accumulation misses them by one ulp and would return
    the NEXT vote instead of averaging — whole cents of index on float
    dust. Four 0.15s and four 0.10s sum to exactly half of the 24-vote
    total at the 12th vote, so the median must AVERAGE the straddling
    votes, never take a side."""
    pairs = [(float(i), w) for i, w in enumerate(
        [0.10, 0.10, 0.10, 0.10, 0.15, 0.10, 0.15, 0.10, 0.15, 0.15, 0.15,
         0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15,
         0.15, 0.15]
    )]
    # cum through value 11.0 = 4*0.10 + 0.15 + 0.10 + 0.15 + 0.10 + 4*0.15
    # = 1.50 exactly = half of W (3.0): boundary -> average(11.0, 12.0).
    assert _weighted_quantile(pairs, Fraction(1, 2)) == 11.5
    with pytest.raises(ValueError):
        _weighted_quantile(pairs, Fraction(0))
    with pytest.raises(ValueError):
        _weighted_quantile(pairs, Fraction(1))
    with pytest.raises(ValueError):
        _weighted_quantile([], Fraction(1, 2))


def test_vote_confidence_floors_and_converts():
    """The CI is max(window pstdev, floor) in filter terms, converted at
    the print's own fx rate. A frozen source (sigma 0) gets exactly the
    floor — staleness is not conviction — and a genuinely volatile source
    keeps its real sigma."""
    frozen = vote_stddev([7.5, 7.5, 7.5], sigma_floor=0.05, fx_factor=1.0)
    assert frozen == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.05,
    }
    # pstdev([6.0, 7.0]) = 0.5 > floor: the real sigma wins.
    volatile = vote_stddev([6.0, 7.0], sigma_floor=0.05, fx_factor=1.0)
    assert volatile == {
        "sigma": 0.5,
        "sigma_floored": False,
        "conf_usd_gpu_hr": 0.5,
    }
    # EUR window: floor applies in RECORDED currency, then converts.
    eur = vote_stddev([7.5, 7.5], sigma_floor=0.05, fx_factor=1.1555)
    assert eur["conf_usd_gpu_hr"] == pytest.approx(0.057775)
    # Day-one / single-entry windows have no dispersion to measure.
    assert vote_stddev([], sigma_floor=0.05, fx_factor=1.0)[
        "conf_usd_gpu_hr"
    ] == 0.05
    assert vote_stddev([9.9], sigma_floor=0.05, fx_factor=1.0)[
        "sigma"
    ] == 0.0


def test_percent_floor_fence_anchors_on_the_window_mean():
    """Percent-form floor (ruling 2026-08-26): band = sigma * max(sd,
    mu * pct/100) — the floor rides the window's own price level. A
    frozen 7.50 window at 3% floors the band at 0.225 native (vs the
    absolute rule's 0.05), and the fence bounds stay symmetric around mu
    because the floor anchors on mu, never on the print under judgment."""
    frozen = evaluate_filter(
        [7.5] * 11, 7.9, warmup=10, sigma_floor_pct=3.0, currency="EUR"
    )
    # dev 0.40 > 2.5 * (7.5 * 0.03) = 0.5625? No: 0.40 <= 0.5625 accepts.
    assert frozen["accepted"] is True
    assert frozen["band"] == pytest.approx(2.5 * 7.5 * 0.03)
    assert (frozen["lo"], frozen["hi"]) == (
        pytest.approx(7.5 - 0.5625),
        pytest.approx(7.5 + 0.5625),
    )
    assert frozen["sigma"] == 0.0  # raw sigma, floor shows in band
    assert frozen["sigma_zero"] is True
    # The same window under the ABSOLUTE floor held that print out —
    # the two semantics are genuinely different rules, not a rescale.
    absolute = evaluate_filter(
        [7.5] * 11, 7.9, warmup=10, sigma_floor=0.05, currency="EUR"
    )
    assert absolute["accepted"] is False
    # pct 0 = no floor in the band (max(sd, 0)): ACCEPTANCE matches the
    # legacy defaults, but the byte shape is CONFIG-keyed (hardening
    # 2026-08-27): any pct-mode verdict records the fence bounds, so an
    # artifact's key set can never depend on the window data. Legacy
    # (no pct key, no currency, floor 0) keeps its published shape.
    legacy_shape = evaluate_filter([7.5] * 11, 7.55, warmup=10)
    assert not {"band", "lo", "hi"} & set(legacy_shape)
    pct_zero = evaluate_filter(
        [7.5] * 11, 7.55, warmup=10, sigma_floor_pct=0.0
    )
    assert pct_zero["accepted"] is legacy_shape["accepted"]
    assert (pct_zero["band"], pct_zero["lo"], pct_zero["hi"]) == (
        0.0, 7.5, 7.5,
    )
    assert {
        k: v for k, v in pct_zero.items() if k not in ("band", "lo", "hi")
    } == legacy_shape
    # USD terms record the fence bounds too whenever the floor binds a
    # positive value (the non-legacy-fence rule, unchanged).
    usd = evaluate_filter(
        [2.0] * 11, 2.03, warmup=10, sigma_floor_pct=3.0
    )
    assert usd["accepted"] is True  # dev 0.03 <= 2.5 * 0.06
    assert usd["band"] == pytest.approx(2.5 * 0.06)


def test_percent_floor_fence_refuses_non_positive_window_mean():
    """Hardening 2026-08-27 (fail-closed): floor = mu * pct/100 disarms
    exactly when the window is degenerate — a non-positive window mean is
    garbage upstream, so percent mode refuses it loudly instead of
    running an unfloored fence. Warm-up returns before mu exists, and
    legacy absolute mode never anchors on mu — both untouched."""
    for window in ([0.0] * 11, [-1.0] * 11, [-1.0, 1.0] * 6):
        with pytest.raises(
            ValueError, match="positive\\s+trailing-window mean"
        ):
            evaluate_filter(window, 2.0, warmup=10, sigma_floor_pct=3.0)
    # Warm-up: shorter than warmup returns before mu is ever computed.
    warm = evaluate_filter([0.0] * 3, 2.0, warmup=10, sigma_floor_pct=3.0)
    assert warm == {"accepted": True, "unfiltered": True, "n_history": 3}
    # Legacy absolute mode on the same degenerate window: untouched.
    legacy = evaluate_filter([0.0] * 11, 2.0, warmup=10, sigma_floor=0.05)
    assert legacy["accepted"] is False
    assert legacy["band"] == pytest.approx(2.5 * 0.05)


def test_percent_floor_vote_anchors_on_the_prints_own_price():
    """Percent-form vote floor: a source cannot claim conviction tighter
    than pct of its OWN quote — the anchor is the print the vote is cast
    at (filter terms), so day-one votes (empty tail, no mean) are still
    well-defined. Converts at the print's fx rate like the absolute
    floor."""
    frozen = vote_stddev(
        [7.5, 7.5, 7.5],
        sigma_floor=0.0,
        sigma_floor_pct=3.0,
        filter_price=7.5,
        fx_factor=1.0,
    )
    assert frozen == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": pytest.approx(0.225),
    }
    # A genuinely volatile window keeps its real sigma once it exceeds
    # the percent floor (pstdev([6,7]) = 0.5 > 0.03 * 6.8).
    volatile = vote_stddev(
        [6.0, 7.0],
        sigma_floor=0.0,
        sigma_floor_pct=3.0,
        filter_price=6.8,
        fx_factor=1.0,
    )
    assert volatile["sigma_floored"] is False
    assert volatile["conf_usd_gpu_hr"] == 0.5
    # Day-one: empty tail, sigma 0 — the floor IS the interval, priced
    # off the print itself, converted at the print's own rate.
    day_one = vote_stddev(
        [],
        sigma_floor=0.0,
        sigma_floor_pct=3.0,
        filter_price=7.5,
        fx_factor=1.1555,
    )
    assert day_one["conf_usd_gpu_hr"] == pytest.approx(
        round(7.5 * 0.03 * 1.1555, 6)
    )
    # Percent mode without the anchor is a coding error, refused loudly.
    with pytest.raises(ValueError, match="requires filter_price"):
        vote_stddev(
            [7.5], sigma_floor=0.0, sigma_floor_pct=3.0, fx_factor=1.0
        )


def test_percent_floor_vote_refuses_non_positive_filter_price():
    """Hardening 2026-08-27: filter_price <= 0 in percent mode would
    resolve floor 0 and — over a frozen tail — mint a conf-0 vote,
    infinite conviction off a garbage print (the floor>0 invariant the
    median_ci_votes gate is built on). Refused loudly; legacy absolute
    mode has no price anchor and is untouched."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="filter_price > 0"):
            vote_stddev(
                [0.0, 0.0],
                sigma_floor=0.0,
                sigma_floor_pct=3.0,
                filter_price=bad,
                fx_factor=1.0,
            )
    # Legacy absolute mode: same inputs, no anchor, no new refusal.
    legacy = vote_stddev([0.0, 0.0], sigma_floor=0.05, fx_factor=1.0)
    assert legacy == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.05,
    }


def test_config_percent_floor_validation_and_conditional_embed():
    """Load rules for the pct keys (floor split, founder ruling
    2026-08-27): the fence pct is a number >= 0, mutually exclusive with
    the absolute key, and FENCE-ONLY — the median_ci_votes gate in the
    percent regime is satisfied by vote_sigma_floor_pct > 0 alone (no
    silent fallback to the fence floor), while the legacy absolute regime
    keeps satisfying it via filter_sigma_floor > 0 untouched. The vote key
    refuses without votes and alongside the absolute floor. Both pct keys
    ride calc_params presence-gated (absent stays absent — the D2
    discipline)."""
    base = json.loads(
        (REPO_ROOT / "config" / "index_basket.json").read_text()
    )
    calc_no_floor = {
        k: v for k, v in base["calc"].items() if k != "filter_sigma_floor"
    }
    pct_pair = {
        "filter_sigma_floor_pct": 3.0,
        "vote_sigma_floor_pct": 3.0,
    }

    def _load(calc, tmp_path, name):
        p = tmp_path / name
        p.write_text(json.dumps({**base, "calc": calc}))
        return load_basket_config(p)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # The pct pair satisfies the median_ci_votes gate and embeds
        # conditionally — both keys, floats, absolute absent.
        cfg = _load({**calc_no_floor, **pct_pair}, tmp_path, "pct.json")
        params = calc_params(cfg)
        assert params["filter_sigma_floor_pct"] == 3.0
        assert params["vote_sigma_floor_pct"] == 3.0
        assert "filter_sigma_floor" not in params
        # The legacy config's params are byte-untouched by this change.
        legacy_params = calc_params(
            _load(dict(base["calc"]), tmp_path, "legacy.json")
        )
        assert legacy_params["filter_sigma_floor"] == 0.05
        assert "filter_sigma_floor_pct" not in legacy_params
        assert "vote_sigma_floor_pct" not in legacy_params
        # Both fence keys at once: refused, one floor semantics per mint.
        with pytest.raises(BasketConfigError, match="mutually exclusive"):
            _load(
                {**base["calc"], "filter_sigma_floor_pct": 3.0},
                tmp_path,
                "both.json",
            )
        # Malformed pct values: refused at load (either knob).
        for bad in (-1, "3", True):
            with pytest.raises(
                BasketConfigError, match="filter_sigma_floor_pct"
            ):
                _load(
                    {**calc_no_floor, **pct_pair,
                     "filter_sigma_floor_pct": bad},
                    tmp_path,
                    "bad.json",
                )
            with pytest.raises(
                BasketConfigError, match="vote_sigma_floor_pct"
            ):
                _load(
                    {**calc_no_floor, **pct_pair,
                     "vote_sigma_floor_pct": bad},
                    tmp_path,
                    "bad_vote.json",
                )
        # Percent-regime votes lane MISSING the vote floor (or at 0): the
        # reworked gate refuses — the fence floor never backs the vote.
        for calc in (
            {**calc_no_floor, "filter_sigma_floor_pct": 3.0},
            {**calc_no_floor, **pct_pair, "vote_sigma_floor_pct": 0},
        ):
            with pytest.raises(
                BasketConfigError, match="vote_sigma_floor_pct > 0"
            ):
                _load(calc, tmp_path, "gate.json")
        # Fence pct 0 with a positive vote floor is a REAL percent ruling
        # now (the fence floor is fence-only) and validates.
        _load(
            {**calc_no_floor, **pct_pair, "filter_sigma_floor_pct": 0},
            tmp_path,
            "fence_zero.json",
        )
        # A vote floor without votes is inert config and refuses.
        no_votes = {
            k: v
            for k, v in calc_no_floor.items()
            if k != "composite_statistic"
        }
        with pytest.raises(
            BasketConfigError,
            match="vote_sigma_floor_pct requires calc.composite_statistic",
        ):
            _load({**no_votes, **pct_pair}, tmp_path, "no_votes.json")
        # A vote floor alongside the ABSOLUTE floor refuses — the absolute
        # key's frozen semantics already govern both sigmas.
        with pytest.raises(
            BasketConfigError,
            match="absolute\\s+calc.filter_sigma_floor are mutually",
        ):
            _load(
                {**base["calc"], "vote_sigma_floor_pct": 3.0},
                tmp_path,
                "vote_plus_absolute.json",
            )


def test_config_refuses_panel_only_vote_sigma_source():
    """calc.vote_sigma_source is a PANEL-ONLY key (ruling 2026-08-27,
    gpu_index.index.panel_config owns it): the daily engine never reads it, so
    silently accepting it here would let a daily config document a vote
    rule its own series never runs — refused loudly at load instead."""
    base = json.loads(
        (REPO_ROOT / "config" / "index_basket.json").read_text()
    )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "panel_key.json"
        cfg = json.loads(json.dumps(base))
        cfg["calc"]["vote_sigma_source"] = "dw_history"
        path.write_text(json.dumps(cfg))
        with pytest.raises(BasketConfigError, match="vote_sigma_source"):
            load_basket_config(path)


def test_compute_day_refuses_ambiguous_floor_pair_in_params():
    """Hardening 2026-08-27: config load refuses the floor pair, but
    params can also arrive from an artifact-embedded/unvalidated dict —
    with both keys present the binding floor is ambiguous, so the
    resolution site refuses loudly naming both keys. No published
    artifact carries both (each carries filter_sigma_floor alone or no
    floor key), so replays never hit this."""
    cfg = json.loads(json.dumps(CONFIG))
    assert "filter_sigma_floor" in cfg["calc"]
    cfg["calc"]["filter_sigma_floor_pct"] = 3.0  # the unvalidated bypass
    with pytest.raises(
        ValueError, match="filter_sigma_floor and\\s+filter_sigma_floor_pct"
    ):
        compute_day(
            config=cfg,
            day="2026-08-10",
            snapshot={"run_id": "x", "late_fill": False, "sources": []},
            substituted_from=None,
            window_history={},
            window_currencies={},
            fx_records={},
            weight_state=_ws(),
            prior_slot_prints={},
        )


def test_compute_day_fence_and_vote_floors_bite_independently():
    """Floor split through the DAILY engine (founder ruling 2026-08-27,
    parity with the panel): fence pct 3.0 / vote pct 5.0 —

      - massedcompute, frozen 11-print 5.95 window, 6.15 print: the FENCE
        band floors at 3% of the window MEAN (band 3.0 * 0.1785 ->
        0.5355, dev 0.2 accepted) while its VOTE floors at 5% of the 6.15
        PRINT: 6.15 * (5.0/100.0) = 0.30750000000000005 -> 0.3075 at the
        6dp wire round — NOT 3% of the print (0.1845) and NOT any pct of
        the mean;
      - hyperstack, [2.4, 2.6] x 6 window (pstdev 0.10000000000000009),
        2.5 print: the real sd EXCEEDS its 3% fence floor (0.075), so the
        fence band is 3.0 * sd -> 0.3 (fence floor no bite), yet the vote
        is STILL floored — 0.1 < 0.125 (5% of 2.5) — pinning that
        sigma_floored reflects the VOTE floor."""
    cfg = json.loads(json.dumps(CONFIG))
    del cfg["calc"]["filter_sigma_floor"]
    cfg["calc"]["filter_sigma_floor_pct"] = 3.0
    cfg["calc"]["vote_sigma_floor_pct"] = 5.0
    cfg["calc"]["methodology_id"] = "annex_a_v0_2_calc_v_test_split"
    payload = compute_day(
        config=cfg,
        day="2026-08-10",
        snapshot={
            "run_id": "split-e2e",
            "late_fill": False,
            "sources": [
                _entry("massedcompute", [_obs("B300", usd=6.15)]),
                _entry("hyperstack", [_obs("B300", usd=2.5)]),
            ],
        },
        substituted_from=None,
        window_history={
            "massedcompute": [5.95] * 11,
            "hyperstack": [2.4, 2.6] * 6,
        },
        window_currencies={"massedcompute": "USD", "hyperstack": "USD"},
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    assert payload["calc_params"]["filter_sigma_floor_pct"] == 3.0
    assert payload["calc_params"]["vote_sigma_floor_pct"] == 5.0
    assert "filter_sigma_floor" not in payload["calc_params"]
    by_id = {s["source_id"]: s for s in payload["sources"]}
    massed = by_id["massedcompute"]["filter"]
    assert massed["accepted"] is True
    assert massed["band"] == pytest.approx(3.0 * 0.1785)
    assert by_id["massedcompute"]["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.3075,
    }
    hyper = by_id["hyperstack"]["filter"]
    assert hyper["accepted"] is True
    assert hyper["band"] == 0.3  # 3.0 * the REAL sd — fence floor no bite
    assert by_id["hyperstack"]["vote"] == {
        "sigma": 0.1,
        "sigma_floored": True,  # the VOTE floor bites where the fence's didn't
        "conf_usd_gpu_hr": 0.125,
    }


def test_compute_day_refuses_percent_regime_params_missing_vote_floor():
    """Era rule (floor split, 2026-08-27), daily mirror of the panel: a
    percent-regime median-votes params dict WITHOUT vote_sigma_floor_pct
    fails LOUDLY — no published daily series is percent-regime, so no
    artifact-embedded params legitimately lack the key, and silently
    falling back to the fence floor would price votes under a rule the
    params never recorded. Legacy absolute params (every published vote
    day) are untouched."""
    cfg = json.loads(json.dumps(CONFIG))
    del cfg["calc"]["filter_sigma_floor"]
    cfg["calc"]["filter_sigma_floor_pct"] = 3.0  # the unvalidated bypass
    with pytest.raises(
        ValueError, match="missing\\s+vote_sigma_floor_pct"
    ):
        compute_day(
            config=cfg,
            day="2026-08-10",
            snapshot={"run_id": "x", "late_fill": False, "sources": []},
            substituted_from=None,
            window_history={},
            window_currencies={},
            fx_records={},
            weight_state=_ws(),
            prior_slot_prints={},
        )


def test_compute_day_refuses_vote_floor_alongside_absolute_params():
    """Floor-split twin of the ambiguous-pair guard, daily mirror: params
    carrying vote_sigma_floor_pct together with the ABSOLUTE
    filter_sigma_floor raise loudly — the absolute key's frozen semantics
    govern both sigmas, so the binding vote floor would be ambiguous. No
    published artifact carries that pair."""
    cfg = json.loads(json.dumps(CONFIG))
    assert "filter_sigma_floor" in cfg["calc"]
    cfg["calc"]["vote_sigma_floor_pct"] = 3.0  # the unvalidated bypass
    with pytest.raises(
        ValueError,
        match="vote_sigma_floor_pct without\\s+filter_sigma_floor_pct",
    ):
        compute_day(
            config=cfg,
            day="2026-08-10",
            snapshot={"run_id": "x", "late_fill": False, "sources": []},
            substituted_from=None,
            window_history={},
            window_currencies={},
            fx_records={},
            weight_state=_ws(),
            prior_slot_prints={},
        )


def test_median_ci_composite_pyth_properties():
    """The properties the Pyth aggregation is adopted FOR, each on a
    hand-checkable book."""
    # Unanimous sources: index = the price, confidence = the shared CI.
    unanimous = median_stddev_composite(
        [("a", 0.5, 7.0), ("b", 0.5, 7.0)], {"a": 0.1, "b": 0.1}
    )
    assert unanimous["value_usd_gpu_hr"] == 7.0
    assert unanimous["confidence_usd_gpu_hr"] == pytest.approx(0.1)
    assert unanimous["vote_p25_usd_gpu_hr"] == 6.9
    assert unanimous["vote_p75_usd_gpu_hr"] == 7.1
    # A far outlier with a TIGHT interval cannot drag the index past the
    # vote quantiles: 4 honest sources at ~7, one liar at 20.
    book = [("a", 0.2, 7.0), ("b", 0.2, 7.1), ("c", 0.2, 6.9),
            ("d", 0.2, 7.0), ("liar", 0.2, 20.0)]
    confs = {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1, "liar": 0.05}
    robust = median_stddev_composite(book, confs)
    assert robust["value_usd_gpu_hr"] == 7.0
    # The same book under the retired weighted mean moves 2.6 dollars.
    assert robust["weighted_mean_usd_gpu_hr"] == pytest.approx(9.6)
    # The index always lies inside the central vote band.
    assert (
        robust["vote_p25_usd_gpu_hr"]
        <= robust["value_usd_gpu_hr"]
        <= robust["vote_p75_usd_gpu_hr"]
    )
    # Zero-CI equal-weight votes collapse to the plain median of prices.
    degenerate = median_stddev_composite(
        [("a", 0.25, 5.0), ("b", 0.25, 6.0), ("c", 0.25, 9.0)],
        {"a": 0.0, "b": 0.0, "c": 0.0},
    )
    assert degenerate["value_usd_gpu_hr"] == 6.0
    # Weights bite: the same three prices with the first source carrying
    # the majority weight pull the median to ITS price.
    weighted = median_stddev_composite(
        [("a", 0.60, 5.0), ("b", 0.20, 6.0), ("c", 0.20, 9.0)],
        {"a": 0.0, "b": 0.0, "c": 0.0},
    )
    assert weighted["value_usd_gpu_hr"] == 5.0
    assert median_stddev_composite([], {}) is None


def test_compute_day_vote_ci_uses_the_filter_window_and_fx():
    """End-to-end vote derivation on a source with real history: the CI is
    the pre-advance window pstdev in RECORDED currency, floored, converted
    at the day's own rate — and never includes today's print."""
    # Scaleway printed 7.40/7.60 EUR on prior days; today prints 7.50 EUR.
    window_history = {"scaleway": [7.40, 7.60]}
    window_currencies = {"scaleway": "EUR"}
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _entry(
                "scaleway",
                [_obs("B300", native=7.50, usd=None, currency="EUR", basis=8)],
            ),
            _entry("verda", [_obs("B300", usd=7.55)]),
        ],
    }
    payload = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history=window_history,
        window_currencies=window_currencies,
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    by_id = {s["source_id"]: s for s in payload["sources"]}
    vote = by_id["scaleway"]["vote"]
    # pstdev([7.40, 7.60]) = 0.10 EUR > floor; x 1.1555 = 0.11555 USD.
    assert vote == {
        "sigma": 0.1,
        "sigma_floored": False,
        "conf_usd_gpu_hr": 0.11555,
    }
    # Today's 7.50 EUR print entered the window only AFTER the vote.
    assert window_history["scaleway"] == [7.40, 7.60, 7.50]
    # The USD source with no history votes at the bare floor.
    assert by_id["verda"]["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.05,
    }


def test_vote_sigma_uses_only_the_filter_window_slice():
    """The vote CI must derive from the SAME trailing slice the fence
    judges against — never the full unbounded history. A 25-entry history
    whose last 20 prints are flat must vote at the floor even though the
    full-history pstdev is large (mutation-proven gap: dropping the slice
    passed every prior test)."""
    history = {"verda": [7.0, 9.0, 7.0, 9.0, 7.0] + [7.5] * 20}
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [_entry("verda", [_obs("B300", usd=7.5)])],
    }
    payload = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history=history,
        window_currencies={"verda": "USD"},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    verda = next(s for s in payload["sources"] if s["source_id"] == "verda")
    assert verda["filter"]["accepted"] is True
    # pstdev of the last 20 (all 7.5) is 0; full-history pstdev ~0.87.
    assert verda["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.05,
    }


def test_vote_fx_falls_back_to_the_implied_recorded_rate():
    """A print captured in USD alongside a trusted native label carries no
    fx_rate block — the vote CI must convert at the rate the recorded
    USD/native pair implies, never silently at parity (mutation-proven
    gap: pricing the fallback at 1.0 passed every prior test)."""
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _entry(
                "scaleway",
                [
                    _obs(
                        "B300",
                        usd=8.6664,
                        native=7.50,
                        currency="EUR",
                        basis=8,
                    )
                ],
            )
        ],
    }
    payload = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    scaleway = next(
        s for s in payload["sources"] if s["source_id"] == "scaleway"
    )
    # Implied rate 8.6664 / 7.50 = 1.15552; floor 0.05 EUR x implied rate.
    assert scaleway["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": pytest.approx(0.057776),
    }


def test_vote_fx_with_no_trustworthy_rate_fails_closed():
    """A non-USD vote with neither a usable fx_rate nor a positive native
    price must kill the day's compute loudly (D1 posture) — never price
    the CI at parity by silent default."""
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _entry(
                "scaleway",
                [
                    # Trusted EUR label with a pathological zero native
                    # price and a captured USD price: no honest rate.
                    _obs(
                        "B300",
                        usd=8.67,
                        native=0.0,
                        currency="EUR",
                        basis=8,
                    )
                ],
            )
        ],
    }
    with pytest.raises(ValueError, match="no trustworthy fx rate"):
        compute_day(
            config=CONFIG,
            day="2026-08-10",
            snapshot=snapshot,
            substituted_from=None,
            window_history={},
            window_currencies={},
            fx_records={},
            weight_state=_ws(),
            prior_slot_prints={},
        )


def test_median_ci_composite_rejects_non_finite_votes():
    """NaN never prices a day: a poisoned price or CI must raise, not
    silently mis-sort the vote ladder (NaN compares False both ways, so
    sorted() scatters it and the cum walk counts its weight arbitrarily —
    a plausible-looking wrong index)."""
    with pytest.raises(ValueError, match="non-finite vote"):
        median_stddev_composite(
            [("a", 0.5, float("nan")), ("b", 0.5, 7.0)],
            {"a": 0.05, "b": 0.05},
        )
    with pytest.raises(ValueError, match="non-finite vote"):
        median_stddev_composite(
            [("a", 0.5, 7.0)], {"a": float("inf")}
        )


def test_weighted_quantile_rejects_non_positive_weights():
    """The docstring promises a clean error, never an IndexError or a
    silent average over a zero-weight book."""
    with pytest.raises(ValueError, match="positive weights"):
        _weighted_quantile([(5.0, 0.0)], Fraction(1, 2))
    with pytest.raises(ValueError, match="positive weights"):
        _weighted_quantile([(5.0, 1.0), (9.0, -0.1)], Fraction(1, 2))


def test_config_rejects_median_votes_without_a_positive_floor(tmp_path):
    """The floor is what stops a frozen list price voting with conviction
    it never earned — a lane config that copies the statistic without the
    floor must fail at load, not pin floor-less params as series law."""
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    for calc_override in (
        {k: v for k, v in base["calc"].items() if k != "filter_sigma_floor"},
        {**base["calc"], "filter_sigma_floor": 0},
    ):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({**base, "calc": calc_override}))
        with pytest.raises(BasketConfigError, match="filter_sigma_floor"):
            load_basket_config(p)


def _hand_walk_votes():
    """The calc_v4 21-vote hand-walk ballot (7 sources, W = 2.70; every
    conf at the 0.05 floor except scaleway's converted 0.057775)."""
    book = [
        ("runpod", 0.15, 7.85),
        ("nebius", 0.15, 7.40),
        ("scaleway", 0.15, 8.66625),
        ("verda", 0.15, 7.50),
        ("hyperstack", 0.10, 7.39),
        ("massed", 0.10, 8.00),
        ("crusoe", 0.10, 5.95),
    ]
    confs = {sid: 0.05 for sid, _, _ in book}
    confs["scaleway"] = 0.057775
    votes = []
    for sid, w, p in book:
        votes.extend([(p - confs[sid], w), (p, w), (p + confs[sid], w)])
    return book, confs, votes


def test_interquantile_mean_matches_its_integral_definition():
    """The interquantile mean is (1/2a) * integral of Q(u) du
    over [1/2 - a, 1/2 + a] — checked against a midpoint Riemann sum with
    the FROZEN _weighted_quantile as the oracle, plus exact closed forms."""
    # Hand-walk: unit weights on 1/2/10, alpha 1/4 -> the band [0.75, 2.25]
    # of W = 3 overlaps the votes 0.25 / 1.0 / 0.25, so the mean is
    # (1*0.25 + 2*1 + 10*0.25) / 1.5 = 19/6 — fractional band edges bite.
    tiny = [(1.0, 1.0), (2.0, 1.0), (10.0, 1.0)]
    assert _interquantile_mean(tiny, Fraction(1, 4)) == Fraction(19, 6)
    _, _, votes = _hand_walk_votes()
    # alpha -> 0 continuity THROUGH the exact-boundary midpoint convention:
    # a 0.001 band straddles the 1.35 boundary half inside verda's point
    # vote and half inside its high vote — exactly the median's midpoint.
    assert _interquantile_mean(votes, Fraction("0.001")) == Fraction("7.525")
    # alpha = 1/2 IS the weighted mean of the votes, exactly.
    total = sum((Fraction(str(w)) for _, w in votes), start=Fraction(0))
    vote_mean = (
        sum(
            (Fraction(str(v)) * Fraction(str(w)) for v, w in votes),
            start=Fraction(0),
        )
        / total
    )
    assert _interquantile_mean(votes, Fraction(1, 2)) == vote_mean
    # The integral definition, discretized: midpoint samples of the frozen
    # quantile function (step-function error bound ~ vote-range / n).
    n = 2001
    for alpha in (Fraction("0.1"), Fraction("0.2")):
        lo = Fraction(1, 2) - alpha
        sampled = (
            sum(
                _weighted_quantile(
                    votes, lo + 2 * alpha * Fraction(2 * i + 1, 2 * n)
                )
                for i in range(n)
            )
            / n
        )
        assert float(_interquantile_mean(votes, alpha)) == pytest.approx(
            sampled, abs=1e-3
        )
    # Refusals: the band must be a band, over a priceable book.
    for bad in (Fraction(0), Fraction(-1, 10), Fraction(51, 100)):
        with pytest.raises(ValueError, match="alpha"):
            _interquantile_mean(votes, bad)
    with pytest.raises(ValueError, match="positive weights"):
        _interquantile_mean([(5.0, 1.0), (9.0, -0.1)], Fraction(1, 4))
    with pytest.raises(ValueError, match="empty"):
        _interquantile_mean([], Fraction(1, 4))


def test_median_composite_iqm_alpha_prices_the_band_mean():
    """nonzero iqm_alpha prices the SAME ballot at the band mean,
    echoes the alpha + point median for artifact self-description, and
    anchors the dispersion at the published (rounded) value; alpha 0 is the
    frozen median path verbatim, key set included."""
    book, confs, _ = _hand_walk_votes()
    base = median_stddev_composite(book, confs)
    assert median_stddev_composite(book, confs, iqm_alpha=0.0) == base
    assert "iqm_alpha" not in base
    assert "vote_median_usd_gpu_hr" not in base
    tuned = median_stddev_composite(book, confs, iqm_alpha=0.1)
    # Exact 545/72, rounded half-even at the 6dp artifact grain.
    assert tuned["value_usd_gpu_hr"] == 7.569444
    assert tuned["iqm_alpha"] == 0.1
    assert tuned["vote_median_usd_gpu_hr"] == base["value_usd_gpu_hr"] == 7.525
    # The ballot's quantiles are unchanged; the dispersion anchors at the
    # published value: max(7.569444 - 7.40, 7.95 - 7.569444).
    assert tuned["vote_p25_usd_gpu_hr"] == base["vote_p25_usd_gpu_hr"]
    assert tuned["vote_p75_usd_gpu_hr"] == base["vote_p75_usd_gpu_hr"]
    assert tuned["confidence_usd_gpu_hr"] == 0.380556
    assert tuned["statistic"] == "median_ci_votes"
    # The Pyth robustness the median was adopted FOR survives a 40-60%
    # band: the liar's tight far vote still cannot drag the index outside
    # the central vote mass.
    liar_book = [
        ("a", 0.2, 7.0),
        ("b", 0.2, 7.1),
        ("c", 0.2, 6.9),
        ("d", 0.2, 7.0),
        ("liar", 0.2, 20.0),
    ]
    liar_confs = {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1, "liar": 0.05}
    robust = median_stddev_composite(liar_book, liar_confs, iqm_alpha=0.1)
    assert (
        robust["vote_p25_usd_gpu_hr"]
        <= robust["value_usd_gpu_hr"]
        <= robust["vote_p75_usd_gpu_hr"]
    )
    assert abs(robust["value_usd_gpu_hr"] - 7.0) < 0.2


def test_iqm_quantization_regimes_pinned_at_an_exact_tie():
    """Review finding: the two rounding regimes are DISTINGUISHABLE only
    at a 7dp-halfway quantity, and no ordinary vector reaches one — so this
    pins them where they genuinely differ, or a mirror (or refactor) using
    the wrong tie rule passes every other test while drifting from prod on
    tie days. The band mean quantizes half-EVEN on the exact rational; the
    vote_median diagnostic keeps the frozen float round() so it prints
    exactly what the alpha-0 branch would have published."""
    # Helper-level: mean of 1.0 and 1.000001 = 2000001/2000000, a true
    # 7dp-halfway tie with an EVEN floor — half-even stays 1.0, half-up
    # would print 1.000001.
    assert (
        float(round(_interquantile_mean([(1.000001, 1.0), (1.0, 1.0)], Fraction(1, 2)), 6))
        == 1.0
    )
    # Even/odd floors through the engine (alpha 1/2 = the vote mean):
    # 1.0000025 has an even floor (stays), 1.0000015 an odd one (goes up)
    # — both land on the even 6dp neighbor 1.000002.
    even = median_stddev_composite(
        [("a", 1.0, 1.000002), ("b", 1.0, 1.000003)],
        {"a": 0.0, "b": 0.0},
        iqm_alpha=0.5,
    )
    assert even["value_usd_gpu_hr"] == 1.000002
    odd = median_stddev_composite(
        [("a", 1.0, 1.000001), ("b", 1.0, 1.000002)],
        {"a": 0.0, "b": 0.0},
        iqm_alpha=0.5,
    )
    assert odd["value_usd_gpu_hr"] == 1.000002
    # The dual-regime ballot: equal weights make the exact cum == target
    # boundary fire, so BOTH the band mean and the point median are the
    # same midpoint 1.0000005 — and the artifact prints it two ways on
    # purpose: 1.0 as the half-even band mean, 1.000001 as the frozen
    # float-round median (byte-identical to what alpha 0 publishes).
    book = [("a", 0.5, 1.0), ("b", 0.5, 1.000001)]
    confs = {"a": 0.05, "b": 0.05}
    dual = median_stddev_composite(book, confs, iqm_alpha=0.05)
    assert dual["value_usd_gpu_hr"] == 1.0
    assert dual["vote_median_usd_gpu_hr"] == 1.000001
    frozen = median_stddev_composite(book, confs)
    assert dual["vote_median_usd_gpu_hr"] == frozen["value_usd_gpu_hr"]
    # Fail-closed on a derived-vote overflow: finite price and stddev whose
    # sum is an infinity must raise, never seat an inf at the ladder edge.
    with pytest.raises(ValueError, match="non-finite vote"):
        median_stddev_composite(
            [("a", 1.0, 1e308), ("b", 1.0, 5.0)],
            {"a": 1e308, "b": 0.05},
            iqm_alpha=0.1,
        )
    with pytest.raises(ValueError, match="non-finite vote"):
        median_stddev_composite(
            [("a", 1.0, 1e308), ("b", 1.0, 5.0)], {"a": 1e308, "b": 0.05}
        )


def test_config_rejects_bad_iqm_alpha(tmp_path):
    """iqm_alpha is load-validated — a band outside [0, 1/2], a
    non-number, or a band without the vote statistic refuses at LOAD, never
    pinning silently-inert or unpriceable params as series law."""
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    p = tmp_path / "c.json"
    for bad in (-0.1, 0.51, True, "0.1", None):
        p.write_text(
            json.dumps({**base, "calc": {**base["calc"], "iqm_alpha": bad}})
        )
        with pytest.raises(BasketConfigError, match="iqm_alpha"):
            load_basket_config(p)
    # The knob is a band of the VOTE ladder — refused without the statistic.
    stripped = {
        k: v for k, v in base["calc"].items() if k != "composite_statistic"
    }
    p.write_text(json.dumps({**base, "calc": {**stripped, "iqm_alpha": 0.1}}))
    with pytest.raises(BasketConfigError, match="iqm_alpha"):
        load_basket_config(p)
    # Well-formed bands load; 0 is legal (the median, explicitly); the
    # param embeds via calc_params — and stays ABSENT when unset (frozen
    # series keep their exact artifact bytes).
    for good in (0, 0.1, 0.5):
        p.write_text(
            json.dumps({**base, "calc": {**base["calc"], "iqm_alpha": good}})
        )
        assert calc_params(load_basket_config(p))["iqm_alpha"] == float(good)
    assert "iqm_alpha" not in calc_params(CONFIG)


def test_compute_day_iqm_alpha_recomputable_from_the_artifact():
    """End to end: a tuned day embeds the alpha in calc_params,
    echoes it in the index block, and the published value reproduces from
    the artifact's own vote blocks + the day's weight vector alone — the
    same recompute the ops mirror runs."""
    config = json.loads(json.dumps(CONFIG))
    config["calc"]["iqm_alpha"] = 0.1
    payload = compute_day(
        config=config,
        day="2026-08-16",
        snapshot=_live_first_capture_snapshot(),
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    index = payload["index"]
    assert payload["calc_params"]["iqm_alpha"] == 0.1
    assert index["iqm_alpha"] == 0.1
    weights = payload["weight_calc"]["weights"]
    votes = []
    for block in payload["sources"]:
        if "vote" not in block:
            continue
        price = block["chosen"]["usd_per_gpu_hr"]
        conf = block["vote"]["conf_usd_gpu_hr"]
        weight = weights[block["source_id"]]
        votes.extend(
            [(price - conf, weight), (price, weight), (price + conf, weight)]
        )
    assert index["value_usd_gpu_hr"] == float(
        round(_interquantile_mean(votes, Fraction("0.1")), 6)
    )
    assert index["vote_median_usd_gpu_hr"] == round(
        _weighted_quantile(votes, Fraction(1, 2)), 6
    )
    # The knob genuinely bit: the band mean moved this day off its median.
    assert index["value_usd_gpu_hr"] != index["vote_median_usd_gpu_hr"]


def test_compute_day_held_out_source_casts_no_vote():
    """A held-out print contributes no vote and no vote block — the
    passing set and the voting set must be the same set."""
    window_history = {"verda": [7.50] * 12}  # armed window, sigma 0
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _entry("verda", [_obs("B300", usd=9.99)]),  # way past the fence
            _entry("nebius", [_obs("B300", usd=7.68)]),
        ],
    }
    payload = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history=window_history,
        window_currencies={"verda": "USD"},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    by_id = {s["source_id"]: s for s in payload["sources"]}
    assert by_id["verda"]["filter"]["accepted"] is False
    assert "vote" not in by_id["verda"]
    assert "vote" in by_id["nebius"]
    assert payload["index"]["sources_used_count"] == 1
    assert payload["index"]["value_usd_gpu_hr"] == 7.68


def test_compute_day_currency_confirmed_votes_from_the_seed_prints():
    """A confirmed currency change votes from the pending seed prints
    (minus today's — a print never self-reports its confidence), in the
    NEW currency's terms, converted at today's rate."""
    window_history = {"scaleway": [8.60, 8.62, 8.61]}  # old USD window
    window_currencies = {"scaleway": "USD"}
    pending = {"scaleway": {"currency": "EUR", "prints": [7.48, 7.52]}}
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _entry(
                "scaleway",
                [_obs("B300", native=7.50, usd=None, currency="EUR", basis=8)],
            ),
        ],
    }
    payload = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history=window_history,
        window_currencies=window_currencies,
        fx_records=FX_2026_08_10,
        pending_currencies=pending,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    scaleway = next(s for s in payload["sources"] if s["source_id"] == "scaleway")
    assert scaleway["filter"]["currency_confirmed"] is True
    vote = scaleway["vote"]
    # pstdev([7.48, 7.52]) = 0.02 EUR < floor 0.05 -> floored, x 1.1555.
    assert vote == {
        "sigma": 0.02,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.057775,
    }


def test_held_out_source_is_excluded_from_the_day_but_enters_window():
    history = {"verda": [7.5] * 12}
    snapshot = _worked_example_snapshot()
    for entry in snapshot["sources"]:
        if entry["source_id"] == "verda":
            entry["observations"] = [_obs("B300", usd=9.0)]
    payload = compute_day(
        config=CONFIG,
        day="2026-09-01",  # outside the calc_v2 manual-exclusion window
        snapshot=snapshot,
        substituted_from=None,
        window_history=history,
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    verda = next(s for s in payload["sources"] if s["source_id"] == "verda")
    assert verda["filter"]["accepted"] is False
    # calc_v3: the verdict names the fence it was judged against, and the
    # floored band (sigma=0 window, floor 0.05 USD) — 9.0 is way outside.
    assert (verda["filter"]["lo"], verda["filter"]["hi"]) == (7.35, 7.65)
    assert verda["filter"]["currency"] == "USD"
    assert payload["index"]["sources_used_count"] == 7
    assert "verda" not in payload["index"]["renormalized_weights"]
    assert history["verda"][-1] == 9.0  # every real print enters the window


def test_native_filter_is_immune_to_fx_moves():
    """R-native (calc_v3), the 2026-08-20 incident inverted: ten identical
    7.50 EUR native prints, then the identical print at a MOVED FX rate.
    The live config accepts (native deviation 0 inside the floored band —
    0.15 EUR at the calc_v4 3.0 multiplier) while the USD composite still
    prices at the day's real rate; the same world under the calc_v2 rules
    (USD filter terms, no floor) held it out — proving the fix bites."""
    fx_moved = {
        "2026-08-20": {
            "source": "ecb_reference_rate",
            "as_of": "2026-08-20",
            "rates": {"USD": 1.1681},  # the real 08-20 rate, ~1% off 1.1555
        }
    }
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _entry(
                "scaleway",
                [_obs("B300", native=7.50, usd=None, currency="EUR", basis=8)],
            )
        ],
    }
    history = {"scaleway": [7.5] * 10}  # ten identical NATIVE prints
    currencies = {"scaleway": "EUR"}
    payload = compute_day(
        config=CONFIG,
        day="2026-08-20",
        snapshot=snapshot,
        substituted_from=None,
        window_history=history,
        fx_records=fx_moved,
        window_currencies=currencies,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    scaleway = next(s for s in payload["sources"] if s["source_id"] == "scaleway")
    verdict = scaleway["filter"]
    assert verdict["accepted"] is True  # FX alone can no longer hold out
    assert verdict["unfiltered"] is False  # a REAL filtered accept
    assert (verdict["mu"], verdict["sigma"], verdict["deviation"]) == (7.5, 0.0, 0.0)
    assert verdict["band"] == pytest.approx(0.15)  # 3.0 * max(0, 0.05) EUR
    assert (verdict["lo"], verdict["hi"]) == (7.35, 7.65)
    assert verdict["currency"] == "EUR"
    assert verdict["sigma_zero"] is True  # informational flag survives
    # The composite still prices in USD at the day's real rate (R2).
    assert scaleway["chosen"]["usd_per_gpu_hr"] == pytest.approx(7.5 * 1.1681)
    assert payload["index"]["value_usd_gpu_hr"] == pytest.approx(7.5 * 1.1681)
    # The window advanced with the NATIVE print, same currency.
    assert history["scaleway"] == [7.5] * 11
    assert currencies["scaleway"] == "EUR"

    # Inversion: the calc_v2 world (USD terms, no floor) on the same facts —
    # a USD window of converted prints and the new rate's print — holds the
    # unmoved native price out. That is the defect calc_v3 fixes.
    cfg_v2 = json.loads(json.dumps(CONFIG))
    del cfg_v2["calc"]["filter_terms"]
    del cfg_v2["calc"]["filter_sigma_floor"]
    del cfg_v2["calc"]["dynamic_weights"]  # calc_v2 predates the v5 mint
    usd_history = {"scaleway": [round(7.5 * 1.1555, 6)] * 10}
    legacy = compute_day(
        config=cfg_v2,
        day="2026-08-20",
        snapshot=snapshot,
        substituted_from=None,
        window_history=usd_history,
        fx_records=fx_moved,
    )
    legacy_verdict = next(
        s for s in legacy["sources"] if s["source_id"] == "scaleway"
    )["filter"]
    assert legacy_verdict["accepted"] is False  # held out purely by FX
    assert "currency" not in legacy_verdict  # legacy verdict shape intact
    assert "band" not in legacy_verdict


def _one_source_day(
    obs_list,
    *,
    history,
    currencies,
    pending,
    weight_state,
    day="2026-09-01",
    fx_records=None,
    prior_slot_prints=None,
):
    """One compute_day over a single-source snapshot, returning that
    source's verdict — the D1 state (history/currencies/pending, and the
    calc_v5 weight_state) threads across calls exactly the way the CLI
    threads it across days."""
    payload = compute_day(
        config=CONFIG,
        day=day,
        snapshot={
            "run_id": "r",
            "late_fill": False,
            "sources": [_entry("scaleway", obs_list)],
        },
        substituted_from=None,
        window_history=history,
        fx_records=fx_records or {},
        window_currencies=currencies,
        pending_currencies=pending,
        weight_state=weight_state,
        prior_slot_prints=prior_slot_prints if prior_slot_prints is not None else {},
    )
    return next(
        s for s in payload["sources"] if s["source_id"] == "scaleway"
    )["filter"]


def test_currency_mismatch_is_held_out_and_preserves_the_window():
    """Ruling D1: a trusted print in a DIFFERENT currency than its window
    is HELD OUT (fail-closed) — never an unfiltered accept — and the old
    window survives untouched; the print enters only the pending streak."""
    history = {"scaleway": [7.5] * 12}
    currencies = {"scaleway": "EUR"}
    pending: dict = {}
    ws = _ws()
    verdict = _one_source_day(
        [_obs("B300", usd=8.70)],
        history=history,
        currencies=currencies,
        pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert verdict == {
        "accepted": False,
        "unfiltered": False,
        "currency_mismatch": True,
        "currency": "USD",
        "window_currency": "EUR",
        "filter_price": 8.70,
        "pending_count": 1,
        "confirm_after": 3,
        "n_history": 12,
    }
    # Window preserved in EUR; the USD print sits ONLY in the streak.
    assert history["scaleway"] == [7.5] * 12
    assert currencies["scaleway"] == "EUR"
    assert pending["scaleway"] == {"currency": "USD", "prints": [8.70]}
    # An untouched same-currency window is a plain filtered verdict.
    steady = _one_source_day(
        [_obs("B300", usd=8.70)],
        history={"scaleway": [8.70] * 12},
        currencies={"scaleway": "USD"},
        pending={},
        weight_state=_ws(),
        prior_slot_prints={},  # fresh world, fresh state
    )
    assert steady["accepted"] is True
    assert "currency_mismatch" not in steady


def test_genuine_currency_change_confirms_on_the_third_consecutive_day():
    """Ruling D1: days 1-2 of a genuine currency change are held out
    (window preserved); the 3rd consecutive same-new-currency print
    CONFIRMS it — old window discarded, new window seeded from the three
    pending prints (n_history=3, still warm-up), that day accepted with
    currency_confirmed."""
    history = {"scaleway": [7.5] * 12}
    currencies = {"scaleway": "EUR"}
    pending: dict = {}
    ws = _ws()
    prints = [8.70, 8.71, 8.69]
    day1 = _one_source_day(
        [_obs("B300", usd=prints[0])],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert (day1["accepted"], day1["pending_count"]) == (False, 1)
    day2 = _one_source_day(
        [_obs("B300", usd=prints[1])],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert (day2["accepted"], day2["pending_count"]) == (False, 2)
    assert history["scaleway"] == [7.5] * 12  # still intact after 2 days
    day3 = _one_source_day(
        [_obs("B300", usd=prints[2])],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert day3 == {
        "accepted": True,
        "unfiltered": True,
        "currency_confirmed": True,
        "currency": "USD",
        "window_currency": "EUR",
        "filter_price": 8.69,
        "n_history": 3,
    }
    # Old window discarded, new one seeded from ALL THREE pending prints.
    assert history["scaleway"] == prints
    assert currencies["scaleway"] == "USD"
    assert pending == {}
    # Day 4 is ordinary warm-up in the new currency.
    day4 = _one_source_day(
        [_obs("B300", usd=8.70)],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert day4 == {
        "accepted": True,
        "unfiltered": True,
        "n_history": 3,
        "currency": "USD",
    }


def test_mislabel_glitch_costs_held_out_days_but_never_the_window():
    """Ruling D1: a mislabel glitch shorter than the 3-day confirmation —
    here two USD-labeled days, then the EUR label heals — costs exactly 2
    held-out source-days, the EUR window survives intact, and the fence is
    still ARMED afterwards (a genuine outlier is still held out)."""
    history = {"scaleway": [7.5] * 12}
    currencies = {"scaleway": "EUR"}
    pending: dict = {}
    ws = _ws()
    held = 0
    for _ in range(2):  # the glitch: label flips to USD for two days
        verdict = _one_source_day(
            [_obs("B300", usd=8.70)],
            history=history, currencies=currencies, pending=pending,
            weight_state=ws,
            prior_slot_prints={},
        )
        assert verdict["currency_mismatch"] is True
        held += 0 if verdict["accepted"] else 1
    heal = _one_source_day(  # label heals: EUR print, unchanged price
        [_obs("B300", native=7.50, usd=None, currency="EUR", basis=8)],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
        day="2026-08-10", fx_records=FX_2026_08_10,
    )
    assert heal["accepted"] is True
    assert heal["currency"] == "EUR"
    assert held == 2  # exactly two held-out source-days
    assert pending == {}  # the streak died with the heal
    assert history["scaleway"] == [7.5] * 13  # window intact + heal print
    # Fence still armed: a genuine EUR outlier is still held out.
    outlier = _one_source_day(
        [_obs("B300", native=9.00, usd=None, currency="EUR", basis=8)],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
        day="2026-08-10", fx_records=FX_2026_08_10,
    )
    assert outlier["accepted"] is False
    assert outlier.get("currency_mismatch") is None


def test_alternating_dual_currency_winner_never_disarms_the_fence():
    """Ruling D1: a mixed-currency source whose cheapest print alternates
    EUR/USD is held out on every flip day — the streak never reaches 3
    consecutive, so the window never resets and the fence never disarms."""
    history = {"scaleway": [7.5] * 12}
    currencies = {"scaleway": "EUR"}
    pending: dict = {}
    ws = _ws()
    for _ in range(3):
        flip = _one_source_day(  # USD day: mismatch, held out
            [_obs("B300", usd=8.70)],
            history=history, currencies=currencies, pending=pending,
            weight_state=ws,
            prior_slot_prints={},
        )
        assert (flip["accepted"], flip["pending_count"]) == (False, 1)
        back = _one_source_day(  # EUR day: filters normally, streak dies
            [_obs("B300", native=7.50, usd=None, currency="EUR", basis=8)],
            history=history, currencies=currencies, pending=pending,
            weight_state=ws,
            prior_slot_prints={},
            day="2026-08-10", fx_records=FX_2026_08_10,
        )
        assert back["accepted"] is True
        assert pending == {}
    assert currencies["scaleway"] == "EUR"  # never reset
    assert history["scaleway"] == [7.5] * 15  # only EUR prints entered


def test_untrusted_currency_inputs_are_held_out_without_reset():
    """Ruling D1 fail-closed variants: a non-USD label with the native
    price missing, and an UNKNOWN label (native present), are UNTRUSTED —
    held out, window preserved, print never enters the window, and no
    reset machinery arms. An untrusted day also breaks a pending streak."""
    history = {"scaleway": [7.5] * 12}
    currencies = {"scaleway": "EUR"}
    pending: dict = {}
    ws = _ws()
    # Variant 1: EUR label, native missing (the chosen carries only USD).
    eur_no_native = _obs("B300", usd=8.66)
    eur_no_native["currency"] = "EUR"
    eur_no_native["price_native_per_gpu_hr"] = None
    v1 = _one_source_day(
        [eur_no_native],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert v1 == {
        "accepted": False,
        "unfiltered": False,
        "untrusted_currency": True,
        "currency_label": "EUR",
        "n_history": 12,
    }
    # Variant 2: UNKNOWN label WITH a native value — still untrusted.
    unknown = _obs("B300", usd=8.66, native=7.5)
    unknown["currency"] = "UNKNOWN"
    v2 = _one_source_day(
        [unknown],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert v2["untrusted_currency"] is True
    assert v2["accepted"] is False
    # Window preserved, currency unchanged, nothing pending, no reset.
    assert history["scaleway"] == [7.5] * 12
    assert currencies["scaleway"] == "EUR"
    assert pending == {}
    # And an untrusted day BREAKS a confirmation streak: 2 mismatch days,
    # an untrusted day, then a 3rd mismatch day is back to count 1.
    _one_source_day([_obs("B300", usd=8.70)],
                    history=history, currencies=currencies, pending=pending,
                    weight_state=ws)
    _one_source_day([_obs("B300", usd=8.71)],
                    history=history, currencies=currencies, pending=pending,
                    weight_state=ws)
    assert pending["scaleway"]["prints"] == [8.70, 8.71]
    _one_source_day([eur_no_native],
                    history=history, currencies=currencies, pending=pending,
                    weight_state=ws)
    assert pending == {}  # streak broken by the untrusted day
    restart = _one_source_day(
        [_obs("B300", usd=8.72)],
        history=history, currencies=currencies, pending=pending,
        weight_state=ws,
        prior_slot_prints={},
    )
    assert restart["pending_count"] == 1  # not confirmed
    assert history["scaleway"] == [7.5] * 12  # window still intact


def test_recorded_mode_requires_window_currencies():
    """Ruling D2: recorded-currency mode without a window_currencies dict
    is a programming error (a replay that cannot detect currency changes),
    not a silent degradation."""
    with pytest.raises(ValueError, match="window_currencies"):
        compute_day(
            config=CONFIG,
            day="2026-09-01",
            snapshot={"run_id": "r", "late_fill": False, "sources": []},
            substituted_from=None,
            window_history={},
            fx_records={},
        )


# ------------------------------------------------------------------ store


def test_read_day_snapshots_earliest_key_wins():
    client = FakeS3()
    day = date(2026, 8, 10)
    early = json.dumps({"run_id": "early"}).encode()
    late = json.dumps({"run_id": "late"}).encode()
    client.objects[
        "index/b300_basket/snapshots/2026-08-10/slot16-20260810T161001Z-aaaa.json"
    ] = early
    client.objects[
        "index/b300_basket/snapshots/2026-08-10/slot16-20260810T161045Z-bbbb.json"
    ] = late
    snaps = read_day_snapshots(client, "curves", prefix="index/b300_basket", day=day)
    assert snaps[16]["run_id"] == "early"


def _composite_payload(day="2026-08-10", value=7.53):
    return {
        "schema_version": 1,
        "kind": "index_basket_composite",
        "basket_id": "b300_annex_a_v0_2",
        "methodology_id": "annex_a_v0_2_calc_v1",
        "date": day,
        "basket_dark": False,
        "index": {"value_usd_gpu_hr": value, "sources_used_count": 8},
        "sources": [],
        "fallback_pool": {"sources": [], "mean_usd_gpu_hr": None},
        "day_missed": False,
    }


def test_composite_store_discipline():
    client = FakeS3()
    out = upload_composite(
        client,
        "curves",
        _composite_payload(),
        prefix="index/b300_basket",
        run_id="20260811T041000Z-aaaa",
        now=datetime(2026, 8, 11, 4, 10, tzinfo=timezone.utc),
    )
    key = out["composite_key"]
    # Deterministic key: ONE object per (methodology, day) — no run_id.
    assert key == "index/b300_basket/composites/annex_a_v0_2_calc_v1/2026-08-10.json"
    assert client.put_order[-1].endswith("latest.json")  # pointer moved last
    assert composite_exists(
        client, "curves", prefix="index/b300_basket", methodology_id="annex_a_v0_2_calc_v1", day="2026-08-10"
    )
    # A racing writer with the IDENTICAL computation is idempotent...
    again = upload_composite(
        client,
        "curves",
        _composite_payload(),
        prefix="index/b300_basket",
        run_id="20260811T041030Z-bbbb",  # different run — bytes identical
    )
    assert again["composite_key"] == key
    # ...but a divergent computation for the same day fails loudly.
    with pytest.raises(BucketPublishError, match="append-only"):
        upload_composite(
            client,
            "curves",
            _composite_payload(value=9.99),
            prefix="index/b300_basket",
            run_id="20260811T041100Z-cccc",
        )
    # Pointer never regresses: publishing an OLDER day keeps the newer pointer.
    upload_composite(
        client,
        "curves",
        _composite_payload(day="2026-08-11", value=7.6),
        prefix="index/b300_basket",
        run_id="20260812T041000Z-bbbb",
    )
    kept = upload_composite(
        client,
        "curves",
        _composite_payload(day="2026-08-09", value=7.4),
        prefix="index/b300_basket",
        run_id="20260812T051000Z-cccc",
    )
    assert kept["status"] == "published_pointer_kept"
    pointer = json.loads(
        client.objects["index/b300_basket/composites/annex_a_v0_2_calc_v1/latest.json"]
    )
    assert pointer["date"] == "2026-08-11"


# -------------------------------------------------------------------- CLI

_CLI_CACHE = {}


def _load_cli():
    if "mod" not in _CLI_CACHE:
        spec = importlib.util.spec_from_file_location(
            "compute_index_composite",
            REPO_ROOT / "scripts" / "compute_index_composite.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _seed_first_capture(client):
    payload = _live_first_capture_snapshot()
    client.objects[
        "index/b300_basket/snapshots/2026-08-10/slot16-20260810T211029Z-b128.json"
    ] = json.dumps(payload).encode()


def _wire_cli(monkeypatch, client, now, argv):
    cli = _load_cli()

    class StubConfig:
        bucket = "curves"

    monkeypatch.setattr(cli.BucketConfig, "from_env", staticmethod(lambda: StubConfig()))
    monkeypatch.setattr(cli, "make_client", lambda cfg: client)
    monkeypatch.setattr(cli, "ensure_rates", lambda *a, **k: dict(FX_2026_08_10))
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr("sys.argv", ["compute_index_composite.py", *argv])
    return cli


def test_cli_sync_computes_closed_days_and_is_idempotent(monkeypatch, capsys):
    client = FakeS3()
    _seed_first_capture(client)
    cli = _wire_cli(
        monkeypatch, client, datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc), ["--sync"]
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    # calc_v4: the vote-median day-1 value, hand-verified in
    # test_golden_first_live_capture (the retired weighted mean was 7.5340).
    assert "2026-08-10: 7.4750 $/GPU-hr (8 sources)" in out
    assert "composites written: 1" in out
    # 08-11 must NOT have been computed — its canonical window is still open.
    assert not composite_exists(
        client, "curves", prefix="index/b300_basket", methodology_id="annex_a_v0_2_calc_v6", day="2026-08-11"
    )

    # Second run: nothing new to write, and no drift warnings.
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "composites written: 0" in out
    assert "DRIFT" not in out


def test_cli_records_day_missed_after_full_close(monkeypatch, capsys):
    """A fully-missed CLOSED day gets an explicit basket_dark artifact —
    a hole in the series must be a record, not an absence."""
    client = FakeS3()
    _seed_first_capture(client)
    cli = _wire_cli(
        monkeypatch, client, datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc), ["--sync"]
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "2026-08-11: BASKET_DARK [day_missed]" in out
    assert composite_exists(
        client, "curves", prefix="index/b300_basket", methodology_id="annex_a_v0_2_calc_v6", day="2026-08-11"
    )


def test_cli_rejects_out_of_range_targets_and_open_windows(monkeypatch, capsys):
    client = FakeS3()
    _seed_first_capture(client)
    now = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)
    # Pre-genesis backfill typo must never look like a successful no-op.
    cli = _wire_cli(monkeypatch, client, now, ["--date", "2025-08-10"])
    assert cli.main() == 1
    assert "outside the replayable range" in capsys.readouterr().out
    # Explicit target whose canonical window is still open: loud exit 1.
    cli = _wire_cli(monkeypatch, client, now, ["--date", "2026-08-11"])
    assert cli.main() == 1
    assert "not yet computable" in capsys.readouterr().out


def test_cli_dry_run_writes_nothing(monkeypatch, capsys):
    client = FakeS3()
    _seed_first_capture(client)
    keys_before = set(client.objects)
    cli = _wire_cli(
        monkeypatch,
        client,
        datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc),
        ["--sync", "--dry-run"],
    )
    assert cli.main() == 0
    assert "2026-08-10: 7.4750" in capsys.readouterr().out
    assert set(client.objects) == keys_before


def test_cli_log_helpers_scrub_workflow_command_injection(monkeypatch, capsys):
    """Artifact-derived strings (verdict currency labels, run ids) ride
    warn()/notice()/error() verbatim — under GITHUB_ACTIONS a smuggled
    newline + '::' would inject a GH workflow command into the job log.
    Control chars must become spaces at the helper level."""
    cli = _load_cli()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    cli.warn("EUR\n::error::payload")
    assert capsys.readouterr().out == "::warning::EUR ::error::payload\n"
    cli.notice("a\rb")
    assert capsys.readouterr().out == "::notice::a b\n"
    cli.error("x\x00y")
    assert capsys.readouterr().out == "::error::x y\n"


def test_cli_pins_replay_to_published_days_and_warns_on_drift(monkeypatch, capsys):
    """Daily-lane review F1 (docs/adversarial-reviews.md): the raw store
    can grow AFTER publication (late upload with an
    earlier run_id). The published composite must stay the replay authority
    — history unchanged, loud DRIFT warning, artifact bytes untouched."""
    client = FakeS3()
    _seed_first_capture(client)
    now = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    composite_key = "index/b300_basket/composites/annex_a_v0_2_calc_v6/2026-08-10.json"
    published_bytes = client.objects[composite_key]

    # A late-landing snapshot with an EARLIER run_id becomes the raw store's
    # earliest key for slot16 — and it carries a different verda print.
    divergent = _live_first_capture_snapshot()
    divergent["run_id"] = "20260810T160500Z-0000"
    for entry in divergent["sources"]:
        if entry["source_id"] == "verda":
            entry["observations"] = [_obs("B300", usd=7.60)]
    client.objects[
        "index/b300_basket/snapshots/2026-08-10/slot16-20260810T160500Z-0000.json"
    ] = json.dumps(divergent).encode()

    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DRIFT 2026-08-10" in out
    assert "selection changed" in out
    assert "verda: published print 7.5" in out
    assert client.objects[composite_key] == published_bytes  # artifact stands
    assert "composites written: 0" in out


# --------------------------------------------------- review-pass hardening


def test_ecb_validation_rejects_poisoned_feeds():
    """USD rates persist immutably first-write-wins — a NaN/zero/out-of-band
    USD value or a non-date key must reject the whole document."""
    with pytest.raises(RuntimeError, match="implausible"):
        parse_ecb_rates(ECB_XML.replace("1.1555", "0.0"))
    with pytest.raises(RuntimeError, match="implausible"):
        parse_ecb_rates(ECB_XML.replace("1.1555", "nan"))
    with pytest.raises(RuntimeError, match="sane band"):
        parse_ecb_rates(ECB_XML.replace("1.1555", "9.9"))
    with pytest.raises(RuntimeError, match="non-date"):
        parse_ecb_rates(ECB_XML.replace("2026-08-10", "../evil"))


def test_ecb_glitched_exotic_currency_never_blinds_usd():
    """The 90d feed re-serves its whole window: rejecting the document over a
    bad NON-USD value would hold EUR conversion out for months. Drop the one
    value instead — nothing bad is stored, USD flows."""
    rates = parse_ecb_rates(ECB_XML.replace("183.0", "nan"))
    assert "JPY" not in rates["2026-08-08"]
    assert rates["2026-08-08"]["USD"] == 1.15
    assert rates["2026-08-10"]["USD"] == 1.1555


def test_ensure_rates_merges_persists_and_survives_feed_failure(monkeypatch, capsys):
    import gpu_index.index.fx as fx_mod

    client = FakeS3()
    monkeypatch.setattr(fx_mod, "fetch", lambda url, timeout=30: ECB_XML)
    stored = fx_mod.ensure_rates(client, "curves", prefix="index/b300_basket")
    assert stored["2026-08-10"]["rates"]["USD"] == 1.1555
    assert "index/b300_basket/fx/ecb-2026-08-10.json" in client.objects

    def _boom(url, timeout=30):
        raise RuntimeError("blocked")

    monkeypatch.setattr(fx_mod, "fetch", _boom)
    stored_again = fx_mod.ensure_rates(client, "curves", prefix="index/b300_basket")
    assert stored_again["2026-08-10"]["rates"]["USD"] == 1.1555  # stored history
    assert "ECB feed unavailable" in capsys.readouterr().out  # loud, not silent

    fresh_client = FakeS3()
    monkeypatch.setattr(fx_mod, "fetch", lambda url, timeout=30: ECB_XML)
    in_memory = fx_mod.ensure_rates(
        fresh_client, "curves", prefix="index/b300_basket", persist=False
    )
    assert in_memory["2026-08-10"]["rates"]["USD"] == 1.1555
    assert not any(k.startswith("index/b300_basket/fx/") for k in fresh_client.objects)


def test_calc_params_pin_for_methodology_v6():
    """THE drift tripwire: changing any value here without minting a NEW
    methodology_id silently recomputes historical filter decisions under
    the same series id. Param change == new methodology_id, always.
    calc_v2 == calc_v1 + the manual exclusion set (pinned below);
    calc_v3 == calc_v2 + the recorded-currency filter terms and
    the 0.05 sigma floor (pinned below); calc_v4 == calc_v3 + the
    median-of-CI-votes statistic; calc_v5 == calc_v4 + the dynamic_weights
    mint (predictive weighting — knobs, risk
    caps, AND the fallback vector pinned below); editing any of them
    without a mint is exactly the drift this test exists to catch."""
    params = calc_params(CONFIG)
    assert params["methodology_id"] == "annex_a_v0_2_calc_v6"
    # calc_v3: the filter runs in the recorded currency and the
    # band floors at 0.05 recorded-currency units.
    assert params["filter_terms"] == "recorded_currency"
    assert params["filter_sigma_floor"] == 0.05
    # calc_v4: the day prices as the median of CI votes.
    assert params["composite_statistic"] == "median_ci_votes"
    assert [
        (e["date"], e["source_id"]) for e in params["manual_exclusions"]
    ] == [
        ("2026-08-11", "vast"),
        ("2026-08-12", "vast"),
        ("2026-08-14", "vast"),
        ("2026-08-15", "vast"),
    ]
    # Reason text rides calc_params into every artifact's BYTES: a wording
    # tweak without a mint forks the embedded params mid-series. Pinned
    # verbatim on purpose — editing a reason must fail here.
    assert [e["reason"] for e in params["manual_exclusions"]] == [
        "capture defect: recorded print known wrong by rule; true print not captured",
        "capture defect: recorded print known wrong by rule; true print not captured",
        "capture defect: recorded print known wrong by rule; true print not captured",
        "capture defect: recorded print known wrong by rule; true print not captured",
    ]
    assert params["target_sku"] == "B300"
    assert params["interruptible_tiers"] == ("spot", "preemptible")
    assert params["filter_window"] == 20
    # calc_v4 (amended pre-publish): the fence
    # loosens to 3.0 — the median of CI votes is the primary outlier
    # defense now, the fence a gross-error screen.
    assert params["filter_sigma"] == 3.0
    assert params["filter_warmup"] == 10
    assert params["manual_verify_pct"] == 15.0
    assert params["promote_tie_break"] == "later"
    assert params["fx_max_staleness_days"] == 7
    assert params["drift_scan_days"] == 14
    assert params["fallback_pool_sku"] == "B200"
    # The pool vector carries three seats. A fourth seat was configured
    # here and never implemented: it held no collector, printed nothing on
    # any day of the series, and rode the vector as a permanent
    # no-print stub. It was dropped from the vector before the public
    # flip. The drop is inert to every published number (a seat with no
    # print never entered the pool mean) and the series is frozen and no
    # longer extended, so no methodology was minted for it; the frozen
    # artifacts keep their own bytes and --verify-published names the
    # difference as calc_params drift, which is what that mode is for.
    assert params["fallback_pool_sources"] == [
        "nebius",
        "e2e",
        "shadeform",
    ]
    # calc_v5: EVERYTHING that shapes a weight rides calc_params — knobs,
    # per-source risk caps, and the opening-weights fallback vector itself — so
    # the D2 refuse-to-extend fence covers weight-methodology edits like
    # any other param. Pinned verbatim.
    assert params["dynamic_weights"] == {
        "scheme": "predictive_v1",
        # R-slots: horizons in HOURS on the
        # capture slot grid; the grid itself is embedded so a cadence
        # change is a mint. 1h joins when capture walks to hourly.
        "lookback_horizons_hours": [6, 24, 48],
        "forward_horizons_hours": [6, 24, 48],
        "slot_hours_utc": [4, 10, 16, 22],
        "history_days": 90,
        "half_life_days": 30.0,
        "ridge_lambda": 1.0,
        "gamma": 4.0,
        "weight_min": 0.025,
        "weight_max": 0.30,
        # R-insample: the only sample gate —
        # q scores the single EW ridge fit in-sample, no OOS evaluations.
        "min_train_samples": 10,
        "target_variance_floor": 1e-12,
        # Pre-publish hardening amendments (2026-08-23): the switch quorum
        # (R-quorum, = the capture claim floor) and the per-return
        # winsorization cap (R-winsor).
        "switch_min_eligible": 5,
        "max_abs_log_return": 0.5,
        # calc_v6: the per-source risk caps are
        # REMOVED — softmax + global weight_max + weight_min + R-winsor are
        # the only fences. The allocator mechanism stays in code, unused.
        "source_weight_caps": {},
        "fallback_weights": {
            "hyperstack": 0.15,
            "latitude": 0.1,
            "massedcompute": 0.1,
            "nebius": 0.15,
            "runpod": 0.1,
            "scaleway": 0.15,
            "vast": 0.1,
            "verda": 0.15,
        },
    }


def test_calc_params_without_com1310_knobs_omits_them():
    """Absence pin (frozen v1/v2/v3/v4 artifact bytes): a config that does
    not set the calc_v3/calc_v4 knobs — or the calc_v5 dynamic_weights
    block — must produce calc_params WITHOUT them: an unconditional key
    (even at the legacy defaults) would fork a frozen series' embedded
    params without a methodology mint."""
    cfg = json.loads(json.dumps(CONFIG))
    del cfg["calc"]["filter_terms"]
    del cfg["calc"]["filter_sigma_floor"]
    del cfg["calc"]["composite_statistic"]
    del cfg["calc"]["dynamic_weights"]
    params = calc_params(cfg)
    assert "filter_terms" not in params
    assert "filter_sigma_floor" not in params
    assert "composite_statistic" not in params
    assert "dynamic_weights" not in params


def test_fx_staleness_config_actually_binds():
    """The config knob must reach the conversion path (was a dead knob)."""
    cfg = json.loads(json.dumps(CONFIG))
    cfg["calc"]["fx_max_staleness_days"] = 0
    stale_fx = {"2026-08-08": {"as_of": "2026-08-08", "rates": {"USD": 1.15}}}
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _entry(
                "scaleway",
                [_obs("B300", native=7.5, usd=None, currency="EUR", basis=8)],
            )
        ],
    }
    tightened = compute_day(
        config=cfg,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records=stale_fx,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    scaleway = next(s for s in tightened["sources"] if s["source_id"] == "scaleway")
    assert scaleway["status"] == "fx_unavailable"

    default_window = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records=stale_fx,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    scaleway = next(
        s for s in default_window["sources"] if s["source_id"] == "scaleway"
    )
    assert scaleway["status"] == "ok"  # 2-day-old rate inside the 7-day window


def test_mixed_currency_fx_outage_is_flagged_not_silent():
    entry = _entry(
        "mixed",
        [_obs("B300", usd=8.5), _obs("B300", native=7.5, usd=None, currency="EUR")],
    )
    chosen = daily_source_observation(
        entry, sku="B300", day="2026-08-10", fx_records={}, interruptible_tiers=()
    )
    assert chosen["usd_per_gpu_hr"] == 8.5
    assert chosen["fx_errors_partial"]  # the dropped EUR print is visible


def test_config_rejects_bad_calc_params(tmp_path):
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    for override, match in (
        ({"calc": {**base["calc"], "filter_window": 1}}, "filter_window"),
        ({"calc": {**base["calc"], "filter_warmup": 0}}, "filter_warmup"),
        ({"calc": {**base["calc"], "filter_sigma": 0}}, "filter_sigma"),
        # calc_v3 knobs: a negative/boolean floor or an unknown terms mode
        # must fail loudly at load, never default into a series.
        (
            {"calc": {**base["calc"], "filter_sigma_floor": -0.01}},
            "filter_sigma_floor",
        ),
        (
            {"calc": {**base["calc"], "filter_sigma_floor": True}},
            "filter_sigma_floor",
        ),
        ({"calc": {**base["calc"], "filter_terms": "native"}}, "filter_terms"),
        # calc_v4 knob: an unknown statistic must fail loudly at load —
        # and so must an explicit null, which calc_params (keyed on
        # presence) would otherwise embed as the string "None" and the
        # drift fence would pin forever.
        (
            {"calc": {**base["calc"], "composite_statistic": "mean_of_votes"}},
            "composite_statistic",
        ),
        (
            {"calc": {**base["calc"], "composite_statistic": None}},
            "composite_statistic",
        ),
        ({"genesis_date": "2026-13-99"}, "genesis_date"),
    ):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({**base, **override}))
        with pytest.raises(BasketConfigError, match=match):
            load_basket_config(p)


def test_drift_ignores_fx_rate_timing_but_flags_native_changes():
    """The day's real ECB rate landing after a walked-back publish is
    routine (R2) and must not page 48x/day forever; a changed NATIVE print
    is real drift."""
    cli = _load_cli()
    params = calc_params(CONFIG)
    stored = {
        "snapshot_run_id": "r",
        "substituted_from_slot": None,
        "day_missed": False,
        "sources": [
            {
                "source_id": "scaleway",
                "chosen": {
                    "usd_per_gpu_hr": 8.625,
                    "native_per_gpu_hr": 7.5,
                    "currency": "EUR",
                    "fx_rate": 1.15,
                    "fx_as_of": "2026-08-08",
                },
                "filter": {"accepted": True},
            }
        ],
    }
    same_native = {
        "run_id": "r",
        "sources": [
            _entry("scaleway", [_obs("B300", native=7.5, usd=None, currency="EUR", basis=8)])
        ],
    }
    msgs = cli.detect_drift(
        stored, (same_native, None), {16: same_native},
        day_str="2026-08-10", params=params, fx_records=FX_2026_08_10,
    )
    assert msgs == []  # newer rate, same native print: not drift

    repriced = {
        "run_id": "r",
        "sources": [
            _entry("scaleway", [_obs("B300", native=7.6, usd=None, currency="EUR", basis=8)])
        ],
    }
    msgs = cli.detect_drift(
        stored, (repriced, None), {16: repriced},
        day_str="2026-08-10", params=params, fx_records=FX_2026_08_10,
    )
    assert any("native print" in m for m in msgs)


def test_drift_flags_vanished_raw_store():
    cli = _load_cli()
    stored = {"day_missed": False, "sources": []}
    msgs = cli.detect_drift(
        stored, None, {}, day_str="2026-08-10",
        params=calc_params(CONFIG), fx_records={},
    )
    assert any("NO snapshots" in m for m in msgs)


def test_drift_scan_respects_horizon(monkeypatch, capsys, tmp_path):
    """Beyond drift_scan_days, published days advance history in one GET
    and skip the raw-store comparison entirely (48 firings/day forever)."""
    client = FakeS3()
    _seed_first_capture(client)
    now = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    # Divergent snapshot lands; with drift_scan_days=0 yesterday is beyond
    # the horizon, so no DRIFT line (and no snapshot reads for that day).
    divergent = _live_first_capture_snapshot()
    divergent["run_id"] = "20260810T160500Z-0000"
    client.objects[
        "index/b300_basket/snapshots/2026-08-10/slot16-20260810T160500Z-0000.json"
    ] = json.dumps(divergent).encode()
    cfg = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    cfg["calc"]["drift_scan_days"] = 0
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    cli = _wire_cli(monkeypatch, client, now, ["--sync", "--config", str(cfg_path)])
    assert cli.main() == 0
    assert "DRIFT" not in capsys.readouterr().out


def test_cli_refuses_out_of_order_target_publication(monkeypatch, capsys):
    """Publishing day D while an earlier computable day is unpublished would
    bake filter provenance the earlier artifacts may not reproduce."""
    client = FakeS3()
    _seed_first_capture(client)
    second = _live_first_capture_snapshot()
    second["run_id"] = "20260811T161000Z-aaaa"
    client.objects[
        "index/b300_basket/snapshots/2026-08-11/slot16-20260811T161000Z-aaaa.json"
    ] = json.dumps(second).encode()
    cli = _wire_cli(
        monkeypatch,
        client,
        datetime(2026, 8, 11, 17, 0, tzinfo=timezone.utc),
        ["--date", "2026-08-11"],
    )
    assert cli.main() == 1
    assert "earlier unpublished" in capsys.readouterr().out
    assert not composite_exists(
        client, "curves", prefix="index/b300_basket",
        methodology_id="annex_a_v0_2_calc_v6", day="2026-08-11",
    )


def test_composite_verify_after_write_and_corrupt_pointer():
    class CorruptingS3(FakeS3):
        def put_object(self, Bucket, Key, Body, **kwargs):
            if "/composites/" in Key and not Key.endswith("latest.json"):
                Body = Body + b"x"
            super().put_object(Bucket, Key, Body, **kwargs)

    client = CorruptingS3()
    with pytest.raises(BucketPublishError, match="Verify-after-write"):
        upload_composite(
            client, "curves", _composite_payload(), prefix="index/b300_basket", run_id="r"
        )
    assert (
        "index/b300_basket/composites/annex_a_v0_2_calc_v1/latest.json"
        not in client.objects
    )

    healthy = FakeS3()
    healthy.objects[
        "index/b300_basket/composites/annex_a_v0_2_calc_v1/latest.json"
    ] = b"{corrupt"
    out = upload_composite(
        healthy, "curves", _composite_payload(), prefix="index/b300_basket", run_id="r"
    )
    assert out["status"] == "published"  # corrupt pointer replaced, not fatal


# ------------------------------------------------ manual exclusions (calc_v2)


def test_manual_exclusion_excludes_print_and_window():
    """calc_v2: an excluded (date, source) contributes NO daily print and
    NO filter-window entry; the artifact names the source and says why.
    Golden: 2026-08-15 with the live-capture world = the 7-source
    renormalized value."""
    window_history: dict = {}
    payload = compute_day(
        config=CONFIG,
        day="2026-08-15",  # vast excluded by config on this date
        snapshot=_live_first_capture_snapshot(),
        substituted_from=None,
        window_history=window_history,
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    vast = next(s for s in payload["sources"] if s["source_id"] == "vast")
    assert vast["status"] == "manually_excluded"
    assert "capture defect" in vast["excluded_reason"]
    assert "chosen" not in vast and "filter" not in vast
    index = payload["index"]
    assert index["sources_used_count"] == 7
    assert "vast" not in index["renormalized_weights"]
    assert "vast" not in window_history  # never poisons sigma history
    # calc_v4 hand-walk, 7 sources (21 votes, W 2.70, all CIs the 0.05
    # floor; scaleway's converts to 0.057775): the median target 1.35
    # lands exactly on verda's mid vote 7.50 (cum 1.20→1.35) so the index
    # averages with verda's high vote (7.50 + 7.55)/2 = 7.525. The retired
    # weighted mean was (0.15*(7.50+7.85+7.40+8.66625)
    # + 0.10*(7.39+5.95+8.0)) / 0.90 = 7.607153.
    assert index["value_usd_gpu_hr"] == pytest.approx(7.525, abs=1e-6)
    assert index["weighted_mean_usd_gpu_hr"] == pytest.approx(
        7.607153, abs=1e-6
    )
    assert payload["basket_dark"] is False
    # calc_v5: the excluded source is outside the day's eligible set, so
    # the fallback weight vector omits it exactly like the window does.
    assert payload["weight_calc"]["mode"] == "fallback"
    assert payload["weight_calc"]["weights"] == {
        "hyperstack": 0.15,
        "latitude": 0.1,
        "massedcompute": 0.1,
        "nebius": 0.15,
        "runpod": 0.1,
        "scaleway": 0.15,
        "verda": 0.15,
    }


def test_manual_exclusion_leaves_other_days_alone():
    """The same world one day outside the exclusion window prices vast
    normally — exclusions are (date, source) pairs, never source-wide."""
    payload = compute_day(
        config=CONFIG,
        day="2026-08-16",  # not excluded; inside the fixture's FX window
        snapshot=_live_first_capture_snapshot(),
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    vast = next(s for s in payload["sources"] if s["source_id"] == "vast")
    assert vast["status"] == "ok"
    assert payload["index"]["sources_used_count"] == 8


def test_manual_exclusion_applies_to_fallback_pool():
    cfg = json.loads(json.dumps(CONFIG))
    cfg["calc"]["manual_exclusions"] = [
        {"date": "2026-08-10", "source_id": "e2e", "reason": "test ruling"}
    ]
    payload = compute_day(
        config=cfg,
        day="2026-08-10",
        snapshot=_live_first_capture_snapshot(),
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    pool = {p["source_id"]: p for p in payload["fallback_pool"]["sources"]}
    assert pool["e2e"]["status"] == "manually_excluded"
    # Pool mean recomputes over the remaining prints (nebius B200 7.15,
    # shadeform B200 3.74).
    assert payload["fallback_pool"]["mean_usd_gpu_hr"] == pytest.approx(
        (7.15 + 3.74) / 2
    )


def test_fallback_pool_rolls_every_configured_seat_including_no_print_ones():
    """The pool block is a COMPLETE roll of the configured pool vector: a
    seat that printed nothing is listed with the status that explains why
    (its recorded snapshot status, or 'missing' when the capture holds no
    entry for it at all), never silently dropped — a shrinking pool block
    would otherwise be indistinguishable from a shrinking pool vector. A
    no-print seat contributes nothing to the mean."""
    cfg = json.loads(json.dumps(CONFIG))
    cfg["calc"]["fallback_pool_sources"] = [
        "nebius",
        "e2e",
        "shadeform",
        "charlie",
        "zulu",
    ]
    snapshot = _live_first_capture_snapshot()
    # charlie: an entry with a recorded status and zero observations.
    # zulu: configured into the vector, absent from the capture entirely.
    snapshot["sources"].append(_entry("charlie", [], status="unimplemented"))
    payload = compute_day(
        config=cfg,
        day="2026-08-10",
        snapshot=snapshot,
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=_ws(),
        prior_slot_prints={},
    )
    pool = payload["fallback_pool"]
    pool_by_id = {p["source_id"]: p for p in pool["sources"]}
    assert set(pool_by_id) == {"nebius", "e2e", "shadeform", "charlie", "zulu"}
    assert pool_by_id["charlie"]["status"] == "unimplemented"
    assert pool_by_id["zulu"]["status"] == "missing"
    assert "chosen" not in pool_by_id["charlie"]
    assert "chosen" not in pool_by_id["zulu"]
    # The mean is the three printing seats only — unchanged by the two
    # no-print seats riding the vector.
    assert pool["mean_usd_gpu_hr"] == pytest.approx((7.15 + 6.99 + 3.74) / 3)


def test_manual_exclusions_are_deterministically_ordered():
    """Artifact bytes must not depend on config list order — racing
    writers with reordered configs would trip the append-only guard."""
    cfg = json.loads(json.dumps(CONFIG))
    cfg["calc"]["manual_exclusions"] = list(
        reversed(cfg["calc"]["manual_exclusions"])
    )
    assert (
        calc_params(cfg)["manual_exclusions"]
        == calc_params(CONFIG)["manual_exclusions"]
    )


def test_config_rejects_bad_manual_exclusions(tmp_path):
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    good = {"date": "2026-08-11", "source_id": "vast", "reason": "r"}
    for override, match in (
        ({**good, "date": "not-a-date"}, "YYYY-MM-DD"),
        ({**good, "source_id": "nonexistent"}, "not a configured source"),
        ({**good, "reason": "  "}, "non-empty reason"),
    ):
        cfg = json.loads(json.dumps(base))
        cfg["calc"]["manual_exclusions"] = [override]
        p = tmp_path / "c.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(BasketConfigError, match=match):
            load_basket_config(p)
    # Duplicate (date, source) pairs are a config bug, not a stronger ruling.
    cfg = json.loads(json.dumps(base))
    cfg["calc"]["manual_exclusions"] = [good, dict(good)]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(BasketConfigError, match="duplicate"):
        load_basket_config(p)


def test_drift_detector_skips_manually_excluded_entries():
    """The artifact diverging from raw is the POINT of an exclusion —
    warning on it 48x/day would bury real drift."""
    cli = _load_cli()
    params = calc_params(CONFIG)
    stored = {
        "snapshot_run_id": "r",
        "substituted_from_slot": None,
        "day_missed": False,
        "sources": [
            {
                "source_id": "vast",
                "weight": 0.10,
                "status": "manually_excluded",
                "excluded_reason": "capture defect",
            }
        ],
    }
    raw = {
        "run_id": "r",
        "sources": [_entry("vast", [_obs("B300", usd=10.9382, basis=8)])],
    }
    msgs = cli.detect_drift(
        stored, (raw, None), {16: raw},
        day_str="2026-08-15", params=params, fx_records={},
    )
    assert msgs == []


def test_v6_replay_writes_its_own_keyspace_and_never_touches_frozen_series(
    monkeypatch, capsys
):
    """End-to-end mint behavior (the dynamic_weights mint, the
    v2/v3/v4 pattern): the v5 keyspace starts EMPTY, so
    --sync under the v5 config replays from genesis into
    composites/annex_a_v0_2_calc_v6/ automatically — and the frozen
    v1/v2/v3/v4 artifacts are never touched. Day 1 prices scaleway from
    the persisted FX record (the calc_v2 bootstrap-mulligan behavior,
    unchanged), and the v5 artifact embeds the calc_v3 filter params,
    the calc_v4 statistic, AND the dynamic_weights block — with every
    published value byte-identical to calc_v4 (fallback mode)."""
    client = FakeS3()
    _seed_first_capture(client)
    # Frozen prior-methodology artifacts must never be touched by the
    # v5 replay — not read as history, not rewritten.
    frozen = {
        "index/b300_basket/composites/annex_a_v0_2_calc_v1/2026-08-10.json": b'{"frozen": "v1"}',
        "index/b300_basket/composites/annex_a_v0_2_calc_v2/2026-08-10.json": b'{"frozen": "v2"}',
        "index/b300_basket/composites/annex_a_v0_2_calc_v2/latest.json": b'{"frozen": "v2-pointer"}',
        "index/b300_basket/composites/annex_a_v0_2_calc_v3/2026-08-10.json": b'{"frozen": "v3"}',
        "index/b300_basket/composites/annex_a_v0_2_calc_v3/latest.json": b'{"frozen": "v3-pointer"}',
        "index/b300_basket/composites/annex_a_v0_2_calc_v4/2026-08-10.json": b'{"frozen": "v4"}',
        "index/b300_basket/composites/annex_a_v0_2_calc_v4/latest.json": b'{"frozen": "v4-pointer"}',
        # The v5 fixture carries the REAL frozen day-1 value: the config
        # names annex_a_v0_2_calc_v5 as fallback_parity_methodology_id, so
        # the replay ALSO exercises the parity tripwire — a silent run here
        # is a genuine machine-checked byte-parity proof, and a dummy value
        # would (correctly) redden the firing.
        "index/b300_basket/composites/annex_a_v0_2_calc_v5/2026-08-10.json": b'{"frozen": "v5", "index": {"value_usd_gpu_hr": 7.475}}',
        "index/b300_basket/composites/annex_a_v0_2_calc_v5/latest.json": b'{"frozen": "v5-pointer"}',
    }
    client.objects.update(frozen)
    cli = _wire_cli(
        monkeypatch, client, datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc), ["--sync"]
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "FALLBACK PARITY" not in out  # the tripwire ran and agreed
    # Genesis day recomputes from raw under v4: every day-1 verdict is a
    # warm-up accept (unchanged), but the day now PRICES as the median of
    # CI votes — every source votes at the 0.05 floor (scaleway's converts
    # at the day's ECB rate), and the median lands on an exact weight
    # boundary: (7.45 + 7.50) / 2 = 7.475 (hand-verified; the retired
    # weighted mean, 7.5340, stays in the artifact as a diagnostic).
    assert "2026-08-10: 7.4750 $/GPU-hr (8 sources)" in out
    v5_key = "index/b300_basket/composites/annex_a_v0_2_calc_v6/2026-08-10.json"
    assert v5_key in client.objects
    for key, frozen_bytes in frozen.items():
        assert client.objects[key] == frozen_bytes
    stored = json.loads(client.objects[v5_key])
    assert stored["methodology_id"] == "annex_a_v0_2_calc_v6"
    assert stored["index"]["value_usd_gpu_hr"] == 7.475
    assert stored["index"]["statistic"] == "median_ci_votes"
    assert stored["index"]["weighted_mean_usd_gpu_hr"] == 7.533988
    # The calc_v3 filter params, the calc_v4 statistic, and the calc_v5
    # dynamic_weights block ride every v5 artifact's calc_params — and the
    # day-one weight_calc audit block records fallback mode.
    assert stored["calc_params"]["filter_terms"] == "recorded_currency"
    assert stored["calc_params"]["filter_sigma_floor"] == 0.05
    assert stored["calc_params"]["composite_statistic"] == "median_ci_votes"
    assert stored["calc_params"]["dynamic_weights"]["scheme"] == "predictive_v1"
    assert stored["weight_calc"]["mode"] == "fallback"
    assert [
        (e["date"], e["source_id"])
        for e in stored["calc_params"]["manual_exclusions"]
    ] == [
        ("2026-08-11", "vast"),
        ("2026-08-12", "vast"),
        ("2026-08-14", "vast"),
        ("2026-08-15", "vast"),
    ]
    # Every verdict records the filter's operating currency (calc_v3):
    # EUR for scaleway's native window, USD for everyone else.
    by_id = {s["source_id"]: s for s in stored["sources"] if s.get("filter")}
    assert by_id["scaleway"]["filter"]["currency"] == "EUR"
    assert by_id["verda"]["filter"]["currency"] == "USD"
    # Sanity pin for ruling D1: none of the REAL B300 sources has ever
    # printed a currency anomaly (scaleway is EUR-with-native since
    # genesis), so the fail-closed guard changes NOTHING for the real
    # series — no verdict carries any D1 flag, and the golden values above
    # (plus the 08-20 flip goldens in the filter tests) are unchanged.
    for detail in stored["sources"]:
        verdict = detail.get("filter") or {}
        assert "untrusted_currency" not in verdict
        assert "currency_mismatch" not in verdict
        assert "currency_confirmed" not in verdict


def _seed_currency_change_world(client):
    """08-10 EUR scaleway (the real capture) + 08-11/12/13 USD scaleway:
    a genuine currency change that confirms on its third USD day."""
    _seed_first_capture(client)
    for i, day in enumerate(("2026-08-11", "2026-08-12", "2026-08-13")):
        snap = _live_first_capture_snapshot()
        snap["run_id"] = f"2026081{i + 1}T161000Z-ccc{i}"
        for entry in snap["sources"]:
            if entry["source_id"] == "scaleway":
                # Scaleway starts publishing USD list prices.
                entry["observations"] = [
                    _obs("B300", usd=8.70 + i * 0.01, basis=8)
                ]
        client.objects[
            f"index/b300_basket/snapshots/{day}/slot16-"
            f"{snap['run_id']}.json"
        ] = json.dumps(snap).encode()


def test_cli_currency_change_is_loud_red_and_confirms_on_day_three(
    monkeypatch, capsys
):
    """Ruling D1 end to end: the two mismatch days publish HELD-OUT
    scaleway verdicts with a WARNING and exit 1 (loud, the
    exclusion-conflict precedent — still publishing); the third
    consecutive USD day confirms the change (warned, not red), reseeds the
    window, and the replay run is quiet, drift-free, and byte-stable."""
    client = FakeS3()
    _seed_currency_change_world(client)
    now = datetime(2026, 8, 14, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 1  # mismatch days redden the firing
    out = capsys.readouterr().out
    assert "composites written: 4" in out  # ...but every day published
    assert (
        "WARNING: 2026-08-11: scaleway recorded currency changed EUR -> USD "
        "— held out (1/3 toward confirmation); window preserved" in out
    )
    assert "held out (2/3 toward confirmation)" in out
    assert (
        "WARNING: 2026-08-13: scaleway currency change CONFIRMED EUR -> USD"
        in out
    )

    def _stored_verdict(day):
        stored = json.loads(
            client.objects[
                f"index/b300_basket/composites/annex_a_v0_2_calc_v6/{day}.json"
            ]
        )
        return next(
            s for s in stored["sources"] if s["source_id"] == "scaleway"
        ), stored

    day1, stored1 = _stored_verdict("2026-08-11")
    assert day1["filter"]["currency_mismatch"] is True
    assert day1["filter"]["accepted"] is False
    assert (day1["filter"]["pending_count"], day1["filter"]["confirm_after"]) == (1, 3)
    # Held out: the mismatch print reaches neither the index nor the
    # renormalized weights (fail-closed).
    assert "scaleway" not in stored1["index"]["renormalized_weights"]
    day2, _ = _stored_verdict("2026-08-12")
    assert day2["filter"]["pending_count"] == 2
    day3, stored3 = _stored_verdict("2026-08-13")
    assert day3["filter"] == {
        "accepted": True,
        "unfiltered": True,
        "currency_confirmed": True,
        "currency": "USD",
        "window_currency": "EUR",
        "filter_price": 8.72,
        "n_history": 3,
        # The confirmation day is unfiltered, so the R3 warm-up screen
        # still applies — 8.72 sits ~17% above the day's basket mean.
        "manual_verify": True,
    }
    assert "scaleway" in stored3["index"]["renormalized_weights"]

    # Replay: state rebuilds from the published artifacts through the SAME
    # advance rule — quiet, drift-free, byte-stable, exit 0 (the anomaly
    # was loud on its own days only).
    published = dict(client.objects)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "composites written: 0" in out
    assert "DRIFT" not in out
    assert "recorded currency changed" not in out
    assert client.objects == published


def test_cli_untrusted_currency_is_loud_and_red(monkeypatch, capsys):
    """Ruling D1: an untrusted filter input (UNKNOWN label) on a computed
    day warns, exits 1, and still publishes — with the source held out and
    its window preserved for the next trustworthy day."""
    client = FakeS3()
    _seed_first_capture(client)
    snap = _live_first_capture_snapshot()
    snap["run_id"] = "20260811T161000Z-uuuu"
    for entry in snap["sources"]:
        if entry["source_id"] == "scaleway":
            bad = _obs("B300", usd=8.66, native=7.5, basis=8)
            bad["currency"] = "UNKNOWN"
            entry["observations"] = [bad]
    client.objects[
        "index/b300_basket/snapshots/2026-08-11/slot16-20260811T161000Z-uuuu.json"
    ] = json.dumps(snap).encode()
    now = datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "composites written: 2" in out  # still publishes
    assert (
        "WARNING: 2026-08-11: scaleway untrusted currency label 'UNKNOWN' "
        "— print held out fail-closed; window preserved" in out
    )
    stored = json.loads(
        client.objects[
            "index/b300_basket/composites/annex_a_v0_2_calc_v6/2026-08-11.json"
        ]
    )
    scaleway = next(s for s in stored["sources"] if s["source_id"] == "scaleway")
    assert scaleway["filter"]["untrusted_currency"] is True
    assert scaleway["filter"]["accepted"] is False
    assert "scaleway" not in stored["index"]["renormalized_weights"]


def test_cli_replay_is_deterministic_across_restarts(monkeypatch, capsys):
    """Ruling D1 determinism pin: the same world replayed (a) in one shot
    and (b) day by day across container restarts (fresh in-memory state
    each run — exactly what the pending tracker must survive) produces
    BYTE-IDENTICAL artifacts, currency change included; and re-running the
    full sync changes nothing."""
    world_a = FakeS3()
    _seed_currency_change_world(world_a)
    final_now = datetime(2026, 8, 14, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, world_a, final_now, ["--sync"])
    assert cli.main() == 1  # the mismatch days redden this world too
    capsys.readouterr()

    def _artifacts(client):
        # Day artifacts only: the latest.json pointer legitimately embeds
        # run_id/published_at, which differ across replay schedules.
        return {
            k: v
            for k, v in client.objects.items()
            if "/composites/" in k and not k.endswith("latest.json")
        }

    artifacts_a = _artifacts(world_a)

    # World B: identical snapshots, but synced one day at a time with a
    # FRESH process each day (mid-series restarts).
    world_b = FakeS3()
    _seed_currency_change_world(world_b)
    for day_offset in range(4):
        run_now = datetime(2026, 8, 11 + day_offset, 5, 0, tzinfo=timezone.utc)
        cli = _wire_cli(monkeypatch, world_b, run_now, ["--sync"])
        cli.main()  # exit codes vary by day (mismatch days are red)
        capsys.readouterr()
    artifacts_b = _artifacts(world_b)
    assert set(artifacts_a) == set(artifacts_b)
    for key, blob in artifacts_a.items():
        assert artifacts_b[key] == blob, key

    # And a second full sync of world A is a byte-stable no-op.
    cli = _wire_cli(monkeypatch, world_a, final_now, ["--sync"])
    assert cli.main() == 0
    assert "composites written: 0" in capsys.readouterr().out
    assert _artifacts(world_a) == artifacts_a


def test_cli_refuses_to_extend_series_on_calc_params_drift(
    monkeypatch, capsys, tmp_path
):
    """Ruling D2: the mint rule made mechanical for EVERY param — after a
    day publishes, a live config whose calc_params drift from the last
    published artifact's (any key, either direction) errors loudly naming
    the key, exits 1, and refuses to publish new days under it."""
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    drifted_configs = {
        "filter_sigma": {"filter_sigma": 2.0},
        "filter_sigma_floor": {"filter_sigma_floor": 0.10},
        "filter_terms": {"filter_terms": "usd"},
        "filter_window": {"filter_window": 15},
        "filter_warmup": {"filter_warmup": 5},
        "manual_verify_pct": {"manual_verify_pct": 20},
        "fallback_pool_sources": {
            "fallback_pool_sources": ["nebius", "e2e"]
        },
    }
    for key, override in drifted_configs.items():
        client = FakeS3()
        _seed_first_capture(client)
        second = _live_first_capture_snapshot()
        second["run_id"] = "20260811T161000Z-dddd"
        client.objects[
            "index/b300_basket/snapshots/2026-08-11/slot16-20260811T161000Z-dddd.json"
        ] = json.dumps(second).encode()
        now = datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc)
        cli = _wire_cli(monkeypatch, client, now, ["--sync"])
        assert cli.main() == 0
        capsys.readouterr()

        cfg = json.loads(json.dumps(base))
        cfg["calc"] = {**cfg["calc"], **override}
        cfg_path = tmp_path / f"drift-{key}.json"
        cfg_path.write_text(json.dumps(cfg))
        later = datetime(2026, 8, 13, 5, 0, tzinfo=timezone.utc)
        cli = _wire_cli(
            monkeypatch, client, later, ["--sync", "--config", str(cfg_path)]
        )
        assert cli.main() == 1, key
        out = capsys.readouterr().out
        assert "calc_params drift" in out, key
        assert f"'{key}'" in out, key
        assert "mint a new methodology_id" in out, key
        assert "NOT published" in out, key
        assert not composite_exists(
            client, "curves", prefix="index/b300_basket",
            methodology_id="annex_a_v0_2_calc_v6", day="2026-08-12",
        ), key
    # The mint's headline fence claim, proven end-to-end (review finding
    # 2026-08-23): weight-METHODOLOGY edits — a gamma bump, a risk-cap
    # change, and a source's opening weight (the fallback vector, which
    # sat OUTSIDE calc_params entirely before calc_v5) — drift the
    # embedded dynamic_weights block and refuse to extend the series.
    dynamic_drifts = {
        "gamma": lambda cfg: cfg["calc"]["dynamic_weights"].update(
            {"gamma": 6.0}
        ),
        # calc_v6 removed the per-source caps; REINTRODUCING one without a
        # mint is exactly the drift the fence must refuse.
        "caps": lambda cfg: cfg["calc"]["dynamic_weights"].update(
            {"source_weight_caps": {"vast": 0.2}}
        ),
        "fallback_weights": lambda cfg: (
            cfg["sources"][0].update({"weight": 0.2}),
            cfg["sources"][1].update({"weight": 0.1}),
        ),
    }
    for label, mutate in dynamic_drifts.items():
        client = FakeS3()
        _seed_first_capture(client)
        now = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)
        cli = _wire_cli(monkeypatch, client, now, ["--sync"])
        assert cli.main() == 0
        capsys.readouterr()
        cfg = json.loads(json.dumps(base))
        mutate(cfg)
        cfg_path = tmp_path / f"dyn-drift-{label}.json"
        cfg_path.write_text(json.dumps(cfg))
        second = _live_first_capture_snapshot()
        second["run_id"] = "20260811T161000Z-eeee"
        client.objects[
            "index/b300_basket/snapshots/2026-08-11/slot16-20260811T161000Z-eeee.json"
        ] = json.dumps(second).encode()
        later = datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc)
        cli = _wire_cli(
            monkeypatch, client, later, ["--sync", "--config", str(cfg_path)]
        )
        assert cli.main() == 1, label
        out = capsys.readouterr().out
        assert "calc_params drift" in out, label
        assert "'dynamic_weights'" in out, label
        assert not composite_exists(
            client, "curves", prefix="index/b300_basket",
            methodology_id="annex_a_v0_2_calc_v6", day="2026-08-11",
        ), label

    # A REMOVED key drifts too (union of keys, both directions). NOTE:
    # removing filter_sigma_floor no longer reaches the drift fence — the
    # calc_v4 coupling (median_ci_votes requires a positive floor) refuses
    # it at config LOAD, pinned by
    # test_config_rejects_median_votes_without_a_positive_floor — so the
    # removed key exercised here is filter_terms.
    client = FakeS3()
    _seed_first_capture(client)
    now = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    cfg = json.loads(json.dumps(base))
    del cfg["calc"]["filter_terms"]
    cfg_path = tmp_path / "drift-removed.json"
    cfg_path.write_text(json.dumps(cfg))
    later = datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc)
    second = _live_first_capture_snapshot()
    second["run_id"] = "20260811T161000Z-eeee"
    client.objects[
        "index/b300_basket/snapshots/2026-08-11/slot16-20260811T161000Z-eeee.json"
    ] = json.dumps(second).encode()
    cli = _wire_cli(
        monkeypatch, client, later, ["--sync", "--config", str(cfg_path)]
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "'filter_terms'" in out
    # The unchanged config still extends the series cleanly.
    cli = _wire_cli(monkeypatch, client, later, ["--sync"])
    assert cli.main() == 0
    assert composite_exists(
        client, "curves", prefix="index/b300_basket",
        methodology_id="annex_a_v0_2_calc_v6", day="2026-08-11",
    )


def test_config_rejects_non_canonical_exclusion_dates(tmp_path):
    """fromisoformat accepts compact/week ISO forms that the exact-string
    exclusion match can never fire on — a silently inert incident exclusion
    is the worst failure mode this mechanism can have."""
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    for bad_date in ("20260811", "2026-W33-2"):
        cfg = json.loads(json.dumps(base))
        cfg["calc"]["manual_exclusions"] = [
            {"date": bad_date, "source_id": "vast", "reason": "r"}
        ]
        p = tmp_path / "c.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(BasketConfigError, match="canonical"):
            load_basket_config(p)
    # genesis_date gets the same round-trip fence.
    cfg = json.loads(json.dumps(base))
    cfg["genesis_date"] = "20260810"
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(BasketConfigError, match="canonical"):
        load_basket_config(p)


def _seed_excluded_day_world(client):
    """08-10 clean + 08-11 with the REAL poisoned vast print in the raw
    store — the excluded-day world, end to end."""
    _seed_first_capture(client)
    poisoned = _live_first_capture_snapshot()
    poisoned["run_id"] = "20260811T161004Z-32b7"
    for entry in poisoned["sources"]:
        if entry["source_id"] == "vast":
            entry["observations"] = [_obs("B300", usd=10.0007, basis=8)]
    client.objects[
        "index/b300_basket/snapshots/2026-08-11/slot16-20260811T161004Z-32b7.json"
    ] = json.dumps(poisoned).encode()


def test_cli_publishes_excluded_day_end_to_end(monkeypatch, capsys):
    """The whole chain on real snapshot data: 2026-08-11 publishes as a
    7-source composite with vast manually_excluded + the reason verbatim,
    the poisoned print stays out of the filter window, and the second run
    is drift-silent."""
    client = FakeS3()
    _seed_excluded_day_world(client)
    now = datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "2026-08-11: " in out and "(7 sources)" in out

    stored = json.loads(
        client.objects[
            "index/b300_basket/composites/annex_a_v0_2_calc_v6/2026-08-11.json"
        ]
    )
    vast = next(s for s in stored["sources"] if s["source_id"] == "vast")
    assert vast["status"] == "manually_excluded"
    assert "capture defect" in vast["excluded_reason"]
    assert "chosen" not in vast
    assert "vast" not in stored["index"]["renormalized_weights"]

    # Second run: replay advances from the published artifact — no vast
    # window entry, no DRIFT (the artifact-vs-raw divergence is the point).
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DRIFT" not in out
    assert "composites written: 0" in out


def test_cli_refuses_to_extend_series_when_exclusions_contradict_published(
    monkeypatch, capsys, tmp_path
):
    """The mint-v3 rule is MECHANICAL: an exclusion edit touching a
    published day (either direction) errors loudly, exits 1, and blocks
    new-day publication under the contradicting config."""
    client = FakeS3()
    _seed_excluded_day_world(client)
    now = datetime(2026, 8, 12, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    # Direction 1: ADD an exclusion for the published-clean 2026-08-10.
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    cfg = json.loads(json.dumps(base))
    cfg["calc"]["manual_exclusions"].append(
        {"date": "2026-08-10", "source_id": "vast", "reason": "forbidden edit"}
    )
    cfg_path = tmp_path / "add.json"
    cfg_path.write_text(json.dumps(cfg))
    cli = _wire_cli(
        monkeypatch, client, now, ["--sync", "--config", str(cfg_path)]
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "contradict the PUBLISHED" in out
    assert "mint a new methodology_id" in out

    # Direction 2: REMOVE the exclusion the published 2026-08-11 pins.
    cfg = json.loads(json.dumps(base))
    cfg["calc"]["manual_exclusions"] = [
        e for e in cfg["calc"]["manual_exclusions"] if e["date"] != "2026-08-11"
    ]
    cfg_path = tmp_path / "remove.json"
    cfg_path.write_text(json.dumps(cfg))
    cli = _wire_cli(
        monkeypatch, client, now, ["--sync", "--config", str(cfg_path)]
    )
    assert cli.main() == 1
    assert "contradict the PUBLISHED" in capsys.readouterr().out

    # And a contradicting config must never publish a NEW day: advance the
    # clock so 2026-08-12 is computable, keep the forbidden add.
    later = datetime(2026, 8, 13, 5, 0, tzinfo=timezone.utc)
    cfg_path = tmp_path / "add.json"
    cli = _wire_cli(
        monkeypatch, client, later, ["--sync", "--config", str(cfg_path)]
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "NOT published" in out
    assert not composite_exists(
        client, "curves", prefix="index/b300_basket",
        methodology_id="annex_a_v0_2_calc_v6", day="2026-08-12",
    )
