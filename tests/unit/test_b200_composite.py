# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""The B200 daily composite — now under annex_a2_v0_3_calc_v5
(the shared calculator's dynamic_weights mint;
fallback mode, so every calc_v3 golden value below is unchanged).

One calculator, parameterized by basket config. Pins here:

  - the B300 lane's calc_params SHAPE (the byte-identity tripwire: a lane
    knob leaking an unconditional key would change B300 artifact bytes
    mid-series without a methodology mint);
  - the day-one golden from the REAL first captured B200 snapshot
    (tests/fixtures/; every price-bearing byte as captured);
  - lane rules: no warm-up publish gate, 5-of-9 claim
    floor, the vast order-book statistic, USD-only (fx_lane none);
  - replay discipline identical to B300 calc_v2: params-pin, replay
    determinism, drift via the SAME statistic, mechanical mint rule,
    pointer no-regress, and the b200 job's capture-then-composite wiring.
"""

from __future__ import annotations

import importlib.util
import io
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_index.index.composite import (
    _weighted_median,
    calc_params,
    compute_day,
    resolve_daily_print,
    vast_vwm_verified_us_ca,
)
from gpu_index.index.config import BasketConfigError, load_basket_config
from gpu_index.index.snapshot import build_capture_snapshot
from gpu_index.common.store import (
    composite_exists,
    snapshot_bytes,
    upload_composite,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
B200_CONFIG_PATH = REPO_ROOT / "config" / "index_basket_b200.json"
B200_CONFIG = load_basket_config(B200_CONFIG_PATH)
B300_CONFIG = load_basket_config(REPO_ROOT / "config" / "index_basket.json")

# The REAL first captured B200 snapshot (production store, slot 22 on the
# 2026-08-16 genesis day). Every price-bearing byte is as captured; the
# capturer job/hostname labels and one prose note were sanitized for
# publication.
FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "b200_snapshot_2026-08-16_slot22-20260816T221613Z-1fdf.json"
)
SNAPSHOT_KEY = (
    "index/b200_basket/snapshots/2026-08-16/slot22-20260816T221613Z-1fdf.json"
)
COMPOSITE_KEY_DAY1 = (
    "index/b200_basket/composites/annex_a2_v0_3_calc_v5/2026-08-16.json"
)
POINTER_KEY = "index/b200_basket/composites/annex_a2_v0_3_calc_v5/latest.json"


def _ws():
    """Fresh calc_v4 weight state — REQUIRED by compute_day whenever the
    config sets calc.dynamic_weights (the CLI threads its own)."""
    return {"prices": {}, "vectors": {}, "mode": "fallback"}


# Hand-verified day-one goldens from the fixture. vast is manually
# excluded on 2026-08-16 (ruling A: the capture truncated the §5
# population to the whole-book cheapest 5), so day one is 8 constituents.
# calc_v3 (the median-of-votes mint) prices the day as the median of CI votes: every
# day-one CI is the 0.05 floor (USD lane), 24 votes, total weight 2.76.
# Hand-walked ladder: the median target 1.38 is crossed inside the 6.74
# votes (runpod's low and lambda's high; cum 1.26→1.36→1.48), so the
# index is 6.74; p25 (target 0.69) is crossed inside verda's low vote
# 6.06 (cum 0.66→0.78); p75 (target 2.07) inside together's low vote 8.14
# (cum 2.04→2.16). Confidence = max(6.74−6.06, 8.14−6.74) = 1.40 — wide,
# because the day-one B200 book genuinely spans 4.71→8.60. The retired
# weighted mean rides as a diagnostic:
#   (0.12*(6.11+7.15+6.00+6.69+8.60+8.19) + 0.10*(4.7125+6.79)) / 0.92
GOLDEN_DAY1_VALUE = 6.74
GOLDEN_DAY1_P25 = 6.06
GOLDEN_DAY1_P75 = 8.14
GOLDEN_DAY1_CONFIDENCE = 1.40
GOLDEN_DAY1_WEIGHTED_MEAN = 6.825054
GOLDEN_DAY1_UNWEIGHTED = 6.780312
# On a non-excluded day the same recorded book prices vast by the §5
# statistic: NOT the book minimum (4.1271 sits on a deverified host).
GOLDEN_VAST_STATISTIC = 5.8757
VAST_EXCLUSION_REASON = (
    "capture defect: recorded print known wrong by rule; "
    "true print not captured"
)


def _fixture_snapshot() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)


# ------------------------------------------------------------------ fakes
# (Same fakes as test_index_composite.py — tests/unit is not a package, so
# the helpers are kept local rather than imported across test modules.)


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


_CLI_CACHE = {}


def _load_cli():
    if "mod" not in _CLI_CACHE:
        spec = importlib.util.spec_from_file_location(
            "compute_index_composite_b200",
            REPO_ROOT / "scripts" / "compute_index_composite.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _wire_cli(monkeypatch, client, now, argv):
    cli = _load_cli()

    class StubConfig:
        bucket = "curves"

    monkeypatch.setattr(
        cli.BucketConfig, "from_env", staticmethod(lambda: StubConfig())
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: client)
    # fx_lane none: the ECB lane must never be touched for this basket.
    monkeypatch.setattr(
        cli,
        "ensure_rates",
        lambda *a, **k: pytest.fail("USD-only basket reached the FX lane"),
    )
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(
        "sys.argv",
        ["compute_index_composite.py", "--config", str(B200_CONFIG_PATH), *argv],
    )
    return cli


def _seed_day1(client):
    client.objects[SNAPSHOT_KEY] = FIXTURE_PATH.read_bytes()


def _obs(sku="B200", usd=6.0, tier="on-demand", basis=1, **extra):
    return {
        "sku": sku,
        "price_usd_gpu_hr": usd,
        "price_native_per_gpu_hr": usd,
        "currency": "USD",
        "tier": tier,
        "gpu_count_basis": basis,
        "raw_value": str(usd),
        "raw_unit": "usd_per_gpu_hr",
        "implausible": False,
        "notes": "",
        **extra,
    }


# ------------------------------------------------------------- params pins


def test_b200_params_pin_for_annex_a2_v0_3_calc_v5():
    """THE drift tripwire for the B200 series: any value change without a
    NEW methodology_id silently recomputes history under the same id.
    calc_v2 == calc_v1 + the recorded-currency filter terms and
    0.05 sigma floor (the shared calculator's mint — this lane is USD-only,
    so the floor is its only behavioral change); calc_v3 == calc_v2 + the
    median-of-CI-votes statistic (the shared calculator's second
    mint); calc_v4 == calc_v3 + the dynamic_weights mint (the shared
    calculator's third — predictive weighting;
    fallback mode keeps calc_v3 values byte-identical)."""
    params = calc_params(B200_CONFIG)
    assert params == {
        "methodology_id": "annex_a2_v0_3_calc_v5",
        "target_sku": "B200",
        "interruptible_tiers": ("spot", "preemptible"),
        "filter_window": 20,
        "filter_sigma": 3.0,
        "filter_warmup": 10,
        # Recorded-currency mint (shared calculator):
        "filter_terms": "recorded_currency",
        "filter_sigma_floor": 0.05,
        # Median-of-votes mint (shared calculator):
        "composite_statistic": "median_ci_votes",
        "manual_verify_pct": 15.0,
        "promote_tie_break": "later",
        "fx_max_staleness_days": 7,
        "drift_scan_days": 14,
        "fallback_pool_sku": "B300",
        "fallback_pool_sources": [
            "verda",
            "nebius",
            "hyperstack",
            "massedcompute",
            "runpod",
            "vast",
        ],
        # Day one's vast recording was
        # truncated by the capture; the reason rides every artifact's
        # bytes verbatim — editing it requires a methodology mint.
        "manual_exclusions": [
            {
                "date": "2026-08-16",
                "source_id": "vast",
                "reason": VAST_EXCLUSION_REASON,
            }
        ],
        # Lane rules (all final):
        "min_sources_to_claim": 5,
        "source_statistics": {"vast": "vast_vwm_verified_us_ca"},
        "fx_lane": "none",
        # calc_v4 (dynamic_weights mint): knobs, per-source risk caps, AND
        # the opening-weights fallback vector ride calc_params verbatim, so the
        # mint fence covers weight-methodology edits like any other param.
        "dynamic_weights": {
            "scheme": "predictive_v1",
            # R-slots + R-insample: horizons
            # in HOURS on the embedded capture slot grid; min_train is the
            # only sample gate (in-sample scoring, no OOS evaluations).
            "lookback_horizons_hours": [6, 24, 48],
            "forward_horizons_hours": [6, 24, 48],
            "slot_hours_utc": [4, 10, 16, 22],
            "history_days": 90,
            "half_life_days": 30.0,
            "ridge_lambda": 1.0,
            "gamma": 4.0,
            "weight_min": 0.025,
            "weight_max": 0.30,
            "min_train_samples": 10,
            "target_variance_floor": 1e-12,
            # Pre-publish hardening amendments (2026-08-23): switch quorum
            # (R-quorum, = the claim floor) + winsorization cap (R-winsor).
            "switch_min_eligible": 5,
            "max_abs_log_return": 0.5,
            # a2 calc_v5: per-source risk caps
            # removed — the same ruling as the B300 calc_v6 mint.
            "source_weight_caps": {},
            "fallback_weights": {
                "coreweave": 0.12,
                "hyperstack": 0.12,
                "lambda": 0.12,
                "massedcompute": 0.1,
                "nebius": 0.12,
                "runpod": 0.1,
                "together": 0.12,
                "vast": 0.08,
                "verda": 0.12,
            },
        },
    }


def test_b300_calc_params_shape_untouched_by_the_lane_knobs():
    """B300 byte-identity: the lane knobs must be ABSENT from the B300
    lane's calc_params (they are embedded verbatim in every artifact, so
    presence alone would change published-series bytes), and a B300
    artifact's serialized bytes must not carry any of the new key names."""
    params = calc_params(B300_CONFIG)
    assert set(params) == {
        "methodology_id",
        "target_sku",
        "interruptible_tiers",
        "filter_window",
        "filter_sigma",
        # The recorded-currency filter knobs and the vote statistic are NOT lane
        # knobs: both live configs set them (each under a freshly minted
        # methodology_id).
        "filter_sigma_floor",
        "filter_terms",
        "composite_statistic",
        "filter_warmup",
        "manual_verify_pct",
        "promote_tie_break",
        "fx_max_staleness_days",
        "drift_scan_days",
        "fallback_pool_sku",
        "fallback_pool_sources",
        "manual_exclusions",
        # The dynamic_weights block is NOT a lane knob either: both live
        # configs set it (each under a freshly minted methodology_id).
        "dynamic_weights",
    }
    payload = compute_day(
        config=B300_CONFIG,
        day="2026-09-01",
        snapshot={"run_id": "r", "late_fill": False, "sources": []},
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    raw = snapshot_bytes(payload)
    for needle in (b"min_sources_to_claim", b"source_statistics", b"fx_lane"):
        assert needle not in raw


# --------------------------------------------------------- vast §5 statistic


def test_weighted_median_generalizes_statistics_median():
    for values in ([5.0], [5.0, 7.0], [3.0, 5.0, 9.0], [1.0, 2.0, 3.0, 10.0]):
        assert _weighted_median([(v, 1) for v in values]) == statistics.median(
            values
        )
    # Volume weighting bites: 4 GPUs at 6.0 outweigh two 1x offers.
    assert _weighted_median([(4.0, 1), (6.0, 4), (9.0, 1)]) == 6.0
    # Exact half-boundary averages the straddling values (median convention).
    assert _weighted_median([(4.0, 2), (8.0, 2)]) == 6.0
    with pytest.raises(ValueError):
        _weighted_median([])


def test_vast_statistic_screens_and_weights():
    """The vast order-book statistic: rentable on-demand asks, VERIFIED hosts only, US/CA
    only, per-GPU terms, volume = offer GPU count. Everything else in the
    book — deverified/unverified, non-US/CA, spot, implausible, other-sku —
    stays out of the statistic (but remains recorded in the raw snapshot)."""
    entry = {
        "source_id": "vast",
        "status": "ok",
        "observations": [
            _obs(usd=4.1271, verification="deverified", region="Virginia, US"),
            _obs(usd=3.9, verification="unverified", region="Oregon, US"),
            _obs(usd=4.0, verification="verified", region="Taiwan, TW"),
            _obs(usd=3.5, verification="verified", region="Texas, US", tier="spot"),
            _obs(usd=3.6, verification="verified", region="Texas, US"),
            _obs(
                usd=9000.0,
                verification="verified",
                region="Texas, US",
                implausible=True,
            ),
            _obs(sku="B300", usd=7.5, verification="verified", region="Utah, US"),
            # The count-4 marketplace slice is eligible (rule:
            # count 4 alone never quarantines) and carries weight 4.
            _obs(usd=6.0, verification="verified", region=", US", basis=4),
            _obs(usd=5.0, verification="verified", region="Quebec, CA"),
        ],
    }
    stat = vast_vwm_verified_us_ca(
        entry,
        statistic="vast_vwm_verified_us_ca",
        sku="B200",
        interruptible_tiers=("spot", "preemptible"),
    )
    # Book: (3.6, 1), (5.0, 1), (6.0, 4) -> W=6, half at 3 -> 6.0.
    assert stat == {
        "usd_per_gpu_hr": 6.0,
        "statistic": "vast_vwm_verified_us_ca",
        "currency": "USD",
        "n_eligible_prints": 3,
        "gpu_volume": 6,
    }

    # Empty eligible book -> None (source held out for the day).
    dark = vast_vwm_verified_us_ca(
        {
            "source_id": "vast",
            "status": "ok",
            "observations": [
                _obs(usd=4.0, verification="deverified", region="Virginia, US")
            ],
        },
        statistic="vast_vwm_verified_us_ca",
        sku="B200",
        interruptible_tiers=(),
    )
    assert dark is None


def test_resolver_dispatches_statistic_only_for_configured_sources():
    params = calc_params(B200_CONFIG)
    entry = {
        "source_id": "vast",
        "status": "ok",
        "observations": [
            _obs(usd=5.0, verification="verified", region="Oregon, US"),
            _obs(usd=4.0, verification="deverified", region="Oregon, US"),
        ],
    }
    by_statistic = resolve_daily_print(
        entry,
        source_id="vast",
        params=params,
        sku="B200",
        day="2026-08-16",
        fx_records={},
    )
    assert by_statistic["usd_per_gpu_hr"] == 5.0  # median rule, not min
    assert by_statistic["statistic"] == "vast_vwm_verified_us_ca"
    by_min = resolve_daily_print(
        entry,
        source_id="verda",
        params=params,
        sku="B200",
        day="2026-08-16",
        fx_records={},
    )
    assert by_min["usd_per_gpu_hr"] == 4.0  # default R1 lowest-print rule
    assert "statistic" not in by_min


# ---------------------------------------------------------- day-one golden


def test_golden_b200_day_one_from_the_real_capture():
    """The real 2026-08-16 genesis capture: publishes on day ONE (no
    warm-up gate) as an 8-constituent index — vast is manually excluded
    (the capture truncated the statistic population) with the reason
    verbatim in the artifact, and every passing source is unfiltered."""
    history: dict = {}
    payload = compute_day(
        config=B200_CONFIG,
        day="2026-08-16",
        snapshot=_fixture_snapshot(),
        substituted_from=22,
        window_history=history,
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    assert payload["basket_dark"] is False
    assert payload["methodology_id"] == "annex_a2_v0_3_calc_v5"
    assert payload["basket_id"] == "b200_annex_a2_v0_3"
    index = payload["index"]
    assert index["value_usd_gpu_hr"] == pytest.approx(GOLDEN_DAY1_VALUE, abs=1e-6)
    assert index["statistic"] == "median_ci_votes"
    assert index["vote_p25_usd_gpu_hr"] == pytest.approx(
        GOLDEN_DAY1_P25, abs=1e-6
    )
    assert index["vote_p75_usd_gpu_hr"] == pytest.approx(
        GOLDEN_DAY1_P75, abs=1e-6
    )
    assert index["confidence_usd_gpu_hr"] == pytest.approx(
        GOLDEN_DAY1_CONFIDENCE, abs=1e-6
    )
    assert index["weighted_mean_usd_gpu_hr"] == pytest.approx(
        GOLDEN_DAY1_WEIGHTED_MEAN, abs=1e-6
    )
    assert index["unweighted_mean_usd_gpu_hr"] == pytest.approx(
        GOLDEN_DAY1_UNWEIGHTED, abs=1e-6
    )
    assert index["sources_used_count"] == 8
    # Every passing source's vote is the bare floor on day one (USD lane,
    # no history) — recorded in the artifact for legibility.
    for detail in payload["sources"]:
        if detail["status"] == "ok":
            assert detail["vote"] == {
                "sigma": 0.0,
                "sigma_floored": True,
                "conf_usd_gpu_hr": 0.05,
            }
    # Each renormalized weight rounds to 6dp in the artifact, so the sum
    # carries rounding dust — same convention as the B300 series.
    assert sum(index["renormalized_weights"].values()) == pytest.approx(
        1.0, abs=1e-5
    )

    by_id = {s["source_id"]: s for s in payload["sources"]}
    assert len(by_id) == 9
    # vast excluded on day one, reason verbatim, no window entry.
    vast = by_id.pop("vast")
    assert vast["status"] == "manually_excluded"
    assert vast["excluded_reason"] == VAST_EXCLUSION_REASON
    assert "chosen" not in vast and "filter" not in vast
    assert "vast" not in index["renormalized_weights"]
    # Ruling 1: warm-up prints PASS — status ok, unfiltered, counted.
    for detail in by_id.values():
        assert detail["status"] == "ok"
        assert detail["filter"]["accepted"] is True
        assert detail["filter"]["unfiltered"] is True
    # R3 warm-up manual-verify marks (>15% from the day's basket mean).
    flagged = {
        sid
        for sid, s in by_id.items()
        if s["filter"].get("manual_verify") is True
    }
    assert flagged == {"coreweave", "together", "massedcompute"}
    # Every day-one print entered its source's window — and vast's did NOT.
    assert {k: len(v) for k, v in history.items()} == dict.fromkeys(by_id, 1)
    # Fallback pool direction (B300 proxy prints); the
    # exclusion reaches the pool too (same (date, source) rule as B300).
    pool = payload["fallback_pool"]
    pool_by_id = {p["source_id"]: p for p in pool["sources"]}
    assert set(pool_by_id) == {
        "verda",
        "nebius",
        "hyperstack",
        "massedcompute",
        "runpod",
        "vast",
    }
    assert pool_by_id["vast"]["status"] == "manually_excluded"
    assert pool["mean_usd_gpu_hr"] == pytest.approx(
        (7.5 + 7.85 + 7.4 + 5.95 + 7.89) / 5
    )
    # calc_v4 fallback mode: the pinned weight vector is the config opening
    # 2.1 weights over the day's eligible set — vast (excluded) is outside
    # the weight domain exactly like the filter window.
    assert payload["weight_calc"]["mode"] == "fallback"
    assert payload["weight_calc"]["weights"] == {
        "coreweave": 0.12,
        "hyperstack": 0.12,
        "lambda": 0.12,
        "massedcompute": 0.1,
        "nebius": 0.12,
        "runpod": 0.1,
        "together": 0.12,
        "verda": 0.12,
    }


def test_golden_vast_statistic_on_a_non_excluded_day():
    """The same recorded book one day outside the exclusion window prices
    vast by the §5 statistic — the volume-weighted median over the
    verified US/CA rows, never the (deverified) book minimum."""
    payload = compute_day(
        config=B200_CONFIG,
        day="2026-08-17",
        snapshot=_fixture_snapshot(),
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    vast = next(s for s in payload["sources"] if s["source_id"] == "vast")
    assert vast["status"] == "ok"
    chosen = vast["chosen"]
    assert chosen["usd_per_gpu_hr"] == pytest.approx(GOLDEN_VAST_STATISTIC)
    assert chosen["statistic"] == "vast_vwm_verified_us_ca"
    assert (chosen["n_eligible_prints"], chosen["gpu_volume"]) == (3, 6)
    assert payload["index"]["sources_used_count"] == 9


def test_artifact_and_pointer_shapes_match_the_b300_contract():
    """The computable consumer validates the B300 composite contract —
    treat those artifacts as the contract, byte-shape included."""
    payload = compute_day(
        config=B200_CONFIG,
        day="2026-08-16",
        snapshot=_fixture_snapshot(),
        substituted_from=22,
        window_history={},
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    assert set(payload) == {
        "schema_version",
        "kind",
        "basket_id",
        "methodology_id",
        "calc_params",
        "date",
        "basket_dark",
        "index",
        "sources",
        "fallback_pool",
        "snapshot_run_id",
        "snapshot_late_fill",
        "substituted_from_slot",
        "day_missed",
        # calc_v4 (dynamic_weights): the day's weight audit block — present
        # ONLY under a dynamic-weights config, absent from frozen series.
        "weight_calc",
    }
    assert payload["schema_version"] == 1
    assert payload["kind"] == "index_basket_composite"
    assert set(payload["index"]) == {
        "value_usd_gpu_hr",
        "statistic",
        "confidence_usd_gpu_hr",
        "vote_p25_usd_gpu_hr",
        "vote_p75_usd_gpu_hr",
        "weighted_mean_usd_gpu_hr",
        "unweighted_mean_usd_gpu_hr",
        "renormalized_weights",
        "sources_used_count",
    }
    for entry in payload["sources"]:
        if entry["status"] == "ok":
            assert {
                "source_id",
                "weight",
                "status",
                "chosen",
                "filter",
                "vote",
            } <= set(entry)
        else:  # the day-one vast exclusion (ruling A)
            assert entry["source_id"] == "vast"
            assert set(entry) == {
                "source_id",
                "weight",
                "status",
                "excluded_reason",
            }
    # calc_params travel verbatim (tuples serialized as lists).
    assert payload["calc_params"]["source_statistics"] == {
        "vast": "vast_vwm_verified_us_ca"
    }
    assert payload["calc_params"]["interruptible_tiers"] == ["spot", "preemptible"]
    assert payload["calc_params"]["manual_exclusions"] == [
        {
            "date": "2026-08-16",
            "source_id": "vast",
            "reason": VAST_EXCLUSION_REASON,
        }
    ]

    client = FakeS3()
    out = upload_composite(
        client,
        "curves",
        payload,
        prefix="index/b200_basket",
        run_id="20260817T051000Z-aaaa",
        now=datetime(2026, 8, 17, 5, 10, tzinfo=timezone.utc),
    )
    assert out["composite_key"] == COMPOSITE_KEY_DAY1
    pointer = json.loads(client.objects[POINTER_KEY])
    assert set(pointer) == {
        "pointer_version",
        "kind",
        "basket_id",
        "methodology_id",
        "composite_key",
        "sha256",
        "date",
        "basket_dark",
        "index_value_usd_gpu_hr",
        "sources_used_count",
        "run_id",
        "published_at",
    }
    assert pointer["index_value_usd_gpu_hr"] == pytest.approx(GOLDEN_DAY1_VALUE)
    assert pointer["sources_used_count"] == 8
    assert pointer["basket_dark"] is False


# ------------------------------------------------------- 5-of-9 claim floor


def _subset_snapshot(keep):
    snap = _fixture_snapshot()
    snap["sources"] = [s for s in snap["sources"] if s["source_id"] in keep]
    return snap


def test_claim_floor_four_passers_publish_basket_dark():
    """Lane rule 2: below 5 passing constituents the day publishes as
    an explicit basket_dark artifact with index null — B300's dark shape,
    never a silent hole and never a 4-source index."""
    history: dict = {}
    payload = compute_day(
        config=B200_CONFIG,
        day="2026-08-16",
        snapshot=_subset_snapshot({"verda", "nebius", "hyperstack", "lambda"}),
        substituted_from=None,
        window_history=history,
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    assert payload["basket_dark"] is True
    assert payload["index"] is None
    # The dark day still records all nine constituents (4 ok, 5 missing)...
    by_id = {s["source_id"]: s for s in payload["sources"]}
    assert len(by_id) == 9
    assert by_id["verda"]["status"] == "ok"
    assert by_id["coreweave"]["status"] == "missing"
    # ...and the four REAL prints still advance their filter windows.
    assert sorted(history) == ["hyperstack", "lambda", "nebius", "verda"]


def test_claim_floor_five_passers_publish_an_index():
    keep = {"verda", "nebius", "hyperstack", "lambda", "coreweave"}
    payload = compute_day(
        config=B200_CONFIG,
        day="2026-08-16",
        snapshot=_subset_snapshot(keep),
        substituted_from=None,
        window_history={},
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    assert payload["basket_dark"] is False
    index = payload["index"]
    assert index["sources_used_count"] == 5
    assert set(index["renormalized_weights"]) == keep
    assert sum(index["renormalized_weights"].values()) == pytest.approx(1.0)
    # Five equal 12% weights renormalize to exactly 1/5 each. calc_v3
    # hand-walk (15 votes, all at the 0.05 floor, W 1.80): the median
    # target 0.90 is crossed inside lambda's mid vote 6.69 (cum
    # 0.84->0.96), so the index is 6.69; the retired weighted mean (= the
    # plain mean here) rides as a diagnostic.
    assert index["value_usd_gpu_hr"] == pytest.approx(6.69)
    assert index["weighted_mean_usd_gpu_hr"] == pytest.approx(
        (6.11 + 7.15 + 6.00 + 6.69 + 8.60) / 5
    )


def test_claim_floor_counts_passers_not_reporters():
    """A source held out by the sigma filter is a reporter but not a
    passer — 5 reporting with only 4 accepted is still a dark day."""
    history = {"verda": [6.11] * 12}  # sigma 0: any move is held out
    snap = _subset_snapshot(
        {"verda", "nebius", "hyperstack", "lambda", "coreweave"}
    )
    for entry in snap["sources"]:
        if entry["source_id"] == "verda":
            entry["observations"] = [_obs(usd=9.99)]
    payload = compute_day(
        config=B200_CONFIG,
        day="2026-08-16",
        snapshot=snap,
        substituted_from=None,
        window_history=history,
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    verda = next(s for s in payload["sources"] if s["source_id"] == "verda")
    assert verda["filter"]["accepted"] is False
    assert payload["basket_dark"] is True
    assert payload["index"] is None
    assert history["verda"][-1] == 9.99  # held-out print still enters


# ------------------------------------------------- warm-up filter semantics


def test_warmup_sources_pass_while_seasoned_sources_filter():
    """Ruling 1: sources without enough trailing history pass unfiltered
    (and read as passing to the index math); a source WITH history is
    filtered normally on the same day."""
    history = {
        "verda": [6.11] * 9,  # one short of warm-up: still unfiltered
        "nebius": [7.15, 7.10, 7.20, 7.15, 7.12, 7.18, 7.15, 7.16, 7.14, 7.15, 7.17, 7.13],
    }
    snap = _fixture_snapshot()
    for entry in snap["sources"]:
        if entry["source_id"] == "nebius":
            # Far outside nebius's tight sigma band, but mild enough that
            # the day's basket mean stays inside every warm-up source's
            # 15% manual_verify threshold.
            entry["observations"] = [_obs(usd=7.5)]
    payload = compute_day(
        config=B200_CONFIG,
        day="2026-08-20",  # outside the day-one exclusion window
        snapshot=snap,
        substituted_from=None,
        window_history=history,
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    by_id = {s["source_id"]: s for s in payload["sources"]}
    assert by_id["verda"]["filter"] == {
        "accepted": True,
        "unfiltered": True,
        "n_history": 9,
        # calc_v2 (recorded-currency mint): every verdict names its operating currency.
        "currency": "USD",
    }
    assert by_id["nebius"]["filter"]["accepted"] is False
    assert by_id["nebius"]["filter"]["unfiltered"] is False
    # calc_v2 (recorded-currency mint): the fence nebius was judged against is explicit
    # in the verdict — band = filter_sigma * max(sigma, 0.05) around the
    # window mu (3.0 since the median-of-votes loosening).
    nebius_verdict = by_id["nebius"]["filter"]
    assert nebius_verdict["band"] == pytest.approx(
        3.0 * max(nebius_verdict["sigma"], 0.05), abs=1e-6
    )
    assert nebius_verdict["lo"] == pytest.approx(
        nebius_verdict["mu"] - nebius_verdict["band"], abs=1e-6
    )
    assert nebius_verdict["hi"] == pytest.approx(
        nebius_verdict["mu"] + nebius_verdict["band"], abs=1e-6
    )
    index = payload["index"]
    assert index["sources_used_count"] == 8  # warm-up passers all count
    assert "nebius" not in index["renormalized_weights"]
    assert "verda" in index["renormalized_weights"]
    assert history["nebius"][-1] == 7.5  # every real print enters the window


# ----------------------------------------------------- CLI replay + drift


def test_cli_publishes_day_one_eagerly_with_no_fx_lane(monkeypatch, capsys):
    """No warm-up gate end to end: --sync publishes the genesis day from
    its FIRST (slot-22-promoted) observation, touches no ECB machinery,
    and the second run is idempotent and drift-silent — proving the drift
    scan prices vast by the §5 statistic, not the min rule."""
    client = FakeS3()
    _seed_day1(client)
    now = datetime(2026, 8, 17, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    # calc_v3 (median-of-votes mint): the vote-median day-one value (the retired
    # weighted mean was 6.8251).
    assert "2026-08-16: 6.7400 $/GPU-hr (8 sources)" in out
    assert "[substituted from slot 22]" in out
    assert "composites written: 1" in out
    assert COMPOSITE_KEY_DAY1 in client.objects

    stored = json.loads(client.objects[COMPOSITE_KEY_DAY1])
    assert stored["substituted_from_slot"] == 22
    assert stored["snapshot_run_id"] == "20260816T221613Z-1fdf"
    assert stored["calc_params"]["fx_lane"] == "none"

    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "composites written: 0" in out
    assert "DRIFT" not in out


def _day2_snapshot():
    snap = _fixture_snapshot()
    snap["capture_date"] = "2026-08-17"
    snap["run_id"] = "20260817T161001Z-2222"
    snap["slot_hour_utc"] = 16
    for entry in snap["sources"]:
        if entry["source_id"] == "nebius":
            for obs in entry["observations"]:
                if obs.get("sku") == "B200":
                    obs["price_usd_gpu_hr"] = 7.2
                    obs["price_native_per_gpu_hr"] = 7.2
    return snap


def test_cli_multiday_replay_pins_to_published_days(monkeypatch, capsys):
    """Replay discipline (ruling 4): day 2 derives its filter history from
    day 1's PUBLISHED artifact; a later mutation of day 1's raw store warns
    DRIFT (via the statistic) but never changes published bytes."""
    client = FakeS3()
    _seed_day1(client)
    client.objects[
        "index/b200_basket/snapshots/2026-08-17/slot16-20260817T161001Z-2222.json"
    ] = snapshot_bytes(_day2_snapshot())
    now = datetime(2026, 8, 18, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "composites written: 2" in out

    day2 = json.loads(
        client.objects[
            "index/b200_basket/composites/annex_a2_v0_3_calc_v5/2026-08-17.json"
        ]
    )
    by_id = {s["source_id"]: s for s in day2["sources"]}
    # Day 2 history comes from day 1's PUBLISHED artifact: one print per
    # source — except vast, whose excluded day-one print never entered.
    assert by_id["vast"]["filter"]["n_history"] == 0
    assert all(
        s["filter"]["n_history"] == 1
        for sid, s in by_id.items()
        if sid != "vast"
    )
    assert by_id["vast"]["chosen"]["usd_per_gpu_hr"] == pytest.approx(
        GOLDEN_VAST_STATISTIC
    )
    assert by_id["nebius"]["chosen"]["usd_per_gpu_hr"] == 7.2
    assert day2["substituted_from_slot"] is None  # canonical slot present
    pointer = json.loads(client.objects[POINTER_KEY])
    assert pointer["date"] == "2026-08-17"

    # Raw stores mutate AFTER publication. Day 1's vast is manually
    # excluded, so its divergence is the POINT of the exclusion — silent.
    # Day 2's vast is a live statistic source: its verified US book
    # changing means the §5 statistic re-prices -> loud DRIFT, artifacts
    # untouched on both days.
    day2_key = (
        "index/b200_basket/composites/annex_a2_v0_3_calc_v5/2026-08-17.json"
    )
    published_day1 = client.objects[COMPOSITE_KEY_DAY1]
    published_day2 = client.objects[day2_key]

    def _with_mutated_vast(snap):
        for entry in snap["sources"]:
            if entry["source_id"] == "vast":
                entry["observations"] = [
                    _obs(usd=5.0, verification="verified", region="Oregon, US")
                ]
        return snap

    client.objects[SNAPSHOT_KEY] = snapshot_bytes(
        _with_mutated_vast(_fixture_snapshot())
    )
    client.objects[
        "index/b200_basket/snapshots/2026-08-17/slot16-20260817T161001Z-2222.json"
    ] = snapshot_bytes(_with_mutated_vast(_day2_snapshot()))
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DRIFT 2026-08-16" not in out  # excluded day: divergence is the point
    assert "DRIFT 2026-08-17" in out
    assert "vast: published print 5.8757" in out
    assert client.objects[COMPOSITE_KEY_DAY1] == published_day1
    assert client.objects[day2_key] == published_day2
    assert "composites written: 0" in out


def test_pointer_never_regresses_for_the_b200_series():
    client = FakeS3()

    def _payload(day, value):
        return {
            "schema_version": 1,
            "kind": "index_basket_composite",
            "basket_id": "b200_annex_a2_v0_3",
            "methodology_id": "annex_a2_v0_3_calc_v5",
            "date": day,
            "basket_dark": False,
            "index": {"value_usd_gpu_hr": value, "sources_used_count": 9},
            "sources": [],
            "fallback_pool": {"sources": [], "mean_usd_gpu_hr": None},
            "day_missed": False,
        }

    upload_composite(
        client, "curves", _payload("2026-08-17", 6.8),
        prefix="index/b200_basket", run_id="r1",
    )
    kept = upload_composite(
        client, "curves", _payload("2026-08-16", 6.7),
        prefix="index/b200_basket", run_id="r2",
    )
    assert kept["status"] == "published_pointer_kept"
    pointer = json.loads(client.objects[POINTER_KEY])
    assert pointer["date"] == "2026-08-17"
    assert composite_exists(
        client, "curves", prefix="index/b200_basket",
        methodology_id="annex_a2_v0_3_calc_v5", day="2026-08-16",
    )


# ------------------------------------------------------- manual exclusions


def test_manual_exclusion_machinery_works_from_day_one():
    """An exclusion ADDED on top of the ruled day-one set applies the same
    day — (date, source) pairs compose, never overwrite."""
    cfg = json.loads(json.dumps(B200_CONFIG))
    cfg["calc"]["manual_exclusions"] = cfg["calc"]["manual_exclusions"] + [
        {
            "date": "2026-08-16",
            "source_id": "coreweave",
            "reason": "test incident",
        }
    ]
    history: dict = {}
    payload = compute_day(
        config=cfg,
        day="2026-08-16",
        snapshot=_fixture_snapshot(),
        substituted_from=22,
        window_history=history,
        window_currencies={},
        fx_records={},
        weight_state=_ws(),
        prior_slot_prints={},
    )
    by_id = {s["source_id"]: s for s in payload["sources"]}
    coreweave = by_id["coreweave"]
    assert coreweave["status"] == "manually_excluded"
    assert coreweave["excluded_reason"] == "test incident"
    assert "chosen" not in coreweave and "filter" not in coreweave
    assert by_id["vast"]["status"] == "manually_excluded"  # ruled A entry
    assert payload["index"]["sources_used_count"] == 7
    assert "coreweave" not in payload["index"]["renormalized_weights"]
    assert "coreweave" not in history  # never poisons sigma history
    assert "vast" not in history


def test_b200_config_rejects_bad_exclusions_and_lane_knobs(tmp_path):
    base = json.loads(B200_CONFIG_PATH.read_text())
    good = {"date": "2026-08-16", "source_id": "vast", "reason": "r"}
    cases = (
        ({"manual_exclusions": [{**good, "date": "20260816"}]}, "canonical"),
        (
            {"manual_exclusions": [{**good, "source_id": "scaleway"}]},
            "not a configured source",
        ),
        ({"manual_exclusions": [good, dict(good)]}, "duplicate"),
        ({"min_sources_to_claim": 0}, "min_sources_to_claim"),
        ({"min_sources_to_claim": 10}, "min_sources_to_claim"),
        ({"min_sources_to_claim": True}, "min_sources_to_claim"),
        ({"fx_lane": "usd"}, "fx_lane"),
        (
            {"source_statistics": {"scaleway": "vast_vwm_verified_us_ca"}},
            "not a configured source",
        ),
        (
            {"source_statistics": {"vast": "nonexistent_stat"}},
            "unknown statistic",
        ),
        ({"source_statistics": ["vast"]}, "must be an object"),
        # Daily-lane review F3: the capture
        # coverage line reads the top-level knob, the
        # composite floors on the calc knob — they must agree.
        ({"min_sources_to_claim": 4}, "one claim floor"),
    )
    for override, match in cases:
        cfg = json.loads(json.dumps(base))
        cfg["calc"] = {**cfg["calc"], **override}
        p = tmp_path / "c.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(BasketConfigError, match=match):
            load_basket_config(p)


def test_mint_rule_is_mechanical_for_the_b200_series(
    monkeypatch, capsys, tmp_path
):
    """Ruling 4: an exclusion edit touching a PUBLISHED b200 day errors
    loudly and refuses to extend the series under the contradicting
    config — same mechanics as the B300 calc_v2 mint rule."""
    client = FakeS3()
    _seed_day1(client)
    now = datetime(2026, 8, 17, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    base = json.loads(B200_CONFIG_PATH.read_text())
    cfg = json.loads(json.dumps(base))
    # The published 2026-08-16 pins its exclusion set (vast, ruled A) —
    # REMOVING it is the forbidden edit; fixing history means minting.
    cfg["calc"]["manual_exclusions"] = []
    cfg_path = tmp_path / "contradicting.json"
    cfg_path.write_text(json.dumps(cfg))
    cli = _load_cli()

    class StubConfig:
        bucket = "curves"

    monkeypatch.setattr(
        cli.BucketConfig, "from_env", staticmethod(lambda: StubConfig())
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg_: client)
    monkeypatch.setattr(
        cli,
        "ensure_rates",
        lambda *a, **k: pytest.fail("USD-only basket reached the FX lane"),
    )
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(
        "sys.argv",
        ["compute_index_composite.py", "--sync", "--config", str(cfg_path)],
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "contradict the PUBLISHED" in out
    assert "mint a new methodology_id" in out


# --------------------------------------- full §5 population capture


def _vast_body(offers):
    return json.dumps({"offers": offers})


def _offer(oid, machine, host, n, per_gpu, geo, verification):
    return {
        "id": oid,
        "machine_id": machine,
        "host_id": host,
        "num_gpus": n,
        "dph_total": per_gpu * n,
        "geolocation": geo,
        "verification": verification,
    }


# 9-machine synthetic book: the whole-book cheapest 5 are a mix (deverified/
# unverified/verified), and four more machines sit beyond the cut — two of
# them verified US/CA (the §5 population rows exist for).
_BOOK_OFFERS = [
    _offer(1, 101, 11, 1, 4.0, "Virginia, US", "deverified"),
    _offer(2, 102, 12, 8, 4.5, "Taiwan, TW", "unverified"),
    _offer(3, 103, 13, 1, 5.0, "Oregon, US", "verified"),
    _offer(4, 104, 14, 2, 5.5, ", CA", "verified"),
    _offer(5, 105, 15, 4, 6.0, "Texas, US", "verified"),
    _offer(6, 106, 16, 8, 6.5, "Utah, US", "verified"),  # population
    _offer(7, 107, 17, 8, 7.0, "Nevada, US", "unverified"),
    _offer(8, 108, 18, 4, 7.5, "Berlin, DE", "verified"),
    _offer(9, 109, 19, 1, 8.0, "Quebec, CA", "verified"),  # population
]


def test_vast_records_full_verified_us_ca_population_for_b200():
    """Ruling A: beyond the legacy cheapest-5, every verified-US/CA machine
    records as a marked §5 population row — per-GPU ascending, one row per
    machine, identity fields intact; the legacy rows are byte-for-byte what
    the old recipe produced (continuity), and B300 books are untouched."""
    from gpu_index.index.sources import parse_vast_offers, select_vast_observations

    candidates = parse_vast_offers(_vast_body(_BOOK_OFFERS))
    rows = select_vast_observations(candidates, "B200")
    assert [
        (r["machine_id"], r["price_usd_gpu_hr"], r.get("book_scope")) for r in rows
    ] == [
        (101, 4.0, None),
        (102, 4.5, None),
        (103, 5.0, None),
        (104, 5.5, None),
        (105, 6.0, None),
        (106, 6.5, "verified_us_ca_population"),
        (109, 8.0, "verified_us_ca_population"),
    ]
    # The recorded MINIMUM is unchanged by the population step — the B300
    # lane's min rule and the capture screens' lowest-print delta are
    # untouched by rows that price above the cheapest 5.
    assert min(r["price_usd_gpu_hr"] for r in rows) == 4.0
    # Identity + §5 marker ride the population rows.
    for row in rows[5:]:
        assert row["verification"] == "verified"
        assert "§5 population row" in row["notes"]
        assert row["offer_id"] and row["host_id"]
    # B300: legacy cheapest-5 only, no population step.
    b300_rows = select_vast_observations(candidates, "B300")
    assert len(b300_rows) == 5
    assert not any("book_scope" in r for r in b300_rows)


def test_vast_book_stats_make_truncation_visible(monkeypatch):
    import gpu_index.index.sources as sources_mod

    candidates = sources_mod.parse_vast_offers(_vast_body(_BOOK_OFFERS))
    stats = sources_mod.vast_book_stats(candidates, "B200")
    assert stats == {
        "machines_total": 9,
        "verified_us_ca_machines": 5,  # 103/104/105 in the cheapest-5 + 106/109
        "rows_recorded": 7,
        "population_overflow": False,
    }
    rows = sources_mod.select_vast_observations(candidates, "B200")
    assert stats["rows_recorded"] == len(rows)  # the two stay in lockstep

    # The safety bound is generous, never silent: past it, the flag fires
    # and rows cap.
    monkeypatch.setattr(sources_mod, "VAST_POPULATION_LIMIT", 1)
    capped = sources_mod.select_vast_observations(candidates, "B200")
    assert [r["machine_id"] for r in capped] == [101, 102, 103, 104, 105, 106]
    capped_stats = sources_mod.vast_book_stats(candidates, "B200")
    assert capped_stats["population_overflow"] is True
    assert capped_stats["rows_recorded"] == len(capped)

    # B300 never records a population, so it can never overflow.
    b300_stats = sources_mod.vast_book_stats(candidates, "B300")
    assert b300_stats["population_overflow"] is False
    assert b300_stats["rows_recorded"] == 5


def test_statistic_consumes_the_full_recorded_population():
    """End to end for ruling A's point: the §5 statistic over the enlarged
    recording differs from the cheapest-5-only recording — the truncation
    WAS biasing the median low."""
    from gpu_index.index.sources import parse_vast_offers, select_vast_observations

    candidates = parse_vast_offers(_vast_body(_BOOK_OFFERS))
    full_rows = select_vast_observations(candidates, "B200")
    entry = {"source_id": "vast", "status": "ok", "observations": full_rows}
    stat = vast_vwm_verified_us_ca(
        entry,
        statistic="vast_vwm_verified_us_ca",
        sku="B200",
        interruptible_tiers=("spot", "preemptible"),
    )
    # Eligible book: (5.0,1) (5.5,2) (6.0,4) (6.5,8) (8.0,1) -> W=16,
    # half at 8 -> 6.5. Truncated to the cheapest-5 it was 6.0.
    assert stat["usd_per_gpu_hr"] == 6.5
    assert (stat["n_eligible_prints"], stat["gpu_volume"]) == (5, 16)
    truncated = {
        "source_id": "vast",
        "status": "ok",
        "observations": full_rows[:5],
    }
    stat_truncated = vast_vwm_verified_us_ca(
        truncated,
        statistic="vast_vwm_verified_us_ca",
        sku="B200",
        interruptible_tiers=("spot", "preemptible"),
    )
    assert stat_truncated["usd_per_gpu_hr"] == 6.0  # the ruled-A bias


def test_collect_vast_wires_book_stats_and_snapshot_carries_them(monkeypatch):
    import gpu_index.index.sources as sources_mod

    def fake_fetch(url, timeout=30):
        return _vast_body(_BOOK_OFFERS if "B200" in url else _BOOK_OFFERS[:3])

    monkeypatch.setattr(sources_mod, "fetch", fake_fetch)
    result = sources_mod.collect_vast()
    assert result["book_stats"]["B200"]["machines_total"] == 9
    assert result["book_stats"]["B200"]["population_overflow"] is False
    assert result["book_stats"]["B300"]["rows_recorded"] == 3

    cfg = {
        "basket_id": "b200_annex_a2_v0_3",
        "basket_role": "b200_basket",
        "target_sku": "B200",
        "bucket_prefix": "index/b200_basket",
        "capture_slots_utc": [4, 16],
        "canonical_slot_utc": 16,
        "sources": [
            {"source_id": "vast", "display_name": "Vast.ai",
             "role": "b200_basket", "weight": 0.5,
             "source_type": "marketplace"},
            {"source_id": "verda", "display_name": "Verda",
             "role": "b200_basket", "weight": 0.5,
             "source_type": "direct_principal"},
        ],
    }
    snapshot = build_capture_snapshot(
        config=cfg,
        source_results=[
            {**result, "status": "ok"},
            {
                "source_id": "verda",
                "status": "ok",
                "observations": [
                    {"sku": "B200", "price_usd_gpu_hr": 6.11, "raw_value": "6.11"}
                ],
            },
        ],
        captured_at=datetime(2026, 8, 18, 16, 16, tzinfo=timezone.utc),
        run_id="r",
        slot_date=datetime(2026, 8, 18, tzinfo=timezone.utc).date(),
        slot_hour_utc=16,
        canonical=True,
        capturer={"job": "test", "version": "t/0"},
    )
    by_id = {s["source_id"]: s for s in snapshot["sources"]}
    # book_stats survive snapshot assembly; sources without them stay bare.
    assert by_id["vast"]["book_stats"]["B200"]["machines_total"] == 9
    assert "book_stats" not in by_id["verda"]
    # The §5 marker survives observation normalization.
    scopes = {
        o.get("book_scope")
        for o in by_id["vast"]["observations"]
        if o.get("sku") == "B200"
    }
    assert "verified_us_ca_population" in scopes
