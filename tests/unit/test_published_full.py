# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Raw-only full reproduction of published observations."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from gpu_index.index.weights import EVENT_NO_PRICE, EVENT_SKIP
from gpu_index.published.full import (
    FullReproductionRefusal,
    VERDICT_MATCH,
    public_attendance_events,
    public_weight_print,
    read_full_history,
    reproduce_full_history,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _params():
    return {
        "aggregation": "median_ci_votes",
        "iqm_alpha": 0.16666,
        "min_sources_to_publish": 5,
        "manual_exclusions": [],
        "members": [
            {"source_id": f"s{i}", "opening_weight": 0.2}
            for i in range(5)
        ],
        "liveness": {
            "scheme": "predictive_v1",
            "lookback_horizons_hours": [1],
            "forward_horizons_hours": [1],
            "history_days": 1,
            "half_life_days": 1,
            "ridge_lambda": 1.0,
            "gamma": 4.0,
            "weight_min": 0.025,
            "weight_max": 0.3,
            "min_train_samples": 10,
            "target_variance_floor": 1e-12,
            "switch_min_eligible": 5,
            "max_abs_log_return": 0.5,
            "attendance_floor": 0.5,
            "attendance_half_life_hours": 6,
            "attendance_eta": 0.5,
            "no_price_exclusion_hours": 24,
        },
    }


def _observation():
    return {
        "kind": "gpu_price_index_observation",
        "sku": "H100",
        "methodology_id": "h100_sxm_v1_calc_v8",
        "observed_at": "2026-09-01T00:00:00.000Z",
        "status": "ok",
        "reason": None,
        "calc_params": _params(),
        "value_usd_gpu_hr": 3.0,
        "stability_band_usd_gpu_hr": 1.1,
        "receipts": [
            {
                "source_id": f"s{i}",
                "upstream_status": "ok",
                "carry_basis": None,
                "filter_verdict": "accepted",
                "price_disclosure": "published",
                "price": float(i + 1),
                "sd": 0.1,
                "currency": "USD",
                "fx_rate": None,
                # Forbidden as derivation inputs. Deliberately nonsense.
                "weight": 0.2,
                "liveness_score": None,
                "attendance_factor": 1.0,
            }
            for i in range(5)
        ],
    }


def test_full_reproduction_never_consumes_published_derived_intermediates():
    observation = _observation()
    first = reproduce_full_history([observation], target_date="2026-09-01")

    tampered = copy.deepcopy(observation)
    for index, receipt in enumerate(tampered["receipts"]):
        receipt["weight"] = 10_000 + index
        receipt["liveness_score"] = -10_000 - index
        receipt["attendance_factor"] = index / 10
    second = reproduce_full_history([tampered], target_date="2026-09-01")

    assert len(first.checks) == 1
    check = first.checks[0]
    assert check.verdict == VERDICT_MATCH
    assert check.derived_value == 3.0
    assert check.derived_band == 1.1
    assert check.derived_weights == {f"s{i}": 0.2 for i in range(5)}
    tampered_check = second.checks[0]
    assert tampered_check.derived_value == check.derived_value
    assert tampered_check.derived_band == check.derived_band
    assert tampered_check.derived_weights == check.derived_weights
    assert tampered_check.first_divergence.quantity == "attendance"
    assert tampered_check.first_divergence.source_id == "s0"


@pytest.mark.parametrize("carry_basis", ["no_price", None])
def test_full_reproduction_rebuilds_carried_votes_from_prior_raw_bytes(
    carry_basis,
):
    first = _observation()
    second = copy.deepcopy(first)
    second["observed_at"] = "2026-09-01T00:15:00.000Z"
    carried = second["receipts"][2]
    carried.update(
        {
            "upstream_status": "carried",
            "carry_basis": carry_basis,
            # Published carry outputs are derived intermediates. Corrupting
            # them must not change a raw-only reconstruction.
            "price": 999.0,
            "sd": 999.0,
            "weight": 0.2,
        }
    )

    result = reproduce_full_history(
        [first, second], target_date="2026-09-01"
    )

    assert [check.verdict for check in result.checks] == [
        VERDICT_MATCH,
        VERDICT_MATCH,
    ]
    assert result.checks[1].derived_value == 3.0
    assert result.checks[1].derived_band == 1.1


def test_public_attendance_classifier_preserves_the_engine_event_table():
    observation = _observation()
    observation["receipts"] = [
        {
            "source_id": "present_but_filtered",
            "upstream_status": "ok",
            "price": 1.0,
            "filter_verdict": "rejected",
        },
        {
            "source_id": "no_usable_price",
            "upstream_status": "ok",
            "price": None,
            "filter_verdict": "not_evaluated",
        },
        {
            "source_id": "provider_carry",
            "upstream_status": "carried",
            "carry_basis": "no_price",
        },
        {
            "source_id": "collector_carry",
            "upstream_status": "carried",
            "carry_basis": None,
        },
        {
            "source_id": "collector_error",
            "upstream_status": "error",
        },
        {
            "source_id": "ramp_in",
            "upstream_status": "missing",
        },
    ]

    assert public_attendance_events(observation) == {
        "no_usable_price": EVENT_NO_PRICE,
        "provider_carry": EVENT_NO_PRICE,
        "collector_carry": EVENT_SKIP,
        "collector_error": EVENT_SKIP,
    }


def test_public_top_level_outage_flag_skips_every_seat():
    observation = _observation()
    observation["reason"] = "observation_missed"

    assert public_attendance_events(observation) == {
        f"s{i}": EVENT_SKIP for i in range(5)
    }


def test_public_weight_print_restores_recorded_currency_terms():
    assert public_weight_print(
        {
            "source_id": "eu-cloud",
            "price": 2.3286,
            "currency": "EUR",
            "fx_rate": 1.1643,
        },
        observed_at="2026-09-01T00:00:00.000Z",
    ) == {"usd": 2.3286, "native": 2.0, "currency": "EUR"}
    assert public_weight_print(
        {
            "source_id": "rounded-eu-cloud",
            "price": 3.686523,
            "currency": "EUR",
            "fx_rate": 1.1643,
        },
        observed_at="2026-09-01T00:00:00.000Z",
    )["native"] == 3.1663


def test_full_reproduction_typed_refusal_names_missing_status_history():
    observation = _observation()
    del observation["receipts"][2]["upstream_status"]

    with pytest.raises(FullReproductionRefusal) as caught:
        reproduce_full_history([observation], target_date="2026-09-01")

    assert caught.value.code == "missing_upstream_status"
    assert "s2" in str(caught.value)


def test_history_loader_uses_the_public_series_origin_and_verified_days():
    observation = _observation()

    class Reader:
        def read_series(self, _range, *, sku):
            assert (_range, sku) == ("90d", "H100")
            return {
                "meta": {"from_observed_at": observation["observed_at"]},
                "data": {
                    "observations": [
                        {"observed_at": observation["observed_at"]}
                    ]
                },
            }

        def read_day(self, date, *, sku):
            assert sku == "H100"
            if date == "2026-08-31":
                return None
            if date == "2026-09-01":
                return {"data": {"observations": [observation]}}
            raise AssertionError(f"unexpected day read {date}")

    history = read_full_history(Reader(), sku="H100", target_date="2026-09-01")

    assert history == [observation]


def test_history_loader_typed_refusal_names_the_missing_bound():
    class Reader:
        def read_series(self, _range, *, sku):
            return {"meta": {"from_observed_at": "2026-08-31T00:00:00.000Z"}}

        def read_day(self, date, *, sku):
            return None

    with pytest.raises(FullReproductionRefusal) as caught:
        read_full_history(Reader(), sku="H100", target_date="2026-09-01")

    assert caught.value.code == "insufficient_observable_history"
    assert "public corpus origin 2026-08-31" in str(caught.value)
    assert "published day 2026-08-31 is unavailable" in str(caught.value)


def test_history_loader_refuses_when_series_and_day_lattices_disagree():
    observation = _observation()

    class Reader:
        def read_series(self, _range, *, sku):
            return {
                "meta": {"from_observed_at": observation["observed_at"]},
                "data": {
                    "observations": [
                        {"observed_at": observation["observed_at"]},
                        {"observed_at": "2026-09-01T00:15:00.000Z"},
                    ]
                },
            }

        def read_day(self, date, *, sku):
            if date == "2026-08-31":
                return None
            return {"data": {"observations": [observation]}}

    with pytest.raises(FullReproductionRefusal) as caught:
        read_full_history(Reader(), sku="H100", target_date="2026-09-01")

    assert caught.value.code == "insufficient_observable_history"
    assert "series/day observation lattice differs" in str(caught.value)


def test_full_cli_prints_derived_vector_and_value_match(monkeypatch, capsys):
    observation = _observation()

    class Reader:
        def describe(self):
            return "test public record"

        def read_series(self, _range, *, sku):
            return {
                "meta": {"from_observed_at": observation["observed_at"]},
                "data": {
                    "observations": [
                        {"observed_at": observation["observed_at"]}
                    ]
                },
            }

        def read_day(self, date, *, sku):
            if date == "2026-08-31":
                return None
            return {"data": {"observations": [observation]}}

    spec = importlib.util.spec_from_file_location(
        "verify_published_record_full_test",
        REPO_ROOT / "scripts" / "verify_published_record.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "PublishedRecordReader", Reader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_published_record.py",
            "--sku",
            "H100",
            "--date",
            "2026-09-01",
            "--full",
        ],
    )

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "raw-only full reproduction" in output
    assert "derived 3.0" in output
    assert "published 3.0" in output
    assert "MATCH" in output
    assert "weights: s0=0.2" in output
    assert "1 MATCH, 0 MISMATCH" in output

    observation["receipts"][1]["weight"] = 999.0
    assert cli.main() == 1
    output = capsys.readouterr().out
    assert (
        "FIRST DIVERGENCE: 2026-09-01T00:00:00.000Z s1 weight "
        "derived 0.2 published 999.0"
    ) in output
