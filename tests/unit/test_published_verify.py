# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Recompute-and-match over published observations.

The published observation (schema ``gpu_price_index_observation``)
carries per-provider receipts {price, sd, weight, status,
filter_verdict} plus the disclosure flag the publisher's disclosure
pass writes. The verifier rebuilds the passing set (status ok +
filter_verdict accepted, the exact set the publisher marks
contributing), re-derives the declared vote IQM (or frozen-v1 median)
of the three sd-votes per source with the panel engine's own
median_stddev_composite, and must land exactly on the published value
and stability band.

Era note: the b300/b200 lanes ran a 4-slot grid before 2026-08-24,
hourly next, and all four public lanes moved to 15 minutes from
2026-08-29. Historical hourly and slot densities are fixture-covered;
the projected era3 case carries the declared IQM alpha explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.published.artifacts import (
    PublishedRecordError,
    decode_and_verify_artifact,
    payload_digest,
)
from gpu_index.published.verify import (
    VERDICT_DEGRADED,
    VERDICT_MATCH,
    VERDICT_MISMATCH,
    UnsupportedStatisticError,
    recompute_observation,
    select_observations,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "published"


def _envelope(key: str) -> dict:
    return decode_and_verify_artifact((FIXTURES / key).read_bytes())


def _redigest(document: dict) -> dict:
    """Re-mint the envelope digest after a tamper, so digest verification
    PASSES and only the recompute can catch the change."""
    payload = {k: document[k] for k in ("data", "meta", "license")}
    document["artifact_sha256"] = payload_digest(payload)
    return decode_and_verify_artifact(json.dumps(document).encode())


def _tampered(key: str, mutate) -> dict:
    document = json.loads((FIXTURES / key).read_bytes())
    mutate(document)
    return _redigest(document)


def _projected_iqm_observation() -> dict:
    """A minimal public observation with disclosed alpha in calc_params.

    The price-scale expected values also pin the engine tuple contract as
    (source_id, weight, price). Swapping the numeric fields silently produces
    a weight-scale result and fails this fixture rather than raising.
    """
    return {
        "kind": "gpu_price_index_observation",
        "sku": "H100",
        "observed_at": "2026-08-30T12:15:00.000Z",
        "status": "ok",
        "calc_params": {
            "aggregation": "median_ci_votes",
            "iqm_alpha": 0.16666,
            "min_sources_to_publish": 2,
        },
        "value_usd_gpu_hr": 12.199898,
        "stability_band_usd_gpu_hr": 7.800102,
        "receipts": [
            {
                "source_id": "alpha",
                "price_disclosure": "published",
                "status": "ok",
                "filter_verdict": "accepted",
                "price": 10.0,
                "sd": 0.5,
                "weight": 0.6,
            },
            {
                "source_id": "bravo",
                "price_disclosure": "published",
                "status": "ok",
                "filter_verdict": "accepted",
                "price": 20.0,
                "sd": 0.5,
                "weight": 0.4,
            },
        ],
    }


# ---------------------------------------------------------------- full MATCH


def test_hourly_era_day_matches_exactly():
    envelope = _envelope("observations/2026/08/25.json")
    checks = [
        recompute_observation(obs)
        for obs in select_observations(envelope, sku="H100")
    ]
    assert len(checks) == 2
    for check in checks:
        assert check.verdict == VERDICT_MATCH
        assert check.recomputed_value == check.published_value
        assert check.recomputed_band == check.published_band
        assert check.messages == ()


def test_v1_artifact_without_iqm_alpha_remains_all_match():
    observations = select_observations(
        _envelope("observations/2026/08/25.json"), sku="H100"
    )
    assert observations
    assert all("iqm_alpha" not in obs["calc_params"] for obs in observations)
    assert all(
        recompute_observation(obs).verdict == VERDICT_MATCH
        for obs in observations
    )


def test_projected_era3_iqm_matches_and_pins_receipt_tuple_order():
    check = recompute_observation(_projected_iqm_observation())
    assert check.verdict == VERDICT_MATCH
    assert check.recomputed_value == 12.199898
    assert check.recomputed_band == 7.800102
    assert check.messages == ()


def test_pre_relabel_aggregation_uses_the_same_iqm_engine_path():
    observation = _projected_iqm_observation()
    observation["calc_params"]["aggregation"] = "median_stddev_votes"
    check = recompute_observation(observation)
    assert check.verdict == VERDICT_MATCH
    assert check.recomputed_value == 12.199898
    assert check.recomputed_band == 7.800102


def test_slot_era_day_matches_including_no_print_consistency():
    envelope = _envelope("observations/2026/08/20.json")
    checks = {
        obs["observed_at"][:13]: recompute_observation(obs)
        for obs in select_observations(envelope, sku="B200")
    }
    assert set(checks) == {"2026-08-20T04", "2026-08-20T10"}
    assert checks["2026-08-20T04"].verdict == VERDICT_MATCH
    assert checks["2026-08-20T04"].status == "ok"
    # The no-print observation is checked for consistency: passing set
    # below min_sources_to_publish and the receipt-derived reason agrees.
    assert checks["2026-08-20T10"].verdict == VERDICT_MATCH
    assert checks["2026-08-20T10"].status == "no_print"
    assert checks["2026-08-20T10"].published_value is None


def test_latest_pointer_observations_match_too():
    envelope = _envelope("latest.json")
    checks = [
        recompute_observation(obs)
        for obs in select_observations(envelope)
    ]
    assert {c.sku for c in checks} == {"H100", "B200"}
    assert all(c.verdict == VERDICT_MATCH for c in checks)


# ------------------------------------------------------------------- tampers


def test_tampered_value_mismatches_naming_the_field():
    def mutate(document):
        document["data"]["observations"][0]["value_usd_gpu_hr"] += 0.01

    envelope = _tampered("observations/2026/08/25.json", mutate)
    check = recompute_observation(
        select_observations(envelope, sku="H100")[0]
    )
    assert check.verdict == VERDICT_MISMATCH
    assert any("value_usd_gpu_hr" in m for m in check.messages)
    assert all("stability_band" not in m for m in check.messages)


def test_tampered_band_mismatches_naming_the_field():
    def mutate(document):
        document["data"]["observations"][0][
            "stability_band_usd_gpu_hr"
        ] *= 2

    envelope = _tampered("observations/2026/08/25.json", mutate)
    check = recompute_observation(
        select_observations(envelope, sku="H100")[0]
    )
    assert check.verdict == VERDICT_MISMATCH
    assert any("stability_band_usd_gpu_hr" in m for m in check.messages)


def test_tampered_receipt_price_mismatches():
    def mutate(document):
        receipts = document["data"]["observations"][0]["receipts"]
        contributing = next(
            r
            for r in receipts
            if r["status"] == "ok" and r["filter_verdict"] == "accepted"
        )
        contributing["price"] += 0.5

    envelope = _tampered("observations/2026/08/25.json", mutate)
    check = recompute_observation(
        select_observations(envelope, sku="H100")[0]
    )
    assert check.verdict == VERDICT_MISMATCH


def test_missing_iqm_disclosure_mismatches_with_targeted_hint():
    observation = _projected_iqm_observation()
    del observation["calc_params"]["iqm_alpha"]
    check = recompute_observation(observation)
    assert check.verdict == VERDICT_MISMATCH
    assert any("calc_params.iqm_alpha is absent" in m for m in check.messages)
    assert any("frozen-v1 default 0.0" in m for m in check.messages)


def test_declared_iqm_mismatch_does_not_claim_missing_disclosure():
    observation = _projected_iqm_observation()
    observation["value_usd_gpu_hr"] += 0.01
    check = recompute_observation(observation)
    assert check.verdict == VERDICT_MISMATCH
    assert any("value_usd_gpu_hr" in m for m in check.messages)
    assert all("iqm_alpha is absent" not in m for m in check.messages)


def test_no_print_that_could_print_mismatches():
    def mutate(document):
        observation = document["data"]["observations"][1]
        assert observation["status"] == "no_print"
        missing = observation["receipts"][3]
        missing.update(
            {
                "status": "ok",
                "filter_verdict": "accepted",
                "price": 3.19,
                "sd": 0.03,
                "weight": 0.2,
                "source_url": "https://delta.example.com/pricing",
                "last_seen": observation["observed_at"],
            }
        )

    envelope = _tampered("observations/2026/08/20.json", mutate)
    check = recompute_observation(
        select_observations(envelope, sku="B200", stamp="2026-08-20T10")[0]
    )
    assert check.verdict == VERDICT_MISMATCH
    assert any("no_print" in m for m in check.messages)


def test_tampered_no_print_reason_mismatches():
    def mutate(document):
        document["data"]["observations"][1]["reason"] = (
            "no_eligible_sources"
        )

    envelope = _tampered("observations/2026/08/20.json", mutate)
    check = recompute_observation(
        select_observations(envelope, sku="B200", stamp="2026-08-20T10")[0]
    )
    assert check.verdict == VERDICT_MISMATCH
    assert any("reason" in m for m in check.messages)


# ------------------------------------------------------------------ withheld


def test_withheld_contributing_source_degrades_to_digest_only():
    envelope = _envelope("observations/2026/08/23.json")
    check = recompute_observation(
        select_observations(envelope, sku="H100")[0]
    )
    assert check.verdict == VERDICT_DEGRADED
    assert check.withheld_sources == ("charlie",)
    assert check.recomputed_value is None  # nothing was recomputed
    assert any("withheld" in m for m in check.messages)
    assert any("digest" in m for m in check.messages)


def test_withheld_non_contributing_source_does_not_degrade():
    # A withheld receipt that never contributed (rejected by the filter)
    # does not impair the vote rebuild: applyDisclosure nulls price+sd
    # but the passing set never contained it.
    def mutate(document):
        observation = document["data"]["observations"][0]
        rejected = next(
            r
            for r in observation["receipts"]
            if r["filter_verdict"] == "rejected"
        )
        rejected["price"] = None
        rejected["sd"] = None
        rejected["price_disclosure"] = "withheld"
        observation["restatements"] = [
            {
                "source_id": rejected["source_id"],
                "note": "Provider price withheld by the effective "
                "disclosure policy; the published index value is "
                "unchanged.",
            }
        ]
        document["meta"]["disclosure_restatement_count"] += 1

    envelope = _tampered("observations/2026/08/25.json", mutate)
    check = recompute_observation(
        select_observations(envelope, sku="H100", stamp="2026-08-25T14")[0]
    )
    assert check.verdict == VERDICT_MATCH


# ----------------------------------------------------------------- selection


def test_select_filters_by_sku_and_stamp():
    envelope = _envelope("observations/2026/08/20.json")
    assert select_observations(envelope, sku="H100") == []
    only = select_observations(envelope, sku="B200", stamp="2026-08-20T04")
    assert len(only) == 1
    assert only[0]["observed_at"] == "2026-08-20T04:00:00.000Z"


def test_series_artifacts_refuse_recompute_selection():
    envelope = _envelope("series/24h.json")
    with pytest.raises(PublishedRecordError, match="series"):
        select_observations(envelope, sku="H100")


def test_missing_receipt_field_refuses_loudly():
    def mutate(document):
        del document["data"]["observations"][0]["receipts"][0][
            "price_disclosure"
        ]

    envelope = _tampered("observations/2026/08/25.json", mutate)
    with pytest.raises(PublishedRecordError, match="price_disclosure"):
        recompute_observation(select_observations(envelope, sku="H100")[0])


def test_unsupported_declared_statistic_refuses_with_named_error():
    observation = _projected_iqm_observation()
    observation["calc_params"]["aggregation"] = "weighted_mean"
    with pytest.raises(UnsupportedStatisticError, match="weighted_mean"):
        recompute_observation(observation)


def test_unsupported_statistic_refuses_before_withheld_degradation():
    observation = _projected_iqm_observation()
    observation["calc_params"]["aggregation"] = "future_statistic"
    observation["receipts"][0]["price_disclosure"] = "withheld"
    with pytest.raises(UnsupportedStatisticError, match="future_statistic"):
        recompute_observation(observation)


@pytest.mark.parametrize("invalid_alpha", ["bad", True, -0.1, 0.51, float("nan")])
def test_invalid_iqm_alpha_refuses_before_withheld_degradation(invalid_alpha):
    observation = _projected_iqm_observation()
    observation["calc_params"]["iqm_alpha"] = invalid_alpha
    observation["receipts"][0]["price_disclosure"] = "withheld"
    with pytest.raises(PublishedRecordError, match="calc_params.iqm_alpha"):
        recompute_observation(observation)


def test_invalid_iqm_alpha_refuses_on_insufficient_no_print():
    observation = _projected_iqm_observation()
    observation["status"] = "no_print"
    observation["reason"] = "insufficient_coverage"
    observation["value_usd_gpu_hr"] = None
    observation["stability_band_usd_gpu_hr"] = None
    observation["calc_params"]["min_sources_to_publish"] = 3
    observation["calc_params"]["iqm_alpha"] = 0.75
    with pytest.raises(PublishedRecordError, match="calc_params.iqm_alpha"):
        recompute_observation(observation)
