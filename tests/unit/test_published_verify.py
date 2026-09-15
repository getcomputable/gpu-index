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
the projected era3 case carries the declared IQM alpha explicitly. The
2026-09-14 smoothing-armed generations (EWMA vote pre-smoothing +
fence_reject_carry; disclosed cast prices) are covered in the final
section of this file.
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


def test_latest_pointer_admits_the_activation_bundle():
    """The publisher's replay-from-anchor bundle rides ``latest.json`` as
    ``data.activation``; the reader contract admits it present, absent, or
    null. The verifier recomputes values from receipts and does not read
    the bundle, so it is admitted as an opaque object: the pointer that
    went live on 2026-09-03 must still verify and reproduce."""
    def with_bundle(document):
        document["data"]["activation"] = {
            "schema_version": "1",
            "anchor": {"sequence": 3, "record_sha256": "a" * 64},
            "current": {"sequence": 6, "record_sha256": "b" * 64},
            "chain": [],
            "checkpoints": [],
            "projections": [],
            "changelog_tail": [],
        }

    def with_null(document):
        document["data"]["activation"] = None

    for mutate in (with_bundle, with_null):
        envelope = _tampered("latest.json", mutate)
        checks = [
            recompute_observation(obs)
            for obs in select_observations(envelope)
        ]
        assert all(c.verdict == VERDICT_MATCH for c in checks)


def test_latest_pointer_refuses_a_malformed_bundle_and_unknown_keys():
    """Admitting one named key does not open the contract: a non-object
    bundle and any other unexpected key still refuse."""
    def bad_bundle(document):
        document["data"]["activation"] = "not-a-bundle"

    def unknown_key(document):
        document["data"]["extra"] = {}

    with pytest.raises(PublishedRecordError, match="activation"):
        _tampered("latest.json", bad_bundle)
    with pytest.raises(PublishedRecordError, match="unexpected"):
        _tampered("latest.json", unknown_key)


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


# ---------------------------------------- smoothing-armed generations
# EWMA vote pre-smoothing + fence_reject_carry (the
# 2026-09-14 calc_v17/calc_v16 mints). On an armed lane
# (calc_params.pre_smoothing_half_life_hours present) every voting
# receipt disclosed smoothed_vote_usd -- the EXACT cast price the engine
# aggregated -- and the recompute votes that number, never the raw print.
# The expected values below are the upstream extraction pass's pinned
# vectors (oracle: the engine's own median_stddev_composite).


def _armed_receipt(
    sid,
    *,
    price,
    cast,
    sd=0.15,
    weight=0.15,
    verdict="accepted",
    upstream="ok",
    carried_vote=None,
    disclosure="published",
):
    receipt = {
        "source_id": sid,
        "price_disclosure": disclosure,
        "status": "ok",
        "upstream_status": upstream,
        "filter_verdict": verdict,
        "price": price,
        "sd": sd,
        "weight": weight,
    }
    if cast is not None:
        receipt["smoothed_vote_usd"] = cast
    if carried_vote is not None:
        # The corpus flattens the engine's carried_vote block to receipt-level
        # keys (the publisher's projection; the cross-repo pin suite holds
        # the shape).
        receipt["carried_vote_from"] = carried_vote["from"]
        if "carry_basis" in carried_vote:
            receipt["carry_basis"] = carried_vote["carry_basis"]
    return receipt


def _fence_reject_receipt(sid, *, cast, price=2.09):
    """The fence-reject carry shape: the REAL rejected print stays on the row as
    price, the vote was substituted from the carry book and disclosed."""
    return _armed_receipt(
        sid,
        price=price,
        cast=cast,
        verdict="rejected",
        carried_vote={
            "from": "2026-09-14T00:45:00.000Z",
            "carry_basis": "no_price",
        },
    )


def _armed_observation(receipts, *, min_publish=5, value=None, band=None):
    return {
        "kind": "gpu_price_index_observation",
        "sku": "H100",
        "observed_at": "2026-09-14T01:00:00.000Z",
        "status": "ok",
        "reason": None,
        "calc_params": {
            "aggregation": "median_ci_votes",
            "iqm_alpha": 0,
            "pre_smoothing_half_life_hours": 1,
            "min_sources_to_publish": min_publish,
        },
        "value_usd_gpu_hr": value,
        "stability_band_usd_gpu_hr": band,
        "receipts": receipts,
    }


def test_armed_lane_votes_disclosed_cast_prices_never_raw():
    """V2: six fresh seats print 7.5 but each cast its EWMA-smoothed
    7.61 -- the recompute must price the disclosed casts exactly, and
    voting the raw prints instead reproduces 7.5 != published 7.61 (the
    contrapositive proves the vote centers moved)."""
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.61) for sid in "abcdef"
    ]
    check = recompute_observation(
        _armed_observation(receipts, value=7.61, band=0.15)
    )
    assert check.verdict == VERDICT_MATCH
    assert check.recomputed_value == 7.61
    assert check.recomputed_band == 0.15

    raw_published = recompute_observation(
        _armed_observation(receipts, value=7.5, band=0.15)
    )
    assert raw_published.verdict == VERDICT_MISMATCH
    assert raw_published.recomputed_value == 7.61


def test_fence_reject_carried_vote_admits_at_its_disclosed_cast_price():
    """V3: seat f's fresh 2.09 print was fence-rejected but the engine
    cast its booked smoothed 7.62 -- the row votes 7.62 (never 2.09,
    never nothing), is classified carried, and the five observed seats
    alone satisfy min_sources_to_publish 5."""
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcde"
    ] + [_fence_reject_receipt("f", cast=7.62)]
    check = recompute_observation(
        _armed_observation(receipts, value=7.5, band=0.15)
    )
    assert check.verdict == VERDICT_MATCH
    assert check.recomputed_value == 7.5
    assert check.recomputed_band == 0.15


def test_armed_lane_missing_cast_price_refuses_loudly_naming_the_seat():
    """V4 (the 2026-09-14T01:00Z observation shape): a participating
    fence-reject row without smoothed_vote_usd = the engine cast a price
    this artifact does not disclose. REFUSE -- never price the rejected
    2.09 (that derives 7.575/0.275 != published 7.635/0.215, so the
    fallback would be a non-vacuous wrong answer) and never drop the
    seat (a different ballot)."""
    receipts = [
        _armed_receipt(sid, price=price, cast=price)
        for sid, price in zip("abcde", (7.4, 7.5, 7.6, 7.85, 8.0))
    ] + [_fence_reject_receipt("sesterce", cast=None)]
    observation = _armed_observation(receipts, value=7.635, band=0.215)
    with pytest.raises(PublishedRecordError) as caught:
        recompute_observation(observation)
    message = str(caught.value)
    assert "sesterce" in message
    assert "2026-09-14T01:00:00.000Z" in message
    assert "smoothed_vote_usd" in message


@pytest.mark.parametrize(
    "unusable", ["7.62", float("nan"), 0, -1, float("inf")]
)
def test_armed_lane_unusable_cast_price_refuses_loudly(unusable):
    """V4's present-but-unusable arm: a malformed disclosure is a torn
    artifact, never a silent chosen fall-back."""
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcde"
    ] + [_fence_reject_receipt("sesterce", cast=unusable)]
    with pytest.raises(PublishedRecordError, match="sesterce"):
        recompute_observation(
            _armed_observation(receipts, value=7.635, band=0.215)
        )


def test_armed_lane_missing_cast_price_on_a_fresh_seat_refuses_too():
    """The disclosure law covers EVERY voting flavor: an ok+accepted row
    on an armed lane without smoothed_vote_usd refuses identically."""
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcde"
    ] + [_armed_receipt("f", price=7.5, cast=None)]
    with pytest.raises(PublishedRecordError, match="f"):
        recompute_observation(
            _armed_observation(receipts, value=7.5, band=0.15)
        )


def test_status_carried_seat_recasts_its_frozen_smoothed_state():
    """V5: a carried seat's row keeps the booked raw print (7.9) as
    price but disclosed its frozen smoothed state (7.58) -- the vote is
    7.58, classified carried, and the five observed seats satisfy the
    floor."""
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcde"
    ] + [_armed_receipt("f", price=7.9, cast=7.58, upstream="carried")]
    check = recompute_observation(
        _armed_observation(receipts, value=7.5, band=0.15)
    )
    assert check.verdict == VERDICT_MATCH
    assert check.recomputed_value == 7.5
    assert check.recomputed_band == 0.15


def test_carried_votes_never_satisfy_the_observed_floor():
    """V6: six seats vote but two are fence-reject carries -- only 4
    observed. min_sources_to_publish 5 rebuilds NO composite (a
    published ok value mismatches; a published no_print is consistent);
    min 4 rebuilds value 7.5 over all six votes."""
    def receipts():
        return [
            _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcd"
        ] + [
            _fence_reject_receipt(sid, cast=7.5) for sid in ("e", "f")
        ]

    over_floor = recompute_observation(
        _armed_observation(receipts(), min_publish=5, value=7.5, band=0.15)
    )
    assert over_floor.verdict == VERDICT_MISMATCH
    assert any("4 observed" in m for m in over_floor.messages)

    dark = _armed_observation(receipts(), min_publish=5)
    dark["status"] = "no_print"
    dark["reason"] = "insufficient_coverage"
    assert recompute_observation(dark).verdict == VERDICT_MATCH

    lower_floor = recompute_observation(
        _armed_observation(receipts(), min_publish=4, value=7.5, band=0.15)
    )
    assert lower_floor.verdict == VERDICT_MATCH
    assert lower_floor.recomputed_value == 7.5


def test_pre_smoothing_observations_keep_the_frozen_law_byte_identically():
    """The pin the task demands: WITHOUT the calc knob the disclosures
    are ignored like any unfamiliar receipt field -- fresh seats vote
    their raw prints, a fence-reject row never votes, and the floor
    counts every passer."""
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.61) for sid in "abcde"
    ] + [_fence_reject_receipt("f", cast=7.62)]
    observation = _armed_observation(receipts, value=7.5, band=0.15)
    del observation["calc_params"]["pre_smoothing_half_life_hours"]
    check = recompute_observation(observation)
    assert check.verdict == VERDICT_MATCH  # raw 7.5 votes, f dropped
    assert check.recomputed_value == 7.5

    smoothed_published = _armed_observation(receipts, value=7.61, band=0.15)
    del smoothed_published["calc_params"]["pre_smoothing_half_life_hours"]
    assert recompute_observation(smoothed_published).verdict == (
        VERDICT_MISMATCH
    )


def test_armed_lane_requires_upstream_status_on_voting_rows():
    """Carried classification (the floor law) reads upstream_status --
    a voting row without it on an armed lane is a torn artifact."""
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcdef"
    ]
    del receipts[2]["upstream_status"]
    with pytest.raises(PublishedRecordError, match="upstream_status"):
        recompute_observation(
            _armed_observation(receipts, value=7.5, band=0.15)
        )


@pytest.mark.parametrize("invalid", ["1h", -1, 0, None, True, float("nan")])
def test_invalid_pre_smoothing_half_life_refuses(invalid):
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcdef"
    ]
    observation = _armed_observation(receipts, value=7.5, band=0.15)
    observation["calc_params"]["pre_smoothing_half_life_hours"] = invalid
    with pytest.raises(
        PublishedRecordError, match="pre_smoothing_half_life_hours"
    ):
        recompute_observation(observation)


def test_withheld_carried_voter_degrades_before_any_cast_price_demand():
    """The disclosure pass nulls a withheld row's prices; a withheld
    VOTING row (here a fence-reject carry) degrades the observation to
    digest-only exactly like any withheld contributor -- never a
    spurious missing-cast-price refusal."""
    withheld = _fence_reject_receipt("f", cast=None)
    withheld.update(price_disclosure="withheld", price=None, sd=None)
    receipts = [
        _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcde"
    ] + [withheld]
    check = recompute_observation(
        _armed_observation(receipts, value=7.5, band=0.15)
    )
    assert check.verdict == VERDICT_DEGRADED
    assert check.withheld_sources == ("f",)


def test_non_string_carried_vote_from_dressing_reads_as_absent():
    """The shared presence fence: number/object/array/empty dressing on the
    flat carried_vote_from key is NOT a disclosure -- the row falls back
    onto the plain law and, being fence-rejected, does not vote (fail
    closed). A nested carried_vote object is one of the dressings, pinned
    refused."""
    for dressing in ({"from": "2026-09-14T00:45:00.000Z"}, 1, ["no_price"], True, ""):
        row = _armed_receipt(
            "f", price=2.09, cast=7.62, verdict="rejected"
        )
        row["carried_vote_from"] = dressing
        receipts = [
            _armed_receipt(sid, price=7.5, cast=7.5) for sid in "abcde"
        ] + [row]
        check = recompute_observation(
            _armed_observation(receipts, value=7.5, band=0.15)
        )
        assert check.verdict == VERDICT_MATCH  # five observed seats only
