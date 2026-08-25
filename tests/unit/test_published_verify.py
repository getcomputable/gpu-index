# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Recompute-and-match over published observations.

The published observation (schema ``gpu_price_index_observation``,
computable-mcp src/publisher/projector.ts:69-106) carries per-provider
receipts {price, sd, weight, status, filter_verdict} plus the disclosure
flag (artifacts.ts:20-24, applyDisclosure :89-108). The verifier rebuilds
the passing set (status ok + filter_verdict accepted, the exact set
projector.ts:236-259 marks contributing), re-derives the weighted median
of the three sd-votes per source with the panel engine's own
median_stddev_composite, and must land exactly on the published value
and stability band.

Era note: the b300/b200 lanes ran a 4-slot grid before 2026-08-24 and
hourly after; the PUBLISHED schema does not distinguish the eras
structurally (projector.ts:125-126 pins calc_params.collection_interval
to the literal "hourly"), the eras differ only in how many stamps a day
file carries. Both densities are covered: the H100 fixture day is
hourly-era, the B200 fixture day is slot-era (observations at T04/T10).
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
