# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory lium collector -- fixture pins (live response 2026-08-22).

House style per the runpod exemplar: (1) parse the recorded fixture, (2)
pin exact prints for known rows incl. this source's edge cases (the single
spot row, miner-set float residue, lookalike labels, identical duplicate
asks), (3) prove the framework normalization maps this source's real
labels. The fixture is a 17-row excerpt of the real 86-row /api/executors
response captured 2026-08-22 -- every distinct machine_name kept, with the
bulky ``specs`` blob pruned to the one field the collector reads
(is_spot); all other bytes are verbatim. Structural-drift and
pending-price cases do not exist on the live surface today, so those pins
are exercised with synthetic bodies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.lium import SOURCE_ID, parse_lium

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "lium" / "executors.json"
)


def _fixture_rows():
    return json.loads(FIXTURE.read_text())


def _body(rows) -> str:
    return json.dumps(rows)


@pytest.fixture(scope="module")
def parsed():
    return parse_lium(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "lium"


def test_every_fixture_row_recorded(parsed):
    rows, notes, stats = parsed
    assert len(rows) == 17
    assert notes == []
    assert stats == {
        "machines_total": 17,
        "rows_recorded": 17,
        "rows_skipped": 0,
        "distinct_labels": 14,
        "by_lium_tier": {"secure": 16, "spot": 1},
    }


def test_pins_exact_prices_for_known_rows(rows):
    b300 = next(r for r in rows if r["machine_id"].startswith("42ef8a0b"))
    assert b300["sku_identifier"] == "NVIDIA B300 SXM6 AC"
    assert b300["price_usd_gpu_hr"] == 7.97
    assert b300["raw_value"] == "7.97"
    assert b300["tier"] == "on-demand"
    assert b300["region"] == "US"
    assert b300["extra"]["gpu_count"] == 8

    # Two IDENTICAL 1.75 asks from two different machines -- both must
    # print (per-machine book, deduped by id, never by price).
    h100 = [r for r in rows if r["sku_identifier"] == "NVIDIA H100 80GB HBM3"]
    assert [r["price_usd_gpu_hr"] for r in h100] == [1.75, 1.75]
    assert len({r["machine_id"] for r in h100}) == 2

    pcie = next(r for r in rows if r["sku_identifier"] == "NVIDIA H100 PCIe")
    assert pcie["price_usd_gpu_hr"] == 2.24


def test_float_residue_recorded_verbatim(rows):
    """Miner-set prices carry float residue -- raw_value keeps the figure
    exactly as published while the normalized price rounds to 4dp."""
    residue = next(r for r in rows if r["machine_id"].startswith("8b121617"))
    assert residue["raw_value"] == "8.000300000000001"
    assert residue["price_usd_gpu_hr"] == 8.0003
    b200 = next(r for r in rows if r["sku_identifier"] == "NVIDIA B200")
    assert b200["raw_value"] == "6.5318000000000005"
    assert b200["price_usd_gpu_hr"] == 6.5318


def test_spot_row_stays_labeled(rows):
    spot = [r for r in rows if r["tier"] == "spot"]
    assert len(spot) == 1
    assert spot[0]["sku_identifier"] == "NVIDIA H200"
    assert spot[0]["price_usd_gpu_hr"] == 3.1
    assert spot[0]["extra"]["lium_tier"] == "spot"
    # ... and the secure H200 book stays separate from the spot print.
    secure_h200 = [
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA H200" and r["tier"] == "on-demand"
    ]
    assert [r["price_usd_gpu_hr"] for r in secure_h200] == [4.2262]


def test_per_gpu_basis_and_currency(rows):
    """price_per_gpu is already per-GPU: basis stays 1 so price*basis
    reproduces the raw figure; machine size is metadata, not basis."""
    for r in rows:
        assert r["gpu_count_basis"] == 1
        assert r["currency"] == "USD"
        assert r["raw_unit"] == "usd_per_gpu_hr"
        assert r["price_usd_gpu_hr"] == round(float(r["raw_value"]), 4)
        assert r["extra"]["gpu_count"] >= 1
        assert r["machine_id"]


def test_lookalike_labels_stay_distinct(rows):
    """The book's lookalike families must never collapse at capture time:
    the identifier is the record."""
    idents = {r["sku_identifier"] for r in rows}
    assert "NVIDIA RTX PRO 6000 Blackwell Server Edition" in idents
    assert "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" in idents
    assert "NVIDIA RTX 6000 Ada Generation" in idents
    assert "NVIDIA H200" in idents and "NVIDIA H200 NVL" in idents
    assert "NVIDIA H100 80GB HBM3" in idents and "NVIDIA H100 PCIe" in idents
    assert "NVIDIA L40S" in idents and "NVIDIA L40" in idents
    server = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    )
    workstation = next(
        r
        for r in rows
        if r["sku_identifier"]
        == "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    )
    assert server["price_usd_gpu_hr"] == 1.01
    assert workstation["price_usd_gpu_hr"] == 1.3


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA B300 SXM6 AC"] == "B300"
    assert mapped["NVIDIA B200"] == "B200"
    # True lookalikes preserved at capture now land on DIFFERENT skus
    # (H-series variant split, design section 7).
    assert mapped["NVIDIA H200"] == "H200"
    assert mapped["NVIDIA H200 NVL"] == "H200_NVL"
    assert mapped["NVIDIA H100 80GB HBM3"] == "H100"
    assert mapped["NVIDIA H100 PCIe"] == "H100_PCIE"
    assert mapped["NVIDIA RTX PRO 6000 Blackwell Server Edition"] == "RTX_PRO_6000"
    assert (
        mapped["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]
        == "RTX_PRO_6000"
    )
    assert mapped["NVIDIA RTX 6000 Ada Generation"] == "RTX_6000_ADA"
    assert mapped["NVIDIA GeForce RTX 5090"] == "RTX_5090"
    assert mapped["NVIDIA GeForce RTX 4090"] == "RTX_4090"
    assert mapped["NVIDIA GeForce RTX 3090"] == "RTX_3090"
    assert mapped["NVIDIA L40S"] == "L40S"
    assert mapped["NVIDIA L40"] == "L40"
    # Every KNOWN lium label maps today; a genuinely-new chip appearing
    # live records unmapped and the capture warns -- this pin is about the
    # known labels staying mapped.
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known lium labels now unmapped: {unmapped}"


def test_pending_price_never_recorded_as_current():
    """pending_price_per_hour + price_change_effective_date are a
    SCHEDULED FUTURE change -- the recorded price must stay price_per_gpu,
    with the pending pair visible in extra."""
    row = _fixture_rows()[0]
    assert row["price_per_gpu"] == 7.97
    row["pending_price_per_hour"] = 9.5
    row["price_change_effective_date"] = "2020-01-01T00:00:00Z"
    obs = parse_lium(_body([row]))[0][0]
    assert obs["price_usd_gpu_hr"] == 7.97
    assert obs["raw_value"] == "7.97"
    assert obs["extra"]["pending_price_per_hour"] == 9.5
    assert (
        obs["extra"]["price_change_effective_date"] == "2020-01-01T00:00:00Z"
    )


def test_no_pending_fields_when_unset(rows):
    for r in rows:
        assert "pending_price_per_hour" not in r["extra"]


def test_unknown_tier_skipped_not_guessed():
    good, weird = (dict(r) for r in _fixture_rows()[:2])
    weird["tier"] = "reserved-enterprise"
    weird["specs"] = {"is_spot": None}
    obs, notes, stats = parse_lium(_body([good, weird]))
    assert len(obs) == 1
    assert obs[0]["machine_id"] == good["id"]
    assert stats["rows_skipped"] == 1
    assert any("1 unmapped_tier" in n for n in notes)


def test_tier_spot_contradiction_skipped():
    row = dict(_fixture_rows()[0])
    assert row["tier"] == "secure"
    row["specs"] = {"is_spot": True}  # contradicts tier
    keep = _fixture_rows()[1]
    obs, notes, stats = parse_lium(_body([row, keep]))
    assert [o["machine_id"] for o in obs] == [keep["id"]]
    assert any("1 tier_spot_mismatch" in n for n in notes)


def test_non_finite_price_skipped_not_recorded():
    """json.loads accepts NaN/Infinity literals; both pass the structural
    number pin AND a <= 0 comparison -- without the isfinite value pin a
    miner-set NaN would record as a live USD print (runpod/scaleway carry
    the same guard)."""
    for weird in (float("nan"), float("inf"), float("-inf")):
        row = dict(_fixture_rows()[0])
        row["price_per_gpu"] = weird
        keep = _fixture_rows()[1]
        # json.dumps emits the bare NaN/Infinity literals lium's own
        # Python backend could produce.
        obs, notes, stats = parse_lium(_body([row, keep]))
        assert [o["machine_id"] for o in obs] == [keep["id"]]
        assert stats["rows_skipped"] == 1
        assert any("1 non_finite_price" in n for n in notes)


def test_zero_price_skipped_not_recorded_as_free():
    row = dict(_fixture_rows()[0])
    row["price_per_gpu"] = 0.0
    obs, notes, _ = parse_lium(_body([row, _fixture_rows()[1]]))
    assert len(obs) == 1
    assert any("1 unpriced" in n for n in notes)


def test_duplicate_machine_id_deduped():
    row = _fixture_rows()[0]
    obs, notes, stats = parse_lium(_body([row, dict(row)]))
    assert len(obs) == 1
    assert stats["machines_total"] == 2
    assert any("1 duplicate_id" in n for n in notes)


def test_structural_drift_raises():
    rows = _fixture_rows()
    with pytest.raises(RuntimeError, match="did not return a JSON list"):
        parse_lium(json.dumps({"executors": rows}))
    with pytest.raises(RuntimeError, match="EMPTY"):
        parse_lium("[]")
    missing = dict(rows[0])
    del missing["machine_name"]
    with pytest.raises(RuntimeError, match="'machine_name'"):
        parse_lium(_body([missing]))
    mistyped = dict(rows[0])
    mistyped["price_per_gpu"] = "7.97"  # stringly-typed price = reshaped row
    with pytest.raises(RuntimeError, match="'price_per_gpu'"):
        parse_lium(_body([mistyped]))
    unlabeled = dict(rows[0])
    unlabeled["machine_name"] = "  "
    with pytest.raises(RuntimeError, match="'machine_name'"):
        parse_lium(_body([unlabeled]))


def test_all_rows_skipped_raises():
    """A book where every row failed the value pins is a shape change,
    not a thin market -- must raise, never claim a healthy zero-row
    source."""
    rows = [dict(r) for r in _fixture_rows()[:3]]
    for r in rows:
        r["tier"] = "mystery"
        r["specs"] = {"is_spot": None}
    with pytest.raises(RuntimeError, match="ZERO survived"):
        parse_lium(_body(rows))
