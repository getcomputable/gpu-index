# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory lium collector -- fixture pins (live response 2026-08-22;
availability fixtures 2026-08-25).

House style per the runpod exemplar: (1) parse the recorded fixture, (2)
pin exact prints for known rows incl. this source's edge cases (the single
spot row, miner-set float residue, lookalike labels, identical duplicate
asks), (3) prove the framework normalization maps this source's real
labels. The executors fixture is a 17-row excerpt of the real 86-row
/api/executors response captured 2026-08-22 -- every distinct machine_name
kept, with the bulky ``specs`` blob pruned to the one field the collector
reads (is_spot); all other bytes are verbatim. Structural-drift and
pending-price cases do not exist on the live surface today, so those pins
are exercised with synthetic bodies.

Availability fixtures (availability-accrual rungs 1-3), captured live 2026-08-25:
machines.json is a 17-row excerpt of the real 91-row /api/machines
response -- all 14 machine_names in the executors fixture kept (verbatim
join coverage), plus the zero-fleet edge rows A30 and A40 (the absence
signal: rental_rate 0.0 AND total_gpu_count 0) and the fully-rented
A100-SXM4-80GB (rental_rate 1.0); rows untrimmed, values verbatim.
capacity.json is the FULL 13-row /api/machines/capacity payload verbatim
(13 base models x buckets {1, 8}). Fail-open paths (availability fetch
raising, availability surface reshaped) are exercised with monkeypatched
fetches and synthetic bodies -- price rows must record regardless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.lium import (
    CAPACITY_URL,
    MACHINES_URL,
    SOURCE_ID,
    URL,
    attach_model_stats,
    collect,
    parse_capacity,
    parse_lium,
    parse_machines,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "lium" / "executors.json"
)
MACHINES_FIXTURE = FIXTURE.with_name("machines.json")
CAPACITY_FIXTURE = FIXTURE.with_name("capacity.json")


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
        "availability_skips": {},
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


# --- availability accrual (rungs 1-3, fixtures 2026-08-25) ----


def _collect(monkeypatch, *, executors=None, machines=None, capacity=None):
    """Drive collect() with per-URL bodies. A value that is an Exception
    raises at fetch time (the transport failing); a string body parses;
    None means the recorded live fixture."""
    bodies = {
        URL: executors if executors is not None else FIXTURE.read_text(),
        MACHINES_URL: (
            machines if machines is not None else MACHINES_FIXTURE.read_text()
        ),
        CAPACITY_URL: (
            capacity if capacity is not None else CAPACITY_FIXTURE.read_text()
        ),
    }

    def fake_fetch(url, timeout=None):
        body = bodies[url]
        if isinstance(body, Exception):
            raise body
        return body

    monkeypatch.setattr("gpu_index.observatory.sources.lium.fetch", fake_fetch)
    return collect()


def test_collect_attaches_chip_occupancy(monkeypatch):
    """Rung 2 happy path: every observation gains the chip-level
    occupancy pair via the verbatim machine_name == name join, and the
    catalog aggregates (incl. the zero-fleet absence signal) land in
    book_stats."""
    out = _collect(monkeypatch)
    rows = out["observations"]
    assert len(rows) == 17
    assert "partial_errors" not in out
    b300 = next(r for r in rows if r["machine_id"].startswith("42ef8a0b"))
    assert b300["extra"]["model_rental_rate"] == 1.0  # fully-rented fleet
    assert b300["extra"]["model_total_gpu_count"] == 57
    b200 = next(r for r in rows if r["sku_identifier"] == "NVIDIA B200")
    assert b200["extra"]["model_rental_rate"] == 0.75
    assert b200["extra"]["model_total_gpu_count"] == 52
    for r in rows:
        assert r["extra"]["model_rental_rate"] is not None
        assert r["extra"]["model_total_gpu_count"] is not None
        # rung 1: every fixture count is a valid int within gpu_count.
        avail = r["extra"]["available_gpu_count"]
        assert isinstance(avail, int) and not isinstance(avail, bool)
        assert 0 <= avail <= r["extra"]["gpu_count"]
    stats = out["book_stats"]
    assert stats["availability_skips"] == {}
    assert stats["models_in_catalog"] == 17  # trimmed excerpt of 91 live
    assert stats["models_zero_capacity"] == 2  # A30 + A40: priced, no fleet


def test_capacity_buckets_recorded_verbatim_never_joined(monkeypatch):
    """Rung 3: the capacity matrix rides whole in book_stats -- and its
    collapsed base_model vocabulary must NEVER be joined onto
    machine_name (it cannot tell H100 PCIe from 80GB HBM3)."""
    out = _collect(monkeypatch)
    buckets = out["book_stats"]["capacity_buckets"]
    assert buckets == json.loads(CAPACITY_FIXTURE.read_text())
    assert len(buckets) == 13
    b300 = next(r for r in buckets if r["base_model"] == "B300")
    assert b300["buckets"]["1"] == {
        "max_cap": 10,
        "unrented_count": 0,
        "hourly_rate": 5.1,
    }
    for r in out["observations"]:
        assert "base_model" not in r["extra"]
        assert "capacity_buckets" not in r["extra"]


def test_available_gpu_count_pin_fails_open_per_row():
    """Rung 1 hostile variants: over-count, bool, stringly int, negative
    -- each records None + a counted availability skip while the PRICE
    ROW STILL PRINTS (fail-open per row, never a dark source)."""
    for hostile in (99, True, "8", -1):
        row = dict(_fixture_rows()[0])
        row["available_gpu_count"] = hostile
        obs, notes, stats = parse_lium(_body([row, _fixture_rows()[1]]))
        assert len(obs) == 2  # nothing skipped
        assert obs[0]["extra"]["available_gpu_count"] is None
        assert obs[0]["price_usd_gpu_hr"] == 7.97  # print unaffected
        assert stats["rows_skipped"] == 0
        assert stats["availability_skips"] == {"bad_available_gpu_count": 1}
        assert notes == []  # book_stats-only, not a partial_error


def test_available_gpu_count_absent_records_none_uncounted():
    """Absent is not a violation: the pin only bites when present."""
    row = dict(_fixture_rows()[0])
    del row["available_gpu_count"]
    obs, _, stats = parse_lium(_body([row]))
    assert obs[0]["extra"]["available_gpu_count"] is None
    assert stats["availability_skips"] == {}


def test_parse_machines_happy_path():
    by_name, value_skips, catalog_stats = parse_machines(
        MACHINES_FIXTURE.read_text()
    )
    assert value_skips == {}
    assert catalog_stats == {
        "models_in_catalog": 17,
        "models_zero_capacity": 2,
    }
    assert by_name["NVIDIA B200"] == {
        "model_rental_rate": 0.75,
        "model_total_gpu_count": 52,
    }
    # The absence signal: a priced model with zero fleet -- rental_rate
    # 0.0 here means 'no fleet', which is why both fields are recorded.
    assert by_name["NVIDIA A30"] == {
        "model_rental_rate": 0.0,
        "model_total_gpu_count": 0,
    }
    assert by_name["NVIDIA A100-SXM4-80GB"] == {
        "model_rental_rate": 1.0,
        "model_total_gpu_count": 39,
    }
    # rental_rate float residue rides verbatim, never rounded.
    rtx3090 = by_name["NVIDIA GeForce RTX 3090"]
    assert rtx3090["model_rental_rate"] == 0.48700000000000004


def test_parse_machines_reshape_fails_closed():
    """Structural drift on the availability surface raises INSIDE the
    availability parse (collect() turns it into a partial_error)."""
    rows = json.loads(MACHINES_FIXTURE.read_text())
    with pytest.raises(RuntimeError, match="non-empty JSON list"):
        parse_machines(json.dumps({"machines": rows}))
    with pytest.raises(RuntimeError, match="non-empty JSON list"):
        parse_machines("[]")
    missing = dict(rows[0])
    del missing["name"]
    stringly = dict(rows[0], rental_rate="0.75")
    booly = dict(rows[0], total_gpu_count=True)
    floaty = dict(rows[0], total_gpu_count=39.0)
    unlabeled = dict(rows[0], name="  ")
    for hostile in (missing, stringly, booly, floaty, unlabeled):
        with pytest.raises(
            RuntimeError, match="name/rental_rate/total_gpu_count"
        ):
            parse_machines(json.dumps([hostile]))


def test_parse_machines_value_guards_fail_open():
    """Out-of-range values on single models degrade to None + count --
    they never blank the whole map (and NaN survives json.loads, same
    trap as the price pin)."""
    rows = [dict(r) for r in json.loads(MACHINES_FIXTURE.read_text())[:3]]
    rows[0]["rental_rate"] = 1.7  # out of [0, 1]
    rows[1]["rental_rate"] = float("nan")  # json.dumps emits bare NaN
    rows[2]["total_gpu_count"] = -3
    by_name, value_skips, catalog_stats = parse_machines(json.dumps(rows))
    assert value_skips == {
        "bad_model_rental_rate": 2,
        "bad_model_total_gpu_count": 1,
    }
    assert by_name[rows[0]["name"]]["model_rental_rate"] is None
    # ... while the row's OTHER half survives independently.
    assert (
        by_name[rows[0]["name"]]["model_total_gpu_count"]
        == rows[0]["total_gpu_count"]
    )
    assert by_name[rows[1]["name"]]["model_rental_rate"] is None
    assert by_name[rows[2]["name"]]["model_total_gpu_count"] is None
    assert by_name[rows[2]["name"]]["model_rental_rate"] == 1.0
    # -3 is invalid, not zero-fleet: rows[0]+rows[1] are the zero pair.
    assert catalog_stats == {
        "models_in_catalog": 3,
        "models_zero_capacity": 2,
    }


def test_model_absent_from_map_counts_not_guesses():
    """A book label missing from the machines catalog records None +
    model_not_in_machines -- never a guessed occupancy."""
    obs, _, _ = parse_lium(FIXTURE.read_text())
    by_name, _, _ = parse_machines(MACHINES_FIXTURE.read_text())
    del by_name["NVIDIA B200"]
    counts = attach_model_stats(obs, by_name)
    assert counts == {"model_not_in_machines": 1}
    b200 = next(r for r in obs if r["sku_identifier"] == "NVIDIA B200")
    assert b200["extra"]["model_rental_rate"] is None
    assert b200["extra"]["model_total_gpu_count"] is None


def test_price_rows_unaffected_when_machines_fetch_raises(monkeypatch):
    """THE fail-open contract: /api/machines dying must not dark the
    price lane -- fields land as None (stable schema), the partial_error
    is the only trace, and no per-row absence is counted."""
    out = _collect(monkeypatch, machines=RuntimeError("boom"))
    rows = out["observations"]
    assert len(rows) == 17  # price parse untouched
    for r in rows:
        assert r["extra"]["model_rental_rate"] is None
        assert r["extra"]["model_total_gpu_count"] is None
    assert out["book_stats"]["availability_skips"] == {}
    assert "models_in_catalog" not in out["book_stats"]
    assert out["partial_errors"] == [
        "availability enrichment skipped -- /api/machines fetch/parse "
        "failed (price rows unaffected): boom"
    ]
    # rung 3 still lands: the two availability surfaces fail independently.
    assert len(out["book_stats"]["capacity_buckets"]) == 13


def test_price_rows_unaffected_when_machines_reshaped(monkeypatch):
    out = _collect(monkeypatch, machines=json.dumps({"machines": []}))
    assert len(out["observations"]) == 17
    assert any(
        n.startswith("availability enrichment skipped")
        and "non-empty JSON list" in n
        for n in out["partial_errors"]
    )


def test_price_rows_unaffected_when_capacity_fails(monkeypatch):
    out = _collect(monkeypatch, capacity=RuntimeError("dead redis"))
    assert len(out["observations"]) == 17
    assert "capacity_buckets" not in out["book_stats"]
    assert out["partial_errors"] == [
        "capacity buckets skipped -- /api/machines/capacity fetch/parse "
        "failed (price rows unaffected): dead redis"
    ]
    # rung 2 still attached.
    assert out["observations"][0]["extra"]["model_total_gpu_count"] == 57


def test_both_availability_surfaces_down_price_lane_survives(monkeypatch):
    out = _collect(
        monkeypatch,
        machines=RuntimeError("a"),
        capacity=json.dumps({"reshaped": True}),
    )
    assert len(out["observations"]) == 17
    assert len(out["partial_errors"]) == 2
    stats = out["book_stats"]
    assert "capacity_buckets" not in stats
    assert "models_in_catalog" not in stats
    assert stats["rows_recorded"] == 17


def test_parse_capacity_verbatim_and_reshape_fences():
    doc = json.loads(CAPACITY_FIXTURE.read_text())
    assert parse_capacity(CAPACITY_FIXTURE.read_text()) == doc
    with pytest.raises(RuntimeError, match="non-empty JSON list"):
        parse_capacity(json.dumps({"capacity": doc}))
    with pytest.raises(RuntimeError, match="non-empty JSON list"):
        parse_capacity("[]")
    nobuckets = dict(doc[0])
    del nobuckets["buckets"]
    unlabeled = dict(doc[0], base_model="  ")
    mistyped = dict(doc[0], buckets=[["1", {}]])
    for hostile in (nobuckets, unlabeled, mistyped):
        with pytest.raises(RuntimeError, match="base_model/buckets"):
            parse_capacity(json.dumps([hostile]))
