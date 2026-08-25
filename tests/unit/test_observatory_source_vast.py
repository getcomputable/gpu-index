# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory vast collector -- fixture pins (live responses 2026-08-22).

Fixtures are REAL bundles-API bodies captured live 2026-08-22 with small
limit values (a limit-8/limit-4 query IS the surface's own bytes, so no
hand trimming was needed). They deliberately preserve the edge cases this
source exposes:

  - h100_sxm_asc.json / h100_sxm_desc.json: ASC head + DESC tail of one
    book; machine 146215 appears in BOTH windows (1x slice in ASC, 8x box
    in DESC) with the 8x box CHEAPER per GPU -- the exact 08-13 burial
    class the DESC guard exists for; verification values verified /
    unverified / deverified; non-US geolocations.
  - b300_asc.json: thin whole book -- ONE machine (144888) listing two
    slice sizes (2x and 4x), so machine dedup must record exactly one row.
  - h200_nvl_asc.json: lookalike label ('H200 NVL' is not 'H200'; since
    the H-series variant split it normalizes to its own H200_NVL sku).

The verified-US/CA population branch (hourly panel design section 6) is
covered here too: the h100_sxm fixtures genuinely contain exactly two
verified US/CA machines (59380, 57753), which exercises the screen against
real bytes; cap/overflow depth is synthesized in-test because no fixture
carries 200+ eligible machines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.vast import (
    SOURCE_ID,
    chip_book_stats,
    collect,
    merge_candidate_books,
    parse_vast_offers,
    pin_candidates,
    select_chip_observations,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "observatory" / "vast"


def _offer(**overrides):
    base = {
        "id": 1,
        "machine_id": 10,
        "host_id": 100,
        "num_gpus": 1,
        "dph_total": 2.0,
        "is_bid": False,
        "geolocation": "Texas, US",
        "verification": "verified",
        "gpu_name": "H100 SXM",
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def h100_candidates():
    asc = parse_vast_offers((FIXTURES / "h100_sxm_asc.json").read_text())
    desc = parse_vast_offers((FIXTURES / "h100_sxm_desc.json").read_text())
    merged = merge_candidate_books(asc, desc)
    pinned, skipped = pin_candidates(merged, "H100 SXM")
    assert skipped == 0  # real responses honor the eq filter
    return pinned


@pytest.fixture(scope="module")
def h100_rows(h100_candidates):
    return select_chip_observations(h100_candidates, 10)


def test_source_id_matches_module():
    assert SOURCE_ID == "vast"


def test_cheapest_machine_pinned_exactly(h100_rows):
    row = h100_rows[0]
    assert row["sku_identifier"] == "H100 SXM"
    assert row["price_usd_gpu_hr"] == 1.3867
    assert row["price_native_per_gpu_hr"] == 1.3867
    assert row["currency"] == "USD"
    assert row["raw_value"] == "1.3867407407407408"
    assert row["raw_unit"] == "usd_per_instance_hr"
    assert row["gpu_count_basis"] == 1
    assert row["tier"] == "on-demand"
    assert row["region"] == ", US"  # geolocation exactly as published
    assert row["offer_id"] == 48285735
    assert row["machine_id"] == 148199
    assert row["host_id"] == 635532
    assert row["verification"] == "unverified"
    assert "machine 148199" in row["notes"]


def test_multi_gpu_box_wins_machine_dedup_across_windows(h100_rows):
    """Machine 146215 lists a 1x slice (ASC window) and an 8x box (DESC
    window) and the 8x box is CHEAPER per GPU -- exactly one row, the 8x
    offer, and price*basis must reproduce the raw instance total (L0)."""
    rows = [r for r in h100_rows if r["machine_id"] == 146215]
    assert len(rows) == 1
    row = rows[0]
    assert row["offer_id"] == 48370130
    assert row["gpu_count_basis"] == 8
    assert row["price_usd_gpu_hr"] == 1.9335
    raw = float(row["raw_value"])
    basis = row["gpu_count_basis"]
    assert abs(row["price_usd_gpu_hr"] * basis - raw) <= 0.005 * basis


def test_record_limit_and_per_gpu_ranking(h100_rows):
    assert len(h100_rows) == 10  # 14 machines in the merged book, limit 10
    prices = [r["price_usd_gpu_hr"] for r in h100_rows]
    assert prices == sorted(prices)
    assert [r["machine_id"] for r in h100_rows[:3]] == [148199, 148240, 140072]
    # deverified is a real verification state and rides through untouched.
    deverified = [r for r in h100_rows if r["verification"] == "deverified"]
    assert [r["machine_id"] for r in deverified] == [142073]


def test_one_machine_two_slice_sizes_records_one_row():
    """B300 thin book: machine 144888 lists 2x and 4x slices; the 4x is
    cheaper per GPU and must be the single recorded row."""
    candidates = parse_vast_offers((FIXTURES / "b300_asc.json").read_text())
    pinned, skipped = pin_candidates(candidates, "B300")
    assert (len(pinned), skipped) == (2, 0)
    rows = select_chip_observations(pinned, 10)
    assert len(rows) == 1
    row = rows[0]
    assert row["sku_identifier"] == "B300"
    assert row["offer_id"] == 47419091
    assert row["machine_id"] == 144888
    assert row["gpu_count_basis"] == 4
    assert row["price_usd_gpu_hr"] == 7.5005
    assert row["raw_value"] == "30.00208333333334"
    assert row["region"] == "Utah, US"


def test_h200_nvl_rows_pin():
    candidates = parse_vast_offers(
        (FIXTURES / "h200_nvl_asc.json").read_text()
    )
    pinned, skipped = pin_candidates(candidates, "H200 NVL")
    assert (len(pinned), skipped) == (4, 0)
    rows = select_chip_observations(pinned, 10)
    assert rows[0]["price_usd_gpu_hr"] == 2.6689
    assert rows[0]["region"] == "California, US"
    assert all(r["sku_identifier"] == "H200 NVL" for r in rows)


def test_book_stats_account_for_truncation(h100_candidates):
    stats = chip_book_stats(
        h100_candidates, 10, fetch_truncated=True, coverage_gap=False
    )
    assert stats == {
        "candidate_offers": 16,
        "machines_total": 14,
        "rows_recorded": 10,
        "fetch_truncated": True,
        "coverage_gap": False,
        "per_gpu_min": 1.3867,
        "per_gpu_max": 5.0939,
    }
    assert chip_book_stats([], 10) == {
        "candidate_offers": 0,
        "machines_total": 0,
        "rows_recorded": 0,
        "fetch_truncated": False,
        "coverage_gap": False,
    }


def test_identity_pin_skips_mismatched_gpu_name():
    """An offer the server returns under the wrong label (eq filter broke)
    must be skipped and counted, never recorded into the queried book."""
    body = json.dumps(
        {
            "offers": [
                _offer(id=1, gpu_name="H100 SXM"),
                _offer(id=2, machine_id=11, gpu_name="H100 NVL"),
                _offer(id=3, machine_id=12, gpu_name=None),
            ]
        }
    )
    pinned, skipped = pin_candidates(parse_vast_offers(body), "H100 SXM")
    assert [c["offer_id"] for c in pinned] == [1]
    assert skipped == 2


def test_bids_and_invalid_basis_rows_never_print():
    """A spot BID recorded as an on-demand ask would bias the print low; a
    missing/zero num_gpus must never default to a per-GPU basis of 1."""
    body = json.dumps(
        {
            "offers": [
                _offer(id=1, is_bid=True),
                _offer(id=2, num_gpus=0),
                _offer(id=3, num_gpus=None),
                _offer(id=4, dph_total=None),
            ]
        }
    )
    assert parse_vast_offers(body) == []


def test_merge_dedups_overlapping_offer_ids():
    asc = parse_vast_offers(
        json.dumps({"offers": [_offer(id=1), _offer(id=2, machine_id=11)]})
    )
    desc = parse_vast_offers(
        json.dumps({"offers": [_offer(id=2, machine_id=11), _offer(id=3, machine_id=12)]})
    )
    merged = merge_candidate_books(asc, desc)
    assert sorted(c["offer_id"] for c in merged) == [1, 2, 3]


def test_collect_fails_closed_without_options():
    """No config, no chip list: collect must raise before any fetch."""
    with pytest.raises(RuntimeError, match="refusing to guess"):
        collect(options=None)
    with pytest.raises(RuntimeError, match="gpu_names"):
        collect(options={"record_limit_per_gpu": 10})
    with pytest.raises(RuntimeError, match="duplicates"):
        collect(
            options={
                "gpu_names": ["B200", "B200"],
                "record_limit_per_gpu": 10,
            }
        )
    with pytest.raises(RuntimeError, match="record_limit_per_gpu"):
        collect(options={"gpu_names": ["B200"], "record_limit_per_gpu": 0})


# ------------------------------------------- verified-US/CA population branch


def test_population_rows_appended_screened_and_marked(h100_candidates):
    """Design section 6: after the cheapest-N rows, every OTHER verified
    US/CA machine rides in per-GPU ascending, marked with book_scope.
    Fixture truth (record_limit=3): cheapest-3 = 148199/148240/140072;
    the only verified US/CA machines beyond them are 59380 (1.7356) and
    57753 (5.0939); verified non-US/CA (68475 Japan, 33102 India) and
    unverified-US (146215) machines must NOT be appended."""
    rows = select_chip_observations(h100_candidates, 3, population=True)
    head, population = rows[:3], rows[3:]
    assert [r["machine_id"] for r in head] == [148199, 148240, 140072]
    assert all("book_scope" not in r for r in head)
    assert [r["machine_id"] for r in population] == [59380, 57753]
    assert [r["price_usd_gpu_hr"] for r in population] == [1.7356, 5.0939]
    for row in population:
        assert row["book_scope"] == "verified_us_ca_population"
        assert row["verification"] == "verified"
        assert row["sku_identifier"] == "H100 SXM"
        assert "population row" in row["notes"]
    recorded = {r["machine_id"] for r in rows}
    assert 68475 not in recorded  # verified, Japan
    assert 33102 not in recorded  # verified, India
    assert 146215 not in {r["machine_id"] for r in population}  # unverified


def test_non_population_chips_record_exactly_the_old_shape(h100_candidates):
    """A chip NOT in population_gpu_names must record byte-identical rows
    to the pre-branch behavior: cheapest-N only, no book_scope anywhere,
    no population note."""
    default = select_chip_observations(h100_candidates, 3)
    explicit = select_chip_observations(h100_candidates, 3, population=False)
    assert default == explicit
    assert len(default) == 3
    assert all("book_scope" not in r for r in default)
    assert all("population row" not in r["notes"] for r in default)
    # The population call's head is the same continuity record.
    with_pop = select_chip_observations(h100_candidates, 3, population=True)
    assert with_pop[:3] == default


def test_population_cap_and_overflow(monkeypatch):
    """The cap is a safety bound, never silent: rows stop at the limit and
    book_stats flags the overflow. Depth synthesized (no fixture carries
    200+ eligible machines); the limit is monkeypatched down instead of
    faking 200 offers."""
    monkeypatch.setattr("gpu_index.observatory.sources.vast.VAST_POPULATION_LIMIT", 2)
    offers = [
        _offer(id=i, machine_id=100 + i, dph_total=1.0 + 0.1 * i)
        for i in range(6)  # all verified, 'Texas, US'
    ]
    candidates = parse_vast_offers(json.dumps({"offers": offers}))
    rows = select_chip_observations(candidates, 1, population=True)
    assert [r["machine_id"] for r in rows] == [100, 101, 102]
    assert [r.get("book_scope") for r in rows] == [
        None,
        "verified_us_ca_population",
        "verified_us_ca_population",
    ]
    stats = chip_book_stats(candidates, 1, population=True)
    assert stats["machines_total"] == 6
    assert stats["verified_us_ca_machines"] == 6
    assert stats["rows_recorded"] == 3  # cheapest-1 + capped population 2
    assert stats["population_recorded"] is True
    assert stats["population_overflow"] is True


def test_book_stats_population_accounting(h100_candidates):
    """Population stats ride only on population chips; the screened count
    covers the WHOLE deduped book (truncation never invisible).
    An empty population book still records population_recorded=true --
    'the branch ran and found nothing' is distinguishable from 'the
    branch did not exist'."""
    stats = chip_book_stats(h100_candidates, 3, population=True)
    assert stats["machines_total"] == 14
    assert stats["verified_us_ca_machines"] == 2
    assert stats["rows_recorded"] == 5  # cheapest-3 + 2 population rows
    assert stats["population_recorded"] is True
    assert stats["population_overflow"] is False
    # Non-population chips: none of the new keys, shape unchanged.
    plain = chip_book_stats(h100_candidates, 3)
    for key in (
        "verified_us_ca_machines",
        "population_recorded",
        "population_overflow",
    ):
        assert key not in plain
    empty = chip_book_stats([], 3, population=True)
    assert empty["verified_us_ca_machines"] == 0
    assert empty["population_recorded"] is True
    assert empty["population_overflow"] is False
    assert empty["rows_recorded"] == 0


def test_population_options_validation():
    """population_gpu_names is config-is-the-contract: malformed values,
    duplicates, and chips that are never queried all fail loudly at load,
    before any fetch."""
    base = {"gpu_names": ["B200", "H100 SXM"], "record_limit_per_gpu": 10}
    with pytest.raises(RuntimeError, match="population_gpu_names"):
        collect(options={**base, "population_gpu_names": "B200"})
    with pytest.raises(RuntimeError, match="population_gpu_names"):
        collect(options={**base, "population_gpu_names": [1]})
    with pytest.raises(
        RuntimeError, match="population_gpu_names contains duplicates"
    ):
        collect(options={**base, "population_gpu_names": ["B200", "B200"]})
    # A population chip missing from gpu_names would silently record no
    # population -- the invisible-truncation class this branch kills.
    with pytest.raises(RuntimeError, match="never queried"):
        collect(options={**base, "population_gpu_names": ["H200 NVL"]})


def test_collect_wires_population_from_config(monkeypatch, h100_candidates):
    """collect() must pass the per-chip population flag through from
    options.population_gpu_names -- rows and book_stats both prove which
    branch ran for which chip."""
    monkeypatch.setattr("gpu_index.observatory.sources.vast.REQUEST_SPACING_SECONDS", 0)

    def fake_fetch(gpu_name, timeout):
        cands = [dict(c, gpu_name=gpu_name) for c in h100_candidates]
        return len(cands), cands, False, False

    monkeypatch.setattr(
        "gpu_index.observatory.sources.vast._fetch_chip_book", fake_fetch
    )
    out = collect(
        options={
            "gpu_names": ["H100 SXM", "H100 PCIE"],
            "record_limit_per_gpu": 3,
            "population_gpu_names": ["H100 SXM"],
        }
    )
    sxm_rows = [
        r for r in out["observations"] if r["sku_identifier"] == "H100 SXM"
    ]
    pcie_rows = [
        r for r in out["observations"] if r["sku_identifier"] == "H100 PCIE"
    ]
    assert [r.get("book_scope") for r in sxm_rows] == [
        None,
        None,
        None,
        "verified_us_ca_population",
        "verified_us_ca_population",
    ]
    assert len(pcie_rows) == 3
    assert all("book_scope" not in r for r in pcie_rows)
    sxm_stats = out["book_stats"]["H100 SXM"]
    pcie_stats = out["book_stats"]["H100 PCIE"]
    assert sxm_stats["population_recorded"] is True
    assert sxm_stats["verified_us_ca_machines"] == 2
    assert sxm_stats["rows_recorded"] == 5
    assert "population_recorded" not in pcie_stats
    assert "verified_us_ca_machines" not in pcie_stats
    assert pcie_stats["rows_recorded"] == 3


def test_real_config_pins_population_chips():
    """The shipped config must arm the population branch for exactly the
    design section 6 chip list, every name also queried."""
    cfg = json.loads(
        (REPO_ROOT / "config" / "raw_observatory.json").read_text()
    )
    vast_src = next(s for s in cfg["sources"] if s["source_id"] == "vast")
    opts = vast_src["options"]
    assert opts["population_gpu_names"] == [
        "B200",
        "H100 SXM",
        "H100 NVL",
        "H100 PCIE",
        "H200",
        "H200 NVL",
    ]
    assert set(opts["population_gpu_names"]) <= set(opts["gpu_names"])


def test_real_labels_normalize_through_catalog():
    """Framework normalization over this source's real labels. The full
    configured gpu_names list is covered by the catalog today; genuinely
    new chips belong in catalog_suggestions, not in this pin."""
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    labels = set()
    for name in (
        "h100_sxm_asc.json",
        "h100_sxm_desc.json",
        "b300_asc.json",
        "h200_nvl_asc.json",
    ):
        for cand in parse_vast_offers((FIXTURES / name).read_text()):
            labels.add(cand["gpu_name"])
    mapped = {
        label: (match_sku(catalog, label) or {"sku": None})["sku"]
        for label in labels
    }
    assert mapped["H100 SXM"] == "H100"
    assert mapped["B300"] == "B300"
    # Variant separation (design section 7): 'H200 NVL' lands on its own
    # H200_NVL sku now -- never on H200, and never on H20.
    assert mapped["H200 NVL"] == "H200_NVL"
    unmapped = [label for label, sku in mapped.items() if sku is None]
    assert not unmapped, f"known vast labels now unmapped: {unmapped}"
