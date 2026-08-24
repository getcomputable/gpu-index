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
  - h200_nvl_asc.json: lookalike label ('H200 NVL' is not 'H200', and must
    still normalize to the H200 catalog sku).
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
    # Lookalike honesty: 'H200 NVL' is its own vast label but the same part
    # family -- it must land on H200 (and never on H20).
    assert mapped["H200 NVL"] == "H200"
    unmapped = [label for label, sku in mapped.items() if sku is None]
    assert not unmapped, f"known vast labels now unmapped: {unmapped}"
