# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Runcrate observatory collector pins from live bytes captured 2026-11-08.

The recorded fixture is the public /api/gpu-pricing feed trimmed to ten GPU
types and to the fields the recipe reads. It keeps the identity edges this
recipe needs: an H100 row (no SXM/PCIe split in the feed, never recorded), a
V100 row (two memory sizes, never recorded), an out-of-stock MI350X row that
still carries a price, and H200 beside H200 NVL. Every hostile-shape test
below is a minimal mutation of that exact fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.runcrate import (
    SOURCE_ID,
    URL,
    collect,
    parse_runcrate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "runcrate"
    / "gpu_pricing.json"
)


def _body(mutate=None) -> str:
    payload = json.loads(FIXTURE.read_text())
    if mutate:
        mutate(payload)
    return json.dumps(payload)


def _row(payload, type_id):
    return next(g for g in payload["gpus"] if g["gpu_type"] == type_id)


@pytest.fixture(scope="module")
def parsed():
    return parse_runcrate(FIXTURE.read_text())


def test_source_id_matches_module():
    assert SOURCE_ID == "runcrate"


def test_config_classifies_runcrate_as_first_party_marketplace_source():
    config = json.loads(
        (REPO_ROOT / "config" / "raw_observatory.json").read_text()
    )
    source = next(
        row for row in config["sources"] if row["source_id"] == SOURCE_ID
    )
    assert source["display_name"] == "Runcrate"
    assert source["source_type"] == "marketplace"
    assert source["first_party"] is True


def test_live_fixture_contains_required_edge_rows():
    payload = json.loads(FIXTURE.read_text())
    ids = [g["gpu_type"] for g in payload["gpus"]]
    assert len(ids) == 10
    assert {"H100", "V100_32G", "H200", "H200_nvl"} <= set(ids)
    stale = _row(payload, "MI350X")
    assert stale["available_options"] == 0
    assert stale["min_price_hr"] > 0


def test_every_known_print_is_pinned_exactly(parsed):
    rows, _ = parsed
    actual = {
        (
            row["sku_identifier"],
            row["gpu_count_basis"],
            row["raw_value"],
            row["price_usd_gpu_hr"],
        )
        for row in rows
    }
    assert actual == {
        ("B300", 1, "17.02", 17.02),
        ("H200", 1, "5.13", 5.13),
        ("H200 NVL", 1, "4.67", 4.67),
        ("A100 80GB", 1, "1.73", 1.73),
        ("L40S", 1, "0.55", 0.55),
        ("RTX 5090", 1, "0.72", 0.72),
        ("A6000", 1, "0.57", 0.57),
    }
    assert len(rows) == 7
    assert all(row["raw_unit"] == "usd_per_gpu_hr" for row in rows)
    assert all(row["currency"] == "USD" for row in rows)
    assert all(row["tier"] == "on-demand" for row in rows)
    assert all(row["region"] == "global" for row in rows)


def test_ambiguous_and_unpinned_types_are_never_recorded(parsed):
    rows, errors = parsed
    labels = {row["sku_identifier"] for row in rows}
    assert "H100" not in labels
    assert "V100 32GB" not in labels
    assert "MI350X" not in labels
    # Outside the identity pin is a silent skip, not a counted anomaly.
    assert errors == []


def test_min_price_is_recorded_not_max_or_avg(parsed):
    rows, _ = parsed
    payload = json.loads(FIXTURE.read_text())
    by_label = {row["sku_identifier"]: row for row in rows}
    h200 = _row(payload, "H200")
    assert h200["max_price_hr"] != h200["min_price_hr"]
    assert by_label["H200"]["price_usd_gpu_hr"] == h200["min_price_hr"]


def test_availability_extra_is_preserved(parsed):
    rows, _ = parsed
    by_label = {row["sku_identifier"]: row for row in rows}
    assert by_label["H200"]["extra"] == {
        "gpu_type_id": "H200",
        "available_options": 15,
    }


def test_real_labels_normalize_through_catalog(parsed):
    rows, _ = parsed
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        row["sku_identifier"]: (
            match_sku(catalog, row["sku_identifier"]) or {"sku": None}
        )["sku"]
        for row in rows
    }
    assert mapped == {
        "B300": "B300",
        "H200": "H200",
        "H200 NVL": "H200_NVL",
        "A100 80GB": "A100",
        "L40S": "L40S",
        "RTX 5090": "RTX_5090",
        "A6000": "RTX_A6000",
    }


@pytest.mark.parametrize("bad", [[], None, "gpus", 3, {"gpus": {}}, {"rows": []}])
def test_reshaped_top_level_raises(bad):
    with pytest.raises(RuntimeError, match="reshaped"):
        parse_runcrate(json.dumps(bad))


def test_out_of_stock_pinned_type_is_counted_and_not_recorded():
    def mutate(payload):
        _row(payload, "L40S")["available_options"] = 0

    rows, errors = parse_runcrate(_body(mutate))
    assert "L40S" not in {r["sku_identifier"] for r in rows}
    assert len(rows) == 6
    assert len(errors) == 1
    assert "L40S" in errors[0] and "no live offers" in errors[0]


@pytest.mark.parametrize(
    "bad", [None, "5.13", True, 0, -1.0, float("nan"), float("inf"), [5.13]]
)
def test_unusable_price_is_counted_and_not_recorded(bad):
    def mutate(payload):
        _row(payload, "H200")["min_price_hr"] = bad

    rows, errors = parse_runcrate(_body(mutate))
    assert "H200" not in {r["sku_identifier"] for r in rows}
    assert len(rows) == 6
    assert len(errors) == 1
    assert "H200" in errors[0] and "not a usable price" in errors[0]


def test_missing_price_is_counted_and_not_recorded():
    def mutate(payload):
        del _row(payload, "H200")["min_price_hr"]

    rows, errors = parse_runcrate(_body(mutate))
    assert len(rows) == 6
    assert len(errors) == 1 and "not a usable price" in errors[0]


@pytest.mark.parametrize("bad", [None, "15", True, 1.5, [15]])
def test_retyped_availability_is_counted_and_not_recorded(bad):
    def mutate(payload):
        _row(payload, "H200")["available_options"] = bad

    rows, errors = parse_runcrate(_body(mutate))
    assert "H200" not in {r["sku_identifier"] for r in rows}
    assert len(errors) == 1 and "available_options retyped" in errors[0]


@pytest.mark.parametrize("bad", [None, "", "  ", 7])
def test_blank_or_retyped_label_is_counted_and_not_recorded(bad):
    def mutate(payload):
        _row(payload, "H200")["name"] = bad

    rows, errors = parse_runcrate(_body(mutate))
    assert "H200" not in {r["sku_identifier"] for r in rows}
    assert len(errors) == 1 and "name is not a non-empty string" in errors[0]


def test_row_without_type_id_or_not_an_object_is_counted():
    def mutate(payload):
        payload["gpus"].append({"name": "H200", "min_price_hr": 1.0})
        payload["gpus"].append("H200")

    rows, errors = parse_runcrate(_body(mutate))
    assert len(rows) == 7
    assert len(errors) == 2


def test_h200_nvl_never_lands_on_plain_h200():
    def mutate(payload):
        _row(payload, "H200")["available_options"] = 0

    rows, _ = parse_runcrate(_body(mutate))
    assert {r["sku_identifier"] for r in rows} >= {"H200 NVL"}
    assert "H200" not in {r["sku_identifier"] for r in rows}


def test_collect_wires_shared_transport_and_result(monkeypatch):
    calls = []

    def fake_fetch(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return FIXTURE.read_text()

    monkeypatch.setattr("gpu_index.observatory.sources.runcrate.fetch", fake_fetch)
    out = collect(timeout=7.5, options={"ignored": True})
    assert calls == [(URL, {"accept": "application/json"}, 7.5)]
    assert out["source_id"] == "runcrate"
    assert out["method"] == "api-json"
    assert out["url"] == URL
    assert len(out["observations"]) == 7
    assert "partial_errors" not in out


def test_collect_reports_partial_errors(monkeypatch):
    def mutate(payload):
        _row(payload, "L40S")["available_options"] = 0

    monkeypatch.setattr(
        "gpu_index.observatory.sources.runcrate.fetch",
        lambda url, headers=None, timeout=None: _body(mutate),
    )
    out = collect()
    assert len(out["observations"]) == 6
    assert len(out["partial_errors"]) == 1


def test_collect_raises_when_feed_parses_nothing(monkeypatch):
    monkeypatch.setattr(
        "gpu_index.observatory.sources.runcrate.fetch",
        lambda url, headers=None, timeout=None: json.dumps({"gpus": []}),
    )
    with pytest.raises(RuntimeError, match="zero GPU price observations"):
        collect()
