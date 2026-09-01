# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Hyperbolic observatory collector pins from live bytes captured 2026-08-29.

The real response churned down to 13 rows at branch cut. It still contains
the identity and normalization edges this recipe needs: an available PCIe
row, 8-GPU VMs, and a three-region bare-metal ladder. Every hostile-shape
test below is a minimal mutation of that exact live fixture.

The providerCostPerHourCents values in the recorded fixture are SYNTHETIC
stand-ins: Hyperbolic was promised its supplier cost would never be
republished and this repo is public. Customer prices, counts and field
shapes are as recorded, and the recipe never reads that field, so every
pin below is unchanged by the substitution.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.collect import collect_all
from gpu_index.observatory.sources.hyperbolic import (
    SOURCE_ID,
    URL,
    collect,
    parse_hyperbolic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "hyperbolic"
    / "rental_options.json"
)


def _body(mutate=None) -> str:
    payload = json.loads(FIXTURE.read_text())
    if mutate:
        mutate(payload)
    return json.dumps(payload)


@pytest.fixture(scope="module")
def rows():
    return parse_hyperbolic(FIXTURE.read_text())


def test_source_id_matches_module():
    assert SOURCE_ID == "hyperbolic"


def test_config_classifies_hyperbolic_as_first_party_partnered_source():
    config = json.loads(
        (REPO_ROOT / "config" / "raw_observatory.json").read_text()
    )
    source = next(
        row for row in config["sources"] if row["source_id"] == SOURCE_ID
    )
    assert source["display_name"] == "Hyperbolic"
    assert source["source_type"] == "direct_partnered"
    assert source["first_party"] is True


def test_live_fixture_contains_required_edge_rows():
    payload = json.loads(FIXTURE.read_text())
    assert len(payload) == 13
    assert any(
        row["gpuFormFactor"] == "pcie" and "totalAvailable" in row
        for row in payload
    )
    assert any(
        row["machineType"] == "virtual-machine" and row["gpuCount"] > 1
        for row in payload
    )
    assert sum(row["machineType"] == "bare-metal" for row in payload) == 3


def test_every_known_print_is_pinned_exactly(rows):
    actual = {
        (
            row["region"],
            row["sku_identifier"],
            row["gpu_count_basis"],
            row["raw_value"],
            row["price_usd_gpu_hr"],
        )
        for row in rows
    }
    assert actual == {
        ("us-east-1", "H100 SXM5", 1, "319", 3.19),
        ("us-east-2", "H200 SXM5", 1, "399", 3.99),
        ("ca-east-3", "H100 SXM5", 1, "485", 4.85),
        ("us-east-1", "H200 SXM5", 1, "492", 4.92),
        ("eu-north-6", "H100 SXM5", 1, "358", 3.58),
        ("eu-north-7", "H100 SXM5", 1, "358", 3.58),
        ("ca-east-1", "H100 PCIE", 1, "275", 2.75),
        ("uk-southeast-3", "H100 SXM5", 8, "2552", 3.19),
        ("eu-west-3", "H100 SXM5", 8, "2552", 3.19),
        ("eu-west-4", "H100 SXM5", 8, "2552", 3.19),
        ("eu-north-4", "H100 SXM5", 8, "2552", 3.19),
        ("eu-north-5", "H200 SXM5", 8, "3192", 3.99),
        ("jp-east-1", "H100 SXM5", 8, "2552", 3.19),
    }
    assert len(rows) == 13
    assert all(row["raw_unit"] == "cents_per_instance_hr" for row in rows)
    assert all(row["tier"] == "on-demand" for row in rows)


def test_whole_option_arithmetic_never_records_instance_total_as_per_gpu(rows):
    for row in rows:
        expected = (
            Decimal(row["raw_value"])
            / Decimal(row["gpu_count_basis"])
            / Decimal(100)
        )
        assert Decimal(str(row["price_usd_gpu_hr"])) == expected
    multi_gpu = [row for row in rows if row["gpu_count_basis"] == 8]
    assert multi_gpu
    assert all(
        row["price_usd_gpu_hr"] != float(Decimal(row["raw_value"]) / 100)
        for row in multi_gpu
    )


def test_bare_metal_ladder_resolves_to_same_per_gpu_rate(rows):
    ladder = [
        row
        for row in rows
        if row["extra"]["machine_type"] == "bare-metal"
    ]
    assert {row["region"] for row in ladder} == {
        "uk-southeast-3",
        "eu-west-3",
        "eu-west-4",
    }
    assert {row["price_usd_gpu_hr"] for row in ladder} == {3.19}


def test_live_network_metadata_is_preserved_as_validated_text(rows):
    first = rows[0]
    assert first["extra"]["connection_type"] == "ethernet"
    assert first["extra"]["ethernet_variant"] == "normal"


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        row["sku_identifier"]: (
            match_sku(catalog, row["sku_identifier"]) or {"sku": None}
        )["sku"]
        for row in rows
    }
    assert mapped == {
        "H100 SXM5": "H100",
        "H100 PCIE": "H100_PCIE",
        "H200 SXM5": "H200",
    }


@pytest.mark.parametrize("bad", [{"options": []}, None, "options", 3])
def test_non_list_top_level_raises(bad):
    with pytest.raises(RuntimeError, match="top level is not a list"):
        parse_hyperbolic(json.dumps(bad))


def test_non_object_row_raises():
    with pytest.raises(RuntimeError, match="row 0 is not an object"):
        parse_hyperbolic(json.dumps(["h100"]))


def test_missing_or_renamed_cost_raises():
    for mutate in (
        lambda payload: payload[0].pop("costPerHourCents"),
        lambda payload: payload[0].update(
            {"hourlyCostCents": payload[0].pop("costPerHourCents")}
        ),
    ):
        with pytest.raises(RuntimeError, match="costPerHourCents"):
            parse_hyperbolic(_body(mutate))


@pytest.mark.parametrize(
    "bad",
    [
        0,
        -1,
        "",
        "contact us",
        None,
        True,
        float("inf"),
        float("nan"),
        "1e10000",
    ],
)
def test_nonpositive_nonfinite_or_unparseable_cost_raises(bad):
    body = _body(
        lambda payload: payload[0].__setitem__("costPerHourCents", bad)
    )
    with pytest.raises(RuntimeError, match="costPerHourCents"):
        parse_hyperbolic(body)


def test_missing_gpu_count_raises_instead_of_defaulting_to_one():
    body = _body(lambda payload: payload[0].pop("gpuCount"))
    with pytest.raises(RuntimeError, match="refusing to default to 1"):
        parse_hyperbolic(body)


@pytest.mark.parametrize("bad", [0, -1, None, True, "8", 8.0])
def test_non_integer_or_nonpositive_gpu_count_raises(bad):
    body = _body(lambda payload: payload[0].__setitem__("gpuCount", bad))
    with pytest.raises(RuntimeError, match="gpuCount"):
        parse_hyperbolic(body)


def test_gpu_count_that_underflows_normalized_float_raises():
    body = _body(
        lambda payload: payload[0].__setitem__("gpuCount", 10**400)
    )
    with pytest.raises(RuntimeError, match="positive finite per-GPU"):
        parse_hyperbolic(body)


def test_positive_price_that_rounds_to_published_zero_raises():
    body = _body(
        lambda payload: payload[0].__setitem__("costPerHourCents", "0.001")
    )
    with pytest.raises(RuntimeError, match="positive finite per-GPU"):
        parse_hyperbolic(body)


def test_unknown_per_hour_cents_field_raises():
    body = _body(
        lambda payload: payload[0].__setitem__("spotPerHourCents", 99)
    )
    with pytest.raises(RuntimeError, match="unknown price field"):
        parse_hyperbolic(body)


@pytest.mark.parametrize("field", ["gpuType", "gpuFormFactor"])
@pytest.mark.parametrize("bad", [None, "", "   ", 100])
def test_missing_or_blank_label_component_raises(field, bad):
    body = _body(lambda payload: payload[0].__setitem__(field, bad))
    with pytest.raises(RuntimeError, match=field):
        parse_hyperbolic(body)


@pytest.mark.parametrize("bad", [None, "", "   ", 1])
def test_missing_or_blank_region_raises(bad):
    body = _body(lambda payload: payload[0].__setitem__("region", bad))
    with pytest.raises(RuntimeError, match="region"):
        parse_hyperbolic(body)


@pytest.mark.parametrize(
    "field", ["connectionType", "machineType", "ethernetVariant"]
)
def test_metadata_text_fields_cannot_smuggle_nested_supplier_cost(field):
    body = _body(
        lambda payload: payload[0].__setitem__(
            field, {"providerCostPerHourCents": 123}
        )
    )
    with pytest.raises(RuntimeError, match=field):
        parse_hyperbolic(body)


def test_malformed_availability_becomes_unknown_without_smuggling_cost():
    body = _body(
        lambda payload: payload[0].__setitem__(
            "totalAvailable", {"providerCostPerHourCents": 123}
        )
    )
    rows = parse_hyperbolic(body)
    assert len(rows) == 13
    assert rows[0]["price_usd_gpu_hr"] == 3.19
    assert rows[0]["extra"]["available_gpu_count"] is None
    assert "providerCostPerHourCents" not in json.dumps(rows, sort_keys=True)


@pytest.mark.parametrize("bad", [True, -1])
def test_invalid_scalar_availability_becomes_unknown_without_gating_price(bad):
    rows = parse_hyperbolic(
        _body(lambda payload: payload[0].__setitem__("totalAvailable", bad))
    )
    assert len(rows) == 13
    assert rows[0]["price_usd_gpu_hr"] == 3.19
    assert rows[0]["extra"]["available_gpu_count"] is None


def test_rejected_supplier_cost_never_leaks_through_collect_all_error(
    monkeypatch,
):
    body = _body(
        lambda payload: payload[0].__setitem__(
            "connectionType", {"providerCostPerHourCents": 441}
        )
    )
    monkeypatch.setattr(
        "gpu_index.observatory.sources.hyperbolic.fetch",
        lambda url, headers=None, timeout=None: body,
    )
    config = {
        "capture_budget_seconds": 5,
        "per_source_deadline_seconds": 2,
        "per_source_timeout_seconds": 1,
        "sources": [{"source_id": "hyperbolic"}],
    }
    out = collect_all(config, {"hyperbolic": collect})
    assert out[0]["status"] == "error"
    assert out[0]["failure_kind"] == "parse"
    assert "providerCostPerHourCents" not in json.dumps(out, sort_keys=True)


@pytest.mark.parametrize("bad", [None, 1, 0, "true"])
def test_non_bool_enabled_raises(bad):
    body = _body(lambda payload: payload[0].__setitem__("enabled", bad))
    with pytest.raises(RuntimeError, match="enabled"):
        parse_hyperbolic(body)


def test_disabled_row_still_records_published_price():
    rows = parse_hyperbolic(
        _body(lambda payload: payload[0].__setitem__("enabled", False))
    )
    assert len(rows) == 13
    assert rows[0]["price_usd_gpu_hr"] == 3.19
    assert rows[0]["extra"]["enabled"] is False


def test_missing_availability_is_unknown_and_never_gates_price(rows):
    unknown = [
        row
        for row in rows
        if row["extra"]["available_gpu_count"] is None
    ]
    assert len(unknown) == 12
    assert all(row["price_usd_gpu_hr"] > 0 for row in unknown)
    pcie = next(row for row in rows if row["region"] == "ca-east-1")
    assert pcie["extra"]["available_gpu_count"] == 2
    assert pcie["price_usd_gpu_hr"] == 2.75


def test_zero_availability_still_records_published_price():
    rows = parse_hyperbolic(
        _body(lambda payload: payload[0].__setitem__("totalAvailable", 0))
    )
    assert len(rows) == 13
    assert rows[0]["extra"]["available_gpu_count"] == 0
    assert rows[0]["price_usd_gpu_hr"] == 3.19


def test_price_uses_customer_cost_not_supplier_cost():
    body = _body(
        lambda payload: payload[0].__setitem__(
            "providerCostPerHourCents", 999999
        )
    )
    first = parse_hyperbolic(body)[0]
    assert first["raw_value"] == "319"
    assert first["price_usd_gpu_hr"] == 3.19
    assert first["price_usd_gpu_hr"] != 9999.99


def test_supplier_cost_never_appears_in_observations_or_book_stats(monkeypatch):
    body = _body(
        lambda payload: payload[0]["nodes"][0].__setitem__(
            "providerCostPerHourCents", 123
        )
    )
    monkeypatch.setattr(
        "gpu_index.observatory.sources.hyperbolic.fetch",
        lambda url, headers=None, timeout=None: body,
    )
    out = collect()
    assert "book_stats" not in out
    assert "providerCostPerHourCents" not in json.dumps(out, sort_keys=True)


def test_collect_wires_shared_transport_and_result(monkeypatch):
    calls = []

    def fake_fetch(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return FIXTURE.read_text()

    monkeypatch.setattr("gpu_index.observatory.sources.hyperbolic.fetch", fake_fetch)
    out = collect(timeout=7.5, options={"ignored": True})
    assert calls == [(URL, {"accept": "application/json"}, 7.5)]
    assert out["source_id"] == "hyperbolic"
    assert out["method"] == "api-json"
    assert out["url"] == URL
    assert len(out["observations"]) == 13
    assert "partial_errors" not in out


def test_collect_raises_when_live_array_parses_nothing(monkeypatch):
    monkeypatch.setattr(
        "gpu_index.observatory.sources.hyperbolic.fetch",
        lambda url, headers=None, timeout=None: "[]",
    )
    with pytest.raises(RuntimeError, match="zero GPU price observations"):
        collect()
