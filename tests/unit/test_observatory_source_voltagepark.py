# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory voltagepark collector -- fixture pins (live response 2026-08-22).

House style per the runpod exemplar: (1) parse the recorded fixture (REAL
bytes from the live locations API, captured 2026-08-22 -- one h100-sxm5-80gb
row, both availability counts genuinely 0 while prices stayed published),
(2) pin exact prints for the known rows incl. this source's edge cases,
(3) prove the framework normalization maps this source's real label, plus
the fail-closed pins: pagination activation, total-count mismatch, renamed
price fields (with their numeric count-field lookalikes still present), and
the string-vs-number price type flip the app schema allows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.voltagepark import SOURCE_ID, parse_voltagepark

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "voltagepark"
    / "locations.json"
)


def _body(mutate=None) -> str:
    """The real fixture, optionally mutated in place -- every fail-closed
    test is a minimal perturbation of genuinely-live bytes."""
    payload = json.loads(FIXTURE.read_text())
    if mutate:
        mutate(payload)
    return json.dumps(payload)


@pytest.fixture(scope="module")
def rows():
    return parse_voltagepark(FIXTURE.read_text())


def test_source_id_matches_module():
    assert SOURCE_ID == "voltagepark"


def test_both_network_tiers_pinned_exactly(rows):
    assert len(rows) == 2
    by_net = {r["extra"]["network"]: r for r in rows}
    assert set(by_net) == {"ethernet", "infiniband"}
    eth = by_net["ethernet"]
    assert eth["sku_identifier"] == "h100-sxm5-80gb"
    assert eth["price_usd_gpu_hr"] == 1.99
    assert eth["raw_value"] == "1.99"
    ib = by_net["infiniband"]
    assert ib["sku_identifier"] == "h100-sxm5-80gb"
    assert ib["price_usd_gpu_hr"] == 2.49
    assert ib["raw_value"] == "2.49"
    # Both are networking variants of the same on-demand product -- NOT an
    # on-demand/reserved split; the native label lives in extra only.
    assert all(r["tier"] == "on-demand" for r in rows)
    assert all(r["currency"] == "USD" for r in rows)


def test_per_gpu_basis_never_multiplied_by_node_count(rows):
    """Prices are per GPU per hour on an 8-GPU node: basis stays 1 and
    price * basis reproduces the raw figure exactly."""
    for r in rows:
        assert r["gpu_count_basis"] == 1
        assert r["raw_unit"] == "usd_per_gpu_hr"
        assert r["price_usd_gpu_hr"] * r["gpu_count_basis"] == float(
            r["raw_value"]
        )
        assert r["extra"]["node_gpu_count"] == 8


def test_zero_availability_does_not_gate_price(rows):
    """Live capture had gpu_count_ethernet == gpu_count_infiniband == 0
    while both prices stayed published -- availability is metadata, never a
    price gate."""
    assert all(r["extra"]["available_gpu_count"] == 0 for r in rows)
    assert len(rows) == 2


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {"h100-sxm5-80gb": "H100"}


def test_top_level_reshape_raises():
    for key in ("results", "total_result_count"):
        body = _body(lambda p, k=key: p.pop(k))
        with pytest.raises(RuntimeError, match="top level reshaped"):
            parse_voltagepark(body)


def test_has_next_true_raises():
    body = _body(lambda p: p.__setitem__("has_next", True))
    with pytest.raises(RuntimeError, match="has_next"):
        parse_voltagepark(body)


def test_total_count_mismatch_raises():
    body = _body(lambda p: p.__setitem__("total_result_count", 2))
    with pytest.raises(RuntimeError, match="total_result_count"):
        parse_voltagepark(body)


def test_renamed_price_field_raises_despite_count_lookalike():
    """Drop gpu_price_infiniband but leave gpu_count_infiniband (numeric,
    present) -- the pin is on the exact price field name, so a lookalike
    count must never stand in for a price."""
    body = _body(lambda p: p["results"][0].pop("gpu_price_infiniband"))
    with pytest.raises(RuntimeError, match="gpu_price_infiniband"):
        parse_voltagepark(body)


def test_unknown_price_column_raises():
    """A NEW gpu_price_* column (a spot tier, a reserved price going public)
    is a published price this recipe never saw -- it must raise, never parse
    the two known tiers and silently under-report the third."""
    for new_key in ("gpu_price_spot", "reserved_gpu_price_cluster_one"):
        body = _body(
            lambda p, k=new_key: p["results"][0].__setitem__(k, "0.99")
        )
        with pytest.raises(RuntimeError, match="unknown price field"):
            parse_voltagepark(body)


def test_price_type_flip_to_number_still_parses():
    """The app schema accepts number-or-string; a flip to a JSON number must
    keep parsing (with the same value) rather than darking the source."""
    body = _body(
        lambda p: p["results"][0].__setitem__("gpu_price_ethernet", 1.99)
    )
    rows = parse_voltagepark(body)
    eth = next(r for r in rows if r["extra"]["network"] == "ethernet")
    assert eth["price_usd_gpu_hr"] == 1.99
    assert eth["raw_value"] == "1.99"


def test_nonpositive_or_unparseable_price_raises():
    for bad in ("0", "-1.00", "", "contact us", None, True):
        body = _body(
            lambda p, b=bad: p["results"][0].__setitem__(
                "gpu_price_ethernet", b
            )
        )
        with pytest.raises(RuntimeError, match="gpu_price_ethernet"):
            parse_voltagepark(body)


def test_missing_gpu_model_raises():
    """One row, one chip: a lost identity label is a reshape, not a row to
    quietly skip."""
    for mutate in (
        lambda p: p["results"][0]["specs_per_node"].__setitem__(
            "gpu_model", ""
        ),
        lambda p: p["results"][0].pop("specs_per_node"),
    ):
        with pytest.raises(RuntimeError, match="voltagepark"):
            parse_voltagepark(_body(mutate))
