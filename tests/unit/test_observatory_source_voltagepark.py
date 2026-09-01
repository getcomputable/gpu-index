# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory voltagepark collector -- fixture pins (live 2026-08-22/25).

House style per the runpod exemplar: (1) parse the recorded fixture (from
the live locations API, captured 2026-08-22 and re-verified live
2026-08-25 -- one h100-sxm5-80gb row, both availability
counts genuinely 0 while prices stayed published; the location UUID is a
SYNTHETIC stand-in, prices and field shapes are as recorded), (2) pin exact prints for
the known rows incl. this source's edge cases, (3) prove the framework
normalization maps this source's real label, plus the fail-closed pins:
pagination activation, total-count mismatch, renamed price fields (with
their numeric count-field lookalikes still present), and the
string-vs-number price type flip the app schema allows.

Availability fixtures (availability accrual), both captured live
2026-08-25, untrimmed (the payloads are tiny). Location and preset UUIDs
are SYNTHETIC stand-ins; prices, field shapes and counts are as recorded:

  - instant_deploy_presets.json -- all 4 published H100 VM presets
    (1x/2x/4x/8x; edge rows kept: the 8-GPU "15.120000" and 1-GPU
    "1.890000" per-INSTANCE rates), every
    location_ids_with_availability genuinely [];
  - vm_instant_locations.json -- the live empty page verbatim (results=[],
    total_result_count=0): emptiness IS today's claimed-availability truth.

The availability tests pin: verbatim book_stats capture, the fail-closed
fences within each VM parse, and -- the load-bearing posture -- that a VM
fetch failure or reshape is FAIL-OPEN (partial_error) and never darks the
proven bare-metal price lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.voltagepark import (
    PRESETS_URL,
    SOURCE_ID,
    URL,
    VM_LOCATIONS_URL,
    collect,
    parse_vm_locations,
    parse_vm_presets,
    parse_voltagepark,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "observatory" / "voltagepark"
FIXTURE = FIXTURE_DIR / "locations.json"
PRESETS_FIXTURE = FIXTURE_DIR / "instant_deploy_presets.json"
VM_LOCATIONS_FIXTURE = FIXTURE_DIR / "vm_instant_locations.json"


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


# ---- availability accrual: rung 1 -- reserved cluster counts --


def test_reserved_cluster_counts_recorded_verbatim(rows):
    """Both reserved-capacity fields ride in every row's extra, with the
    live nulls recorded as-is (nullable ints per the openapi.json schema) --
    presence of the KEY is the accrual, so a null must not vanish."""
    for r in rows:
        assert "reserved_gpu_count_infiniband_cluster_one" in r["extra"]
        assert "reserved_gpu_count_infiniband_cluster_two" in r["extra"]
        assert r["extra"]["reserved_gpu_count_infiniband_cluster_one"] is None
        assert r["extra"]["reserved_gpu_count_infiniband_cluster_two"] is None


def test_reserved_cluster_counts_never_gate_price():
    """A nonzero reserved count (the day the semantics proof lands) parses
    straight through -- availability metadata, never a price gate."""
    body = _body(
        lambda p: p["results"][0].__setitem__(
            "reserved_gpu_count_infiniband_cluster_one", 512
        )
    )
    rows = parse_voltagepark(body)
    assert len(rows) == 2
    assert all(
        r["extra"]["reserved_gpu_count_infiniband_cluster_one"] == 512
        for r in rows
    )
    assert all(r["price_usd_gpu_hr"] in (1.99, 2.49) for r in rows)


# ---- availability accrual: rung 2 -- vm_instant book_stats ----


def _presets_body(mutate=None) -> str:
    payload = json.loads(PRESETS_FIXTURE.read_text())
    if mutate:
        mutate(payload)
    return json.dumps(payload)


def _vm_locations_body(mutate=None) -> str:
    payload = json.loads(VM_LOCATIONS_FIXTURE.read_text())
    if mutate:
        mutate(payload)
    return json.dumps(payload)


def test_vm_presets_recorded_verbatim_with_derived_counts():
    """The presets surface lands verbatim (all 4 live presets, byte-honest
    against the fixture) plus the ONLY derived figure -- per-preset
    available_location_count -- recomputed here from the fixture itself."""
    payload = json.loads(PRESETS_FIXTURE.read_text())
    block = parse_vm_presets(PRESETS_FIXTURE.read_text())
    assert block["presets"] == payload  # verbatim, nothing dropped/reshaped
    assert block["presets_url"] == PRESETS_URL
    assert block["presets_deprecated_in_spec"] is True
    assert block["preset_available_location_counts"] == {
        p["id"]: len(p["location_ids_with_availability"]) for p in payload
    }
    # Live 2026-08-25: 4 presets, every availability array genuinely empty.
    assert len(payload) == 4
    assert set(block["preset_available_location_counts"].values()) == {0}


def test_vm_preset_rates_stay_per_instance_metadata():
    """compute_rate_hourly is per-INSTANCE ("15.120000" = the whole 8-GPU
    box; the 1-GPU preset prints "1.890000") and must ride only as labeled
    book_stats metadata -- never divided into a per-GPU price row (the
    2026-08-25 ruling: voltagepark is a seated live-panel source)."""
    block = parse_vm_presets(PRESETS_FIXTURE.read_text())
    assert block["compute_rate_hourly_basis"] == "per-instance"
    by_gpus = {
        p["resources"]["gpus"]["h100-sxm5-80gb"]["count"]: p[
            "compute_rate_hourly"
        ]
        for p in block["presets"]
    }
    assert by_gpus == {
        8: "15.120000",
        4: "7.560000",
        2: "3.780000",
        1: "1.890000",
    }


def test_vm_locations_empty_page_recorded_verbatim():
    """The live empty page IS the claimed-availability print: results=[]
    and total_result_count=0 record verbatim rather than raising."""
    block = parse_vm_locations(VM_LOCATIONS_FIXTURE.read_text())
    assert block["locations"] == []
    assert block["locations_total_result_count"] == 0
    assert block["locations_url"] == VM_LOCATIONS_URL


def test_vm_locations_rows_ride_verbatim_when_stock_lands():
    """The day rows finally land (the semantics proof), dict rows of ANY
    internal shape record verbatim -- the accrual must capture the first
    nonzero print, not fence it away."""
    row = {
        "id": "00000000-0000-4000-8000-000000000027",
        "available_presets": [
            {"id": "00000000-0000-4000-8000-000000000024", "available_vms": 3}
        ],
    }
    body = _vm_locations_body(
        lambda p: (
            p.__setitem__("results", [row]),
            p.__setitem__("total_result_count", 1),
        )
    )
    block = parse_vm_locations(body)
    assert block["locations"] == [row]
    assert block["locations_total_result_count"] == 1


def test_vm_presets_reshape_raises():
    """Fail-closed fences WITHIN the presets parse: top level not a list,
    preset not an object, id lost, duplicate ids (the count map would
    silently collapse), and a reshaped availability array."""
    with pytest.raises(RuntimeError, match="not a list"):
        parse_vm_presets(json.dumps({"presets": []}))
    with pytest.raises(RuntimeError, match="not an object"):
        parse_vm_presets(json.dumps(["h100"]))
    with pytest.raises(RuntimeError, match="no string id"):
        parse_vm_presets(_presets_body(lambda p: p[0].pop("id")))
    with pytest.raises(RuntimeError, match="duplicate preset id"):
        parse_vm_presets(
            _presets_body(lambda p: p[1].__setitem__("id", p[0]["id"]))
        )
    for bad in (3, None, "us-east", [{"id": "x"}], ["ok", 7]):
        with pytest.raises(
            RuntimeError, match="location_ids_with_availability"
        ):
            parse_vm_presets(
                _presets_body(
                    lambda p, b=bad: p[0].__setitem__(
                        "location_ids_with_availability", b
                    )
                )
            )


def test_vm_locations_reshape_raises():
    """Same envelope pins as the price page: reshaped top level, activated
    pagination, total-count mismatch, non-object rows."""
    for mutate in (
        lambda p: p.pop("results"),
        lambda p: p.pop("total_result_count"),
    ):
        with pytest.raises(RuntimeError, match="top level reshaped"):
            parse_vm_locations(_vm_locations_body(mutate))
    with pytest.raises(RuntimeError, match="has_next"):
        parse_vm_locations(
            _vm_locations_body(lambda p: p.__setitem__("has_next", True))
        )
    with pytest.raises(RuntimeError, match="total_result_count"):
        parse_vm_locations(
            _vm_locations_body(
                lambda p: p.__setitem__("total_result_count", 3)
            )
        )
    with pytest.raises(RuntimeError, match="not an object"):
        parse_vm_locations(
            _vm_locations_body(
                lambda p: (
                    p.__setitem__("results", ["stock"]),
                    p.__setitem__("total_result_count", 1),
                )
            )
        )


# ---- collect(): fail-open posture -- VM surfaces never dark the price lane


def _wire_fetch(monkeypatch, bodies):
    """Route collect()'s three fetches by URL; an Exception value raises."""
    def fake_fetch(url, headers=None, timeout=None):
        body = bodies[url]
        if isinstance(body, Exception):
            raise body
        return body

    monkeypatch.setattr(
        "gpu_index.observatory.sources.voltagepark.fetch", fake_fetch
    )


def test_collect_happy_path_records_prices_and_vm_book_stats(monkeypatch):
    _wire_fetch(
        monkeypatch,
        {
            URL: FIXTURE.read_text(),
            PRESETS_URL: PRESETS_FIXTURE.read_text(),
            VM_LOCATIONS_URL: VM_LOCATIONS_FIXTURE.read_text(),
        },
    )
    out = collect()
    assert len(out["observations"]) == 2
    assert "partial_errors" not in out
    vm = out["book_stats"]["vm_instant"]
    assert vm["semantics"].startswith("claimed-availability")
    assert len(vm["presets"]) == 4
    assert vm["locations"] == []
    assert vm["locations_total_result_count"] == 0


def test_collect_vm_fetch_failure_is_fail_open(monkeypatch):
    """Both VM fetches down: price observations still record, each surface
    leaves exactly one labeled partial_error, and no vm_instant block is
    emitted -- the proven bare-metal lane never darks."""
    _wire_fetch(
        monkeypatch,
        {
            URL: FIXTURE.read_text(),
            PRESETS_URL: RuntimeError("HTTP 503"),
            VM_LOCATIONS_URL: RuntimeError("HTTP 404"),
        },
    )
    out = collect()
    assert {r["price_usd_gpu_hr"] for r in out["observations"]} == {
        1.99,
        2.49,
    }
    assert out["partial_errors"] == [
        "vm_instant presets not recorded (fail-open, price lane "
        "unaffected): HTTP 503",
        "vm_instant locations not recorded (fail-open, price lane "
        "unaffected): HTTP 404",
    ]
    assert "book_stats" not in out


def test_collect_vm_reshape_is_fail_open_per_surface(monkeypatch):
    """A hostile reshape on ONE VM surface (presets flips to an object)
    fires that surface's fail-closed fence, lands as its partial_error, and
    the OTHER surface plus every price row still record."""
    _wire_fetch(
        monkeypatch,
        {
            URL: FIXTURE.read_text(),
            PRESETS_URL: json.dumps({"unexpected": "object"}),
            VM_LOCATIONS_URL: VM_LOCATIONS_FIXTURE.read_text(),
        },
    )
    out = collect()
    assert len(out["observations"]) == 2
    assert len(out["partial_errors"]) == 1
    assert "vm_instant presets not recorded" in out["partial_errors"][0]
    assert "not a list" in out["partial_errors"][0]
    vm = out["book_stats"]["vm_instant"]
    assert "presets" not in vm
    assert vm["locations"] == []
    assert vm["semantics"].startswith("claimed-availability")


def test_collect_price_lane_stays_fail_closed(monkeypatch):
    """The ruling cuts one way only: a darked PRICE surface still raises
    (fail-closed) even with both VM surfaces healthy."""
    _wire_fetch(
        monkeypatch,
        {
            URL: json.dumps({"results": [], "total_result_count": 0}),
            PRESETS_URL: PRESETS_FIXTURE.read_text(),
            VM_LOCATIONS_URL: VM_LOCATIONS_FIXTURE.read_text(),
        },
    )
    with pytest.raises(RuntimeError, match="zero GPU price observations"):
        collect()
