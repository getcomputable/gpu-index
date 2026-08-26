# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory runpod collector -- fixture pins (live response 2026-08-25).

House style for per-source observatory tests: (1) parse the recorded
fixture, (2) pin exact prints for a few known rows incl. the edge cases
this source exposes (zero-price surfaces, AMD parts, consumer cards),
(3) prove the framework normalization maps this source's real labels,
(4) prove present-but-unusable values are counted, never silently dropped.

Fixture: real bytes captured live 2026-08-25 with the merged availability
query (one POST: gpuTypes incl. the three lowestPrice stock probes +
dataCenters matrix), TRIMMED for size: 48 -> 9 gpuTypes rows (the same nine
chips the pre-availability fixture pinned) and 50 -> 8 dataCenters, with
each kept DC's gpuAvailability filtered to the nine kept chips. Edge rows
kept deliberately: MI300X (null stockStatus on all three probes), B200 (lp8
stockStatus null at
a 1x-quotable chip; one listed-available DC + one UNLISTED DC), B300 (three
listed-available DCs), H100 (available/unavailable/unlisted DC mix),
RTX 3070 + A100-SXM4-40GB (secureCloud:false chips, lp1s null, absent from
the DC matrix), CA-MTL-2 (unlisted DC with empty gpuAvailability), H200
("Medium" tier, absent from the trimmed matrix).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.runpod import SOURCE_ID, parse_runpod

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "runpod" / "gputypes.json"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_runpod(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def _chip_extra(rows, sku):
    """The availability extras are chip-level: identical on every surface
    row of the chip. Assert that invariant, return the shared extra."""
    chip_rows = [r for r in rows if r["sku_identifier"] == sku]
    assert chip_rows, f"fixture row went missing: {sku}"
    keys = (
        "stock_status",
        "stock_status_8x",
        "stock_status_secure",
        "max_gpu_count",
        "offered_secure",
        "offered_community",
        "dc_availability",
        "dc_available_count",
    )
    stocks = [{k: r["extra"][k] for k in keys} for r in chip_rows]
    assert all(s == stocks[0] for s in stocks[1:])
    return stocks[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "runpod"


def test_fixture_parse_is_clean(parsed):
    """The recorded capture must parse without a single anomaly note --
    a partial_errors entry on the pinned fixture means the parse pins and
    the recorded bytes disagree."""
    assert parsed[1] == []


def test_all_four_price_surfaces_recorded(rows):
    b200 = [r for r in rows if r["sku_identifier"] == "NVIDIA B200"]
    surfaces = {(r["tier"], r["extra"]["cloud"]): r["price_usd_gpu_hr"] for r in b200}
    assert surfaces == {
        ("on-demand", "secure"): 6.79,
        ("on-demand", "community"): 5.98,
        ("spot", "secure"): 6.79,
        ("spot", "community"): 5.98,
    }


def test_zero_price_surface_is_skipped_not_recorded_as_free(rows):
    """A100-SXM4-40GB has securePrice 0 (not offered on secure cloud) --
    that must never print as a $0 observation."""
    a100_40 = [r for r in rows if r["sku_identifier"] == "NVIDIA A100-SXM4-40GB"]
    assert a100_40, "fixture row went missing"
    assert all(r["extra"]["cloud"] == "community" for r in a100_40)
    assert all(r["price_usd_gpu_hr"] == 1.0 for r in a100_40)


def test_memory_label_and_notes(rows):
    mi300x = next(
        r
        for r in rows
        if r["sku_identifier"] == "AMD Instinct MI300X OAM"
        and r["tier"] == "on-demand"
        and r["extra"]["cloud"] == "secure"
    )
    assert mi300x["memory_gb_label"] == 192
    assert mi300x["price_usd_gpu_hr"] == 2.39
    assert "MI300X" in mi300x["notes"]


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
    assert mapped["AMD Instinct MI300X OAM"] == "MI300X"
    assert mapped["NVIDIA GeForce RTX 4090"] == "RTX_4090"
    assert mapped["NVIDIA GeForce RTX 3070"] == "RTX_3070"
    assert mapped["NVIDIA H100 80GB HBM3"] == "H100"
    assert mapped["NVIDIA L40S"] == "L40S"
    # Nothing in this source's fixture should be unmapped -- if a new chip
    # appears live it records unmapped and the capture warns; this pin is
    # about the KNOWN labels staying mapped.
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known runpod labels now unmapped: {unmapped}"


# ---------------------------------------------------------------------------
# Availability extras (availability accrual) -- happy-path pins from the fixture
# ---------------------------------------------------------------------------


def test_b200_availability_extras_pinned(rows):
    """B200 live 2026-08-25: quotable at 1x ('Low' unfiltered and on secure
    cloud) while the 8x probe nulls out -- the capacity-at-size signal --
    and the trimmed DC matrix has one listed-available DC plus one UNLISTED
    DC whose row must ride verbatim but never count."""
    extra = _chip_extra(rows, "NVIDIA B200")
    assert extra == {
        "stock_status": "Low",
        "stock_status_8x": None,
        "stock_status_secure": "Low",
        "max_gpu_count": 8,
        "offered_secure": True,
        "offered_community": True,
        "dc_availability": {
            "EU-RO-1": {"available": True, "stock_status": "Low", "listed": True},
            "US-TX-6": {"available": False, "stock_status": None, "listed": False},
        },
        "dc_available_count": 1,
    }


def test_all_null_stock_probes_record_none_silently(rows, parsed):
    """MI300X live 2026-08-25 has NO public quote at any probed size/filter
    (null stockStatus inside all three probe objects) -- that is the API's
    documented answer and must record as None without a partial_errors
    entry, never read as 'delisted' (its four price surfaces still print)."""
    extra = _chip_extra(rows, "AMD Instinct MI300X OAM")
    assert extra["stock_status"] is None
    assert extra["stock_status_8x"] is None
    assert extra["stock_status_secure"] is None
    assert extra["offered_secure"] is True
    assert extra["offered_community"] is False
    # In the matrix with zero available DCs: count is 0, not None.
    assert extra["dc_availability"] == {
        "EU-RO-1": {"available": False, "stock_status": None, "listed": True}
    }
    assert extra["dc_available_count"] == 0
    assert parsed[1] == []


def test_chip_absent_from_dc_matrix_records_none_not_zero(rows):
    """The matrix is sparse: RTX 3070 has no gpuAvailability row anywhere in
    the capture, so both dc fields must be None (no signal) -- conflating
    absence with 0 would fabricate a stockout. Its probes also pin the
    community-only shape: secureCloud false, lp1s null, 8x community 'Low'."""
    extra = _chip_extra(rows, "NVIDIA GeForce RTX 3070")
    assert extra["dc_availability"] is None
    assert extra["dc_available_count"] is None
    assert extra["offered_secure"] is False
    assert extra["offered_community"] is True
    assert extra["stock_status"] == "Low"
    assert extra["stock_status_8x"] == "Low"
    assert extra["stock_status_secure"] is None


def test_dc_available_count_counts_only_listed_available(rows):
    """H100 spans the whole DC-row taxonomy in the trimmed capture: one
    listed+available (AP-IN-1), one listed+unavailable (EU-NL-1), one
    UNLISTED+unavailable (AP-IN-2) -- only the first counts. B300's three
    listed+available DCs pin the multi-DC sum."""
    h100 = _chip_extra(rows, "NVIDIA H100 80GB HBM3")
    assert h100["dc_availability"] == {
        "AP-IN-1": {"available": True, "stock_status": "Low", "listed": True},
        "AP-IN-2": {"available": False, "stock_status": None, "listed": False},
        "EU-NL-1": {"available": False, "stock_status": None, "listed": True},
    }
    assert h100["dc_available_count"] == 1
    assert h100["stock_status_8x"] == "Low"
    b300 = _chip_extra(rows, "NVIDIA B300 SXM6 AC")
    assert b300["dc_available_count"] == 3
    assert sorted(b300["dc_availability"]) == ["EU-NL-1", "EUR-IS-1", "US-WA-2"]


def test_availability_extras_recompute_from_fixture(rows):
    """Every observation's stock_status and dc_available_count must equal a
    from-scratch recomputation over the raw fixture bytes -- the collector
    adds no interpretation beyond the listed+available sum."""
    raw = json.loads(FIXTURE.read_text())
    lowest = {
        g["id"]: (g["lowestPrice"] or {}).get("stockStatus")
        for g in raw["data"]["gpuTypes"]
    }
    counts = {}
    for dc in raw["data"]["dataCenters"]:
        for entry in dc["gpuAvailability"]:
            counts.setdefault(entry["gpuTypeId"], 0)
            if dc["listed"] is True and entry["available"] is True:
                counts[entry["gpuTypeId"]] += 1
    assert counts == {
        "NVIDIA H100 80GB HBM3": 1,
        "NVIDIA B300 SXM6 AC": 3,
        "NVIDIA L40S": 0,
        "NVIDIA GeForce RTX 4090": 2,
        "NVIDIA B200": 1,
        "AMD Instinct MI300X OAM": 0,
    }
    for r in rows:
        sku = r["sku_identifier"]
        assert r["extra"]["stock_status"] == lowest[sku]
        assert r["extra"]["dc_available_count"] == counts.get(sku)


# ---------------------------------------------------------------------------
# Fail-open fences: stock-subtree anomalies are counted, price rows NEVER
# thin. (Availability rides in the same POST as the prices, so there is no
# separate fetch to kill -- the reshape variants below ARE the failure mode.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reshape", ["missing", "string", "object"])
def test_missing_or_misshapen_datacenters_is_failopen(rows, reshape):
    """Losing the whole dataCenters block (RunPod gating the field, a rename,
    a retype) must cost ZERO price observations: one loud partial_error,
    dc fields None everywhere, every price row still lands."""
    raw = json.loads(FIXTURE.read_text())
    if reshape == "missing":
        del raw["data"]["dataCenters"]
    elif reshape == "string":
        raw["data"]["dataCenters"] = "gone"
    else:
        raw["data"]["dataCenters"] = {"EU-RO-1": {}}
    got, errors = parse_runpod(json.dumps(raw))
    assert len(got) == len(rows)
    assert len(errors) == 1
    assert "dataCenters block missing/misshapen" in errors[0]
    assert "price rows unaffected" in errors[0]
    for r in got:
        assert r["extra"]["dc_availability"] is None
        assert r["extra"]["dc_available_count"] is None
    # The other stock probes are untouched by a dataCenters failure.
    assert any(r["extra"]["stock_status"] == "Low" for r in got)


def test_misshapen_lowest_price_is_counted_price_unaffected(rows):
    """Hostile variant 1: lowestPrice retyped to a bare string. The probe
    records None + one partial_error naming the chip; all four B200 price
    surfaces still print at the fixture prices."""
    raw = json.loads(FIXTURE.read_text())
    for g in raw["data"]["gpuTypes"]:
        if g["id"] == "NVIDIA B200":
            g["lowestPrice"] = "Low"
    got, errors = parse_runpod(json.dumps(raw))
    assert len(got) == len(rows)
    assert len(errors) == 1
    assert "NVIDIA B200" in errors[0]
    assert "lowestPrice misshapen" in errors[0]
    b200 = [r for r in got if r["sku_identifier"] == "NVIDIA B200"]
    assert sorted(r["price_usd_gpu_hr"] for r in b200) == [5.98, 5.98, 6.79, 6.79]
    assert all(r["extra"]["stock_status"] is None for r in b200)
    # The sibling probes on the same chip are independent and keep printing.
    assert all(r["extra"]["stock_status_secure"] == "Low" for r in b200)


def test_retyped_stock_status_is_counted_not_recorded(rows):
    """Hostile variant 2: stockStatus retyped to a number inside an intact
    probe object. Opaque-string discipline means we record verbatim STRINGS
    only -- a non-string lands as None + partial_error, never as junk."""
    raw = json.loads(FIXTURE.read_text())
    for g in raw["data"]["gpuTypes"]:
        if g["id"] == "NVIDIA B200":
            g["lp1s"] = {"stockStatus": 3}
    got, errors = parse_runpod(json.dumps(raw))
    assert len(got) == len(rows)
    assert len(errors) == 1
    assert "lp1s.stockStatus retyped" in errors[0]
    b200 = [r for r in got if r["sku_identifier"] == "NVIDIA B200"]
    assert all(r["extra"]["stock_status_secure"] is None for r in b200)
    assert all(r["extra"]["stock_status"] == "Low" for r in b200)


def test_missing_probe_alias_is_counted():
    """GraphQL echoes every requested field (null when unanswerable), so an
    ABSENT probe alias is a reshape (alias dropped server-side) and must be
    counted -- unlike a null probe, which is a documented answer."""
    raw = json.loads(FIXTURE.read_text())
    for g in raw["data"]["gpuTypes"]:
        if g["id"] == "NVIDIA B200":
            del g["lp8"]
    got, errors = parse_runpod(json.dumps(raw))
    assert len(errors) == 1
    assert "lp8 probe missing" in errors[0]
    b200 = [r for r in got if r["sku_identifier"] == "NVIDIA B200"]
    assert len(b200) == 4
    assert all(r["extra"]["stock_status_8x"] is None for r in b200)


def test_retyped_gputype_scalars_record_none_and_count():
    raw = json.loads(FIXTURE.read_text())
    for g in raw["data"]["gpuTypes"]:
        if g["id"] == "NVIDIA B200":
            g["maxGpuCount"] = "8"
            g["secureCloud"] = "yes"
    got, errors = parse_runpod(json.dumps(raw))
    assert sorted(e.split(": ")[1].split(" ")[0] for e in errors) == [
        "maxGpuCount",
        "secureCloud",
    ]
    b200 = [r for r in got if r["sku_identifier"] == "NVIDIA B200"]
    assert len(b200) == 4
    assert all(r["extra"]["max_gpu_count"] is None for r in b200)
    assert all(r["extra"]["offered_secure"] is None for r in b200)
    assert all(r["extra"]["offered_community"] is True for r in b200)


def test_malformed_dc_entries_are_skipped_not_fatal(rows):
    """One bad DC (or one bad gpuAvailability row) must not erase the rest
    of the matrix: each anomaly is counted, the intact rows still fold."""
    raw = json.loads(FIXTURE.read_text())
    raw["data"]["dataCenters"].extend(
        [
            {"name": "no id", "listed": True, "gpuAvailability": []},
            {"id": "XX-BAD-1", "listed": True, "gpuAvailability": "nope"},
            {
                "id": "XX-BAD-2",
                "listed": True,
                "gpuAvailability": [{"available": True, "stockStatus": "Low"}],
            },
        ]
    )
    got, errors = parse_runpod(json.dumps(raw))
    assert len(got) == len(rows)
    assert errors == [
        "dataCenters entry without id -- skipped",
        "XX-BAD-1: gpuAvailability misshapen (str) -- skipped",
        "XX-BAD-2: gpuAvailability entry without gpuTypeId -- skipped",
    ]
    b200 = next(r for r in got if r["sku_identifier"] == "NVIDIA B200")
    assert b200["extra"]["dc_available_count"] == 1


def test_dc_available_count_guards_retyped_flags(rows):
    """available/stockStatus in the matrix are recorded VERBATIM (opaque by
    ruling), but the derived scalar guards with `is True` -- a truthy
    retyped flag ("true", 1) must never inflate dc_available_count."""
    raw = json.loads(FIXTURE.read_text())
    for dc in raw["data"]["dataCenters"]:
        for entry in dc["gpuAvailability"]:
            if entry["available"] is True:
                entry["available"] = "true"
    got, errors = parse_runpod(json.dumps(raw))
    assert errors == []
    assert len(got) == len(rows)
    b300 = next(r for r in got if r["sku_identifier"] == "NVIDIA B300 SXM6 AC")
    assert b300["extra"]["dc_available_count"] == 0
    assert (
        b300["extra"]["dc_availability"]["EU-NL-1"]["available"] == "true"
    )  # verbatim


# ---------------------------------------------------------------------------
# Price-surface fences (pre-availability behavior, unchanged)
# ---------------------------------------------------------------------------


def test_parse_counts_unlabeled_rows_in_partial_errors():
    body = json.dumps(
        {
            "data": {
                "gpuTypes": [
                    {"id": "", "displayName": "?", "securePrice": 1.0}
                ],
                "dataCenters": [],
            }
        }
    )
    rows, errors = parse_runpod(body)
    assert rows == []
    assert len(errors) == 1
    assert "without id" in errors[0]


def test_retyped_price_field_is_counted_not_silently_dropped():
    """If one surface's price field is re-typed (schema drift), the other
    surfaces must keep printing AND the anomaly must land in
    partial_errors -- a silently thinner record is a permanent capture gap."""
    fixture = json.loads(FIXTURE.read_text())
    for g in fixture["data"]["gpuTypes"]:
        if isinstance(g.get("securePrice"), (int, float)):
            g["securePrice"] = str(g["securePrice"])
    rows, errors = parse_runpod(json.dumps(fixture))
    assert not any(
        r["tier"] == "on-demand" and r["extra"]["cloud"] == "secure" for r in rows
    )
    # every other surface still prints
    assert any(
        r["tier"] == "on-demand" and r["extra"]["cloud"] == "community"
        for r in rows
    )
    stringified = sum(
        1
        for g in fixture["data"]["gpuTypes"]
        if isinstance(g.get("securePrice"), str)
    )
    assert len(errors) == stringified
    assert all("securePrice" in e for e in errors)


@pytest.mark.parametrize(
    "bad_price", ["6.79", True, float("nan"), float("inf"), -6.79]
)
def test_unusable_price_shapes_never_become_observations(bad_price):
    body = json.dumps(
        {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA B200",
                        "displayName": "B200",
                        "memoryInGb": 180,
                        "securePrice": bad_price,
                        "communityPrice": 5.98,
                        "lowestPrice": None,
                        "lp8": None,
                        "lp1s": None,
                    }
                ],
                "dataCenters": [],
            }
        }
    )
    rows, errors = parse_runpod(body)
    assert [r["price_usd_gpu_hr"] for r in rows] == [5.98]
    assert all(math.isfinite(r["price_usd_gpu_hr"]) for r in rows)
    assert len(errors) == 1 and "securePrice" in errors[0]


def test_null_and_zero_prices_skip_silently():
    """Null/zero are the API's documented 'not offered' -- they must NOT
    pollute partial_errors (16 zero fields exist on the live surface
    today)."""
    body = json.dumps(
        {
            "data": {
                "gpuTypes": [
                    {
                        "id": "NVIDIA B200",
                        "displayName": "B200",
                        "securePrice": 0,
                        "communityPrice": None,
                        "secureSpotPrice": 6.79,
                        "lowestPrice": None,
                        "lp8": None,
                        "lp1s": None,
                    }
                ],
                "dataCenters": [],
            }
        }
    )
    rows, errors = parse_runpod(body)
    assert errors == []
    assert [(r["tier"], r["extra"]["cloud"]) for r in rows] == [("spot", "secure")]
