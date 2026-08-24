# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory runpod collector -- fixture pins (live response 2026-08-22).

House style for per-source observatory tests: (1) parse the recorded
fixture, (2) pin exact prints for a few known rows incl. the edge cases
this source exposes (zero-price surfaces, AMD parts, consumer cards),
(3) prove the framework normalization maps this source's real labels,
(4) prove present-but-unusable values are counted, never silently dropped.
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


def test_parse_counts_unlabeled_rows_in_partial_errors():
    body = json.dumps(
        {"data": {"gpuTypes": [{"id": "", "displayName": "?", "securePrice": 1.0}]}}
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
                    }
                ]
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
                    }
                ]
            }
        }
    )
    rows, errors = parse_runpod(body)
    assert errors == []
    assert [(r["tier"], r["extra"]["cloud"]) for r in rows] == [("spot", "secure")]
