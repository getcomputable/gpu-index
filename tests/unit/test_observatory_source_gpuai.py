# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory gpuai collector -- fixture pins (live response 2026-08-22).

Fixture: real bytes from GET https://api.gpu.ai/v1/pricing?limit=200
captured 2026-08-22, trimmed to 22 of the live book's 272 offer rows with
the edge rows kept: the 8x-only B300 config, an odd 7x gpu_count, the 4x
RTX 5090 outlier row ($154.67 -- the framework flags it implausible, the
collector records it raw), rows missing the optional environment key, and
every lookalike label pair (h100_sxm/pcie/nvl, h200_sxm/nvl,
a100_40gb/80gb, rtx_6000_ada vs rtx_pro_6000). pricing_page1.json keeps
the REAL next_cursor string from the live page; pricing_page2.json is the
book tail with next_cursor null.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.gpuai import (
    SOURCE_ID,
    parse_pricing_page,
    row_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "observatory" / "gpuai"
PAGE1 = FIXTURE_DIR / "pricing_page1.json"
PAGE2 = FIXTURE_DIR / "pricing_page2.json"


@pytest.fixture(scope="module")
def pages():
    return [
        parse_pricing_page(PAGE1.read_text()),
        parse_pricing_page(PAGE2.read_text()),
    ]


@pytest.fixture(scope="module")
def rows(pages):
    return [obs for page in pages for obs in page["observations"]]


def _by_offer(rows, offer_id):
    return next(r for r in rows if r["offer_id"] == offer_id)


def _valid_row():
    """A REAL fixture row (deep-copied) as the mutation base for pin
    tests -- synthetic bodies stay one field away from live bytes."""
    return copy.deepcopy(json.loads(PAGE1.read_text())["data"][0])


def _page_body(row):
    return json.dumps({"data": [row], "next_cursor": None})


def test_source_id_matches_module():
    assert SOURCE_ID == "gpuai"


def test_per_instance_price_divides_to_per_gpu(rows):
    """price_per_hour covers the WHOLE configuration -- the 8x B300 must
    record $6.3275/GPU-hr, never $50.62."""
    b300 = _by_offer(rows, "a0a48d83d5f6")
    assert b300["sku_identifier"] == "b300"
    assert b300["price_usd_gpu_hr"] == 6.3275
    assert b300["raw_value"] == "50.62"
    assert b300["raw_unit"] == "usd_per_instance_hr"
    assert b300["gpu_count_basis"] == 8
    assert b300["tier"] == "on-demand"
    assert b300["region"] == "ca-central"
    assert b300["extra"]["capacity_class"] == "secure"
    assert b300["currency"] == "USD"
    assert "USD implied" in b300["notes"]


def test_capacity_classes_recorded_separately(rows):
    """secure and community are separate labeled observations (runpod
    pattern), including per-count configs of the same chip."""
    b200 = {
        r["offer_id"]: (
            r["extra"]["capacity_class"],
            r["gpu_count_basis"],
            r["price_usd_gpu_hr"],
        )
        for r in rows
        if r["sku_identifier"] == "b200"
    }
    assert b200 == {
        "89ce2d94cfa2": ("community", 1, 5.32),
        "e8cf1fc47222": ("secure", 1, 6.79),
        "b25408688458": ("community", 8, 7.1275),
    }


def test_recon_cross_checked_h100_sxm_row(rows):
    """2x $3.47 us-west community -- the /gpus page renders it $1.74."""
    h100 = _by_offer(rows, "ec95461f7fc5")
    assert h100["sku_identifier"] == "h100_sxm"
    assert h100["price_usd_gpu_hr"] == 1.735
    assert h100["raw_value"] == "3.47"
    assert h100["gpu_count_basis"] == 2
    assert h100["extra"]["capacity_class"] == "community"


def test_odd_gpu_count_basis(rows):
    """gpu_count is whatever the provider listed -- a 7x config divides
    by 7 (rounded to the lane's 4 decimals)."""
    r3090 = _by_offer(rows, "8132b1195c60")
    assert r3090["sku_identifier"] == "rtx_3090"
    assert r3090["gpu_count_basis"] == 7
    assert r3090["raw_value"] == "1.5"
    assert r3090["price_usd_gpu_hr"] == round(1.5 / 7, 4)


def test_outlier_row_recorded_raw_not_screened(rows):
    """The live 4x RTX 5090 at $154.67 is recorded honestly -- flagging
    it implausible is the FRAMEWORK's job, never the collector's."""
    outlier = _by_offer(rows, "79ff8ac1608a")
    assert outlier["sku_identifier"] == "rtx_5090"
    assert outlier["price_usd_gpu_hr"] == 38.6675
    assert outlier["raw_value"] == "154.67"


def test_rows_missing_optional_environment_key_still_record(rows):
    """environment is absent on 4/272 live rows -- not pinned, not fatal."""
    h200 = _by_offer(rows, "aea648d95704")
    assert h200["sku_identifier"] == "h200_sxm"
    assert h200["price_usd_gpu_hr"] == 3.99
    assert h200["extra"]["capacity_class"] == "secure"


def test_pagination_cursor_shape(pages):
    assert isinstance(pages[0]["next_cursor"], str)
    assert pages[0]["next_cursor"]
    assert pages[1]["next_cursor"] is None
    assert pages[0]["raw_row_count"] == 16
    assert pages[1]["raw_row_count"] == 6


def test_offer_identity_and_availability_metadata(rows):
    assert all(r["offer_id"] for r in rows)
    assert len({r["offer_id"] for r in rows}) == len(rows)
    assert all(r["extra"]["available"] >= 1 for r in rows)


def test_empty_capacity_class_falls_back_to_community_flag():
    """The schema documents transiently-empty capacity_class (pre-field
    cache entries); the required boolean twin is the schema-warranted
    fallback, noted in extra."""
    row = _valid_row()
    row["capacity_class"] = ""
    row["community"] = True
    obs = row_observation(row)
    assert obs["extra"]["capacity_class"] == "community"
    assert obs["extra"]["capacity_class_source"] == "community_flag_fallback"
    row["community"] = False
    assert row_observation(row)["extra"]["capacity_class"] == "secure"


def test_capacity_class_disagreeing_with_boolean_twin_refuses():
    row = _valid_row()
    row["capacity_class"] = "secure"
    row["community"] = True
    with pytest.raises(RuntimeError, match="boolean twin"):
        parse_pricing_page(_page_body(row))


def test_future_currency_field_refuses_unless_usd():
    row = _valid_row()
    row["currency"] = "EUR"
    with pytest.raises(RuntimeError, match="implied-USD warrant"):
        parse_pricing_page(_page_body(row))
    row["currency"] = "USD"
    assert parse_pricing_page(_page_body(row))["observations"]


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"offering_id": None}, "offering_id"),
        ({"tier": "reserved"}, "pinned enum"),
        ({"gpu_count": 0}, "positive integer"),
        ({"gpu_count": True}, "positive integer"),
        ({"price_per_hour": 0}, "positive number"),
        ({"region": ""}, "region"),
        ({"capacity_class": "premium"}, "capacity_class"),
    ],
)
def test_row_pin_violations_refuse_the_capture(mutation, message):
    """First-party structured API: a violating row = a reshaped surface;
    the whole capture refuses, never a silent row-skip."""
    row = _valid_row()
    row.update(mutation)
    with pytest.raises(RuntimeError, match=message):
        parse_pricing_page(_page_body(row))


def test_envelope_pin_violations_refuse_the_capture():
    with pytest.raises(RuntimeError, match="data list"):
        parse_pricing_page(json.dumps({"data": {}, "next_cursor": None}))
    with pytest.raises(RuntimeError, match="next_cursor key"):
        parse_pricing_page(json.dumps({"data": []}))
    with pytest.raises(RuntimeError, match="next_cursor"):
        parse_pricing_page(json.dumps({"data": [], "next_cursor": ""}))
    with pytest.raises(RuntimeError, match="JSON object"):
        parse_pricing_page(json.dumps([]))


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    # The lookalike pairs the catalog must keep discriminating.
    assert mapped["b300"] == "B300"
    assert mapped["b200"] == "B200"
    assert mapped["h100_sxm"] == "H100"
    assert mapped["h100_pcie"] == "H100"
    assert mapped["h100_nvl"] == "H100"
    assert mapped["h200_sxm"] == "H200"
    assert mapped["h200_nvl"] == "H200"
    assert mapped["a100_40gb"] == "A100"
    assert mapped["a100_80gb"] == "A100"
    assert mapped["rtx_6000_ada"] == "RTX_6000_ADA"
    assert mapped["rtx_pro_6000"] == "RTX_PRO_6000"
    assert mapped["rtx_pro_4500"] == "RTX_PRO_4500"
    assert mapped["l4"] == "L4"
    assert mapped["l40"] == "L40"
    assert mapped["l40s"] == "L40S"
    assert mapped["v100"] == "V100"
    # Nothing in this source's fixture should be unmapped -- if a new chip
    # appears live it records unmapped and the capture warns; this pin is
    # about the KNOWN labels staying mapped.
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known gpuai labels now unmapped: {unmapped}"
