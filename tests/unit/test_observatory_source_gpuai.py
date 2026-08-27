# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory gpuai collector -- fixture pins (live response 2026-08-22).

Fixture: real bytes from GET https://api.gpu.ai/v1/pricing?limit=200
captured 2026-08-22, trimmed to 22 of the live book's 272 offer rows with
the edge rows kept: the 8x-only B300 config, an odd 7x gpu_count, the 4x
RTX 5090 outlier row ($154.67 -- the framework flags it implausible, the
collector records it raw), rows missing the optional environment key, and
every lookalike label pair (h100_sxm/pcie/nvl, h200_sxm/nvl,
a100_40gb/80gb, rtx_6000_ada vs rtx_pro_6000). The per-offer
``offering_id`` values are SYNTHETIC -- sequential 12-hex stand-ins
(aa0000000001...) that preserve the stable-identity and cross-page
uniqueness relationships the pins depend on; prices, availability counts,
field shapes and the next_cursor are as recorded. pricing_page1.json keeps
the REAL next_cursor string from the live page; pricing_page2.json is the
book tail with next_cursor null, plus three real sold-out rows (available
0: 1x a100_80gb, 1x a40, 8x b200) appended verbatim from the live
include_unavailable=true book on 2026-08-25.
"""

from __future__ import annotations

import copy
import json
import urllib.parse
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.gpuai import (
    SOURCE_ID,
    collect,
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
    b300 = _by_offer(rows, "aa0000000006")
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
        "aa0000000003": ("community", 1, 5.32),
        "aa0000000004": ("secure", 1, 6.79),
        "aa0000000005": ("community", 8, 7.1275),
        # Sold out (available 0) -- records beside the in-stock offers.
        "aa0000000025": ("secure", 8, 5.4),
    }


def test_recon_cross_checked_h100_sxm_row(rows):
    """2x $3.47 us-west community -- the /gpus page renders it $1.74."""
    h100 = _by_offer(rows, "aa0000000009")
    assert h100["sku_identifier"] == "h100_sxm"
    assert h100["price_usd_gpu_hr"] == 1.735
    assert h100["raw_value"] == "3.47"
    assert h100["gpu_count_basis"] == 2
    assert h100["extra"]["capacity_class"] == "community"


def test_odd_gpu_count_basis(rows):
    """gpu_count is whatever the provider listed -- a 7x config divides
    by 7 (rounded to the lane's 4 decimals)."""
    r3090 = _by_offer(rows, "aa0000000015")
    assert r3090["sku_identifier"] == "rtx_3090"
    assert r3090["gpu_count_basis"] == 7
    assert r3090["raw_value"] == "1.5"
    assert r3090["price_usd_gpu_hr"] == round(1.5 / 7, 4)


def test_outlier_row_recorded_raw_not_screened(rows):
    """The live 4x RTX 5090 at $154.67 is recorded honestly -- flagging
    it implausible is the FRAMEWORK's job, never the collector's."""
    outlier = _by_offer(rows, "aa0000000017")
    assert outlier["sku_identifier"] == "rtx_5090"
    assert outlier["price_usd_gpu_hr"] == 38.6675
    assert outlier["raw_value"] == "154.67"


def test_rows_missing_optional_environment_key_still_record(rows):
    """environment is absent on 4/272 live rows -- not pinned, not fatal."""
    h200 = _by_offer(rows, "aa0000000011")
    assert h200["sku_identifier"] == "h200_sxm"
    assert h200["price_usd_gpu_hr"] == 3.99
    assert h200["extra"]["capacity_class"] == "secure"


def test_pagination_cursor_shape(pages):
    assert isinstance(pages[0]["next_cursor"], str)
    assert pages[0]["next_cursor"]
    assert pages[1]["next_cursor"] is None
    assert pages[0]["raw_row_count"] == 16
    assert pages[1]["raw_row_count"] == 9


def test_offer_identity_and_availability_metadata(rows):
    assert all(r["offer_id"] for r in rows)
    assert len({r["offer_id"] for r in rows}) == len(rows)
    assert all(r["extra"]["available"] >= 0 for r in rows)
    sold_out = {r["offer_id"] for r in rows if r["extra"]["available"] == 0}
    assert sold_out == {"aa0000000023", "aa0000000024", "aa0000000025"}


def test_sold_out_row_records_as_listed_price():
    """An available==0 row is an observation of the book -- full shape,
    stable offering_id, price recorded (a LISTED price; consumers fence on
    extra.available > 0 before any offered-price statistic)."""
    page2 = json.loads(PAGE2.read_text())
    row = next(
        r for r in page2["data"] if r["offering_id"] == "aa0000000023"
    )
    obs = row_observation(row)
    assert obs["sku_identifier"] == "a100_80gb"
    assert obs["extra"]["available"] == 0
    assert obs["price_usd_gpu_hr"] == 1.07
    assert obs["raw_value"] == "1.07"
    assert obs["offer_id"] == "aa0000000023"


def test_collect_requests_include_unavailable_and_counts_stockouts(
    monkeypatch,
):
    """The one-flag posture change: collect() must ask for the sold-out
    rows on EVERY page fetch, cursor-bearing pages included -- silently
    losing the param on later pages is exactly the unpinnable default-book
    revert the module docstring warns about. book_stats carries the
    silent-revert tripwire count."""
    bodies = [PAGE1.read_text(), PAGE2.read_text()]
    urls = []

    def fake_fetch(url, timeout=None):
        urls.append(url)
        return bodies[len(urls) - 1]

    monkeypatch.setattr("gpu_index.observatory.sources.gpuai.fetch", fake_fetch)
    res = collect()
    base = "https://api.gpu.ai/v1/pricing?limit=200&include_unavailable=true"
    cursor = json.loads(PAGE1.read_text())["next_cursor"]
    assert urls == [
        base,
        base + "&cursor=" + urllib.parse.quote(cursor, safe=""),
    ]
    assert res["book_stats"]["pages_fetched"] == 2
    assert res["book_stats"]["rows_recorded"] == 25
    assert res["book_stats"]["rows_zero_available"] == 3


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
        ({"available": -1}, "non-negative integer"),
        ({"available": None}, "non-negative integer"),
        ({"available": True}, "non-negative integer"),
        ({"available": 2.0}, "non-negative integer"),
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
    # H-series variant split (design section 7): PCIe/NVL slugs land on
    # their variant skus; SXM stays on the generic entries by design.
    assert mapped["h100_sxm"] == "H100"
    assert mapped["h100_pcie"] == "H100_PCIE"
    assert mapped["h100_nvl"] == "H100_NVL"
    assert mapped["h200_sxm"] == "H200"
    assert mapped["h200_nvl"] == "H200_NVL"
    assert mapped["a100_40gb"] == "A100"
    assert mapped["a100_80gb"] == "A100"
    assert mapped["rtx_6000_ada"] == "RTX_6000_ADA"
    assert mapped["rtx_pro_6000"] == "RTX_PRO_6000"
    assert mapped["rtx_pro_4500"] == "RTX_PRO_4500"
    assert mapped["l4"] == "L4"
    assert mapped["l40"] == "L40"
    assert mapped["l40s"] == "L40S"
    assert mapped["v100"] == "V100"
    assert mapped["a40"] == "A40"
    # Nothing in this source's fixture should be unmapped -- if a new chip
    # appears live it records unmapped and the capture warns; this pin is
    # about the KNOWN labels staying mapped.
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known gpuai labels now unmapped: {unmapped}"
