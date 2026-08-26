# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory latitude collector -- fixture pins (live page 2026-08-22).

House style per the runpod exemplar: (1) parse the recorded fixture (a real
contiguous slice of the live /pricing flight data holding all 17 bare-metal
plan blobs -- 3 GPU, 14 CPU-only), (2) pin exact prints for known rows incl.
this source's edge cases (null/0-priced unavailable region, native BRL
prints, CPU plans with an empty gpu object, per-node -> per-GPU
normalization at counts 8 and 1), (3) prove the framework normalization
maps this source's real labels, (4) prove the fail-closed identity pins
raise on reshapes instead of guessing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.latitude import (
    HOURS_PER_MONTH,
    SOURCE_ID,
    parse_latitude,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "latitude"
    / "pricing_excerpt.html"
)


def _escaped(plain: str) -> str:
    """Flight-data encoding: the blob sits inside a JS string, so every
    quote is backslash-escaped."""
    return plain.replace('"', '\\"')


@pytest.fixture(scope="module")
def parsed():
    return parse_latitude(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "latitude"


def test_gpu_plans_only_no_partial_errors(parsed):
    rows, partial_errors = parsed
    assert partial_errors == []
    # 3 GPU plans; the 14 CPU-only blobs (empty gpu object) contribute
    # nothing. 6 priced (plan, region) pairs x 2 tiers x 2 currencies.
    assert len(rows) == 24
    assert {r["sku_identifier"] for r in rows} == {
        "NVIDIA HGX B300",
        "NVIDIA H100 80GB",
        "NVIDIA RTX PRO 6000",
    }
    assert all(r["extra"]["plan"].startswith(("g3-", "g4-")) for r in rows)


def test_b300_us_hourly_and_monthly_pins(rows):
    b300_usd = [
        r
        for r in rows
        if r["extra"]["plan"] == "g4-b300-large" and r["currency"] == "USD"
    ]
    by_tier = {r["tier"]: r for r in b300_usd}
    hourly = by_tier["on-demand"]
    assert hourly["price_usd_gpu_hr"] == 16.0
    assert hourly["raw_value"] == "128"
    assert hourly["raw_unit"] == "usd_per_node_hr"
    assert hourly["gpu_count_basis"] == 8
    # price x basis reproduces the raw per-node figure exactly.
    assert hourly["price_usd_gpu_hr"] * hourly["gpu_count_basis"] == 128.0
    assert hourly["region"] == "United States"
    monthly = by_tier["monthly-commit"]
    assert monthly["price_usd_gpu_hr"] == 8.0  # 46720 / 730 / 8
    assert monthly["raw_value"] == "46720"
    assert monthly["raw_unit"] == "usd_per_node_month"
    assert str(HOURS_PER_MONTH) in monthly["notes"]
    # A priced region can still be out of stock -- list price is recorded,
    # stock is metadata.
    assert hourly["extra"]["stock_level"] == "unavailable"
    assert hourly["extra"]["interconnect"] == "800Gbps Dual Plane RoCE"
    assert hourly["memory_gb_label"] == 288
    # Per-site availability, verbatim: unavailable region = zero sites in
    # stock, empty instant-deploy list -- co-occurrence recorded raw,
    # never derived.
    assert hourly["extra"]["locations_available"] == [
        "DAL", "LAX", "NYC", "CHI", "ASH", "MIA2", "LAX2", "SJC2", "ASH2",
    ]
    assert hourly["extra"]["locations_in_stock"] == []
    assert hourly["extra"]["deploys_instantly"] == []


def test_per_site_availability_recorded_verbatim(rows):
    """Site-level beats region-level: a priced, stocked region can hold
    inventory at only SOME of its sites -- the fixture's rtx6kpro rows are
    live SUBSET cases."""
    rtx = [
        r
        for r in rows
        if r["extra"]["plan"] == "g4-rtx6kpro-large"
        and r["currency"] == "USD"
        and r["tier"] == "on-demand"
    ]
    by_region = {r["region"]: r["extra"] for r in rtx}
    au = by_region["Australia"]
    assert au["stock_level"] == "low"
    assert au["locations_available"] == ["SYD", "SYD2"]
    assert au["locations_in_stock"] == ["SYD2"]  # 1 of 2 sites
    assert au["deploys_instantly"] == ["ubuntu24_ml_in_a_box"]
    us = by_region["United States"]
    assert len(us["locations_available"]) == 9
    assert us["locations_in_stock"] == ["CHI"]  # 1 of 9 sites


def test_missing_locations_map_is_noted_never_fatal():
    """The per-site fields are availability METADATA: a region row without
    a readable locations map records its priced observations anyway, with
    the miss noted and the keys absent -- never fabricated."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":[{"name":"United States","deploys_instantly":[],'
        '"stock_level":"low","pricing":{"USD":{"hour":8,"month":2920,'
        '"year":29200}}}],"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    assert len(rows) == 2  # hourly + monthly still record
    assert partial_errors == [
        "g9-test-large/United States: region row missing a readable "
        "locations map -- site availability not recorded"
    ]
    for r in rows:
        assert "locations_available" not in r["extra"]
        assert "locations_in_stock" not in r["extra"]
        assert r["extra"]["deploys_instantly"] == []


def test_h100_single_gpu_pins(rows):
    h100 = [
        r
        for r in rows
        if r["extra"]["plan"] == "g3-h100-small" and r["currency"] == "USD"
    ]
    by_tier = {r["tier"]: r for r in h100}
    assert by_tier["on-demand"]["price_usd_gpu_hr"] == 3.37
    assert by_tier["on-demand"]["gpu_count_basis"] == 1
    assert by_tier["on-demand"]["raw_value"] == "3.37"
    # 1230 / 730 / 1, rounded to 4dp by observation()
    assert by_tier["monthly-commit"]["price_usd_gpu_hr"] == 1.6849
    assert by_tier["monthly-commit"]["raw_value"] == "1230"
    assert by_tier["on-demand"]["memory_gb_label"] == 80


def test_brl_recorded_natively_never_as_usd(rows):
    brl = [r for r in rows if r["currency"] == "BRL"]
    assert len(brl) == 12  # every priced (plan, region, tier) prints BRL too
    assert all(r["price_usd_gpu_hr"] is None for r in brl)
    b300 = next(
        r
        for r in brl
        if r["extra"]["plan"] == "g4-b300-large" and r["tier"] == "on-demand"
    )
    assert b300["price_native_per_gpu_hr"] == 80.0  # 640 / 8
    assert b300["raw_value"] == "640"
    assert b300["raw_unit"] == "brl_per_node_hr"


def test_unpriced_region_skipped_not_zero(rows):
    """g4-rtx6kpro-large United Kingdom publishes null USD / 0 BRL (region
    unavailable) -- that must never print as a $0 observation."""
    rtx = [r for r in rows if r["extra"]["plan"] == "g4-rtx6kpro-large"]
    assert {r["region"] for r in rtx} == {
        "United States",
        "Australia",
        "Japan",
        "Netherlands",
    }
    assert all(r["price_native_per_gpu_hr"] > 0 for r in rows)
    # Per-region prices genuinely differ (not one price fanned out).
    au = next(
        r
        for r in rtx
        if r["region"] == "Australia"
        and r["currency"] == "USD"
        and r["tier"] == "on-demand"
    )
    assert au["price_usd_gpu_hr"] == 8.25  # 66 / 8
    # interconnect is null on this plan -- key stays absent, never fabricated
    assert all("interconnect" not in r["extra"] for r in rtx)


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA HGX B300"] == "B300"
    assert mapped["NVIDIA H100 80GB"] == "H100"
    # Lookalike trap: 'NVIDIA RTX PRO 6000' must hit the Blackwell PRO entry,
    # never RTX_6000 (Quadro-era) or RTX_6000_ADA -- first-match-wins order.
    assert mapped["NVIDIA RTX PRO 6000"] == "RTX_PRO_6000"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known latitude labels now unmapped: {unmapped}"


def test_parse_raises_on_no_plan_blobs():
    with pytest.raises(RuntimeError, match="no bare-metal plan blobs"):
        parse_latitude("<html>nothing that looks like flight data</html>")


def test_parse_raises_on_reshaped_gpu_spec():
    """A non-empty gpu object the strict regex cannot read (reordered
    fields) must RAISE, not silently drop a GPU plan from the record."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"type":"NVIDIA X100","count":8,"vram_per_gpu":80}},'
        '"regions":[{"name":"United States","deploys_instantly":[],'
        '"stock_level":"low","pricing":{"USD":{"hour":8,"month":2920,'
        '"year":29200}}}],"available_operating_systems":[]'
    )
    with pytest.raises(RuntimeError, match="field order/shape changed"):
        parse_latitude(_escaped(plain))


def test_parse_raises_on_unreadable_regions():
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":[{"label":"renamed-field","stock_level":"low"}],'
        '"available_operating_systems":[]'
    )
    with pytest.raises(RuntimeError, match="region shape changed"):
        parse_latitude(_escaped(plain))


def test_reshaped_currency_block_noted_not_silently_dropped():
    """A currency block that no longer starts with "hour" (leading field
    inserted / triple reordered) must surface in partial_errors -- it must
    never just vanish while the source stays 'ok'."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":[{"name":"United States","deploys_instantly":[],'
        '"locations":{"available":["DAL"],"in_stock":[]},'
        '"stock_level":"low","pricing":{'
        '"USD":{"setup":10,"hour":8,"month":2920,"year":29200},'
        '"BRL":{"hour":40,"month":14600,"year":146000}}}],'
        '"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    # BRL still records; the reshaped USD block is a LOUD skip.
    assert [r["currency"] for r in rows] == ["BRL", "BRL"]
    assert partial_errors == [
        "g9-test-large/United States: 1 of 2 currency blocks unreadable -- "
        "skipped those"
    ]


def test_unreadable_region_row_cannot_donate_prices_to_neighbor():
    """A reshaped region row the start regex cannot read is absorbed into
    the previous region's segment -- its pricing must NOT be attributed to
    that region's name, and the drop must be noted."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":['
        '{"name":"United States","deploys_instantly":[],'
        '"locations":{"available":["DAL","CHI"],"in_stock":["CHI"]},'
        '"stock_level":"low","pricing":{'
        '"USD":{"hour":8,"month":2920,"year":29200}}},'
        '{"label":"Ghost","deploys_instantly":["evil_os"],'
        '"locations":{"available":["EVIL"],"in_stock":["EVIL"]},'
        '"stock_level":"low","pricing":{'
        '"EUR":{"hour":99,"month":999,"year":9999}}}],'
        '"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    # The ghost row's EUR prices never print under "United States".
    assert {r["currency"] for r in rows} == {"USD"}
    assert all(r["region"] == "United States" for r in rows)
    # ...and the ghost's site lists never donate either -- the per-site
    # search is bounded to the segment head before this row's own
    # stock_level.
    assert all(
        r["extra"]["locations_in_stock"] == ["CHI"]
        and r["extra"]["locations_available"] == ["DAL", "CHI"]
        and r["extra"]["deploys_instantly"] == []
        for r in rows
    )
    assert partial_errors == [
        "g9-test-large: 2 stock_level markers vs 1 readable region rows -- "
        "region shape partially changed, unreadable rows skipped"
    ]


def test_own_stock_reshaped_skips_row_never_borrows_neighbor_stock():
    """When this row's OWN stock_level value is unreadable, the row is a
    loud skip -- the absorbed neighbor's readable stock_level and site
    lists must never print under this region's name."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":['
        '{"name":"United States","deploys_instantly":["own_os"],'
        '"locations":{"available":["DAL"],"in_stock":["DAL"]},'
        '"stock_level":null,"pricing":{'
        '"USD":{"hour":8,"month":2920,"year":29200}}},'
        '{"label":"Ghost","deploys_instantly":["evil_os"],'
        '"locations":{"available":["EVIL"],"in_stock":["EVIL"]},'
        '"stock_level":"high","pricing":{'
        '"EUR":{"hour":99,"month":999,"year":9999}}}],'
        '"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    assert rows == []
    assert partial_errors == [
        "g9-test-large: 2 stock_level markers vs 1 readable region rows -- "
        "region shape partially changed, unreadable rows skipped",
        "g9-test-large/United States: region row missing stock_level -- "
        "skipped",
    ]


def test_missing_own_stock_key_skips_row_never_borrows_neighbor():
    """When this row has NO stock_level at all and an absorbed neighbor
    has one, the neighbor's marker sits past this row's own pricing map --
    it must read as 'missing stock_level', never as this row's stock, and
    the neighbor's site lists must not donate."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":['
        '{"name":"United States","deploys_instantly":[],'
        '"pricing":{"USD":{"hour":8,"month":2920,"year":29200}}},'
        '{"label":"Ghost","deploys_instantly":["evil_os"],'
        '"locations":{"available":["EVIL"],"in_stock":["EVIL"]},'
        '"stock_level":"low","pricing":{'
        '"EUR":{"hour":99,"month":999,"year":9999}}}],'
        '"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    assert rows == []
    assert partial_errors == [
        "g9-test-large/United States: region row missing stock_level -- "
        "skipped"
    ]


def test_non_flat_list_items_are_a_miss_never_fabricated():
    """A list whose items are no longer flat strings must MISS whole (a
    partial_error), never fabricate site codes from object keys or an
    affirmative empty list from numeric items."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":[{"name":"United States","deploys_instantly":[1,2],'
        '"locations":{"available":[{"code":"MIA","live":true}],'
        '"in_stock":[1,2]},'
        '"stock_level":"low","pricing":{"USD":{"hour":8,"month":2920,'
        '"year":29200}}}],"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    assert len(rows) == 2  # priced observations still record
    assert partial_errors == [
        "g9-test-large/United States: region row missing a readable "
        "locations map -- site availability not recorded",
        "g9-test-large/United States: region row missing a readable "
        "deploys_instantly list -- instant-deploy flag not recorded",
    ]
    for r in rows:
        assert "locations_available" not in r["extra"]
        assert "locations_in_stock" not in r["extra"]
        assert "deploys_instantly" not in r["extra"]


def test_null_deploys_is_noted_while_locations_still_record():
    """The two per-site fields miss independently: deploys_instantly null
    notes its own partial_error while the locations lists still record."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":8,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":[{"name":"United States","deploys_instantly":null,'
        '"locations":{"available":["DAL","CHI"],"in_stock":["DAL"]},'
        '"stock_level":"low","pricing":{"USD":{"hour":8,"month":2920,'
        '"year":29200}}}],"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    assert len(rows) == 2
    assert partial_errors == [
        "g9-test-large/United States: region row missing a readable "
        "deploys_instantly list -- instant-deploy flag not recorded"
    ]
    for r in rows:
        assert r["extra"]["locations_available"] == ["DAL", "CHI"]
        assert r["extra"]["locations_in_stock"] == ["DAL"]
        assert "deploys_instantly" not in r["extra"]


def test_gpu_count_zero_is_skipped_with_note():
    """count 0 makes per-GPU normalization impossible -- skip + note, never
    a divide-by-zero or a guessed basis."""
    plain = (
        '"slug":"g9-test-large","name":"g9.test.large","specs":{'
        '"gpu":{"count":0,"type":"NVIDIA X100","vram_per_gpu":80,'
        '"interconnect":null}},'
        '"regions":[{"name":"United States","deploys_instantly":[],'
        '"stock_level":"low","pricing":{"USD":{"hour":8,"month":2920,'
        '"year":29200}}}],"available_operating_systems":[]'
    )
    rows, partial_errors = parse_latitude(_escaped(plain))
    assert rows == []
    assert partial_errors == [
        "g9-test-large: gpu count 0 -- per-GPU normalization impossible, "
        "plan skipped"
    ]
