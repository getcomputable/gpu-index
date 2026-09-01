# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory nebius collector - fixture pins (live page 2026-08-22).

Fixture is a real contiguous excerpt of nebius.com/prices captured
2026-08-22: the escaped-JSON GPU table (with its unpriced GB300/GB200 NVL72
'Contact us' rows and the L40S 'from $' floor-price rows) PLUS the page's
lookalike non-GPU tables (CPU-only 'Price per hour', Storage, Other
services) that the exact-header pin must never match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.nebius import SOURCE_ID, parse_nebius

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "nebius" / "prices.html"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_nebius(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def skipped(parsed):
    return parsed[1]


def test_source_id_matches_module():
    assert SOURCE_ID == "nebius"


def test_both_price_columns_pinned_exact(rows):
    by_key = {(r["sku_identifier"], r["tier"]): r for r in rows}
    b300_od = by_key[("NVIDIA HGX B300", "on-demand")]
    assert b300_od["price_usd_gpu_hr"] == 7.85
    assert b300_od["raw_value"] == "$7.85"
    assert b300_od["gpu_count_basis"] == 1
    assert b300_od["currency"] == "USD"
    assert by_key[("NVIDIA HGX B300", "preemptible")]["price_usd_gpu_hr"] == 4.30
    assert by_key[("NVIDIA HGX H100", "on-demand")]["price_usd_gpu_hr"] == 3.85
    assert by_key[("NVIDIA HGX H100", "preemptible")]["price_usd_gpu_hr"] == 2.15
    assert by_key[("NVIDIA RTX PRO 6000", "on-demand")]["price_usd_gpu_hr"] == 1.80


def test_every_priced_row_records_both_tiers(rows):
    items = {r["sku_identifier"] for r in rows}
    assert items == {
        "NVIDIA HGX B300",
        "NVIDIA HGX B200",
        "NVIDIA HGX H200",
        "NVIDIA HGX H100",
        "NVIDIA RTX PRO 6000",
        "NVIDIA L40S with Intel CPU",
        "NVIDIA L40S with AMD CPU",
    }
    assert len(rows) == len(items) * 2  # on-demand + preemptible each
    for item in items:
        tiers = {r["tier"] for r in rows if r["sku_identifier"] == item}
        assert tiers == {"on-demand", "preemptible"}


def test_contact_us_rows_skipped_and_counted(rows, skipped):
    """GB300/GB200 NVL72 print '--' / '[Contact us]' - no observation is
    ever guessed for them, and the skip is countable in partial_errors."""
    items = {r["sku_identifier"] for r in rows}
    assert not any("NVL72" in item for item in items)
    assert (
        "unpriced row skipped (no numeric price published): NVIDIA GB300 NVL72"
        in skipped
    )
    assert (
        "unpriced row skipped (no numeric price published): NVIDIA GB200 NVL72"
        in skipped
    )
    assert len(skipped) == 2


def test_from_floor_prices_keep_their_qualifier(rows):
    l40s = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA L40S with Intel CPU"
        and r["tier"] == "on-demand"
    )
    assert l40s["price_usd_gpu_hr"] == 1.82
    assert l40s["raw_value"] == "from $1.82"  # the figure exactly as published
    assert l40s["extra"]["price_qualifier"] == "from"
    assert "floor price" in l40s["notes"]
    amd_pre = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA L40S with AMD CPU"
        and r["tier"] == "preemptible"
    )
    assert amd_pre["price_usd_gpu_hr"] == 0.74
    # firm-priced rows carry no qualifier
    b200 = next(r for r in rows if r["sku_identifier"] == "NVIDIA HGX B200")
    assert "price_qualifier" not in b200["extra"]


def test_header_pin_fails_closed_on_renamed_column():
    mutated = FIXTURE.read_text().replace(
        "Preemptible, GPU-hour", "Reserved, GPU-hour"
    )
    with pytest.raises(RuntimeError, match="header"):
        parse_nebius(mutated)


def test_duplicate_table_fails_closed():
    body = FIXTURE.read_text()
    with pytest.raises(RuntimeError, match="more than one"):
        parse_nebius(body + body)


def test_unrecognized_price_cell_fails_closed():
    """A price printed without the pinned '$' shape (currency no longer
    attributable) must raise, never be assumed USD."""
    mutated = FIXTURE.read_text().replace("$7.85", "7.85")
    with pytest.raises(RuntimeError, match="unrecognized price cell"):
        parse_nebius(mutated)


def test_malformed_thousands_separator_fails_closed():
    """A comma in a price cell must be a well-formed group of three -
    '$1,23.45' concatenating to 123.45 would be a silently wrong figure."""
    mutated = FIXTURE.read_text().replace("$7.85", "$1,23.45")
    with pytest.raises(RuntimeError, match="unrecognized price cell"):
        parse_nebius(mutated)


def test_region_recorded_as_unspecified(rows):
    """The page publishes no region qualifier - the observation must not
    invent one."""
    assert {r["region"] for r in rows} == {"unspecified"}


def test_row_with_extra_price_column_fails_closed():
    """Successor to the basket parser's third-price-column refusal: a row
    that grows a cell (e.g. a new Reserved price) must raise, never let the
    positional tier attribution silently shift."""
    import gpu_index.observatory.sources.nebius as nebius

    body = (
        '{"table":{"content":['
        + nebius._HEADER_ROW
        + ',["NVIDIA HGX H100","16","200","$2.15","$3.85","$9.99"]'
        + '],"customColumnWidth":[4,2,2,2]}}'
    )
    with pytest.raises(RuntimeError, match="string cells"):
        parse_nebius(body)


def test_zero_price_cell_skipped_not_recorded_as_free(parsed):
    mutated = FIXTURE.read_text().replace("$7.85", "$0.00")
    rows, skipped = parse_nebius(mutated)
    b300 = [r for r in rows if r["sku_identifier"] == "NVIDIA HGX B300"]
    assert [r["tier"] for r in b300] == ["preemptible"]  # on-demand dropped
    assert any(
        s.startswith("zero-price cell skipped (on-demand): NVIDIA HGX B300")
        for s in skipped
    )


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA HGX B300"] == "B300"
    assert mapped["NVIDIA HGX B200"] == "B200"
    assert mapped["NVIDIA HGX H200"] == "H200"
    assert mapped["NVIDIA HGX H100"] == "H100"
    assert mapped["NVIDIA RTX PRO 6000"] == "RTX_PRO_6000"
    assert mapped["NVIDIA L40S with Intel CPU"] == "L40S"
    assert mapped["NVIDIA L40S with AMD CPU"] == "L40S"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known nebius labels now unmapped: {unmapped}"
    # Lookalike discipline for the (today unpriced) superchip rows: if
    # Nebius ever prices them, they must land on GB300/GB200 - never B300/
    # B200.
    assert match_sku(catalog, "NVIDIA GB300 NVL72")["sku"] == "GB300"
    assert match_sku(catalog, "NVIDIA GB200 NVL72")["sku"] == "GB200"
