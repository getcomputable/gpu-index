# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory sesterce collector -- fixture pins (live homepage 2026-08-22).

Fixture: real bytes fetched from https://www.sesterce.com/ on 2026-08-22
(Next.js ISR page, render timestamp 18:31:24 UTC embedded), trimmed to the
CLI-demo marketing section + the #pricing live-offers section + an RSC
flight excerpt. The trim deliberately preserves every lookalike surface the
collector must refuse: the bbg-ticker marquee (every row duplicated twice),
the RSC dollar-escaped table copy ('$$8.48'), and the static CLI-demo
capacity strip whose figures conflict with the live table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.sesterce import SOURCE_ID, parse_sesterce

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "sesterce"
    / "homepage_excerpt.html"
)

MIDDOT = "\u00b7"


@pytest.fixture(scope="module")
def fixture_html():
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parsed(fixture_html):
    return parse_sesterce(fixture_html)


def test_source_id_matches_module():
    assert SOURCE_ID == "sesterce"


def test_pins_exact_prices_for_all_six_rows(parsed):
    rows, _ = parsed
    prices = {r["sku_identifier"]: r["price_usd_gpu_hr"] for r in rows}
    assert prices == {
        "B300": 8.48,
        "B200": 4.11,
        "H200": 2.48,
        "H100": 1.83,
        "A100": 1.49,
        "L40S": 0.76,
    }
    for r in rows:
        # Currency + tier are pinned by the section marker; the unit basis
        # by the pinned thead -- price*basis must reproduce the raw cell.
        assert r["currency"] == "USD"
        assert r["tier"] == "on-demand"
        assert r["region"] == "global"
        assert r["gpu_count_basis"] == 1
        assert (
            f"${r['price_usd_gpu_hr'] * r['gpu_count_basis']:.2f}"
            == r["raw_value"]
        )
        assert r["extra"]["price_basis"] == "from_floor"
        assert "From / GPU / hr" in r["notes"]


def test_ticker_and_rsc_duplicates_never_print(fixture_html, parsed):
    """The marquee repeats every row twice and the RSC flight payload
    carries the table a third time with $-escaped prices -- exactly one
    print per chip must survive."""
    assert fixture_html.count('class="bbg-ticker-item"') == 12
    assert "$$8.48" in fixture_html  # RSC dollar-escaping present
    rows, _ = parsed
    assert len(rows) == 6
    identifiers = [r["sku_identifier"] for r in rows]
    assert len(identifiers) == len(set(identifiers))


def test_cli_demo_capacity_strip_skipped_and_counted(fixture_html, parsed):
    """The static 'Sesterce CLI' terminal graphic shows a 4-chip capacity
    strip ($4.50 B200 etc.) that conflicts with the live table -- it must
    never print, and the skip must be countable in partial_errors."""
    assert fixture_html.count('class="scloud-capacity__price"') == 4
    rows, partial_errors = parsed
    recorded = {r["price_usd_gpu_hr"] for r in rows}
    assert not recorded & {4.50, 2.25, 1.66, 1.24}
    strip_notes = [p for p in partial_errors if "capacity strip" in p]
    assert len(strip_notes) == 1
    assert "4 price chips" in strip_notes[0]


def test_availability_is_recorded_but_never_a_price_qualifier(parsed):
    """B300 says 'Limited' with 0/7 live regions yet carries a real price
    -- availability is a separate published column, recorded in extra."""
    rows, _ = parsed
    b300 = next(r for r in rows if r["sku_identifier"] == "B300")
    assert b300["price_usd_gpu_hr"] == 8.48
    assert b300["extra"]["available_now"] == "Limited"
    assert b300["extra"]["availability_trend"] == "flat"
    assert b300["extra"]["listed_gpus"] == 23
    assert b300["extra"]["live_regions"] == 0
    assert b300["extra"]["total_regions"] == 7
    assert b300["extra"]["vendor_line"] == f"NVIDIA Blackwell {MIDDOT} SXM6"
    h100 = next(r for r in rows if r["sku_identifier"] == "H100")
    # One 'From' floor blends SXM5 / PCIe -- kept as qualifier, never split.
    expected_h100_vendor = f"NVIDIA Hopper {MIDDOT} SXM5 / PCIe"
    assert h100["extra"]["vendor_line"] == expected_h100_vendor
    assert h100["extra"]["available_now"] == "9 GPUs"
    assert "page_rendered_at" in b300["extra"]  # ISR staleness audit field


def test_currency_marker_flip_fails_closed(fixture_html):
    flipped = fixture_html.replace(
        f" {MIDDOT} USD {MIDDOT} ", f" {MIDDOT} EUR {MIDDOT} "
    )
    with pytest.raises(RuntimeError, match="currency\\+tier marker"):
        parse_sesterce(flipped)


def test_dropped_on_demand_marker_fails_closed(fixture_html):
    dropped = fixture_html.replace(f" {MIDDOT} ON-DEMAND", "")
    with pytest.raises(RuntimeError, match="currency\\+tier marker"):
        parse_sesterce(dropped)


def test_missing_pricing_anchor_fails_closed(fixture_html):
    with pytest.raises(RuntimeError, match="section anchor"):
        parse_sesterce(fixture_html.replace('id="pricing"', 'id="prices"'))


def test_reshaped_thead_fails_closed(fixture_html):
    """A renamed price column header (the unit-basis pin) must dark the
    source, never misattribute which number is the price."""
    reshaped = fixture_html.replace("From / GPU / hr", "Price / node / hr")
    with pytest.raises(RuntimeError, match="pinned table header"):
        parse_sesterce(reshaped)


def test_duplicated_section_fails_closed(fixture_html):
    """A second render of the section would double-print every row."""
    with pytest.raises(RuntimeError, match="section anchor"):
        parse_sesterce(fixture_html + fixture_html)


def test_price_format_change_raises_never_guessed(fixture_html):
    mutated = fixture_html.replace(
        '<span class="bbg-price">$8.48</span>',
        '<span class="bbg-price">\u20ac8.48</span>',
    )
    with pytest.raises(RuntimeError, match="misses the exact"):
        parse_sesterce(mutated)


def test_unpriced_row_is_skipped_and_counted(fixture_html):
    mutated = fixture_html.replace(
        '<td><span class="bbg-price">$8.48</span></td>', "<td></td>"
    )
    rows, partial_errors = parse_sesterce(mutated)
    assert len(rows) == 5
    assert "B300" not in {r["sku_identifier"] for r in rows}
    assert any("'B300': no price cell" in p for p in partial_errors)


def test_real_labels_normalize_through_catalog(parsed):
    rows, _ = parsed
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {
        "B300": "B300",
        "B200": "B200",
        "H200": "H200",
        "H100": "H100",
        "A100": "A100",
        "L40S": "L40S",
    }
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known sesterce labels now unmapped: {unmapped}"
