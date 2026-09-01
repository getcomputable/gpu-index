# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory together collector -- fixture pins (live page 2026-08-22).

Fixture: real bytes of https://www.together.ai/pricing trimmed to the
dedicated-inference -> sandbox section span, so every trap stays armed in
the fixture: the Dedicated Inference HGX B200 lookalike row ($8.99, with
the same is-right-border price-cell markup) BEFORE the gpu-clusters slice,
the hidden (<div class="hide">) Hardware/Hourly duplicate table INSIDE the
slice, the em-dash unpriced GB200/GB300 NVL72 + HGX B300 rows, the
'Contact us' 181+ days and colspan=4 cells, and the leading-space ' $3.45'
price cell.

Source stays ASCII-only: the page's em-dash (U+2014 unpriced cells) and
the NBSP (U+00A0) inside the hardware-label span separators are referenced
via chr() codepoints.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.together import (
    SOURCE_ID,
    _TABLE_ANCHOR,
    parse_together_pricing,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "together" / "pricing.html"
)

EM_DASH = chr(0x2014)
NBSP = chr(0xA0)


@pytest.fixture(scope="module")
def parsed():
    return parse_together_pricing(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def partial_errors(parsed):
    return parsed[1]


def _mutate_after_anchor(text, old, new, count=1):
    """Apply a replacement only past the combined-table heading, so the
    dedicated-inference and hidden-table copies of similar bytes are never
    the ones mutated."""
    i = text.index(_TABLE_ANCHOR)
    return text[:i] + text[i:].replace(old, new, count)


def test_source_id_matches_module():
    assert SOURCE_ID == "together"


def test_row_totals_and_tiers(rows):
    """3 priced chips x (1 on-demand + 3 priced tenors); the 3 em-dash
    chips print nothing."""
    assert len(rows) == 12
    assert sum(1 for r in rows if r["tier"] == "on-demand") == 3
    assert sum(1 for r in rows if r["tier"] == "reserved") == 9
    assert all(r["currency"] == "USD" for r in rows)
    assert all(r["gpu_count_basis"] == 1 for r in rows)


def test_on_demand_pins(rows):
    od = {
        r["sku_identifier"]: r["price_usd_gpu_hr"]
        for r in rows
        if r["tier"] == "on-demand"
    }
    assert od == {
        "NVIDIA HGX H100": 3.99,
        "NVIDIA HGX H200": 5.99,
        "NVIDIA HGX B200": 8.19,
    }
    b200 = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA HGX B200" and r["tier"] == "on-demand"
    )
    assert b200["raw_value"] == "$8.19"
    assert b200["extra"]["column"] == "ON-Demand (Pay as you go)"


def test_reserved_tenor_pins(rows):
    def tenors(identifier):
        return {
            r["extra"]["tenor"]: r["price_usd_gpu_hr"]
            for r in rows
            if r["sku_identifier"] == identifier and r["tier"] == "reserved"
        }

    assert tenors("NVIDIA HGX H100") == {
        "7-30 days": 3.69,
        "31-90 days": 3.45,
        "91-180 days": 3.19,
    }
    assert tenors("NVIDIA HGX H200") == {
        "7-30 days": 4.99,
        "31-90 days": 4.15,
        "91-180 days": 3.99,
    }
    assert tenors("NVIDIA HGX B200") == {
        "7-30 days": 7.99,
        "31-90 days": 7.79,
        "91-180 days": 6.79,
    }


def test_leading_space_price_cell_still_pins_exactly(rows):
    """The live 31-90d H100 cell is ' $3.45' (leading space inside the
    <p>); whitespace collapse must normalize it without loosening the
    $D.DD pin."""
    assert FIXTURE.read_text().count(" $3.45") == 1
    h100 = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA HGX H100"
        and r["tier"] == "reserved"
        and r["extra"]["tenor"] == "31-90 days"
    )
    assert h100["price_usd_gpu_hr"] == 3.45
    assert h100["raw_value"] == "$3.45"


def test_em_dash_chips_skipped_silently(rows, partial_errors):
    """GB200/GB300 NVL72 and HGX B300 carry em-dash on-demand cells: no
    print, no $0, and no partial_errors noise for the dash itself."""
    dark = {"NVIDIA GB200 NVL72", "NVIDIA GB300 NVL72", "NVIDIA HGX B300"}
    assert not [r for r in rows if r["sku_identifier"] in dark]
    assert all(EM_DASH not in e for e in partial_errors)


def test_contact_us_cells_counted(partial_errors):
    """3 x '181+ days' cells + 3 x colspan=4 cells are 'Contact us' --
    each counted, none guessed."""
    assert len(partial_errors) == 6
    assert all("'Contact us'" in e and "skipped" in e for e in partial_errors)
    spanning = [e for e in partial_errors if "7-30 days/31-90 days" in e]
    assert len(spanning) == 3


def test_dedicated_inference_lookalike_excluded(rows):
    """The fixture keeps the Dedicated Inference HGX B200 row ($8.99, same
    is-right-border cell markup) BEFORE the gpu-clusters anchor -- the
    section slice must keep it out."""
    text = FIXTURE.read_text()
    assert text.count("$8.99") == 1  # the trap row is present in the bytes
    assert text.count("HGX B200") == 3  # dedicated-inference + hidden + visible
    assert not [r for r in rows if r["price_usd_gpu_hr"] == 8.99]


def test_hidden_duplicate_table_not_double_counted(rows):
    """The slice's first table (<div class="hide">) duplicates every
    on-demand price byte-identically; each chip must print exactly once
    per tier/tenor."""
    assert FIXTURE.read_text().count('<div class="hide">') == 1
    keys = [(r["sku_identifier"], r["tier"], r.get("extra", {}).get("tenor")) for r in rows]
    assert len(keys) == len(set(keys))
    assert sum(1 for r in rows if r["sku_identifier"] == "NVIDIA HGX B200") == 4


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {
        "NVIDIA HGX H100": "H100",
        "NVIDIA HGX H200": "H200",
        "NVIDIA HGX B200": "B200",
    }
    # The unpriced-today labels are still this surface's vocabulary; if
    # they gain prices they must land on the right entries -- in
    # particular the GB-superchip rows must NEVER fall through to B200/
    # B300 (catalog boundary-aware ordering).
    for label, sku in (
        ("NVIDIA GB200 NVL72", "GB200"),
        ("NVIDIA GB300 NVL72", "GB300"),
        ("NVIDIA HGX B300", "B300"),
    ):
        entry = match_sku(catalog, label)
        assert entry and entry["sku"] == sku, (label, entry)


def test_missing_section_anchor_raises():
    text = FIXTURE.read_text().replace('id="gpu-clusters"', 'id="gpu-cluster"')
    with pytest.raises(RuntimeError, match="section anchors"):
        parse_together_pricing(text)


def test_missing_identity_pin_raises():
    text = FIXTURE.read_text().replace(
        "All prices are per GPU per hour", "All prices are per node per hour"
    )
    with pytest.raises(RuntimeError, match="identity pin"):
        parse_together_pricing(text)


def test_duplicated_table_anchor_raises():
    """Two combined-table headings in the slice = ambiguous attribution --
    the exactly-once pin must refuse, not pick one."""
    text = FIXTURE.read_text().replace(
        _TABLE_ANCHOR, _TABLE_ANCHOR + '</p><p class="h4">' + _TABLE_ANCHOR, 1
    )
    with pytest.raises(RuntimeError, match="identity pin"):
        parse_together_pricing(text)


def test_header_caption_drift_raises():
    """Tenor attribution rides on header ORDER -- any caption drift must
    refuse to attribute prices to tiers."""
    text = FIXTURE.read_text().replace(
        '<p class="caption-m">31-90 days</p>',
        '<p class="caption-m">30-90 days</p>',
    )
    with pytest.raises(RuntimeError, match="column order/labels reshaped"):
        parse_together_pricing(text)


def test_on_demand_border_binding_raises():
    """A row whose column-1 cell loses is-right-border can no longer prove
    which column is ON-Demand -- refuse, never guess."""
    label = (
        "<span>NVIDIA</span><span>" + NBSP + "</span><span>HGX H100</span>"
        "</p></td>"
    )
    text = FIXTURE.read_text()
    assert text.count(label + '<td class="is-right-border">') == 1
    text = text.replace(
        label + '<td class="is-right-border">', label + "<td>", 1
    )
    with pytest.raises(RuntimeError, match="ON-Demand column binding broke"):
        parse_together_pricing(text)


def test_reshaped_price_cell_raises_never_guesses():
    """A digit-bearing cell that misses the exact $D.DD pin must raise
    (currency/format honesty), not record."""
    text = FIXTURE.read_text().replace("$8.19", "8.19 EUR")
    with pytest.raises(RuntimeError, match="refusing to guess"):
        parse_together_pricing(text)


def test_priced_colspan_cell_skipped_never_attributed():
    """A priced cell spanning all four tenor columns has no honest
    single-tenor identity -- skipped + counted, never attributed."""
    text = FIXTURE.read_text()
    # The td form, searched from the combined-table heading -- the thead's
    # Reserved <th> also carries colspan="4", and the dedicated-inference
    # excerpt holds a td copy before the slice.
    j = text.index(
        'colspan="4" class="text-align-center"', text.index(_TABLE_ANCHOR)
    )
    text = text[:j] + text[j:].replace("Contact us", "$9.99", 1)
    rows, errs = parse_together_pricing(text)
    assert not [r for r in rows if r["price_usd_gpu_hr"] == 9.99]
    assert any("spans 4 tenor columns" in e for e in errs)


def test_tenor_column_coverage_mismatch_raises():
    # The td form, mutated only past the combined-table heading (see the
    # priced-colspan test for why).
    text = _mutate_after_anchor(
        FIXTURE.read_text(),
        'colspan="4" class="text-align-center"',
        'colspan="3" class="text-align-center"',
    )
    with pytest.raises(RuntimeError, match="tenor cells cover 3 columns"):
        parse_together_pricing(text)


def test_zero_rows_raises():
    text = _mutate_after_anchor(
        FIXTURE.read_text(), 'fs-list-element="item"', 'fs-list-element="x"', 6
    )
    with pytest.raises(RuntimeError, match="zero hardware rows"):
        parse_together_pricing(text)
