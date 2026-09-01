# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory fal collector -- fixture pins (live page 2026-08-22).

Fixture: real bytes of https://fal.ai/pricing trimmed to two excerpts that
keep every trap armed: (1) the 'Serverless & Compute Pricing' section span
holding all THREE near-identical <table> blocks -- the pinned GPU table
plus both model-API lookalikes (per-output prices, header Model | Unit |
Price | Output per $1) whose header tuple is the ONLY discriminator -- and
the page's own 'as low as $1.89/hr for H100' prose cross-check; (2) an RSC
flight excerpt carrying the framework-duplicated bytes of the GPU rows
(GPU-B300/GPU-B200 row keys, a second '$8.50/h'/'$4.49/h', a second
'List Price'/'As low as' header), so a whole-page regex sweep would
double-print and only table-scoped parsing stays honest.

Mutation care: the '$1.89/h' bytes appear in the PROSE before the table,
and '$4.49/h' appears in the table BEFORE its RSC duplicate -- replace-once
mutations below target the table copy via those orderings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.fal import SOURCE_ID, parse_fal_pricing

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "fal" / "pricing.html"
)

_MINI_HEADER = (
    "<table><thead><tr><th>GPU</th><th>VRAM</th><th>List Price</th>"
    "<th>As low as</th></tr></thead><tbody>"
)


def _mini_table(*rows: str) -> str:
    """A minimal synthetic table carrying the pinned header."""
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r.split("|")) + "</tr>"
        for r in rows
    )
    return _MINI_HEADER + body + "</tbody></table>"


@pytest.fixture(scope="module")
def rows():
    return parse_fal_pricing(FIXTURE.read_text())


def test_source_id_matches_module():
    assert SOURCE_ID == "fal"


def test_row_totals_and_tiers(rows):
    """5 chips x 2 price columns; both columns USD per single GPU-hour."""
    assert len(rows) == 10
    assert sum(1 for r in rows if r["tier"] == "serverless") == 5
    assert sum(1 for r in rows if r["tier"] == "serverless_as_low_as") == 5
    assert all(r["currency"] == "USD" for r in rows)
    assert all(r["gpu_count_basis"] == 1 for r in rows)


def test_list_price_pins(rows):
    listed = {
        r["sku_identifier"]: r["price_usd_gpu_hr"]
        for r in rows
        if r["tier"] == "serverless"
    }
    assert listed == {
        "B300": 8.50,
        "B200": 6.25,
        "H200": 4.50,
        "H100": 4.50,
        "RTX PRO 6000": 2.99,
    }
    b300 = next(
        r
        for r in rows
        if r["sku_identifier"] == "B300" and r["tier"] == "serverless"
    )
    assert b300["raw_value"] == "$8.50/h"
    assert b300["extra"]["column"] == "List Price"
    assert b300["memory_gb_label"] == 288


def test_as_low_as_floor_pins(rows):
    """The famous $4.49 B300 sighting is the FLOOR column -- it must carry
    its own tier label and floor marker, never the list tier. The H100
    floor cross-checks against the page's own rendered prose ('as low as
    $1.89/hr for H100'), which the fixture preserves."""
    assert "$1.89/hr for H100" in FIXTURE.read_text()
    floors = {
        r["sku_identifier"]: r["price_usd_gpu_hr"]
        for r in rows
        if r["tier"] == "serverless_as_low_as"
    }
    assert floors == {
        "B300": 4.49,
        "B200": 3.49,
        "H200": 2.10,
        "H100": 1.89,
        "RTX PRO 6000": 1.10,
    }
    for r in rows:
        if r["tier"] == "serverless_as_low_as":
            assert r["extra"]["pricing_basis"] == "as_low_as_floor"
            assert "not the bookable list rate" in r["notes"]
    # The floor price must never leak into the list tier.
    assert not [
        r
        for r in rows
        if r["tier"] == "serverless" and r["price_usd_gpu_hr"] == 4.49
    ]


def test_h100_h200_share_list_price_not_deduped(rows):
    """H100 and H200 both list $4.50/h live -- rows are keyed by label and
    a price-keyed dedup would silently drop one chip's history."""
    at_450 = sorted(
        r["sku_identifier"]
        for r in rows
        if r["tier"] == "serverless" and r["price_usd_gpu_hr"] == 4.50
    )
    assert at_450 == ["H100", "H200"]


def test_memory_labels(rows):
    mem = {r["sku_identifier"]: r["memory_gb_label"] for r in rows}
    assert mem == {
        "B300": 288,
        "B200": 180,
        "H200": 141,
        "H100": 80,
        "RTX PRO 6000": 96,
    }
    assert all(r["extra"]["vram_label"].endswith("GB") for r in rows)


def test_model_api_lookalike_tables_excluded(rows):
    """The fixture keeps both model-API tables (per-output prices in
    markup identical to the GPU table's, discriminated ONLY by the header
    tuple) -- none of their rows may print."""
    text = FIXTURE.read_text()
    assert text.count("Output per $1") == 2  # both lookalikes present
    assert "Wan 2.5" in text and "Seedream V4" in text
    assert {r["sku_identifier"] for r in rows} == {
        "B300",
        "B200",
        "H200",
        "H100",
        "RTX PRO 6000",
    }
    assert all(r["price_usd_gpu_hr"] >= 1.0 for r in rows)  # no $0.0x prints


def test_rsc_flight_duplicate_bytes_not_double_counted(rows):
    """Every price string appears twice in the document (HTML table + RSC
    flight duplicate); the fixture preserves the duplication for the B300
    row and each observation must still print exactly once."""
    text = FIXTURE.read_text()
    assert text.count("$8.50/h") == 2  # the trap bytes are present
    assert text.count("GPU-B300") == 1  # the RSC row key rides along
    keys = [(r["sku_identifier"], r["tier"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert sum(1 for r in rows if r["sku_identifier"] == "B300") == 2


def test_real_labels_normalize_through_catalog(rows):
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
        # Bare 'H100' (80GB): the catalog's policy entry for unqualified
        # H100 labels -- the VRAM cell rides in memory_gb_label.
        "H100": "H100",
        # Workstation Blackwell -- must land on RTX_PRO_6000, not a
        # datacenter lookalike.
        "RTX PRO 6000": "RTX_PRO_6000",
    }
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known fal labels now unmapped: {unmapped}"


def test_header_pin_zero_matches_raises():
    """Renaming one header cell must leave zero pinned tables -- the two
    model-API tables are never an acceptable fallback."""
    text = FIXTURE.read_text()
    assert text.count(">GPU</th>") == 1
    with pytest.raises(RuntimeError, match="no table with the pinned header"):
        parse_fal_pricing(text.replace(">GPU</th>", ">Chip</th>"))


def test_header_pin_client_side_rendering_fails_closed():
    """A future move to CSR empties the server HTML of tables entirely --
    same zero-match refusal, never a healthy zero-observation capture."""
    with pytest.raises(RuntimeError, match="no table with the pinned header"):
        parse_fal_pricing("<html><body><div>loading...</div></body></html>")


def test_header_pin_two_matches_raises():
    """A duplicate render of the GPU table would double-print every row --
    the exactly-one pin must refuse, not pick."""
    text = FIXTURE.read_text()
    first_table = text[
        text.index("<table") : text.index("</table>") + len("</table>")
    ]
    assert ">GPU</th>" in first_table  # the fixture's first table is the GPU one
    with pytest.raises(RuntimeError, match="refusing to pick one"):
        parse_fal_pricing(text + first_table)


def test_contact_us_cell_refuses_capture():
    """An unpriced/contact-us cell is a reshape signal on this five-row
    rate card -- the whole capture refuses rather than skipping the row.
    (Replace-once hits the table's '$4.49/h'; its RSC duplicate comes
    later in the fixture bytes.)"""
    text = FIXTURE.read_text().replace("$4.49/h", "Contact us", 1)
    with pytest.raises(RuntimeError, match="refusing to guess"):
        parse_fal_pricing(text)


def test_currency_or_format_change_refuses_capture():
    """A digit-bearing cell that misses the exact $D.DD/h pin must raise
    -- a currency/format change is never guessed into USD."""
    text = FIXTURE.read_text().replace("$2.10/h", "2.10 EUR/h", 1)
    with pytest.raises(RuntimeError, match="refusing to guess"):
        parse_fal_pricing(text)


def test_vram_fence_refuses_reshaped_column():
    text = FIXTURE.read_text().replace(">288GB<", ">288 GB VRAM<", 1)
    with pytest.raises(RuntimeError, match="digits\\+GB pin"):
        parse_fal_pricing(text)


def test_wrong_cell_count_refuses_capture():
    with pytest.raises(RuntimeError, match="expected 4"):
        parse_fal_pricing(_mini_table("B300|288GB|$8.50/h|$4.49/h|extra"))


def test_zero_data_rows_raises():
    with pytest.raises(RuntimeError, match="zero data rows"):
        parse_fal_pricing(_mini_table())


def test_duplicate_label_refuses_capture():
    with pytest.raises(RuntimeError, match="more than one table row"):
        parse_fal_pricing(
            _mini_table("B300|288GB|$8.50/h|$4.49/h", "B300|288GB|$9.00/h|$5.00/h")
        )


def test_empty_label_refuses_capture():
    with pytest.raises(RuntimeError, match="empty label"):
        parse_fal_pricing(_mini_table(" |288GB|$8.50/h|$4.49/h"))
