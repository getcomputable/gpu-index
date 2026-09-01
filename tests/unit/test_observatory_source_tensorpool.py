# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory tensorpool collector -- fixture pins (live page 2026-08-22).

Fixture: the FULL real byte stream of https://tensorpool.dev/pricing
(~31KB, server-rendered, self-contained), captured 2026-08-22, so every
trap stays armed in the fixture:

  - the mobile per-provider cards (TensorPool / Lambda Labs / Traditional
    Clouds) render the same table a second time WITHOUT the /hr suffix --
    double-print bait and the equality-tripwire input;
  - the desktop grid's competitor columns repeat every chip label with
    different prices ($3.29 Lambda H100 vs $1.99 TensorPool; $18
    Traditional B200) and B200 SXM is $4.99 in BOTH the Lambda and
    TensorPool columns, so a mis-pinned parser passes a B200 spot-check
    by accident;
  - N/A not-offered cells, the priced-but-not-a-GPU CPU row ($0.015/hr),
    and the storage section ($100/TB/mo, $50/TB/mo) below the grid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.tensorpool import SOURCE_ID, parse_tensorpool_pricing

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "tensorpool"
    / "pricing.html"
)

EXPECTED_PRICES = {
    "B200 SXM": 4.99,
    "H200 SXM": 2.99,
    "B300 SXM": 5.49,
    "H100 SXM": 1.99,
    "L40S": 1.49,
}


@pytest.fixture(scope="module")
def parsed():
    return parse_tensorpool_pricing(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def partial_errors(parsed):
    return parsed[1]


def test_source_id_matches_module():
    assert SOURCE_ID == "tensorpool"


def test_tensorpool_column_pins(rows):
    """Exactly the five GPU rows, TensorPool-column prices only."""
    assert {r["sku_identifier"]: r["price_usd_gpu_hr"] for r in rows} == (
        EXPECTED_PRICES
    )
    assert all(r["tier"] == "on-demand" for r in rows)
    assert all(r["currency"] == "USD" for r in rows)
    assert all(r["gpu_count_basis"] == 1 for r in rows)
    assert all(r["raw_unit"] == "usd_per_gpu_hr" for r in rows)
    assert all(r["extra"]["column"] == "TensorPool" for r in rows)
    b300 = next(r for r in rows if r["sku_identifier"] == "B300 SXM")
    assert b300["raw_value"] == "$5.49/hr"
    assert b300["extra"]["grid_category"] == "ON-DEMAND"


def test_no_partial_errors_on_live_shape(partial_errors):
    """All five GPU rows were priced on the captured page."""
    assert partial_errors == []


def test_competitor_columns_never_recorded(rows):
    """Lambda ($3.29 H100) and Traditional Clouds ($18/$10/$14/$3) cells
    share the recorded rows' chip labels -- none may print."""
    text = FIXTURE.read_text()
    for armed in ("$3.29/hr", "$18/hr", "$10/hr", "$14/hr", "$3/hr"):
        assert armed in text, f"competitor trap {armed!r} left the fixture"
    competitor_prices = {3.29, 18.0, 10.0, 14.0, 3.0}
    assert not competitor_prices & {r["price_usd_gpu_hr"] for r in rows}


def test_b200_price_collision_discriminated_via_h100(rows):
    """B200 is $4.99 in BOTH the Lambda and TensorPool columns, so its pin
    proves nothing about column binding; H100 ($1.99 TensorPool vs $3.29
    Lambda) is the discriminating row."""
    h100 = next(r for r in rows if r["sku_identifier"] == "H100 SXM")
    assert h100["price_usd_gpu_hr"] == 1.99


def test_cpu_row_fenced(rows):
    """The CPU row is priced in the TensorPool column ($0.015/hr) but is
    not a GPU -- fenced by label, silently."""
    assert "$0.015/hr" in FIXTURE.read_text()
    assert not [r for r in rows if r["sku_identifier"] == "CPU"]
    assert 0.015 not in {r["price_usd_gpu_hr"] for r in rows}


def test_storage_section_fenced(rows):
    """Storage prices ($100/TB/mo shared, $50/TB/mo object) sit below the
    grid slice and must never print."""
    assert "Storage Pricing" in FIXTURE.read_text()
    assert not {100.0, 50.0} & {r["price_usd_gpu_hr"] for r in rows}


def test_na_tensorpool_cell_skipped_and_counted():
    """A TensorPool N/A cell (mutated onto H200 in BOTH renderings so the
    equality tripwire still holds) is skipped + counted, never a $0."""
    text = (
        FIXTURE.read_text()
        .replace("$2.99/hr", "N/A")  # grid TensorPool cell
        .replace("$2.99</strong>", "N/A</strong>")  # mobile-card row
    )
    rows, errors = parse_tensorpool_pricing(text)
    assert {r["sku_identifier"] for r in rows} == (
        set(EXPECTED_PRICES) - {"H200 SXM"}
    )
    assert len(errors) == 1 and "H200 SXM" in errors[0]


def test_card_grid_mismatch_raises():
    """The mobile card is the consistency tripwire: a price that differs
    between the two renderings must refuse the whole capture."""
    text = FIXTURE.read_text().replace("$1.99</strong>", "$1.89</strong>")
    with pytest.raises(RuntimeError, match="renderings of the same table"):
        parse_tensorpool_pricing(text)


def test_chip_missing_from_card_raises():
    """A grid row absent from the card rendering (or vice versa) is a
    reshape -- the tripwire compares complete label sets."""
    text = FIXTURE.read_text().replace(
        '<div class="mb-2">L40S - <strong>$1.49</strong></div>', "", 1
    )
    with pytest.raises(RuntimeError, match="renderings of the same table"):
        parse_tensorpool_pricing(text)


def test_highlight_missing_from_tensorpool_cell_raises():
    """The bg-[#B8E7F5] class dropping off a TensorPool data cell (e.g. a
    Tailwind restyle) must fail closed, not fall back to position."""
    text = FIXTURE.read_text().replace(
        "font-medium bg-[#B8E7F5] border-t", "font-medium border-t", 1
    )
    with pytest.raises(RuntimeError, match="no longer discriminates"):
        parse_tensorpool_pricing(text)


def test_highlight_leaking_onto_competitor_cell_raises():
    """The class appearing on a Lambda cell means it no longer identifies
    the TensorPool column -- refuse, never guess by position."""
    text = FIXTURE.read_text().replace(
        'font-medium border-t border-[#e0e0e0]">$3.29/hr',
        'font-medium bg-[#B8E7F5] border-t border-[#e0e0e0]">$3.29/hr',
    )
    with pytest.raises(RuntimeError, match="no longer discriminates"):
        parse_tensorpool_pricing(text)


def test_header_drift_raises():
    """Column attribution rides on header order -- any drift in the four
    pinned captions refuses extraction."""
    text = FIXTURE.read_text().replace(
        ">TensorPool</div>", ">Tensor Pool</div>"
    )
    with pytest.raises(RuntimeError, match="column order/labels reshaped"):
        parse_tensorpool_pricing(text)


def test_grid_anchor_missing_raises():
    text = FIXTURE.read_text().replace(
        "grid-cols-[200px_200px_1fr_1fr_1fr]", "grid-cols-5"
    )
    with pytest.raises(RuntimeError, match="grid anchor"):
        parse_tensorpool_pricing(text)


def test_grid_anchor_duplicated_raises():
    """Two grids matching the anchor = ambiguous attribution -- the
    exactly-once pin must refuse, not pick one."""
    text = FIXTURE.read_text() + (
        '<div class="hidden md:grid grid-cols-[200px_200px_1fr_1fr_1fr]">'
        "</div>"
    )
    with pytest.raises(RuntimeError, match="grid anchor"):
        parse_tensorpool_pricing(text)


def test_first_category_not_on_demand_raises():
    """The ON-DEMAND Category cell is the tier pin -- renamed/relabeled
    means tier attribution is unverified."""
    text = FIXTURE.read_text().replace("ON-DEMAND", "RESERVED")
    with pytest.raises(RuntimeError, match="tier attribution unverified"):
        parse_tensorpool_pricing(text)


def test_second_category_section_raises():
    """A new non-empty Category label further down (a future reserved
    section gaining prices) must raise for re-verification, never print
    its rows as on-demand."""
    text = FIXTURE.read_text().replace(
        '<div class="p-5"></div>', '<div class="p-5">RESERVED</div>', 1
    )
    with pytest.raises(RuntimeError, match="second tier section"):
        parse_tensorpool_pricing(text)


def test_price_format_drift_raises():
    """A digit-bearing TensorPool cell that misses the exact $D[.DD]/hr
    pin (currency change, added text) must raise, never guess USD."""
    text = FIXTURE.read_text().replace("$5.49/hr", "5.49/hr")
    with pytest.raises(RuntimeError, match="refusing to guess"):
        parse_tensorpool_pricing(text)


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {
        "B200 SXM": "B200",
        "H200 SXM": "H200",
        "B300 SXM": "B300",
        "H100 SXM": "H100",
        "L40S": "L40S",
    }
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known tensorpool labels now unmapped: {unmapped}"
