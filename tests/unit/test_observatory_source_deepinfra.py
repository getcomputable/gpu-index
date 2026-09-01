# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory deepinfra collector -- fixture pins (live page 2026-08-22).

Fixture is real bytes from https://deepinfra.com/pricing (fetched
2026-08-22), trimmed to the Custom LLMs section plus the page's trap
structures: the Tier|Scheduling|Price multiplier table ('1x base price'
cells), the Tier|Qualification-threshold table, one serverless per-token
Model table, and the i18n JSON blob that carries the same chip labels
('gpu_b200':'B200') next to an unpriced '${{price}} / hour' template.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.deepinfra import SOURCE_ID, parse_deepinfra_pricing

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "deepinfra"
    / "pricing.html"
)


@pytest.fixture(scope="module")
def fixture_text():
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def parsed(fixture_text):
    return parse_deepinfra_pricing(fixture_text)


def _table(data_rows, header=("GPU", "Memory", "Price")):
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in data_rows
    )
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def test_source_id_matches_module():
    assert SOURCE_ID == "deepinfra"


def test_all_five_rows_pinned_exactly(parsed):
    rows, partial_errors = parsed
    assert partial_errors == []
    pinned = {
        r["sku_identifier"]: (
            r["price_usd_gpu_hr"],
            r["raw_value"],
            r["memory_gb_label"],
        )
        for r in rows
    }
    assert pinned == {
        "A100": (0.89, "$0.89 / GPU-hour", 80),
        "H100": (2.20, "$2.20 / GPU-hour", 80),
        "H200": (2.69, "$2.69 / GPU-hour", 141),
        "B200": (3.69, "$3.69 / GPU-hour", 180),
        "B300": (4.89, "$4.89 / GPU-hour", 270),
    }


def test_tier_currency_and_basis(parsed):
    rows, _ = parsed
    for r in rows:
        assert r["tier"] == "on-demand"
        assert r["currency"] == "USD"
        assert r["gpu_count_basis"] == 1
        assert r["raw_unit"] == "usd_per_gpu_hr"
        # price * basis must reproduce the raw figure.
        assert f"${r['price_usd_gpu_hr']:.2f}" in r["raw_value"]
        assert r["extra"]["product"] == "custom_llm_dedicated_deployment"


def test_lookalike_tables_and_i18n_blob_excluded(fixture_text, parsed):
    """The fixture deliberately keeps the page's trap structures; parsing
    must yield ONLY the five Custom LLMs rows -- one print per chip."""
    # Guard the guards: if the fixture is ever retrimmed without the traps,
    # this test must fail rather than go vacuous.
    assert "1x base price" in fixture_text  # Tier|Scheduling|Price table
    assert "Qualification &amp; Invoicing Threshold" in fixture_text
    assert "$ per 1M input tokens" in fixture_text  # serverless Model table
    assert '"gpu_b200":"B200"' in fixture_text  # i18n label blob
    assert "${{price}} / hour" in fixture_text  # unpriced i18n template
    rows, _ = parsed
    labels = [r["sku_identifier"] for r in rows]
    assert sorted(labels) == ["A100", "B200", "B300", "H100", "H200"]


def test_missing_header_pin_raises():
    page = _table([["B200", "180GB", "$3.69 / GPU-hour"]], header=("Chip", "Memory", "Price"))
    with pytest.raises(RuntimeError, match="no table with header"):
        parse_deepinfra_pricing(page)


def test_digit_bearing_nonmatching_price_raises():
    """A currency/format change must never be guessed into USD."""
    for bad_cell in ("3.69 EUR / GPU-hour", "$3.69 / hour", "$3.69"):
        page = _table([["B200", "180GB", bad_cell]])
        with pytest.raises(RuntimeError, match="refusing to guess"):
            parse_deepinfra_pricing(page)


def test_unpriced_contact_row_skipped_and_counted():
    page = _table(
        [
            ["B200", "180GB", "$3.69 / GPU-hour"],
            ["MI300X", "192GB", "Contact us"],
        ]
    )
    rows, partial_errors = parse_deepinfra_pricing(page)
    assert [r["sku_identifier"] for r in rows] == ["B200"]
    assert partial_errors == ["row 'MI300X': unpriced cell 'Contact us' -- skipped"]


def test_duplicated_identical_tables_print_once():
    """Tab-panel duplication (seen on this page's other tables) must not
    double-print; diverging copies must abort."""
    one = _table([["B200", "180GB", "$3.69 / GPU-hour"]])
    rows, _ = parse_deepinfra_pricing(one + one)
    assert len(rows) == 1
    other = _table([["B200", "180GB", "$9.99 / GPU-hour"]])
    with pytest.raises(RuntimeError, match="DIFFERING rows"):
        parse_deepinfra_pricing(one + other)


def test_duplicate_gpu_label_raises():
    page = _table(
        [
            ["B200", "180GB", "$3.69 / GPU-hour"],
            ["B200", "270GB", "$4.89 / GPU-hour"],
        ]
    )
    with pytest.raises(RuntimeError, match="appears twice"):
        parse_deepinfra_pricing(page)


def test_malformed_row_raises():
    page = _table([["B200", "$3.69 / GPU-hour"]])  # 2 cells, not 3
    with pytest.raises(RuntimeError, match="exactly 3"):
        parse_deepinfra_pricing(page)


def test_zero_data_rows_raises():
    page = _table([])
    with pytest.raises(RuntimeError, match="zero data rows"):
        parse_deepinfra_pricing(page)


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
        "A100": "A100",
        "H100": "H100",
        "H200": "H200",
        "B200": "B200",
        "B300": "B300",
    }
