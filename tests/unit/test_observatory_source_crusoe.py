# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory crusoe collector -- fixture pins (live page 2026-08-22).

The fixture is a real contiguous excerpt of
https://www.crusoe.ai/cloud/pricing (captured 2026-08-22) spanning the GPU
instances section PLUS every lookalike surface that must never print: the
CPU instance rows ($0.04/vCPU-hr, same row container class), the Serverless
Inference table ($/M-token prices in the same 'pricing-rich' cell class),
and the Self-Serve Deployments rows (identical 'NVIDIA H100'/'NVIDIA H200'
headings priced bare $5.50/$6.00 for a different managed product). Parsing
the whole excerpt and getting EXACTLY the six pinned GPU-rental prints is
the proof the fences hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.crusoe import SOURCE_ID, parse_crusoe_pricing

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "crusoe" / "pricing.html"
)

# The six real GPU-rental prints on the captured page -- heading + spec
# tags is the row identity (the two A100 rows share a bare heading).
EXPECTED_ON_DEMAND = {
    "NVIDIA H200 141GB HGX": 4.29,
    "NVIDIA H100 80GB HGX": 3.90,
    "NVIDIA A100 80GB SXM": 2.30,
    "NVIDIA A100 80GB PCIe": 2.00,
    "NVIDIA L40S 48GB": 1.50,
    "AMD MI300X 192GB": 3.45,
}

_HEADER_PIN = (
    '<div class="pricing-heading is-d">GPU model</div>'
    '<div class="pricing-heading is-d">On-demand</div>'
    '<div class="pricing-heading is-d">Current spot</div>'
)


def _row(heading, mem="80GB", form="HGX", cells=None):
    if cells is None:
        cells = ("$1.00/GPU-hr", '<a href="/contact-sales">Contact sales</a>')
    cell_html = "".join(
        f'<div class="pricing-rich w-richtext"><p>{c}</p></div>' for c in cells
    )
    return (
        '<div role="listitem" class="prixing-item w-dyn-item">'
        '<div class="pricing_gpu-item">'
        f'<h4 class="pricing-item-heading">{heading}</h4>'
        f'<div class="pricing-tag is-dark">{mem}</div>'
        f'<div class="pricing-tag">{form}</div>'
        f"{cell_html}</div></div>"
    )


def _page(rows_html):
    return (
        "<h3>GPU instances pricing</h3>"
        + _HEADER_PIN
        + rows_html
        + "<h3>CPU instances pricing</h3>"
    )


@pytest.fixture(scope="module")
def parsed():
    return parse_crusoe_pricing(FIXTURE.read_text())


def test_source_id_matches_module():
    assert SOURCE_ID == "crusoe"


def test_exactly_the_six_gpu_rental_prints(parsed):
    """The excerpt contains ~40 lookalike rows (CPU, inference, Self-Serve
    Deployments) -- exactly six observations proves the section fence and
    the /GPU-hr pin exclude every one of them."""
    rows, _ = parsed
    assert {
        r["sku_identifier"]: r["price_usd_gpu_hr"] for r in rows
    } == EXPECTED_ON_DEMAND
    assert all(r["tier"] == "on-demand" for r in rows)
    assert all(r["currency"] == "USD" for r in rows)


def test_raw_value_reproduces_price(parsed):
    rows, _ = parsed
    for r in rows:
        assert r["raw_value"] == f"${r['price_usd_gpu_hr']:.2f}/GPU-hr"
        assert r["gpu_count_basis"] == 1
        assert r["raw_unit"] == "usd_per_gpu_hr"


def test_a100_variants_do_not_collide(parsed):
    rows, _ = parsed
    a100 = {
        r["sku_identifier"]: r["price_usd_gpu_hr"]
        for r in rows
        if "A100" in r["sku_identifier"]
    }
    assert a100 == {
        "NVIDIA A100 80GB SXM": 2.30,
        "NVIDIA A100 80GB PCIe": 2.00,
    }


def test_l40s_empty_form_factor_tag_tolerated(parsed):
    rows, _ = parsed
    l40s = next(r for r in rows if "L40S" in r["sku_identifier"])
    assert l40s["sku_identifier"] == "NVIDIA L40S 48GB"
    assert l40s["memory_gb_label"] == 48
    assert l40s["extra"]["form_factor_tag"] is None


def test_contact_sales_rows_skipped_and_counted_never_priced(parsed):
    """GB200/B200/MI355X are contact-sales lookalikes in the same table --
    they must never print, but their skips are counted; every Current-spot
    cell is contact-sales on the captured page, so no spot prints at all."""
    rows, partial_errors = parsed
    identifiers = {r["sku_identifier"] for r in rows}
    assert not any("GB200" in i or "B200" in i or "MI355X" in i for i in identifiers)
    assert not any(r["tier"] == "spot" for r in rows)
    joined = "\n".join(partial_errors)
    for label in ("NVIDIA GB200", "NVIDIA B200", "AMD MI355X"):
        assert label in joined, f"skipped row {label!r} not accounted for"
    # 3 fully-unpriced rows x 2 cells + 6 priced rows x 1 spot cell.
    assert len(partial_errors) == 12
    assert all("unpriced cell" in e for e in partial_errors)


def test_real_labels_normalize_through_catalog(parsed):
    rows, _ = parsed
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA H200 141GB HGX"] == "H200"
    assert mapped["NVIDIA H100 80GB HGX"] == "H100"
    assert mapped["NVIDIA A100 80GB SXM"] == "A100"
    assert mapped["NVIDIA A100 80GB PCIe"] == "A100"
    assert mapped["NVIDIA L40S 48GB"] == "L40S"
    assert mapped["AMD MI300X 192GB"] == "MI300X"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known crusoe labels now unmapped: {unmapped}"
    # The contact-sales rows print the day Crusoe publishes numbers -- their
    # real fixture labels must already map so that day is not an unmapped
    # surprise.
    assert match_sku(catalog, "NVIDIA GB200 186GB NVL72")["sku"] == "GB200"
    assert match_sku(catalog, "NVIDIA B200 180GB HGX")["sku"] == "B200"
    assert match_sku(catalog, "AMD MI355X 288GB")["sku"] == "MI355X"


def test_missing_section_heading_raises():
    body = FIXTURE.read_text().replace("CPU instances pricing", "CPUs")
    with pytest.raises(RuntimeError, match="identity pin"):
        parse_crusoe_pricing(body)


def test_header_relabel_raises():
    """A column relabel (spot -> reserved, say) must fail loud -- tier
    attribution rides on the pinned caption order."""
    body = FIXTURE.read_text().replace("Current spot", "Reserved")
    with pytest.raises(RuntimeError, match="identity pin"):
        parse_crusoe_pricing(body)


def test_offpin_priced_cell_raises_not_guessed():
    """A cell that keeps digits but drops the /GPU-hr basis suffix must
    raise, never print (basis/currency change)."""
    body = FIXTURE.read_text().replace("$4.29/GPU-hr", "$4.29/hr")
    with pytest.raises(RuntimeError, match="refusing to guess"):
        parse_crusoe_pricing(body)


def test_trailing_paragraph_qualifier_raises_not_dropped():
    """A basis qualifier in a SECOND <p> of the price cell (the page's
    Storage table already splits cells across two <p>s) must break the pin
    and raise -- never print $4.29 with the 'per 8-GPU node' part silently
    dropped."""
    body = FIXTURE.read_text().replace(
        "<p>$4.29/GPU-hr</p>", "<p>$4.29/GPU-hr</p><p>per 8-GPU node</p>"
    )
    with pytest.raises(RuntimeError, match="refusing to guess"):
        parse_crusoe_pricing(body)


def test_post_paragraph_qualifier_raises_not_dropped():
    """Same attack with the qualifier as bare text after the first </p>."""
    body = FIXTURE.read_text().replace(
        "<p>$4.29/GPU-hr</p>", "<p>$4.29/GPU-hr</p>per 8-GPU node"
    )
    with pytest.raises(RuntimeError, match="refusing to guess"):
        parse_crusoe_pricing(body)


def test_nested_div_in_cell_raises():
    """A nested <div> would truncate the whole-cell capture -- fail closed
    instead of classifying a partial body."""
    body = FIXTURE.read_text().replace(
        "<p>$4.29/GPU-hr</p>", "<div><p>$4.29/GPU-hr</p></div>"
    )
    with pytest.raises(RuntimeError, match="nested <div>"):
        parse_crusoe_pricing(body)


def test_wrong_cell_count_raises():
    row = _row("NVIDIA H200", mem="141GB", cells=("$4.29/GPU-hr",))
    with pytest.raises(RuntimeError, match="exactly 2 price cells"):
        parse_crusoe_pricing(_page(row))


def test_unpinnable_vendor_row_skipped_and_counted():
    rows_html = "".join(
        _row(h, mem=m)
        for h, m in (
            ("NVIDIA H200", "141GB"),
            ("NVIDIA H100", "80GB"),
            ("NVIDIA A100", "80GB"),
            ("AMD MI300X", "192GB"),
        )
    ) + _row("Coming soon")
    rows, partial_errors = parse_crusoe_pricing(_page(rows_html))
    assert len(rows) == 4
    assert any("Coming soon" in e and "skipped" in e for e in partial_errors)


def test_thin_capture_raises():
    """Fewer than 4 priced on-demand rows is a reshaped page, not a
    healthy capture."""
    body = FIXTURE.read_text()
    for price in ("$1.50/GPU-hr", "$2.00/GPU-hr", "$2.30/GPU-hr"):
        body = body.replace(price, "Contact sales")
    with pytest.raises(RuntimeError, match="thin capture"):
        parse_crusoe_pricing(body)
