# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory civo collector -- fixture pins (live page 2026-08-22).

The fixture is a contiguous real-bytes excerpt of civo.com/pricing spanning
a lookalike NON-GPU pricing table (RAM Optimized compute instances) before
the fence, the full section#nvidia-gpus (all 6 GPU tables, 17 rows, incl.
the duplicate inner <div id="nvidia-gpus"> and the commitment-only B200
row), and the Object store pricing table after -- so the census pins prove
the section fence excludes both neighbors. Live cross-checks at capture
time: L40S 8x $10.32 = 8 x $1.29; the headline per-GPU B200 $3.79 is the
36-MONTH commitment, not on-demand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.civo import SOURCE_ID, parse_civo_pricing

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "civo"
    / "pricing_excerpt.html"
)

EXPECTED_IDENTIFIERS = {
    "NVIDIA L40S 48GB",
    "NVIDIA A100 40GB",
    "NVIDIA A100 80GB",
    "NVIDIA H100 SXM",
    "NVIDIA H200 SXM",
    "NVIDIA B200",
}


@pytest.fixture(scope="module")
def parsed():
    return parse_civo_pricing(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "civo"


def test_census_and_section_fence(parsed):
    """Exactly the 6 GPU tables' labels print -- the fixture carries full
    lookalike pricing tables on BOTH sides of the section fence (compute
    instances before, object store after) and none of their rows may leak.
    17 data rows x 5 hourly surfaces = 85, minus 2 genuine N/As (B200
    on-demand + 6-month) = 83; every skip was pinned, so no partials."""
    rows, partial_errors = parsed
    assert partial_errors == []
    assert {r["sku_identifier"] for r in rows} == EXPECTED_IDENTIFIERS
    assert len(rows) == 83
    assert {r["tier"] for r in rows} == {"on-demand", "reserved"}
    assert all(r["currency"] == "USD" for r in rows)
    assert all(r["raw_unit"] == "usd_per_instance_hr" for r in rows)


def test_h200_small_all_five_price_surfaces(rows):
    small = [
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA H200 SXM"
        and r["extra"]["size_name"] == "Small"
    ]
    surfaces = {
        (r["tier"], r["extra"].get("commitment_months")): r["price_usd_gpu_hr"]
        for r in small
    }
    assert surfaces == {
        ("on-demand", None): 3.49,
        ("reserved", 6): 3.29,
        ("reserved", 12): 3.19,
        ("reserved", 24): 3.09,
        ("reserved", 36): 2.99,
    }
    assert all(r["gpu_count_basis"] == 1 for r in small)
    assert all(r["memory_gb_label"] == 141 for r in small)
    od = next(r for r in small if r["tier"] == "on-demand")
    assert od["raw_value"] == "$3.49"
    assert od["extra"]["instance_detail"] == "1 x NVIDIA H200 - 141GB HBM3e"


def test_b200_is_commitment_only_never_on_demand(rows):
    """THE tier trap: every published B200 number is a 12/24/36-month
    commitment price (on-demand and 6-month are N/A by design). The
    headline $3.79/GPU-hr must print as tier reserved/36mo -- an on-demand
    label would look ~40% under market -- and the N/As must never
    zero-fill."""
    b200 = [r for r in rows if r["sku_identifier"] == "NVIDIA B200"]
    assert all(r["tier"] == "reserved" for r in b200)
    surfaces = {
        r["extra"]["commitment_months"]: (r["price_usd_gpu_hr"], r["raw_value"])
        for r in b200
    }
    assert surfaces == {
        12: (4.49, "$35.92"),
        24: (3.99, "$31.92"),
        36: (3.79, "$30.32"),
    }
    assert all(r["gpu_count_basis"] == 8 for r in b200)
    assert all(r["price_usd_gpu_hr"] > 0 for r in b200)


def test_per_gpu_division_reproduces_the_instance_price(rows):
    """price * gpu_count_basis == the raw per-instance figure, every row
    (e.g. L40S Extra Large $10.32 = 8 x $1.29)."""
    for r in rows:
        raw = float(r["raw_value"].replace("$", "").replace(",", ""))
        assert r["price_usd_gpu_hr"] * r["gpu_count_basis"] == pytest.approx(
            raw, abs=0.005
        )
    l40s_xl = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA L40S 48GB"
        and r["extra"]["size_name"] == "Extra Large"
        and r["tier"] == "on-demand"
    )
    assert (l40s_xl["raw_value"], l40s_xl["price_usd_gpu_hr"]) == ("$10.32", 1.29)


def test_a100_memory_lookalikes_stay_distinct(rows):
    """A100 40GB vs A100 80GB differ only by the memory size in caption and
    product-detail -- they must print as distinct identifiers with their own
    prices."""

    def small_od(identifier):
        return next(
            r
            for r in rows
            if r["sku_identifier"] == identifier
            and r["tier"] == "on-demand"
            and r["extra"]["size_name"] == "Small"
        )

    a40 = small_od("NVIDIA A100 40GB")
    a80 = small_od("NVIDIA A100 80GB")
    assert a40["price_usd_gpu_hr"] == 1.09
    assert a80["price_usd_gpu_hr"] == 1.79
    assert a40["memory_gb_label"] == 40
    assert a80["memory_gb_label"] == 80


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA L40S 48GB"] == "L40S"
    assert mapped["NVIDIA A100 40GB"] == "A100"
    assert mapped["NVIDIA A100 80GB"] == "A100"
    assert mapped["NVIDIA H100 SXM"] == "H100"
    assert mapped["NVIDIA H200 SXM"] == "H200"
    assert mapped["NVIDIA B200"] == "B200"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known civo labels now unmapped: {unmapped}"


# ---- fail-closed pins, exercised on minimal synthetic bodies ----

SECTION_OPEN = '<section class="pricing-product" id="nvidia-gpus">'


def _hourly(value):
    return (
        '<div id="price-value-hourly" data-option="hourly" '
        f'class="price-value">{value}<span>per hour</span></div>'
    )


def _row(size, count, model, od_value, terms):
    terms_html = "".join(
        f'<div id="{months}_months">{_hourly(value)}</div>'
        for months, value in terms
    )
    return (
        f'<tr><td data-title="Size">{size}'
        f'<div class="product-detail">{count} x NVIDIA {model}</div></td>'
        '<td data-title="On-demand price" '
        f'class="pricing-data on-demand-pricing">{_hourly(od_value)}</td>'
        '<td data-title="Commitment price" '
        f'class="pricing-data commitment-pricing">{terms_html}</td></tr>'
    )


def _table(caption, rows_html):
    return (
        f"<table><caption>NVIDIA {caption} GPU pricing</caption>"
        "<thead><tr><th>Size</th><th>On-demand</th><th>Commitment</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )


def _page(inner):
    return f"<html>{SECTION_OPEN}{inner}</section></html>"


_GOOD_ROW = _row("Small", 1, "H200 - 141GB HBM3e", "$3.49", [(36, "$2.99")])


def test_synthetic_round_trip_and_na_skip():
    body = _page(
        _table(
            "H200 SXM",
            _row("Small", 1, "H200 - 141GB HBM3e", "N/A", [(36, "$2.99")]),
        )
    )
    rows, partial_errors = parse_civo_pricing(body)
    assert partial_errors == []
    assert [
        (r["tier"], r["price_usd_gpu_hr"], r["extra"]["commitment_months"])
        for r in rows
    ] == [("reserved", 2.99, 36)]


def test_missing_section_fence_raises():
    with pytest.raises(RuntimeError, match="nvidia-gpus"):
        parse_civo_pricing(
            '<html><div id="nvidia-gpus">' + _table("B200", _GOOD_ROW) + "</div></html>"
        )


def test_zero_gpu_tables_inside_section_raises():
    with pytest.raises(RuntimeError, match="zero"):
        parse_civo_pricing(_page("<p>no tables here</p>"))


def test_reshaped_price_cell_raises_instead_of_guessing():
    """The hourly print pin (data-option + per-hour span + exact
    N/A-or-$D.DD) failing on a present cell must raise, not skip -- e.g. a
    currency switch away from '$' (pound sign escape-coded: source stays
    ASCII-only)."""
    body = _page(
        _table("H200 SXM", _GOOD_ROW.replace("$3.49", "\u00a32.79"))
    )
    with pytest.raises(RuntimeError, match="hourly print"):
        parse_civo_pricing(body)


def test_caption_model_mismatch_raises():
    """A row landing under the wrong caption (the A100 40GB/80GB lookalike
    hazard) must raise, never mislabel."""
    body = _page(
        _table("A100 40GB", _row("Small", 1, "A100 - 80GB", "$1.79", [(36, "$1.39")]))
    )
    with pytest.raises(RuntimeError, match="does not match"):
        parse_civo_pricing(body)


def test_priced_row_without_identity_pin_is_counted_not_guessed():
    no_pin_row = (
        '<tr><td data-title="Size">Mystery</td>'
        '<td data-title="On-demand price" '
        f'class="pricing-data on-demand-pricing">{_hourly("$9.99")}</td>'
        '<td data-title="Commitment price" '
        'class="pricing-data commitment-pricing">'
        f'<div id="36_months">{_hourly("$8.99")}</div></td></tr>'
    )
    body = _page(_table("H200 SXM", _GOOD_ROW + no_pin_row))
    rows, partial_errors = parse_civo_pricing(body)
    assert len(rows) == 2  # the pinned row still prints
    assert len(partial_errors) == 1
    assert "without a product-detail identity pin" in partial_errors[0]


def test_attributed_tr_row_still_extracts():
    """A data row whose <tr> gains an attribute (class, data-*) must still
    be scanned -- a bare-<tr>-only scanner would silently drop all five of
    its price surfaces with no raise and no partial_error."""
    body = _page(
        _table(
            "H200 SXM",
            _GOOD_ROW.replace("<tr>", '<tr class="featured">', 1),
        )
    )
    rows, partial_errors = parse_civo_pricing(body)
    assert partial_errors == []
    assert {(r["tier"], r["price_usd_gpu_hr"]) for r in rows} == {
        ("on-demand", 3.49),
        ("reserved", 2.99),
    }


def test_priced_cell_outside_any_scanned_row_raises():
    """The priced-cell census: price cells that the row scanner cannot see
    (row markup reshaped past even the attribute-tolerant <tr> pattern)
    must raise, never vanish silently."""
    floating_cell = (
        '<td class="pricing-data on-demand-pricing">'
        f"{_hourly('$9.99')}</td>"
    )
    body = _page(
        _table("H200 SXM", _GOOD_ROW).replace(
            "</tbody>", f"{floating_cell}</tbody>"
        )
    )
    with pytest.raises(RuntimeError, match="outside any scanned"):
        parse_civo_pricing(body)


def test_unattributed_print_before_first_term_block_raises():
    bad = _GOOD_ROW.replace(
        'class="pricing-data commitment-pricing">',
        'class="pricing-data commitment-pricing">' + _hourly("$1.00"),
    )
    with pytest.raises(RuntimeError, match="before the first"):
        parse_civo_pricing(_page(_table("H200 SXM", bad)))
