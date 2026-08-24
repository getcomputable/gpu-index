# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory coreweave collector -- fixture pins (live page 2026-08-22).

Fixture: real bytes of coreweave.com/pricing trimmed to the GPU-instances
section (both region tables) PLUS the head of the CPU section below it --
the CPU table reuses the same row markup, h3 shape and a second
'REGION: NORTH AMERICA' heading, so keeping it in the fixture proves the
section slice (not luck) is what fences those prices out. Edge rows kept:
GB300 NVL72 (fully unpriced, 'Contact sales'), GB200 NVL72 (literal '4^1'
GPU Count cell, spot N/A), HGX B300 (spot-only), GH200 (1-GPU basis), and
the EUROPE table's N/A spot cells.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.coreweave import SOURCE_ID, parse_coreweave

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "coreweave"
    / "pricing.html"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_coreweave(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def partial_errors(parsed):
    return parsed[1]


def _one(rows, name, region, tier):
    matches = [
        r
        for r in rows
        if r["sku_identifier"] == name
        and r["region"] == region
        and r["tier"] == tier
    ]
    assert len(matches) == 1, f"{name}/{region}/{tier}: {len(matches)} rows"
    return matches[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "coreweave"


def test_census(rows):
    """Both region tables print; tier vocabulary is exactly the two
    labeled spans (the inference span is a separate product and the
    unlabeled duplicate price divs are never a pin)."""
    assert len(rows) == 32
    by_region = {"NORTH AMERICA": 0, "EUROPE": 0}
    for r in rows:
        by_region[r["region"]] += 1
    assert by_region == {"NORTH AMERICA": 17, "EUROPE": 15}
    assert {r["tier"] for r in rows} == {"on-demand", "spot"}


def test_b200_exact_pins(rows):
    od = _one(rows, "NVIDIA HGX B200", "NORTH AMERICA", "on-demand")
    assert od["price_usd_gpu_hr"] == 8.6
    assert od["currency"] == "USD"
    assert od["raw_value"] == "$68.80"
    assert od["raw_unit"] == "usd_per_instance_hr"
    assert od["gpu_count_basis"] == 8
    # price * basis reproduces the raw per-instance figure exactly.
    assert od["price_usd_gpu_hr"] * od["gpu_count_basis"] == 68.80
    assert od["memory_gb_label"] == 180
    assert od["extra"]["data_product"] == "nvidia-b200"
    assert od["extra"]["gpu_count_cell"] == "8"
    assert od["extra"]["vcpus"] == 128
    assert od["extra"]["system_ram_gb"] == 2048
    assert od["extra"]["local_storage_tb"] == 61.44
    spot = _one(rows, "NVIDIA HGX B200", "EUROPE", "spot")
    assert spot["raw_value"] == "$34.87"
    assert spot["price_usd_gpu_hr"] * 8 == pytest.approx(34.87, abs=1e-3)


def test_gb200_superchip_count_cell(rows):
    """GB200 NVL72 is priced per 4-GPU instance and its GPU Count cell is
    the literal footnote text '4^1' (2 superchips x 2 GPUs) -- the leading
    integer is the basis, the raw cell is preserved, and the part is a
    first-class sku here (not the basket lanes' quarantine class)."""
    for region in ("NORTH AMERICA", "EUROPE"):
        od = _one(rows, "NVIDIA GB200 NVL72", region, "on-demand")
        assert od["price_usd_gpu_hr"] == 10.5
        assert od["raw_value"] == "$42.00"
        assert od["gpu_count_basis"] == 4
        assert od["extra"]["gpu_count_cell"] == "4^1"
    # Spot cell is 'N/A' in both regions -- never a print.
    assert not [
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA GB200 NVL72" and r["tier"] == "spot"
    ]


def test_gb300_contact_sales_row_skipped_not_guessed(rows, partial_errors):
    assert not [
        r for r in rows if r["sku_identifier"] == "NVIDIA GB300 NVL72"
    ]
    gb300_errors = [e for e in partial_errors if "GB300" in e]
    assert len(gb300_errors) == 2
    assert any("NORTH AMERICA" in e for e in gb300_errors)
    assert any("EUROPE" in e for e in gb300_errors)
    assert all("contact-sales" in e for e in gb300_errors)


def test_b300_is_spot_only(rows):
    """HGX B300's On-Demand item-value is empty (page hides the span) --
    only the labeled spot print records."""
    b300 = [r for r in rows if r["sku_identifier"] == "NVIDIA HGX B300"]
    assert {(r["region"], r["tier"]) for r in b300} == {
        ("NORTH AMERICA", "spot"),
        ("EUROPE", "spot"),
    }
    assert _one(rows, "NVIDIA HGX B300", "NORTH AMERICA", "spot")[
        "price_usd_gpu_hr"
    ] == 4.48
    assert _one(rows, "NVIDIA HGX B300", "EUROPE", "spot")[
        "price_usd_gpu_hr"
    ] == 4.5875


def test_gh200_single_gpu_basis(rows):
    for region in ("NORTH AMERICA", "EUROPE"):
        od = _one(rows, "NVIDIA GH200", region, "on-demand")
        assert od["gpu_count_basis"] == 1
        assert od["price_usd_gpu_hr"] == 6.5
        assert od["raw_value"] == "$6.50"


def test_regional_spot_divergence(rows):
    """EUROPE drops spot for L40/L40S ('N/A') while NORTH AMERICA prices
    them; GH200 spot is 'N/A' in BOTH regions -- N/A tiers are skipped
    per-region, never copied over."""
    assert _one(rows, "NVIDIA L40S", "NORTH AMERICA", "spot")[
        "price_usd_gpu_hr"
    ] == 0.985
    eu_spotless = [
        r
        for r in rows
        if r["region"] == "EUROPE"
        and r["tier"] == "spot"
        and r["sku_identifier"] in ("NVIDIA L40", "NVIDIA L40S", "NVIDIA GH200")
    ]
    assert eu_spotless == []
    assert not [
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA GH200" and r["tier"] == "spot"
    ]
    assert _one(rows, "NVIDIA HGX H200", "EUROPE", "spot")[
        "price_usd_gpu_hr"
    ] == 2.58


def test_inference_span_never_prints(rows):
    """The 'Inference Single CPU Price' span (footnote 2: inference
    platform customers only) is a different product. Its NA per-GPU
    figures ($8.60 B200, $6.16 H100) must not appear as raw prints --
    every raw_value here is a per-instance labeled od/spot figure."""
    raws = {r["raw_value"] for r in rows}
    assert "$8.60" not in raws
    assert "$6.16" not in raws
    assert "$10.50" not in raws  # GB200 inference figure


def test_cpu_section_lookalike_never_prints(rows):
    """The fixture keeps the CPU section head on purpose: same row-split
    marker, same h3 shape, priced rows, and a second 'REGION: NORTH
    AMERICA' heading. The GPU/CPU slice must fence it all out."""
    text = FIXTURE.read_text()
    assert "AMD Genoa" in text
    assert text.count("REGION: NORTH AMERICA") == 2
    assert not [r for r in rows if "Genoa" in r["sku_identifier"]]


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA HGX B200"] == "B200"
    assert mapped["NVIDIA HGX B300"] == "B300"
    assert mapped["NVIDIA GB200 NVL72"] == "GB200"
    assert (
        mapped["NVIDIA RTX PRO 6000 Blackwell Server Edition"]
        == "RTX_PRO_6000"
    )
    assert mapped["NVIDIA HGX H100"] == "H100"
    assert mapped["NVIDIA HGX H200"] == "H200"
    assert mapped["NVIDIA GH200"] == "GH200"
    assert mapped["NVIDIA L40"] == "L40"
    assert mapped["NVIDIA L40S"] == "L40S"
    assert mapped["NVIDIA A100"] == "A100"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known coreweave labels now unmapped: {unmapped}"
    # The unpriced GB300 row prints nothing today, but the label must map
    # for the day it gains a price.
    assert match_sku(catalog, "NVIDIA GB300 NVL72")["sku"] == "GB300"


# --------------------------------------------------------- synthetic shapes

_ROW_TMPL = (
    '<div role="listitem" class="table-row-v2 w-dyn-item '
    'kubernetes-gpu-pricing"><div class="table-grid">\n'
    '<h3 data-product="{product}" class="table-model-name">{name}</h3>\n'
    '<div class="table-meta-value">\n'
    '<span class="instance-price">On-Demand Price: '
    '<span class="item-value">{od}</span> / Hour<br/></span>\n'
    '<span class="spot-price">Spot Price: '
    '<span class="item-value">{spot}</span> / Hour<br/></span></div>\n'
    '<div class="table-meta-value">{count}</div>\n<div>GPU Count</div>\n'
    '<div class="table-meta-value">{vram}</div>\n<div>VRAM</div>\n'
    "</div>"
)


def _row(
    product="hgx-h100",
    name="NVIDIA HGX H100",
    od="$49.24",
    spot="N/A",
    count="8",
    vram="80",
    extra_markup="",
):
    return (
        _ROW_TMPL.format(
            product=product, name=name, od=od, spot=spot, count=count,
            vram=vram,
        )
        + extra_markup
    )


def _page(na_rows, eu_rows):
    return (
        "<html>On-demand GPU instances "
        "REGION: NORTH AMERICA " + "".join(na_rows)
        + " REGION: EUROPE " + "".join(eu_rows)
        + " On-demand CPU instances REGION: NORTH AMERICA cpu tail</html>"
    )


def test_duplicated_section_anchor_raises():
    body = _page([_row()], [_row()]) + "On-demand GPU instances"
    with pytest.raises(RuntimeError, match="section anchor"):
        parse_coreweave(body)


def test_unexpected_region_heading_raises():
    body = _page([_row()], [_row()]).replace(
        "REGION: EUROPE", "REGION: ASIA PACIFIC"
    )
    with pytest.raises(RuntimeError, match="region heading"):
        parse_coreweave(body)


def test_impure_row_segment_raises():
    """Two different h3 labels inside one split segment mean the row-split
    marker no longer isolates rows -- every boundary is suspect, so the
    parse must refuse outright (not skip one row)."""
    leaked = _row(extra_markup=(
        '<h3 data-product="hgx-h200" class="table-model-name">'
        "NVIDIA HGX H200</h3>"
    ))
    with pytest.raises(RuntimeError, match="multiple model labels"):
        parse_coreweave(_page([leaked], [_row()]))


def test_unpinnable_count_cell_skips_row_into_partial_errors():
    bad = _row(product="mystery", name="NVIDIA MYSTERY", count="N/A")
    rows, errors = parse_coreweave(_page([_row(), bad], [_row()]))
    assert not [r for r in rows if r["sku_identifier"] == "NVIDIA MYSTERY"]
    assert any("MYSTERY" in e and "leading integer" in e for e in errors)


def test_non_dollar_price_is_never_assumed_usd():
    """A span value that is not a plain $ figure (new currency symbol,
    range, text) is a skipped pin plus a partial error -- never recorded
    as USD."""
    euro = _row(
        product="mystery", name="NVIDIA MYSTERY", od="\u20ac42.00",
        spot="N/A",
    )
    rows, errors = parse_coreweave(_page([_row(), euro], [_row()]))
    assert not [r for r in rows if r["sku_identifier"] == "NVIDIA MYSTERY"]
    assert any(
        "MYSTERY" in e and "never" in e and "USD" in e for e in errors
    )


def test_reshaped_model_h3_raises():
    """A model h3 that gains an attribute (or any reshape the strict pin
    no longer matches) must raise, not silently drop its row: every
    'table-model-name' marker in the GPU section is counted against
    strict matches."""
    body = _page([_row()], [_row()]).replace(
        '<h3 data-product="hgx-h100"', '<h3 id="x" data-product="hgx-h100"', 1
    )
    with pytest.raises(RuntimeError, match="strict h3 pin"):
        parse_coreweave(body)


def test_priced_listitem_without_model_h3_raises():
    """A row listitem carrying price markup but no model h3 at all has an
    unpinnable identity -- its price must never vanish without a trace."""
    anon = (
        '<div role="listitem" class="table-row-v2 w-dyn-item anon">'
        "<div>$9.99</div></div>"
    )
    with pytest.raises(RuntimeError, match="identity unpinnable"):
        parse_coreweave(_page([_row(), anon], [_row()]))


def test_new_tier_span_skips_row_loudly():
    """An item-value span the two tier pins + excluded inference span
    cannot account for (a NEW tier appearing) must skip the row into
    partial_errors -- recording only the known tiers would silently
    under-report the surface."""
    bad = _row(
        product="mystery",
        name="NVIDIA MYSTERY",
        extra_markup=(
            '<span class="reserved-price">Reserved Price: '
            '<span class="item-value">$30.00</span> / Hour</span>'
        ),
    )
    rows, errors = parse_coreweave(_page([_row(), bad], [_row()]))
    assert not [r for r in rows if r["sku_identifier"] == "NVIDIA MYSTERY"]
    assert not [r for r in rows if r["raw_value"] == "$30.00"]
    assert any(
        "MYSTERY" in e and "unattributable price span" in e for e in errors
    )


def test_renamed_tier_label_is_loud_not_not_offered():
    """If the page renames 'On-Demand Price:' the pin stops matching --
    without span accounting the row would print spot-only, exactly the
    shape of the page's own not-offered encoding. It must instead skip
    loudly."""
    bad = _row(product="mystery", name="NVIDIA MYSTERY").replace(
        "On-Demand Price:", "On Demand Price:"
    )
    rows, errors = parse_coreweave(_page([_row(), bad], [_row()]))
    assert not [r for r in rows if r["sku_identifier"] == "NVIDIA MYSTERY"]
    assert any(
        "MYSTERY" in e and "unattributable price span" in e for e in errors
    )


def test_all_rows_unpinnable_in_one_region_raises():
    """Row-level skips are honest one by one, but a whole region printing
    zero observations means the pins stopped fitting the table."""
    unpriced = _row(od="", spot="N/A")
    with pytest.raises(RuntimeError, match="zero observations"):
        parse_coreweave(_page([unpriced], [_row()]))
