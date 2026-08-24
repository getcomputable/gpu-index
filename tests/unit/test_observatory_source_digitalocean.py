# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory digitalocean collector -- fixture pins (live page 2026-08-22).

Fixture is the full docs GPU Droplet pricing page as fetched live on
2026-08-22 (64KB, server-rendered Hugo): 11 on-demand rows, 8 spot rows,
plus the three exclusion surfaces the pins exist for -- the unpriced
contract table (bare 'NVIDIA B300' label, 'Per contract after contacting
sales' cells), the CPU Droplet section above, and the bandwidth prose
($0.01/GiB) below. Edge-case labels live in the real rows: lowercase
'L40s', Ada-less 'RTX 4000'/'RTX 6000', and the two B300 cooling variants
at identical prices.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.digitalocean import SOURCE_ID, parse_digitalocean

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "digitalocean"
    / "droplets_pricing.html"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_digitalocean(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "digitalocean"


def test_row_counts_and_clean_parse(parsed):
    rows, errors = parsed
    assert errors == []  # both sliced tables parse fully on the live page
    assert len([r for r in rows if r["tier"] == "on-demand"]) == 11
    assert len([r for r in rows if r["tier"] == "spot"]) == 8
    assert {r["tier"] for r in rows} == {"on-demand", "spot"}


def test_on_demand_pins(rows):
    by_id = {
        r["sku_identifier"]: r for r in rows if r["tier"] == "on-demand"
    }
    h100 = by_id["NVIDIA H100"]
    assert h100["price_usd_gpu_hr"] == 4.41
    assert h100["gpu_count_basis"] == 1
    assert h100["raw_value"] == "$4.41 per hour"
    assert h100["currency"] == "USD"
    h200_node = by_id["NVIDIA H200 (8x)"]
    assert h200_node["gpu_count_basis"] == 8
    assert h200_node["raw_value"] == "$35.76 per hour"
    assert h200_node["price_usd_gpu_hr"] == 4.47  # 35.76 / 8
    assert by_id["NVIDIA L40s"]["price_usd_gpu_hr"] == 1.57
    assert by_id["AMD MI300X"]["price_usd_gpu_hr"] == 2.59
    assert by_id["NVIDIA RTX 4000"]["price_usd_gpu_hr"] == 0.76


def test_spot_pins_including_b300_cooling_variants(rows):
    """The widely quoted $11.19 B300 is the SPOT tier; the two cooling
    variants are distinct rows at the same price and both must record."""
    spot = {r["sku_identifier"]: r for r in rows if r["tier"] == "spot"}
    assert spot["NVIDIA B300, air-cooled"]["price_usd_gpu_hr"] == 11.19
    assert spot["NVIDIA B300, liquid-cooled"]["price_usd_gpu_hr"] == 11.19
    node = spot["NVIDIA B300, air-cooled (8x)"]
    assert node["raw_value"] == "$89.52 per hour"
    assert node["gpu_count_basis"] == 8
    assert node["price_usd_gpu_hr"] == 11.19
    assert spot["AMD MI350X"]["price_usd_gpu_hr"] == 4.0
    assert spot["AMD MI355X (8x)"]["price_usd_gpu_hr"] == 4.5  # 36.00 / 8


def test_contract_table_and_off_slice_prices_never_recorded(rows):
    """The contract table's bare 'NVIDIA B300' label exists ONLY there --
    it must never print (its cells are unpriced by design), and nothing
    from the CPU/bandwidth surfaces may leak in."""
    identifiers = {r["sku_identifier"] for r in rows}
    assert "NVIDIA B300" not in identifiers
    assert all(
        r["extra"]["section_anchor"]
        in ("on-demand-gpu-droplet-pricing", "spot-gpu-droplet-pricing")
        for r in rows
    )
    assert all("contract" not in r["raw_value"].lower() for r in rows)


def test_per_gpu_price_times_basis_reproduces_published_figure(rows):
    for r in rows:
        published = float(r["raw_value"].split(" ")[0].lstrip("$"))
        assert round(r["price_native_per_gpu_hr"] * r["gpu_count_basis"], 2) == published


def _row(label, price):
    return f"<tr><td>{label}</td><td>{price}</td></tr>"


def _page(on_demand_body, spot_body):
    def section(anchor, title, body):
        return (
            f'<h2 id="{anchor}-pricing">{title}</h2>\n<div><table>\n'
            "<thead><tr><th>GPU</th><th>Price</th></tr></thead>\n"
            f"<tbody>{body}</tbody>\n</table></div>\n"
            f'<h3 id="{anchor}-billing">Billing</h3>\n'
        )

    return (
        section("on-demand-gpu-droplet", "On-Demand", on_demand_body)
        + section("spot-gpu-droplet", "Spot", spot_body)
        + '<h2 id="contract-gpu-droplet-pricing">Contract</h2>\n'
        '<div><table><thead><tr><th>GPU</th><th>Price</th></tr></thead>'
        "<tbody><tr><td>NVIDIA B300</td><td>Per contract after "
        '<a href="https://example.invalid">contacting sales</a></td></tr>'
        "</tbody></table></div>\n"
        '<h3 id="contract-gpu-droplet-billing">Billing</h3>'
    )


_OK_ROW = _row("NVIDIA H100", "$4.41 per hour")
_OK_SPOT = _row("NVIDIA B300, air-cooled", "$11.19 per hour")


def test_unknown_vendor_and_unpriced_rows_skip_and_count():
    body = _page(
        _OK_ROW
        + _row("Intel Gaudi 3", "$1.00 per hour")
        + _row("AMD MI300X", "Per contract after contacting sales"),
        _OK_SPOT,
    )
    rows, errors = parse_digitalocean(body)
    assert [r["sku_identifier"] for r in rows] == [
        "NVIDIA H100",
        "NVIDIA B300, air-cooled",
    ]
    assert len(errors) == 2
    assert "vendor prefix" in errors[0] and "Intel Gaudi 3" in errors[0]
    assert "unpriced" in errors[1] and "AMD MI300X" in errors[1]


def test_count_token_off_label_end_skips_and_counts():
    """A footnote marker (or renamed suffix) after '(8x)' must never
    default the basis to 1 -- that would record the whole-droplet price
    as a per-GPU price, 8x too high. Skip + count instead."""
    for drifted in ("NVIDIA H200 (8x)*", "NVIDIA H200 (8x) SXM"):
        page = _page(
            _OK_ROW + _row(drifted, "$35.76 per hour"),
            _OK_SPOT,
        )
        rows, errors = parse_digitalocean(page)
        assert [r["sku_identifier"] for r in rows] == [
            "NVIDIA H100",
            "NVIDIA B300, air-cooled",
        ]
        assert len(errors) == 1
        assert "not label-final" in errors[0] and drifted in errors[0]


def test_digit_bearing_price_that_misses_the_pin_raises():
    # \u20ac = euro sign, escape-coded to keep the source ASCII-only.
    for bad in ("$4.41 per month", "\u20ac4.41 per hour", "4.41 per hour"):
        page = _page(_OK_ROW + _row("NVIDIA H200", bad), _OK_SPOT)
        with pytest.raises(RuntimeError, match="refusing to guess"):
            parse_digitalocean(page)


def test_row_with_wrong_cell_count_raises():
    page = _page(
        _OK_ROW + "<tr><td>NVIDIA H200</td><td>x</td><td>y</td></tr>",
        _OK_SPOT,
    )
    with pytest.raises(RuntimeError, match="binding suspect"):
        parse_digitalocean(page)


def test_section_yielding_zero_observations_raises():
    page = _page(_row("Intel Gaudi 3", "$1.00 per hour"), _OK_SPOT)
    with pytest.raises(RuntimeError, match="zero observations"):
        parse_digitalocean(page)


def test_missing_or_duplicated_section_anchor_raises():
    body = FIXTURE.read_text()
    with pytest.raises(RuntimeError, match="appears 0x"):
        parse_digitalocean(
            body.replace('<h2 id="spot-gpu-droplet-pricing">', "<h2>")
        )
    with pytest.raises(RuntimeError, match="appears 2x"):
        parse_digitalocean(
            body + '<h2 id="on-demand-gpu-droplet-pricing">dup</h2>'
        )


def test_slice_swallowing_a_second_table_raises():
    """If the headings fencing the spot section off the contract table
    vanish, the slice would hold two tables -- must refuse, because the
    second is the unpriced contract table."""
    body = (
        FIXTURE.read_text()
        .replace('<h3 id="spot-gpu-droplet-billing">', "<div>")
        .replace('<h2 id="contract-gpu-droplet-pricing">', "<div>")
    )
    with pytest.raises(RuntimeError, match="2 tables"):
        parse_digitalocean(body)


def test_header_drift_raises():
    body = FIXTURE.read_text().replace(">GPU</th>", ">Chip</th>")
    with pytest.raises(RuntimeError, match="header cells"):
        parse_digitalocean(body)


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA H100"] == "H100"
    assert mapped["NVIDIA H200 (8x)"] == "H200"
    assert mapped["NVIDIA L40s"] == "L40S"  # lowercase-s label
    assert mapped["AMD MI300X (8x)"] == "MI300X"
    assert mapped["AMD MI325X"] == "MI325X"
    assert mapped["AMD MI355X"] == "MI355X"
    assert mapped["NVIDIA B300, air-cooled"] == "B300"
    assert mapped["NVIDIA B300, liquid-cooled (8x)"] == "B300"
    # Catalog ruling: bare generation-less labels ('RTX 6000', 'RTX 4000')
    # map to their own AMBIGUOUS bucket skus rather than being folded into
    # either generation (DO's marketing says Ada; a marketplace bare label
    # is usually the Quadro-era part) -- the raw sku_identifier stays the
    # record either way. This pin fires if a catalog reorder silently
    # reassigns them to a generation.
    assert mapped["NVIDIA RTX 6000"] == "RTX_6000"
    assert mapped["NVIDIA RTX 4000"] == "RTX_4000"
    assert mapped["AMD MI350X"] == "MI350X"
    assert mapped["AMD MI350X (8x)"] == "MI350X"
    unmapped = {k for k, v in mapped.items() if v is None}
    assert not unmapped, f"digitalocean labels now unmapped: {sorted(unmapped)}"
