# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory hyperstack collector -- fixture pins (live page 2026-08-22).

Fixture: tests/fixtures/observatory/hyperstack/gpu-pricing.html -- the live
https://www.hyperstack.cloud/gpu-pricing document captured 2026-08-22,
trimmed by stripping script/style/noscript/svg bodies and comments only
(parse output verified identical to the untrimmed page at capture time).
It preserves all three price tables (on-demand 13 rows / reservation 12 /
spot 5), the CPU-pricing table that end-fences the spot slice, the 'B300s
are coming' banner, and the priceless nav/footer lookalike labels
(GB200 NVL72, GB300 NVL72, HGX B200) that must never print.

House style: (1) parse the recorded fixture, (2) pin exact prints incl.
this source's edge cases (spec-less reserved/spot rows, lookalike labels,
unpriced rows), (3) prove the framework normalization maps the real
labels, (4) prove the fail-closed anchor fences raise on reshape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.hyperstack import SOURCE_ID, parse_hyperstack

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "hyperstack"
    / "gpu-pricing.html"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_hyperstack(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def _one(rows, identifier, tier):
    matched = [
        r
        for r in rows
        if r["sku_identifier"] == identifier and r["tier"] == tier
    ]
    assert len(matched) == 1, f"expected exactly one {identifier} {tier} row"
    return matched[0]


# Synthetic minimal page for the fail-closed fence tests: the pipe
# transform turns whitespace into '|', so plain text stands in for markup.
_SYNTH_OD_HEADER = (
    "On-Demand GPU Pricing GPU Model VRAM (GB) Max pCPUs per GPU "
    "Max RAM (GB) per GPU Pricing Per Hour"
)
_SYNTH_RESERVED_HEADER = "Reservation Pricing GPU Model Starting from Reserve"
_SYNTH_SPOT_HEADER = "Spot VM Pricing Learn More GPU Model Spot VM"
_SYNTH_CPU_HEADER = "On-Demand CPU Pricing"


def _synth(
    od_rows="NVIDIA H100 80 28 180 $2.50",
    reserved_rows="NVIDIA H100 $1.75 Reserve here",
    spot_rows="NVIDIA H100 PCIe $2.00",
):
    return " ".join(
        [
            _SYNTH_OD_HEADER,
            od_rows,
            _SYNTH_RESERVED_HEADER,
            reserved_rows,
            _SYNTH_SPOT_HEADER,
            spot_rows,
            _SYNTH_CPU_HEADER,
        ]
    )


def test_source_id_matches_module():
    assert SOURCE_ID == "hyperstack"


def test_tier_counts_and_no_partials(parsed):
    rows, partial_errors = parsed
    assert partial_errors == []
    tiers = {t: sum(1 for r in rows if r["tier"] == t) for t in
             ("on-demand", "reserved", "spot")}
    assert tiers == {"on-demand": 13, "reserved": 12, "spot": 5}
    assert len(rows) == 30


def test_on_demand_pins(rows):
    h200 = _one(rows, "NVIDIA H200 SXM", "on-demand")
    assert h200["price_usd_gpu_hr"] == 3.99
    assert h200["raw_value"] == "$3.99"
    assert h200["memory_gb_label"] == 141
    assert h200["extra"]["vram_gb"] == 141
    assert h200["extra"]["max_pcpus_per_gpu"] == 22
    assert h200["extra"]["max_ram_gb_per_gpu"] == 225
    assert h200["extra"]["table"] == "on_demand_gpu_pricing"
    assert _one(rows, "NVIDIA B200", "on-demand")["price_usd_gpu_hr"] == 6.00
    a4000 = _one(rows, "NVIDIA A4000", "on-demand")
    assert a4000["price_usd_gpu_hr"] == 0.15
    assert a4000["memory_gb_label"] == 16


def test_b300_on_demand_carries_forward_listed_banner_note(rows):
    b300 = _one(rows, "NVIDIA B300", "on-demand")
    assert b300["price_usd_gpu_hr"] == 7.40
    assert b300["memory_gb_label"] == 288
    # The fixture page still shows the 'B300s are coming' banner -- the
    # priced row may be forward-listed and the note must say so.
    assert "forward-listed" in b300["notes"]


def test_reserved_pins(rows):
    h200 = _one(rows, "NVIDIA H200 SXM", "reserved")
    assert h200["price_usd_gpu_hr"] == 2.79
    assert h200["raw_value"] == "$2.79"
    assert h200["extra"]["pricing_basis"] == "starting_from"
    assert h200["extra"]["table"] == "reservation_pricing"
    assert "Starting from" in h200["notes"]
    assert _one(rows, "NVIDIA B200", "reserved")["price_usd_gpu_hr"] == 5.10
    assert _one(rows, "NVIDIA A4000", "reserved")["price_usd_gpu_hr"] == 0.11
    # B300 has no reservation row on the page -- must not be invented.
    assert not [
        r
        for r in rows
        if r["tier"] == "reserved" and r["sku_identifier"] == "NVIDIA B300"
    ]


def test_spot_pins(rows):
    h100_pcie = _one(rows, "NVIDIA H100 PCIe", "spot")
    assert h100_pcie["price_usd_gpu_hr"] == 2.00
    assert h100_pcie["extra"]["table"] == "spot_vm_pricing"
    assert _one(rows, "NVIDIA A6000", "spot")["price_usd_gpu_hr"] == 0.40


def test_raw_value_reproduces_price_per_gpu(rows):
    for r in rows:
        assert r["currency"] == "USD"
        assert r["gpu_count_basis"] == 1
        raw = float(r["raw_value"].lstrip("$").replace(",", ""))
        assert r["price_usd_gpu_hr"] * r["gpu_count_basis"] == raw


def test_lookalike_nav_and_cpu_rows_never_print(rows):
    """Nav/footer carry priceless GB200 NVL72 / GB300 NVL72 / HGX B200
    labels and the CPU table carries priced non-GPU flavors -- none may
    surface as observations."""
    identifiers = {r["sku_identifier"] for r in rows}
    for fragment in ("NVL72", "GB200", "GB300", "HGX"):
        assert not [i for i in identifiers if fragment in i], fragment
    assert all(i.startswith("NVIDIA ") for i in identifiers)
    assert not [i for i in identifiers if "cpu" in i.lower()]


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA H200 SXM"] == "H200"
    assert mapped["NVIDIA H100 SXM"] == "H100"
    assert mapped["NVIDIA H100 NVLink"] == "H100"
    assert mapped["NVIDIA H100 PCIe"] == "H100"
    assert mapped["NVIDIA RTX Pro 6000 SE"] == "RTX_PRO_6000"
    assert mapped["NVIDIA A100 SXM"] == "A100"
    assert mapped["NVIDIA A6000"] == "RTX_A6000"
    assert mapped["NVIDIA A4000"] == "RTX_A4000"
    assert mapped["NVIDIA L40"] == "L40"
    assert mapped["NVIDIA B200"] == "B200"
    assert mapped["NVIDIA B300"] == "B300"
    # This pin is about the KNOWN labels staying mapped -- a genuinely new
    # chip appearing live records unmapped and the capture warns.
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known hyperstack labels now unmapped: {unmapped}"


def test_synth_page_parses_all_three_tiers():
    rows, partial_errors = parse_hyperstack(_synth())
    assert partial_errors == []
    assert [(r["tier"], r["sku_identifier"], r["price_usd_gpu_hr"])
            for r in rows] == [
        ("on-demand", "NVIDIA H100", 2.50),
        ("reserved", "NVIDIA H100", 1.75),
        ("spot", "NVIDIA H100 PCIe", 2.00),
    ]


def test_missing_section_anchor_raises():
    page = _synth().replace("Reservation Pricing", "Reservation Prices")
    with pytest.raises(RuntimeError, match="reservation table header"):
        parse_hyperstack(page)


def test_renamed_on_demand_column_raises():
    # nebius-style header guard: ANY column change must raise, never
    # misattribute which number is the price.
    page = _synth().replace("Pricing Per Hour", "Pricing Per Month")
    with pytest.raises(RuntimeError, match="on-demand table header"):
        parse_hyperstack(page)


def test_duplicate_table_render_raises():
    page = _synth() + " " + _SYNTH_OD_HEADER + " NVIDIA H100 80 28 180 $2.50"
    with pytest.raises(RuntimeError, match="more than once"):
        parse_hyperstack(page)


def test_unpriced_spot_row_is_skipped_and_counted():
    rows, partial_errors = parse_hyperstack(
        _synth(spot_rows="NVIDIA B300 Coming soon NVIDIA L40 $0.80")
    )
    spot = [r for r in rows if r["tier"] == "spot"]
    assert [(r["sku_identifier"], r["price_usd_gpu_hr"]) for r in spot] == [
        ("NVIDIA L40", 0.80)
    ]
    # The unpriced B300 row must not be glued onto its neighbour's price.
    assert not [r for r in rows if "Coming" in r["sku_identifier"]]
    assert partial_errors == [
        "spot table: 1 of 2 rows had no pinnable price cell -- skipped, "
        "not guessed"
    ]


def test_spec_less_row_never_matches_on_demand_fence():
    """The prior-art exactly-3-numeric-spec-fields fence: a priced row
    WITHOUT the per-GPU spec columns (the reserved-row shape) must never
    print from the on-demand table."""
    rows, partial_errors = parse_hyperstack(
        _synth(od_rows="NVIDIA H300 $9.99 NVIDIA H100 80 28 180 $2.50")
    )
    od = [r for r in rows if r["tier"] == "on-demand"]
    assert [(r["sku_identifier"], r["price_usd_gpu_hr"]) for r in od] == [
        ("NVIDIA H100", 2.50)
    ]
    # The unfenced row is priced AND vendor-celled, so BOTH tallies fire.
    assert partial_errors == [
        "on-demand table: 1 of 2 rows had no pinnable price cell -- "
        "skipped, not guessed",
        "on-demand table: 1 of 2 price cells matched no pinnable NVIDIA "
        "row -- new vendor or reshaped row skipped, not guessed",
    ]


def test_reserved_row_requires_reserve_here_fence():
    rows, partial_errors = parse_hyperstack(
        _synth(
            reserved_rows=(
                "NVIDIA H300 $9.99 NVIDIA H100 $1.75 Reserve here"
            )
        )
    )
    reserved = [r for r in rows if r["tier"] == "reserved"]
    assert [
        (r["sku_identifier"], r["price_usd_gpu_hr"]) for r in reserved
    ] == [("NVIDIA H100", 1.75)]
    # The unfenced row is priced AND vendor-celled, so BOTH tallies fire.
    assert partial_errors == [
        "reserved table: 1 of 2 rows had no pinnable price cell -- "
        "skipped, not guessed",
        "reserved table: 1 of 2 price cells matched no pinnable NVIDIA "
        "row -- new vendor or reshaped row skipped, not guessed",
    ]


def test_section_with_zero_pinnable_rows_raises():
    with pytest.raises(RuntimeError, match="spot table header present"):
        parse_hyperstack(_synth(spot_rows="no gpu rows here today"))


def test_foreign_vendor_priced_row_skips_loudly():
    """A priced row whose vendor cell is not NVIDIA (a new vendor appearing
    on the page) is invisible to the row-start tally -- the price-cell
    tally must surface it in partial_errors, never drop it silently."""
    rows, partial_errors = parse_hyperstack(
        _synth(spot_rows="AMD MI300X $1.99 NVIDIA H100 PCIe $2.00")
    )
    spot = [r for r in rows if r["tier"] == "spot"]
    assert [(r["sku_identifier"], r["price_usd_gpu_hr"]) for r in spot] == [
        ("NVIDIA H100 PCIe", 2.00)
    ]
    assert not [r for r in rows if "MI300X" in r["sku_identifier"]]
    assert partial_errors == [
        "spot table: 1 of 2 price cells matched no pinnable NVIDIA row -- "
        "new vendor or reshaped row skipped, not guessed"
    ]
