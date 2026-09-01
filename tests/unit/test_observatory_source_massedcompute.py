# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory massedcompute collector -- fixture pins.

Fixture: tests/fixtures/observatory/massedcompute/pricing_rsc_excerpt.html --
REAL bytes captured live from https://vm.massedcompute.com/pricing on
2026-08-22, spliced down to the keyed row segments (mobile card + desktop
table copies) with every edge case preserved: the desktop contact-us
'Request' rows (H100 80GB x8, L40 48GB x8), the deferred-price stray chunk
('7d:...$$23.36') that sits INSIDE the H100 x8 key-to-key gap (the
price-steal trap), desktop rows whose cells are RSC-deferred refs (B200
SXM6), and the lookalike label families (L40S/L40, RTX 6000 ADA/RTX A6000,
H200 NVL/H100 NVL). Splices were verified byte-parity: the excerpt parse
equals the full-page parse restricted to the kept rows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.massedcompute import SOURCE_ID, parse_massedcompute

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "massedcompute"
    / "pricing_rsc_excerpt.html"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_massedcompute(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def partials(parsed):
    return parsed[1]


def by_row(rows):
    return {(r["sku_identifier"], r["gpu_count_basis"]): r for r in rows}


def test_source_id_matches_module():
    assert SOURCE_ID == "massedcompute"


def test_flagship_price_pins(rows):
    b300 = by_row(rows)[("B300 SXM6", 8)]
    assert b300["price_usd_gpu_hr"] == 6.6
    assert b300["raw_value"] == "52.80"
    assert b300["raw_unit"] == "usd_per_node_hr"
    assert b300["currency"] == "USD"
    assert b300["tier"] == "on-demand"
    assert b300["extra"]["row_variants"] == ["desktop", "mobile"]

    h200 = by_row(rows)[("H200 NVL (141GB)", 1)]
    assert h200["price_usd_gpu_hr"] == 3.62
    assert h200["gpu_count_basis"] == 1


def test_deferred_desktop_cells_fall_back_to_mobile_copy(rows):
    """The desktop B200 SXM6 row defers its price cell to a later RSC chunk
    ('$L31'-style refs) -- only the mobile card carries $$43.46 inline, and
    the print must say so."""
    b200 = by_row(rows)[("B200 SXM6", 8)]
    assert b200["price_usd_gpu_hr"] == 5.4325
    assert b200["raw_value"] == "43.46"
    assert b200["extra"]["row_variants"] == ["mobile"]

    sxm5 = by_row(rows)[("H100 SXM5 (80GB)", 8)]
    assert sxm5["price_usd_gpu_hr"] == 3.14
    assert sxm5["extra"]["row_variants"] == ["mobile"]


def test_contact_us_row_never_steals_deferred_neighbor_price(rows, partials):
    """THE identity pin this source earned: the desktop 'H100 (80GB) x8'
    row publishes 'Request' (no price), but the deferred chunk carrying
    'H100 NVL (94GB) NVLink x8's $$23.36 sits inside its key-to-key gap.
    Without the chunk-line fence that steals as a silent wrong print
    (2.92 vs the true 2.73 per GPU). The fixture keeps those exact bytes."""
    fixture = FIXTURE.read_text(encoding="utf-8")
    key = re.search(r'\\"H100 \(80GB\)-x 8-3\\"', fixture)
    nxt = re.compile(r'\\"[^"\\]{1,60}?-x\s*\d+-\d+\\"').search(
        fixture, key.end()
    )
    assert "$$23.36" in fixture[key.end() : nxt.start()], (
        "fixture lost the stray deferred-price chunk -- the trap is unarmed"
    )

    rows_by = by_row(rows)
    assert ("H100 (80GB)", 8) not in rows_by
    assert rows_by[("H100 (80GB)", 1)]["price_usd_gpu_hr"] == 2.73
    # ...and the stray's true owner still prints, at its own rate.
    nvlink8 = rows_by[("H100 NVL (94GB) NVLink", 8)]
    assert nvlink8["raw_value"] == "23.36"
    assert nvlink8["price_usd_gpu_hr"] == 2.92
    assert any("H100 (80GB) x8" in p for p in partials)


def test_unpriced_request_rows_skip_and_are_noted(rows, partials):
    """L40 (48GB) x8 is 'Request' in BOTH page copies -- no print, one
    honest partial note."""
    rows_by = by_row(rows)
    assert ("L40 (48GB)", 8) not in rows_by
    assert rows_by[("L40 (48GB)", 1)]["price_usd_gpu_hr"] == 0.86
    unpriced_notes = [p for p in partials if "unpriced rows skipped" in p]
    assert len(unpriced_notes) == 1
    assert "L40 (48GB) x8" in unpriced_notes[0]


def test_mobile_desktop_copies_dedupe_to_one_print(rows):
    seen = [(r["sku_identifier"], r["gpu_count_basis"]) for r in rows]
    assert len(seen) == len(set(seen)), "a page copy double-printed"
    both = by_row(rows)[("RTX A6000 (48GB)", 1)]
    assert both["extra"]["row_variants"] == ["desktop", "mobile"]


def test_per_gpu_normalization_reproduces_node_price(rows):
    """price * gpu_count_basis must reproduce the raw node figure, for
    every observation."""
    assert rows, "fixture parsed empty"
    for r in rows:
        node = float(r["raw_value"].replace(",", ""))
        assert round(r["price_usd_gpu_hr"] * r["gpu_count_basis"], 2) == node
        assert r["raw_unit"] == "usd_per_node_hr"
        assert r["currency"] == "USD"
        assert r["region"] == "unspecified"


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["B300 SXM6"] == "B300"
    assert mapped["B200 SXM6"] == "B200"
    # H-series variant split (hourly panel design section 7): NVL labels
    # land on their own variant skus now, never on the generic parts.
    assert mapped["H200 NVL (141GB)"] == "H200_NVL"
    assert mapped["H200 NVL (141GB) NVLink"] == "H200_NVL"
    assert mapped["H100 (80GB)"] == "H100"
    assert mapped["H100 NVL"] == "H100_NVL"
    assert mapped["H100 NVL (94GB) NVLink"] == "H100_NVL"
    assert mapped["H100 SXM5 (80GB)"] == "H100"
    assert mapped["A100 (80GB)"] == "A100"
    assert mapped["A100 SXM4 (80GB)"] == "A100"
    assert mapped["DGX A100 (80GB)"] == "A100"
    # Lookalike families must land on DIFFERENT skus.
    assert mapped["L40S (48GB)"] == "L40S"
    assert mapped["L40 (48GB)"] == "L40"
    assert mapped["RTX 6000 ADA (48GB)"] == "RTX_6000_ADA"
    assert mapped["RTX A6000 (48GB)"] == "RTX_A6000"
    assert mapped["RTX A6000 (48GB) NVLink"] == "RTX_A6000"
    assert mapped["RTX A6000 (48GB) [ALT Config]"] == "RTX_A6000"
    assert mapped["RTX A6000 (48GB) [Premium]"] == "RTX_A6000"
    assert mapped["RTX PRO 6000 Blackwell (96GB)"] == "RTX_PRO_6000"
    assert mapped["RTX PRO 4500 Blackwell (32GB)"] == "RTX_PRO_4500"
    assert mapped["RTX A5000 (24GB)"] == "RTX_A5000"
    assert mapped["A30 (24GB)"] == "A30"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known massedcompute labels now unmapped: {unmapped}"


def test_raises_when_row_keys_vanish():
    with pytest.raises(RuntimeError, match="zero RSC row keys"):
        parse_massedcompute("<html>pricing moved somewhere else</html>")


def test_raises_when_keys_exist_but_no_price_pins():
    body = '\\"H100 (80GB)-x 1-0\\",Request,Contact]\\n'
    with pytest.raises(RuntimeError, match="not one pinnable price"):
        parse_massedcompute(body)


def test_conflicting_page_copies_skip_rather_than_double_print():
    body = (
        '\\"H100 (80GB)-x 1-0\\",td,$$2.73]\\n'
        '\\"H100 (80GB)-mobile-x 1-0\\",card,$$2.99]\\n'
        '\\"A30 (24GB)-x 1-0\\",td,$$0.35]\\n'
    )
    rows, partials = parse_massedcompute(body)
    assert [(r["sku_identifier"], r["price_usd_gpu_hr"]) for r in rows] == [
        ("A30 (24GB)", 0.35)
    ]
    assert any("CONFLICTING" in p and "H100 (80GB) x1" in p for p in partials)


def test_two_prices_in_one_row_window_is_ambiguous_not_a_guess():
    body = (
        '\\"L40 (48GB)-x 1-0\\",td,$$0.99,$$0.86]\\n'
        '\\"A30 (24GB)-x 1-0\\",td,$$0.35]\\n'
    )
    rows, partials = parse_massedcompute(body)
    assert [(r["sku_identifier"], r["price_usd_gpu_hr"]) for r in rows] == [
        ("A30 (24GB)", 0.35)
    ]
    assert any("ambiguous rows skipped" in p and "L40 (48GB) x1" in p
               for p in partials)
