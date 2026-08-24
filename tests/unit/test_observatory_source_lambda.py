# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory lambda collector -- fixture pins (live page 2026-08-22).

Fixture: real bytes of https://lambda.ai/pricing trimmed to the two pricing
islands plus the window.__islands script blobs that carry unicode-escaped
duplicates of every table (the escaped-duplicate trap stays live in the
fixture). Edge cases preserved: unpriced em-dash '1 year+' committed rows,
'256+' GPU-count labels, two 'NVIDIA A100 SXM' rows in one tab differing
only by VRAM, and the committed 'NVIDIA H100'/'NVIDIA HGX B200' lookalikes
of the instance labels.

The module lives at observatory/sources/lambda.py -- 'lambda' is a Python
keyword, so imports go through importlib (the registry does the same).

Source stays ASCII-only: the page's en-dash (U+2013 in DURATION cells) and
em-dash (U+2014 unpriced price cells) are referenced via chr() codepoints.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku

lambda_mod = importlib.import_module("gpu_index.observatory.sources.lambda")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "lambda" / "pricing.html"
)

EN_DASH = chr(0x2013)
EM_DASH = chr(0x2014)


@pytest.fixture(scope="module")
def parsed():
    return lambda_mod.parse_lambda_pricing(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def partial_errors(parsed):
    return parsed[1]


def test_source_id_matches_module():
    assert lambda_mod.SOURCE_ID == "lambda"


def test_row_totals_and_tiers(rows):
    """22 instance rows + 6 priced committed rows; nothing else."""
    assert len(rows) == 28
    assert sum(1 for r in rows if r["tier"] == "on-demand") == 22
    assert sum(1 for r in rows if r["tier"] == "reserved") == 6


def test_same_plan_priced_differently_per_tab_size(rows):
    """The page's core trap: one plan string, four prices -- the tab label
    (config GPU count) is identity and must ride as gpu_count_basis."""
    b200 = {
        r["gpu_count_basis"]: r["price_usd_gpu_hr"]
        for r in rows
        if r["sku_identifier"] == "NVIDIA B200 SXM6"
    }
    assert b200 == {8: 6.69, 4: 6.79, 2: 6.89, 1: 6.99}
    one_x = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA B200 SXM6"
        and r["gpu_count_basis"] == 1
    )
    assert one_x["raw_value"] == "$6.99"
    assert one_x["extra"]["tab"] == "1x"
    assert one_x["memory_gb_label"] == 180


def test_duplicate_plan_rows_distinguished_by_vram(rows):
    """Two 'NVIDIA A100 SXM' rows in the 8x tab (80 GB vs 40 GB) -- both
    must print, distinguished by the memory label and specs."""
    a100_8x = [
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA A100 SXM" and r["gpu_count_basis"] == 8
    ]
    by_vram = {r["memory_gb_label"]: r["price_usd_gpu_hr"] for r in a100_8x}
    assert by_vram == {80: 2.79, 40: 1.99}


def test_known_instance_pins(rows):
    def one(identifier, basis):
        matches = [
            r
            for r in rows
            if r["sku_identifier"] == identifier
            and r["gpu_count_basis"] == basis
        ]
        assert len(matches) == 1, (identifier, basis, matches)
        return matches[0]

    v100 = one("NVIDIA Tesla V100", 8)
    assert v100["price_usd_gpu_hr"] == 0.79
    assert v100["extra"]["storage"] == "5.8 TiB SSD"
    gh200 = one("NVIDIA GH200", 1)
    assert gh200["price_usd_gpu_hr"] == 2.29
    assert gh200["memory_gb_label"] == 96
    quadro = one("NVIDIA Quadro RTX 6000", 1)
    assert quadro["price_usd_gpu_hr"] == 0.69
    assert all(r["currency"] == "USD" for r in rows)


def test_committed_cluster_tiers_recorded_as_reserved(rows):
    hgx = {
        r["gpu_count_basis"]: r["price_usd_gpu_hr"]
        for r in rows
        if r["sku_identifier"] == "NVIDIA HGX B200"
    }
    assert hgx == {16: 9.86, 64: 9.36, 256: 8.87}
    h100 = {
        r["gpu_count_basis"]: r["price_usd_gpu_hr"]
        for r in rows
        if r["sku_identifier"] == "NVIDIA H100"
    }
    assert h100 == {16: 6.16, 64: 5.85, 256: 5.54}
    big = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA HGX B200"
        and r["gpu_count_basis"] == 256
    )
    # '256+' is the provider's own count label -- basis carries the numeric
    # floor, extra keeps the exact string.
    assert big["extra"]["gpu_count_label"] == "256+"
    assert big["extra"]["duration"] == "2 weeks " + EN_DASH + " 1 year"
    assert all(
        r["tier"] == "reserved"
        for r in rows
        if r["sku_identifier"] in ("NVIDIA HGX B200", "NVIDIA H100")
    )


def test_unpriced_committed_rows_skipped_and_counted(rows, partial_errors):
    """Both '1 year+' rows carry an em-dash price cell -- skipped, counted,
    never printed as $0 or guessed."""
    assert len(partial_errors) == 2
    for err in partial_errors:
        assert "skipped unpriced row" in err
        assert "1 year+" in err
        assert EM_DASH in err  # the unpriced cell text, quoted verbatim
    assert all(r["price_usd_gpu_hr"] is not None for r in rows)


def test_escaped_island_duplicates_not_double_counted(rows):
    """Every table also exists unicode-escaped inside window.__islands; the
    fixture keeps those bytes so this trap stays armed. 30 plain rows + 30
    escaped copies must parse to exactly 30 attributed rows (28 priced +
    2 unpriced skips), never 60."""
    text = FIXTURE.read_text()
    assert len(lambda_mod._ROW_START_RE.findall(text)) == 30
    assert text.count('data-plan=\\"') == 30
    assert len(rows) == 28


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {
        "NVIDIA B200 SXM6": "B200",
        "NVIDIA HGX B200": "B200",
        "NVIDIA H100 SXM": "H100",
        "NVIDIA H100 PCIe": "H100",
        "NVIDIA H100": "H100",
        "NVIDIA A100 SXM": "A100",
        "NVIDIA A100 PCIe": "A100",
        "NVIDIA GH200": "GH200",
        "NVIDIA Tesla V100": "V100",
        "NVIDIA A10": "A10",
        "NVIDIA A6000": "RTX_A6000",
        "NVIDIA Quadro RTX 6000": "RTX_6000_QUADRO",
    }
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known lambda labels now unmapped: {unmapped}"


def test_missing_anchor_raises():
    text = FIXTURE.read_text().replace(
        ">Instances pricing</h2>", ">Instance pricing</h2>"
    )
    with pytest.raises(RuntimeError, match="Instances pricing"):
        lambda_mod.parse_lambda_pricing(text)


def test_reshaped_price_cell_raises_never_guesses():
    """A digit-bearing price cell that misses the exact $D.DD pin must
    raise (currency/format honesty), not record."""
    text = FIXTURE.read_text().replace("$6.69", "6.69 USD", 1)
    with pytest.raises(RuntimeError, match="refusing to guess"):
        lambda_mod.parse_lambda_pricing(text)


def test_plain_row_in_trailing_text_raises_not_last_tab():
    """A plain data-plan row materializing AFTER the last panel's table
    (where the escaped window.__islands copies live) must trip the
    row-count cross-check -- never ride the last tab with its basis."""
    text = FIXTURE.read_text()
    row = re.search(
        r'<tr[^>]*data-plan="NVIDIA B200 SXM6"[^>]*>.*?</tr>', text, re.DOTALL
    ).group(0)
    with pytest.raises(RuntimeError, match="rows outside the tabbed islands"):
        lambda_mod.parse_lambda_pricing(text + row)


def test_duplicate_cell_label_raises_never_picks_one():
    """A duplicated PRICE cell in one row (was/now promo pair) must raise,
    never silently record whichever cell comes last."""
    text = FIXTURE.read_text()
    target = '<td data-label="PRICE/GPU/HR*">$6.69</td>'
    assert target in text
    dup = target + '<td data-label="PRICE/GPU/HR*">$9.99</td>'
    with pytest.raises(RuntimeError, match="duplicate .* cell in one row"):
        lambda_mod.parse_lambda_pricing(text.replace(target, dup, 1))


def test_reshaped_tab_label_raises():
    """The instances tab label is the only honest source of the config GPU
    count -- a reshaped label must raise, never default."""
    text = FIXTURE.read_text().replace(">8x</button>", ">8 GPUs</button>", 1)
    with pytest.raises(RuntimeError, match="cannot derive the config GPU count"):
        lambda_mod.parse_lambda_pricing(text)
