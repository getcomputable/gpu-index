# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory e2e collector - fixture pins (live page captured 2026-08-22).

Fixture (tests/fixtures/observatory/e2e/pricing_us.html) = three REAL
excerpts of https://www.e2enetworks.com/pricing?region=us fetched
2026-08-22: (a) the document head with both application/ld+json scripts,
the one USD pricing <table> and the per-GPU-hour footnote; (b, c) two
flight-payload slices carrying the ESCAPED lookalikes - nav GPU labels and
a full escaped copy of the JSON-LD offer catalog - that the identity pins
must ignore. House style per the runpod exemplar: (1) parse the fixture,
(2) pin exact prints incl. this source's edge cases (three commit tiers,
INR JSON-LD offers recorded natively, same-label A100 80/40GB rows, CPU
instance offers excluded), (3) prove real labels normalize through the
catalog. Synthetic-HTML tests cover the fail-closed paths the live page
does not currently exhibit (contact-us cells, column reorder, INR table
cells, reshapes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.e2e import SOURCE_ID, parse_e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "e2e"
    / "pricing_us.html"
)

RUPEE = "\u20b9"

_HEADER = (
    "Item",
    "vRAM",
    "vCPUs",
    "RAM, GB",
    "Hourly/On-Demand",
    "Monthly",
    "Annually",
)

# A well-formed offer catalog for synthetic pages, so the JSON-LD surface
# does not add a zero-offers partial error to table-focused tests.
_LD_STUB = (
    '<script type="application/ld+json">{"@type":"OfferCatalog",'
    '"itemListElement":[{"@type":"Offer","name":"NVIDIA H100 GPU Cloud '
    'Instance","category":"GPU Cloud Instance","priceCurrency":"INR",'
    '"price":362,"priceSpecification":{"@type":"UnitPriceSpecification",'
    '"price":362,"priceCurrency":"INR","unitText":"per hour"},'
    '"itemOffered":{"@type":"Product","category":"GPU Cloud Instance",'
    '"description":"80 GB VRAM, 26 vCPUs, 250 GB RAM"}}]}</script>'
)

_GOOD_ROW = ("NVIDIA H100", "80", "26", "250", "$3.77", "$2,363.01", "$23,914.80")


# The real page's basis footnote (right under the table) - the pin that
# fences the per-GPU claim against the table's 1x/2x/4x/8x multiplier.
_FOOTNOTE = "<p>Prices are per GPU-hour and do not include taxes.</p>"


def _page(rows, header=_HEADER, ld=_LD_STUB, tables=1, footnote=_FOOTNOTE):
    def tr(cells, tag):
        inner = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
        return f"<tr>{inner}</tr>"

    body = tr(header, "th") + "".join(tr(r, "td") for r in rows)
    table = f"<table>{body}</table>" * tables
    return f"<html><head>{ld}</head><body>{table}{footnote}</body></html>"


@pytest.fixture(scope="module")
def parsed():
    return parse_e2e(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "e2e"


def test_fixture_parses_clean(parsed):
    rows, partial_errors = parsed
    assert partial_errors == []
    assert len(rows) == 40  # 10 table rows x 3 tiers + 10 JSON-LD offers


def test_usd_table_covers_every_gpu_row(rows):
    table = [r for r in rows if r["extra"]["surface"] == "usd_table"]
    assert len(table) == 30
    assert {r["sku_identifier"] for r in table} == {
        "NVIDIA B200",
        "NVIDIA H200",
        "NVIDIA H100",
        "NVIDIA RTX PRO 6000",
        "NVIDIA A100",
        "NVIDIA A40",
        "NVIDIA L40S",
        "NVIDIA A30",
        "NVIDIA L4",
    }
    assert all(r["currency"] == "USD" for r in table)
    assert {r["tier"] for r in table} == {
        "on-demand",
        "monthly-commit",
        "reserved",
    }


def test_b200_three_tiers_exact(rows):
    b200 = {
        r["tier"]: r
        for r in rows
        if r["sku_identifier"] == "NVIDIA B200"
        and r["extra"]["surface"] == "usd_table"
    }
    assert b200["on-demand"]["price_usd_gpu_hr"] == 6.99
    assert b200["on-demand"]["raw_value"] == "$6.99"
    assert b200["on-demand"]["raw_unit"] == "usd_per_gpu_hr"
    # Monthly/annual keep the published figure verbatim; the normalized
    # price must reproduce it through the stated hours convention.
    assert b200["monthly-commit"]["raw_value"] == "$4,983.16"
    assert b200["monthly-commit"]["raw_unit"] == "usd_per_gpu_month"
    assert b200["monthly-commit"]["price_usd_gpu_hr"] == round(
        4983.16 / 730.0, 4
    )
    assert b200["reserved"]["raw_value"] == "$55,089.45"
    assert b200["reserved"]["raw_unit"] == "usd_per_gpu_year"
    assert b200["reserved"]["price_usd_gpu_hr"] == round(55089.45 / 8760.0, 4)
    assert all(r["memory_gb_label"] == 192 for r in b200.values())


def test_cheapest_row_pinned_l4(rows):
    l4 = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA L4" and r["tier"] == "on-demand"
    )
    assert l4["price_usd_gpu_hr"] == 0.57
    assert l4["memory_gb_label"] == 24


def test_same_label_a100_rows_disambiguated_by_vram(rows):
    a100 = [
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA A100" and r["tier"] == "on-demand"
    ]
    assert {(r["memory_gb_label"], r["price_usd_gpu_hr"]) for r in a100} == {
        (80, 2.10),
        (40, 1.98),
    }


def test_jsonld_inr_offers_recorded_natively(rows):
    ld = [r for r in rows if r["extra"]["surface"] == "jsonld_offer_catalog"]
    assert len(ld) == 10
    assert all(r["currency"] == "INR" for r in ld)
    assert all(r["price_usd_gpu_hr"] is None for r in ld)
    assert all(r["raw_unit"] == "inr_per_gpu_hr" for r in ld)
    b200 = [
        r for r in ld if r["sku_identifier"] == "NVIDIA B200 GPU Cloud Instance"
    ]
    # Exactly ONE print: the flight payload's escaped copy of the whole
    # offer catalog (present in the fixture) must never double-record.
    assert len(b200) == 1
    assert b200[0]["price_native_per_gpu_hr"] == 671.0
    assert b200[0]["raw_value"] == "671"
    assert b200[0]["memory_gb_label"] == 192
    assert (
        b200[0]["extra"]["url"] == "https://www.e2enetworks.com/gpus/nvidia-b200"
    )


def test_cpu_instances_and_flight_lookalikes_excluded(rows):
    # The JSON-LD catalog also lists C3/M3/SDC3/E1LC CPU instances, and the
    # fixture's flight slices carry escaped "NVIDIA B200" nav labels: none
    # of those may print.
    assert all("CPU Instance" not in r["sku_identifier"] for r in rows)
    b200_prints = [r for r in rows if "B200" in r["sku_identifier"]]
    assert len(b200_prints) == 4  # 3 table tiers + 1 JSON-LD offer


def test_contact_us_cell_skips_tier_not_row():
    rows, errs = parse_e2e(
        _page(
            [
                ("NVIDIA H100", "80", "26", "250", "$2.00", "Contact us", "$12,000.00"),
            ]
        )
    )
    table = [r for r in rows if r["extra"]["surface"] == "usd_table"]
    assert {r["tier"] for r in table} == {"on-demand", "reserved"}
    assert any("'Contact us'" in e and "monthly-commit" in e for e in errs)


def test_column_reorder_fails_whole_row():
    swapped = ("NVIDIA H200", "141", "30", "375", "$2,837.51", "$4.54", "$29,608.80")
    rows, errs = parse_e2e(_page([_GOOD_ROW, swapped]))
    assert all(r["sku_identifier"] != "NVIDIA H200" for r in rows)
    assert any(
        "NVIDIA H200" in e and "cross-check" in e for e in errs
    ), errs
    # the well-formed sibling row still records all three tiers
    assert (
        len([r for r in rows if r["sku_identifier"] == "NVIDIA H100"]) == 3
    )


def test_inr_table_cells_record_natively_never_as_usd():
    inr_row = (
        "NVIDIA L4",
        "24",
        "25",
        "110",
        f"{RUPEE}49",
        f"{RUPEE}30,660",
        f"{RUPEE}330,000",
    )
    rows, _ = parse_e2e(_page([inr_row]))
    l4 = [r for r in rows if r["sku_identifier"] == "NVIDIA L4"]
    assert len(l4) == 3
    assert all(r["currency"] == "INR" for r in l4)
    assert all(r["price_usd_gpu_hr"] is None for r in l4)
    hourly = next(r for r in l4 if r["tier"] == "on-demand")
    assert hourly["price_native_per_gpu_hr"] == 49.0
    assert hourly["raw_value"] == f"{RUPEE}49"
    assert hourly["raw_unit"] == "inr_per_gpu_hr"


def test_unrecognized_price_text_is_skipped_not_guessed():
    rows, errs = parse_e2e(
        _page(
            [
                ("NVIDIA H100", "80", "26", "250", "4.00 EUR", "$2,363.01", "$23,914.80"),
            ]
        )
    )
    tiers = {r["tier"] for r in rows if r["extra"]["surface"] == "usd_table"}
    assert "on-demand" not in tiers
    assert any("'4.00 EUR'" in e for e in errs)


def test_reshaped_pages_raise():
    with pytest.raises(RuntimeError, match="expected exactly one pricing table"):
        parse_e2e("<html><body>no tables here</body></html>")
    shuffled = tuple(reversed(_HEADER))
    with pytest.raises(RuntimeError, match="expected exactly one pricing table"):
        parse_e2e(_page([_GOOD_ROW], header=shuffled))
    with pytest.raises(RuntimeError, match="found 2"):
        parse_e2e(_page([_GOOD_ROW], tables=2))
    # header intact but every row failing its pins = whole-surface reshape
    with pytest.raises(RuntimeError, match="zero pinned GPU rows"):
        parse_e2e(_page([("NVIDIA H100", "spec", "26", "250", "$1", "$500", "$5,000")]))


def test_per_gpu_footnote_gone_raises():
    # The 1x/2x/4x/8x multiplier sits on this table: a multiplied default
    # view would pass the header pin and every ratio band, so the per-GPU
    # claim is fenced on the page's own footnote.
    with pytest.raises(RuntimeError, match="per-GPU basis footnote"):
        parse_e2e(_page([_GOOD_ROW], footnote=""))


def test_jsonld_surface_dark_is_partial_error_not_failure():
    rows, errs = parse_e2e(_page([_GOOD_ROW], ld=""))
    assert len(rows) == 3  # the table alone keeps the source alive
    assert any("zero GPU offers" in e for e in errs)


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA B200"] == "B200"
    assert mapped["NVIDIA H200"] == "H200"
    assert mapped["NVIDIA H100"] == "H100"
    assert mapped["NVIDIA RTX PRO 6000"] == "RTX_PRO_6000"
    assert mapped["NVIDIA A100"] == "A100"
    assert mapped["NVIDIA A40"] == "A40"
    assert mapped["NVIDIA L40S"] == "L40S"
    assert mapped["NVIDIA A30"] == "A30"
    assert mapped["NVIDIA L4"] == "L4"
    # the JSON-LD suffix must not break token matching
    assert mapped["NVIDIA B200 GPU Cloud Instance"] == "B200"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known e2e labels now unmapped: {unmapped}"
