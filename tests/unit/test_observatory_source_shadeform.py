# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory shadeform collector -- fixture pins (live page 2026-08-22).

Fixture is a real-byte excerpt of the shadeform.com homepage (103 of the
276 escaped listing blobs, trimmed in page order with real surrounding
context) preserving the edge cases this source exposes: CPU-only instances,
the RTX6000 / RTX6000Ada / RTXPro6000 and A4000 / RTX4000Ada lookalike
labels, near-duplicate same-price offers, baremetal deployment_type, and
US / non-US / mixed availability regions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.shadeform import SOURCE_ID, parse_shadeform

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "shadeform"
    / "homepage_excerpt.html"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_shadeform(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "shadeform"


def test_fixture_parses_clean(parsed):
    rows, errors = parsed
    # 103 blobs in the excerpt: 99 GPU rows + 4 CPU-only instances that are
    # skipped by scope rule, NOT counted as pin failures.
    assert len(rows) == 99
    assert errors == []


def test_cents_per_instance_normalizes_per_gpu(rows):
    """lambdalabs B200 ladder: integer cents per INSTANCE hour divided by
    the row's own num_gpus."""
    lam = [
        r
        for r in rows
        if r["sku_identifier"] == "B200"
        and r["extra"]["cloud"] == "lambdalabs"
    ]
    ladder = {
        (r["gpu_count_basis"], r["raw_value"]): r["price_usd_gpu_hr"]
        for r in lam
    }
    assert ladder == {
        (1, "699"): 6.99,
        (2, "1378"): 6.89,
        (4, "2716"): 6.79,
        (8, "5352"): 6.69,
    }
    assert all(r["raw_unit"] == "cents_per_instance_hr" for r in lam)
    assert all(r["currency"] == "USD" for r in lam)
    assert all(r["memory_gb_label"] == 192 for r in lam)


def test_price_times_basis_reproduces_raw_cents(rows):
    for r in rows:
        cents = int(r["raw_value"])
        reproduced = r["price_usd_gpu_hr"] * r["gpu_count_basis"] * 100
        # price is rounded to 4 decimals per GPU; the tolerance is that
        # rounding scaled back to instance cents.
        tolerance = 0.005 * r["gpu_count_basis"] + 1e-6
        assert abs(reproduced - cents) <= tolerance, r


def test_rounding_case_a16_16x(rows):
    """919 cents / 16 GPUs = 0.574375 -- the 4-decimal rounding case."""
    a16 = next(
        r
        for r in rows
        if r["sku_identifier"] == "A16" and r["gpu_count_basis"] == 16
    )
    assert a16["raw_value"] == "919"
    assert a16["price_usd_gpu_hr"] == 0.5744


def test_baremetal_is_on_demand_with_form_factor_in_extra(rows):
    boost = next(
        r
        for r in rows
        if r["sku_identifier"] == "B200"
        and r["extra"]["cloud"] == "boostrun"
    )
    assert boost["tier"] == "on-demand"
    assert boost["extra"]["deployment_type"] == "baremetal"
    assert boost["price_usd_gpu_hr"] == 3.74
    assert boost["region"] == "US"


def test_region_rollup_us_nonus_mixed(rows):
    scaleway = next(
        r
        for r in rows
        if r["extra"]["cloud"] == "scaleway" and r["sku_identifier"] == "B300"
    )
    assert scaleway["price_usd_gpu_hr"] == 9.0075
    assert scaleway["region"] == "non-US"
    # Availability (per-region stock flags) is recorded verbatim in extra.
    assert scaleway["extra"]["availability"] == [
        {
            "region": "paris-france-1",
            "display_name": "FR, Paris",
            "available": False,
        }
    ]
    mixed = next(
        r
        for r in rows
        if r["extra"]["cloud"] == "lambdalabs"
        and r["sku_identifier"] == "A6000"
    )
    assert mixed["region"] == "mixed"
    gaudi = next(r for r in rows if r["sku_identifier"] == "GAUDI2")
    assert gaudi["region"] == "US"
    assert gaudi["extra"]["gpu_manufacturer"] == "intel"
    # The fixture keeps rows with genuinely-available regions too -- stock
    # disclosure must survive the trim.
    assert any(
        a["available"] for r in rows for a in r["extra"]["availability"]
    )


def test_cpu_rows_skipped_silently_not_as_errors(parsed):
    rows, errors = parsed
    assert not any(r["sku_identifier"] == "CPU" for r in rows)
    assert not any("CPU" in e for e in errors)


def test_near_duplicate_offers_both_recorded(rows):
    """excesssupply publishes two distinct 8x H200 rows at the same price
    (different blob bytes) -- both are real offers and both must print;
    only byte-identical duplicate blobs dedup."""
    pair = [
        r
        for r in rows
        if r["extra"]["cloud"] == "excesssupply"
        and r["sku_identifier"] == "H200"
        and r["gpu_count_basis"] == 8
        and r["raw_value"] == "3613"
    ]
    assert len(pair) == 2
    assert all(r["price_usd_gpu_hr"] == 4.5163 for r in pair)


def test_reseller_cloud_attribution_on_every_row(rows):
    assert all(r["extra"]["cloud"] for r in rows)
    assert all(
        r["extra"]["deployment_type"] in ("vm", "baremetal") for r in rows
    )
    assert all("via " + r["extra"]["cloud"] in r["notes"] for r in rows)


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    # The lookalike ladder: three different RTX 6000-generation parts and
    # two different 4000-class parts must land on DIFFERENT skus.
    assert mapped["RTX6000"] == "RTX_6000"
    assert mapped["RTX6000Ada"] == "RTX_6000_ADA"
    assert mapped["RTXPro6000"] == "RTX_PRO_6000"
    assert mapped["A4000"] == "RTX_A4000"
    assert mapped["RTX4000Ada"] == "RTX_4000_ADA"
    # Variant labels collapse to the part.
    assert mapped["H100_nvl"] == "H100"
    assert mapped["A100_80G"] == "A100"
    assert mapped["V100_32G"] == "V100"
    assert mapped["B300"] == "B300"
    assert mapped["B200"] == "B200"
    assert mapped["GAUDI2"] == "GAUDI2"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known shadeform labels now unmapped: {unmapped}"


def test_vram_label_distinguishes_rtx6000_generations(rows):
    quadro = next(r for r in rows if r["sku_identifier"] == "RTX6000")
    ada = next(r for r in rows if r["sku_identifier"] == "RTX6000Ada")
    assert quadro["memory_gb_label"] == 24
    assert ada["memory_gb_label"] == 48


def _blob(cloud, gpu_type, num_gpus, hourly_price, deployment_type):
    # Regex-conformant synthetic blob: "cloud" first, "deployment_type" last.
    return (
        '{"cloud":"%s","shade_instance_type":"synthetic","gpu_type":"%s",'
        '"num_gpus":%s,"hourly_price":%s,"availability":[],'
        '"deployment_type":"%s"}'
        % (cloud, gpu_type, num_gpus, hourly_price, deployment_type)
    )


def test_unknown_deployment_type_skipped_never_tier_guessed():
    body = ",".join(
        [
            _blob("verda", "B300", 1, 791, "vm"),
            _blob("verda", "B300", 1, 500, "spot_preview"),
        ]
    )
    rows, errors = parse_shadeform(body)
    assert [r["raw_value"] for r in rows] == ["791"]
    assert len(errors) == 1
    assert "deployment_type 'spot_preview'" in errors[0]


def test_non_integer_cents_skipped_and_counted():
    body = ",".join(
        [
            _blob("verda", "B300", 1, 791, "vm"),
            _blob("verda", "B200", 1, "6.52", "vm"),  # dollars-float reshape
            _blob("verda", "H200", 1, 0, "vm"),  # zero is not a price
        ]
    )
    rows, errors = parse_shadeform(body)
    assert [r["raw_value"] for r in rows] == ["791"]
    assert len(errors) == 2
    assert all("not positive integer cents" in e for e in errors)


def test_byte_identical_duplicate_blobs_dedup():
    one = _blob("verda", "B300", 1, 791, "vm")
    rows, errors = parse_shadeform(",".join([one, one]))
    assert len(rows) == 1
    assert errors == []


def test_all_rows_failing_pins_raises_specific():
    body = _blob("verda", "B300", 1, "6.52", "vm")
    with pytest.raises(RuntimeError, match="zero passed the identity pins"):
        parse_shadeform(body)


def test_no_blobs_parses_empty():
    # A page with no listing blobs returns nothing; collect()'s result()
    # then raises the parsed-nothing fence.
    assert parse_shadeform("<html>redesigned</html>") == ([], [])
