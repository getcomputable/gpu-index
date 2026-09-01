# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory verda collector -- fixture pins (live JSON-LD 2026-08-22).

Fixture: the real <script type="application/ld+json"> element sliced
byte-for-byte from https://verda.com/pricing on 2026-08-22 (163 Offer
nodes: instance, serverless, and instant-cluster rows, plus every edge
case the surface exposes -- blank-model CPU-instance offers, storage
offers, verbatim-duplicated H100 rows, confidential-compute 'CC'
variants, and the GB300/B300 lookalike pair).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.verda import SOURCE_ID, parse_verda

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "verda"
    / "pricing_jsonld.html"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_verda(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def partials(parsed):
    return parsed[1]


def _one(rows, name):
    matches = [r for r in rows if r["sku_identifier"] == name]
    assert len(matches) == 1, f"{name!r}: expected exactly one row, got {matches}"
    return matches[0]


def _jsonld(offers):
    graph = [{"@type": "Offer", **o} for o in offers]
    return (
        '<script type="application/ld+json">'
        + json.dumps({"@context": "https://schema.org", "@graph": graph})
        + "</script>"
    )


def test_source_id_matches_module():
    assert SOURCE_ID == "verda"


def test_instance_rows_exact_prices(rows):
    b300 = _one(rows, "1x B300 SXM6 268GB on-demand")
    assert b300["price_usd_gpu_hr"] == 7.5
    assert b300["currency"] == "USD"
    assert b300["raw_value"] == "7.5"
    assert b300["raw_unit"] == "usd_per_gpu_hr"
    assert b300["gpu_count_basis"] == 1
    assert b300["tier"] == "on-demand"
    assert b300["extra"]["product"] == "instance"

    b200_spot = _one(rows, "1x B200 SXM6 180GB spot")
    assert b200_spot["price_usd_gpu_hr"] == 3.055
    assert b200_spot["tier"] == "spot"


def test_instant_cluster_normalizes_by_leading_count(rows):
    cluster = _one(rows, "104x B300 SXM6 268GB instant cluster")
    assert cluster["raw_value"] == "780"
    assert cluster["raw_unit"] == "usd_per_cluster_hr"
    assert cluster["gpu_count_basis"] == 104
    assert cluster["price_usd_gpu_hr"] == 7.5  # 780 / 104
    assert cluster["tier"] == "on-demand"
    assert cluster["extra"]["product"] == "instant-cluster"
    # the recorded normalization reproduces the published figure exactly
    assert cluster["price_usd_gpu_hr"] * cluster["gpu_count_basis"] == 780


def test_serverless_rows_labeled_serverless(rows):
    cont = _one(rows, "8x B200 SXM6 180GB serverless continuous")
    assert cont["tier"] == "serverless"
    assert cont["raw_value"] == "53.77"
    assert cont["raw_unit"] == "usd_per_container_hr"
    assert cont["gpu_count_basis"] == 8
    assert cont["price_usd_gpu_hr"] == 6.7213  # round(53.77 / 8, 4)
    assert cont["extra"]["serverless_billing"] == "continuous"

    spot = _one(rows, "1x B300 SXM6 268GB serverless spot")
    assert spot["tier"] == "serverless"
    assert spot["price_usd_gpu_hr"] == 4.13
    assert spot["raw_unit"] == "usd_per_gpu_hr"
    assert spot["extra"]["serverless_billing"] == "spot"


def test_gb300_lookalike_recorded_alongside_b300(rows):
    gb300 = _one(rows, "1x GB300 SXM6 288GB on-demand")
    assert gb300["price_usd_gpu_hr"] == 8.62
    assert gb300["memory_gb_label"] == 288
    b300 = _one(rows, "1x B300 SXM6 268GB on-demand")
    assert b300["memory_gb_label"] == 268


def test_confidential_compute_variant_flagged(rows):
    cc = _one(rows, "1x B200 CC SXM6 180GB on-demand")
    assert cc["price_usd_gpu_hr"] == 6.232
    assert cc["extra"]["confidential_compute"] is True
    plain = _one(rows, "1x B200 SXM6 180GB on-demand")
    assert "confidential_compute" not in plain["extra"]


def test_blank_model_rows_skipped_and_counted(rows, partials):
    assert not [r for r in rows if r["sku_identifier"].startswith(" ")]
    blank_notes = [p for p in partials if "blank model label" in p]
    assert len(blank_notes) == 1
    assert blank_notes[0].startswith("skipped 24 ")
    # and that is the ONLY partial this fixture produces
    assert partials == blank_notes


def test_storage_rows_excluded_silently(rows, partials):
    assert not [r for r in rows if "storage" in r["sku_identifier"].lower()]
    assert not [p for p in partials if "storage" in p.lower()]


def test_verbatim_duplicate_offers_dedup(rows):
    # the live page repeats the H100 rows verbatim (featured grid + full
    # list) -- one print each, but _one() already asserts uniqueness
    _one(rows, "1x H100 SXM5 80GB on-demand")
    _one(rows, "1x H100 SXM5 80GB spot")


def test_per_gpu_times_basis_reproduces_every_raw_figure(rows):
    for r in rows:
        native = r["price_native_per_gpu_hr"]
        basis = r["gpu_count_basis"]
        raw = float(r["raw_value"])
        # native is rounded to 4dp, so allow half an ulp scaled by basis
        assert abs(native * basis - raw) <= 0.0001 * basis + 1e-9, r


def test_tier_vocabulary(rows):
    assert {r["tier"] for r in rows} == {"on-demand", "spot", "serverless"}


def test_description_mismatch_raises():
    html = _jsonld(
        [
            {
                "name": "1x B200 SXM6 180GB spot",
                "description": "Hourly on-demand instance price",
                "price": 1.0,
                "priceCurrency": "USD",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="tier labels reshaped"):
        parse_verda(html)


def test_missing_currency_records_unknown_never_usd():
    html = _jsonld(
        [
            {
                "name": "1x B200 SXM6 180GB spot",
                "description": "Hourly spot instance price",
                "price": 1.5,
            }
        ]
    )
    rows, partials = parse_verda(html)
    assert partials == []
    (row,) = rows
    assert row["currency"] == "UNKNOWN"
    assert row["price_usd_gpu_hr"] is None
    assert row["price_native_per_gpu_hr"] == 1.5
    assert row["raw_unit"] == "unknown_per_gpu_hr"


def test_cluster_without_count_skipped_via_partial():
    html = _jsonld(
        [
            {
                "name": "B200 SXM6 180GB instant cluster",
                "description": "Hourly instant cluster price",
                "price": 100,
                "priceCurrency": "USD",
            }
        ]
    )
    rows, partials = parse_verda(html)
    assert rows == []
    assert any("without a leading 'Nx ' GPU count" in p for p in partials)


def test_disagreeing_currency_labels_skipped_never_usd():
    # top level says USD, the offer's own priceSpecification says EUR --
    # recording either label would be a guess (and recording USD could
    # mislabel a EUR figure forever), so the row must skip loudly.
    html = _jsonld(
        [
            {
                "name": "1x B200 SXM6 180GB spot",
                "description": "Hourly spot instance price",
                "price": 1.5,
                "priceCurrency": "USD",
                "priceSpecification": {"price": 1.5, "priceCurrency": "EUR"},
            }
        ]
    )
    rows, partials = parse_verda(html)
    assert rows == []
    assert any("disagreeing currency labels" in p for p in partials)


def test_disagreeing_price_fields_skipped_via_partial():
    html = _jsonld(
        [
            {
                "name": "1x B200 SXM6 180GB spot",
                "description": "Hourly spot instance price",
                "price": 1.5,
                "priceCurrency": "USD",
                "priceSpecification": {"price": 2.5, "priceCurrency": "USD"},
            }
        ]
    )
    rows, partials = parse_verda(html)
    assert rows == []
    assert any("disagreeing price fields" in p for p in partials)


def test_no_jsonld_block_raises():
    with pytest.raises(RuntimeError, match="no application/ld"):
        parse_verda("<html><body>maintenance</body></html>")


def test_zero_offer_nodes_raises():
    html = (
        '<script type="application/ld+json">'
        '{"@context":"https://schema.org","@graph":[{"@type":"WebSite"}]}'
        "</script>"
    )
    with pytest.raises(RuntimeError, match="zero Offer nodes"):
        parse_verda(html)


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["1x B300 SXM6 268GB on-demand"] == "B300"
    # lookalike stays out of B300 (boundary-aware catalog matching)
    assert mapped["1x GB300 SXM6 288GB on-demand"] == "GB300"
    assert mapped["1x B200 CC SXM6 180GB on-demand"] == "B200"
    assert mapped["104x B300 SXM6 268GB instant cluster"] == "B300"
    assert mapped["1x H200 SXM5 141GB on-demand"] == "H200"
    assert mapped["1x H100 SXM5 80GB on-demand"] == "H100"
    assert mapped["1x A100 SXM4 80GB on-demand"] == "A100"
    assert mapped["1x L40S 48GB on-demand"] == "L40S"
    assert mapped["1x RTX 6000 Ada 48GB on-demand"] == "RTX_6000_ADA"
    assert mapped["1x RTX A6000 48GB on-demand"] == "RTX_A6000"
    assert mapped["1x RTX PRO 6000 96GB on-demand"] == "RTX_PRO_6000"
    assert mapped["1x RTX PRO 6000 CC 96GB on-demand"] == "RTX_PRO_6000"
    assert mapped["1x Tesla V100 16GB on-demand"] == "V100"
    # Nothing in this source's fixture should be unmapped -- a genuinely
    # new chip appearing live records unmapped and the capture warns; this
    # pin is about the KNOWN labels staying mapped.
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known verda labels now unmapped: {unmapped}"
