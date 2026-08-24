# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory ovh collector -- fixture pins (live responses 2026-08-22).

Fixtures are trimmed real bodies from both billing subsidiaries:
api.ovh.com FR book (EUR, full lineup) and api.us.ovhcloud.com US book
(USD, smaller genuine lineup), captured 2026-08-22. They preserve the
edge rows this source exposes: 8-GPU multi-GPU basis, monthly.postpaid
tier, win-* Windows-license flavors (incl. the cent-rounded
formattedPrice), t*-le-* legacy variants, coming_soon-tagged priced rows,
the Quadro-RTX5000 lookalike label, US rows with intervalUnit "none" on
BOTH hourly and monthly modes, and an ai-serving addon that carries a full
gpu blob but is priced per minute (the brick fence's reason to exist).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.ovh import SOURCE_ID, parse_ovh

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "observatory" / "ovh"
FR_FIXTURE = FIXTURE_DIR / "order_catalog_public_cloud_fr.json"
US_FIXTURE = FIXTURE_DIR / "order_catalog_public_cloud_us.json"


@pytest.fixture(scope="module")
def fr():
    return parse_ovh(
        FR_FIXTURE.read_text(),
        expected_currency="EUR",
        expected_subsidiary="FR",
    )


@pytest.fixture(scope="module")
def us():
    return parse_ovh(
        US_FIXTURE.read_text(),
        expected_currency="USD",
        expected_subsidiary="US",
    )


def _one(rows, plan_code):
    matches = [r for r in rows if r["extra"]["plan_code"] == plan_code]
    assert len(matches) == 1, f"{plan_code}: {len(matches)} rows"
    return matches[0]


def _synthetic_gpu_addon(
    plan_code,
    model="H100",
    number=1,
    price=280000000,
    formatted="2.80 EUR",
    pricings=None,
):
    return {
        "planCode": plan_code,
        "pricings": (
            pricings
            if pricings is not None
            else [
                {
                    "price": price,
                    "formattedPrice": formatted,
                    "interval": 1,
                    "intervalUnit": "hour",
                }
            ]
        ),
        "blobs": {
            "commercial": {"brick": "gpu"},
            "tags": ["active"],
            "technical": {
                "gpu": {
                    "model": model,
                    "number": number,
                    "memory": {"size": 80},
                },
                "os": {"family": "linux"},
            },
        },
    }


def _synthetic_body(addons, currency="EUR", subsidiary="FR"):
    return json.dumps(
        {
            "locale": {
                "currencyCode": currency,
                "subsidiary": subsidiary,
                "taxRate": 20,
            },
            "addons": addons,
        }
    )


def test_source_id_matches_module():
    assert SOURCE_ID == "ovh"


def test_fixture_row_counts(fr, us):
    fr_rows, fr_skipped = fr
    us_rows, us_skipped = us
    # 18 FR gpu addons + 10 US gpu addons in the fixtures, every one priced
    # and pinnable; the non-gpu addons (incl. the ai-serving one) never
    # print and never count as skips -- they are out of scope by the brick
    # fence, not failed rows.
    assert len(fr_rows) == 18
    assert len(us_rows) == 10
    assert fr_skipped == []
    assert us_skipped == []


def test_fr_h100_hourly_pin(fr):
    row = _one(fr[0], "h100-380.consumption")
    assert row["sku_identifier"] == "H100"
    assert row["tier"] == "on-demand"
    assert row["currency"] == "EUR"
    assert row["price_native_per_gpu_hr"] == 2.8
    assert row["price_usd_gpu_hr"] is None  # EUR recorded natively
    assert row["gpu_count_basis"] == 1
    assert row["raw_value"] == "280000000"
    assert row["raw_unit"] == "eur_1e-8_per_node_hr"
    assert row["memory_gb_label"] == 80
    assert row["region"] == "FR subsidiary"
    # U+20AC is the euro sign in the FR book's display strings -- kept as
    # an escape: source files stay ASCII-only.
    assert row["extra"]["formatted_price"] == "2.80 \u20ac"


def test_multi_gpu_basis_reproduces_raw(fr):
    row = _one(fr[0], "h200-1920.consumption")
    assert row["sku_identifier"] == "H200"
    assert row["gpu_count_basis"] == 8
    assert row["price_native_per_gpu_hr"] == 5.25
    assert row["raw_value"] == "4200000000"  # 42.00 EUR/hr for the node
    # price * basis must reproduce the published figure exactly.
    assert (
        row["price_native_per_gpu_hr"] * row["gpu_count_basis"] * 1e8
        == float(row["raw_value"])
    )
    assert row["memory_gb_label"] == 141


def test_monthly_postpaid_tier(fr):
    row = _one(fr[0], "h100-380.monthly.postpaid")
    assert row["tier"] == "monthly-commit"
    assert row["raw_value"] == "194000000000"  # 1940.00 EUR/month
    assert row["raw_unit"] == "eur_1e-8_per_node_month"
    # 1940 / 730 h/month, rounded to the framework's 4 decimals.
    assert row["price_native_per_gpu_hr"] == round(1940.0 / 730.0, 4)
    assert "730h/month" in row["notes"]


def test_us_book_is_usd_and_suffix_beats_intervalunit(us):
    # US consumption rows publish intervalUnit "none"/interval 0 -- tier
    # must come from the planCode suffix.
    hourly = _one(us[0], "l40s-90.consumption")
    assert hourly["tier"] == "on-demand"
    assert hourly["currency"] == "USD"
    assert hourly["price_usd_gpu_hr"] == 1.8
    assert hourly["raw_value"] == "180000000"
    assert hourly["region"] == "US subsidiary"
    # ...and some US MONTHLY rows carry intervalUnit "none" too, so the
    # suffix rule must hold in both directions.
    monthly = _one(us[0], "win-l4-90.monthly.postpaid")
    assert monthly["tier"] == "monthly-commit"
    assert monthly["raw_value"] == "138700000000"


def test_windows_flavors_labeled_not_mixed(fr, us):
    win_fr = _one(fr[0], "win-t1-45.consumption")
    assert win_fr["sku_identifier"] == "Tesla V100"
    assert win_fr["price_native_per_gpu_hr"] == 0.9776
    assert win_fr["extra"]["windows_license_included"] is True
    assert win_fr["extra"]["os_family"] == "windows"
    assert "Windows license included" in win_fr["notes"]
    linux_fr = _one(fr[0], "t1-45.consumption")
    assert "windows_license_included" not in linux_fr["extra"]
    assert linux_fr["extra"]["os_family"] == "linux"
    # The cent-rounded display: 155819000 -> 1.55819 shown as "$1.56 USD".
    # The tripwire must tolerate display rounding, not demand equality.
    win_us = _one(us[0], "win-t2-45.consumption")
    assert win_us["price_native_per_gpu_hr"] == 1.5582
    assert win_us["extra"]["formatted_price"] == "$1.56 USD"


def test_legacy_and_coming_soon_metadata(fr):
    legacy = _one(fr[0], "t1-le-45.consumption")
    assert legacy["extra"]["legacy_le_variant"] is True
    assert legacy["price_native_per_gpu_hr"] == 0.7
    plain = _one(fr[0], "t1-45.consumption")
    assert "legacy_le_variant" not in plain["extra"]
    assert plain["price_native_per_gpu_hr"] == 0.7
    soon = _one(fr[0], "a10-45.consumption")
    assert soon["sku_identifier"] == "A10"
    assert soon["price_native_per_gpu_hr"] == 0.76
    assert "coming_soon" in soon["extra"]["tags"]


def test_managed_ai_rows_fenced_out(fr):
    """The FR fixture includes a REAL ai-serving addon with a full gpu blob
    (model 'NVIDIA Ada Lovelace L4', priced PER MINUTE). If the fence were
    gpu-blob presence instead of brick=='gpu', that per-minute price would
    print as an absurdly cheap hourly row."""
    rows, skipped = fr
    assert not any(
        r["extra"]["plan_code"].startswith("ai-") for r in rows
    )
    assert not any("Ada Lovelace" in r["sku_identifier"] for r in rows)
    # Out-of-brick addons are out of scope, not failures.
    assert not any("ai-app" in note for note in skipped)


def test_wrong_currency_or_subsidiary_raises():
    body = FR_FIXTURE.read_text()
    with pytest.raises(RuntimeError, match="wrong billing entity"):
        parse_ovh(body, expected_currency="USD", expected_subsidiary="FR")
    with pytest.raises(RuntimeError, match="wrong billing entity"):
        parse_ovh(body, expected_currency="EUR", expected_subsidiary="US")


def test_zero_gpu_addons_raises():
    non_gpu = {
        "planCode": "cloud.credit",
        "pricings": [],
        "blobs": {"commercial": {"brick": None}},
    }
    with pytest.raises(RuntimeError, match="ZERO brick=='gpu'"):
        parse_ovh(
            _synthetic_body([non_gpu]),
            expected_currency="EUR",
            expected_subsidiary="FR",
        )


def test_formatted_price_tripwire_fails_row():
    """price/1e8 disagreeing with the row's own formattedPrice beyond the
    half-cent display rounding = the 1e-8 scale convention broke."""
    addon = _synthetic_gpu_addon(
        "h100-380.consumption", price=280000000, formatted="3.80 EUR"
    )
    rows, skipped = parse_ovh(
        _synthetic_body([addon]),
        expected_currency="EUR",
        expected_subsidiary="FR",
    )
    assert rows == []
    assert len(skipped) == 1
    assert "h100-380.consumption" in skipped[0]
    assert "scale convention broken" in skipped[0]


def test_unparseable_formatted_price_fails_row():
    addon = _synthetic_gpu_addon(
        "h100-380.consumption", formatted="1,940.00 EUR"  # thousands sep
    )
    rows, skipped = parse_ovh(
        _synthetic_body([addon]),
        expected_currency="EUR",
        expected_subsidiary="FR",
    )
    assert rows == []
    assert len(skipped) == 1
    assert "cross-check cannot run" in skipped[0]


def test_unknown_billing_suffix_skipped():
    addon = _synthetic_gpu_addon("h100-380.hourly.prepaid")
    rows, skipped = parse_ovh(
        _synthetic_body([addon]),
        expected_currency="EUR",
        expected_subsidiary="FR",
    )
    assert rows == []
    assert len(skipped) == 1
    assert "unknown billing-mode suffix" in skipped[0]


def test_multiple_pricing_phases_skipped():
    base = _synthetic_gpu_addon("h100-380.consumption")
    two_phase = _synthetic_gpu_addon(
        "h100-760.consumption",
        pricings=[
            {"price": 560000000, "formattedPrice": "5.60 EUR"},
            {"price": 500000000, "formattedPrice": "5.00 EUR"},
        ],
    )
    rows, skipped = parse_ovh(
        _synthetic_body([base, two_phase]),
        expected_currency="EUR",
        expected_subsidiary="FR",
    )
    assert [r["extra"]["plan_code"] for r in rows] == [
        "h100-380.consumption"
    ]
    assert len(skipped) == 1
    assert "2 pricing phases" in skipped[0]


def test_missing_model_or_count_skipped():
    no_model = _synthetic_gpu_addon("x1-45.consumption", model="")
    no_count = _synthetic_gpu_addon("x2-45.consumption", number=0)
    fractional = _synthetic_gpu_addon("x3-45.consumption", number=0.5)
    ok = _synthetic_gpu_addon("h100-380.consumption")
    rows, skipped = parse_ovh(
        _synthetic_body([no_model, no_count, fractional, ok]),
        expected_currency="EUR",
        expected_subsidiary="FR",
    )
    assert [r["extra"]["plan_code"] for r in rows] == [
        "h100-380.consumption"
    ]
    assert len(skipped) == 3


def test_duplicate_plan_code_raises():
    a = _synthetic_gpu_addon("h100-380.consumption")
    b = _synthetic_gpu_addon("h100-380.consumption")
    with pytest.raises(RuntimeError, match="appears twice"):
        parse_ovh(
            _synthetic_body([a, b]),
            expected_currency="EUR",
            expected_subsidiary="FR",
        )


def test_real_labels_normalize_through_catalog(fr, us):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in fr[0] + us[0]
    }
    assert mapped["H100"] == "H100"
    assert mapped["H200"] == "H200"
    assert mapped["A100"] == "A100"
    assert mapped["A10"] == "A10"
    assert mapped["L4"] == "L4"
    assert mapped["L40S"] == "L40S"
    assert mapped["Tesla V100"] == "V100"
    # The two OVH-specific labels have dedicated catalog entries (verified
    # 2026-08-22): Tesla V100S maps to the boundary-fenced V100S variant
    # entry (NOT folded into V100), and the Turing-era Quadro-RTX5000 maps
    # to RTX_5000, which sits BELOW RTX_5000_ADA exactly so this label
    # stays off the Ada part. Pinned exactly: a label going unmapped here
    # is a catalog regression or a genuinely new OVH part.
    assert mapped["Tesla V100S"] == "V100S"
    assert mapped["Quadro-RTX5000"] == "RTX_5000"
    unmapped = {k for k, v in mapped.items() if v is None}
    assert unmapped == set()
