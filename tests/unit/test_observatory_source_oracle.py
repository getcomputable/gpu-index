# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory oracle collector -- fixture pins (live response 2026-08-22).

Fixture: real cetools /products/?currencyCode=USD rows captured 2026-08-22
(payload lastUpdated 2026-08-14T09:09:55.872Z), trimmed to all 29
'GPU Per Hour' rows plus the lookalike rows the fences must exclude
(VMware per-NODE commit tiers, Roving Edge / Cloud@Customer per-day
possession rows, one OCPU row as generic chaff).

House style: (1) parse the recorded fixture, (2) pin exact prints for
known rows incl. this source's edge cases (license lookalikes, on-prem
twins with identical displayNames, per-node commit tiers, H100T variant),
(3) prove the framework normalization maps this source's real labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.oracle import SOURCE_ID, parse_oracle

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "oracle"
    / "products_usd.json"
)

# Every partNumber the metric fence sees but the policy fences must exclude.
LICENSE_PARTS = {
    "B111824", "B111825", "B111826", "B111827",
    "B111828", "B111829", "B111830", "B111831",
}
ON_PREM_PARTS = {"B110965", "B111454", "B111455"}
# In the fixture but carrying non-GPU metrics (per NODE / per day / OCPU):
NON_GPU_METRIC_PARTS = {"B108806", "B108807", "B108808", "B109493", "B111463"}


def _body(items):
    return json.dumps({"lastUpdated": "2026-08-14T09:09:55.872Z", "items": items})


def _gpu_row(
    part="B98415",
    display="Oracle Cloud Infrastructure - Compute - GPU - H100",
    category="Compute - GPU",
    locs=None,
):
    if locs is None:
        locs = [
            {
                "currencyCode": "USD",
                "prices": [{"model": "PAY_AS_YOU_GO", "value": 10}],
            }
        ]
    return {
        "partNumber": part,
        "displayName": display,
        "metricName": "GPU Per Hour",
        "serviceCategory": category,
        "currencyCodeLocalizations": locs,
    }


@pytest.fixture(scope="module")
def parsed():
    return parse_oracle(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def by_part(rows):
    out = {r["extra"]["part_number"]: r for r in rows}
    assert len(out) == len(rows), "partNumber stopped being unique per row"
    return out


def test_source_id_matches_module():
    assert SOURCE_ID == "oracle"


def test_public_rental_row_pins(by_part):
    """Exact prices for the flagship rows (cross-checked at recon time
    against the pricing page's data-partnumber bindings and Oracle's own
    blogs: H100 $10/GPU/hr, MI300X '$6 per GPU-hour')."""
    pins = {
        "B98415": 10.0,   # H100
        "B110519": 10.0,  # H200 -- 'same as H100' per Oracle's GA blog
        "B110978": 14.0,  # B200
        "B112237": 15.0,  # B300
        "B110979": 16.0,  # GB200
        "B112140": 18.0,  # GB300
        "B109485": 6.0,   # MI300X
        "B111758": 8.6,   # MI355X
        "B109479": 3.5,   # L40S
        "B95907": 4.0,    # A100 80GB (v2)
        "B95909": 2.0,    # A10
        "B112613": 4.5,   # RTX PRO 6000
    }
    for part, price in pins.items():
        obs = by_part[part]
        assert obs["price_usd_gpu_hr"] == price, part
        assert obs["currency"] == "USD"
        assert obs["tier"] == "on-demand"
        assert obs["gpu_count_basis"] == 1
        assert obs["raw_unit"] == "usd_per_gpu_hr"


def test_records_exactly_the_public_rental_rows(rows, parsed):
    """29 GPU-metric rows - 8 licenses - 3 on-prem = 18 public rentals."""
    assert len(rows) == 18
    _, skipped, stats = parsed
    assert skipped == []  # current surface has no anomalies to note
    assert stats["gpu_metric_rows"] == 29
    assert stats["recorded"] == 18
    assert stats["software_license_rows_excluded"] == 8
    assert stats["on_prem_appliance_rows_excluded"] == 3
    assert stats["catalog_last_updated"] == "2026-08-14T09:09:55.872Z"


def test_h100t_variant_stays_distinct(by_part):
    """H100T (B109480, 'Compute - GPU - Other') is a distinct variant at a
    distinct price -- it must print under its own label, never fold into
    H100."""
    h100t = by_part["B109480"]
    assert h100t["price_usd_gpu_hr"] == 10.75
    assert h100t["sku_identifier"] == "OCI - Compute - GPU - H100T"
    assert h100t["extra"]["service_category"] == "Compute - GPU - Other"


def test_software_license_lookalikes_fenced(by_part):
    """'OCI - NVIDIA AI Enterprise - H100' ($2.50) shares serviceCategory
    AND metric with the real H100 rental ($10) -- the license rows must
    never print as rentals."""
    assert not LICENSE_PARTS & set(by_part)
    assert all(
        "NVIDIA AI Enterprise" not in r["sku_identifier"]
        for r in by_part.values()
    )
    assert by_part["B98415"]["price_usd_gpu_hr"] == 10.0


def test_on_prem_twins_fenced(by_part, rows):
    """Two Cloud@Customer rows share the IDENTICAL displayName
    ('... Compute - GPU.L40S', $3.50 vs $2.90) -- only partNumber
    discriminates, so they are excluded, and the only L40S print left is
    the public B109479 at $3.50."""
    assert not ON_PREM_PARTS & set(by_part)
    l40s = [r for r in rows if "L40S" in r["sku_identifier"].upper()]
    assert len(l40s) == 1
    assert l40s[0]["extra"]["part_number"] == "B109479"
    assert l40s[0]["price_usd_gpu_hr"] == 3.5


def test_per_node_and_per_day_metrics_fenced(by_part):
    """VMware BM.GPU.A10.64 commit tiers bill per NODE ($16/$13/$11) and
    Roving Edge / Cloud@Customer infra bill per DAY -- the exact-metric
    fence keeps them out; re-adding them per-GPU would be dishonest."""
    assert not NON_GPU_METRIC_PARTS & set(by_part)
    # (the GB200 rental legitimately costs $16.00/GPU-hr, same figure as
    # the VMware node rate -- so pin on identity, never on price values)
    for r in by_part.values():
        assert "VMWARE" not in r["sku_identifier"].upper()
        assert "ROVING EDGE" not in r["sku_identifier"].upper()


def test_top_level_reshape_raises():
    with pytest.raises(RuntimeError, match="reshaped"):
        parse_oracle(json.dumps({"products": []}))
    with pytest.raises(RuntimeError, match="reshaped"):
        parse_oracle(json.dumps({"lastUpdated": "x", "items": {}}))


def test_multi_currency_payload_raises():
    locs = [
        {"currencyCode": "USD", "prices": [{"model": "PAY_AS_YOU_GO", "value": 10}]},
        {"currencyCode": "EUR", "prices": [{"model": "PAY_AS_YOU_GO", "value": 9}]},
    ]
    with pytest.raises(RuntimeError, match="currency"):
        parse_oracle(_body([_gpu_row(locs=locs)]))


def test_non_usd_localization_raises():
    """We requested currencyCode=USD; a EUR row back means the server-side
    filter broke -- never record under an assumed currency."""
    locs = [{"currencyCode": "EUR", "prices": [{"model": "PAY_AS_YOU_GO", "value": 9}]}]
    with pytest.raises(RuntimeError, match="EUR"):
        parse_oracle(_body([_gpu_row(locs=locs)]))


def test_part_number_name_drift_raises():
    """If B98415 stops saying H100, Oracle remapped part numbers -- the
    catalog would derive a wrong sku from the new name; fail closed."""
    row = _gpu_row(display="Oracle Cloud Infrastructure - Compute - GPU - H200")
    with pytest.raises(RuntimeError, match="drifted"):
        parse_oracle(_body([row]))
    # ...and the check is boundary-aware, so the real H100T row (which
    # contains 'H100' as a substring but not as a token) would raise too if
    # it ever claimed B98415's part number.
    with pytest.raises(RuntimeError, match="drifted"):
        parse_oracle(_body([_gpu_row(display="OCI - Compute - GPU - H100T")]))


def test_non_numeric_price_raises():
    locs = [{"currencyCode": "USD", "prices": [{"model": "PAY_AS_YOU_GO", "value": "10"}]}]
    with pytest.raises(RuntimeError, match="not a plain number"):
        parse_oracle(_body([_gpu_row(locs=locs)]))


def test_double_payg_price_raises():
    locs = [
        {
            "currencyCode": "USD",
            "prices": [
                {"model": "PAY_AS_YOU_GO", "value": 10},
                {"model": "PAY_AS_YOU_GO", "value": 8},
            ],
        }
    ]
    with pytest.raises(RuntimeError, match="ambiguous"):
        parse_oracle(_body([_gpu_row(locs=locs)]))


def test_unpriced_and_zero_rows_skip_with_note():
    rows, skipped, stats = parse_oracle(
        _body(
            [
                _gpu_row(locs=[{"currencyCode": "USD", "prices": []}]),
                _gpu_row(
                    part="B95909",
                    display="Compute  - GPU - A10",
                    locs=[
                        {
                            "currencyCode": "USD",
                            "prices": [{"model": "PAY_AS_YOU_GO", "value": 0}],
                        }
                    ],
                ),
                _gpu_row(part="B99999", display="OCI - Compute - GPU - Z9000", locs=[]),
            ]
        )
    )
    assert rows == []
    assert stats["recorded"] == 0
    assert len(skipped) == 3
    assert all("not a $0 print" in n or "not a real print" in n for n in skipped)


def test_new_price_model_records_known_tier_and_notes_the_new_one():
    locs = [
        {
            "currencyCode": "USD",
            "prices": [
                {"model": "PAY_AS_YOU_GO", "value": 10},
                {"model": "ANNUAL_COMMIT", "value": 7},
            ],
        }
    ]
    rows, skipped, _ = parse_oracle(_body([_gpu_row(locs=locs)]))
    assert len(rows) == 1
    assert rows[0]["price_usd_gpu_hr"] == 10.0
    assert len(skipped) == 1
    assert "ANNUAL_COMMIT" in skipped[0]


def test_unknown_new_part_records_honestly():
    """A new GPU-Per-Hour partNumber outside the pin map records under the
    provider's own label (the catalog maps it or reports it unmapped) --
    never regexed, never guessed."""
    rows, skipped, _ = parse_oracle(
        _body([_gpu_row(part="B99999", display="OCI - Compute - GPU - Z9000")])
    )
    assert skipped == []
    assert len(rows) == 1
    assert rows[0]["sku_identifier"] == "OCI - Compute - GPU - Z9000"
    assert rows[0]["extra"]["part_number"] == "B99999"


def test_unrecognized_service_category_skips_with_note():
    """A NEW product line (the next Cloud@Customer) must fail closed, not
    print as a public rental."""
    row = _gpu_row(
        part="B99998",
        display="Oracle SuperCluster - Compute - GPU - H100",
        category="SuperCluster Appliance",
    )
    rows, skipped, _ = parse_oracle(_body([row]))
    assert rows == []
    assert len(skipped) == 1
    assert "SuperCluster Appliance" in skipped[0]


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["Oracle Cloud Infrastructure - Compute - GPU - H100"] == "H100"
    assert mapped["OCI - Compute - GPU - H200"] == "H200"
    assert mapped["OCI - Compute - GPU - B200"] == "B200"
    assert mapped["Oracle Cloud Infrastructure - Compute - GPU - B300"] == "B300"
    assert mapped["OCI - Compute - GPU - GB200"] == "GB200"
    assert mapped["OCI - Compute - GPU - GB300"] == "GB300"
    assert mapped["OCI - Compute - GPU - MI300X"] == "MI300X"
    # 'OCI- Compute' (missing space) and 'Compute  - GPU' (double space)
    # are real formatting quirks the catalog prep must absorb:
    assert mapped["OCI- Compute - GPU - MI355X"] == "MI355X"
    assert mapped["Compute  - GPU - A10"] == "A10"
    assert mapped["Compute - GPU - L40S"] == "L40S"
    assert mapped["Compute - GPU - A100 - v2"] == "A100"
    assert mapped["OCI - Compute - GPU - RTX PRO 6000"] == "RTX_PRO_6000"
    # Labels that MUST stay unmapped rather than guessed: H100T is a
    # distinct variant (boundary matching keeps H100 out of it), and the
    # X7/V2/E3/E4 rows are hardware-generation labels with no chip token.
    expected_unmapped_ok = {
        "OCI - Compute - GPU - H100T",
        "Compute - Bare Metal GPU Standard - X7",
        "Compute - Virtual Machine GPU Standard - X7",
        "Compute - GPU Standard - V2",
        "Compute - GPU - E3",
        "OCI - Compute - GPU - E4",
    }
    unmapped = {k for k, v in mapped.items() if v is None}
    # Subset, not equality: the catalog growing an entry for one of these
    # is fine; a KNOWN-mapped oracle label going unmapped is not.
    assert unmapped <= expected_unmapped_ok, (
        f"known oracle labels now unmapped: {unmapped - expected_unmapped_ok}"
    )
