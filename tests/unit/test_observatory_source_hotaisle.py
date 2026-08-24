# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory hotaisle collector -- fixture pins (live page 2026-08-22).

House style per the runpod exemplar: (1) parse the recorded fixture,
(2) pin exact prints for the known rows incl. this source's edge cases
(the one-month-minimum bare-metal tier, the HTML-escaped '2x &amp; 4x'
count, the lowercase-x chip label, the grandfathered legacy prose row,
the priceless MI355X lookalikes, the meta-description price echoes),
(3) prove the framework normalization maps this source's real labels.

The fixture is a trimmed excerpt of the real page: head metas + JSON-LD
(priceless OfferCatalog trap), the nav MI355X link, the pricing hero
(summary lookalike cards + grandfathered prose) and all three plan cards,
and the footer product links.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.hotaisle import SOURCE_ID, parse_hotaisle

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "hotaisle"
    / "pricing_excerpt.html"
)


@pytest.fixture(scope="module")
def body():
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def parsed(body):
    return parse_hotaisle(body)


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "hotaisle"


def test_exactly_the_four_published_rows(parsed):
    """3 plan cards + 1 grandfathered prose row -- nothing double-printed
    from the hero lookalike cards or the meta-description echoes, nothing
    invented for the priceless MI355X mentions."""
    rows, partial_errors = parsed
    assert len(rows) == 4
    assert partial_errors == []
    assert {r["sku_identifier"] for r in rows} == {"MI300X"}


def test_vm_cards_pin_exact_on_demand_prices(rows):
    vm = {
        r["extra"]["plan"]: r for r in rows if r.get("extra", {}).get("card_kicker") == "VM"
    }
    assert set(vm) == {"Small", "Medium"}
    for plan in ("Small", "Medium"):
        assert vm[plan]["tier"] == "on-demand"
        assert vm[plan]["price_usd_gpu_hr"] == 2.99
        assert vm[plan]["currency"] == "USD"
        assert vm[plan]["raw_value"] == "$2.99/GPU/hr"
        # price is already per GPU-hour: basis 1, price*basis == raw figure
        assert vm[plan]["gpu_count_basis"] == 1
    assert vm["Small"]["extra"]["node_gpus_label"] == "1x"


def test_medium_card_html_escaped_ampersand_count(rows):
    medium = next(r for r in rows if r.get("extra", {}).get("plan") == "Medium")
    assert medium["extra"]["node_gpus_label"] == "2x & 4x"


def test_bare_metal_is_monthly_commit_not_on_demand(rows):
    """$3.39 is a one-month-minimum committed rate -- the committed-tier
    confusion trap; it must never print as on-demand."""
    large = next(
        r for r in rows if r.get("extra", {}).get("card_kicker") == "Bare metal"
    )
    assert large["extra"]["plan"] == "Large"
    assert large["tier"] == "monthly-commit"
    assert large["price_usd_gpu_hr"] == 3.39
    assert large["raw_value"] == "$3.39/GPU/hr"
    assert large["extra"]["node_gpus_label"] == "8x"
    assert "one-month minimum" in large["notes"]


def test_chip_label_upcased_from_lowercase_x(rows):
    """The card span prints 'MI300x'; the identifier is upcased, the
    verbatim label kept in extra."""
    card_rows = [r for r in rows if "card_kicker" in r.get("extra", {})]
    assert card_rows
    for r in card_rows:
        assert r["sku_identifier"] == "MI300X"
        assert r["extra"]["chip_label_as_published"] == "MI300x"


def test_grandfathered_legacy_row(rows):
    legacy = [r for r in rows if r["tier"] == "legacy"]
    assert len(legacy) == 1
    assert legacy[0]["sku_identifier"] == "MI300X"
    assert legacy[0]["price_usd_gpu_hr"] == 1.99
    assert legacy[0]["raw_value"] == "$1.99/GPU/hr"
    assert legacy[0]["extra"]["availability"] == "existing customers only"


def test_no_rows_invented_for_priceless_mi355x(body, rows):
    """MI355X lives in the fixture bytes (nav, JSON-LD, footer) with no
    published price -- it must never become an observation."""
    assert "MI355X" in body
    assert all("MI355" not in r["sku_identifier"] for r in rows)


def test_grandfathered_prose_absent_is_not_an_error(body):
    """The legacy rate is optional prose -- its retirement must not fail
    the whole source."""
    mutated = body.replace("grandfathered at", "formerly billed near")
    rows, partial_errors = parse_hotaisle(mutated)
    assert len(rows) == 3
    assert partial_errors == []
    assert all(r["tier"] != "legacy" for r in rows)


def test_grandfathered_without_chip_pin_is_skipped_and_counted(body):
    """If the adjacent mi300x blog link (slug AND link text both carry the
    chip today) vanishes there is no honest chip attribution -- skipped +
    counted, never guessed."""
    mutated = body.replace(
        "why-we-raised-our-mi300x-price", "why-we-raised-it"
    ).replace("Why we raised our MI300X price", "Why we raised it")
    rows, partial_errors = parse_hotaisle(mutated)
    assert all(r["tier"] != "legacy" for r in rows)
    assert len(partial_errors) == 1
    assert "chip" in partial_errors[0]


def test_fewer_than_three_cards_fails_closed(body):
    """Tailwind anchor churn / pulled cards must raise, not partial-print."""
    truncated = body[: body.index('uppercase tracking-wide">Bare metal</p>')]
    with pytest.raises(RuntimeError, match="plan-card kickers"):
        parse_hotaisle(truncated)


def test_card_without_per_gpu_hr_basis_pin_fails_closed(body):
    """A price that loses its '/GPU/hr' suffix has an unpinned basis --
    the card must raise, never guess a unit."""
    mutated = body.replace("$3.39/GPU/hr", "$3.39/hr")
    with pytest.raises(RuntimeError, match="GPU/hr"):
        parse_hotaisle(mutated)


def test_bare_metal_without_commit_phrase_fails_closed(body):
    """Dropping the one-month-minimum copy would make the monthly-commit
    label dishonest -- refuse the tier rather than mislabel."""
    mutated = body.replace("one-month minimum", "flexible terms").replace(
        "One-month minimum", "Flexible terms"
    )
    with pytest.raises(RuntimeError, match="commit"):
        parse_hotaisle(mutated)


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {"MI300X": "MI300X"}
