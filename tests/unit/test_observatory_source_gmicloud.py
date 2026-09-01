# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory gmicloud collector -- fixture pins (live page 2026-08-22).

Fixture is a trimmed real-byte excerpt of https://www.gmicloud.ai/en/pricing
captured 2026-08-22: document head (meta/og/twitter descriptions carrying
'from $2.00' dollar noise), the JSON-LD block (no per-GPU offers), the five
NVIDIA GPU cards incl. the unpriced GB300 'Pre order' card, the FAQ stating
on-demand and committed pricing differ (the from_floor tier ruling), and an
RSC-flight slice with the escaped 'from $2.00' duplicate. House style:
(1) parse the fixture, (2) pin exact prints incl. the edge cases, (3) prove
framework normalization maps this source's real labels -- with the
GB200/GB300 Grace-Blackwell lookalikes staying distinct from B200/B300.

Second fixture (idcs_excerpt.json): trimmed real bytes of the console
/api/v1/idcs roster captured live 2026-08-25 (200 JSON, 27 entries live,
all "status":"available"). Trim keeps 10 of 27 entries VERBATIM (entry
bytes sliced from the capture, wrapper preserved): the leading 'global'
pseudo-entry (edge row -- must be excluded from counts) plus 9 real DCs
covering every country in the roster (US Denver-1/2, Ohio, Iowa-1,
Iowa-15; TW twaif + taiwan3 last row; SG singapore1; JP japan1). Non-
'available' statuses have NEVER been observed live, so full/maintenance/
unexpected variants below are synthetic mutations of real entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.gmicloud import (
    IDC_URL,
    SOURCE_ID,
    URL,
    collect,
    parse_gmicloud,
    parse_idcs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "gmicloud"
    / "pricing_excerpt.html"
)
IDC_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "gmicloud"
    / "idcs_excerpt.json"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_gmicloud(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


@pytest.fixture(scope="module")
def partial_errors(parsed):
    return parsed[1]


def test_source_id_matches_module():
    assert SOURCE_ID == "gmicloud"


def test_exact_prices_for_all_priced_cards(rows):
    """One from-floor print per priced card, price*basis == raw figure."""
    pinned = {
        r["sku_identifier"]: (r["price_usd_gpu_hr"], r["raw_value"])
        for r in rows
    }
    assert pinned == {
        "NVIDIA GB200": (8.0, "from $8.00"),
        "NVIDIA H100 GPU": (2.0, "from $2.00"),
        "NVIDIA H200": (2.6, "from $2.60"),
        "NVIDIA B200 GPU": (4.0, "from $4.00"),
    }
    for r in rows:
        assert r["gpu_count_basis"] == 1
        assert r["currency"] == "USD"
        assert r["tier"] == "from_floor"  # FAQ: on-demand != committed
        assert r["raw_unit"] == "usd_per_gpu_hr"


def test_preorder_gb300_skipped_not_guessed(rows, partial_errors):
    """The GB300 card is price-shaped ('Pre order' + dangling '/GPU-hour'
    span) -- it must be skipped with a countable reason, never recorded."""
    assert not [r for r in rows if "GB300" in r["sku_identifier"]]
    assert len(partial_errors) == 1
    assert "NVIDIA GB300" in partial_errors[0]
    assert "Pre order" in partial_errors[0]


def test_dollar_noise_never_free_grepped(rows):
    """'from $2.00' rides 6x in the fixture bytes (meta description, og/
    twitter tags, JSON-LD, RSC flight) -- exactly ONE H100 print may come
    out, from the card's own span pair."""
    body = FIXTURE.read_text()
    assert body.count("from $2.00") >= 4  # the noise is really in the bytes
    assert len([r for r in rows if r["price_usd_gpu_hr"] == 2.0]) == 1
    assert len(rows) == 4


def test_availability_badges_recorded(rows):
    badges = {
        r["sku_identifier"]: r["extra"]["availability_badge"] for r in rows
    }
    assert badges == {
        "NVIDIA GB200": "AVAILABLE NOW",
        "NVIDIA H100 GPU": "AVAILABLE NOW",
        "NVIDIA H200": "AVAILABLE NOW",
        "NVIDIA B200 GPU": "Limited Availability",
    }


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {
        "NVIDIA GB200": "GB200",  # Grace-Blackwell superchip, NOT B200
        "NVIDIA H100 GPU": "H100",  # ' GPU' suffix is inconsistent; catalog
        "NVIDIA H200": "H200",  # normalization absorbs it
        "NVIDIA B200 GPU": "B200",
    }
    # The unpriced GB300 card's label must also stay a first-class
    # Grace-Blackwell sku the day it prices up -- never B300.
    assert match_sku(catalog, "NVIDIA GB300")["sku"] == "GB300"


# --- fail-closed pins on synthetic reshapes ---------------------------------

_CARD = (
    '<div><div class="absolute x">AVAILABLE NOW</div>'
    "<h3>NVIDIA H100 GPU</h3>"
    "<span>from $2.00</span><span>/GPU-hour</span></div>"
)


def test_no_nvidia_cards_raises():
    with pytest.raises(RuntimeError, match="no 'NVIDIA <chip>' h3"):
        parse_gmicloud("<h3>Pricing Philosophy</h3><p>words</p>")


def test_two_unit_spans_in_one_card_raises():
    body = (
        "<h3>NVIDIA H100 GPU</h3>"
        "<span>from $2.00</span><span>/GPU-hour</span>"
        "<span>from $9.00</span><span>/GPU-hour</span>"
    )
    with pytest.raises(RuntimeError, match="can no longer be attributed"):
        parse_gmicloud(body)


def test_unit_span_outside_nvidia_cards_raises():
    """A priced '/GPU-hour' row under a non-NVIDIA heading (or an h3 that
    grew nested tags) is a row the pin cannot attribute -- refuse."""
    body = (
        _CARD
        + "<h3>Some New Section</h3>"
        + "<span>from $9.99</span><span>/GPU-hour</span>"
    )
    with pytest.raises(RuntimeError, match="outside NVIDIA card segments"):
        parse_gmicloud(body)


def test_unit_span_with_no_adjacent_price_slot_raises():
    body = "<h3>NVIDIA H100 GPU</h3><p>now /GPU-hour costs vary</p>"
    with pytest.raises(RuntimeError, match="no adjacent price slot"):
        parse_gmicloud(body)


def test_all_cards_unpriced_raises():
    body = (
        "<h3>NVIDIA H100 GPU</h3>"
        "<span>Contact sales</span><span>/GPU-hour</span>"
    )
    with pytest.raises(RuntimeError, match="zero priced observations"):
        parse_gmicloud(body)


def test_zero_dollar_floor_never_prints():
    """A 'from $0.00' slot is skipped (never a $0 print); if it is the ONLY
    card the parse raises rather than claiming a healthy empty source."""
    zero_card = (
        "<h3>NVIDIA B200 GPU</h3>"
        "<span>from $0.00</span><span>/GPU-hour</span>"
    )
    with pytest.raises(RuntimeError, match="zero priced observations"):
        parse_gmicloud(zero_card)
    rows, errors = parse_gmicloud(zero_card + _CARD)
    assert [r["sku_identifier"] for r in rows] == ["NVIDIA H100 GPU"]
    assert any("not a positive price" in e for e in errors)


# --- IDC region capacity metadata (console /api/v1/idcs, 2026-08-25) --------


def test_idc_fixture_book_stats_exact():
    """Counts + exceptions recomputed from the fixture: 10 entries = the
    'global' pseudo-entry (excluded, flagged verbatim) + 9 real DCs, all
    'available' -- exactly what the live roster printed 2026-08-25."""
    body = IDC_FIXTURE.read_text()
    assert len(json.loads(body)["idcs"]) == 10  # trim really kept 10 rows
    assert parse_idcs(body) == {
        "idc_total": 9,
        "idc_status_counts": {"available": 9},
        "idcs_not_available": [],
        "unexpected_statuses": [],
        "global_pseudo_status": "available",
    }


def test_idc_global_pseudo_entry_never_masks_a_real_full():
    """Mutate a real fixture entry to the documented 'full' while 'global'
    stays 'available': the full DC must land verbatim in idcs_not_available
    and the counts -- the pseudo-entry is excluded, not averaged in."""
    doc = json.loads(IDC_FIXTURE.read_text())
    for entry in doc["idcs"]:
        if entry["idcId"] == "us-denver-1":
            entry["status"] = "full"
    stats = parse_idcs(json.dumps(doc))
    assert stats["idc_status_counts"] == {"available": 8, "full": 1}
    assert stats["idcs_not_available"] == [
        {
            "country": "US",
            "countrySubdivision": "US-CO",
            "idcId": "us-denver-1",
            "name": "Denver IDC-1",
            "status": "full",
        }
    ]
    assert stats["unexpected_statuses"] == []  # 'full' is documented enum
    assert stats["global_pseudo_status"] == "available"


def test_idc_unexpected_status_recorded_verbatim_never_coerced():
    doc = json.loads(IDC_FIXTURE.read_text())
    doc["idcs"][1]["status"] = "Sold Out"  # outside documented enum
    doc["idcs"][2]["status"] = "maintenance"  # documented, not unexpected
    stats = parse_idcs(json.dumps(doc))
    assert stats["unexpected_statuses"] == ["Sold Out"]
    assert stats["idc_status_counts"] == {
        "available": 7,
        "Sold Out": 1,
        "maintenance": 1,
    }
    assert {e["status"] for e in stats["idcs_not_available"]} == {
        "Sold Out",
        "maintenance",
    }


def test_idc_global_drift_is_flagged_but_stays_out_of_counts():
    """A weird 'global' status must surface in the flag verbatim while
    every count stays real-DC only."""
    doc = json.loads(IDC_FIXTURE.read_text())
    assert doc["idcs"][0]["idcId"] == "global"
    doc["idcs"][0]["status"] = "full"
    stats = parse_idcs(json.dumps(doc))
    assert stats["global_pseudo_status"] == "full"
    assert stats["idc_status_counts"] == {"available": 9}
    assert stats["idcs_not_available"] == []


def test_idc_parse_fences_fail_closed():
    """Hostile reshapes raise WITHIN the parse (collect turns them into
    one partial_error): non-JSON, missing/renamed list, empty roster,
    wrong-typed entries."""
    with pytest.raises(RuntimeError, match="not JSON"):
        parse_idcs("<html>maintenance page</html>")
    with pytest.raises(RuntimeError, match="top-level 'idcs' list"):
        parse_idcs('{"datacenters": []}')
    with pytest.raises(RuntimeError, match="top-level 'idcs' list"):
        parse_idcs('{"idcs": {"global": "available"}}')
    with pytest.raises(RuntimeError, match="idcs list is empty"):
        parse_idcs('{"idcs": []}')
    with pytest.raises(RuntimeError, match="entry #1 lacks string"):
        parse_idcs('{"idcs": [{"idcId": "a", "status": "available"}, '
                   '{"idcId": "b", "status": 3}]}')
    with pytest.raises(RuntimeError, match="entry #0 lacks string"):
        parse_idcs('{"idcs": ["us-denver-1"]}')


def _collect_with(monkeypatch, idc_body=None, idc_exc=None):
    """Run collect() with fetch stubbed: pricing fixture for URL, and
    either a body or a raise for IDC_URL."""
    def fake_fetch(url, timeout=None):
        if url == URL:
            return FIXTURE.read_text()
        assert url == IDC_URL
        if idc_exc is not None:
            raise idc_exc
        return idc_body
    monkeypatch.setattr("gpu_index.observatory.sources.gmicloud.fetch", fake_fetch)
    return collect()


def test_collect_rides_idc_book_stats_next_to_price_rows(monkeypatch):
    res = _collect_with(monkeypatch, idc_body=IDC_FIXTURE.read_text())
    assert len(res["observations"]) == 4
    assert res["book_stats"]["idc_total"] == 9
    assert res["book_stats"]["idc_status_counts"] == {"available": 9}
    # the only partial_error is the pricing parse's own GB300 pre-order
    assert len(res["partial_errors"]) == 1
    assert "GB300" in res["partial_errors"][0]


def test_price_lane_unaffected_when_idc_fetch_dies(monkeypatch):
    """Fail-SOFT ruling: the IDC fetch raising must cost exactly one
    partial_error and the book_stats -- never a price row."""
    res = _collect_with(
        monkeypatch, idc_exc=OSError("connection refused")
    )
    assert len(res["observations"]) == 4  # price rows fully intact
    assert "book_stats" not in res
    idc_errors = [e for e in res["partial_errors"] if e.startswith("idcs:")]
    assert idc_errors == [
        "idcs: region capacity fetch/parse failed, book_stats dropped "
        "(connection refused)"
    ]


def test_price_lane_unaffected_when_idc_body_reshapes(monkeypatch):
    """Same ruling on the parse side: a reshaped 200 body trips the
    fail-closed fence inside parse_idcs, and collect converts it to the
    single fail-open partial_error."""
    res = _collect_with(monkeypatch, idc_body='{"datacenters": []}')
    assert len(res["observations"]) == 4
    assert "book_stats" not in res
    assert any(
        e.startswith("idcs:") and "top-level 'idcs' list" in e
        for e in res["partial_errors"]
    )
