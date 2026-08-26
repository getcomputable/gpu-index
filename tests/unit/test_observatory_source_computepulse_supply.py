# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory computepulse_supply collector -- fixture pins (live 2026-08-25).

supply_selling.json is the REAL /api/supply?intent=selling&include_ended=
false&limit=200 body captured live 2026-08-25, trimmed from 25 to 7 rows
(envelope kept verbatim -- page.total 25, next_offset null; each kept row
is a byte-identical span of the fetched body). Edge rows preserved:

  - portal-labs-infrastructure-h100-cf5243: priced, min_duration_hours
    null -> on-demand; min_bookable_gpus null.
  - packet-ai-l40s-b8d333: 730h floor -> monthly-commit; PCIe.
  - packet-ai-a100-87e501: region is the em-dash placeholder (U+2014).
  - daring-pelican: the REAL Anonymous UNPRICED 8x H100 InfiniBand offer
    (price + price_currency null, min_duration_hours exactly 720,
    gpus_per_node 8, available_from set).
  - sharon-ai-h200-8d7a74: 1016-GPU block, 8736h floor, full
    available_from/available_until window, min_bookable_gpus 128.
  - neevcloud-t4-5b8d41: 168h floor -> on-demand; both window dates.
  - btc-com-rtx-4090-0f9238: 672h floor -> on-demand (just under the
    720h monthly-commit boundary); 160 GPUs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources import computepulse_supply
from gpu_index.observatory.sources.computepulse_supply import (
    MAX_PAGES,
    SOURCE_ID,
    TIER_DERIVATION,
    collect,
    offer_observation,
    parse_supply_page,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    REPO_ROOT / "tests" / "fixtures" / "observatory" / "computepulse_supply"
)


def _fixture_body() -> str:
    return (FIXTURES / "supply_selling.json").read_text()


def _row(**overrides):
    """A synthetic capacity offer shaped like the live surface (defaults
    are real field values from the 2026-08-25 capture)."""
    base = {
        "slug": "packet-ai-l40s-b8d333",
        "intent": "capacity_offer",
        "company": "packet.ai",
        "gpu_model": "L40S",
        "gpu_key": "L40S",
        "gpu_count": 4,
        "gpus_per_node": None,
        "node_count": None,
        "region": "us",
        "interconnect": "PCIe",
        "cores_per_node": None,
        "node_ram_gb": None,
        "nvme_gb": None,
        "cluster_interface": None,
        "min_bookable_gpus": 1,
        "min_duration_hours": 730,
        "contract_months": None,
        "price_per_gpu_hour": 0.92,
        "price_currency": "USD",
        "price_observed_at": "2026-07-30",
        "availability_state": "now",
        "available_now": True,
        "available_from": None,
        "available_until": None,
        "created_at": "2026-07-30T09:52:34.759+00:00",
        "claimed": False,
        "shareable_spec": False,
        "notes": None,
        "links": {
            "canonical": "https://compute-pulse.com/supply/packet-ai-l40s-b8d333",
            "marketplace": "https://compute-pulse.com/supply",
            "next_action": "https://compute-pulse.com/rfq",
        },
    }
    base.update(overrides)
    return base


def _body(rows, *, visibility="published_public_fields_only", total=None,
          next_offset=None, offset=0):
    return json.dumps(
        {
            "data": rows,
            "page": {
                "total": len(rows) if total is None else total,
                "limit": 200,
                "offset": offset,
                "next_offset": next_offset,
            },
            "meta": {
                "visibility": visibility,
                "notice": (
                    "Listings are discovery records, not guaranteed "
                    "quotes. Confirm identity, availability, price, and "
                    "terms before transacting."
                ),
            },
            "links": {
                "self": "https://compute-pulse.com/api/supply",
                "next": None,
                "marketplace": "https://compute-pulse.com/supply",
                "documentation": "https://compute-pulse.com/agents",
                "openapi": "https://compute-pulse.com/openapi.json",
            },
        }
    )


@pytest.fixture(scope="module")
def book():
    return parse_supply_page(_fixture_body())


def _by_slug(page, slug):
    return next(
        o for o in page["observations"] if o["extra"]["slug"] == slug
    )


def test_source_id_matches_module():
    assert SOURCE_ID == "computepulse_supply"


def test_fixture_envelope_and_counts(book):
    """Recomputed from the fixture: 7 rows kept of the live 25-offer book,
    6 priced -> observations, 1 real unpriced Anonymous offer counted (not
    dropped, never a $0 row), all 7 declared available now."""
    assert book["raw_row_count"] == 7
    assert book["total"] == 25  # live envelope kept verbatim
    assert book["next_offset"] is None
    assert len(book["observations"]) == 6
    assert book["unpriced"] == 1
    assert book["available_now"] == 7  # includes the unpriced offer
    assert book["skips"] == {}
    # The unpriced Anonymous offer is NOT an observation under any slug.
    slugs = {o["extra"]["slug"] for o in book["observations"]}
    assert "daring-pelican" not in slugs


def test_no_min_duration_prices_on_demand_with_full_extra(book):
    row = _by_slug(book, "portal-labs-infrastructure-h100-cf5243")
    assert row["sku_identifier"] == "H100"
    assert row["price_usd_gpu_hr"] == 1.49
    assert row["currency"] == "USD"
    assert row["raw_value"] == "1.49"
    assert row["raw_unit"] == "usd_per_gpu_hr_seller_declared"
    assert row["gpu_count_basis"] == 4
    assert row["tier"] == "on-demand"  # min_duration_hours null -> else-branch
    assert row["region"] == "ap-australia"
    assert "Portal Labs Infrastructure capacity offer" in row["notes"]
    extra = row["extra"]
    assert extra["company"] == "Portal Labs Infrastructure"
    assert extra["intent"] == "capacity_offer"
    assert extra["available_now"] is True
    assert extra["availability_state"] == "now"
    assert extra["available_from"] is None
    assert extra["available_until"] is None
    assert extra["gpus_per_node"] is None
    assert extra["min_bookable_gpus"] is None
    assert extra["min_duration_hours"] is None
    assert extra["interconnect"] == "Ethernet"
    assert extra["tier_derivation"] == TIER_DERIVATION


def test_monthly_floor_derives_monthly_commit(book):
    row = _by_slug(book, "packet-ai-l40s-b8d333")
    assert row["tier"] == "monthly-commit"  # 730h >= 720h
    assert row["price_usd_gpu_hr"] == 0.92
    assert row["extra"]["min_duration_hours"] == 730
    assert row["extra"]["min_bookable_gpus"] == 1
    assert row["extra"]["interconnect"] == "PCIe"


def test_availability_window_rides_verbatim(book):
    row = _by_slug(book, "sharon-ai-h200-8d7a74")
    assert row["sku_identifier"] == "H200"
    assert row["price_usd_gpu_hr"] == 2.95
    assert row["gpu_count_basis"] == 1016
    assert row["tier"] == "monthly-commit"  # 8736h floor
    assert row["extra"]["available_from"] == "2025-05-31"
    assert row["extra"]["available_until"] == "2028-06-29"
    assert row["extra"]["min_bookable_gpus"] == 128
    assert row["extra"]["interconnect"] == "Infiniband 3.2 Tb"


def test_sub_monthly_floors_price_on_demand(book):
    t4 = _by_slug(book, "neevcloud-t4-5b8d41")
    assert (t4["tier"], t4["extra"]["min_duration_hours"]) == ("on-demand", 168)
    assert t4["price_usd_gpu_hr"] == 0.29
    rtx = _by_slug(book, "btc-com-rtx-4090-0f9238")
    # 672h (28 days) sits just under the 720h boundary.
    assert (rtx["tier"], rtx["extra"]["min_duration_hours"]) == ("on-demand", 672)
    assert rtx["gpu_count_basis"] == 160
    assert rtx["sku_identifier"] == "RTX 4090"


def test_em_dash_region_placeholder_recorded_verbatim(book):
    row = _by_slug(book, "packet-ai-a100-87e501")
    assert row["region"] == "\u2014"  # the marketplace's own em-dash placeholder
    assert row["price_usd_gpu_hr"] == 1.43


def test_tier_boundary_and_hostile_min_duration_types():
    """720 exactly is monthly-commit; below, null, bool True, or a STRING
    '720' (wrong type) all fall to on-demand -- and the verbatim value
    still rides in extra so the derivation stays auditable."""
    cases = [
        (720, "monthly-commit"),
        (719.99, "on-demand"),
        (None, "on-demand"),
        (True, "on-demand"),   # bool is not a duration
        ("720", "on-demand"),  # reshaped type never flatters the tier
    ]
    for mdh, want in cases:
        obs, skip = offer_observation(_row(min_duration_hours=mdh))
        assert skip is None
        assert obs["tier"] == want, (mdh, want)
        assert obs["extra"]["min_duration_hours"] == mdh


def test_unpriced_offers_never_become_zero_dollar_rows():
    """price null (openapi: 'not stated'), zero, negative, or a STRING
    price (wrong type) all count as unpriced -- no observation, no $0."""
    rows = [
        _row(slug="a", price_per_gpu_hour=None, price_currency=None),
        _row(slug="b", price_per_gpu_hour=0),
        _row(slug="c", price_per_gpu_hour=-1.0),
        _row(slug="d", price_per_gpu_hour="0.92"),
    ]
    page = parse_supply_page(_body(rows))
    assert page["observations"] == []
    assert page["unpriced"] == 4
    assert page["skips"] == {}
    assert page["available_now"] == 4  # the capacity signal survives


def test_buyer_requirement_rows_are_never_supply_prices():
    """A priced buyer_requirement is a BID -- recording it as an offer
    would invert the side; skipped + counted, never guessed."""
    rows = [
        _row(slug="bid", intent="buyer_requirement", price_per_gpu_hour=9.99),
        _row(slug="junk", intent="something_new"),
        _row(),
    ]
    page = parse_supply_page(_body(rows))
    assert [o["extra"]["slug"] for o in page["observations"]] == [
        "packet-ai-l40s-b8d333"
    ]
    assert page["skips"] == {"non_offer_intent": 2}


def test_currency_pins_fail_closed():
    """USD needs the row's own price_currency warrant; a non-USD ISO code
    records natively; null/junk currency on a priced row is skipped."""
    page = parse_supply_page(
        _body(
            [
                _row(slug="usd"),
                _row(slug="eur", price_currency="EUR", price_per_gpu_hour=2.0),
                _row(slug="null-cur", price_currency=None),
                _row(slug="junk", price_currency="credits"),
            ]
        )
    )
    by = {o["extra"]["slug"]: o for o in page["observations"]}
    assert set(by) == {"usd", "eur"}
    assert by["usd"]["price_usd_gpu_hr"] == 0.92
    eur = by["eur"]
    assert eur["currency"] == "EUR"
    assert eur["price_usd_gpu_hr"] is None
    assert eur["price_native_per_gpu_hr"] == 2.0
    assert eur["raw_unit"] == "eur_per_gpu_hr_seller_declared"
    assert page["skips"] == {"unpinned_currency": 2}


def test_dishonest_rows_are_skipped_and_counted():
    rows = [
        _row(slug="a", gpu_count=0),
        _row(slug="b", gpu_count=None),
        _row(slug="c", gpu_count=4.0),  # float basis is not a stated count
        _row(slug="d", gpu_model=""),
        "not-a-dict",
        _row(),
    ]
    page = parse_supply_page(_body(rows))
    assert len(page["observations"]) == 1
    assert page["skips"] == {
        "bad_gpu_count": 3,
        "unlabeled": 1,
        "malformed_row": 1,
    }


def test_unknown_availability_state_records_verbatim():
    """Fail-open by plan: soon/ended are schema-documented but unobserved,
    and an undocumented state is signal, not an error."""
    for state in ("soon", "ended", "unheard_of"):
        obs, skip = offer_observation(_row(availability_state=state))
        assert skip is None
        assert obs["extra"]["availability_state"] == state


def test_envelope_fences_fail_closed():
    with pytest.raises(RuntimeError, match="publication contract"):
        parse_supply_page(_body([_row()], visibility="everything"))
    with pytest.raises(RuntimeError, match="meta block"):
        body = json.loads(_body([_row()]))
        del body["meta"]
        parse_supply_page(json.dumps(body))
    with pytest.raises(RuntimeError, match="data list"):
        body = json.loads(_body([_row()]))
        body["data"] = {"rows": []}
        parse_supply_page(json.dumps(body))
    with pytest.raises(RuntimeError, match="page.total"):
        parse_supply_page(_body([_row()], total=-1))
    with pytest.raises(RuntimeError, match="next_offset"):
        parse_supply_page(_body([_row()], next_offset="200"))


def test_collect_end_to_end_on_fixture(monkeypatch):
    """collect() over the real fixture: one page, no sort param (supply
    400s on the listings sort enum), book_stats recomputed exactly."""
    urls = []

    def fake_fetch(url, timeout=None):
        urls.append(url)
        return _fixture_body()

    monkeypatch.setattr(computepulse_supply, "fetch", fake_fetch)
    out = collect()
    assert len(urls) == 1
    assert "intent=selling" in urls[0]
    assert "include_ended=false" in urls[0]
    assert "limit=200" in urls[0]
    assert "sort" not in urls[0]
    assert out["source_id"] == SOURCE_ID
    assert out["first_party_observation"] is False
    assert len(out["observations"]) == 6
    assert out["book_stats"] == {
        "offers_total": 25,
        "pages_fetched": 1,
        "raw_rows": 7,
        "offers_priced": 6,
        "offers_unpriced": 1,
        "offers_available_now": 7,
        "fetch_truncated": False,
    }
    assert any("unpriced offer(s) counted" in note
               for note in out["partial_errors"])


def test_collect_pages_and_dedups_by_slug(monkeypatch):
    """A book wider than one page: offsets must advance, replayed slugs
    drop (openapi: slug is the stable public key), counters aggregate."""
    pages = {
        0: _body([_row(slug="one"), _row(slug="two")],
                 total=3, next_offset=200),
        200: _body([_row(slug="two"), _row(slug="three")],
                   total=3, next_offset=None, offset=200),
    }

    def fake_fetch(url, timeout=None):
        offset = int(url.split("offset=")[1].split("&")[0])
        return pages[offset]

    monkeypatch.setattr(computepulse_supply, "fetch", fake_fetch)
    monkeypatch.setattr(computepulse_supply.time, "sleep", lambda s: None)
    out = collect()
    assert [o["extra"]["slug"] for o in out["observations"]] == [
        "one", "two", "three"
    ]
    assert out["book_stats"]["pages_fetched"] == 2
    assert out["book_stats"]["offers_priced"] == 3
    # The dedup guards EVERY book-level rollup: slug "two" is declared
    # available on both pages but counts once (3, not 4).
    assert out["book_stats"]["offers_available_now"] == 3
    assert any("1 duplicate_slug" in note for note in out["partial_errors"])


def test_cross_page_replay_counts_once_in_unpriced_rollup(monkeypatch):
    """An UNPRICED offer replayed across pages (the exact book-shift case
    the slug dedup exists for) must count once in offers_unpriced /
    offers_available_now -- the availability rollup is the signal this
    source exists to accrue and must never exceed the unique book."""
    unpriced = {"price_per_gpu_hour": None, "price_currency": None}
    pages = {
        0: _body(
            [_row(slug="priced-one"), _row(slug="ghost", **unpriced)],
            total=3,
            next_offset=200,
        ),
        200: _body(
            [_row(slug="ghost", **unpriced), _row(slug="priced-two")],
            total=3,
            next_offset=None,
            offset=200,
        ),
    }

    def fake_fetch(url, timeout=None):
        offset = int(url.split("offset=")[1].split("&")[0])
        return pages[offset]

    monkeypatch.setattr(computepulse_supply, "fetch", fake_fetch)
    monkeypatch.setattr(computepulse_supply.time, "sleep", lambda s: None)
    out = collect()
    assert out["book_stats"]["offers_priced"] == 2
    assert out["book_stats"]["offers_unpriced"] == 1
    assert out["book_stats"]["offers_available_now"] == 3
    assert out["book_stats"]["raw_rows"] == 4


def test_collect_refuses_non_advancing_pagination(monkeypatch):
    monkeypatch.setattr(
        computepulse_supply,
        "fetch",
        lambda url, timeout=None: _body([_row()], next_offset=0),
    )
    with pytest.raises(RuntimeError, match="refusing to loop"):
        collect()
    assert MAX_PAGES >= 2  # the guard, not the cap, must catch the loop


def test_collect_raises_when_all_priced_rows_flunk_pins(monkeypatch):
    """Priced rows exist but every one fails a pin: the intent filter (or
    shape) changed -- raise, never a healthy zero-print capture."""
    body = _body([_row(intent="buyer_requirement", slug=f"s{i}")
                  for i in range(3)])
    monkeypatch.setattr(
        computepulse_supply, "fetch", lambda url, timeout=None: body
    )
    with pytest.raises(RuntimeError, match="ZERO survived"):
        collect()


def test_collect_all_unpriced_book_raises_parsed_nothing(monkeypatch):
    """An all-unpriced book has zero price observations -- result() must
    refuse it (a capture with no prints never looks healthy)."""
    body = _body([_row(slug="a", price_per_gpu_hour=None, price_currency=None)])
    monkeypatch.setattr(
        computepulse_supply, "fetch", lambda url, timeout=None: body
    )
    with pytest.raises(RuntimeError, match="parsed zero GPU price"):
        collect()


def test_real_labels_normalize_through_catalog(book):
    """Framework normalization over this source's real seller labels."""
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    labels = {o["sku_identifier"] for o in book["observations"]}
    mapped = {
        label: (match_sku(catalog, label) or {"sku": None})["sku"]
        for label in labels
    }
    assert mapped == {
        "H100": "H100",
        "L40S": "L40S",
        "A100": "A100",
        "H200": "H200",
        "T4": "T4",
        "RTX 4090": "RTX_4090",
    }
