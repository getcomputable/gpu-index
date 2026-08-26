# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory computepulse collector -- fixture pins (live responses 2026-08-22).

Fixtures are REAL /api/listings bodies captured live 2026-08-22, trimmed to
row subsets (envelope kept verbatim, each kept row's content byte-identical
to the fetched response; mi325x_listings.json is the full body untouched).
Edge cases preserved:

  - b300_listings.json: USD spot/on_demand rows across all three
    source_kinds; the three Scaleway EUR-native rows (aggregator publishes
    its own USD conversion alongside); available true/false/null.
  - rtx4090_listings.json: compact 'RTX4090' label (catalog compact-token
    match); SwissGPU CHF-native row; a REAL overflowing envelope
    (page.total 250, next_offset 200).
  - h100_listings.json: reserved tier rows; Cyfuture AI INR-native
    provider_scrape row (source_updated_at null); SXM/PCIe variants.
  - mi325x_listings.json: genuinely empty book (data [], total 0).
  - b300_available_probe.json: REAL ?gpu=nvidia-b300&available=true&limit=1
    body captured live 2026-08-25, FULL and untouched (1 row, page.total 3)
    -- the whole-book available_total probe envelope.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources import computepulse
from gpu_index.observatory.sources.computepulse import (
    MAX_PAGES_PER_GPU,
    SOURCE_ID,
    collect,
    dedup_new_observations,
    parse_listings_page,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "observatory" / "computepulse"


def _page(name: str, slug: str):
    return parse_listings_page((FIXTURES / name).read_text(), slug)


def _row(**overrides):
    """A synthetic listing row shaped like the live surface (defaults are
    real field values from the 2026-08-22 capture)."""
    base = {
        "id": "11629",
        "provider": "Verda",
        "gpu_model": "B300",
        "gpu_count": 8,
        "region": "unspecified",
        "price_per_gpu_hour": 3.75,
        "billing_type": "spot",
        "available": None,
        "gpu_variant": None,
        "currency": None,
        "price_native": None,
        "source_kind": "open_api",
        "source_url": "https://example.com/v2",
        "source_updated_at": None,
        "fetched_at": "2026-08-22T18:07:08.477+00:00",
        "human_verified": False,
        "verification_level": 0,
        "links": {
            "gpu": "https://compute-pulse.com/gpu/nvidia-b300",
            "provider": "https://compute-pulse.com/provider/verda",
            "source": None,
        },
    }
    base.update(overrides)
    return base


def _body(rows, *, status="live_or_cached", price_unit="USD per GPU hour",
          total=None, next_offset=None):
    return json.dumps(
        {
            "data": rows,
            "page": {
                "total": len(rows) if total is None else total,
                "limit": 200,
                "offset": 0,
                "next_offset": next_offset,
            },
            "meta": {
                "catalog_status": status,
                "freshest_observation": None,
                "price_unit": price_unit,
                "notice": "Observed public prices are not quotes.",
            },
            "links": {
                "self": "https://compute-pulse.com/api/listings",
                "next": None,
                "documentation": "https://compute-pulse.com/agents",
                "openapi": "https://compute-pulse.com/openapi.json",
                "llms": "https://compute-pulse.com/llms.txt",
            },
        }
    )


@pytest.fixture(scope="module")
def b300():
    return _page("b300_listings.json", "nvidia-b300")


@pytest.fixture(scope="module")
def rtx4090():
    return _page("rtx4090_listings.json", "nvidia-rtx-4090")


@pytest.fixture(scope="module")
def h100():
    return _page("h100_listings.json", "nvidia-h100")


def _by_listing_id(page, listing_id):
    return next(
        o for o in page["observations"]
        if o["extra"]["listing_id"] == listing_id
    )


def test_source_id_matches_module():
    assert SOURCE_ID == "computepulse"


def test_usd_spot_row_pins_exactly(b300):
    row = _by_listing_id(b300, "11629")
    assert row["sku_identifier"] == "B300"
    assert row["price_usd_gpu_hr"] == 3.75
    assert row["price_native_per_gpu_hr"] == 3.75
    assert row["currency"] == "USD"
    assert row["raw_value"] == "3.75"
    assert row["raw_unit"] == "usd_per_gpu_hr_aggregator_stated"
    assert row["gpu_count_basis"] == 8
    assert row["tier"] == "spot"
    assert row["region"] == "unspecified"
    assert row["memory_gb_label"] == 268
    assert "Verda spot lead via compute-pulse aggregator" in row["notes"]
    # Trust metadata rides in extra, complete -- these are LEADS not PRINTS
    # and a consumer must be able to screen on provenance.
    extra = row["extra"]
    assert extra["provider"] == "Verda"
    assert extra["source_kind"] == "open_api"
    assert extra["source_url"].startswith("https://dstack-gpu-pricing")
    assert extra["human_verified"] is False
    assert extra["verification_level"] == 0
    assert extra["available"] is None
    assert extra["aggregator_fetched_at"] == "2026-08-22T18:07:08.477+00:00"
    assert extra["source_updated_at"] == "2026-08-21T00:00:00+00:00"


def test_eur_native_row_recorded_natively_not_as_usd(b300):
    """Scaleway quotes EUR; the aggregator publishes its own USD conversion.
    The record must stay native (FX at aggregation time is the aggregator's
    number, visible in extra but never a USD list price)."""
    row = _by_listing_id(b300, "11966")
    assert row["currency"] == "EUR"
    assert row["price_usd_gpu_hr"] is None
    assert row["price_native_per_gpu_hr"] == 7.5
    assert row["raw_value"] == "7.5"
    assert row["raw_unit"] == "eur_per_gpu_hr_aggregator_stated"
    assert row["gpu_count_basis"] == 8
    assert row["extra"]["aggregator_usd_per_gpu_hr"] == 8.7743
    assert row["extra"]["provider"] == "Scaleway"
    # All three Scaleway instance sizes record, each on its own basis.
    eur = [o for o in b300["observations"] if o["currency"] == "EUR"]
    assert {(o["gpu_count_basis"], o["price_native_per_gpu_hr"]) for o in eur} == {
        (8, 7.5), (4, 8.52), (2, 9.48),
    }


def test_chf_native_row_and_compact_label(rtx4090):
    row = _by_listing_id(rtx4090, "3250")
    assert row["sku_identifier"] == "RTX4090"  # aggregator's compact label
    assert row["currency"] == "CHF"
    assert row["price_usd_gpu_hr"] is None
    assert row["price_native_per_gpu_hr"] == 0.4
    assert row["raw_unit"] == "chf_per_gpu_hr_aggregator_stated"
    assert row["gpu_count_basis"] == 1
    assert row["extra"]["aggregator_usd_per_gpu_hr"] == 0.4923
    assert row["extra"]["provider"] == "SwissGPU"
    assert row["tier"] == "on-demand"


def test_inr_reserved_scrape_row(h100):
    row = _by_listing_id(h100, "16195")
    assert row["currency"] == "INR"
    assert row["price_native_per_gpu_hr"] == 205.75
    assert row["price_usd_gpu_hr"] is None
    assert row["tier"] == "reserved"
    assert row["gpu_count_basis"] == 8
    assert row["extra"]["source_kind"] == "provider_scrape"
    assert row["extra"]["gpu_variant"] == "SXM"
    assert row["extra"]["source_updated_at"] is None
    assert row["extra"]["aggregator_usd_per_gpu_hr"] == 2.1499


def test_reserved_and_spot_tiers_map(h100):
    hyperstack = _by_listing_id(h100, "7933")
    assert hyperstack["tier"] == "reserved"
    assert hyperstack["price_usd_gpu_hr"] == 1.75
    assert hyperstack["extra"]["gpu_variant"] == "PCIe"
    vast_spot = _by_listing_id(h100, "17947")
    assert vast_spot["tier"] == "spot"
    assert vast_spot["price_usd_gpu_hr"] == 0.302
    assert vast_spot["gpu_count_basis"] == 2


def test_availability_states_ride_through(b300):
    assert _by_listing_id(b300, "1754")["extra"]["available"] is True
    assert _by_listing_id(b300, "8385")["extra"]["available"] is False
    assert _by_listing_id(b300, "11370")["extra"]["available"] is None


def test_empty_book_page_parses_to_zero_rows_without_raising():
    page = _page("mi325x_listings.json", "amd-mi325x")
    assert page["observations"] == []
    assert page["raw_row_count"] == 0
    assert page["total"] == 0
    assert page["next_offset"] is None
    assert page["skips"] == {}


def test_real_overflow_envelope_extracts_pagination(rtx4090):
    """The rtx4090 fixture keeps its REAL envelope: a 250-listing book with
    next_offset 200 -- the shape the 2-page cap and overflow note key on."""
    assert MAX_PAGES_PER_GPU == 2
    assert rtx4090["total"] == 250
    assert rtx4090["next_offset"] == 200


def test_identity_pin_skips_rows_attributed_to_another_slug():
    """A row whose own links.gpu points at a different chip page means the
    server-side gpu filter stopped discriminating for it -- skipped and
    counted, never recorded into the queried book."""
    rows = [
        _row(),
        _row(id="2", links={"gpu": "https://compute-pulse.com/gpu/nvidia-b200",
                            "provider": "x", "source": None}),
        _row(id="3", links=None),
        _row(id="4", links={"provider": "x", "source": None}),
    ]
    page = parse_listings_page(_body(rows), "nvidia-b300")
    assert [o["extra"]["listing_id"] for o in page["observations"]] == ["11629"]
    assert page["skips"] == {"identity_mismatch": 3}


def test_seed_fallback_is_refused():
    with pytest.raises(RuntimeError, match="seeded/unknown fallback"):
        parse_listings_page(_body([_row()], status="seed_fallback"), "nvidia-b300")
    with pytest.raises(RuntimeError, match="catalog_status"):
        parse_listings_page(_body([_row()], status="something_new"), "nvidia-b300")


def test_changed_price_unit_is_refused():
    with pytest.raises(RuntimeError, match="refusing to interpret"):
        parse_listings_page(
            _body([_row()], price_unit="EUR per GPU hour"), "nvidia-b300"
        )


def test_dishonest_rows_are_skipped_and_counted():
    rows = [
        _row(id="1", price_per_gpu_hour=None),           # unpriced lead
        _row(id="2", price_per_gpu_hour=0),              # zero is not a price
        _row(id="3", gpu_count=0),                       # basis unstatable
        _row(id="4", gpu_count=None),
        _row(id="5", billing_type="custom"),             # bespoke terms
        _row(id="6", gpu_model=""),                      # unlabeled
        _row(id="7", provider=""),                       # unattributed lead
        _row(id="8"),                                    # the one honest row
    ]
    page = parse_listings_page(_body(rows), "nvidia-b300")
    assert [o["extra"]["listing_id"] for o in page["observations"]] == ["8"]
    assert page["skips"] == {
        "unpriced": 2,
        "bad_gpu_count": 2,
        "unmapped_billing": 1,
        "unlabeled": 1,
        "unattributed": 1,
    }


def test_ambiguous_currency_records_unknown_never_usd():
    rows = [_row(currency="credits", price_native=5.0)]
    page = parse_listings_page(_body(rows), "nvidia-b300")
    row = page["observations"][0]
    assert row["currency"] == "UNKNOWN"
    assert row["price_usd_gpu_hr"] is None
    assert row["price_native_per_gpu_hr"] == 5.0
    assert row["raw_unit"] == "native_per_gpu_hr_aggregator_stated"
    assert row["extra"]["currency_raw"] == "credits"
    assert row["extra"]["aggregator_usd_per_gpu_hr"] == 3.75


def test_stated_currency_without_native_amount_keeps_usd_and_flags_it():
    rows = [_row(currency="EUR", price_native=None)]
    page = parse_listings_page(_body(rows), "nvidia-b300")
    row = page["observations"][0]
    assert row["currency"] == "USD"  # the aggregator's genuine USD statement
    assert row["price_usd_gpu_hr"] == 3.75
    assert row["extra"]["provider_currency"] == "EUR"


def test_bad_pagination_shapes_raise():
    with pytest.raises(RuntimeError, match="next_offset"):
        parse_listings_page(_body([_row()], next_offset=0), "nvidia-b300")
    with pytest.raises(RuntimeError, match="next_offset"):
        parse_listings_page(_body([_row()], next_offset="200"), "nvidia-b300")
    with pytest.raises(RuntimeError, match="page.total"):
        parse_listings_page(_body([_row()], total=-1), "nvidia-b300")


def test_dedup_across_pages_drops_replayed_listing_ids(b300):
    """sort=price_asc pages are fetched seconds apart; a shifting book can
    replay a listing on page 2. Replays drop, id-less rows are kept."""
    obs = list(b300["observations"])
    seen = set()
    first, dups = dedup_new_observations(obs, seen)
    assert (len(first), dups) == (len(obs), 0)
    replay, dups = dedup_new_observations(obs, seen)
    assert (replay, dups) == ([], len(obs))
    anon = dict(obs[0], extra=dict(obs[0]["extra"], listing_id=None))
    kept, dups = dedup_new_observations([anon, anon], seen)
    assert (len(kept), dups) == (2, 0)


def test_collect_fails_closed_without_options():
    """No config, no slug list: collect must raise before any fetch."""
    with pytest.raises(RuntimeError, match="refusing to guess"):
        collect(options=None)
    with pytest.raises(RuntimeError, match="options.gpus"):
        collect(options={"limit_per_gpu": 200})
    with pytest.raises(RuntimeError, match="duplicates"):
        collect(
            options={
                "gpus": ["nvidia-b300", "nvidia-b300"],
                "limit_per_gpu": 200,
            }
        )
    with pytest.raises(RuntimeError, match="limit_per_gpu"):
        collect(options={"gpus": ["nvidia-b300"], "limit_per_gpu": 0})
    with pytest.raises(RuntimeError, match="limit_per_gpu"):
        collect(options={"gpus": ["nvidia-b300"], "limit_per_gpu": 201})


def _one_slug_collect(monkeypatch, probe_behavior):
    """collect() over the real single-page b300 fixture with the
    available_total probe answered by probe_behavior(url) -- returns
    (result, requested_urls)."""
    urls = []

    def fake_fetch(url, timeout=None):
        urls.append(url)
        if "available=true" in url:
            return probe_behavior(url)
        assert "sort=price_asc" in url
        return (FIXTURES / "b300_listings.json").read_text()

    monkeypatch.setattr(computepulse, "fetch", fake_fetch)
    monkeypatch.setattr(computepulse.time, "sleep", lambda s: None)
    out = collect(options={"gpus": ["nvidia-b300"], "limit_per_gpu": 200})
    return out, urls


def test_collect_records_whole_book_available_total(monkeypatch, b300):
    """Happy path recomputed from the fixtures: the probe's page.total (3,
    live 2026-08-25) lands next to listings_total; the probe URL uses the
    server-side available filter with limit=1 and NO sort param."""
    out, urls = _one_slug_collect(
        monkeypatch,
        lambda url: (FIXTURES / "b300_available_probe.json").read_text(),
    )
    assert len(urls) == 2  # one price page (next_offset null) + one probe
    assert urls[1].endswith("?gpu=nvidia-b300&available=true&limit=1")
    assert "sort" not in urls[1]
    assert out["book_stats"]["nvidia-b300"] == {
        "listings_total": 98,
        "available_total": 3,
        "pages_fetched": 1,
        "raw_rows": 9,
        "rows_recorded": len(b300["observations"]),
        "fetch_truncated": False,
    }
    assert len(out["observations"]) == len(b300["observations"])
    assert "partial_errors" not in out


def test_probe_fetch_failure_never_darks_the_price_lane(monkeypatch, b300):
    """FAIL-OPEN BY RULING: the probe's transport blowing up records
    available_total null + a partial note -- every price row still lands
    and the slug never counts as failed."""
    def boom(url):
        raise OSError("connection reset by peer")

    out, _ = _one_slug_collect(monkeypatch, boom)
    assert len(out["observations"]) == len(b300["observations"])
    stats = out["book_stats"]["nvidia-b300"]
    assert stats["available_total"] is None
    assert stats["listings_total"] == 98
    assert out["partial_errors"] == [
        "nvidia-b300: available_total probe failed "
        "(OSError: connection reset by peer) -- whole-book availability "
        "count missing this capture; price rows unaffected"
    ]


def test_probe_poisoned_envelope_is_fail_open_too(monkeypatch, b300):
    """The probe body flunking the envelope pins (seed_fallback -- canned
    data) must fail open exactly like a transport error: the pin's refusal
    text lands in the note, the price rows are untouched."""
    out, _ = _one_slug_collect(
        monkeypatch, lambda url: _body([_row()], status="seed_fallback")
    )
    assert len(out["observations"]) == len(b300["observations"])
    assert out["book_stats"]["nvidia-b300"]["available_total"] is None
    (note,) = out["partial_errors"]
    assert note.startswith("nvidia-b300: available_total probe failed")
    assert "seeded/unknown fallback" in note
    assert "price rows unaffected" in note


def test_real_labels_normalize_through_catalog(b300, rtx4090, h100):
    """Framework normalization over this source's real labels, including
    the compact RTX form the aggregator uses. Genuinely new chips belong in
    catalog_suggestions, not in this pin."""
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    labels = {
        o["sku_identifier"]
        for page in (b300, rtx4090, h100)
        for o in page["observations"]
    }
    mapped = {
        label: (match_sku(catalog, label) or {"sku": None})["sku"]
        for label in labels
    }
    assert mapped["B300"] == "B300"
    assert mapped["RTX4090"] == "RTX_4090"  # compact-token catalog match
    assert mapped["H100"] == "H100"
    unmapped = [label for label, sku in mapped.items() if sku is None]
    assert not unmapped, f"known computepulse labels now unmapped: {unmapped}"
