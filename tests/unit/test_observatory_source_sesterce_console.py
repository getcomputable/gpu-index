# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory sesterce_console collector -- fixture pins (live console
/compute page 2026-08-25).

Fixture: real bytes fetched from https://cloud.sesterce.com/compute on
2026-08-25 (471,771-byte page, 20 flight chunks, 82 offers across 21
gpuNames), trimmed to the document head, the flight bootstrap + first
chunk, the two chunks immediately before the offers-bearing chunk, and the
offers chunk itself with the filteredOffers array cut from 82 to 9
representative offers (book indices 0, 1, 2, 42, 48, 67, 79, 80, 81 --
edge rows kept: the first offer H200x8 and the last three CPU rows). The
trim deliberately preserves every hostile shape the parser must handle:
the '$1b:1:props:filteredOffers:N' RSC reference strings (51 of them, must
NOT match the key pin), '$undefined' sentinels OUTSIDE the array slice,
per-region available:false entries on live offers (RTXPro6000x8 TYO4),
null countryCodes, float price artifacts (30.008000000000003), the
gpuCount=0 CPU vm tiers, and instance-total prices needing per-GPU
division. All splices cut between real top-level array elements; every
byte inside the kept chunks is verbatim from the live page except the four
cloud._id record ids, which are length-preserving 24-hex stand-ins
(aa00000000000000000000NN) so the flight chunk byte lengths stay exact.
The parser reads cloud.name and never cloud._id, so no pin moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources import COLLECTORS
from gpu_index.observatory.sources import sesterce as sesterce_home
from gpu_index.observatory.sources.sesterce_console import (
    SOURCE_ID,
    URL,
    _offers_slice,
    parse_console_book,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "sesterce_console"
    / "compute_excerpt.html"
)
HOMEPAGE_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "sesterce"
    / "homepage_excerpt.html"
)


def make_page(offers_json: str) -> str:
    """A minimal one-chunk flight page around a synthetic offers array --
    the surgical harness for per-offer type-pin tests (the fixture keeps
    the realistic multi-chunk shape)."""
    payload = '0:["x",{"filteredOffers":' + offers_json + "}]"
    chunk = json.dumps(payload)[1:-1]
    return (
        '<html><body><script>self.__next_f.push([1,"'
        + chunk
        + '"])</script></body></html>'
    )


def offer_dict(**overrides):
    base = {
        "gpuName": "H100",
        "gpuCount": 1,
        "spotOfferAvailable": False,
        "hourlySpotPrice": 0,
        "hourlyPrice": 2.189,
        "configuration": {"interconnect": "pcie", "vRamGB": 80},
        "nvlink": False,
        "availability": [
            {
                "name": "AMS",
                "region": "AMS",
                "countryCode": None,
                "available": True,
            }
        ],
        "deploymentType": "baremetal",
        "cloud": {"_id": "x", "name": "AZ_02"},
        "instanceId": "H100",
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def fixture_html():
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parsed(fixture_html):
    return parse_console_book(fixture_html)


def test_source_id_matches_module_and_both_surfaces_register():
    assert SOURCE_ID == "sesterce_console"
    # Sibling surfaces: separate modules, separate fetches, separate URLs
    # -- a console reshape can never dark the homepage price lane.
    assert "sesterce" in COLLECTORS and "sesterce_console" in COLLECTORS
    assert COLLECTORS["sesterce"] is not COLLECTORS["sesterce_console"]
    assert URL != sesterce_home.URL


def test_pins_exact_per_gpu_prices_for_all_six_gpu_offers(parsed):
    rows, _, _ = parsed
    prices = {
        (r["sku_identifier"], r["extra"]["instance_id"], r["extra"]["deployment_type"]): r[
            "price_usd_gpu_hr"
        ]
        for r in rows
    }
    # Instance-total hourlyPrice / gpuCount, recomputed from the fixture.
    assert prices == {
        ("H200", "H200x8", "vm"): 3.52,  # 28.16 / 8
        ("RTXPro6000", "RTXPro6000x8", "baremetal"): 3.751,  # 30.008.../8
        ("A100_80G", "A100_sxm4_80Gx8", "vm"): 3.069,  # 24.552.../8
        ("H100", "H100", "baremetal"): 2.189,
        ("H100", "H100_sxm5", "vm"): 4.796,
        ("A100", "A100_sxm4", "vm"): 2.189,
    }
    for r in rows:
        assert r["currency"] == "USD"
        assert r["tier"] == "on-demand"
        assert r["region"] == "global"
        assert r["raw_unit"] == "usd_per_instance_hr"
        # raw_value is the untouched instance-total print (float artifacts
        # kept); price * basis must re-derive from it exactly.
        assert r["raw_value"] == str(r["extra"]["hourly_price_instance"])
        assert r["gpu_count_basis"] == r["extra"]["gpu_count"]
        assert r["price_usd_gpu_hr"] == round(
            r["extra"]["hourly_price_instance"] / r["gpu_count_basis"], 4
        )
    # Float artifacts recorded verbatim, never tidied.
    rtx = next(r for r in rows if r["sku_identifier"] == "RTXPro6000")
    assert rtx["raw_value"] == "30.008000000000003"


def test_availability_array_recorded_verbatim_with_false_entries(parsed):
    rows, _, _ = parsed
    rtx = next(r for r in rows if r["sku_identifier"] == "RTXPro6000")
    # The per-region explicit booleans, byte-for-byte from the live book:
    # a false entry on a LIVE offer (TYO4) and null countryCodes.
    assert rtx["extra"]["availability"] == [
        {"name": "TYO4", "region": "TYO4", "countryCode": None, "available": False},
        {"name": "AMS", "region": "AMS", "countryCode": None, "available": True},
        {"name": "SYD2", "region": "SYD2", "countryCode": None, "available": True},
        {
            "name": "chicago-usa-4",
            "region": "chicago-usa-4",
            "countryCode": None,
            "available": True,
        },
        {
            "name": "ashburn-usa-7",
            "region": "ashburn-usa-7",
            "countryCode": None,
            "available": True,
        },
    ]
    assert rtx["extra"]["regions_listed"] == 5
    assert rtx["extra"]["regions_available"] == 4
    assert rtx["extra"]["nvlink"] is False
    assert rtx["extra"]["interconnect"] == "pcie"
    assert rtx["extra"]["vram_gb"] == 768
    assert rtx["extra"]["cloud_name"] == "AZ_02"
    assert rtx["extra"]["spot_offer_available"] is False
    assert rtx["extra"]["hourly_spot_price"] == 0
    h200 = next(r for r in rows if r["sku_identifier"] == "H200")
    assert h200["extra"]["nvlink"] is True
    assert h200["extra"]["interconnect"] == "sxm5"
    assert h200["extra"]["vram_gb"] == 1128
    assert h200["extra"]["cloud_name"] == "AZ_22"
    assert h200["extra"]["regions_listed"] == 1
    assert h200["extra"]["regions_available"] == 1


def test_cpu_zero_count_offers_skipped_and_counted(parsed):
    rows, partial_errors, _ = parsed
    assert "CPU" not in {r["sku_identifier"] for r in rows}
    notes = [p for p in partial_errors if "non_gpu_zero_count" in p]
    assert len(notes) == 1
    assert "skipped 3 non_gpu_zero_count offer(s)" in notes[0]


def test_homepage_only_chips_tripwire_notes_never_fails(parsed):
    """B200/B300 print on the homepage with zero offers in the console
    book (live 2026-08-25) -- noted per capture, never a failure."""
    rows, partial_errors, _ = parsed
    assert len(rows) == 6  # the note rides beside a full healthy record
    notes = [p for p in partial_errors if "homepage-advertised" in p]
    assert len(notes) == 1
    assert "B200, B300" in notes[0]
    assert len(partial_errors) == 2  # the skip note and this tripwire


def test_book_stats_census(parsed):
    _, _, stats = parsed
    assert stats == {
        "offers_in_book": 9,
        "offers_recorded": 6,
        "offers_skipped": 3,
        "gpu_names": ["A100", "A100_80G", "CPU", "H100", "H200", "RTXPro6000"],
        "region_entries": 33,
        "region_entries_available": 19,
    }


def test_rsc_reference_strings_never_match_the_key_pin(fixture_html):
    """The payload carries 51 '$1b:1:props:filteredOffers:N' reference
    strings -- lookalikes that must not count against the exactly-one
    key pin."""
    assert fixture_html.count("filteredOffers") > 1
    rows, _, _ = parse_console_book(fixture_html)
    assert len(rows) == 6


def test_duplicated_offers_key_fails_closed(fixture_html):
    lookalike = (
        '<script>self.__next_f.push([1,"\\"filteredOffers\\":[]"])</script>'
    )
    with pytest.raises(RuntimeError, match="need exactly 1"):
        parse_console_book(fixture_html + lookalike)


def test_zero_flight_chunks_fails_closed(fixture_html):
    broken = fixture_html.replace("self.__next_f.push", "self.__nxt_f.push")
    with pytest.raises(RuntimeError, match="zero self.__next_f.push"):
        parse_console_book(broken)


def test_undefined_sentinel_inside_slice_fails_closed(fixture_html):
    """'$undefined' lives elsewhere in the payload (kept in the fixture)
    without tripping the pin -- INSIDE the slice it must raise."""
    assert "$undefined" in fixture_html
    mutated = fixture_html.replace(
        '\\"available\\":false', '\\"available\\":\\"$undefined\\"', 1
    )
    with pytest.raises(RuntimeError, match="undefined.*sentinel"):
        parse_console_book(mutated)


def test_bad_chunk_escaping_fails_closed():
    # \x41 is a JS-only escape json.loads refuses -- a chunk-encoding
    # change must dark the source, never be guessed around.
    page = '<script>self.__next_f.push([1,"\\x41"])</script>'
    with pytest.raises(RuntimeError, match="not one valid JSON string"):
        parse_console_book(page)


def test_unbalanced_array_fails_closed():
    with pytest.raises(RuntimeError, match="never closes"):
        _offers_slice('x{"filteredOffers":[{"a":[1,2}')


def test_bracket_scan_is_string_aware():
    # A ']' inside a JSON string must not close the array early.
    flight = '{"filteredOffers":[{"s":"]"}],"z":1}'
    assert _offers_slice(flight) == '[{"s":"]"}]'


def test_slice_that_is_not_valid_json_fails_closed(fixture_html):
    # Stealing availability's '[' shifts the balanced-bracket close onto
    # availability's stray ']' -- the slice must refuse at json.loads.
    mutated = fixture_html.replace(
        '\\"availability\\":[', '\\"availability\\":', 1
    )
    with pytest.raises(RuntimeError, match="not valid JSON"):
        parse_console_book(mutated)


def test_reshaped_gpu_name_fails_closed(fixture_html):
    mutated = fixture_html.replace(
        '{\\"gpuName\\":', '{\\"gpu_name\\":', 1
    )
    with pytest.raises(RuntimeError, match="gpuName is None"):
        parse_console_book(mutated)


def test_stringified_price_fails_closed(fixture_html):
    mutated = fixture_html.replace(
        '\\"hourlyPrice\\":28.16', '\\"hourlyPrice\\":\\"28.16\\"', 1
    )
    with pytest.raises(RuntimeError, match="hourlyPrice"):
        parse_console_book(mutated)


def test_stringified_gpu_count_fails_closed(fixture_html):
    mutated = fixture_html.replace(
        '\\"gpuCount\\":8', '\\"gpuCount\\":\\"8\\"', 1
    )
    with pytest.raises(RuntimeError, match="gpuCount"):
        parse_console_book(mutated)


def test_non_boolean_available_fails_closed(fixture_html):
    """A truthy non-boolean would silently miscount regions_available --
    the explicit-boolean pin must refuse instead."""
    mutated = fixture_html.replace(
        '\\"available\\":false', '\\"available\\":1', 1
    )
    with pytest.raises(RuntimeError, match="explicit boolean"):
        parse_console_book(mutated)


def test_availability_not_a_list_fails_closed():
    page = make_page(json.dumps([offer_dict(availability=42)]))
    with pytest.raises(RuntimeError, match="not a list"):
        parse_console_book(page)


def test_offer_not_an_object_fails_closed():
    page = make_page('["H100"]')
    with pytest.raises(RuntimeError, match="not an object"):
        parse_console_book(page)


def test_missing_instance_id_fails_closed():
    page = make_page(json.dumps([offer_dict(instanceId=None)]))
    with pytest.raises(RuntimeError, match="instanceId"):
        parse_console_book(page)


def test_empty_book_fails_closed():
    with pytest.raises(RuntimeError, match="book pulled or reshaped"):
        parse_console_book(make_page("[]"))


def test_all_offers_skipped_fails_closed():
    page = make_page(
        json.dumps([offer_dict(gpuName="CPU", gpuCount=0, instanceId="cpu")])
    )
    with pytest.raises(RuntimeError, match="ZERO survived"):
        parse_console_book(page)


def test_zero_priced_offer_never_prints():
    page = make_page(
        json.dumps(
            [offer_dict(), offer_dict(hourlyPrice=0.0, instanceId="H100_z")]
        )
    )
    rows, partial_errors, stats = parse_console_book(page)
    assert len(rows) == 1
    assert rows[0]["price_usd_gpu_hr"] == 2.189
    assert any("1 zero_price" in p for p in partial_errors)
    assert stats["offers_skipped"] == 1


def test_homepage_price_lane_unaffected_by_console_reshape():
    """Rule 4 posture: the availability-rich console surface is a separate
    module with its own fetch -- a console reshape raises HERE while the
    homepage price parse still records every row."""
    with pytest.raises(RuntimeError):
        parse_console_book("<html>no flight payload at all</html>")
    home_rows, _ = sesterce_home.parse_sesterce(
        HOMEPAGE_FIXTURE.read_text(encoding="utf-8")
    )
    assert len(home_rows) == 6


def test_real_labels_normalize_through_catalog(parsed):
    rows, _, _ = parsed
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped == {
        "H200": "H200",
        "RTXPro6000": "RTX_PRO_6000",
        "A100_80G": "A100",
        "H100": "H100",
        "A100": "A100",
    }
