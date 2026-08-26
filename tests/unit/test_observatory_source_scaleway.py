# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory scaleway collector -- fixture pins (catalog live
2026-08-22, availability sibling live 2026-08-25).

Catalog fixture: real bytes from api.scaleway.com instance catalog
(fr-par-2), trimmed to every GPU row (17 across both pages: B300-SXM,
H100-PCIe, H100-SXM, L4, L40S on page 1; P100/RENDER-S on page 2) plus
CPU-only lookalike rows (BASIC2/BASIC3/POP2 families) that the gpu>=1
fence must exclude.

Availability fixture: real bytes from the sibling
.../products/servers/availability endpoint (fr-par-2, captured
2026-08-25; live page 1 = 100 entries, page 2 = 36 disjoint entries,
page 3 = empty), trimmed to 25 + 4 entries: all 17 GPU catalog rows
(shortage on B300-SXM-*/H100-2-80G/L4-8-24G/L40S-4-48G/L40S-8-48G,
available on the rest), all 5 'scarce' rows of the capture (the rare
middle state -- non-GPU today), and shortage/available non-GPU edge rows
including page 2's first/last sorted keys.

The surface publishes no unpriced GPU rows and no out-of-enum
availability values today, so the skip, join-miss, enum-drift and
fail-closed paths are exercised with synthetic bodies, exemplar-style.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources.scaleway import (
    AVAIL_URL,
    SOURCE_ID,
    URL,
    collect,
    parse_availability,
    parse_scaleway,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "observatory" / "scaleway"


def _fixture_bodies():
    return [
        (FIXTURE_DIR / "servers_page1.json").read_text(),
        (FIXTURE_DIR / "servers_page2.json").read_text(),
    ]


def _avail_bodies():
    return [
        (FIXTURE_DIR / "availability_page1.json").read_text(),
        (FIXTURE_DIR / "availability_page2.json").read_text(),
    ]


def _body(servers):
    return json.dumps({"servers": servers})


_VALID_GPU_SPEC = {
    "gpu": 2,
    "gpu_info": {
        "gpu_manufacturer": "NVIDIA",
        "gpu_name": "H100-SXM",
        "gpu_memory": 85899345920,
    },
    "mig_profile": None,
    "hourly_price": 6.6198,
    "monthly_price": 4832.454,
    "end_of_service": False,
}


@pytest.fixture(scope="module")
def avail_map():
    return parse_availability(_avail_bodies())


@pytest.fixture(scope="module")
def parsed(avail_map):
    return parse_scaleway(_fixture_bodies(), avail_map)


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "scaleway"


def test_every_gpu_row_and_only_gpu_rows(parsed):
    rows, skipped = parsed
    assert len(rows) == 17
    assert skipped == []
    instance_types = {r["extra"]["instance_type"] for r in rows}
    # CPU-only lookalike rows kept in the fixture must never print.
    assert not instance_types & {
        "BASIC2-A2C-4G",
        "BASIC3-X2C-4G",
        "POP2-HN-3",
        "POP2-HM-2C-16G",
    }


def test_eur_recorded_natively_never_as_usd(rows):
    assert all(r["currency"] == "EUR" for r in rows)
    assert all(r["price_usd_gpu_hr"] is None for r in rows)
    assert all(r["raw_unit"] == "eur_per_node_hr" for r in rows)


def test_h100_sxm_8_exact_pin(rows):
    r = next(
        x for x in rows if x["extra"]["instance_type"] == "H100-SXM-8-80G"
    )
    assert r["sku_identifier"] == "H100-SXM"
    assert r["price_native_per_gpu_hr"] == 3.1663  # 25.3308 EUR node / 8
    assert r["raw_value"] == "25.3308"
    assert r["gpu_count_basis"] == 8
    assert r["tier"] == "on-demand"
    assert r["region"] == "fr-par-2"
    assert r["memory_gb_label"] == 80
    assert r["extra"]["gpu_manufacturer"] == "NVIDIA"
    assert r["extra"]["availability"] == "available"


def test_b300_sxm_2_exact_pin(rows):
    r = next(
        x for x in rows if x["extra"]["instance_type"] == "B300-SXM-2-288G"
    )
    assert r["sku_identifier"] == "B300-SXM"
    assert r["price_native_per_gpu_hr"] == 9.48  # 18.96 EUR node / 2
    assert r["gpu_count_basis"] == 2
    assert r["memory_gb_label"] == 288
    # monthly_price is exactly hourly*730 on this surface -- a derived
    # figure carried for audit, never a second observation.
    assert r["extra"]["monthly_price_eur_node"] == 13840.8
    assert r["extra"]["availability"] == "shortage"


def test_page_two_render_s_exact_pin(rows):
    r = next(x for x in rows if x["extra"]["instance_type"] == "RENDER-S")
    assert r["sku_identifier"] == "P100"
    assert r["price_native_per_gpu_hr"] == 1.221
    assert r["gpu_count_basis"] == 1
    assert r["memory_gb_label"] == 16
    # Page-2 catalog row joined against the page-2 availability map entry.
    assert r["extra"]["availability"] == "available"


def test_per_gpu_times_basis_reproduces_node_price(rows):
    for r in rows:
        node_price = float(r["raw_value"])
        reproduced = r["price_native_per_gpu_hr"] * r["gpu_count_basis"]
        # per-GPU price is rounded to 4 decimals by the framework; the
        # reproduction tolerance is exactly that rounding times the basis.
        assert abs(reproduced - node_price) <= 5e-5 * r["gpu_count_basis"] + 1e-9


def test_unpriced_and_unlabeled_and_mig_rows_are_skipped_not_guessed():
    unpriced = dict(_VALID_GPU_SPEC)
    del unpriced["hourly_price"]
    unlabeled = dict(_VALID_GPU_SPEC, gpu_info={"gpu_memory": 1})
    mig = dict(_VALID_GPU_SPEC, mig_profile="nvidia-mig-1g-10gb")
    free = dict(_VALID_GPU_SPEC, hourly_price=0)
    rows, skipped = parse_scaleway(
        [
            _body(
                {
                    "GPU-NOPRICE": unpriced,
                    "GPU-NONAME": unlabeled,
                    "GPU-MIG": mig,
                    "GPU-FREE": free,
                    "GPU-OK": _VALID_GPU_SPEC,
                }
            )
        ],
        {"GPU-OK": "available"},
    )
    assert [r["extra"]["instance_type"] for r in rows] == ["GPU-OK"]
    # Skipped rows never reach the join, so none of them adds a
    # map-miss note on top of its skip note.
    assert len(skipped) == 4
    assert any("GPU-NOPRICE" in s and "hourly_price" in s for s in skipped)
    assert any("GPU-NONAME" in s and "gpu_name" in s for s in skipped)
    assert any("GPU-MIG" in s and "mig_profile" in s for s in skipped)
    assert any("GPU-FREE" in s for s in skipped)


def test_reshaped_servers_mapping_raises():
    with pytest.raises(RuntimeError, match="reshaped"):
        parse_scaleway([json.dumps({"servers": [_VALID_GPU_SPEC]})], {})


def test_reshaped_price_field_raises():
    reshaped = dict(
        _VALID_GPU_SPEC, hourly_price={"currency": "EUR", "units": 6}
    )
    with pytest.raises(RuntimeError, match="hourly_price"):
        parse_scaleway([_body({"GPU-RESHAPED": reshaped})], {})


def test_nan_price_raises_not_printed():
    # json.loads accepts a bare NaN literal; without the finiteness fence
    # it would print a NaN observation that poisons the snapshot.
    body = _body({"GPU-NAN": _VALID_GPU_SPEC}).replace("6.6198", "NaN")
    with pytest.raises(RuntimeError, match="hourly_price"):
        parse_scaleway([body], {})


def test_reshaped_gpu_count_raises():
    # A numeric-string count would otherwise read as "CPU-only" and
    # silently drop the row.
    reshaped = dict(_VALID_GPU_SPEC, gpu="2")
    with pytest.raises(RuntimeError, match="gpu count"):
        parse_scaleway([_body({"GPU-STRCOUNT": reshaped})], {})


def test_fractional_gpu_count_skipped_not_misdivided():
    # gpu=2.5 must never print with basis int(2.5)=2 -- that mis-normalizes
    # the per-GPU price (same dishonest-division rule as mig_profile).
    frac = dict(_VALID_GPU_SPEC, gpu=2.5)
    rows, skipped = parse_scaleway(
        [_body({"GPU-FRAC": frac, "GPU-OK": _VALID_GPU_SPEC})],
        {"GPU-OK": "available"},
    )
    assert [r["extra"]["instance_type"] for r in rows] == ["GPU-OK"]
    assert any("GPU-FRAC" in s and "fractional" in s for s in skipped)


def test_instance_repeating_across_pages_raises():
    page = _body({"GPU-OK": _VALID_GPU_SPEC})
    with pytest.raises(RuntimeError, match="two pages"):
        parse_scaleway([page, page], {"GPU-OK": "available"})


def test_availability_map_parses_fixture_verbatim(avail_map):
    # 25 + 4 trimmed entries across the two pages, values verbatim.
    assert len(avail_map) == 29
    assert avail_map["B300-SXM-2-288G"] == "shortage"
    assert avail_map["H100-SXM-8-80G"] == "available"
    assert avail_map["GP1-L"] == "scarce"  # the rare middle state
    assert avail_map["RENDER-S"] == "available"  # page-2 entry
    assert set(avail_map.values()) == {"available", "scarce", "shortage"}


def test_every_gpu_row_carries_availability_from_the_map(parsed):
    rows, skipped = parsed
    states = [r["extra"]["availability"] for r in rows]
    # fr-par-2 2026-08-25: every GPU catalog row present in the map --
    # no unknowns, and the exact split recomputes from the fixture.
    assert None not in states
    assert sorted(states).count("shortage") == 7
    assert sorted(states).count("available") == 10
    assert skipped == []


def test_gpu_row_missing_from_map_is_unknown_with_note():
    # Proven possible live: fr-par-1's map omits RENDER/P100 catalog
    # rows. The row must still price, with availability None and a loud
    # note -- never a guess, never a darkened lane.
    rows, skipped = parse_scaleway([_body({"GPU-OK": _VALID_GPU_SPEC})], {})
    assert [r["extra"]["instance_type"] for r in rows] == ["GPU-OK"]
    assert rows[0]["extra"]["availability"] is None
    assert rows[0]["price_native_per_gpu_hr"] == 3.3099
    assert len(skipped) == 1
    assert "GPU-OK" in skipped[0] and "availability map" in skipped[0]


def test_enum_drift_recorded_verbatim_with_note():
    # A fourth state must not dark the source OR be silently normalized:
    # record verbatim, note loudly (ordinal reading needs a human).
    rows, skipped = parse_scaleway(
        [_body({"GPU-OK": _VALID_GPU_SPEC})], {"GPU-OK": "preorder"}
    )
    assert rows[0]["extra"]["availability"] == "preorder"
    assert len(skipped) == 1
    assert "'preorder'" in skipped[0] and "enum" in skipped[0]


def test_reshaped_availability_servers_raises():
    with pytest.raises(RuntimeError, match="availability page 1 reshaped"):
        parse_availability(
            [json.dumps({"servers": [{"availability": "available"}]})]
        )


def test_reshaped_availability_entry_raises():
    # Hostile variants: value nested one level deeper, value the bare
    # string, value a number -- each must hit the fence, never join.
    for entry in ({"availability": {"state": "available"}},
                  "available",
                  {"availability": 3}):
        with pytest.raises(RuntimeError, match="string 'availability'"):
            parse_availability([json.dumps({"servers": {"GPU-OK": entry}})])


def test_availability_name_on_two_pages_raises():
    page = json.dumps({"servers": {"GPU-OK": {"availability": "scarce"}}})
    with pytest.raises(RuntimeError, match="two availability pages"):
        parse_availability([page, page])


def _fake_fetch_fixture_pages(url, timeout=None):
    if url == f"{URL}?per_page=100&page=1":
        return (FIXTURE_DIR / "servers_page1.json").read_text()
    if url == f"{AVAIL_URL}?per_page=100&page=1":
        return (FIXTURE_DIR / "availability_page1.json").read_text()
    raise AssertionError(f"unexpected fetch: {url}")


def test_collect_wires_catalog_and_availability(monkeypatch):
    # Both trimmed fixture pages are short (<100 entries), so the loop
    # stops after page 1 of each surface: 16 GPU rows, all joined.
    monkeypatch.setattr(
        "gpu_index.observatory.sources.scaleway.fetch", _fake_fetch_fixture_pages
    )
    out = collect()
    assert len(out["observations"]) == 16
    assert all(
        r["extra"]["availability"] in {"available", "scarce", "shortage"}
        for r in out["observations"]
    )
    assert "partial_errors" not in out


def test_collect_darkens_when_availability_reshapes(monkeypatch):
    # Fail-closed BY PLAN RULING: a reshaped availability sibling darks
    # the whole source (same API -- the catalog parse is then suspect
    # too). This is the deliberate exception to the fail-open doctrine
    # for availability surfaces.
    def fake_fetch(url, timeout=None):
        if url.startswith(AVAIL_URL):
            return json.dumps({"servers": [{"availability": "available"}]})
        return _fake_fetch_fixture_pages(url, timeout)

    monkeypatch.setattr("gpu_index.observatory.sources.scaleway.fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="availability page 1 reshaped"):
        collect()


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"server_types": {"H100-1-80G": {"availability": "available"}}}),
        json.dumps({"servers": None}),
        json.dumps([]),
    ],
)
def test_collect_darkens_when_availability_envelope_vanishes(
    monkeypatch, body
):
    """The fence must fire on the LIKELY reshapes too: a renamed, nulled,
    or delisted 'servers' envelope must reach the raising parser -- the
    paging loop must never swallow it into an empty healthy-looking map
    that records weeks of availability=None."""

    def fake_fetch(url, timeout=None):
        if url.startswith(AVAIL_URL):
            return body
        return _fake_fetch_fixture_pages(url, timeout)

    monkeypatch.setattr("gpu_index.observatory.sources.scaleway.fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="availability page 1 reshaped"):
        collect()


def test_collect_darkens_when_availability_map_is_empty(monkeypatch):
    """A well-formed but EMPTY total map (~140 entries live) is an emptied
    or relocated surface, not a stocked-out zone -- fail closed, never a
    healthy capture of all-unknown availability."""

    def fake_fetch(url, timeout=None):
        if url.startswith(AVAIL_URL):
            return json.dumps({"servers": {}})
        return _fake_fetch_fixture_pages(url, timeout)

    monkeypatch.setattr("gpu_index.observatory.sources.scaleway.fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="availability map came back empty"):
        collect()


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["B300-SXM"] == "B300"
    # Variant split (design section 7): the PCIe label lands on H100_PCIE;
    # SXM stays on the generic H100 entry (no H100_SXM sku by design).
    assert mapped["H100-PCIe"] == "H100_PCIE"
    assert mapped["H100-SXM"] == "H100"
    assert mapped["L40S"] == "L40S"  # must not fall through to L40 or L4
    assert mapped["L4"] == "L4"  # boundary-aware: L4 never matches L40S
    assert mapped["P100"] == "P100"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known scaleway labels now unmapped: {unmapped}"
