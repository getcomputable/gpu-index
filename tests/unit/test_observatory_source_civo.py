# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Observatory civo collector -- fixture pins (live page 2026-08-22).

The fixture is a contiguous real-bytes excerpt of civo.com/pricing spanning
a lookalike NON-GPU pricing table (RAM Optimized compute instances) before
the fence, the full section#nvidia-gpus (all 6 GPU tables, 17 rows, incl.
the duplicate inner <div id="nvidia-gpus"> and the commitment-only B200
row), and the Object store pricing table after -- so the census pins prove
the section fence excludes both neighbors. Live cross-checks at capture
time: L40S 8x $10.32 = 8 x $1.29; the headline per-GPU B200 $3.79 is the
36-MONTH commitment, not on-demand.

Availability hub fixture (cloud_gpu_hub_excerpt.html, captured live
2026-08-25): a contiguous ~11KB real-bytes excerpt of the 344KB
civo.com/ai/cloud-gpu page holding two adjacent self.__next_f.push script
chunks -- a sectionHeader block (lookalike "blockType" key, no Status
column; the prefilter must skip it) and the full Status-columned
pricingTable (ALL 7 cells-rows, all 10 statusText fields incl. the ones on
non-status-variant cells that must be ignored). Edge rows kept: the hub's
H200 row claims 80GB against /pricing's 141GB (must land in partial_errors
unmatched), H100 SXM and PCIe share one (1, H100, 80GB) identity (agreeing
"In stock" annotates once), and the B200 status cell carries an
incongruous $$1.09 hourlyPrice that must never be read as a price.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources import civo as civo_module
from gpu_index.observatory.sources.civo import (
    SOURCE_ID,
    annotate_availability,
    parse_civo_availability,
    parse_civo_pricing,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "civo"
    / "pricing_excerpt.html"
)
HUB_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "observatory"
    / "civo"
    / "cloud_gpu_hub_excerpt.html"
)

EXPECTED_IDENTIFIERS = {
    "NVIDIA L40S 48GB",
    "NVIDIA A100 40GB",
    "NVIDIA A100 80GB",
    "NVIDIA H100 SXM",
    "NVIDIA H200 SXM",
    "NVIDIA B200",
}


@pytest.fixture(scope="module")
def parsed():
    return parse_civo_pricing(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rows(parsed):
    return parsed[0]


def test_source_id_matches_module():
    assert SOURCE_ID == "civo"


def test_census_and_section_fence(parsed):
    """Exactly the 6 GPU tables' labels print -- the fixture carries full
    lookalike pricing tables on BOTH sides of the section fence (compute
    instances before, object store after) and none of their rows may leak.
    17 data rows x 5 hourly surfaces = 85, minus 2 genuine N/As (B200
    on-demand + 6-month) = 83; every skip was pinned, so no partials."""
    rows, partial_errors = parsed
    assert partial_errors == []
    assert {r["sku_identifier"] for r in rows} == EXPECTED_IDENTIFIERS
    assert len(rows) == 83
    assert {r["tier"] for r in rows} == {"on-demand", "reserved"}
    assert all(r["currency"] == "USD" for r in rows)
    assert all(r["raw_unit"] == "usd_per_instance_hr" for r in rows)


def test_h200_small_all_five_price_surfaces(rows):
    small = [
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA H200 SXM"
        and r["extra"]["size_name"] == "Small"
    ]
    surfaces = {
        (r["tier"], r["extra"].get("commitment_months")): r["price_usd_gpu_hr"]
        for r in small
    }
    assert surfaces == {
        ("on-demand", None): 3.49,
        ("reserved", 6): 3.29,
        ("reserved", 12): 3.19,
        ("reserved", 24): 3.09,
        ("reserved", 36): 2.99,
    }
    assert all(r["gpu_count_basis"] == 1 for r in small)
    assert all(r["memory_gb_label"] == 141 for r in small)
    od = next(r for r in small if r["tier"] == "on-demand")
    assert od["raw_value"] == "$3.49"
    assert od["extra"]["instance_detail"] == "1 x NVIDIA H200 - 141GB HBM3e"


def test_b200_is_commitment_only_never_on_demand(rows):
    """THE tier trap: every published B200 number is a 12/24/36-month
    commitment price (on-demand and 6-month are N/A by design). The
    headline $3.79/GPU-hr must print as tier reserved/36mo -- an on-demand
    label would look ~40% under market -- and the N/As must never
    zero-fill."""
    b200 = [r for r in rows if r["sku_identifier"] == "NVIDIA B200"]
    assert all(r["tier"] == "reserved" for r in b200)
    surfaces = {
        r["extra"]["commitment_months"]: (r["price_usd_gpu_hr"], r["raw_value"])
        for r in b200
    }
    assert surfaces == {
        12: (4.49, "$35.92"),
        24: (3.99, "$31.92"),
        36: (3.79, "$30.32"),
    }
    assert all(r["gpu_count_basis"] == 8 for r in b200)
    assert all(r["price_usd_gpu_hr"] > 0 for r in b200)


def test_per_gpu_division_reproduces_the_instance_price(rows):
    """price * gpu_count_basis == the raw per-instance figure, every row
    (e.g. L40S Extra Large $10.32 = 8 x $1.29)."""
    for r in rows:
        raw = float(r["raw_value"].replace("$", "").replace(",", ""))
        assert r["price_usd_gpu_hr"] * r["gpu_count_basis"] == pytest.approx(
            raw, abs=0.005
        )
    l40s_xl = next(
        r
        for r in rows
        if r["sku_identifier"] == "NVIDIA L40S 48GB"
        and r["extra"]["size_name"] == "Extra Large"
        and r["tier"] == "on-demand"
    )
    assert (l40s_xl["raw_value"], l40s_xl["price_usd_gpu_hr"]) == ("$10.32", 1.29)


def test_a100_memory_lookalikes_stay_distinct(rows):
    """A100 40GB vs A100 80GB differ only by the memory size in caption and
    product-detail -- they must print as distinct identifiers with their own
    prices."""

    def small_od(identifier):
        return next(
            r
            for r in rows
            if r["sku_identifier"] == identifier
            and r["tier"] == "on-demand"
            and r["extra"]["size_name"] == "Small"
        )

    a40 = small_od("NVIDIA A100 40GB")
    a80 = small_od("NVIDIA A100 80GB")
    assert a40["price_usd_gpu_hr"] == 1.09
    assert a80["price_usd_gpu_hr"] == 1.79
    assert a40["memory_gb_label"] == 40
    assert a80["memory_gb_label"] == 80


def test_real_labels_normalize_through_catalog(rows):
    catalog = load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")
    mapped = {
        r["sku_identifier"]: (
            match_sku(catalog, r["sku_identifier"]) or {"sku": None}
        )["sku"]
        for r in rows
    }
    assert mapped["NVIDIA L40S 48GB"] == "L40S"
    assert mapped["NVIDIA A100 40GB"] == "A100"
    assert mapped["NVIDIA A100 80GB"] == "A100"
    assert mapped["NVIDIA H100 SXM"] == "H100"
    assert mapped["NVIDIA H200 SXM"] == "H200"
    assert mapped["NVIDIA B200"] == "B200"
    unmapped = [k for k, v in mapped.items() if v is None]
    assert not unmapped, f"known civo labels now unmapped: {unmapped}"


# ---- fail-closed pins, exercised on minimal synthetic bodies ----

SECTION_OPEN = '<section class="pricing-product" id="nvidia-gpus">'


def _hourly(value):
    return (
        '<div id="price-value-hourly" data-option="hourly" '
        f'class="price-value">{value}<span>per hour</span></div>'
    )


def _row(size, count, model, od_value, terms):
    terms_html = "".join(
        f'<div id="{months}_months">{_hourly(value)}</div>'
        for months, value in terms
    )
    return (
        f'<tr><td data-title="Size">{size}'
        f'<div class="product-detail">{count} x NVIDIA {model}</div></td>'
        '<td data-title="On-demand price" '
        f'class="pricing-data on-demand-pricing">{_hourly(od_value)}</td>'
        '<td data-title="Commitment price" '
        f'class="pricing-data commitment-pricing">{terms_html}</td></tr>'
    )


def _table(caption, rows_html):
    return (
        f"<table><caption>NVIDIA {caption} GPU pricing</caption>"
        "<thead><tr><th>Size</th><th>On-demand</th><th>Commitment</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )


def _page(inner):
    return f"<html>{SECTION_OPEN}{inner}</section></html>"


_GOOD_ROW = _row("Small", 1, "H200 - 141GB HBM3e", "$3.49", [(36, "$2.99")])


def test_synthetic_round_trip_and_na_skip():
    body = _page(
        _table(
            "H200 SXM",
            _row("Small", 1, "H200 - 141GB HBM3e", "N/A", [(36, "$2.99")]),
        )
    )
    rows, partial_errors = parse_civo_pricing(body)
    assert partial_errors == []
    assert [
        (r["tier"], r["price_usd_gpu_hr"], r["extra"]["commitment_months"])
        for r in rows
    ] == [("reserved", 2.99, 36)]


def test_missing_section_fence_raises():
    with pytest.raises(RuntimeError, match="nvidia-gpus"):
        parse_civo_pricing(
            '<html><div id="nvidia-gpus">' + _table("B200", _GOOD_ROW) + "</div></html>"
        )


def test_zero_gpu_tables_inside_section_raises():
    with pytest.raises(RuntimeError, match="zero"):
        parse_civo_pricing(_page("<p>no tables here</p>"))


def test_reshaped_price_cell_raises_instead_of_guessing():
    """The hourly print pin (data-option + per-hour span + exact
    N/A-or-$D.DD) failing on a present cell must raise, not skip -- e.g. a
    currency switch away from '$' (pound sign escape-coded: source stays
    ASCII-only)."""
    body = _page(
        _table("H200 SXM", _GOOD_ROW.replace("$3.49", "\u00a32.79"))
    )
    with pytest.raises(RuntimeError, match="hourly print"):
        parse_civo_pricing(body)


def test_caption_model_mismatch_raises():
    """A row landing under the wrong caption (the A100 40GB/80GB lookalike
    hazard) must raise, never mislabel."""
    body = _page(
        _table("A100 40GB", _row("Small", 1, "A100 - 80GB", "$1.79", [(36, "$1.39")]))
    )
    with pytest.raises(RuntimeError, match="does not match"):
        parse_civo_pricing(body)


def test_priced_row_without_identity_pin_is_counted_not_guessed():
    no_pin_row = (
        '<tr><td data-title="Size">Mystery</td>'
        '<td data-title="On-demand price" '
        f'class="pricing-data on-demand-pricing">{_hourly("$9.99")}</td>'
        '<td data-title="Commitment price" '
        'class="pricing-data commitment-pricing">'
        f'<div id="36_months">{_hourly("$8.99")}</div></td></tr>'
    )
    body = _page(_table("H200 SXM", _GOOD_ROW + no_pin_row))
    rows, partial_errors = parse_civo_pricing(body)
    assert len(rows) == 2  # the pinned row still prints
    assert len(partial_errors) == 1
    assert "without a product-detail identity pin" in partial_errors[0]


def test_attributed_tr_row_still_extracts():
    """A data row whose <tr> gains an attribute (class, data-*) must still
    be scanned -- a bare-<tr>-only scanner would silently drop all five of
    its price surfaces with no raise and no partial_error."""
    body = _page(
        _table(
            "H200 SXM",
            _GOOD_ROW.replace("<tr>", '<tr class="featured">', 1),
        )
    )
    rows, partial_errors = parse_civo_pricing(body)
    assert partial_errors == []
    assert {(r["tier"], r["price_usd_gpu_hr"]) for r in rows} == {
        ("on-demand", 3.49),
        ("reserved", 2.99),
    }


def test_priced_cell_outside_any_scanned_row_raises():
    """The priced-cell census: price cells that the row scanner cannot see
    (row markup reshaped past even the attribute-tolerant <tr> pattern)
    must raise, never vanish silently."""
    floating_cell = (
        '<td class="pricing-data on-demand-pricing">'
        f"{_hourly('$9.99')}</td>"
    )
    body = _page(
        _table("H200 SXM", _GOOD_ROW).replace(
            "</tbody>", f"{floating_cell}</tbody>"
        )
    )
    with pytest.raises(RuntimeError, match="outside any scanned"):
        parse_civo_pricing(body)


def test_unattributed_print_before_first_term_block_raises():
    bad = _GOOD_ROW.replace(
        'class="pricing-data commitment-pricing">',
        'class="pricing-data commitment-pricing">' + _hourly("$1.00"),
    )
    with pytest.raises(RuntimeError, match="before the first"):
        parse_civo_pricing(_page(_table("H200 SXM", bad)))


# ---- availability annotation (the /ai/cloud-gpu hub, 2026-08-25) ----


@pytest.fixture(scope="module")
def hub_status_rows():
    return parse_civo_availability(HUB_FIXTURE.read_text())


def test_availability_fixture_all_seven_status_rows_verbatim(
    hub_status_rows,
):
    """All 7 cells-rows of the live pricingTable print with their exact
    text-cell identity and VERBATIM statusText -- and the lookalike
    sectionHeader push chunk before the table contributes nothing."""
    status_rows, partial_errors = hub_status_rows
    assert partial_errors == []
    assert [(r["label"], r["detail"]) for r in status_rows] == [
        ("B200 SXM", "8 x NVIDIA B200 - 180GB"),
        ("A100 40GB", "1 x NVIDIA A100 - 40GB"),
        ("A100 80GB", "1 x NVIDIA A100 - 80GB"),
        ("Small H100 SXM", "1 x NVIDIA H100 - 80GB"),
        ("Small H100 PCIe", "1 x NVIDIA H100 - 80GB"),
        ("Small H200 SXM", "1 x NVIDIA H200 - 80GB"),
        ("Small L40s", "1 x NVIDIA L40S - 48GB"),
    ]
    assert {r["status"] for r in status_rows} == {"In stock"}


def test_availability_join_annotates_only_matched_identities(
    rows, hub_status_rows
):
    """The parsed-identity join against the /pricing fixture: 23 of 83
    observations gain the two availability keys (the 1x sizes of A100
    40/80, H100, L40S -- 5 surfaces each -- plus the three 8x B200
    commitment rows), NOTHING else changes, and the hub's H200 row (CMS
    claims 80GB vs /pricing's 141GB) lands in partial_errors unmatched --
    never guessed onto the 141GB row."""
    observations = copy.deepcopy(rows)
    before = copy.deepcopy(observations)
    partial_errors = annotate_availability(
        observations, hub_status_rows[0]
    )
    assert partial_errors == [
        "availability: hub row(s) 'Small H200 SXM' "
        "('1 x NVIDIA H200 - 80GB') matched zero /pricing rows on "
        "(gpu_count, chip, memory GB) (1, 'H200', 80) -- status not "
        "recorded"
    ]
    annotated = [
        o for o in observations if "availability_status" in o["extra"]
    ]
    assert len(annotated) == 23
    counts = {}
    for o in annotated:
        key = (o["sku_identifier"], o["gpu_count_basis"])
        counts[key] = counts.get(key, 0) + 1
        assert o["extra"]["availability_status"] == "In stock"
        assert o["extra"]["availability_source"] == (
            "www.civo.com/ai/cloud-gpu"
        )
    assert counts == {
        ("NVIDIA A100 40GB", 1): 5,
        ("NVIDIA A100 80GB", 1): 5,
        ("NVIDIA H100 SXM", 1): 5,
        ("NVIDIA L40S 48GB", 1): 5,
        ("NVIDIA B200", 8): 3,
    }
    # No H200 observation was annotated, and NOTHING but the two extra
    # availability keys changed anywhere -- prices are never read from
    # status cells (the fixture's B200 status cell carries $$1.09).
    for pre, post in zip(before, observations):
        stripped = copy.deepcopy(post)
        stripped["extra"].pop("availability_status", None)
        stripped["extra"].pop("availability_source", None)
        assert stripped == pre
    h200 = [
        o
        for o in observations
        if o["sku_identifier"] == "NVIDIA H200 SXM"
    ]
    assert h200
    assert all("availability_status" not in o["extra"] for o in h200)


def _route_fetch(monkeypatch, pricing_body, hub_body=None, hub_exc=None):
    def _fetch(url, timeout=None):
        if url == civo_module.URL:
            return pricing_body
        assert url == civo_module.AVAILABILITY_URL
        if hub_exc is not None:
            raise hub_exc
        return hub_body

    monkeypatch.setattr(civo_module, "fetch", _fetch)


def test_collect_wires_annotation_and_join_errors(monkeypatch):
    _route_fetch(
        monkeypatch,
        FIXTURE.read_text(),
        hub_body=HUB_FIXTURE.read_text(),
    )
    out = civo_module.collect()
    assert len(out["observations"]) == 83
    annotated = [
        o
        for o in out["observations"]
        if o["extra"].get("availability_status") == "In stock"
    ]
    assert len(annotated) == 23
    assert out["partial_errors"] == [
        "availability: hub row(s) 'Small H200 SXM' "
        "('1 x NVIDIA H200 - 80GB') matched zero /pricing rows on "
        "(gpu_count, chip, memory GB) (1, 'H200', 80) -- status not "
        "recorded"
    ]


def test_collect_availability_fetch_failure_is_fail_open(monkeypatch):
    """Rule 4: a hub fetch failure must never dark the /pricing lane --
    all 83 price observations still record, no availability fields land,
    and the loss is a partial_error."""
    _route_fetch(
        monkeypatch,
        FIXTURE.read_text(),
        hub_exc=RuntimeError("hub unreachable"),
    )
    out = civo_module.collect()
    assert len(out["observations"]) == 83
    assert not any(
        "availability_status" in o["extra"] for o in out["observations"]
    )
    assert out["partial_errors"] == [
        "availability annotation failed (price lane unaffected): "
        "hub unreachable"
    ]


def test_collect_availability_reshape_is_fail_open(monkeypatch):
    """The fail-closed fence INSIDE the availability parse (zero
    Status-columned pricingTable blocks) surfaces as a partial_error at
    the collect() boundary -- prices still record."""
    _route_fetch(
        monkeypatch,
        FIXTURE.read_text(),
        hub_body="<html><p>marketing rewrite, no flight data</p></html>",
    )
    out = civo_module.collect()
    assert len(out["observations"]) == 83
    assert not any(
        "availability_status" in o["extra"] for o in out["observations"]
    )
    assert len(out["partial_errors"]) == 1
    assert "availability annotation failed" in out["partial_errors"][0]
    assert "refusing to annotate" in out["partial_errors"][0]


# ---- availability fail-closed pins, on minimal synthetic RSC pages ----


def _rsc_page(*payload_parts):
    """A minimal Next.js page: each part becomes one self.__next_f.push
    chunk whose JS string literal is built exactly like the real page's
    (JSON.stringify quoting)."""
    scripts = "".join(
        f"<script>self.__next_f.push([1,{json.dumps(part)}])</script>"
        for part in payload_parts
    )
    return f"<html><body>{scripts}</body></html>"


def _hub_table(rows, header=("Model", "Status", "On demand", "Commitment")):
    return {
        "blockType": "pricingTable",
        "caption": "All NVIDIA GPUs pricing by Civo",
        "headerRow": [{"headerText": h} for h in header],
        "rows": rows,
    }


def _hub_row(label, detail, status="In stock"):
    return {
        "rowVariant": "cells",
        "cells": [
            {
                "variant": "text",
                "primaryText": label,
                "secondaryText": detail,
            },
            {
                "variant": "status",
                "statusText": status,
                "statusIcon": "lightning",
            },
            {"variant": "price", "hourlyPrice": "$$9.99"},
        ],
    }


def _hub_payload(*blocks):
    return "2f:" + json.dumps(list(blocks), separators=(",", ":"))


_GOOD_HUB_ROW = _hub_row("Small L40s", "1 x NVIDIA L40S - 48GB")


def test_synthetic_hub_round_trip():
    body = _rsc_page(_hub_payload(_hub_table([_GOOD_HUB_ROW])))
    status_rows, partial_errors = parse_civo_availability(body)
    assert partial_errors == []
    assert status_rows == [
        {
            "label": "Small L40s",
            "detail": "1 x NVIDIA L40S - 48GB",
            "status": "In stock",
        }
    ]


def test_zero_status_columned_tables_raises():
    """A pricingTable block whose headerRow lost the Status column is the
    reshape fence -- fail closed, never an empty-handed success."""
    table = _hub_table(
        [_GOOD_HUB_ROW], header=("Model", "On demand", "Commitment")
    )
    with pytest.raises(RuntimeError, match="0 with a Status headerRow"):
        parse_civo_availability(_rsc_page(_hub_payload(table)))


def test_no_pricing_table_block_at_all_raises():
    with pytest.raises(RuntimeError, match="0 with a Status headerRow"):
        parse_civo_availability(
            _rsc_page("2f:" + json.dumps({"blockType": "sectionHeader"}))
        )


def test_undecodable_pricing_table_block_raises():
    """The anchor present but the JSON behind it broken = RSC payload
    reshaped -- raise, never scan past garbage."""
    body = _rsc_page('2f:{"blockType":"pricingTable",###broken###')
    with pytest.raises(RuntimeError, match="not decodable"):
        parse_civo_availability(body)


def test_status_table_without_rows_list_raises():
    table = _hub_table([_GOOD_HUB_ROW])
    table["rows"] = "reshaped-into-a-string"
    with pytest.raises(RuntimeError, match="lost its rows list"):
        parse_civo_availability(_rsc_page(_hub_payload(table)))


def test_status_table_with_zero_status_rows_raises():
    """A Status-columned table whose rows all lost their status cells
    would otherwise succeed with zero signal -- silence is not health."""
    row = _hub_row("Small L40s", "1 x NVIDIA L40S - 48GB")
    row["cells"] = [row["cells"][0], row["cells"][2]]  # drop status cell
    with pytest.raises(RuntimeError, match="zero status-celled"):
        parse_civo_availability(_rsc_page(_hub_payload(_hub_table([row]))))


def test_hostile_non_string_status_text_is_partial_error_not_crash():
    """Adversarial reshape: statusText arrives as a number. The row is
    skipped with a partial_error; the good row still prints."""
    bad = _hub_row("B200 SXM", "8 x NVIDIA B200 - 180GB", status="In stock")
    bad["cells"][1]["statusText"] = 7
    body = _rsc_page(_hub_payload(_hub_table([bad, _GOOD_HUB_ROW])))
    status_rows, partial_errors = parse_civo_availability(body)
    assert [r["label"] for r in status_rows] == ["Small L40s"]
    assert partial_errors == [
        "availability: hub row 'B200 SXM': status cell statusText "
        "missing or not text -- skipped"
    ]


def test_rsc_dollar_forms_decoded_or_skipped_never_recorded_raw():
    """RSC flight strings escape a literal leading '$' by doubling it (the
    block's own price cells carry '$$1.09'); any other leading '$' is a
    protocol form ('$undefined' sentinel, '$3f' string-dedup reference) --
    the escape decodes to the published text, wire artifacts skip loudly,
    and neither pollutes the availability accrual record."""
    escaped = _hub_row(
        "B200 SXM", "8 x NVIDIA B200 - 180GB", status="$$3.79/GPU reserved"
    )
    sentinel = _hub_row("H200", "8 x NVIDIA H200 - 141GB", status="$undefined")
    ref = _hub_row("A100 80GB", "1 x NVIDIA A100 - 80GB", status="$3f")
    body = _rsc_page(
        _hub_payload(_hub_table([escaped, sentinel, ref, _GOOD_HUB_ROW]))
    )
    status_rows, partial_errors = parse_civo_availability(body)
    assert {r["label"]: r["status"] for r in status_rows} == {
        "B200 SXM": "$3.79/GPU reserved",
        "Small L40s": "In stock",
    }
    assert partial_errors == [
        "availability: hub row 'H200': statusText '$undefined' is an RSC "
        "protocol form, not published text -- skipped",
        "availability: hub row 'A100 80GB': statusText '$3f' is an RSC "
        "protocol form, not published text -- skipped",
    ]


def test_hostile_row_without_text_identity_cell_is_partial_error():
    """Adversarial reshape: the leading text cell became a status cell.
    A status with no identity is unjoinable -- partial_error, never a
    guess."""
    bad = {
        "rowVariant": "cells",
        "cells": [
            {"variant": "status", "statusText": "Sold out"},
        ],
    }
    body = _rsc_page(_hub_payload(_hub_table([bad, _GOOD_HUB_ROW])))
    status_rows, partial_errors = parse_civo_availability(body)
    assert [r["label"] for r in status_rows] == ["Small L40s"]
    assert partial_errors == [
        "availability: hub row with status 'Sold out' lost its leading "
        "text identity cell -- not joinable, skipped"
    ]


def test_hostile_disagreeing_status_cells_in_one_row_is_partial_error():
    bad = _hub_row("A100 40GB", "1 x NVIDIA A100 - 40GB")
    bad["cells"].append({"variant": "status", "statusText": "Sold out"})
    body = _rsc_page(_hub_payload(_hub_table([bad])))
    status_rows, partial_errors = parse_civo_availability(body)
    assert status_rows == []
    assert partial_errors == [
        "availability: hub row 'A100 40GB': 2 status cells disagree on "
        "statusText -- ambiguous, skipped"
    ]


def test_join_refuses_conflicting_statuses_for_one_identity(rows):
    """The H100 SXM/PCIe hazard, hostile variant: two hub rows share the
    (1, H100, 80GB) identity but DISAGREE -- refuse to record either,
    never pick a winner."""
    observations = copy.deepcopy(rows)
    partial_errors = annotate_availability(
        observations,
        [
            {
                "label": "Small H100 SXM",
                "detail": "1 x NVIDIA H100 - 80GB",
                "status": "In stock",
            },
            {
                "label": "Small H100 PCIe",
                "detail": "1 x NVIDIA H100 - 80GB",
                "status": "Sold out",
            },
        ],
    )
    assert not any(
        "availability_status" in o["extra"] for o in observations
    )
    assert len(partial_errors) == 1
    assert "disagree on status" in partial_errors[0]
    assert "'In stock'" in partial_errors[0]
    assert "'Sold out'" in partial_errors[0]


def test_join_skips_unparseable_identity_with_partial_error(rows):
    observations = copy.deepcopy(rows)
    partial_errors = annotate_availability(
        observations,
        [
            {
                "label": "Mystery",
                "detail": "a rack of GPUs",
                "status": "In stock",
            }
        ],
    )
    assert not any(
        "availability_status" in o["extra"] for o in observations
    )
    assert partial_errors == [
        "availability: hub row 'Mystery' ('a rack of GPUs'): cannot "
        "parse (gpu_count, chip, memory GB) identity -- not joined"
    ]
