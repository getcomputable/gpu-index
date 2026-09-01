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

Dedicated-availability fixtures (availability accrual) are trimmed real bodies from
the two dedicated-server availability books, captured live 2026-08-25:
the EU book (api.ovh.com, 8,973 entries / 169 gpu rows / 4.33 MB live)
trimmed to 11 entries (8 gpu + 3 non-gpu) and the US book
(api.us.ovhcloud.com, 15,129 entries / 319 gpu rows / 6.34 MB live)
trimmed to 11 entries (9 gpu + 2 non-gpu). The trim keeps every gpu part
published that day (2x/4x L4, 2x/4x L40S-48g, tesla-v100s, radeon
rx6700xt-12g), the sibling-config state divergence (23scalegpu01-v2
ram-192g gra="72H" vs ram-384g gra="1H-low"), the eu-west-par-a/b/c
"unknown" local zones, the -eu/-ca/-us US plan variants incl. the 94
live -v1-eu configs with EMPTY datacenters lists, the book heads, and
non-gpu rows carrying "comingSoon", "1H-high" and the extended "1440H"
lead-time state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import load_sku_catalog, match_sku
from gpu_index.observatory.sources import ovh
from gpu_index.observatory.sources.ovh import (
    SOURCE_ID,
    dedicated_gpu_summaries,
    parse_dedicated_availabilities,
    parse_ovh,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "observatory" / "ovh"
FR_FIXTURE = FIXTURE_DIR / "order_catalog_public_cloud_fr.json"
US_FIXTURE = FIXTURE_DIR / "order_catalog_public_cloud_us.json"
DEDICATED_EU_FIXTURE = FIXTURE_DIR / "dedicated_availabilities_eu.json"
DEDICATED_US_FIXTURE = FIXTURE_DIR / "dedicated_availabilities_us.json"


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


@pytest.fixture(scope="module")
def dedicated_eu():
    return parse_dedicated_availabilities(
        DEDICATED_EU_FIXTURE.read_text(), book="EU"
    )


@pytest.fixture(scope="module")
def dedicated_us():
    return parse_dedicated_availabilities(
        DEDICATED_US_FIXTURE.read_text(), book="US"
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


def _dedicated_entry(
    fqn,
    plan_code="23scalegpu01-v2",
    gpu="gpu-2xnvidia-l4",
    datacenters=None,
):
    return {
        "fqn": fqn,
        "memory": "ram-192g-ecc-4800",
        "planCode": plan_code,
        "server": plan_code.split("-")[0],
        "storage": "noraid-0",
        "systemStorage": "softraid-2x960nvme-system",
        "gpu": gpu,
        "datacenters": (
            datacenters
            if datacenters is not None
            else [{"availability": "72H", "datacenter": "gra"}]
        ),
    }


def _route_fetch(monkeypatch, dedicated="fixtures"):
    """Monkeypatch ovh.fetch: cloud catalogs always serve the price
    fixtures; the dedicated-availability GETs serve fixtures, raise, or
    return a reshaped body per ``dedicated``."""
    cloud_fr = FR_FIXTURE.read_text()
    cloud_us = US_FIXTURE.read_text()
    dedicated_eu = DEDICATED_EU_FIXTURE.read_text()
    dedicated_us = DEDICATED_US_FIXTURE.read_text()

    def fake_fetch(url, timeout=None):
        if "order/catalog/public/cloud" in url:
            return cloud_fr if "ovhSubsidiary=FR" in url else cloud_us
        assert "dedicated/server/datacenter/availabilities" in url
        if dedicated == "raise":
            raise RuntimeError("socket exploded")
        if dedicated == "reshaped":
            return '{"oops": true}'
        return (
            dedicated_eu
            if url.startswith("https://api.ovh.com/")
            else dedicated_us
        )

    monkeypatch.setattr(ovh, "fetch", fake_fetch)


def test_dedicated_fixture_gpu_row_counts(dedicated_eu, dedicated_us):
    eu_rows, eu_notes = dedicated_eu
    us_rows, us_notes = dedicated_us
    # 8 of the 11 EU fixture entries and 9 of the 11 US entries carry a
    # gpu part; the non-gpu configs (incl. the comingSoon/1H-high/1440H
    # rows) are out of scope by the gpu-key filter, never skips.
    assert len(eu_rows) == 8
    assert len(us_rows) == 9
    assert eu_notes == []
    assert us_notes == []
    for row in eu_rows + us_rows:
        assert set(row) == {"fqn", "planCode", "gpu", "datacenters"}
        for dc in row["datacenters"]:
            assert set(dc) == {"datacenter", "availability"}


def test_dedicated_eu_rows_recorded_verbatim(dedicated_eu):
    rows, _ = dedicated_eu
    by_fqn = {r["fqn"]: r for r in rows}
    # The evidence row, pinned whole -- and its ram-384g sibling showing
    # the same part+datacenter in a DIFFERENT state (72H vs 1H-low), the
    # live proof that per_datacenter must count states, not pick one.
    small = by_fqn[
        "23scalegpu01-v2.ram-192g-ecc-4800.noraid-0"
        ".softraid-2x960nvme-system.gpu-2xnvidia-l4"
    ]
    assert small == {
        "fqn": (
            "23scalegpu01-v2.ram-192g-ecc-4800.noraid-0"
            ".softraid-2x960nvme-system.gpu-2xnvidia-l4"
        ),
        "planCode": "23scalegpu01-v2",
        "gpu": "gpu-2xnvidia-l4",
        "datacenters": [
            {"datacenter": "gra", "availability": "72H"},
            {"datacenter": "rbx", "availability": "unavailable"},
        ],
    }
    big = by_fqn[
        "23scalegpu01-v2.ram-384g-ecc-4800.softraid-2x7680nvme"
        ".softraid-2x960nvme-system.gpu-2xnvidia-l4"
    ]
    assert big["datacenters"][0] == {
        "datacenter": "gra",
        "availability": "1H-low",
    }


def test_dedicated_us_v100s_and_empty_datacenters(dedicated_us):
    rows, notes = dedicated_us
    v100s = [r for r in rows if r["gpu"] == "gpu-4xnvidia-tesla-v100s"]
    assert len(v100s) == 2
    states = {
        r["fqn"]: {
            d["datacenter"]: d["availability"] for d in r["datacenters"]
        }
        for r in v100s
    }
    assert sorted(s["vin"] for s in states.values()) == ["1H-low", "72H"]
    assert all(s["hil"] == "unavailable" for s in states.values())
    # Empty datacenters lists are REAL on the US -v1-eu L4 configs (94
    # live 2026-08-25) -- recorded verbatim, never a reshape note.
    empties = [
        r for r in rows if r["planCode"] == "23scalegpu01-v1-eu"
    ]
    assert len(empties) == 2
    assert all(r["datacenters"] == [] for r in empties)
    assert notes == []


def test_dedicated_summaries_join_dedupe_and_counts(
    dedicated_eu, dedicated_us
):
    by_model, notes = dedicated_gpu_summaries(
        [("EU", dedicated_eu[0]), ("US", dedicated_us[0])]
    )
    assert notes == []
    assert set(by_model) == {"L4", "L40S", "V100S", "RX6700XT"}
    l4_parts = by_model["L4"]
    assert [p["gpu_part"] for p in l4_parts] == [
        "gpu-2xnvidia-l4",
        "gpu-4xnvidia-l4",
    ]
    two_l4 = l4_parts[0]
    assert two_l4["gpu_count"] == 2
    # Recomputed from the fixtures: three EU 2xL4 configs contribute gra
    # 72H / 1H-low / unavailable (one each) and rbx unavailable x3; the
    # US 2xL4 row has an empty datacenters list and contributes nothing.
    assert two_l4["per_datacenter"] == {
        "gra": {"1H-low": 1, "72H": 1, "unavailable": 1},
        "rbx": {"unavailable": 3},
    }
    assert two_l4["plan_codes"] == ["23scalegpu01-v1-eu", "23scalegpu01-v2"]
    v100s = by_model["V100S"]
    assert len(v100s) == 1
    assert v100s[0]["gpu_count"] == 4
    assert v100s[0]["per_datacenter"] == {
        "hil": {"unavailable": 2},
        "vin": {"1H-low": 1, "72H": 1},
    }
    assert v100s[0]["plan_codes"] == ["21hgrai012-us"]
    # RX6700XT rows exist in BOTH books under DIFFERENT fqns (-eu plan
    # variant), so both count -- the fqn+datacenter dedupe must not
    # collapse genuinely distinct configs.
    assert by_model["RX6700XT"][0]["per_datacenter"] == {
        "gra": {"unknown": 2},
        "rbx": {"1H-low": 2},
    }


def test_dedicated_cross_book_dedupe_keeps_first_and_notes_conflict():
    shared_fqn = "23scalegpu01-v2.ram-192g.x.y.gpu-2xnvidia-l4"
    eu_rows = [
        {
            "fqn": shared_fqn,
            "planCode": "23scalegpu01-v2",
            "gpu": "gpu-2xnvidia-l4",
            "datacenters": [
                {"datacenter": "gra", "availability": "72H"},
                {"datacenter": "rbx", "availability": "unavailable"},
            ],
        }
    ]
    us_rows = [
        {
            "fqn": shared_fqn,
            "planCode": "23scalegpu01-v2",
            "gpu": "gpu-2xnvidia-l4",
            "datacenters": [
                {"datacenter": "gra", "availability": "72H"},  # agrees
                {"datacenter": "rbx", "availability": "1H-low"},  # conflict
                {"datacenter": "vin", "availability": "72H"},  # new pair
            ],
        }
    ]
    by_model, notes = dedicated_gpu_summaries(
        [("EU", eu_rows), ("US", us_rows)]
    )
    # Each fqn+datacenter pair counts ONCE; the first-seen (EU) state
    # wins and the disagreement is a visible note, never a silent merge.
    assert by_model["L4"][0]["per_datacenter"] == {
        "gra": {"72H": 1},
        "rbx": {"unavailable": 1},
        "vin": {"72H": 1},
    }
    assert len(notes) == 1
    assert "duplicate fqn+datacenter across books disagrees" in notes[0]
    assert "'unavailable' vs '1H-low'" in notes[0]


def test_dedicated_unknown_vocabulary_noted_not_failed():
    body = json.dumps(
        [
            _dedicated_entry(
                "f1.gpu-2xnvidia-l4",
                datacenters=[
                    {"availability": "sold-out", "datacenter": "gra"},
                    {"availability": "sold-out", "datacenter": "rbx"},
                    # The NNNH lead-time family is known vocabulary --
                    # 240H/720H/1440H were live on US rows 2026-08-25.
                    {"availability": "1440H", "datacenter": "bhs"},
                ],
            )
        ]
    )
    rows, notes = parse_dedicated_availabilities(body, book="EU")
    assert len(rows) == 1  # recorded verbatim despite the new string
    assert rows[0]["datacenters"][0]["availability"] == "sold-out"
    assert len(notes) == 1  # one note per DISTINCT new string, not per row
    assert "'sold-out' (x2)" in notes[0]
    assert "outside the known vocabulary" in notes[0]


def test_dedicated_reshaped_book_raises():
    with pytest.raises(RuntimeError, match="not a JSON list"):
        parse_dedicated_availabilities('{"oops": true}', book="EU")
    # A book with no gpu-part rows at all = lineup pulled or vocabulary
    # changed; refusing beats recording an empty availability book.
    non_gpu = json.dumps(
        [{"fqn": "a.b.c", "planCode": "a", "datacenters": []}]
    )
    with pytest.raises(RuntimeError, match="ZERO gpu-part"):
        parse_dedicated_availabilities(non_gpu, book="EU")
    dup = json.dumps(
        [
            _dedicated_entry("same.fqn.gpu-2xnvidia-l4"),
            _dedicated_entry("same.fqn.gpu-2xnvidia-l4"),
        ]
    )
    with pytest.raises(RuntimeError, match="appears twice"):
        parse_dedicated_availabilities(dup, book="EU")


def test_dedicated_hostile_rows_skipped_with_notes():
    good = _dedicated_entry("good.gpu-2xnvidia-l4")
    non_string_gpu = _dedicated_entry("bad1", gpu=12)
    no_fqn = _dedicated_entry("bad2")
    del no_fqn["fqn"]
    dc_not_list = _dedicated_entry("bad3", datacenters={"gra": "72H"})
    dc_bad_types = _dedicated_entry(
        "bad4",
        datacenters=[{"availability": None, "datacenter": "gra"}],
    )
    body = json.dumps(
        [good, non_string_gpu, no_fqn, dc_not_list, dc_bad_types]
    )
    rows, notes = parse_dedicated_availabilities(body, book="US")
    assert [r["fqn"] for r in rows] == ["good.gpu-2xnvidia-l4"]
    assert len(notes) == 4
    assert "non-string gpu part 12" in notes[0]
    assert "without string fqn/planCode" in notes[1]
    assert "datacenters is not a list" in notes[2]
    assert "datacenters entries reshaped" in notes[3]


def test_dedicated_unparseable_part_kept_out_of_join():
    rows = [
        _dedicated_entry("f1.gpu-2xnvidia-l4"),
        _dedicated_entry("f2.gpu-weird", gpu="gpu-weird"),
        # A new vendor token must fail into a note, never a guessed label.
        _dedicated_entry("f3.intel", gpu="gpu-2xintel-max1550"),
    ]
    parsed, parse_notes = parse_dedicated_availabilities(
        json.dumps(rows), book="EU"
    )
    assert len(parsed) == 3  # all three stay verbatim in book_stats
    assert parse_notes == []
    by_model, notes = dedicated_gpu_summaries([("EU", parsed)])
    assert set(by_model) == {"L4"}
    assert len(notes) == 2
    assert all("excluded from the cloud-observation join" in n for n in notes)
    assert "'gpu-weird'" in notes[0]
    assert "'gpu-2xintel-max1550'" in notes[1]


def test_collect_attaches_baremetal_availability(monkeypatch):
    _route_fetch(monkeypatch)
    out = ovh.collect()
    assert len(out["observations"]) == 28  # 18 FR + 10 US price rows
    assert "partial_errors" not in out  # fixture bodies: no warns, no notes
    l4 = [o for o in out["observations"] if o["sku_identifier"] == "L4"]
    l40s = [o for o in out["observations"] if o["sku_identifier"] == "L40S"]
    assert l4 and l40s
    for obs in l4 + l40s:
        extra = obs["extra"]
        assert extra["dedicated_availability_product_line"] == "baremetal"
        assert [
            p["gpu_part"] for p in extra["dedicated_gpu_availability"]
        ] == (
            ["gpu-2xnvidia-l4", "gpu-4xnvidia-l4"]
            if obs["sku_identifier"] == "L4"
            else ["gpu-2xnvidia-l40s-48g", "gpu-4xnvidia-l40s-48g"]
        )
    # Models with no baremetal part (H100 etc.) get NO availability
    # fields -- absence means "no baremetal twin", never "sold out".
    h100 = [o for o in out["observations"] if o["sku_identifier"] == "H100"]
    assert h100
    for obs in h100:
        assert "dedicated_gpu_availability" not in obs["extra"]
        assert "dedicated_availability_product_line" not in obs["extra"]
    # Channel (a): the verbatim GPU rows persist whole per subsidiary.
    stats = out["book_stats"]["dedicated_gpu_availability"]
    assert len(stats["EU"]) == 8
    assert len(stats["US"]) == 9
    v100s_extra = [
        o
        for o in out["observations"]
        if o["sku_identifier"] == "Tesla V100S"
        and "dedicated_gpu_availability" in o.get("extra", {})
    ]
    # The cloud label is "Tesla V100S", the baremetal label parses to
    # "V100S" -- deliberately NOT joined (the V100S rows still live whole
    # in book_stats); a fuzzy join is the consumer's call, not ours.
    assert v100s_extra == []


def test_collect_price_rows_survive_availability_fetch_failure(monkeypatch):
    """Hard rule: availability must never gate price collection. Both
    dedicated GETs exploding must cost exactly two partial_errors."""
    _route_fetch(monkeypatch, dedicated="raise")
    out = ovh.collect()
    assert len(out["observations"]) == 28
    assert not any(
        "dedicated_gpu_availability" in o.get("extra", {})
        for o in out["observations"]
    )
    assert "dedicated_gpu_availability" not in out["book_stats"]
    errors = out["partial_errors"]
    assert len(errors) == 2
    for book, err in zip(("EU", "US"), errors):
        assert f"dedicated {book}: availability fetch failed" in err
        assert "socket exploded" in err
        assert "price rows unaffected" in err


def test_collect_price_rows_survive_availability_reshape(monkeypatch):
    _route_fetch(monkeypatch, dedicated="reshaped")
    out = ovh.collect()
    assert len(out["observations"]) == 28
    assert "dedicated_gpu_availability" not in out["book_stats"]
    errors = out["partial_errors"]
    assert len(errors) == 2
    for book, err in zip(("EU", "US"), errors):
        assert f"dedicated {book}: availability parse failed" in err
        assert "not a JSON list" in err
        assert "price rows unaffected" in err


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
