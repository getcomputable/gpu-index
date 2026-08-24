# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for the index-basket capture lane.

Collection only: these tests pin (1) the parsers against fixture payloads,
(2) the slot gate, (3) the append-only store discipline, and (4) that the
snapshot document stores per-source prints and NO composite — the ruling
this lane exists to honor.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path

import pytest

from gpu_index.index.config import BasketConfigError, load_basket_config
from gpu_index.common.http import _HttpsOnlyRedirect, read_body_capped
from gpu_index.index.snapshot import normalize_observation
from gpu_index.common.slots import (
    current_slot,
    is_canonical,
    latest_pointer_key,
    slot_key_prefix,
    snapshot_key,
)
from gpu_index.index.snapshot import build_capture_snapshot, derive_basis_pairs
from gpu_index.index.screens import (
    QUARANTINE_REASON,
    apply_jump_screen,
    lowest_eligible,
)
from gpu_index.index.sources import (
    _extraction_consistent,
    parse_e2e,
    parse_hyperstack,
    parse_latitude,
    parse_massedcompute,
    parse_nebius,
    parse_runpod,
    parse_scaleway,
    parse_shadeform_b200,
    parse_vast,
    parse_vast_offers,
    parse_verda,
    select_vast_observations,
)
from gpu_index.common.store import (
    BucketPublishError,
    previous_day_has_snapshots,
    slot_already_captured,
    slot_hours_present,
    upload_capture_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_basket_env(monkeypatch):
    """An exported BASKET_CONFIG_PATH must never leak into these tests, and
    GITHUB_ACTIONS flips warn()/notice() to '::warning::' format — CI itself
    sets it, so any assertion on 'WARNING' text breaks ON CI without this."""
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


# ------------------------------------------------------------------ config


def test_real_config_loads_and_validates():
    cfg = load_basket_config(REPO_ROOT / "config" / "index_basket.json")
    assert cfg["basket_id"] == "b300_annex_a_v0_2"
    basket = [s for s in cfg["sources"] if s["role"] == "b300_basket"]
    assert len(basket) == 8  # eight standing constituents
    assert abs(sum(s["weight"] for s in basket) - 1.0) < 1e-9
    assert cfg["canonical_slot_utc"] in cfg["capture_slots_utc"]
    # Shadeform must never appear as a B300 basket constituent (double-counts
    # Verda) — pool only.
    shadeform = next(s for s in cfg["sources"] if s["source_id"] == "shadeform")
    assert shadeform["role"] == "b200_pool"


def _minimal_cfg(**overrides):
    cfg = {
        "basket_id": "t",
        "bucket_prefix": "index/t",
        "capture_slots_utc": [4, 16],
        "canonical_slot_utc": 16,
        "sources": [
            {"source_id": "a", "role": "b300_basket", "weight": 0.5},
            {"source_id": "b", "role": "b300_basket", "weight": 0.5},
        ],
    }
    cfg.update(overrides)
    return cfg


def test_config_rejects_bad_weights(tmp_path):
    bad = _minimal_cfg()
    bad["sources"][0]["weight"] = 0.7
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(BasketConfigError, match="sum to 1.0"):
        load_basket_config(p)


def test_config_rejects_cadence_outside_ruling(tmp_path):
    """Ruling 2026-08-10: 2-4 captures/day. 1 or 5 slots must be a loud edit."""
    for slots in ([16], [0, 4, 8, 12, 16]):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(_minimal_cfg(capture_slots_utc=slots)))
        with pytest.raises(BasketConfigError, match="2-4"):
            load_basket_config(p)


def test_config_rejects_canonical_outside_slots(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_minimal_cfg(canonical_slot_utc=12)))
    with pytest.raises(BasketConfigError, match="canonical"):
        load_basket_config(p)


def test_config_rejects_claim_threshold_outside_constituent_count(tmp_path):
    for bad in (0, 3, "5"):  # _minimal_cfg has 2 basket constituents
        p = tmp_path / "c.json"
        p.write_text(json.dumps(_minimal_cfg(min_basket_sources_to_claim=bad)))
        with pytest.raises(BasketConfigError, match="min_basket_sources_to_claim"):
            load_basket_config(p)


def test_config_rejects_prefix_outside_index_keyspace(tmp_path):
    """The basket shares the curve bucket — a prefix outside index/ (e.g. a
    typo'd curves/...) must be refused, not written to."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_minimal_cfg(bucket_prefix="curves/dpc-v2/oops")))
    with pytest.raises(BasketConfigError, match="index/"):
        load_basket_config(p)


def test_config_rejects_dot_segment_prefix_escapes(tmp_path):
    """'index/../curves/...' startswith 'index/' but a path-normalizing
    gateway could route it into the curve keyspace — refuse dot segments."""
    for bad in ("index/../curves/dpc-v2", "index/./x", "index//x", "index/x\\y"):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(_minimal_cfg(bucket_prefix=bad)))
        with pytest.raises(BasketConfigError, match="clean path"):
            load_basket_config(p)


def test_explicit_config_path_beats_env(tmp_path, monkeypatch):
    """CLI --config must win over an exported BASKET_CONFIG_PATH — the env
    var silently eating the flag was reproduced killing 4 tests."""
    good = tmp_path / "c.json"
    good.write_text(json.dumps(_minimal_cfg()))
    monkeypatch.setenv("BASKET_CONFIG_PATH", str(tmp_path / "missing.json"))
    cfg = load_basket_config(good)
    assert cfg["basket_id"] == "t"


# ------------------------------------------------------------------ slots


def test_current_slot_picks_latest_mark_at_or_before_now():
    slots = [4, 10, 16, 22]
    assert current_slot(_utc(2026, 8, 10, 16, 24), slots) == (
        datetime(2026, 8, 10).date(),
        16,
    )
    assert current_slot(_utc(2026, 8, 10, 15, 54), slots) == (
        datetime(2026, 8, 10).date(),
        10,
    )
    assert current_slot(_utc(2026, 8, 10, 4, 0), slots) == (
        datetime(2026, 8, 10).date(),
        4,
    )


def test_current_slot_wraps_to_previous_day_before_first_mark():
    day, slot = current_slot(_utc(2026, 8, 10, 2, 24), [4, 10, 16, 22])
    assert (day.isoformat(), slot) == ("2026-08-09", 22)


# ------------------------------------------------------------------ http hardening


def test_redirect_to_plaintext_is_refused():
    """An https source silently redirecting to http would put an audit-grade
    price observation on an unauthenticated hop — must surface as an error."""
    handler = _HttpsOnlyRedirect()
    req = urllib.request.Request("https://example.com/pricing")
    with pytest.raises(urllib.error.HTTPError, match="non-https"):
        handler.redirect_request(
            req, None, 302, "Found", Message(), "http://example.com/pricing"
        )
    new_req = handler.redirect_request(
        req, None, 302, "Found", Message(), "https://example.com/moved"
    )
    assert new_req.full_url == "https://example.com/moved"


class _FakeResp:
    def __init__(self, chunk: bytes, delay: float = 0.0):
        self._chunk = chunk
        self._delay = delay

    def read(self, n: int) -> bytes:
        if self._delay:
            time.sleep(self._delay)
        return self._chunk


def test_read_body_capped_refuses_oversize_bodies():
    with pytest.raises(RuntimeError, match="exceeds"):
        read_body_capped(_FakeResp(b"x" * 70000), limit=100000)


def test_read_body_capped_refuses_slow_drip():
    """urllib's timeout is per-recv; the wall-clock limit is what bounds a
    provider trickling bytes forever."""
    with pytest.raises(RuntimeError, match="still streaming"):
        read_body_capped(
            _FakeResp(b"x", delay=0.03), limit=10**9, wall_clock_limit=0.05
        )


def test_slot_key_layout():
    day = datetime(2026, 8, 10).date()
    assert (
        snapshot_key("index/b300_basket", day, 16, "20260810T162400Z")
        == "index/b300_basket/snapshots/2026-08-10/slot16-20260810T162400Z.json"
    )
    assert slot_key_prefix("index/b300_basket", day, 4).endswith("/slot04-")
    assert latest_pointer_key("index/b300_basket") == "index/b300_basket/latest.json"
    assert is_canonical(16, 16) and not is_canonical(4, 16)


# ------------------------------------------------------------------ parsers

VERDA_HTML = (
    '{"@type":"Offer","name":"1x B300 SXM6 268GB on-demand","priceCurrency":"USD","price":7.50}'
    '{"@type":"Offer","name":"1x B300 SXM6 268GB on-demand","priceCurrency":"USD","price":7.50}'
    '{"@type":"Offer","name":"1x B300 SXM6 268GB spot","priceCurrency":"USD","price":3.75}'
    '{"@type":"Offer","name":"2x B300 SXM6 268GB on-demand","priceCurrency":"USD","price":15.00}'
    '{"@type":"Offer","name":"1x B200 SXM5 180GB on-demand","priceCurrency":"USD","price":5.98}'
    '{"@type":"Offer","name":"1x B300 SXM6 268GB serverless spot","priceCurrency":"USD","price":4.13}'
    '{"@type":"Offer","name":"16x B300 SXM6 268GB instant cluster","priceCurrency":"USD","price":120.00}'
    '{"@type":"Offer","name":"GB300 SXM6 288GB on-demand","priceCurrency":"USD","price":8.62}'
)


def test_parse_verda_dedupes_normalizes_and_labels_tiers():
    rows = parse_verda(VERDA_HTML)
    assert [
        (r["sku"], r["tier"], r["price_usd_gpu_hr"], r["gpu_count_basis"])
        for r in rows
    ] == [
        ("B300", "on-demand", 7.5, 1),
        ("B300", "spot", 3.75, 1),
        ("B300", "on-demand", 7.5, 2),  # 2x node $15 normalized per-GPU
        ("B200", "on-demand", 5.98, 1),
    ]
    # serverless-spot skipped (would pollute the spot tier), instant cluster
    # skipped, GB300 (distinct Grace-Blackwell sku) normalized away,
    # duplicate offer dropped


def test_parse_verda_records_non_usd_natively():
    """A Nordic operator flipping its JSON-LD to EUR must never be silently
    recorded as USD."""
    html = (
        '{"@type":"Offer","name":"1x B300 SXM6 268GB on-demand",'
        '"priceCurrency":"EUR","price":7.10}'
    )
    rows = parse_verda(html)
    assert rows[0]["currency"] == "EUR"
    assert rows[0]["price_usd_gpu_hr"] is None
    assert rows[0]["price_native_per_gpu_hr"] == 7.1


def test_parse_verda_missing_currency_is_unknown_never_assumed_usd():
    """priceCurrency dropped from the JSON-LD must degrade visibly
    (currency UNKNOWN, no USD print) — not silently enter the USD series."""
    html = '{"@type":"Offer","name":"1x B300 SXM6 268GB on-demand","price":7.50}'
    rows = parse_verda(html)
    assert rows[0]["currency"] == "UNKNOWN"
    assert rows[0]["price_usd_gpu_hr"] is None
    assert rows[0]["price_native_per_gpu_hr"] == 7.5


def test_parse_verda_currency_search_never_bleeds_into_next_offer():
    """The currency lookup is bounded at THIS offer's closing brace — the
    next offer's EUR must not label this offer's print."""
    html = (
        '{"@type":"Offer","name":"1x B300 SXM6 268GB on-demand","price":7.50}'
        '{"@type":"Offer","name":"other","priceCurrency":"EUR","price":1.00}'
    )
    rows = parse_verda(html)
    assert rows[0]["currency"] == "UNKNOWN"


RUNPOD_BODY = json.dumps(
    {
        "data": {
            "gpuTypes": [
                {"id": "b300", "displayName": "NVIDIA B300", "memoryInGb": 288, "securePrice": 7.39},
                {"id": "b200", "displayName": "B200", "memoryInGb": 180, "securePrice": 5.98},
                {"id": "gb300", "displayName": "GB300 NVL72", "memoryInGb": 288, "securePrice": 11.0},
                {"id": "h100", "displayName": "H100 80GB", "memoryInGb": 80, "securePrice": 1.99},
                {"id": "b300x", "displayName": "B300 (paused)", "memoryInGb": 288, "securePrice": None},
            ]
        }
    }
)


def test_parse_runpod_secure_only():
    rows = parse_runpod(RUNPOD_BODY)
    assert [(r["sku"], r["price_usd_gpu_hr"]) for r in rows] == [
        ("B300", 7.39),
        ("B200", 5.98),
    ]
    # GB300 excluded, H100 out of basket scope, null securePrice skipped;
    # communityPrice is never even requested (excluded tier).


VAST_BODY = json.dumps(
    {
        "offers": [
            {"num_gpus": 8, "dph_total": 65.04, "geolocation": "US"},
            {"num_gpus": 16, "dph_total": 140.8, "geolocation": "Iceland"},
        ]
    }
)

# The REAL B300 book as fetched live 2026-08-15 (incident forensics):
# 9 offers, 5 machines, 4 hosts, $6.25-$10.94/GPU-hr simultaneously. Host
# 543558 prices ~$10.94/GPU at every slice size; the cheapest hosts are
# verification="unverified".
VAST_LIVE_BOOK_2026_08_15 = json.dumps(
    {
        "offers": [
            {"id": 47779742, "machine_id": 144429, "host_id": 543558, "num_gpus": 1, "dph_total": 10.938888888888888, "geolocation": ", CA", "verified": None, "verification": "verified"},
            {"id": 47419090, "machine_id": 144888, "host_id": 620186, "num_gpus": 2, "dph_total": 15.002083333333335, "geolocation": "Utah, US", "verified": None, "verification": "verified"},
            {"id": 47779740, "machine_id": 144429, "host_id": 543558, "num_gpus": 2, "dph_total": 21.87638888888889, "geolocation": ", CA", "verified": None, "verification": "verified"},
            {"id": 47676702, "machine_id": 147162, "host_id": 1801, "num_gpus": 4, "dph_total": 25.002083333333335, "geolocation": "Taiwan, TW", "verified": None, "verification": "unverified"},
            {"id": 47419091, "machine_id": 144888, "host_id": 620186, "num_gpus": 4, "dph_total": 30.002083333333335, "geolocation": "Utah, US", "verified": None, "verification": "verified"},
            {"id": 47779738, "machine_id": 144429, "host_id": 543558, "num_gpus": 4, "dph_total": 43.75138888888889, "geolocation": ", CA", "verified": None, "verification": "verified"},
            {"id": 47676700, "machine_id": 147162, "host_id": 1801, "num_gpus": 8, "dph_total": 50.002083333333335, "geolocation": "Taiwan, TW", "verified": None, "verification": "unverified"},
            {"id": 46949777, "machine_id": 146594, "host_id": 246706, "num_gpus": 8, "dph_total": 55.002083333333335, "geolocation": "Texas, US", "verified": None, "verification": "unverified"},
            {"id": 47106639, "machine_id": 144407, "host_id": 543558, "num_gpus": 8, "dph_total": 85.00555555555556, "geolocation": ", CA", "verified": None, "verification": "verified"},
        ]
    }
)


def test_parse_vast_normalizes_per_gpu():
    rows = parse_vast(VAST_BODY, "B300")
    assert rows[0]["price_usd_gpu_hr"] == pytest.approx(8.13)
    assert rows[0]["gpu_count_basis"] == 8
    assert rows[0]["raw_unit"] == "usd_per_instance_hr"
    assert rows[0]["raw_value"] == "65.04"
    assert rows[1]["price_usd_gpu_hr"] == pytest.approx(8.8)


def test_parse_vast_dedups_by_machine_and_ranks_per_gpu():
    """THE 08-13 incident pin: against the real book, the
    recorded print must be $6.2503 (Taiwan, TRUE 8-GPU basis) — never host
    543558's $10.94 slice pricing, and never an instance-total ranking."""
    rows = parse_vast(VAST_LIVE_BOOK_2026_08_15, "B300")
    # One row per machine, cheapest machines first, per-GPU ranked.
    assert [
        (r["machine_id"], r["price_usd_gpu_hr"], r["gpu_count_basis"])
        for r in rows
    ] == [
        (147162, pytest.approx(6.2503, abs=1e-4), 8),
        (146594, pytest.approx(6.8753, abs=1e-4), 8),
        (144888, pytest.approx(7.5005, abs=1e-4), 4),
        (144407, pytest.approx(10.6257, abs=1e-4), 8),
        (144429, pytest.approx(10.9378, abs=1e-4), 4),
    ]
    # The would-be daily print (lowest eligible) is the Taiwan 8x box.
    lowest = min(rows, key=lambda r: r["price_usd_gpu_hr"])
    assert (lowest["host_id"], lowest["region"]) == (1801, "Taiwan, TW")
    assert lowest["verification"] == "unverified"  # metadata, not a screen
    assert lowest["offer_id"] == 47676700
    # Identity fields ride every row (L2).
    assert all(
        r["machine_id"] and r["host_id"] and r["offer_id"] for r in rows
    )
    # No machine appears twice.
    machines = [r["machine_id"] for r in rows]
    assert len(machines) == len(set(machines))


def test_extraction_consistency_tripwire():
    """L0: recorded price x basis must reproduce the raw offer total —
    fires only if a future edit derives price and raw_value from different
    fields (exactly the 08-13 failure class)."""
    assert _extraction_consistent(6.2503, 8, 50.002083333333335)
    assert not _extraction_consistent(10.9382, 8, 21.87638888888889)


def test_select_vast_never_collapses_unidentified_offers():
    """Offers without a machine_id must all survive dedup — collapsing
    them would silently drop real machines."""
    candidates = parse_vast_offers(VAST_BODY)
    assert all(c["machine_id"] is None for c in candidates)
    rows = select_vast_observations(candidates, "B300")
    assert len(rows) == 2


HYPERSTACK_HTML = (
    "<table><tr><td>NVIDIA</td><td>B300</td><td>288</td><td>8</td><td>1.5</td>"
    "<td>$7.40</td></tr><tr><td>NVIDIA</td><td>B200</td><td>180</td><td>8</td>"
    "<td>1.2</td><td>$4.20</td></tr><tr><td>NVIDIA</td><td>A100</td><td>80</td>"
    "<td>8</td><td>1.0</td><td>$1.35</td></tr></table>"
)


def test_parse_hyperstack_rows():
    rows = parse_hyperstack(HYPERSTACK_HTML)
    assert [(r["sku"], r["price_usd_gpu_hr"]) for r in rows] == [
        ("B300", 7.4),
        ("B200", 4.2),
    ]


def test_parse_hyperstack_excludes_reserved_style_rows():
    """The exact-3-numeric-fields fence is load-bearing: RESERVED price rows
    (no spec columns) must never be recorded as on-demand prints."""
    html = HYPERSTACK_HTML + "<tr><td>NVIDIA</td><td>B300</td><td>$5.10</td></tr>"
    rows = parse_hyperstack(html)
    assert [(r["sku"], r["price_usd_gpu_hr"]) for r in rows] == [
        ("B300", 7.4),
        ("B200", 4.2),
    ]


def test_comma_prices_parse_whole_not_truncated():
    """'$1,299' must parse as 1299 (out of band, flagged) — '[\\d.]+' style
    regexes would truncate it to 1, an in-band plausible-but-wrong print."""
    html = (
        "<table><tr><td>NVIDIA</td><td>B300</td><td>288</td><td>8</td>"
        "<td>1.5</td><td>$1,299</td></tr></table>"
    )
    rows = parse_hyperstack(html)
    assert rows[0]["price_usd_gpu_hr"] == 1299.0


SHADEFORM_HTML = (
    '{\\"cloud\\":\\"nebius\\",\\"gpu_type\\":\\"B200\\",\\"num_gpus\\":8,'
    '\\"hourly_price\\":4800,\\"availability\\":[{\\"display_name\\":\\"US-East\\"}],'
    '\\"deployment_type\\":\\"vm\\"}'
    '{\\"cloud\\":\\"datacrunch\\",\\"gpu_type\\":\\"B300\\",\\"num_gpus\\":8,'
    '\\"hourly_price\\":6000,\\"availability\\":[{\\"display_name\\":\\"FIN-01\\"}],'
    '\\"deployment_type\\":\\"vm\\"}'
    '{\\"cloud\\":\\"scaleway\\",\\"gpu_type\\":\\"B200\\",\\"num_gpus\\":4,'
    '\\"hourly_price\\":2600,\\"availability\\":[{\\"display_name\\":\\"EU-West\\"}],'
    '\\"deployment_type\\":\\"vm\\"}'
)


def test_parse_shadeform_b200_only_and_cents_math():
    rows = parse_shadeform_b200(SHADEFORM_HTML)
    assert [(r["sku"], r["price_usd_gpu_hr"], r["region"]) for r in rows] == [
        ("B200", 6.0, "US"),
        ("B200", 6.5, "non-US"),
    ]
    # The B300 blob (Verda re-published) must never be recorded here.
    assert all("via " in r["notes"] for r in rows)


def test_parse_vast_skips_offers_without_gpu_count():
    """A missing/zero/absurd num_gpus must never default to 1 — that would
    record a whole-instance price as a per-GPU print."""
    body = json.dumps(
        {
            "offers": [
                {"num_gpus": 0, "dph_total": 55.0},
                {"dph_total": 60.0},
                {"num_gpus": 400, "dph_total": 4000.0},
                {"num_gpus": True, "dph_total": 55.0},
                {"num_gpus": 8.0, "dph_total": 65.04, "geolocation": "US"},
            ]
        }
    )
    rows = parse_vast(body, "B300")
    assert len(rows) == 1
    assert rows[0]["price_usd_gpu_hr"] == pytest.approx(8.13)
    assert rows[0]["gpu_count_basis"] == 8  # integral float accepted as int


def test_parse_vast_skips_spot_bids():
    """A spot BID recorded as an on-demand ask would bias the print low."""
    body = json.dumps(
        {
            "offers": [
                {"num_gpus": 8, "dph_total": 40.0, "is_bid": True},
                {"num_gpus": 8, "dph_total": 65.04, "geolocation": "US"},
            ]
        }
    )
    rows = parse_vast(body, "B300")
    assert [(r["price_usd_gpu_hr"]) for r in rows] == [pytest.approx(8.13)]


def test_collect_vast_single_call_per_sku_logs_candidates(monkeypatch, capsys):
    """ONE unfiltered query per sku (the old 8x-preferred query
    short-circuited on a single expensive host and the verified filter
    dropped the cheapest hosts); the FULL candidate set is logged with
    identity; a dead sku surfaces in partial_errors instead of silently
    thinning the source."""
    import gpu_index.index.sources as sources_mod

    calls = []

    def fake_fetch(url, timeout=None, **kwargs):
        calls.append(urllib.parse.unquote(url))
        if "B200" in url:
            raise RuntimeError("B200 feed down")
        return VAST_LIVE_BOOK_2026_08_15

    monkeypatch.setattr(sources_mod, "fetch", fake_fetch)
    result = sources_mod.collect_vast()
    assert len(calls) == 2  # one per sku, never a second preferred-basis call
    assert all("verified" not in c for c in calls)  # broken filter is gone
    assert all("num_gpus" not in c for c in calls)  # no 8x preference
    assert len(result["observations"]) == 5
    assert result["observations"][0]["price_usd_gpu_hr"] == pytest.approx(6.2503, abs=1e-4)
    assert any("B200" in e for e in result["partial_errors"])
    out = capsys.readouterr().out
    assert out.count("vast candidate B300") == 9  # full set, pre-dedup
    assert "machine=147162" in out and "host=1801" in out


def test_collect_vast_log_lines_strip_control_chars(monkeypatch, capsys):
    """Remote geo strings reach the job log — a newline + '::' sequence is
    a GH Actions workflow command and must never survive printing."""
    import gpu_index.index.sources as sources_mod

    body = json.dumps(
        {
            "offers": [
                {
                    # ids come off the wire too: a hostile response must
                    # not inject via ANY field on the candidate log line.
                    "id": "1\n::stop-commands::a",
                    "machine_id": "2\n::error::b",
                    "host_id": "3\n::add-mask::c",
                    "num_gpus": 8,
                    "dph_total": 50.0,
                    "geolocation": "evil\n::stop-commands::x",
                    "verification": "verified",
                }
            ]
        }
    )
    monkeypatch.setattr(
        sources_mod, "fetch", lambda url, timeout=None, **kw: body
    )
    result = sources_mod.collect_vast()
    out = capsys.readouterr().out
    assert "\n::stop-commands" not in out
    assert "\n::error" not in out
    assert "\n::add-mask" not in out
    # The stored observation keeps the raw string (raw is raw).
    assert result["observations"][0]["region"] == "evil\n::stop-commands::x"


NEBIUS_HTML = (
    'chrome [\\"Item\\",\\"vCPUs\\",\\"RAM, GB\\",\\"Preemptible, GPU-hour\\",\\"On-demand, GPU-hour\\"],'
    '[\\"NVIDIA HGX B300\\",\\"24\\",\\"346\\",\\"$4.30\\",\\"$7.85\\"],'
    '[\\"NVIDIA HGX B200\\",\\"20\\",\\"224\\",\\"$3.95\\",\\"$7.15\\"],'
    '[\\"NVIDIA HGX B300\\",\\"24\\",\\"346\\",\\"$4.30\\",\\"$7.85\\"] tail'
)


def test_parse_nebius_reads_both_tiers_and_dedupes():
    rows = parse_nebius(NEBIUS_HTML)
    assert [(r["sku"], r["tier"], r["price_usd_gpu_hr"]) for r in rows] == [
        ("B300", "on-demand", 7.85),
        ("B300", "preemptible", 4.3),
        ("B200", "on-demand", 7.15),
        ("B200", "preemptible", 3.95),
    ]
    assert all(r["gpu_count_basis"] == 1 for r in rows)  # per GPU-hour already


def test_parse_nebius_fails_closed_on_header_change():
    """ANY header change must raise, never silently record the wrong tier —
    including a new price column inserted BEFORE the pinned pair, which a
    substring check would have waved through."""
    with pytest.raises(RuntimeError, match="header"):
        parse_nebius('[\\"NVIDIA HGX B300\\",\\"24\\",\\"346\\",\\"$4.30\\",\\"$7.85\\"]')
    reordered = NEBIUS_HTML.replace(
        '\\"RAM, GB\\",\\"Preemptible, GPU-hour\\"',
        '\\"RAM, GB\\",\\"Reserved, GPU-hour\\",\\"Preemptible, GPU-hour\\"',
    )
    with pytest.raises(RuntimeError, match="header"):
        parse_nebius(reordered)


def test_parse_nebius_rejects_rows_with_extra_price_column():
    """Even with the header intact, a row carrying a third price field must
    not match — recording reserved-as-preemptible would be plausible and
    silently wrong."""
    html = (
        'chrome [\\"Item\\",\\"vCPUs\\",\\"RAM, GB\\",\\"Preemptible, GPU-hour\\",\\"On-demand, GPU-hour\\"],'
        '[\\"NVIDIA HGX B300\\",\\"24\\",\\"346\\",\\"$2.00\\",\\"$4.30\\",\\"$7.85\\"] tail'
    )
    assert parse_nebius(html) == []


SCALEWAY_BODY = json.dumps(
    {
        "servers": {
            "B300-SXM-8-288G": {
                "gpu": 8,
                "hourly_price": 60.0,
                "gpu_info": {"gpu_name": "B300-SXM", "gpu_memory": 309237645312},
            },
            "B300-SXM-2-288G": {
                "gpu": 2,
                "hourly_price": 18.96,
                "gpu_info": {"gpu_name": "B300-SXM", "gpu_memory": 309237645312},
            },
            "H100-2-80G": {
                "gpu": 2,
                "hourly_price": 5.0,
                "gpu_info": {"gpu_name": "H100-SXM", "gpu_memory": 85899345920},
            },
        }
    }
)


def test_parse_scaleway_is_eur_native_and_never_pretends_usd():
    rows = parse_scaleway([json.loads(SCALEWAY_BODY)["servers"]])
    assert [(r["gpu_count_basis"], r["price_native_per_gpu_hr"]) for r in rows] == [
        (2, 9.48),
        (8, 7.5),
    ]
    for r in rows:
        assert r["currency"] == "EUR"
        assert r["price_usd_gpu_hr"] is None  # FX is not a collector decision
        assert r["raw_unit"] == "eur_per_node_hr"


MASSED_HTML = (
    'pre \\"B300 SXM6-x 8-0\\" specs 280 vCPU $$47.60 tail '
    'dup \\"B300 SXM6-mobile-x 8-1\\" specs 280 vCPU $$47.60 tail'
)


def test_parse_massedcompute_dedupes_mobile_rows_and_normalizes():
    rows = parse_massedcompute(MASSED_HTML)
    assert len(rows) == 1
    assert rows[0]["price_usd_gpu_hr"] == pytest.approx(5.95)
    assert rows[0]["gpu_count_basis"] == 8
    assert rows[0]["raw_value"] == "47.60"


def test_parse_massedcompute_priceless_row_never_steals_neighbor_price():
    """A 'contact us' row with no $$ must be skipped, not priced with the
    NEXT row's number (which its own GPU count would then divide wrongly)."""
    html = (
        'a \\"B300 SXM6-x 2-0\\" contact-us-no-price '
        '\\"B300 SXM6-x 8-1\\" specs $$47.60 tail'
    )
    rows = parse_massedcompute(html)
    assert [(r["gpu_count_basis"], r["price_usd_gpu_hr"]) for r in rows] == [(8, 5.95)]


def test_parse_massedcompute_ambiguous_multi_price_row_is_skipped():
    """Two $$ figures in one row (promo + strikethrough) is ambiguous —
    skip rather than guess which number is real."""
    html = '\\"B300 SXM6-x 8-0\\" was $$59.50 now $$47.60 tail'
    assert parse_massedcompute(html) == []


LATITUDE_HTML = (
    'pre \\"slug\\":\\"g4-b300-large\\",\\"name\\":\\"g4.b300.large\\" mid '
    '\\"gpu\\":{\\"count\\":8,\\"type\\":\\"NVIDIA HGX B300\\",\\"vram_per_gpu\\":288} '
    '{\\"name\\":\\"United States\\",\\"deploys_instantly\\":[],\\"in_stock\\":[],'
    '\\"stock_level\\":\\"unavailable\\",\\"pricing\\":{\\"USD\\":{\\"hour\\":128,'
    '\\"month\\":46720,\\"year\\":392448}}} available_operating_systems tail'
)


def test_parse_latitude_records_hourly_and_labeled_commit_tier():
    rows = parse_latitude(LATITUDE_HTML)
    assert [(r["tier"], r["price_usd_gpu_hr"], r["gpu_count_basis"]) for r in rows] == [
        ("on-demand", 16.0, 8),  # $128/node-hr / 8 — the true on-demand print
        ("monthly-commit", 8.0, 8),  # $46,720/mo / 730h / 8 — the scoping ballpark
    ]
    assert "stock_level=unavailable" in rows[0]["notes"]


def test_parse_latitude_fails_loud_when_blob_missing():
    with pytest.raises(RuntimeError, match="g4-b300"):
        parse_latitude("<html>no plans here</html>")


E2E_HTML = (
    "<div>NVIDIA B200</div><div>192</div><div>32</div><div>400</div>"
    "<div>$6.99</div><div>$4,983.16</div><div>$55,089.45</div>"
)


def test_parse_e2e_takes_hourly_cell_with_row_identity_check():
    rows = parse_e2e(E2E_HTML)
    assert len(rows) == 1
    assert rows[0]["price_usd_gpu_hr"] == 6.99
    assert rows[0]["raw_value"] == "$6.99"


def test_parse_e2e_fails_loud_without_vram_identity():
    """A column reorder must never silently promote the monthly price."""
    with pytest.raises(RuntimeError, match="identity"):
        parse_e2e("<div>NVIDIA B200</div><div>32</div><div>$4,983.16</div>")


def test_parse_e2e_monthly_ratio_pin_catches_column_reorder():
    """Monthly-first column order: the first $ cell would be ~11x the next,
    nowhere near the ~730 hourly-to-monthly ratio — must raise."""
    html = (
        "<div>NVIDIA B200</div><div>192</div><div>32</div><div>400</div>"
        "<div>$4,983.16</div><div>$55,089.45</div><div>$6.99</div>"
    )
    with pytest.raises(RuntimeError, match="cross-check"):
        parse_e2e(html)


def test_collector_refuses_empty_parse():
    """A silently reshaped page must be an ERROR, not a healthy zero-print."""
    from gpu_index.index.sources import _result

    with pytest.raises(RuntimeError, match="zero"):
        _result("verda", method="jsonld", url="x", observations=[])


# ------------------------------------------------------------------ snapshot


def _cfg_for_snapshot():
    return {
        "basket_id": "b300_annex_a_v0_2",
        "methodology_doc": "docs/annex_a_b300_index_basket.md",
        "bucket_prefix": "index/b300_basket",
        "capture_slots_utc": [4, 16],
        "canonical_slot_utc": 16,
        "sources": [
            {"source_id": "verda", "display_name": "Verda", "role": "b300_basket", "weight": 0.5, "source_type": "direct_principal"},
            {"source_id": "vast", "display_name": "Vast.ai", "role": "b300_basket", "weight": 0.5, "source_type": "marketplace"},
            {"source_id": "computedesk", "display_name": "Compute Desk", "role": "b200_pool", "weight": None, "source_type": "index_provider"},
        ],
    }


def _snapshot():
    results = [
        {
            "source_id": "verda",
            "status": "ok",
            "method": "jsonld",
            "url": "https://verda.com/pricing",
            "first_party_observation": True,
            "fetched_at": "2026-08-10T16:24:01Z",
            "observations": [
                {"sku": "B300", "price_usd_gpu_hr": 7.5, "raw_value": "7.50", "tier": "on-demand"},
                {"sku": "B200", "price_usd_gpu_hr": 6.1, "raw_value": "6.10", "tier": "on-demand"},
            ],
        },
        {
            "source_id": "vast",
            "status": "error",
            "error": "URLError: timed out",
            "observations": [],
        },
        {
            "source_id": "computedesk",
            "status": "unimplemented",
            "error": "collector not implemented yet",
            "observations": [],
        },
    ]
    return build_capture_snapshot(
        config=_cfg_for_snapshot(),
        source_results=results,
        captured_at=_utc(2026, 8, 10, 16, 24),
        run_id="20260810T162400Z",
        slot_date=datetime(2026, 8, 10).date(),
        slot_hour_utc=16,
        canonical=True,
        capturer={"job": "test", "version": "t/0"},
        previous_day_empty=False,
    )


def test_snapshot_stores_per_source_prints_and_no_composite():
    payload = _snapshot()
    # Every configured source appears, failures as visible holes.
    assert [s["source_id"] for s in payload["sources"]] == ["verda", "vast", "computedesk"]
    assert payload["sources_ok"] == ["verda"]
    assert payload["sources_failed"] == ["vast"]
    assert payload["sources_unimplemented"] == ["computedesk"]
    verda = payload["sources"][0]
    assert verda["weight"] == 0.5 and verda["source_type"] == "direct_principal"
    assert verda["observations"][0]["raw_value"] == "7.50"
    # THE ruling: per-source prints only — no composite/index value anywhere.
    assert set(payload.keys()) == {
        "schema_version",
        "kind",
        "basket_id",
        "methodology_doc",
        "captured_at",
        "capture_date",
        "slot_hour_utc",
        "canonical_slot",
        "late_fill",
        "run_id",
        "capturer",
        "sources",
        "basis_pairs",
        "sources_ok",
        "sources_failed",
        "sources_unimplemented",
        "basket_sources_ok",
        "previous_day_empty",
    }
    # Coverage visibility: constituents-only, distinct from pool sources.
    assert payload["basket_sources_ok"] == ["verda"]
    dumped = json.dumps(payload)
    for forbidden in ("index_value", "composite", "weighted_mean", "settlement"):
        assert forbidden not in dumped


def test_snapshot_derives_basis_pair_from_dual_sku_source():
    payload = _snapshot()
    assert payload["basis_pairs"] == [
        {
            "source_id": "verda",
            "b200_usd_gpu_hr": 6.1,
            "b300_usd_gpu_hr": 7.5,
            "ratio_b300_b200": pytest.approx(1.229508, abs=1e-6),
        }
    ]


def test_snapshot_flags_implausible_prints_but_keeps_them():
    pairs = derive_basis_pairs(
        [
            {
                "source_id": "x",
                "status": "ok",
                "observations": [
                    {"sku": "B300", "price_usd_gpu_hr": 900.0, "tier": "on-demand", "implausible": True},
                    {"sku": "B200", "price_usd_gpu_hr": 6.0, "tier": "on-demand", "implausible": False},
                ],
            }
        ]
    )
    assert pairs == []  # implausible print stored upstream but never derived from

    payload = _snapshot()
    obs = payload["sources"][0]["observations"][0]
    assert obs["implausible"] is False


def test_normalize_observation_currency_honesty_and_band_boundaries():
    eur = normalize_observation(
        {"sku": "B300", "price_native_per_gpu_hr": 9.48, "currency": "EUR"}
    )
    assert eur["price_usd_gpu_hr"] is None  # FX is never a collector decision
    assert eur["price_native_per_gpu_hr"] == 9.48
    assert eur["implausible"] is False

    empty = normalize_observation({"sku": "B300"})
    assert empty["price_usd_gpu_hr"] is None
    assert empty["implausible"] is True

    assert normalize_observation({"sku": "B300", "price_usd_gpu_hr": 0.5})["implausible"] is False
    assert normalize_observation({"sku": "B300", "price_usd_gpu_hr": 40.0})["implausible"] is False
    assert normalize_observation({"sku": "B300", "price_usd_gpu_hr": 0.49})["implausible"] is True
    assert normalize_observation({"sku": "B300", "price_usd_gpu_hr": 40.01})["implausible"] is True


# ------------------------------------------------------------------ store


class _NoSuchKey(Exception):
    def __init__(self):
        super().__init__("missing")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.put_order = []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
        self.put_order.append(Key)

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


def test_upload_writes_snapshot_then_pointer_last():
    client = FakeS3()
    payload = _snapshot()
    out = upload_capture_snapshot(
        client, "curves", payload, prefix="index/b300_basket", now=_utc(2026, 8, 10, 16, 25)
    )
    key = "index/b300_basket/snapshots/2026-08-10/slot16-20260810T162400Z.json"
    assert out["snapshot_key"] == key
    assert client.put_order == [key, "index/b300_basket/latest.json"]
    pointer = json.loads(client.objects["index/b300_basket/latest.json"])
    assert pointer["snapshot_key"] == key
    assert pointer["canonical_slot"] is True
    assert pointer["sources_ok"] == ["verda"]
    assert pointer["basket_sources_ok"] == ["verda"]
    assert pointer["published_at"] == "2026-08-10T16:25:00Z"
    stored = json.loads(client.objects[key])
    assert stored["sources"][0]["observations"][0]["price_usd_gpu_hr"] == 7.5


def test_upload_refuses_to_overwrite_different_bytes():
    client = FakeS3()
    payload = _snapshot()
    upload_capture_snapshot(client, "curves", payload, prefix="index/b300_basket")
    mutated = json.loads(json.dumps(payload))
    mutated["sources"][0]["observations"][0]["price_usd_gpu_hr"] = 9.99
    with pytest.raises(BucketPublishError, match="append-only"):
        upload_capture_snapshot(client, "curves", mutated, prefix="index/b300_basket")


def test_pointer_not_moved_when_verify_after_write_fails():
    """Store discipline: readback mismatch must leave latest untouched."""

    class CorruptingS3(FakeS3):
        def put_object(self, Bucket, Key, Body, **kwargs):
            if "snapshots/" in Key:
                Body = Body + b"x"
            super().put_object(Bucket, Key, Body, **kwargs)

    client = CorruptingS3()
    with pytest.raises(BucketPublishError, match="Verify-after-write"):
        upload_capture_snapshot(client, "curves", _snapshot(), prefix="index/b300_basket")
    assert "index/b300_basket/latest.json" not in client.objects


def test_pointer_never_regresses_to_an_older_capture():
    """A late fill of an earlier slot finishing after a newer capture must
    not repoint latest backwards (fresh published_at would mask it)."""
    client = FakeS3()
    newer = _snapshot()
    upload_capture_snapshot(client, "curves", newer, prefix="index/b300_basket")
    older = json.loads(json.dumps(newer))
    older["captured_at"] = "2026-08-10T04:24:00Z"
    older["slot_hour_utc"] = 4
    older["canonical_slot"] = False
    older["run_id"] = "20260810T042400Z-aaaa"
    out = upload_capture_snapshot(client, "curves", older, prefix="index/b300_basket")
    assert out["status"] == "published_pointer_kept"
    pointer = json.loads(client.objects["index/b300_basket/latest.json"])
    assert pointer["slot_hour_utc"] == 16  # still the newer capture
    # The older snapshot itself IS stored (append-only history).
    assert any("slot04-" in k for k in client.objects)


def test_slot_gate_and_missed_day_reads():
    client = FakeS3()
    day = datetime(2026, 8, 10).date()
    prev = datetime(2026, 8, 9).date()
    assert not slot_already_captured(
        client, "curves", prefix="index/b300_basket", day=day, slot_hour=16
    )
    assert not previous_day_has_snapshots(
        client, "curves", prefix="index/b300_basket", day=prev
    )
    upload_capture_snapshot(client, "curves", _snapshot(), prefix="index/b300_basket")
    assert slot_already_captured(
        client, "curves", prefix="index/b300_basket", day=day, slot_hour=16
    )
    assert not slot_already_captured(
        client, "curves", prefix="index/b300_basket", day=day, slot_hour=4
    )
    assert previous_day_has_snapshots(
        client, "curves", prefix="index/b300_basket", day=day
    )


def test_slot_hours_present_reads_coverage():
    """The partial-miss alarm's read: which slots actually recorded."""
    client = FakeS3()
    day = datetime(2026, 8, 10).date()
    assert slot_hours_present(client, "curves", prefix="index/b300_basket", day=day) == set()
    upload_capture_snapshot(client, "curves", _snapshot(), prefix="index/b300_basket")
    assert slot_hours_present(
        client, "curves", prefix="index/b300_basket", day=day
    ) == {16}


# ------------------------------------------------------------------ runner

_RUNNER_CACHE = {}


def _load_runner():
    # Cached: re-executing the script each call would stack sys.path inserts
    # in the shared test process (the script itself also guards them now).
    if "mod" not in _RUNNER_CACHE:
        spec = importlib.util.spec_from_file_location(
            "capture_index_basket", REPO_ROOT / "scripts" / "capture_index_basket.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RUNNER_CACHE["mod"] = mod
    return _RUNNER_CACHE["mod"]


def test_collect_all_partitions_ok_error_unimplemented(monkeypatch):
    runner = _load_runner()

    def ok(timeout):
        return {
            "source_id": "verda",
            "method": "jsonld",
            "url": "u",
            "first_party_observation": True,
            "fetched_at": "t",
            "observations": [{"sku": "B300", "price_usd_gpu_hr": 7.5}],
        }

    def boom(timeout):
        raise RuntimeError("feed down")

    monkeypatch.setattr(runner, "COLLECTORS", {"verda": ok, "vast": boom})
    cfg = _cfg_for_snapshot()
    results = runner.collect_all(cfg)
    by_id = {r["source_id"]: r for r in results}
    assert by_id["verda"]["status"] == "ok"
    assert by_id["vast"]["status"] == "error"
    assert "feed down" in by_id["vast"]["error"]
    assert by_id["computedesk"]["status"] == "unimplemented"
    assert all("elapsed_seconds" in r for r in results if r["status"] != "unimplemented")


def test_collect_all_abandons_a_source_past_its_deadline(monkeypatch):
    """The budget is a HARD cap: an in-flight source is abandoned at its
    share, not merely checked between sources — a 20-minute capture would
    runner-kill the chain and delay the next publish firing."""
    runner = _load_runner()

    def fast(timeout):
        return {
            "source_id": "verda",
            "observations": [{"sku": "B300", "price_usd_gpu_hr": 7.5}],
        }

    def hung(timeout):
        time.sleep(5)
        return {"source_id": "vast", "observations": []}

    monkeypatch.setattr(runner, "COLLECTORS", {"verda": fast, "vast": hung})
    results = runner.collect_all(_cfg_for_snapshot(), budget_seconds=0.2)
    by_id = {r["source_id"]: r for r in results}
    assert by_id["verda"]["status"] == "ok"
    assert by_id["vast"]["status"] == "error"
    assert "budget share" in by_id["vast"]["error"]


def test_collect_all_skips_sources_after_budget_spent(monkeypatch):
    runner = _load_runner()

    def hung(timeout):
        time.sleep(5)
        return {"source_id": "verda", "observations": []}

    def never_called(timeout):
        pytest.fail("source ran after the budget was spent")

    monkeypatch.setattr(
        runner, "COLLECTORS", {"verda": hung, "vast": never_called}
    )
    results = runner.collect_all(_cfg_for_snapshot(), budget_seconds=0.1)
    by_id = {r["source_id"]: r for r in results}
    assert "budget share" in by_id["verda"]["error"]
    assert "exhausted before this source ran" in by_id["vast"]["error"]


def test_print_summary_survives_observation_with_no_price():
    """A malformed single observation must never crash the warn-only lane
    after an otherwise-successful collection."""
    runner = _load_runner()
    payload = {
        "sources": [
            {
                "source_id": "x",
                "status": "ok",
                "observations": [normalize_observation({"sku": "B300"})],
            }
        ],
        "basis_pairs": [],
    }
    runner.print_summary(payload)  # must not raise


def test_clean_strips_workflow_command_injection():
    runner = _load_runner()
    cleaned = runner._clean("line\n::add-mask::secret\r::error::spoof")
    assert "\n" not in cleaned and "\r" not in cleaned
    assert cleaned.startswith("line ")


def _run_main(runner, monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["capture_index_basket.py", *argv])
    return runner.main()


def test_main_all_sources_failed_exits_1_without_writing(monkeypatch):
    runner = _load_runner()

    def boom(timeout):
        raise RuntimeError("down")

    monkeypatch.setattr(runner, "COLLECTORS", {"verda": boom})
    monkeypatch.setattr(
        runner, "write_local_snapshot", lambda p: pytest.fail("wrote on all-failed")
    )
    monkeypatch.setattr(
        runner,
        "upload_capture_snapshot",
        lambda *a, **k: pytest.fail("uploaded on all-failed"),
    )
    assert _run_main(runner, monkeypatch, ["--local-only"]) == 1


def test_main_pool_only_success_does_not_claim_the_slot(monkeypatch):
    """All 8 B300 constituents dark + one B200 pool source ok must exit 1 so
    the next firing retries — pool-only data must not mask a dead basket."""
    runner = _load_runner()

    def shadeform_ok(timeout):
        return {
            "source_id": "shadeform",
            "method": "html-json-blobs",
            "url": "u",
            "first_party_observation": False,
            "fetched_at": "t",
            "observations": [{"sku": "B200", "price_usd_gpu_hr": 6.0}],
        }

    monkeypatch.setattr(runner, "COLLECTORS", {"shadeform": shadeform_ok})
    monkeypatch.setattr(
        runner, "write_local_snapshot", lambda p: pytest.fail("slot claimed by pool-only run")
    )
    assert _run_main(runner, monkeypatch, ["--local-only"]) == 1


def _ok_collector(sid):
    def fn(timeout):
        return {
            "source_id": sid,
            "method": "m",
            "url": "u",
            "first_party_observation": True,
            "fetched_at": "t",
            "observations": [{"sku": "B300", "price_usd_gpu_hr": 7.0}],
        }

    return fn


_FIVE_CONSTITUENTS = ["verda", "nebius", "hyperstack", "scaleway", "runpod"]


def test_main_refuses_to_claim_below_min_coverage(monkeypatch):
    """Ruling 2026-08-10: fewer than 5 of 8 constituents must NOT claim the
    slot — a thin snapshot would stop retries and, under earliest-key-wins,
    be the print a reader sees forever."""
    runner = _load_runner()
    four = {sid: _ok_collector(sid) for sid in _FIVE_CONSTITUENTS[:4]}
    monkeypatch.setattr(runner, "COLLECTORS", four)
    monkeypatch.setattr(
        runner, "write_local_snapshot", lambda p: pytest.fail("thin snapshot claimed the slot")
    )
    assert _run_main(runner, monkeypatch, ["--local-only"]) == 1


def test_main_claims_at_min_coverage_and_reports_it(monkeypatch, tmp_path, capsys):
    runner = _load_runner()
    five = {sid: _ok_collector(sid) for sid in _FIVE_CONSTITUENTS}
    monkeypatch.setattr(runner, "COLLECTORS", five)
    written = {}

    def record_local(payload):
        written["payload"] = payload
        return tmp_path / "snapshot.json"

    monkeypatch.setattr(runner, "write_local_snapshot", record_local)
    assert _run_main(runner, monkeypatch, ["--local-only"]) == 0
    assert written["payload"]["basket_sources_ok"] == sorted(_FIVE_CONSTITUENTS)
    out = capsys.readouterr().out
    assert "basket coverage: 5/8 constituents ok (minimum 5" in out


def test_main_dry_run_and_only_source_never_write(monkeypatch):
    runner = _load_runner()

    def verda_ok(timeout):
        return {
            "source_id": "verda",
            "method": "jsonld",
            "url": "u",
            "first_party_observation": True,
            "fetched_at": "t",
            "observations": [{"sku": "B300", "price_usd_gpu_hr": 7.5}],
        }

    monkeypatch.setattr(runner, "COLLECTORS", {"verda": verda_ok})
    monkeypatch.setattr(
        runner, "write_local_snapshot", lambda p: pytest.fail("dry run wrote locally")
    )
    monkeypatch.setattr(
        runner, "upload_capture_snapshot", lambda *a, **k: pytest.fail("dry run uploaded")
    )
    assert _run_main(runner, monkeypatch, ["--dry-run"]) == 0
    # --only-source implies --dry-run: partial snapshots are never recorded.
    assert _run_main(runner, monkeypatch, ["--only-source", "verda"]) == 0


# ----------------------------------------------------------- capture screens


def _scr_obs(
    sku="B300",
    native=7.5,
    currency="USD",
    tier="on-demand",
    implausible=False,
    machine=None,
):
    obs = {
        "sku": sku,
        "price_usd_gpu_hr": native if currency == "USD" else None,
        "price_native_per_gpu_hr": native,
        "currency": currency,
        "tier": tier,
        "implausible": implausible,
        "gpu_count_basis": 8,
        "notes": "",
    }
    if machine is not None:
        obs["machine_id"] = machine
    return obs


def _scr_src(sid, observations, role="b300_basket", status="ok"):
    return {
        "source_id": sid,
        "role": role,
        "status": status,
        "observations": observations,
    }


def _scr_cfg():
    return {
        "target_sku": "B300",
        "calc": {"interruptible_tiers": ["spot", "preemptible"]},
        "capture_screens": {
            "jump_quarantine_pct": 25,
            "jump_corroborate_pct": 10,
            "jump_min_corroborators": 2,
        },
    }


def _scr_ref(sources):
    return {"capture_date": "2026-08-14", "slot_hour_utc": 16, "sources": sources}


def test_jump_screen_quarantines_uncorroborated_jump():
    """THE incident case: one source +59% while every other source is flat
    -> quarantined via the implausible machinery, spot prints untouched
    (already ineligible), raw prices still present."""
    payload = {
        "sources": [
            _scr_src("verda", [_scr_obs(native=7.5)]),
            _scr_src("nebius", [_scr_obs(native=7.85)]),
            _scr_src(
                "vast",
                [
                    _scr_obs(native=10.9382, machine=144429),
                    _scr_obs(native=3.75, tier="spot"),
                ],
            ),
        ]
    }
    reference = _scr_ref(
        [
            _scr_src("verda", [_scr_obs(native=7.5)]),
            _scr_src("nebius", [_scr_obs(native=7.85)]),
            _scr_src("vast", [_scr_obs(native=6.8753, machine=146594)]),
        ]
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    assert [q["source_id"] for q in report["quarantined"]] == ["vast"]
    assert report["quarantined"][0]["book_pct"] == pytest.approx(59.09, abs=0.05)
    assert report["quarantined"][0]["corroborators"] == 0
    vast = payload["sources"][2]
    ondemand, spot = vast["observations"]
    assert ondemand["implausible"] is True
    assert ondemand["quarantined"] == QUARANTINE_REASON
    assert "L5 QUARANTINE" in ondemand["notes"]
    assert ondemand["price_native_per_gpu_hr"] == 10.9382  # raw stays
    assert "quarantined" not in spot  # ineligible tiers never touched
    # The machine the reference print came from is gone from today's book:
    # same-machine is honestly n/a, not zero.
    vast_delta = next(
        d for d in report["deltas"] if d["source_id"] == "vast"
    )
    assert vast_delta["book_pct"] == pytest.approx(59.09, abs=0.05)
    assert vast_delta["same_machine_pct"] is None


def test_jump_screen_market_wide_move_passes():
    """Corroboration gate: a real market event (three sources -30%) must
    pass; two lone movers (one corroborator each, minimum two) must not."""
    def world(moves):
        payload = {
            "sources": [
                _scr_src(sid, [_scr_obs(native=7.5 * (1 + pct / 100.0))])
                for sid, pct in moves.items()
            ]
        }
        reference = _scr_ref(
            [_scr_src(sid, [_scr_obs(native=7.5)]) for sid in moves]
        )
        return payload, reference

    payload, reference = world(
        {"verda": -30, "nebius": -31, "vast": -29, "runpod": -2}
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    assert report["quarantined"] == []

    payload, reference = world(
        {"verda": 30, "nebius": 31, "vast": 0, "runpod": 0}
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    assert sorted(q["source_id"] for q in report["quarantined"]) == [
        "nebius",
        "verda",
    ]


def test_jump_screen_same_machine_delta_pair():
    """The one-line-ends-the-question pair: a new cheapest machine moves
    the book while the reference machine sits still."""
    payload = {
        "sources": [
            _scr_src(
                "vast",
                [
                    _scr_obs(native=6.88, machine=222),  # new cheapest
                    _scr_obs(native=7.57, machine=111),  # ref machine +10%
                ],
            ),
        ]
    }
    reference = _scr_ref(
        [_scr_src("vast", [_scr_obs(native=6.88, machine=111)])]
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    delta = report["deltas"][0]
    assert delta["book_pct"] == pytest.approx(0.0, abs=0.01)
    assert delta["same_machine_pct"] == pytest.approx(10.03, abs=0.05)
    assert report["quarantined"] == []


def test_jump_screen_compares_in_native_terms():
    """An EUR source's move is its own EUR ratio — no FX involved; +26.7%
    uncorroborated quarantines exactly like a USD source."""
    payload = {
        "sources": [
            _scr_src("scaleway", [_scr_obs(native=9.5, currency="EUR")]),
            _scr_src("verda", [_scr_obs(native=7.5)]),
            _scr_src("nebius", [_scr_obs(native=7.85)]),
        ]
    }
    reference = _scr_ref(
        [
            _scr_src("scaleway", [_scr_obs(native=7.5, currency="EUR")]),
            _scr_src("verda", [_scr_obs(native=7.5)]),
            _scr_src("nebius", [_scr_obs(native=7.85)]),
        ]
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    assert [q["source_id"] for q in report["quarantined"]] == ["scaleway"]


def test_jump_screen_fails_open_and_skips_incomparables():
    """No reference -> report-only. Currency change -> not comparable.
    Pool sources and failed sources are never screened."""
    payload = {
        "sources": [
            _scr_src("verda", [_scr_obs(native=30.0)]),  # +300% but no ref
            _scr_src("scaleway", [_scr_obs(native=30.0, currency="EUR")]),
            _scr_src("shadeform", [_scr_obs(native=30.0)], role="b200_pool"),
            _scr_src("nebius", [], status="error"),
        ]
    }
    report = apply_jump_screen(payload, None, config=_scr_cfg())
    assert report["reference"] is None
    assert report["quarantined"] == []
    assert all(d["note"] == "no reference print" for d in report["deltas"])
    assert {d["source_id"] for d in report["deltas"]} == {"verda", "scaleway"}

    reference = _scr_ref(
        [_scr_src("scaleway", [_scr_obs(native=7.5, currency="USD")])]
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    scaleway = next(d for d in report["deltas"] if d["source_id"] == "scaleway")
    assert scaleway["note"] == "currency changed — not comparable"
    assert report["quarantined"] == []
    assert "quarantined" not in payload["sources"][2]["observations"][0]


def test_normalize_observation_preserves_identity_fields():
    """L2: identity must survive snapshot normalization (it used to strip
    unknown keys) — the same-machine delta depends on it."""
    out = normalize_observation(
        {
            "sku": "B300",
            "price_usd_gpu_hr": 6.25,
            "offer_id": 47676700,
            "machine_id": 147162,
            "host_id": 1801,
            "verification": "unverified",
        }
    )
    assert (out["offer_id"], out["machine_id"], out["host_id"]) == (
        47676700,
        147162,
        1801,
    )
    assert out["verification"] == "unverified"
    plain = normalize_observation({"sku": "B300", "price_usd_gpu_hr": 6.25})
    assert "machine_id" not in plain


def test_quarantined_prints_are_excluded_from_the_composite():
    """End of the chain: a quarantined print must not price the source, not
    enter the index, and not enter the filter window — via the SAME
    implausible rule the calc has always had (no methodology change)."""
    from gpu_index.index.composite import compute_day

    cfg = load_basket_config(REPO_ROOT / "config" / "index_basket.json")
    quarantined_obs = _scr_obs(native=10.9382, machine=144429)
    quarantined_obs["implausible"] = True
    quarantined_obs["quarantined"] = QUARANTINE_REASON
    snapshot = {
        "run_id": "r",
        "late_fill": False,
        "sources": [
            _scr_src("verda", [_scr_obs(native=7.5)]),
            _scr_src("vast", [quarantined_obs]),
        ],
    }
    window_history = {}
    payload = compute_day(
        config=cfg,
        day="2026-08-15",
        snapshot=snapshot,
        substituted_from=None,
        window_history=window_history,
        window_currencies={},
        fx_records={},
        weight_state={"prices": {}, "vectors": {}, "mode": "fallback"},
        prior_slot_prints={},
    )
    weights = (payload["index"] or {}).get("renormalized_weights", {})
    assert "vast" not in weights and "verda" in weights
    assert "vast" not in window_history  # a bad print must not poison sigma


def test_main_capture_applies_screens_and_stores_flags(
    monkeypatch, capsys, tmp_path
):
    """Full incident replay through main(): reference snapshot in the
    bucket, vast +59% uncorroborated -> the UPLOADED snapshot carries the
    quarantine flags, the delta lines and the L5 warning are printed, and
    the capture still exits 0 (flag, never fail)."""
    runner = _load_runner()

    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    base["min_basket_sources_to_claim"] = 1
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(base))

    client = FakeS3()
    ref_key = "index/b300_basket/snapshots/2026-08-14/slot16-20260814T161000Z-aaaa.json"
    reference = {
        "capture_date": "2026-08-14",
        "slot_hour_utc": 16,
        "sources": [
            {"source_id": "verda", "status": "ok", "observations": [_scr_obs(native=7.5)]},
            {"source_id": "nebius", "status": "ok", "observations": [_scr_obs(native=7.85)]},
            {"source_id": "vast", "status": "ok", "observations": [_scr_obs(native=6.8753, machine=146594)]},
        ],
    }
    client.objects[ref_key] = json.dumps(reference).encode()
    client.objects["index/b300_basket/latest.json"] = json.dumps(
        {"snapshot_key": ref_key}
    ).encode()

    def verda(timeout):
        return {
            "source_id": "verda",
            "observations": [
                {"sku": "B300", "price_usd_gpu_hr": 7.5, "raw_value": "7.5"},
                {"sku": "B200", "price_usd_gpu_hr": 6.1, "raw_value": "6.1"},
            ],
        }

    def nebius(timeout):
        return {
            "source_id": "nebius",
            "observations": [{"sku": "B300", "price_usd_gpu_hr": 7.85, "raw_value": "7.85"}],
        }

    def vast(timeout):
        return {
            "source_id": "vast",
            "observations": [
                {
                    "sku": "B300",
                    "price_usd_gpu_hr": 10.9382,
                    "raw_value": "87.5056",
                    "raw_unit": "usd_per_instance_hr",
                    "gpu_count_basis": 8,
                    "offer_id": 47779749,
                    "machine_id": 144429,
                    "host_id": 543558,
                    "verification": "verified",
                },
                # Non-target sku stays unflagged, but the quarantined B300
                # print must not feed the recorded basis pair either.
                {"sku": "B200", "price_usd_gpu_hr": 6.0, "raw_value": "6.0"},
            ],
        }

    class StubConfig:
        bucket = "curves"

    monkeypatch.setattr(
        runner, "COLLECTORS", {"verda": verda, "nebius": nebius, "vast": vast}
    )
    monkeypatch.setattr(
        runner.BucketConfig, "from_env", staticmethod(lambda: StubConfig())
    )
    monkeypatch.setattr(runner, "make_client", lambda cfg: client)
    monkeypatch.setattr(
        runner, "utc_now", lambda: _utc(2026, 8, 15, 16, 24)
    )
    monkeypatch.setattr(
        runner, "write_local_snapshot", lambda p: tmp_path / "local.json"
    )
    assert _run_main(runner, monkeypatch, ["--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "delta B300 verda" in out and "delta B300 vast" in out
    assert "+59.1%" in out
    assert "L5 QUARANTINE vast" in out and "WARNING" in out
    assert "vs 2026-08-14 slot16" in out

    stored_key = next(
        k for k in client.put_order if "snapshots/2026-08-15/" in k
    )
    stored = json.loads(client.objects[stored_key])
    by_id = {s["source_id"]: s for s in stored["sources"]}
    vast_b300 = by_id["vast"]["observations"][0]
    assert vast_b300["implausible"] is True
    assert vast_b300["quarantined"] == QUARANTINE_REASON
    assert "L5 QUARANTINE" in vast_b300["notes"]
    assert vast_b300["price_usd_gpu_hr"] == 10.9382  # raw price recorded
    assert by_id["verda"]["observations"][0].get("implausible") is False
    # Non-target sku untouched by the quarantine...
    vast_b200 = by_id["vast"]["observations"][1]
    assert vast_b200.get("implausible") is False
    assert "quarantined" not in vast_b200
    # ...but the recorded basis pairs were re-derived post-screen: vast's
    # B300 median is gone, so only verda (clean dual-sku) keeps a pair.
    assert [p["source_id"] for p in stored["basis_pairs"]] == ["verda"]


def test_lowest_eligible_mirrors_r1_at_capture_time():
    """Interruptible tiers, flagged prints, and priceless rows never become
    the reference or today's would-be print."""
    src = _scr_src(
        "vast",
        [
            _scr_obs(native=3.75, tier="spot"),
            _scr_obs(native=5.0, implausible=True),
            _scr_obs(native=6.8753, machine=146594),
            _scr_obs(native=7.5),
            {"sku": "B300", "tier": "on-demand", "notes": ""},
        ],
    )
    chosen = lowest_eligible(src, "B300", ("spot", "preemptible"))
    assert chosen["price_native_per_gpu_hr"] == 6.8753
    assert lowest_eligible(None, "B300", ()) is None


# ----------------------------------------------------- capture hardening


def test_collect_vast_fetches_descending_tail_when_at_limit(monkeypatch, capsys):
    """A full-limit ascending response may be truncated, and the offers it
    cuts FIRST are the largest totals — the cheap-per-GPU 8x boxes (the
    08-13 burial class). The collector must fetch the descending tail and
    merge, so the true cheapest machine always arrives."""
    import gpu_index.index.sources as sources_mod

    limit = sources_mod.VAST_FETCH_LIMIT
    asc_offers = [
        {
            "id": i, "machine_id": 1000 + i, "host_id": 543558, "num_gpus": 1,
            "dph_total": 10.94 + i * 0.01, "geolocation": ", CA",
            "verification": "verified",
        }
        for i in range(limit)  # exactly at limit -> suspected truncation
    ]
    cheap_8x = {
        "id": 9999, "machine_id": 147162, "host_id": 1801, "num_gpus": 8,
        "dph_total": 50.0021, "geolocation": "Taiwan, TW",
        "verification": "unverified",
    }
    # Descending tail: biggest totals first, overlapping one asc offer.
    desc_offers = [cheap_8x, asc_offers[-1]]
    calls = []

    def fake_fetch(url, timeout=None, **kwargs):
        q = urllib.parse.unquote(url)
        calls.append(q)
        if "B200" in q:
            return json.dumps({"offers": [asc_offers[0]]})  # thin, no tail
        if '"desc"' in q:
            return json.dumps({"offers": desc_offers})
        return json.dumps({"offers": asc_offers})

    monkeypatch.setattr(sources_mod, "fetch", fake_fetch)
    result = sources_mod.collect_vast()
    # B200: 1 call; B300: asc + desc = 2 calls (the old worst case).
    assert len(calls) == 3
    assert sum(1 for c in calls if '"desc"' in c) == 1
    b300 = [o for o in result["observations"] if o["sku"] == "B300"]
    lowest = min(b300, key=lambda o: o["price_usd_gpu_hr"])
    assert lowest["price_usd_gpu_hr"] == pytest.approx(6.2503, abs=1e-4)
    assert lowest["machine_id"] == 147162
    out = capsys.readouterr().out
    assert "book at fetch limit" in out
    # Not book-exceeds-coverage: the desc window was NOT full.
    assert "partial_errors" not in result or not any(
        "exceeds fetch coverage" in e for e in result["partial_errors"]
    )


def test_collect_vast_flags_book_wider_than_both_windows(monkeypatch):
    """asc AND desc windows both full: mid-book offers may be missing —
    must surface in partial_errors, never silently."""
    import gpu_index.index.sources as sources_mod

    limit = sources_mod.VAST_FETCH_LIMIT
    full = [
        {
            "id": i, "machine_id": i, "host_id": 1, "num_gpus": 1,
            "dph_total": 10.0 + i * 0.01, "geolocation": "US",
            "verification": "verified",
        }
        for i in range(limit)
    ]

    def fake_fetch(url, timeout=None, **kwargs):
        if "B200" in urllib.parse.unquote(url):
            raise RuntimeError("B200 down")
        return json.dumps({"offers": full})

    monkeypatch.setattr(sources_mod, "fetch", fake_fetch)
    result = sources_mod.collect_vast()
    assert any(
        "exceeds fetch coverage" in e for e in result["partial_errors"]
    )


def test_vast_query_pins_limit_and_order():
    """The fetch window IS the correctness boundary: the limit and the
    order pair must not silently regress."""
    import gpu_index.index.sources as sources_mod

    asc = urllib.parse.unquote(sources_mod._vast_query("B300"))
    desc = urllib.parse.unquote(sources_mod._vast_query("B300", order="desc"))
    assert f'"limit": {sources_mod.VAST_FETCH_LIMIT}' in asc
    assert sources_mod.VAST_FETCH_LIMIT >= 50
    assert '["dph_total", "asc"]' in asc
    assert '["dph_total", "desc"]' in desc


def test_l0_tripwire_wiring_drops_offer_loudly(monkeypatch, capsys):
    """The consistency check is unfireable from public inputs by
    construction — pin the WIRING so deleting the guard fails a test, and
    pin that a firing is loud, never silent."""
    import gpu_index.index.sources as sources_mod

    monkeypatch.setattr(
        sources_mod, "_extraction_consistent", lambda *a: False
    )
    offers = parse_vast_offers(VAST_BODY)
    assert offers == []
    out = capsys.readouterr().out
    assert out.count("L0 ANOMALY") == 2
    assert "offer EXCLUDED" in out


def test_jump_screen_skips_quarantine_when_corroboration_starves():
    """With fewer comparable sources than min_corroborators + 1, a genuine
    market-wide move could never gather corroborators — fail open, loudly,
    instead of quarantining a real repricing."""
    payload = {
        "sources": [
            _scr_src("verda", [_scr_obs(native=5.25)]),   # -30%
            _scr_src("nebius", [_scr_obs(native=5.25)]),  # -30%
            _scr_src("vast", [], status="error"),
        ]
    }
    reference = _scr_ref(
        [
            _scr_src("verda", [_scr_obs(native=7.5)]),
            _scr_src("nebius", [_scr_obs(native=7.5)]),
        ]
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    assert report["quarantined"] == []
    assert "quarantine skipped" in report["quarantine_skipped"]
    assert "2 comparable" in report["quarantine_skipped"]
    # Nothing was flagged.
    assert all(
        "quarantined" not in o
        for s in payload["sources"]
        for o in s["observations"]
    )


def test_jump_screen_ignores_unpriceable_currency_prints():
    """An UNKNOWN-currency print (calc can never use it) must not become
    the would-be print or knock the source out of the corroborator pool."""
    payload = {
        "sources": [
            _scr_src(
                "verda",
                [
                    _scr_obs(native=5.0, currency="UNKNOWN"),
                    _scr_obs(native=7.5),
                ],
            ),
            _scr_src("nebius", [_scr_obs(native=7.85)]),
            _scr_src("vast", [_scr_obs(native=10.9382)]),
        ]
    }
    reference = _scr_ref(
        [
            _scr_src("verda", [_scr_obs(native=7.5)]),
            _scr_src("nebius", [_scr_obs(native=7.85)]),
            _scr_src("vast", [_scr_obs(native=6.8753)]),
        ]
    )
    report = apply_jump_screen(payload, reference, config=_scr_cfg())
    verda = next(d for d in report["deltas"] if d["source_id"] == "verda")
    assert verda["book_pct"] == pytest.approx(0.0, abs=0.01)  # USD 7.5 vs 7.5
    # verda counts as a comparable source, so vast's jump still quarantines.
    assert [q["source_id"] for q in report["quarantined"]] == ["vast"]


def test_jump_screen_degrades_on_malformed_reference_shapes():
    """Every reference shape surprise degrades to no-reference/ineligible —
    the screen itself must never raise on bucket-derived JSON."""
    payload = {"sources": [_scr_src("verda", [_scr_obs(native=7.5)])]}
    for reference in (
        [1, 2, 3],  # JSON array at the root
        {"capture_date": "2026-08-14", "slot_hour_utc": None, "sources": []},
        {"capture_date": "x", "slot_hour_utc": 16, "sources": ["oops", None]},
        {
            "capture_date": "x",
            "slot_hour_utc": 16,
            "sources": [
                {"source_id": "verda", "observations": ["oops", {"sku": "B300"}]}
            ],
        },
    ):
        report = apply_jump_screen(payload, reference, config=_scr_cfg())
        assert report["quarantined"] == []


def test_main_capture_survives_screen_crash(monkeypatch, capsys, tmp_path):
    """The screens block is fail-open at the WIRING level too: even a bug
    in apply_jump_screen itself must cost the report, never the slot."""
    runner = _load_runner()

    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    base["min_basket_sources_to_claim"] = 1
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(base))

    client = FakeS3()
    client.objects["index/b300_basket/latest.json"] = b'{"snapshot_key": "k"}'
    client.objects["k"] = b"[1, 2, 3]"

    def verda(timeout):
        return {
            "source_id": "verda",
            "observations": [{"sku": "B300", "price_usd_gpu_hr": 7.5, "raw_value": "7.5"}],
        }

    class StubConfig:
        bucket = "curves"

    def boom(*args, **kwargs):
        raise RuntimeError("intentional screen bug")

    monkeypatch.setattr(runner, "COLLECTORS", {"verda": verda})
    monkeypatch.setattr(
        runner.BucketConfig, "from_env", staticmethod(lambda: StubConfig())
    )
    monkeypatch.setattr(runner, "make_client", lambda cfg: client)
    monkeypatch.setattr(runner, "utc_now", lambda: _utc(2026, 8, 15, 16, 24))
    monkeypatch.setattr(
        runner, "write_local_snapshot", lambda p: tmp_path / "local.json"
    )
    monkeypatch.setattr(runner, "apply_jump_screen", boom)
    assert _run_main(runner, monkeypatch, ["--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "capture screens failed" in out
    assert "capture proceeds unscreened" in out
    assert any("snapshots/2026-08-15/" in k for k in client.put_order)


def test_config_rejects_incoherent_screen_thresholds(tmp_path):
    base = json.loads((REPO_ROOT / "config" / "index_basket.json").read_text())
    for override, match in (
        (
            {"capture_screens": {"jump_quarantine_pct": 25, "jump_corroborate_pct": 40}},
            "must not exceed",
        ),
        ({"capture_screens": [1, 2]}, "must be an object"),
        ({"capture_screens": {"jump_quarantine_pct": -5}}, "positive number"),
        ({"capture_screens": {"jump_min_corroborators": 0}}, "int >= 1"),
    ):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({**base, **override}))
        with pytest.raises(BasketConfigError, match=match):
            load_basket_config(p)


# ---------------------------------------------------------- B200 basket lane


def test_b200_config_loads_and_validates():
    cfg = load_basket_config(REPO_ROOT / "config" / "index_basket_b200.json")
    assert cfg["basket_id"] == "b200_annex_a2_v0_3"
    assert cfg["basket_role"] == "b200_basket"
    assert cfg["bucket_prefix"] == "index/b200_basket"
    assert cfg["target_sku"] == "B200"
    basket = [s for s in cfg["sources"] if s["role"] == "b200_basket"]
    assert len(basket) == 9  # nine standing constituents
    assert abs(sum(s["weight"] for s in basket) - 1.0) < 1e-9
    ids = {s["source_id"] for s in basket}
    assert {"lambda", "coreweave", "together"} <= ids  # the new collectors
    # USD-only lane: the EUR source must never appear here.
    assert "scaleway" not in ids and "latitude" not in ids
    assert cfg["canonical_slot_utc"] == 16  # the lane's canonical slot


def test_gb200_lookalike_quarantine():
    """Product-identity screen — the highest-consequence screen: real providers
    market Grace-coupled parts under the string 'B200' at 2-4x the
    underlying. Every documented lookalike signal quarantines; every
    legitimate labeling convention passes."""
    oracle = normalize_observation(
        {
            "sku": "B200",
            "price_usd_gpu_hr": 14.0,
            "sku_identifier": "BM.GPU.GB200.4 (4x Nvidia B200 189GB NVL72)",
            "gpu_count_basis": 4,
        }
    )
    assert oracle["quarantined"] == "gb200_lookalike"
    assert oracle["implausible"] is True
    assert "GB200 QUARANTINE" in oracle["notes"]
    assert oracle["price_usd_gpu_hr"] == 14.0  # raw stays recorded

    crusoe = normalize_observation(
        {
            "sku": "B200",
            "price_usd_gpu_hr": 10.5,
            "sku_identifier": "NVIDIA GB200 (186GB NVL72)",
        }
    )
    assert crusoe["quarantined"] == "gb200_lookalike"

    for memory in (186, 189):
        obs = normalize_observation(
            {"sku": "B200", "price_usd_gpu_hr": 9.0, "memory_gb_label": memory}
        )
        assert obs["quarantined"] == "gb200_lookalike", memory
    for count in (36, 72):
        obs = normalize_observation(
            {"sku": "B200", "price_usd_gpu_hr": 9.0, "gpu_count_basis": count}
        )
        assert obs["quarantined"] == "gb200_lookalike", count

    # Legitimate labels for the SAME physical part: 180,
    # 183 and 192 GB all pass; 8-GPU HGX identifiers pass.
    for memory in (180, 183, 192):
        obs = normalize_observation(
            {
                "sku": "B200",
                "price_usd_gpu_hr": 6.0,
                "sku_identifier": "p6-b200.48xlarge",
                "memory_gb_label": memory,
                "gpu_count_basis": 8,
            }
        )
        assert "quarantined" not in obs, memory
        assert obs["implausible"] is False

    # A 4-GPU marketplace slice of an HGX box is legitimate — count 4 alone
    # does NOT quarantine (deliberate rule).
    slice4 = normalize_observation(
        {"sku": "B200", "price_usd_gpu_hr": 6.25, "gpu_count_basis": 4}
    )
    assert "quarantined" not in slice4

    # Prose immunity: 'NVLink' in free-text notes must never trip the
    # token screen (it matches only the structured sku_identifier).
    prose = normalize_observation(
        {
            "sku": "B200",
            "price_usd_gpu_hr": 6.0,
            "notes": "NVLink domain of exactly 8, x86 host",
        }
    )
    assert "quarantined" not in prose

    # The screen is scoped to B200 prints — B300 rows pass through.
    b300 = normalize_observation(
        {"sku": "B300", "price_usd_gpu_hr": 7.5, "gpu_count_basis": 72}
    )
    assert "quarantined" not in b300


def test_parse_massedcompute_b200_row():
    """The widened key regex records the B200 node row from the
    same RSC surface; the B300 recipe is unchanged."""
    html = (
        'pre \\"B200 SXM6-x 8-0\\" specs 208 vCPU $$37.70 tail '
        '\\"B300 SXM6-x 8-1\\" specs 280 vCPU $$47.60 tail'
    )
    rows = parse_massedcompute(html)
    assert [(r["sku"], r["price_usd_gpu_hr"], r["gpu_count_basis"]) for r in rows] == [
        ("B200", pytest.approx(4.7125), 8),
        ("B300", pytest.approx(5.95), 8),
    ]


def test_snapshot_basket_ok_follows_config_role():
    """The coverage read (claim threshold input) counts THIS basket's
    constituents, whatever the config names its role."""
    cfg = {
        "basket_id": "b200_annex_a2_v0_3",
        "basket_role": "b200_basket",
        "bucket_prefix": "index/b200_basket",
        "capture_slots_utc": [4, 16],
        "canonical_slot_utc": 16,
        "sources": [
            {"source_id": "verda", "display_name": "Verda", "role": "b200_basket", "weight": 0.5, "source_type": "direct_principal"},
            {"source_id": "lambda", "display_name": "Lambda", "role": "b200_basket", "weight": 0.5, "source_type": "direct_principal"},
        ],
    }
    results = [
        {"source_id": "verda", "status": "ok",
         "observations": [{"sku": "B200", "price_usd_gpu_hr": 6.11, "raw_value": "6.11"}]},
        {"source_id": "lambda", "status": "error", "error": "x", "observations": []},
    ]
    payload = build_capture_snapshot(
        config=cfg,
        source_results=results,
        captured_at=_utc(2026, 8, 17, 16, 16),
        run_id="r",
        slot_date=datetime(2026, 8, 17).date(),
        slot_hour_utc=16,
        canonical=True,
        capturer={"job": "test", "version": "t/0"},
    )
    assert payload["basket_sources_ok"] == ["verda"]
    assert payload["basket_id"] == "b200_annex_a2_v0_3"


def test_jump_screen_follows_config_role():
    payload = {
        "sources": [
            _scr_src("verda", [_scr_obs(sku="B200", native=9.0)], role="b200_basket"),
            _scr_src("nebius", [_scr_obs(sku="B200", native=7.15)], role="b200_basket"),
            _scr_src("runpod", [_scr_obs(sku="B200", native=6.79)], role="b200_basket"),
        ]
    }
    reference = _scr_ref(
        [
            _scr_src("verda", [_scr_obs(sku="B200", native=6.11)]),
            _scr_src("nebius", [_scr_obs(sku="B200", native=7.15)]),
            _scr_src("runpod", [_scr_obs(sku="B200", native=6.79)]),
        ]
    )
    cfg = {**_scr_cfg(), "target_sku": "B200", "basket_role": "b200_basket"}
    report = apply_jump_screen(payload, reference, config=cfg)
    # verda +47% uncorroborated -> quarantined, proving the screen sees
    # b200_basket-role sources under the b200 config.
    assert [q["source_id"] for q in report["quarantined"]] == ["verda"]


# ------------------------------------------- B200 basket collectors

LAMBDA_HTML = (
    'pre <div>1-Click Clusters<br>pricing</div> '
    '<tr data-plan="NVIDIA HGX B200"><td data-label="DURATION">2 weeks</td>'
    '<td data-label="PRICE/GPU/HR*">$9.86</td></tr> mid '
    '<h2 class="h2">Instances pricing</h2> tabs '
    '<tr data-plan="NVIDIA B200 SXM6"><th scope="row">NVIDIA B200 SXM6</th>'
    '<td data-label="VRAM/GPU">180 GB</td><td data-label="vCPUs">208</td>'
    '<td data-label="RAM">2900 GiB</td><td data-label="STORAGE">22 TiB SSD</td>'
    '<td data-label="PRICE/GPU/HR*">$6.69</td></tr> '
    '<tr data-plan="NVIDIA B200 SXM6"><th scope="row">NVIDIA B200 SXM6</th>'
    '<td data-label="VRAM/GPU">180 GB</td><td data-label="vCPUs">104</td>'
    '<td data-label="RAM">1440 GiB</td><td data-label="STORAGE">11 TiB SSD</td>'
    '<td data-label="PRICE/GPU/HR*">$6.79</td></tr> tail'
)


def test_parse_lambda_pins_the_8x_ondemand_row():
    """The committed 1-Click Clusters table appears FIRST in document order
    at $9.86 and the 4x tab prices the same plan at $6.79 — the heading
    scope + spec pins must land on exactly the 8x on-demand row."""
    from gpu_index.index.sources import parse_lambda

    rows = parse_lambda(LAMBDA_HTML)
    assert len(rows) == 1
    obs = rows[0]
    assert obs["price_usd_gpu_hr"] == 6.69
    assert obs["gpu_count_basis"] == 8
    assert obs["sku_identifier"] == "NVIDIA B200 SXM6"
    assert obs["memory_gb_label"] == 180
    assert obs["tier"] == "on-demand"


def test_parse_lambda_fails_loud_on_reshape():
    from gpu_index.index.sources import parse_lambda

    with pytest.raises(RuntimeError, match="Instances pricing"):
        parse_lambda("<html>no heading</html>")
    # Committed plan string leaking into the on-demand scope must refuse.
    with pytest.raises(RuntimeError, match="committed-tier"):
        parse_lambda(
            '<h2 class="h2">Instances pricing</h2> NVIDIA HGX B200 leak'
        )
    # A duplicate pinned row (e.g. the escaped copy un-escaping) must trip
    # the exactly-one fence, never silently pick one.
    row = LAMBDA_HTML.split("</h2>")[1]
    with pytest.raises(RuntimeError, match="exactly one"):
        parse_lambda('<h2 class="h2">Instances pricing</h2>' + row + row)


COREWEAVE_HTML = (
    "pre On-demand GPU instances mid REGION: NORTH AMERICA "
    '<div role="listitem" class="table-row-v2 w-dyn-item kubernetes-gpu-pricing">'
    '<h3 data-product="nvidia-gb200-nvl72" class="table-model-name">NVIDIA GB200 NVL72</h3>'
    '<div class="table-meta-value">4^1</div>\n<div>GPU Count</div>'
    '<span class="instance-price">On-Demand Price: <span class="item-value">$42.00</span> / Hour<br/></span>'
    '<div role="listitem" class="table-row-v2 w-dyn-item kubernetes-gpu-pricing">'
    '<h3 data-product="nvidia-b200" class="table-model-name">NVIDIA HGX B200</h3>'
    '<div class="table-meta-value">8</div>\n<div>GPU Count</div>'
    '<div class="table-meta-value">180</div>\n<div>VRAM</div>'
    '<span class="instance-price">On-Demand Price: <span class="item-value">$68.80</span> / Hour<br/></span>'
    '<span class="spot-price">Spot Price: <span class="item-value">$34.11</span> / Hour</span>'
    "<div>$68.80</div>"
    " REGION: EUROPE eu-rows On-demand CPU instances tail"
)


def test_parse_coreweave_divides_the_labeled_instance_price():
    """The GB200 NVL72 row (Grace-coupled, '4^1' count, $42/4-GPU) sits in
    the SAME table and its data-product contains the substring 'b200' —
    the segment pins + int-count pin must keep the recipe off it, and the
    unlabeled duplicate price cell must never be the anchor."""
    from gpu_index.index.sources import parse_coreweave

    rows = parse_coreweave(COREWEAVE_HTML)
    assert len(rows) == 1
    obs = rows[0]
    assert obs["price_usd_gpu_hr"] == pytest.approx(8.60)
    assert obs["gpu_count_basis"] == 8
    assert obs["raw_unit"] == "usd_per_instance_hr"
    assert obs["memory_gb_label"] == 180


def test_parse_coreweave_fails_loud_on_wrong_row():
    from gpu_index.index.sources import parse_coreweave

    with pytest.raises(RuntimeError, match="section anchors"):
        parse_coreweave("<html>reshaped</html>")
    # GB200 content leaking into the B200 segment must refuse (1.2).
    leaked = COREWEAVE_HTML.replace(
        '<div class="table-meta-value">180</div>',
        'nvidia-gb200-nvl72 <div class="table-meta-value">180</div>',
    )
    with pytest.raises(RuntimeError, match="GB200 NVL72 content leaked"):
        parse_coreweave(leaked)
    # A count-pin mismatch (Grace 4-GPU shape swapped in) must refuse.
    wrong_count = COREWEAVE_HTML.replace(
        '<div class="table-meta-value">8</div>',
        '<div class="table-meta-value">4</div>',
    )
    with pytest.raises(RuntimeError, match="GPU Count pin"):
        parse_coreweave(wrong_count)


TOGETHER_ROW = (
    '<tr><td><p><span>NVIDIA</span><span> </span><span>HGX B200</span></p></td>'
    '<td class="is-right-border"><div class="opacity-70">'
    '<p data-batch="" class="body-m text-weight-medium">$%s</p></div></td>'
    '<td><p class="body-m">$7.99</p></td></tr>'
)

TOGETHER_HTML = (
    'pre id="dedicated-inference" section ' + (TOGETHER_ROW % "8.99")
    + ' mid id="gpu-clusters" class="section-anchor" '
    "On-demand hourly rates and reserved capacity >ON-Demand< "
    "All prices are per GPU per hour "
    '<p class="caption-m">Hardware</p></div></div></th>'
    '<th class="is-right-border"><div class="pricing_inline">'
    '<div class="opacity-70"><p class="caption-m">ON-Demand</p> '
    + (TOGETHER_ROW % "8.19")
    + '<tr><td><p><span>GB200 NVL72</span></p></td>'
    '<td class="is-right-border"><div class="opacity-70">'
    '<p data-batch="" class="body-m text-weight-medium">—</p></div></td></tr>'
    ' tail id="sandbox" class="section-anchor" post'
)


def test_parse_together_slices_past_the_dedicated_inference_twin():
    """The Dedicated Inference section carries a BYTE-IDENTICAL HGX B200
    row at $8.99 before the gpu-clusters anchor — the slice is what keeps
    the recipe on the $8.19 eligible print; GB200 NVL72 rows in the same
    table have em-dash cells the regex cannot match."""
    from gpu_index.index.sources import parse_together

    rows = parse_together(TOGETHER_HTML)
    assert len(rows) == 1
    obs = rows[0]
    assert obs["price_usd_gpu_hr"] == 8.19
    assert obs["raw_unit"] == "usd_per_gpu_hr"
    assert "256-GPU minimum" in obs["notes"]


def test_parse_together_fails_loud_on_reshape():
    from gpu_index.index.sources import parse_together

    with pytest.raises(RuntimeError, match="section anchors"):
        parse_together("<html>reshaped</html>")
    # Anchors present but reordered (sandbox before gpu-clusters).
    reordered = (
        'id="sandbox" class="section-anchor" x '
        'id="gpu-clusters" class="section-anchor" y'
    )
    with pytest.raises(RuntimeError, match="section anchors"):
        parse_together(reordered)
    # Identity pins missing inside the slice.
    hollow = (
        'id="gpu-clusters" class="section-anchor" nothing '
        'id="sandbox" class="section-anchor"'
    )
    with pytest.raises(RuntimeError, match="identity pin"):
        parse_together(hollow)
    # Two eligible-looking rows inside the slice must trip exactly-one.
    doubled = TOGETHER_HTML.replace(
        TOGETHER_ROW % "8.19", (TOGETHER_ROW % "8.19") * 2
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        parse_together(doubled)


def test_b200_capture_dry_run_end_to_end(monkeypatch, capsys, tmp_path):
    """The b200 config drives the SAME runner: nine constituents, coverage
    line reads 9/9, dry run writes nothing."""
    runner = _load_runner()

    def stub(sid, price):
        def collector(timeout):
            return {
                "source_id": sid,
                "observations": [
                    {"sku": "B200", "price_usd_gpu_hr": price, "raw_value": str(price)}
                ],
            }
        return collector

    prices = {
        "verda": 6.11, "nebius": 7.15, "hyperstack": 6.00, "lambda": 6.69,
        "coreweave": 8.60, "together": 8.19, "massedcompute": 4.71,
        "runpod": 6.79, "vast": 5.66,
    }
    monkeypatch.setattr(
        runner, "COLLECTORS", {sid: stub(sid, p) for sid, p in prices.items()}
    )
    monkeypatch.setattr(runner, "utc_now", lambda: _utc(2026, 8, 17, 16, 16))
    monkeypatch.setattr(
        runner,
        "upload_capture_snapshot",
        lambda *a, **k: pytest.fail("dry run uploaded"),
    )
    cfg_path = REPO_ROOT / "config" / "index_basket_b200.json"
    assert _run_main(
        runner, monkeypatch, ["--dry-run", "--config", str(cfg_path)]
    ) == 0
    out = capsys.readouterr().out
    assert "basket coverage: 9/9 constituents ok (minimum 5" in out


# ------------------------------------------------- collector hardening


def test_parse_lambda_never_bridges_into_the_next_row():
    """With plain [^$]*? gaps, an 8x price cell losing its '$' (em-dash
    outage) would let the pins bridge into the 4x row and record $6.79 —
    the tempered gap must fail LOUD instead."""
    from gpu_index.index.sources import parse_lambda

    dashed = LAMBDA_HTML.replace(
        '<td data-label="PRICE/GPU/HR*">$6.69</td>',
        '<td data-label="PRICE/GPU/HR*">—</td>',
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        parse_lambda(dashed)


def test_massed_nvl_row_reaches_the_gb200_screen():
    """The page already names H-series rows 'H200 NVL (141GB) NVLink' — a
    future B200-NVL row must arrive with a structured sku_identifier and
    leave normalization quarantined, never as an eligible print."""
    html = 'pre \\"B200 NVL72-x 4-0\\" specs $$42.00 tail'
    rows = parse_massedcompute(html)
    assert len(rows) == 1
    assert rows[0]["sku_identifier"] == "B200 NVL72"
    out = normalize_observation(rows[0])
    assert out["quarantined"] == "gb200_lookalike"
    assert out["implausible"] is True


def test_runpod_structured_fields_feed_the_screen():
    from gpu_index.index.sources import parse_runpod

    body = json.dumps(
        {
            "data": {
                "gpuTypes": [
                    {"displayName": "NVIDIA B200", "memoryInGb": 180, "securePrice": 6.79},
                    {"displayName": "B200 NVL", "memoryInGb": 186, "securePrice": 10.5},
                ]
            }
        }
    )
    rows = parse_runpod(body)
    by_id = {r["sku_identifier"]: normalize_observation(r) for r in rows}
    legit = by_id["NVIDIA B200"]
    assert "quarantined" not in legit and legit["memory_gb_label"] == 180
    trap = by_id["B200 NVL"]
    assert trap["quarantined"] == "gb200_lookalike"


def test_b200a_token_quarantines():
    """The product-identity screen names B200A a hard exclusion; norm_sku admits the
    substring, so the token screen must carry it."""
    obs = normalize_observation(
        {
            "sku": "B200",
            "price_usd_gpu_hr": 5.0,
            "sku_identifier": "NVIDIA B200A 141GB",
        }
    )
    assert obs["quarantined"] == "gb200_lookalike"


def test_all_nine_b200_recipes_carry_sku_identifier():
    """The recorded SKU identifier is what makes a product-identity substitution
    detectable after the fact — every constituent's rows must carry it."""
    from gpu_index.index.sources import parse_nebius, parse_verda

    assert all("sku_identifier" in r for r in parse_vast(VAST_BODY, "B200"))
    assert all(
        "sku_identifier" in r
        for r in parse_massedcompute(MASSED_HTML)
    )
    assert all("sku_identifier" in r for r in parse_nebius(NEBIUS_HTML))
    assert all("sku_identifier" in r for r in parse_hyperstack(HYPERSTACK_HTML))
    assert all("sku_identifier" in r for r in parse_verda(VERDA_HTML))
    # lambda/coreweave/together assert theirs in their own goldens; runpod
    # in test_runpod_structured_fields_feed_the_screen.


def test_b200_coverage_requires_a_target_sku_print():
    """A dual-sku collector that came back ok with only B300 rows must not
    count toward b200 basket coverage (the slot-claim input)."""
    cfg = {
        "basket_id": "b200_annex_a2_v0_3",
        "basket_role": "b200_basket",
        "target_sku": "B200",
        "bucket_prefix": "index/b200_basket",
        "capture_slots_utc": [4, 16],
        "canonical_slot_utc": 16,
        "sources": [
            {"source_id": "verda", "display_name": "V", "role": "b200_basket", "weight": 0.5, "source_type": "direct_principal"},
            {"source_id": "massedcompute", "display_name": "M", "role": "b200_basket", "weight": 0.5, "source_type": "direct_partnered"},
        ],
    }
    results = [
        {"source_id": "verda", "status": "ok",
         "observations": [{"sku": "B200", "price_usd_gpu_hr": 6.11, "raw_value": "6.11"}]},
        {"source_id": "massedcompute", "status": "ok",
         "observations": [{"sku": "B300", "price_usd_gpu_hr": 5.95, "raw_value": "47.60"}]},
    ]
    payload = build_capture_snapshot(
        config=cfg,
        source_results=results,
        captured_at=_utc(2026, 8, 17, 16, 16),
        run_id="r",
        slot_date=datetime(2026, 8, 17).date(),
        slot_hour_utc=16,
        canonical=True,
        capturer={"job": "test", "version": "t/0"},
    )
    assert payload["basket_sources_ok"] == ["verda"]
    # Both sources still record fully — coverage, not censorship.
    assert payload["sources_ok"] == ["massedcompute", "verda"]
