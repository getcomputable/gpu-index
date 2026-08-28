# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Bucket dashboard: renderer + CLI wiring pins.

What must hold: the report is a pure render of stored artifacts published
to the ONE mutable key under index/ (text/html, no-store), it is warn-only
(a render bug never fails composite publishing), it never emits a
remote-derived string unescaped, and --dry-run never writes it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gpu_index.index.report import render_report

# Sibling-module reuse on purpose: the golden fixtures and the CLI harness
# live in test_index_composite.py and must not fork.
from test_index_composite import (
    FX_2026_08_10,
    FakeS3,
    _composite_payload,
    _entry,
    _live_first_capture_snapshot,
    _obs,
    _seed_first_capture,
    _wire_cli,
)

REPORT_KEY = "index/b300_basket/report/index.html"
NOW = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)


def _ws():
    """Fresh calc_v5 weight state — REQUIRED by compute_day whenever the
    config sets calc.dynamic_weights (the CLI threads its own)."""
    return {"prices": {}, "vectors": {}, "mode": "fallback"}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Autouse fixtures do not cross modules: this module needs its own.
    GITHUB_ACTIONS flips warn() to '::warning::' (no uppercase WARNING) —
    without the delenv the warn-only assertion fails ON CI ITSELF."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)


class RecordingS3(FakeS3):
    """FakeS3 that keeps the PUT kwargs so headers are assertable."""

    def __init__(self):
        super().__init__()
        self.put_kwargs = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        super().put_object(Bucket, Key, Body, **kwargs)
        self.put_kwargs[Key] = kwargs


def _seed_pointer(client):
    snapshot_key = (
        "index/b300_basket/snapshots/2026-08-10/slot16-20260810T211029Z-b128.json"
    )
    client.objects["index/b300_basket/latest.json"] = json.dumps(
        {
            "pointer_version": 1,
            "snapshot_key": snapshot_key,
            "run_id": "20260810T211029Z-b128",
            "capture_date": "2026-08-10",
            "slot_hour_utc": 16,
            "canonical_slot": True,
            "late_fill": True,
            "basket_sources_ok": [
                "verda", "nebius", "hyperstack", "scaleway",
                "runpod", "vast", "massedcompute", "latitude",
            ],
            "captured_at": "2026-08-10T21:10:29Z",
            "published_at": "2026-08-10T21:10:30Z",
        }
    ).encode()


# ----------------------------------------------------------- CLI wiring


def test_sync_publishes_report_with_html_headers(monkeypatch, capsys):
    """End-to-end: --sync renders the dashboard from what it just published
    and PUTs it with ContentType text/html + Cache-Control no-store."""
    client = RecordingS3()
    _seed_first_capture(client)
    _seed_pointer(client)
    cli = _wire_cli(monkeypatch, client, NOW, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert f"report published: {REPORT_KEY}" in out

    assert REPORT_KEY in client.objects
    kwargs = client.put_kwargs[REPORT_KEY]
    assert kwargs["ContentType"] == "text/html; charset=utf-8"
    assert kwargs["CacheControl"] == "no-store"

    html = client.objects[REPORT_KEY].decode("utf-8")
    # Golden content from the real first-capture composite under calc_v4
    # (calc_v4) — unchanged under calc_v5, whose dynamic weights start in
    # fallback mode: the vote median 7.475 with its aggregate confidence
    # and statistic label; the retired weighted mean 7.5340 still renders
    # as the artifact's diagnostic field in the day breakdown.
    assert "7.4750" in html
    assert "median of stddev votes, weighted" in html
    assert "&#177;0.4500 CI" in html
    assert "8/8 basket ok" in html
    for source_id in (
        "verda", "nebius", "hyperstack", "scaleway",
        "runpod", "vast", "massedcompute", "latitude",
    ):
        assert source_id in html
    # Scaleway's audit row: native EUR print + the FX block it converted at.
    assert "8.66625" in html
    assert "1.1555" in html
    assert "EUR (unconverted)" in html
    # Day selector + per-day breakdown panel exist for the published day.
    assert 'id="day-select"' in html
    assert 'data-day-panel="2026-08-10"' in html
    assert "annex_a_v0_2_calc_v6" in html
    # Snapshot run id in the footer.
    assert "20260810T211029Z-b128" in html

    # Overwrite-in-place: a second sync — one that writes ZERO composites —
    # must still rewrite the SAME report key with fresh bytes. (47 of 48
    # daily firings write no composite; gating the report on `wrote` would
    # freeze the dashboard for the rest of the day.)
    later = datetime(2026, 8, 11, 6, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, later, ["--sync"])
    assert cli.main() == 0
    out2 = capsys.readouterr().out
    assert "composites written: 0" in out2
    assert f"report published: {REPORT_KEY}" in out2
    assert client.put_order.count(REPORT_KEY) == 2
    assert "2026-08-11 06:00:00Z" in client.objects[REPORT_KEY].decode()


def test_render_failure_is_warn_only(monkeypatch, capsys):
    """A broken renderer must not touch the exit code or the composites."""
    import gpu_index.index.report as report_mod

    def _boom(**kwargs):
        raise RuntimeError("intentional render bug")

    monkeypatch.setattr(report_mod, "render_report", _boom)
    client = FakeS3()
    _seed_first_capture(client)
    cli = _wire_cli(monkeypatch, client, NOW, ["--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "composites written: 1" in out
    assert "report render failed" in out
    assert "WARNING" in out
    assert REPORT_KEY not in client.objects
    assert (
        "index/b300_basket/composites/annex_a_v0_2_calc_v6/2026-08-10.json"
        in client.objects
    )


def test_dry_run_never_writes_the_report(monkeypatch, capsys):
    client = FakeS3()
    _seed_first_capture(client)
    cli = _wire_cli(monkeypatch, client, NOW, ["--sync", "--dry-run"])
    assert cli.main() == 0
    assert REPORT_KEY not in client.objects
    assert "report published" not in capsys.readouterr().out


def test_targeted_modes_skip_the_report(monkeypatch, capsys):
    """Only --sync (the job's mode) refreshes the dashboard."""
    client = FakeS3()
    _seed_first_capture(client)
    cli = _wire_cli(monkeypatch, client, NOW, ["--date", "2026-08-10"])
    assert cli.main() == 0
    assert REPORT_KEY not in client.objects


# ------------------------------------------------------------- renderer


def _composites_window():
    """Three published days: normal, held-out verda, dark day-missed."""
    from gpu_index.index.composite import compute_day
    from test_index_composite import CONFIG

    window_history: dict = {}
    ws = _ws()  # calc_v5 weight state threads across days like the history
    day1 = compute_day(
        config=CONFIG,
        day="2026-08-10",
        snapshot=_live_first_capture_snapshot(),
        substituted_from=None,
        window_history=window_history,
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=ws,
        prior_slot_prints={},
    )
    dark = compute_day(
        config=CONFIG,
        day="2026-08-11",
        snapshot=None,
        substituted_from=None,
        window_history=window_history,
        window_currencies={},
        fx_records=FX_2026_08_10,
        weight_state=ws,
        prior_slot_prints={},
    )
    # Seasoned windows so day 3 carries REAL filtered verdicts: a held-out
    # verda (USD) and an accepted scaleway judged in its native EUR terms
    # (calc_v3 — the filter cell must show the fence and the currency).
    history = {"verda": [7.5] * 12, "scaleway": [7.5] * 12}
    currencies = {"verda": "USD", "scaleway": "EUR"}
    snap = _live_first_capture_snapshot()
    for entry in snap["sources"]:
        if entry["source_id"] == "verda":
            entry["observations"] = [_obs("B300", usd=9.0)]
    day3 = compute_day(
        config=CONFIG,
        day="2026-08-12",
        snapshot=snap,
        substituted_from=None,
        window_history=history,
        fx_records=FX_2026_08_10,
        window_currencies=currencies,
        weight_state=_ws(),
        prior_slot_prints={},  # fresh state alongside the seeded histories
    )
    return {"2026-08-10": day1, "2026-08-11": dark, "2026-08-12": day3}


def test_render_report_time_series_and_breakdowns():
    composites = _composites_window()
    html = render_report(
        pointer=None,
        latest_snapshot=_live_first_capture_snapshot(),
        composites_by_date=composites,
        now=NOW,
    )
    # One breakdown panel per published day, dark day included.
    for day in ("2026-08-10", "2026-08-11", "2026-08-12"):
        assert f'data-day-panel="{day}"' in html
        assert f'<option value="{day}"' in html
    # The dark day reads as dark, not as a number.
    assert "2026-08-11 — dark" in html.replace("&#x27;", "'")
    # Held-out verda on 08-12: verdict chip + amber held marker data.
    assert "held out" in html
    # Index chart + 8 per-source panels carry hover data.
    assert html.count('data-chart="line"') == 9
    # Filter audit fields surface (mu/sigma from the held-out day).
    assert "sigma" in html
    # calc_v3 legibility: the filter cell shows the explicit
    # fence the verdict was judged against, and the operating currency for
    # non-USD sources — scaleway filters in native EUR terms (sigma=0
    # window, floored band 3.0 * 0.05 = 0.15 around mu 7.5 since the
    # calc_v4 loosening).
    assert "fence [7.35, 7.65]" in html
    assert "in EUR" in html
    # verda (USD terms) shows its fence but no currency label.
    verda_cell = html.split("mu 7.5 &#183; sigma 0 &#183; dev 1.5")[1].split("</td>")[0]
    assert "fence [" in verda_cell
    assert "in USD" not in verda_cell
    # Embedded JSON never emits a raw '<'.
    for chunk in html.split('data-days="')[1:]:
        assert "<" not in chunk.split('"')[0]


def test_render_report_escapes_remote_page_text():
    """EVERY remote-derived string — snapshot notes (arbitrary scraped page
    text), run ids, keys, dates, statuses, tiers, currencies, source ids —
    must be escaped. Removing html.escape from ANY sink must fail here."""
    hostile = '</script><script>alert(1)</script><img src=x onerror=y>'

    def h(tag):
        return f'{tag}"><script>alert(1)</script>'

    obs = _obs("B300", usd=7.5)
    obs.update(
        {
            "sku": h("sku"),
            "tier": h("tier"),
            "region": h("region"),
            "notes": hostile,
            "currency": h("currency"),
        }
    )
    eur_obs = _obs("B300", native=7.5, usd=None, currency="EUR", basis=8)
    eur_obs["notes"] = hostile
    snapshot = {
        "run_id": h("snaprun"),
        "captured_at": h("captured"),
        "late_fill": False,
        "sources": [
            _entry(h("srcid"), [obs, eur_obs]),
            _entry(h("deadsrc"), [], status=h("status")),
        ],
    }
    pointer = {
        "snapshot_key": h("key"),
        "capture_date": h("capdate"),
        "slot_hour_utc": 16,
        "published_at": h("pubat"),
        "late_fill": True,
        "basket_sources_ok": [h("okid")],
    }
    composites = {
        "2026-08-10": {
            **_composite_payload(),
            "methodology_id": h("methodology"),
            "snapshot_run_id": h("comprun"),
            "calc_params": {"promote_tie_break": h("tiebreak")},
            "sources": [
                {
                    "source_id": h("compsrc"),
                    "weight": 0.15,
                    "status": "ok",
                    "chosen": {
                        "usd_per_gpu_hr": 7.5,
                        "tier": h("ctier"),
                        "gpu_count_basis": 1,
                        "currency": h("ccur"),
                        "native_per_gpu_hr": 7.5,
                        "fx_rate": 1.1,
                        "fx_as_of": h("fxasof"),
                        "notes": hostile,
                    },
                    "filter": {"accepted": True, "unfiltered": False,
                               "mu": 7.5, "sigma": 0.1, "deviation": 0.0,
                               "n_history": 20, "band": 0.25, "lo": 7.25,
                               "hi": 7.75, "currency": h("fcur")},
                },
                {"source_id": h("heldsrc"), "weight": 0.1,
                 "status": h("srcstatus")},
            ],
            "fallback_pool": {
                "sources": [{"source_id": h("poolsrc"),
                             "status": h("poolstatus")}],
                "mean_usd_gpu_hr": None,
            },
        }
    }
    html = render_report(
        pointer=pointer,
        latest_snapshot=snapshot,
        composites_by_date=composites,
        now=NOW,
    )
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
    assert '"><script>' not in html  # no attribute breakout anywhere
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # Exactly the one legitimate script block (the dashboard's own JS).
    assert html.count("<script>") == 1


def test_render_report_survives_poisoned_numbers_and_naive_heartbeat():
    """json.loads accepts bare NaN/Infinity and a hand-healed pointer can
    carry a NAIVE published_at: both must degrade (gap / 'unparseable'
    chip), never crash the render — a warn-only crash silently freezes the
    dashboard for every subsequent firing."""
    composites = _composites_window()
    composites["2026-08-12"]["index"]["value_usd_gpu_hr"] = float("nan")
    composites["2026-08-10"]["index"]["value_usd_gpu_hr"] = float("inf")
    pointer = {
        "snapshot_key": "k",
        "capture_date": "2026-08-10",
        "slot_hour_utc": 16,
        "late_fill": False,
        "basket_sources_ok": [],
        "published_at": "2026-08-10T21:10:30",  # naive — no Z, no offset
    }
    html = render_report(
        pointer=pointer,
        latest_snapshot=None,
        composites_by_date=composites,
        now=NOW,
    )
    assert "capture heartbeat unparseable" in html
    assert "NaN" not in html and "Infinity" not in html
    assert "nan" not in html.lower().replace("tabular-nums", "")


def _scaleway_day(filter_verdict):
    return {
        "2026-08-10": {
            **_composite_payload(),
            "sources": [
                {
                    "source_id": "scaleway",
                    "weight": 0.15,
                    "status": "ok",
                    "chosen": {"usd_per_gpu_hr": 8.7, "currency": "USD"},
                    "filter": filter_verdict,
                }
            ],
        }
    }


def test_currency_mismatch_renders_held_out_not_warmup():
    """Ruling D1: a currency_mismatch verdict is HELD OUT (fail-closed,
    window preserved) — chip and filter cell must say exactly that, never
    impersonate a warm-up print, an ordinary sigma hold-out, or the old
    unfiltered-accept 'window reset' semantics."""
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=_scaleway_day(
            {
                "accepted": False,
                "unfiltered": False,
                "currency_mismatch": True,
                "currency": "USD",
                "window_currency": "EUR",
                "filter_price": 8.7,
                "pending_count": 1,
                "confirm_after": 3,
                "n_history": 12,
            }
        ),
        now=NOW,
    )
    assert "held out (currency change 1/3)" in html  # chip
    assert (
        "currency changed EUR &#8594; USD &#183; held out 1/3 &#183; "
        "window preserved" in html
    )
    assert "window reset" not in html  # the retired semantics
    assert "warm-up, n=12" not in html


def test_untrusted_currency_renders_held_out_fail_closed():
    """Ruling D1: an untrusted_currency verdict names the raw label
    (escaped — it is remote-derived), reads as held out fail-closed, and
    says the window survived."""
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=_scaleway_day(
            {
                "accepted": False,
                "unfiltered": False,
                "untrusted_currency": True,
                "currency_label": "UNKNOWN<script>",
                "n_history": 12,
            }
        ),
        now=NOW,
    )
    assert "held out (untrusted currency)" in html  # chip
    assert (
        "untrusted currency label UNKNOWN&lt;script&gt; &#183; held out "
        "fail-closed &#183; window preserved" in html
    )
    assert "UNKNOWN<script>" not in html


def test_currency_confirmed_renders_reseed_and_keeps_manual_verify():
    """Ruling D1: the confirmation day (3rd consecutive same-new-currency
    print) is unfiltered=true, so compute_day's R3 loop can flag it
    manual_verify — the mark must survive in BOTH the verdict chip and the
    filter cell, never vanish behind the confirmation copy."""
    confirmed = {
        "accepted": True,
        "unfiltered": True,
        "currency_confirmed": True,
        "currency": "USD",
        "window_currency": "EUR",
        "filter_price": 8.7,
        "n_history": 3,
    }
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=_scaleway_day(dict(confirmed)),
        now=NOW,
    )
    assert "currency confirmed (window reseeded)" in html  # chip
    assert (
        "currency confirmed EUR &#8594; USD &#183; window reseeded, n=3"
        in html
    )
    # No manual_verify: neither surface carries the mark.
    assert "(window reseeded) manual-verify" not in html
    assert "n=3 &#183; manual-verify" not in html

    flagged = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=_scaleway_day({**confirmed, "manual_verify": True}),
        now=NOW,
    )
    assert "currency confirmed (window reseeded) manual-verify" in flagged
    assert (
        "currency confirmed EUR &#8594; USD &#183; window reseeded, n=3 "
        "&#183; manual-verify" in flagged
    )


def test_params_footer_shows_com1310_knobs_only_when_present():
    """The day-panel footer grows 'terms'/'sigma-floor' segments for
    calc_v3-style calc_params and renders byte-identical to before for a
    legacy artifact (whose frozen calc_params never carry the knobs)."""
    legacy_params = {
        "filter_window": 20,
        "filter_sigma": 2.5,
        "filter_warmup": 10,
        "manual_verify_pct": 15.0,
        "fx_max_staleness_days": 7,
        "promote_tie_break": "later",
    }
    legacy_line = (
        "methodology annex_a_v0_2_calc_v1 &#183; window 20 &#183; "
        "sigma 2.5 &#183; warmup 10 &#183; manual-verify 15.0% &#183; "
        "fx staleness 7d &#183; tie-break later"
    )
    legacy = {
        "2026-08-10": {
            **_composite_payload(),
            "calc_params": dict(legacy_params),
        }
    }
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=legacy,
        now=NOW,
    )
    assert f'<div class="sub params">{legacy_line}</div>' in html
    assert "sigma-floor" not in html

    v3 = {
        "2026-08-10": {
            **_composite_payload(),
            "calc_params": {
                **legacy_params,
                "filter_terms": "recorded_currency",
                "filter_sigma_floor": 0.05,
            },
        }
    }
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=v3,
        now=NOW,
    )
    assert (
        f'<div class="sub params">{legacy_line} &#183; '
        "terms recorded_currency &#183; sigma-floor 0.05</div>" in html
    )

    # Percent-form floor: the segment carries the % suffix — a pct-mode
    # day must never read as an absolute floor.
    pct = {
        "2026-08-10": {
            **_composite_payload(),
            "calc_params": {
                **legacy_params,
                "filter_terms": "recorded_currency",
                "filter_sigma_floor_pct": 3.0,
            },
        }
    }
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=pct,
        now=NOW,
    )
    assert (
        f'<div class="sub params">{legacy_line} &#183; '
        "terms recorded_currency &#183; sigma-floor 3.0%</div>" in html
    )


def test_v4_vote_fields_render_and_legacy_days_stay_untouched():
    """calc_v4 era-correctness both ways: a v4 artifact renders
    its aggregate confidence, statistic label, vote band, and per-source
    vote CIs (floored and non-floored forms); a legacy artifact renders
    none of that v4 chrome. The one deliberate exception is the index
    table's ±CI COLUMN: the column set is a stable view-level choice
    (identical across eras, per the stable-format rule), so legacy days
    show a dash cell under the same header rather than a shifting table
    shape."""
    v4_day = {
        **_composite_payload(),
        "methodology_id": "annex_a_v0_2_calc_v4",
        "index": {
            "value_usd_gpu_hr": 7.475,
            "statistic": "median_ci_votes",
            "confidence_usd_gpu_hr": 0.45,
            "vote_p25_usd_gpu_hr": 7.35,
            "vote_p75_usd_gpu_hr": 7.925,
            "weighted_mean_usd_gpu_hr": 7.533988,
            "unweighted_mean_usd_gpu_hr": 7.529531,
            "renormalized_weights": {"verda": 0.15, "scaleway": 0.15},
            "sources_used_count": 8,
        },
        "sources": [
            {
                "source_id": "verda",
                "weight": 0.15,
                "status": "ok",
                "chosen": {"usd_per_gpu_hr": 7.5, "tier": "on_demand"},
                "filter": {"accepted": True, "unfiltered": True, "n_history": 0},
                "vote": {
                    "sigma": 0.0,
                    "sigma_floored": True,
                    "conf_usd_gpu_hr": 0.05,
                },
            },
            {
                # The mature-window steady state: a real sigma, no floor.
                "source_id": "scaleway",
                "weight": 0.15,
                "status": "ok",
                "chosen": {"usd_per_gpu_hr": 8.66625, "tier": "on_demand"},
                "filter": {"accepted": True, "unfiltered": False,
                           "mu": 7.5, "sigma": 0.1, "deviation": 0.0,
                           "n_history": 12},
                "vote": {
                    "sigma": 0.1,
                    "sigma_floored": False,
                    "conf_usd_gpu_hr": 0.11555,
                },
            },
        ],
    }
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date={"2026-08-10": v4_day},
        now=NOW,
    )
    assert "&#177;0.4500 CI" in html  # hero tile
    assert "median of stddev votes, weighted" in html  # sources tile subtitle
    # Day summary: value ± confidence with the vote band.
    assert "&#177; 0.4500 (votes [7.3500, 7.9250])" in html
    # Per-source vote suffix on the filter cell: floored form names the
    # floor, non-floored form shows the real sigma without the flag.
    assert "vote &#177;0.05 (sigma 0, floored)" in html
    assert "vote &#177;0.11555 (sigma 0.1)" in html
    assert "vote &#177;0.11555 (sigma 0.1, floored)" not in html

    legacy_html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date={"2026-08-10": _composite_payload()},
        now=NOW,
    )
    for needle in (
        " CI</div>",  # hero-tile confidence segment
        "median of stddev votes",
        "vote &#177;",
        "votes [",
        "sigma_floored",
    ):
        assert needle not in legacy_html
    assert "weighted mean, renormalized" in legacy_html
    # The stable ±CI column renders a dash for legacy days.
    assert "&#177;CI" in legacy_html

    # a tuned day self-describes on the dashboard — the sources
    # tile names the band and the point-median diagnostic, so an operator
    # never opens the raw artifact to see which statistic priced the day.
    tuned_day = json.loads(json.dumps(v4_day))
    tuned_day["index"]["iqm_alpha"] = 0.1
    tuned_day["index"]["vote_median_usd_gpu_hr"] = 7.525
    tuned_html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date={"2026-08-10": tuned_day},
        now=NOW,
    )
    assert "band mean &#177;0.1 of stddev votes, median $7.5250" in tuned_html
    assert "median of stddev votes, weighted" not in tuned_html


def test_line_chart_held_last_point_keeps_amber_marker():
    """A held-out latest print must not be repainted as accepted by the
    end-of-line dot (SVG paints in document order)."""
    from gpu_index.index.report import _line_chart

    svg = _line_chart(
        ["2026-08-10", "2026-08-11"], [7.5, 9.0], held_days=["2026-08-11"]
    )
    assert svg.count('class="pt-held"') == 1
    held_cx = svg.split('class="pt-held" cx="')[1].split('"')[0]
    assert f'class="pt" cx="{held_cx}"' not in svg


def test_render_report_survives_an_empty_world():
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date={},
        now=NOW,
    )
    assert "no published composites yet" in html
    assert "no composite published yet" in html


def test_render_report_gaps_stay_gaps():
    """An unpublished day between two published days must appear on the
    axis (time reads true) and in the table as not-published — never as a
    zero or an interpolated value."""
    composites = _composites_window()
    del composites["2026-08-11"]
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=composites,
        now=NOW,
    )
    assert "2026-08-11" in html  # still on the axis / in the table
    assert "not published" in html
    assert 'data-day-panel="2026-08-11"' not in html


def test_report_window_is_bounded():
    """publish_report must never GET a composite older than its window —
    the job fires 48x/day, so genesis-anchored reads would grow by one GET
    per calendar day forever. Genesis here sits 71 days back, far outside
    the window, so this fails if the clamp regresses to `start = genesis`."""
    from datetime import date, timedelta

    from gpu_index.index.report import REPORT_WINDOW_DAYS
    from test_index_composite import _load_cli

    cli = _load_cli()
    client = RecordingS3()
    _seed_first_capture(client)
    _seed_pointer(client)

    reads = []
    orig = client.get_object

    def counting_get(Bucket, Key):
        reads.append(Key)
        return orig(Bucket=Bucket, Key=Key)

    client.get_object = counting_get
    key = cli.publish_report(
        client,
        "curves",
        prefix="index/b300_basket",
        methodology_id="annex_a_v0_2_calc_v6",
        genesis=date(2026, 6, 1),
        now=NOW,
    )
    assert key == REPORT_KEY
    composite_reads = [k for k in reads if "/composites/" in k]
    assert len(composite_reads) == REPORT_WINDOW_DAYS
    window_start = (
        NOW.date() - timedelta(days=REPORT_WINDOW_DAYS - 1)
    ).isoformat()
    for read_key in composite_reads:
        read_day = read_key.rsplit("/", 1)[1].removesuffix(".json")
        assert read_day >= window_start, read_key


def test_ok_source_with_no_eligible_print_is_not_labeled_held_out():
    """A source that collected fine but had nothing eligible (every print
    interruptible/implausible/quarantined) is not a filter hold-out — the
    chip must say so instead of impersonating one."""
    composites = {
        "2026-08-10": {
            **_composite_payload(),
            "sources": [{"source_id": "vast", "weight": 0.1, "status": "ok"}],
        }
    }
    html = render_report(
        pointer=None,
        latest_snapshot=None,
        composites_by_date=composites,
        now=NOW,
    )
    assert "no eligible print" in html
    assert "held out</span>" not in html  # chip text; the JS tooltip string differs
