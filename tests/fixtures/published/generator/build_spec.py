#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Regenerate the published-record fixtures (tests/fixtures/published/).

Two stages, so every digest assertion in the suite is a genuine
cross-implementation check:

  1. this script builds the observation SPEC with engine-true values
     (median_stddev_composite prices each fixture observation from its
     own receipts, exactly like production);
  2. ``envelope.mjs`` (a byte-exact Node mirror of computable-mcp
     src/publisher/artifacts.ts) wraps, digests, and pretty-prints each
     file — JS writes and digests, the Python package re-derives.

    python3 build_spec.py spec.json
    node envelope.mjs spec.json ..   # writes into tests/fixtures/published
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from gpu_index.index.panel import median_stddev_composite  # noqa: E402

INDEX_NAME = "Computable GPU Index"
INDEX_DOMAIN = "index.example.com"
LICENSING_URL = "https://index.example.com/licensing"
SNAP = "d1a5e2c3" * 8  # 64 hex chars


def liveness_block():
    return {
        "scheme": "predictive_v1",
        "lookback_horizons_hours": [24, 72],
        "forward_horizons_hours": [24],
        "history_days": 14,
        "half_life_days": 7.0,
        "ridge_lambda": 0.000001,
        "gamma": 0.5,
        "weight_min": 0.02,
        "weight_max": 0.35,
        "min_train_samples": 6,
        "target_variance_floor": 0.000001,
        "switch_min_eligible": 4,
        "attendance_floor": 0.6,
        "max_abs_log_return": 0.75,
    }


def calc_params(min_publish, members):
    return {
        "collection_interval": "hourly",
        "index_recomputation": "hourly",
        "minimum_panel_members_to_record": min_publish,
        "min_sources_to_publish": min_publish,
        "eligible_tiers": ["verified"],
        "filter_window_observations": 14,
        "filter_sigma": 3.0,
        "filter_warmup_observations": 3,
        "filter_sigma_floor_usd_gpu_hr": 0.02,
        "filter_terms": "recorded_currency",
        "manual_verify_pct": 25.0,
        "aggregation": "median_stddev_votes",
        "fx_source": None,
        "fx_max_staleness_days": None,
        "manual_exclusions": [],
        "members": [
            {"source_id": sid, "opening_weight": w} for sid, w in members
        ],
        "liveness": liveness_block(),
    }


def receipt(sid, price, sd, weight, status="ok", verdict="accepted",
            last_seen=None, liveness=0.8, disclosure="published"):
    url = None if price is None and status != "ok" else (
        f"https://{sid}.example.com/pricing"
    )
    return {
        "source_id": sid,
        "price": price,
        "sd": sd,
        "weight": weight,
        "liveness_score": liveness,
        "source_url": url,
        "status": status,
        "filter_verdict": verdict,
        "last_seen": last_seen,
        "price_disclosure": disclosure,
    }


def observation(sku, methodology, stamp, receipts, min_publish, members,
                restatements=None, force_no_print_reason=None):
    observed_at = f"{stamp}:00:00.000Z"
    generated_at = f"{stamp}:59:30.000Z"
    passing = [
        (r["source_id"], float(r["weight"]), r["price"])
        for r in receipts
        if r["status"] == "ok" and r["filter_verdict"] == "accepted"
        and r["price"] is not None
    ]
    stddevs = {
        r["source_id"]: r["sd"]
        for r in receipts
        if r["status"] == "ok" and r["filter_verdict"] == "accepted"
        and r["sd"] is not None
    }
    if force_no_print_reason:
        status, reason = "no_print", force_no_print_reason
        value = band = None
    else:
        comp = median_stddev_composite(passing, stddevs)
        assert comp is not None and len(passing) >= min_publish
        status, reason = "ok", None
        value = comp["value_usd_gpu_hr"]
        band = comp["confidence_usd_gpu_hr"]
    return {
        "schema_version": 1,
        "kind": "gpu_price_index_observation",
        "sku": sku,
        "methodology_id": methodology,
        "observed_at": observed_at,
        "generated_at": generated_at,
        "status": status,
        "reason": reason,
        "value_usd_gpu_hr": value,
        "stability_band_usd_gpu_hr": band,
        "unit": "USD/GPU/hour",
        "input_snapshot_sha256": SNAP,
        "calc_params": calc_params(min_publish, members),
        "receipts": receipts,
        "restatements": restatements or [],
    }


H100_MEMBERS = [
    ("alpha", 0.15), ("bravo", 0.15), ("charlie", 0.2),
    ("delta", 0.2), ("echo", 0.15), ("foxtrot", 0.15),
]
B200_MEMBERS = [
    ("alpha", 0.25), ("bravo", 0.25), ("charlie", 0.25), ("delta", 0.25),
]


def h100_receipts(base):
    return [
        receipt("alpha", base + 0.013437, 0.03125, 0.0625),
        receipt("bravo", base + 0.087001, 0.041333, 0.15),
        receipt("charlie", base - 0.052499, 0.02, 0.35),
        receipt("delta", base + 0.000001, 0.055725, 0.2),
        receipt("echo", base - 0.10125, 0.03, 0.1375),
        receipt("foxtrot", base + 0.41, 0.02, None, status="ok",
                verdict="rejected"),
    ]


_STAMP = {}


def make_h100(stamp, base):
    _STAMP["current"] = f"{stamp}:00:00.000Z"
    rs = h100_receipts(base)
    for r in rs:
        r["last_seen"] = _STAMP["current"]
    return observation("H100", "h100_sxm_v1_calc_v1", stamp, rs, 5,
                       H100_MEMBERS)


def make_b200_ok(stamp):
    _STAMP["current"] = f"{stamp}:00:00.000Z"
    rs = [
        receipt("alpha", 3.153125, 0.026667, 0.25, last_seen=_STAMP["current"]),
        receipt("bravo", 3.21, 0.04, 0.25, last_seen=_STAMP["current"]),
        receipt("charlie", 3.099999, 0.02, 0.3, last_seen=_STAMP["current"]),
        receipt("delta", 3.18, 0.031, 0.2, last_seen=_STAMP["current"]),
    ]
    return observation("B200", "b200_panel_annex_a2_v0_3_calc_v6", stamp,
                       rs, 4, B200_MEMBERS)


def make_b200_no_print(stamp):
    _STAMP["current"] = f"{stamp}:00:00.000Z"
    rs = [
        receipt("alpha", 3.16, 0.027, 0.25, last_seen=_STAMP["current"]),
        receipt("bravo", 3.2, 0.04, 0.25, last_seen=_STAMP["current"]),
        receipt("charlie", 3.11, 0.02, 0.3, last_seen=_STAMP["current"]),
        receipt("delta", None, None, None, status="missing",
                verdict="not_evaluated", last_seen=None, liveness=None),
    ]
    return observation("B200", "b200_panel_annex_a2_v0_3_calc_v6", stamp,
                       rs, 4, B200_MEMBERS,
                       force_no_print_reason="insufficient_coverage")


def make_h100_withheld(stamp, base):
    obs = make_h100(stamp, base)  # true value computed pre-disclosure
    withheld = obs["receipts"][2]  # charlie, contributing
    withheld["price"] = None
    withheld["sd"] = None
    withheld["price_disclosure"] = "withheld"
    obs["restatements"] = [{
        "source_id": "charlie",
        "note": ("Provider price withheld by the effective disclosure "
                 "policy; the published index value is unchanged."),
    }]
    return obs


def series_row(obs):
    n_passing = sum(
        1 for r in obs["receipts"]
        if r["status"] == "ok" and r["filter_verdict"] == "accepted"
    )
    frozen = sorted(r["source_id"] for r in obs["receipts"]
                    if r["status"] == "frozen")
    excluded = sorted(r["source_id"] for r in obs["receipts"]
                      if r["status"] == "excluded")
    return {
        "sku": obs["sku"],
        "observed_at": obs["observed_at"],
        "generated_at": obs["generated_at"],
        "value_usd_gpu_hr": obs["value_usd_gpu_hr"],
        "stability_band_usd_gpu_hr": obs["stability_band_usd_gpu_hr"],
        "unit": obs["unit"],
        "methodology_id": obs["methodology_id"],
        "input_snapshot_sha256": obs["input_snapshot_sha256"],
        "status": obs["status"],
        "reason": obs["reason"],
        "freshness": "fresh",
        "coverage": {
            "n_sources": len(obs["receipts"]),
            "n_passing": n_passing,
            "frozen": frozen,
            "excluded": excluded,
        },
    }


h100_t14 = make_h100("2026-08-25T14", 2.1)
h100_t15 = make_h100("2026-08-25T15", 2.15)
b200_t04 = make_b200_ok("2026-08-20T04")
b200_t10 = make_b200_no_print("2026-08-20T10")
h100_withheld = make_h100_withheld("2026-08-23T14", 2.05)

spec = {
    "indexName": INDEX_NAME,
    "indexDomain": INDEX_DOMAIN,
    "commercialLicensingUrl": LICENSING_URL,
    "files": [
        {
            "path": "observations/2026/08/25.json",
            "data": {
                "kind": "gpu_index_observation_day",
                "date": "2026-08-25",
                "observations": [h100_t14, h100_t15],
            },
        },
        {
            "path": "observations/2026/08/20.json",
            "data": {
                "kind": "gpu_index_observation_day",
                "date": "2026-08-20",
                "observations": [b200_t04, b200_t10],
            },
        },
        {
            "path": "observations/2026/08/23.json",
            "data": {
                "kind": "gpu_index_observation_day",
                "date": "2026-08-23",
                "observations": [h100_withheld],
            },
        },
        {
            "path": "latest.json",
            "data": {
                "kind": "gpu_index_latest",
                "observations": [h100_t15, b200_t04],
            },
        },
        {
            "path": "series/24h.json",
            "data": {
                "kind": "gpu_index_series",
                "range": "24h",
                "observations": [series_row(h100_t14),
                                 series_row(h100_t15)],
            },
        },
    ],
}

out = Path(sys.argv[1])
out.write_text(json.dumps(spec, indent=1))
print("wrote", out)
print("H100 T14 value", h100_t14["value_usd_gpu_hr"],
      "band", h100_t14["stability_band_usd_gpu_hr"])
print("B200 T04 value", b200_t04["value_usd_gpu_hr"],
      "band", b200_t04["stability_band_usd_gpu_hr"])
