# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for the hourly panel engine core (stage 3 of
METHODOLOGY.md).

Pins: the panel config loader's acceptance shape and every rejection
class (prefix fences, record-source contiguity, member/variant/statistic
rules, calc requirements incl. hour-scoped exclusions and the jump-screen
thresholds, the tier ALLOW-LIST eligible_tiers with the retired
interruptible_tiers key refused, and the member extra_require screen);
the identity/variant screens' boundary-token behavior over
stored sku_identifier (pre-catalog-change history, SXM5/alpha-digit
splits, NVL-vs-NVLink boundaries); the three panel statistics (the
population-accounting gate, both thin-book floors, hand-computed
volume-weighted medians incl. the straddle-average rule and lium's
extra.gpu_count volume); the calc-lane jump screen (quarantine,
corroborated pass, starvation stand-down, missing-reference report-only,
same-machine delta) and its artifact-only verdict reach (no window entry,
no weight-series print, out of the eligible set); FX rules (EUR converts,
BRL/INR never candidates, fx_lane none forces hold-out); the claim-floor
dark artifact; and a full compute_observation golden observation with a
hand-computed median-of-CI-votes index and dispersion.

The daily engine's own suites (test_index_composite.py,
test_dynamic_weights.py) run alongside -- nothing here touches their
functions' behavior.
"""

from __future__ import annotations

import json

import pytest

from gpu_index.index.composite import CURRENCY_CONFIRM_DAYS, SOURCE_STATISTIC_FNS
from gpu_index.index.panel import (
    PANEL_STATISTIC_FNS,
    apply_panel_jump_screen,
    compile_screens,
    compute_observation,
    jump_reference_prints,
    lium_vwm_book_floor,
    lowest_eligible_print,
    member_eligible_rows,
    panel_calc_params,
    record_source_for,
    vast_vwm_verified_us_ca_floor,
    vast_vwm_verified_us_ca_v2,
    _token_patterns,
)
from gpu_index.index.panel_config import (
    KNOWN_LANE_PREFIXES,
    RAW_OBSERVATORY_PREFIX,
    PanelConfigError,
    load_panel_config,
    panel_schedule,
    validate_panel_config,
)
from gpu_index.index.panel_schedule import date_hour_to_stamp
from gpu_index.index.weights import new_weight_state
from gpu_index.observatory.config import RESERVED_LANE_PREFIXES

GENESIS = "2026-08-23"
STAMP0 = date_hour_to_stamp(GENESIS, 0)

FX = {
    GENESIS: {
        "source": "ecb_reference_rate",
        "as_of": GENESIS,
        "rates": {"USD": 1.15},
    }
}


# ----------------------------------------------------------------- fixtures


def _member(sid, weight, skus=("H100",), statistic=None, variant=None):
    member = {"source_id": sid, "weight": weight, "skus": list(skus)}
    if statistic is not None:
        member["statistic"] = statistic
    if variant is not None:
        member["variant"] = variant
    return member


def _config():
    """A valid H100-SXM-shaped panel config (6 members, hourly grid)."""
    sxm = {"mode": "label", "require_tokens": ["SXM"]}
    return {
        "panel_id": "h100_sxm_test",
        "bucket_prefix": "index/h100_sxm",
        "genesis_date": GENESIS,
        "record_sources": [
            {
                "kind": "observatory",
                "prefix": "index/raw_observatory",
                "from_date": GENESIS,
            }
        ],
        "slot_grids": [
            {"from_date": GENESIS, "slot_hours_utc": list(range(24))}
        ],
        # TOP-LEVEL operational knob (never in calc/calc_params).
        "drift_scan_observations": 48,
        "reject_tokens": ["PCIE", "NVL", "H800", "H20", "H100T", "GH200"],
        "members": [
            _member("alpha", 0.3, variant=sxm),
            _member(
                "bravo",
                0.2,
                variant={
                    "mode": "declared",
                    "evidence": "pricing page names the product NVIDIA HGX H100",
                },
            ),
            _member("charlie", 0.2, variant=sxm),
            _member("delta", 0.1, variant=sxm),
            _member(
                "vast", 0.1, statistic="vast_vwm_verified_us_ca_floor", variant=sxm
            ),
            _member(
                "lium",
                0.1,
                statistic="lium_vwm_book_floor",
                variant={"mode": "label", "require_tokens": ["HBM3"]},
            ),
        ],
        "calc": {
            "methodology_id": "h100_sxm_test_calc_v1",
            "min_sources_to_claim": 2,
            "eligible_tiers": ["on-demand"],
            "filter_window": 20,
            "filter_sigma": 3.0,
            "filter_warmup": 10,
            "filter_sigma_floor": 0.05,
            "filter_terms": "recorded_currency",
            "composite_statistic": "median_ci_votes",
            "fx_lane": "ecb",
            "manual_exclusions": [],
            "jump_screen": {
                "quarantine_pct": 25.0,
                "corroborate_pct": 10.0,
                "min_corroborators": 2,
                "reference_max_lookback": 24,
            },
            "dynamic_weights": {
                "scheme": "predictive_v1",
                "lookback_horizons_hours": [1, 2],
                "forward_horizons_hours": [1, 2],
                "history_days": 2,
                "half_life_days": 1,
                "ridge_lambda": 0.001,
                "gamma": 4.0,
                "weight_min": 0.025,
                "weight_max": 0.30,
                "min_train_samples": 3,
                "target_variance_floor": 1e-12,
                "switch_min_eligible": 3,
                "max_abs_log_return": 0.5,
                "source_weight_caps": {},
                "attendance_floor": 0.5,
            },
        },
    }


def _obs(
    sku,
    identifier,
    usd=None,
    native=None,
    currency="USD",
    tier="on-demand",
    basis=1,
    implausible=False,
    machine_id=None,
    host_id=None,
    verification=None,
    region=None,
    extra=None,
):
    row = {
        "sku": sku,
        "sku_match": "catalog",
        "sku_identifier": identifier,
        "price_usd_gpu_hr": usd,
        "price_native_per_gpu_hr": native if native is not None else usd,
        "currency": currency,
        "raw_value": str(usd if usd is not None else native),
        "raw_unit": "usd_per_gpu_hr",
        "gpu_count_basis": basis,
        "tier": tier,
        "region": region or "?",
        "notes": "",
        "implausible": implausible,
    }
    if machine_id is not None:
        row["machine_id"] = machine_id
    if host_id is not None:
        row["host_id"] = host_id
    if verification is not None:
        row["verification"] = verification
    if extra is not None:
        row["extra"] = extra
    return row


def _entry(sid, observations, status="ok", book_stats=None):
    entry = {"source_id": sid, "status": status, "observations": observations}
    if book_stats is not None:
        entry["book_stats"] = book_stats
    return entry


def _snapshot(entries, run_id="run-1", late_fill=False):
    return {"sources": entries, "run_id": run_id, "late_fill": late_fill}


def _state():
    return {
        "window_history": {},
        "window_currencies": {},
        "pending_currencies": {},
        "weight_state": new_weight_state(),
    }


def _compute(config, snapshot, state, stamp=STAMP0, fx=None, ref=None, ref_label=None):
    return compute_observation(
        config=config,
        obs_stamp=stamp,
        snapshot=snapshot,
        fx_records=fx or {},
        window_history=state["window_history"],
        window_currencies=state["window_currencies"],
        pending_currencies=state["pending_currencies"],
        weight_state=state["weight_state"],
        reference_prints=ref,
        reference_label=ref_label,
    )


def _reject(match, mutate):
    cfg = _config()
    mutate(cfg)
    with pytest.raises(PanelConfigError, match=match):
        validate_panel_config(cfg)


# ------------------------------------------------------------ config loader


def test_load_panel_config_roundtrip(tmp_path):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(_config()))
    cfg = load_panel_config(path)
    assert cfg["_config_path"] == str(path)
    params = panel_calc_params(cfg)
    assert params["methodology_id"] == "h100_sxm_test_calc_v1"
    assert params["record_sources"][0]["kind"] == "observatory"
    assert params["jump_screen"]["reference_max_lookback"] == 24
    assert params["dynamic_weights"]["attendance_floor"] == 0.5
    # statistic params resolve chosen priors for exactly the named ids.
    assert params["statistic_params"] == {
        "lium_vwm_book_floor": {
            "min_population_machines": 5,
            "min_population_miners": 3,
        },
        "vast_vwm_verified_us_ca_floor": {
            "min_population_hosts": 3,
            "min_population_machines": 5,
        },
    }
    schedule = panel_schedule(cfg)
    assert schedule.genesis_stamp == STAMP0
    with pytest.raises(PanelConfigError, match="missing"):
        load_panel_config(tmp_path / "nope.json")


def test_config_prefix_fences():
    _reject("under 'index/'", lambda c: c.update(bucket_prefix="curves/x"))
    _reject(
        "clean path segments",
        lambda c: c.update(bucket_prefix="index/../curves"),
    )
    _reject(
        "read-only record",
        lambda c: c.update(bucket_prefix="index/raw_observatory"),
    )
    _reject(
        "read-only record",
        lambda c: c.update(bucket_prefix="index/raw_observatory/panels"),
    )
    _reject(
        "nests with",
        lambda c: c.update(bucket_prefix="index/b300_basket/hourly"),
    )
    # EQUALITY with a non-record lane prefix is sanctioned: the migrated
    # B300/B200 hourly lanes live under their basket lane's prefix.
    cfg = _config()
    cfg["bucket_prefix"] = "index/b300_basket"
    validate_panel_config(cfg)


def test_known_lane_prefixes_lockstep_with_observatory_fence():
    # SET EQUALITY both ways (security review, superseding the old
    # subset assert): the observatory fence must reserve EVERY non-record
    # lane keyspace the panel side knows, and a lane either side forgets
    # can no longer hide behind a one-way subset. The observatory's own
    # prefix is the ONE legitimate difference -- its validator refuses
    # equality with a reserved prefix, so a self-entry would refuse the
    # real observatory config.
    assert set(KNOWN_LANE_PREFIXES) - {RAW_OBSERVATORY_PREFIX} == set(
        RESERVED_LANE_PREFIXES
    )
    assert RAW_OBSERVATORY_PREFIX in KNOWN_LANE_PREFIXES
    assert RAW_OBSERVATORY_PREFIX not in RESERVED_LANE_PREFIXES


def test_config_record_sources_rejections():
    def _two_eras(cfg, second_from):
        cfg["record_sources"] = [
            {
                "kind": "basket",
                "prefix": "index/b300_basket",
                "from_date": GENESIS,
                "to_date": "2026-08-25",
            },
            {
                "kind": "observatory",
                "prefix": "index/raw_observatory",
                "from_date": second_from,
            },
        ]

    _reject("missing 'record_sources'", lambda c: c.update(record_sources=[]))
    _reject(
        "kind must be",
        lambda c: c["record_sources"][0].update(kind="csv"),
    )
    _reject(
        "must equal genesis_date",
        lambda c: c["record_sources"][0].update(from_date="2026-08-24"),
    )
    _reject("contiguous", lambda c: _two_eras(c, "2026-08-28"))
    _reject("contiguous", lambda c: _two_eras(c, "2026-08-25"))
    _reject(
        "open-ended",
        lambda c: c["record_sources"][0].update(to_date="2026-09-01"),
    )
    # Compact ISO ('20260823') parses but is non-canonical -- it would
    # validate and then never string-match downstream (the basket lesson).
    _reject(
        "canonical YYYY-MM-DD",
        lambda c: c["record_sources"][0].update(from_date="20260823"),
    )
    # A valid basket -> observatory cutover loads.
    cfg = _config()
    _two_eras(cfg, "2026-08-26")
    validate_panel_config(cfg)


def test_config_member_rejections():
    _reject(
        "duplicate member",
        lambda c: c["members"].append(_member("alpha", 0.0001)),
    )
    _reject(
        "exact at 6 decimal places",
        lambda c: c["members"][0].update(weight=0.1234567),
    )
    _reject("sum to 1.0", lambda c: c["members"][0].update(weight=0.25))
    _reject("skus must be", lambda c: c["members"][0].update(skus=[]))
    _reject(
        "unknown panel statistic",
        lambda c: c["members"][0].update(statistic="vast_vwm_verified_us_ca"),
    )
    _reject(
        "variant.mode",
        lambda c: c["members"][0].update(variant={"mode": "guess"}),
    )
    _reject(
        "require_tokens",
        lambda c: c["members"][0].update(variant={"mode": "label"}),
    )
    _reject(
        "evidence",
        lambda c: c["members"][0].update(variant={"mode": "declared"}),
    )
    _reject(
        "must not also carry require_tokens",
        lambda c: c["members"][0].update(
            variant={
                "mode": "declared",
                "evidence": "x",
                "require_tokens": ["SXM"],
            }
        ),
    )
    _reject(
        r"normalize to \[A-Z0-9 \]\+",
        lambda c: c.update(reject_tokens=["??"]),
    )


def test_config_calc_rejections():
    def calc(c):
        return c["calc"]

    _reject(
        "min_sources_to_claim",
        lambda c: calc(c).pop("min_sources_to_claim"),
    )
    _reject(
        "min_sources_to_claim",
        lambda c: calc(c).update(min_sources_to_claim=7),
    )
    _reject(
        "filter_sigma_floor > 0",
        lambda c: calc(c).update(filter_sigma_floor=0),
    )
    _reject("fx_lane", lambda c: calc(c).pop("fx_lane"))
    _reject("jump_screen", lambda c: calc(c).pop("jump_screen"))
    _reject(
        "must not exceed",
        lambda c: calc(c)["jump_screen"].update(corroborate_pct=30.0),
    )
    _reject(
        "min_corroborators",
        lambda c: calc(c)["jump_screen"].update(min_corroborators=0),
    )
    _reject(
        "reference_max_lookback",
        lambda c: calc(c)["jump_screen"].update(reference_max_lookback=0),
    )
    _reject(
        "dynamic_weights is required",
        lambda c: calc(c).pop("dynamic_weights"),
    )
    _reject(
        "attendance_floor",
        lambda c: calc(c)["dynamic_weights"].pop("attendance_floor"),
    )
    # The drift-scan bound is a TOP-LEVEL operational key: required there,
    # REFUSED in calc (the retired location must never ride calc_params).
    _reject(
        "drift_scan_observations",
        lambda c: c.pop("drift_scan_observations"),
    )
    _reject(
        "TOP-LEVEL drift_scan_observations",
        lambda c: calc(c).update(drift_scan_observations=48),
    )
    _reject(
        "unknown statistic",
        lambda c: calc(c).update(statistic_params={"nope": {}}),
    )
    _reject(
        "not named by any member",
        lambda c: calc(c).update(
            statistic_params={"vast_vwm_verified_us_ca_v2": {}}
        ),
    )
    _reject(
        "unknown param",
        lambda c: calc(c).update(
            statistic_params={"lium_vwm_book_floor": {"min_hosts": 2}}
        ),
    )

    def _coarse_grid(c):
        # Final era is the 4-slot basket grid: a 7h horizon is inexpressible.
        c["slot_grids"] = [
            {"from_date": GENESIS, "slot_hours_utc": [4, 10, 16, 22]}
        ]
        calc(c)["dynamic_weights"]["lookback_horizons_hours"] = [6, 7]
        calc(c)["dynamic_weights"]["forward_horizons_hours"] = [6, 12]

    _reject("not expressible", _coarse_grid)


def test_config_dw_bounds_and_feasibility_rejections():
    """Coverage stage: the dynamic-weights bound fences -- weight_min
    above weight_max, an N*w_min allocation that cannot sum to 1, a
    history span that can never define a score, and the
    source_weight_caps shape/bounds/membership rules."""

    def dw(c):
        return c["calc"]["dynamic_weights"]

    _reject(
        "weight_min",
        lambda c: dw(c).update(weight_min=0.4, weight_max=0.3),
    )
    # 6 members x 0.2 = 1.2 >= 1: no vector can respect the floor.
    _reject(
        "infeasible",
        lambda c: dw(c).update(weight_min=0.2),
    )
    # history 1 day (24h) < max lookback 2 + max forward 2 +
    # min_train_samples 21 * 1h spacing + 1h = 26h: no score can EVER
    # accumulate min_train_samples -- the silently-dead-scheme class.
    _reject(
        "can never define a score",
        lambda c: dw(c).update(history_days=1, min_train_samples=21),
    )
    _reject(
        "source_weight_caps must be an object",
        lambda c: dw(c).update(source_weight_caps=[["alpha", 0.3]]),
    )
    _reject(
        "not a configured member",
        lambda c: dw(c).update(source_weight_caps={"zed": 0.3}),
    )
    # A cap below weight_min contradicts the floor.
    _reject(
        r"source_weight_caps\['alpha'\]",
        lambda c: dw(c).update(source_weight_caps={"alpha": 0.01}),
    )
    # A well-formed cap validates.
    cfg = _config()
    cfg["calc"]["dynamic_weights"]["source_weight_caps"] = {"alpha": 0.25}
    validate_panel_config(cfg)


def test_config_filter_vocabulary_and_exclusion_shape_rejections():
    """Coverage stage: filter_terms / composite_statistic must name a
    known vocabulary entry, and manual_exclusions must be a list of
    objects (a dict or a bare string would silently exclude nothing)."""
    _reject(
        "filter_terms",
        lambda c: c["calc"].update(filter_terms="usd_only_maybe"),
    )
    _reject(
        "composite_statistic",
        lambda c: c["calc"].update(composite_statistic="mean_of_means"),
    )
    _reject(
        "manual_exclusions must be a list",
        lambda c: c["calc"].update(
            manual_exclusions={"date": GENESIS, "source_id": "alpha"}
        ),
    )
    _reject(
        "entries must be objects",
        lambda c: c["calc"].update(manual_exclusions=["alpha"]),
    )


def test_load_panel_config_refuses_unparseable_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ this is not json")
    with pytest.raises(PanelConfigError, match="unparseable"):
        load_panel_config(path)


def test_config_methodology_id_key_segment_and_frozen_fence():
    """Security review: methodology_id becomes a bucket key segment, and
    under a shared daily-lane prefix it must never name a FROZEN daily
    series (a typo'd id would publish hour-keyed artifacts into the
    frozen keyspace and move its latest.json pointer)."""
    from gpu_index.index.panel_config import FROZEN_METHODOLOGY_IDS

    assert FROZEN_METHODOLOGY_IDS == (
        {f"annex_a_v0_2_calc_v{i}" for i in range(1, 7)}
        | {f"annex_a2_v0_3_calc_v{i}" for i in range(1, 6)}
    )
    for bad in ("a/b", "..", ".", "calc_v1\n", "calc v1", ""):
        _reject(
            "methodology_id",
            lambda c, bad=bad: c["calc"].update(methodology_id=bad),
        )

    def frozen_b300(c):
        c["bucket_prefix"] = "index/b300_basket"
        c["calc"]["methodology_id"] = "annex_a_v0_2_calc_v6"

    _reject("FROZEN daily series", frozen_b300)

    def frozen_b200(c):
        c["bucket_prefix"] = "index/b200_basket"
        c["calc"]["methodology_id"] = "annex_a2_v0_3_calc_v5"

    _reject("FROZEN daily series", frozen_b200)

    # The real mints under the shared prefixes validate...
    cfg = _config()
    cfg["bucket_prefix"] = "index/b300_basket"
    cfg["calc"]["methodology_id"] = "annex_a_v0_2_calc_v7"
    validate_panel_config(cfg)
    # ... and a frozen id under a NON-daily prefix is a different
    # keyspace entirely (no collision to fence).
    cfg = _config()
    cfg["calc"]["methodology_id"] = "annex_a_v0_2_calc_v6"
    validate_panel_config(cfg)


def test_config_unknown_keys_refused_at_every_level():
    """Adversarial review F12: an unread key is silently inert config --
    a typo'd 'rejected_tokens' would ship a panel with NO identity screen
    while its author reads a live one. One rejection per documented
    level."""
    _reject(
        "unrecognized key",
        lambda c: c.update(rejected_tokens=["NVL"]),  # top-level typo
    )
    _reject(
        "unrecognized key",
        lambda c: c["calc"].update(statistic_parms={}),  # calc typo
    )
    _reject(
        "unrecognized key",
        lambda c: c["members"][0].update(weights=0.3),  # member typo
    )
    _reject(
        "unrecognized key",
        lambda c: c["calc"]["dynamic_weights"].update(attendence_floor=0.5),
    )
    _reject(
        "unrecognized key",
        lambda c: c["calc"]["jump_screen"].update(quarantine_pcts=1),
    )
    _reject(
        "unrecognized key",
        lambda c: c["calc"].update(
            manual_exclusions=[
                {
                    "date": GENESIS,
                    "source_id": "alpha",
                    "reason": "x",
                    "hours": 3,  # typo of 'hour': would exclude the DAY
                }
            ]
        ),
    )
    _reject(
        "unrecognized key",
        lambda c: c.update(
            record_exclusions=[
                {"date": GENESIS, "hour": 0, "reason": "x", "source_id": "a"}
            ]
        ),
    )
    # The loader's own bookkeeping key stays admissible: a loaded config
    # must re-validate.
    cfg = _config()
    cfg["_config_path"] = "/tmp/panel.json"
    validate_panel_config(cfg)


def test_config_record_exclusions_validation():
    """The F6 escape hatch's load fences: canonical date, on-grid hour,
    non-empty reason, no duplicates; the resolved entries ride
    calc_params (they shape the series)."""
    base = {"date": GENESIS, "hour": 0, "reason": "poisoned object"}
    _reject(
        "canonical YYYY-MM-DD",
        lambda c: c.update(record_exclusions=[dict(base, date="20260823")]),
    )
    _reject(
        "hour",
        lambda c: c.update(record_exclusions=[dict(base, hour=24)]),
    )
    _reject(
        "hour",
        lambda c: c.update(record_exclusions=[dict(base, hour=True)]),
    )
    _reject(
        "non-empty reason",
        lambda c: c.update(record_exclusions=[dict(base, reason=" ")]),
    )
    _reject(
        "duplicate",
        lambda c: c.update(record_exclusions=[dict(base), dict(base)]),
    )
    # Pre-genesis (never scheduled): would quarantine nothing, refused.
    _reject(
        "not a scheduled observation hour",
        lambda c: c.update(record_exclusions=[dict(base, date="2026-08-22")]),
    )
    _reject("must be a list", lambda c: c.update(record_exclusions={}))
    cfg = _config()
    cfg["record_exclusions"] = [dict(base)]
    validate_panel_config(cfg)
    params = panel_calc_params(cfg)
    assert params["record_exclusions"] == [base]
    # Absent key resolves to the empty pinned list (unconditional embed).
    assert panel_calc_params(_config())["record_exclusions"] == []


def test_config_tier_allow_list_rejections():
    def calc(c):
        return c["calc"]

    # REQUIRED: the allow-list is lane law.
    _reject("eligible_tiers", lambda c: calc(c).pop("eligible_tiers"))
    _reject(
        "non-empty list", lambda c: calc(c).update(eligible_tiers=[])
    )
    _reject(
        "non-empty list",
        lambda c: calc(c).update(eligible_tiers=["on-demand", " "]),
    )
    _reject(
        "must contain 'on-demand'",
        lambda c: calc(c).update(eligible_tiers=["reserved"]),
    )
    # The retired exclusion-list key is refused LOUDLY, naming the
    # successor -- a silently-ignored key would read as a live screen.
    _reject(
        "ALLOW-LIST calc.eligible_tiers",
        lambda c: calc(c).update(interruptible_tiers=["spot", "preemptible"]),
    )


def test_config_extra_require_rejections():
    def alpha(c):
        return c["members"][0]

    _reject(
        "extra_require must be a non-empty object",
        lambda c: alpha(c).update(extra_require={}),
    )
    _reject(
        "extra_require must be a non-empty object",
        lambda c: alpha(c).update(extra_require=["cloud", "secure"]),
    )
    _reject(
        "non-empty string keys",
        lambda c: alpha(c).update(extra_require={"cloud": ""}),
    )
    _reject(
        "non-empty string keys",
        lambda c: alpha(c).update(extra_require={" ": "secure"}),
    )
    _reject(
        "non-empty string keys",
        lambda c: alpha(c).update(extra_require={"cloud": 1}),
    )
    # The well-formed screen validates.
    cfg = _config()
    cfg["members"][0]["extra_require"] = {"cloud": "secure"}
    validate_panel_config(cfg)


def test_config_manual_exclusion_rejections():
    def with_exclusions(c, entries):
        c["calc"]["manual_exclusions"] = entries

    base = {"date": GENESIS, "source_id": "alpha", "reason": "incident"}
    _reject(
        "canonical YYYY-MM-DD",
        lambda c: with_exclusions(c, [dict(base, date="20260823")]),
    )
    _reject(
        "not a configured member",
        lambda c: with_exclusions(c, [dict(base, source_id="zed")]),
    )
    _reject(
        "non-empty reason",
        lambda c: with_exclusions(c, [dict(base, reason=" ")]),
    )
    _reject("hour", lambda c: with_exclusions(c, [dict(base, hour=24)]))
    _reject("hour", lambda c: with_exclusions(c, [dict(base, hour=True)]))
    _reject(
        "duplicate",
        lambda c: with_exclusions(c, [dict(base, hour=3), dict(base, hour=3)]),
    )
    _reject(
        "one scope per",
        lambda c: with_exclusions(c, [dict(base), dict(base, hour=3)]),
    )
    # A well-formed mixed list validates.
    cfg = _config()
    with_exclusions(
        cfg,
        [dict(base), dict(base, date="2026-08-24", hour=3)],
    )
    validate_panel_config(cfg)


def test_panel_registry_is_disjoint_from_the_daily_registry():
    assert set(PANEL_STATISTIC_FNS) == {
        "vast_vwm_verified_us_ca_v2",
        "vast_vwm_verified_us_ca_floor",
        "lium_vwm_book_floor",
    }
    assert not set(PANEL_STATISTIC_FNS) & set(SOURCE_STATISTIC_FNS)


# ------------------------------------------------- identity/variant screens


def _screen_rows(
    rows, *, skus, reject=(), require=None, extra_require=None, counts=None
):
    reject_patterns = [p for t in reject for p in _token_patterns(t)]
    require_patterns = (
        None if require is None else [p for t in require for p in _token_patterns(t)]
    )
    return member_eligible_rows(
        _entry("s", rows),
        skus=set(skus),
        reject_patterns=reject_patterns,
        require_patterns=require_patterns,
        eligible_tiers=("on-demand",),
        extra_require=extra_require,
        screen_counts=counts,
    )


def test_identity_screen_rejects_pre_catalog_variant_history():
    # Pre-catalog-change history: an H200 NVL row FILED under sku "H200"
    # must be rejected by the H200-SXM panel's NVL token; boundary rules
    # keep 'H20' from firing inside 'H200' and 'NVL' inside 'NVLink'.
    rows = [
        _obs("H200", "H200 NVL", 3.1),
        _obs("H200", "NVIDIA H200 SXM", 3.3),
        _obs("H200", "H200 NVLink platform", 3.4),
        _obs("H200", "H200", 3.5, tier="spot"),
        _obs("H200", "H200", 3.6, implausible=True),
        _obs("H100", "H100 SXM", 2.4),  # wrong sku for this member
    ]
    kept = _screen_rows(rows, skus={"H200"}, reject=("NVL", "GH200", "H20"))
    assert [r["sku_identifier"] for r in kept] == [
        "NVIDIA H200 SXM",
        "H200 NVLink platform",
    ]


def test_variant_label_mode_requires_the_token():
    rows = [
        _obs("H100", "H100", 2.1),  # unqualified label proves no variant
        _obs("H100", "H100 SXM5 80GB", 2.4),  # SXM5 matches SXM (digit split)
        _obs("H100", "h100-sxm5-80gb", 2.5),
        _obs("H100", "H100 PCIe", 2.0),  # rejected before the variant rule
    ]
    kept = _screen_rows(
        rows, skus={"H100"}, reject=("PCIE", "NVL"), require=("SXM",)
    )
    assert [r["price_usd_gpu_hr"] for r in kept] == [2.4, 2.5]


def test_variant_declared_mode_admits_bare_labels():
    rows = [
        _obs("H100", "H100", 2.1),
        _obs("H100", "H100 PCIe", 2.0),  # panel reject still applies
    ]
    kept = _screen_rows(rows, skus={"H100"}, reject=("PCIE", "NVL"))
    assert [r["price_usd_gpu_hr"] for r in kept] == [2.1]


def test_tier_allow_list_admits_on_demand_only():
    # Methodology section 5 / the hourly-mint tier reconciliation: the tier
    # screen is an ALLOW-LIST, so reserved, committed, serverless and
    # from-floor rows are ineligible by construction -- including tiers
    # the retired exclusion list never enumerated.
    rows = [
        _obs("H100", "H100 SXM", 2.4),  # on-demand (the _obs default)
        _obs("H100", "H100 SXM", 1.0, tier="reserved"),
        _obs("H100", "H100 SXM", 1.1, tier="monthly-commit"),
        _obs("H100", "H100 SXM", 1.2, tier="serverless"),
        _obs("H100", "H100 SXM", 1.3, tier="from_floor"),
        _obs("H100", "H100 SXM", 1.4, tier="spot"),
        _obs("H100", "H100 SXM", 1.5, tier="preemptible"),
    ]
    kept = _screen_rows(rows, skus={"H100"})
    assert [r["price_usd_gpu_hr"] for r in kept] == [2.4]


def test_extra_require_screens_on_the_structured_extra_dict():
    # FIX 2: key-present-wrong-value is excluded and counted; a row with
    # NO extra dict (basket-era record) or an extra without the key
    # PASSES -- the one documented fail-open (member_eligible_rows).
    rows = [
        _obs("H100", "H100 SXM", 3.29, extra={"cloud": "secure"}),
        _obs("H100", "H100 SXM", 2.69, extra={"cloud": "community"}),
        _obs("H100", "H100 SXM", 3.10),  # basket-era: no extra at all
        _obs("H100", "H100 SXM", 2.50, extra={"display_name": "H100"}),
    ]
    counts = {}
    kept = _screen_rows(
        rows, skus={"H100"}, extra_require={"cloud": "secure"}, counts=counts
    )
    assert [r["price_usd_gpu_hr"] for r in kept] == [3.29, 3.10, 2.5]
    assert counts == {"extra_require_mismatch": 1}


def test_non_finite_prices_are_held_out_at_eligibility():
    """Finiteness fail-closed (adversarial review F9): json admits
    NaN/Infinity and non-USD rows skip the capture plausibility band --
    a non-finite native (or a non-finite USD, when present) is excluded
    and counted, never a candidate. The statistics inherit the screen
    (they only ever see the returned rows)."""
    rows = [
        _obs("H100", "H100 SXM", usd=None, native=float("nan"), currency="EUR"),
        _obs("H100", "H100 SXM", usd=None, native=float("inf"), currency="EUR"),
        _obs("H100", "H100 SXM", usd=float("inf")),
        _obs("H100", "H100 SXM", usd=None, native=None, currency="EUR"),
        _obs("H100", "H100 SXM", 2.4),
        _obs("H100", "H100 SXM", usd=None, native=2.0, currency="EUR"),
    ]
    counts: dict = {}
    kept = _screen_rows(rows, skus={"H100"}, counts=counts)
    assert [r["price_native_per_gpu_hr"] for r in kept] == [2.4, 2.0]
    assert counts == {"non_finite_price": 4}
    # End to end: an Infinity EUR row must never poison the member's
    # lowest-eligible candidate set.
    chosen = lowest_eligible_print(kept, obs_date=GENESIS, fx_records=FX)
    assert chosen["usd_per_gpu_hr"] == 2.3  # EUR 2.0 * 1.15


def test_silent_ok_seat_pins_eligible_rows_and_screen_counts():
    """Dead-seat visibility (adversarial review F1): a member whose rows
    ALL screen out keeps status ok but pins eligible_rows 0 + the
    per-screen screen_counts block; a member with NO rows for its skus
    pins eligible_rows 0 WITHOUT screen_counts -- the two silences stay
    distinguishable forever (the h200 vast dead seat's tripwire)."""
    cfg = _config()
    snapshot = _snapshot(
        [
            # alpha requires SXM: its one bare-label row screens out.
            _entry("alpha", [_obs("H100", "H100", 2.2)]),
            _entry("bravo", [_obs("H100", "H100", 2.5)]),  # declared: prints
            # charlie's only row is a foreign sku: no rows at all.
            _entry("charlie", [_obs("B200", "B200", 1.0)]),
            _entry("delta", [_obs("H100", "H100 SXM", 2.6)]),
        ]
    )
    payload = _compute(cfg, snapshot, _state())
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    alpha = by_sid["alpha"]
    assert alpha["status"] == "ok"  # the F1 shape: ok, silent...
    assert "chosen" not in alpha
    assert alpha["eligible_rows"] == 0  # ...but no longer invisible
    assert alpha["screen_counts"] == {"variant_unmatched": 1}
    charlie = by_sid["charlie"]
    assert charlie["status"] == "ok"
    assert charlie["eligible_rows"] == 0
    assert "screen_counts" not in charlie  # no-rows-at-all: distinct
    bravo = by_sid["bravo"]
    assert bravo["status"] == "ok" and "eligible_rows" not in bravo
    json.dumps(payload)


def test_record_quarantined_observation_artifact_and_guard():
    """F6 at the engine: a quarantined stamp publishes an explicit
    record_quarantined artifact (index null, observation_missed FALSE --
    bytes exist, they are quarantined, not missing), and passing a
    snapshot alongside the quarantine raises (the record must never be
    priced)."""
    cfg = _config()
    state = _state()
    payload = compute_observation(
        config=cfg,
        obs_stamp=STAMP0,
        snapshot=None,
        fx_records={},
        window_history=state["window_history"],
        window_currencies=state["window_currencies"],
        pending_currencies=state["pending_currencies"],
        weight_state=state["weight_state"],
        record_quarantined="poisoned object at this stamp",
    )
    assert payload["observation_missed"] is False
    assert payload["record_quarantined"] == "poisoned object at this stamp"
    assert payload["panel_dark"] is True
    assert payload["index"] is None
    assert all(s["status"] == "missing" for s in payload["sources"])
    json.dumps(payload)
    # Normal observations carry the key as an explicit null.
    normal = _compute(cfg, _golden_snapshot(), _state())
    assert normal["record_quarantined"] is None
    with pytest.raises(ValueError, match="quarantined"):
        compute_observation(
            config=cfg,
            obs_stamp=STAMP0,
            snapshot=_golden_snapshot(),
            fx_records={},
            window_history={},
            window_currencies={},
            pending_currencies={},
            weight_state=new_weight_state(),
            record_quarantined="poisoned object at this stamp",
        )


# --------------------------------------------------------------- statistics


def _vast_rows():
    return [
        _obs(
            "B200", "B200", 2.0, basis=8,
            machine_id="m1", host_id="h1",
            verification="verified", region="Oregon, US",
        ),
        _obs(
            "B200", "B200", 3.0, basis=8,
            machine_id="m2", host_id="h2",
            verification="verified", region=", CA",
        ),
        _obs(
            "B200", "B200", 1.5, basis=8,
            machine_id="m3", host_id="h3",
            verification="deverified", region="Oregon, US",
        ),
        _obs(
            "B200", "B200", 1.6, basis=8,
            machine_id="m4", host_id="h4",
            verification="verified", region="Taiwan, TW",
        ),
    ]


def test_vast_v2_population_accounting_gate():
    rows = _vast_rows()
    # A cheapest-only book (accounting key absent) is held out even though
    # eligible rows exist -- the branch never ran, the book may be truncated.
    held = vast_vwm_verified_us_ca_v2(
        rows,
        source_entry=_entry("vast", rows, book_stats={"B200": {"machines_total": 9}}),
        statistic="vast_vwm_verified_us_ca_v2",
        params={},
    )
    assert held["held_out"]["reason"] == "no_population_accounting"
    assert held["held_out"]["books_missing_accounting"] == ["B200"]
    # No book_stats at all (pre-branch snapshot): same hold-out.
    held = vast_vwm_verified_us_ca_v2(
        rows,
        source_entry=_entry("vast", rows),
        statistic="vast_vwm_verified_us_ca_v2",
        params={},
    )
    assert held["held_out"]["reason"] == "no_population_accounting"
    # With accounting present: VWM over verified US/CA rows only.
    # Book [(2.0, 8), (3.0, 8)]: half = 8, cum hits 8 exactly at 2.0 ->
    # straddle average (2.0 + 3.0) / 2 = 2.5.
    print_ = vast_vwm_verified_us_ca_v2(
        rows,
        source_entry=_entry(
            "vast", rows, book_stats={"B200": {"verified_us_ca_machines": 2}}
        ),
        statistic="vast_vwm_verified_us_ca_v2",
        params={},
    )
    assert print_ == {
        "usd_per_gpu_hr": 2.5,
        "statistic": "vast_vwm_verified_us_ca_v2",
        "currency": "USD",
        "n_eligible_prints": 2,
        "gpu_volume": 16,
    }


def test_vast_floor_holds_out_three_machines_one_host():
    rows = [
        _obs(
            "H100", "H100 SXM", 2.0 + i * 0.1, basis=8,
            machine_id=f"m{i}", host_id="h1",
            verification="verified", region="Oregon, US",
        )
        for i in range(3)
    ]
    stats = {"H100 SXM": {"verified_us_ca_machines": 3}}
    held = vast_vwm_verified_us_ca_floor(
        rows,
        source_entry=_entry("vast", rows, book_stats=stats),
        statistic="vast_vwm_verified_us_ca_floor",
        params={"min_population_machines": 5, "min_population_hosts": 3},
    )
    assert held["held_out"] == {
        "reason": "thin_book",
        "statistic": "vast_vwm_verified_us_ca_floor",
        "population_machines": 3,
        "population_hosts": 1,
        "min_population_machines": 5,
        "min_population_hosts": 3,
    }


def test_vast_floor_passes_and_prices_the_hand_computed_vwm():
    specs = [
        ("m1", "h1", 2.0, 40),
        ("m2", "h1", 2.5, 8),
        ("m3", "h2", 3.0, 16),
        ("m4", "h2", 5.0, 8),
        ("m5", "h3", 9.0, 8),
    ]
    rows = [
        _obs(
            "H100", "H100 SXM", usd, basis=basis,
            machine_id=m, host_id=h,
            verification="verified", region="Oregon, US",
        )
        for m, h, usd, basis in specs
    ]
    stats = {"H100 SXM": {"verified_us_ca_machines": 5}}
    print_ = vast_vwm_verified_us_ca_floor(
        rows,
        source_entry=_entry("vast", rows, book_stats=stats),
        statistic="vast_vwm_verified_us_ca_floor",
        params={"min_population_machines": 5, "min_population_hosts": 3},
    )
    # Volume weighting bites: total weight 80, half 40, cum hits 40 exactly
    # at (2.0, 40) -> straddle average (2.0 + 2.5) / 2 = 2.25 (an
    # unweighted median of the five prices would be 3.0).
    assert print_["usd_per_gpu_hr"] == 2.25
    assert print_["n_eligible_prints"] == 5
    assert print_["gpu_volume"] == 80
    assert print_["population_machines"] == 5
    assert print_["population_hosts"] == 3


def _lium_row(machine, usd, gpu_count, miner):
    return _obs(
        "H100", "H100 80GB HBM3", usd,
        machine_id=machine,
        extra={"gpu_count": gpu_count, "miner_hotkey": miner},
    )


def test_lium_statistic_volume_dedupe_and_hand_computed_vwm():
    rows = [
        _lium_row("A", 1.0, 8, "mA"),
        _lium_row("A", 0.8, 8, "mA"),  # duplicate machine: cheapest wins
        _lium_row("B", 1.2, 8, "mB"),
        _lium_row("C", 1.4, 8, "mC"),
        _lium_row("D", 1.6, 8, "mA"),
        _lium_row("E", 2.0, 16, "mB"),
        _lium_row("F", 1.1, 0, "mC"),  # bad volume: skipped AND counted
    ]
    print_ = lium_vwm_book_floor(
        rows,
        source_entry=_entry("lium", rows),
        statistic="lium_vwm_book_floor",
        params={"min_population_machines": 5, "min_population_miners": 3},
    )
    # Deduped book [(0.8,8),(1.2,8),(1.4,8),(1.6,8),(2.0,16)]: total 48,
    # half 24, cum hits 24 exactly at 1.4 -> straddle (1.4 + 1.6)/2 = 1.5.
    assert print_["usd_per_gpu_hr"] == 1.5
    assert print_["n_eligible_prints"] == 5
    assert print_["gpu_volume"] == 48
    assert print_["population_machines"] == 5
    assert print_["population_miners"] == 3
    assert print_["rows_skipped_bad_volume"] == 1


def test_lium_thin_book_holds_out_with_counts():
    rows = [
        _lium_row("A", 1.0, 8, "mA"),
        _lium_row("B", 1.2, 8, "mA"),
    ]
    held = lium_vwm_book_floor(
        rows,
        source_entry=_entry("lium", rows),
        statistic="lium_vwm_book_floor",
        params={"min_population_machines": 5, "min_population_miners": 3},
    )
    assert held["held_out"]["reason"] == "thin_book"
    assert held["held_out"]["population_machines"] == 2
    assert held["held_out"]["population_miners"] == 1


# ---------------------------------------------------------------------- fx


def test_lowest_eligible_eur_converts_brl_inr_never():
    rows = [
        _obs("H100", "H100 SXM", usd=None, native=2.0, currency="EUR"),
        _obs("H100", "H100 SXM", usd=None, native=3.0, currency="BRL"),
        _obs("H100", "H100 SXM", usd=None, native=250.0, currency="INR"),
        _obs("H100", "H100 SXM", usd=2.5),
    ]
    chosen = lowest_eligible_print(rows, obs_date=GENESIS, fx_records=FX)
    # EUR 2.0 * 1.15 = 2.30 beats USD 2.50; BRL/INR were never candidates.
    assert chosen["usd_per_gpu_hr"] == 2.3
    assert chosen["currency"] == "EUR"
    assert chosen["fx_rate"] == 1.15
    assert chosen["n_eligible_prints"] == 2
    # FX outage with no USD fallback: the loud fx_unavailable sentinel.
    outage = lowest_eligible_print(rows[:1], obs_date=GENESIS, fx_records={})
    assert outage["fx_unavailable"] is True and outage["fx_errors"]


def test_mixed_currency_partial_fx_outage_prices_usd_and_records_errors():
    """Coverage stage (recipe 5): with one EUR row and one USD row and NO
    fx records, the member still prices from the USD print but the
    dropped EUR candidate stays VISIBLE -- chosen.fx_errors_partial
    carries the walk-back error (composite's mixed-currency rule)."""
    rows = [
        _obs("H100", "H100 SXM", usd=None, native=2.0, currency="EUR"),
        _obs("H100", "H100 SXM", usd=2.5),
    ]
    chosen = lowest_eligible_print(rows, obs_date=GENESIS, fx_records={})
    assert chosen["usd_per_gpu_hr"] == 2.5
    assert chosen["currency"] == "USD"
    assert chosen["n_eligible_prints"] == 1  # the EUR row never priced
    assert "fx_unavailable" not in chosen
    assert len(chosen["fx_errors_partial"]) == 1
    assert "USD rate" in chosen["fx_errors_partial"][0]


def test_fx_lane_none_forces_holdout_even_with_rates_passed():
    cfg = _config()
    cfg["calc"]["fx_lane"] = "none"
    validate_panel_config(cfg)
    snapshot = _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 2.4)]),
            _entry(
                "bravo",
                [_obs("H100", "H100", usd=None, native=2.0, currency="EUR")],
            ),
            _entry("charlie", [_obs("H100", "H100 SXM", 2.6)]),
        ]
    )
    payload = _compute(cfg, snapshot, _state(), fx=FX)
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["bravo"]["status"] == "fx_unavailable"
    assert by_sid["alpha"]["status"] == "ok"


def test_eur_member_filters_in_native_terms():
    cfg = _config()
    snapshot = _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 2.4)]),
            _entry(
                "bravo",
                [_obs("H100", "H100", usd=None, native=2.0, currency="EUR")],
            ),
        ]
    )
    state = _state()
    payload = _compute(cfg, snapshot, state, fx=FX)
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["bravo"]["chosen"]["usd_per_gpu_hr"] == 2.3
    assert by_sid["bravo"]["filter"]["currency"] == "EUR"
    # The window holds the NATIVE print (recorded-currency posture).
    assert state["window_history"]["bravo"] == [2.0]
    assert state["window_currencies"]["bravo"] == "EUR"


# -------------------------------------------------------------- jump screen


def _jump_current(prices, rows_by_sid=None):
    current = {}
    for sid, usd in prices.items():
        chosen = None
        if usd is not None:
            chosen = {
                "usd_per_gpu_hr": usd,
                "currency": "USD",
                "native_per_gpu_hr": usd,
            }
        current[sid] = {
            "chosen": chosen,
            "rows": (rows_by_sid or {}).get(sid, []),
        }
    return current


def _jump_ref(prices):
    return {
        sid: {"usd_per_gpu_hr": usd, "currency": "USD", "native_per_gpu_hr": usd}
        for sid, usd in prices.items()
    }


JUMP_PARAMS = {
    "quarantine_pct": 25.0,
    "corroborate_pct": 10.0,
    "min_corroborators": 2,
    "reference_max_lookback": 24,
}


def test_jump_screen_quarantines_an_uncorroborated_jump():
    block = apply_panel_jump_screen(
        _jump_current({"a": 3.25, "b": 2.5, "c": 2.5, "d": 2.5}),
        _jump_ref({"a": 2.5, "b": 2.5, "c": 2.5, "d": 2.5}),
        jump_params=JUMP_PARAMS,
        reference_label="2026-08-23T00",
    )
    assert block["reference"] == "2026-08-23T00"
    assert block["quarantine_skipped"] is None
    assert block["quarantined"] == [
        {"source_id": "a", "book_pct": 30.0, "corroborators": 0}
    ]
    deltas = {d["source_id"]: d for d in block["deltas"]}
    assert deltas["a"]["book_pct"] == 30.0
    assert deltas["b"]["book_pct"] == 0.0


def test_jump_screen_corroborated_move_passes():
    block = apply_panel_jump_screen(
        _jump_current({"a": 3.25, "b": 2.8, "c": 2.8, "d": 2.5}),
        _jump_ref({"a": 2.5, "b": 2.5, "c": 2.5, "d": 2.5}),
        jump_params=JUMP_PARAMS,
    )
    # b and c moved +12% (>= corroborate 10%): a's +30% is a market move.
    assert block["quarantined"] == []


def test_jump_screen_starvation_stands_down():
    block = apply_panel_jump_screen(
        _jump_current({"a": 3.25, "b": 2.5}),
        _jump_ref({"a": 2.5, "b": 2.5}),
        jump_params=JUMP_PARAMS,
    )
    assert block["quarantined"] == []
    assert "fail-open" in block["quarantine_skipped"]
    assert "['a']" in block["quarantine_skipped"]


def test_jump_screen_missing_reference_is_report_only():
    block = apply_panel_jump_screen(
        _jump_current({"a": 3.25, "b": 2.5, "c": 2.5}),
        None,
        jump_params=JUMP_PARAMS,
    )
    assert block["reference"] is None
    assert block["quarantined"] == []
    assert block["quarantine_skipped"] is None
    assert all(d["note"] == "no reference print" for d in block["deltas"])


def test_jump_screen_same_machine_delta():
    rows = [
        _obs("H100", "H100 SXM", 1.2, machine_id="m2"),
        _obs("H100", "H100 SXM", 2.0, machine_id="m1"),
    ]
    ref = {
        "a": {
            "usd_per_gpu_hr": 2.0,
            "currency": "USD",
            "native_per_gpu_hr": 2.0,
            "machine_id": "m1",
        }
    }
    block = apply_panel_jump_screen(
        _jump_current({"a": 1.2, "b": 2.5}, rows_by_sid={"a": rows}),
        dict(ref, b={"usd_per_gpu_hr": 2.5, "currency": "USD", "native_per_gpu_hr": 2.5}),
        jump_params=JUMP_PARAMS,
    )
    deltas = {d["source_id"]: d for d in block["deltas"]}
    # Book -40% beside same-machine 0%: the extraction-artifact signature.
    assert deltas["a"]["book_pct"] == -40.0
    assert deltas["a"]["same_machine_pct"] == 0.0


def test_jump_screen_eur_reference_reselects_the_same_currency_row():
    """Coverage stage (recipe 6): a mixed-currency book whose chosen print
    is USD compares against an EUR reference by RE-SELECTING its lowest
    eligible EUR row (the capture rule: compare in the reference print's
    currency), so the ratio is native-vs-native, never FX-spliced."""
    rows = [
        _obs("H100", "H100 SXM", usd=None, native=2.4, currency="EUR"),
        _obs("H100", "H100 SXM", usd=2.2),
    ]
    current = {
        "a": {
            "chosen": {
                "usd_per_gpu_hr": 2.2,
                "currency": "USD",
                "native_per_gpu_hr": 2.2,
            },
            "rows": rows,
        },
        "b": {
            "chosen": {
                "usd_per_gpu_hr": 2.5,
                "currency": "USD",
                "native_per_gpu_hr": 2.5,
            },
            "rows": [],
        },
    }
    reference = {
        "a": {"usd_per_gpu_hr": 2.3, "currency": "EUR", "native_per_gpu_hr": 2.0},
        "b": {"usd_per_gpu_hr": 2.5, "currency": "USD", "native_per_gpu_hr": 2.5},
    }
    block = apply_panel_jump_screen(
        current, reference, jump_params=JUMP_PARAMS
    )
    deltas = {d["source_id"]: d for d in block["deltas"]}
    # EUR 2.4 vs EUR 2.0 reference: +20% -- comparable, note empty.
    assert deltas["a"]["book_pct"] == 20.0
    assert deltas["a"]["note"] == ""
    assert block["quarantined"] == []


def test_jump_screen_statistic_print_and_nonpositive_reference_are_guarded():
    """Coverage stage (recipe 6): a statistic print (USD by construction,
    no native key) facing a non-USD reference
    cannot re-select a row and reports 'not comparable' WITHOUT
    quarantining, however large the USD move; a non-positive reference
    print is guarded before any ratio."""
    rows = [_obs("H100", "H100 SXM", usd=None, native=9.9, currency="EUR")]
    current = {
        "s": {
            "chosen": {
                "usd_per_gpu_hr": 9.9,
                "currency": "USD",
                "statistic": "lium_vwm_book_floor",
            },
            # An EUR row exists: the statistic path must STILL refuse to
            # re-select (its print never came from one row).
            "rows": rows,
        },
        "z": {
            "chosen": {
                "usd_per_gpu_hr": 2.5,
                "currency": "USD",
                "native_per_gpu_hr": 2.5,
            },
            "rows": [],
        },
    }
    reference = {
        "s": {"usd_per_gpu_hr": 2.3, "currency": "EUR", "native_per_gpu_hr": 2.0},
        "z": {"usd_per_gpu_hr": 2.5, "currency": "USD", "native_per_gpu_hr": 2.5},
    }
    block = apply_panel_jump_screen(current, reference, jump_params=JUMP_PARAMS)
    deltas = {d["source_id"]: d for d in block["deltas"]}
    assert deltas["s"]["note"] == "currency changed -- not comparable"
    assert deltas["s"]["book_pct"] is None
    assert block["quarantined"] == []  # a 4x USD move, still no verdict

    bad_ref = {
        "z": {"usd_per_gpu_hr": 0.0, "currency": "USD", "native_per_gpu_hr": 0.0}
    }
    block = apply_panel_jump_screen(
        {"z": current["z"]}, bad_ref, jump_params=JUMP_PARAMS
    )
    deltas = {d["source_id"]: d for d in block["deltas"]}
    assert deltas["z"]["note"] == "non-positive reference print"
    assert deltas["z"]["book_pct"] is None
    assert block["quarantined"] == []


# -------------------------------------------------------- compute_observation


def _golden_snapshot():
    return _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 2.4, machine_id="a1")]),
            _entry("bravo", [_obs("H100", "H100", 2.5)]),
            _entry("charlie", [_obs("H100", "H100 SXM5 80GB", 2.6)]),
        ]
    )


def test_compute_observation_golden():
    cfg = _config()
    state = _state()
    payload = _compute(cfg, _golden_snapshot(), state)

    assert payload["schema_version"] == 1
    assert payload["kind"] == "index_panel_composite"
    assert payload["panel_id"] == "h100_sxm_test"
    assert payload["methodology_id"] == "h100_sxm_test_calc_v1"
    assert payload["date"] == "2026-08-23T00"
    assert payload["observation_date"] == GENESIS
    assert payload["observation_hour_utc"] == 0
    assert payload["observation_missed"] is False
    assert payload["panel_dark"] is False
    assert payload["record_kind"] == "observatory"
    assert payload["snapshot_run_id"] == "run-1"
    assert payload["snapshot_late_fill"] is False

    # Hand-computed median-of-CI-votes: first-ever prints are warm-up
    # (sigma 0, floored at 0.05), so each member votes its weight at
    # price and price +/- 0.05. Sorted votes with weights
    # (alpha .3 @ 2.35/2.40/2.45, bravo .2 @ 2.45/2.50/2.55,
    #  charlie .2 @ 2.55/2.60/2.65), total weight 2.1:
    # median (1.05) -> 2.45; p25 (0.525) -> 2.40; p75 (1.575) -> 2.55.
    index = payload["index"]
    assert index["value_usd_gpu_hr"] == 2.45
    assert index["statistic"] == "median_ci_votes"
    assert index["confidence_usd_gpu_hr"] == 0.1
    assert index["vote_p25_usd_gpu_hr"] == 2.4
    assert index["vote_p75_usd_gpu_hr"] == 2.55
    assert index["weighted_mean_usd_gpu_hr"] == 2.485714
    assert index["unweighted_mean_usd_gpu_hr"] == 2.5
    assert index["renormalized_weights"] == {
        "alpha": 0.428571,
        "bravo": 0.285714,
        "charlie": 0.285714,
    }
    assert index["sources_used_count"] == 3

    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert [s["source_id"] for s in payload["sources"]] == [
        "alpha", "bravo", "charlie", "delta", "lium", "vast",
    ]
    alpha = by_sid["alpha"]
    assert alpha["status"] == "ok"
    assert alpha["weight"] == 0.3
    assert alpha["chosen"]["usd_per_gpu_hr"] == 2.4
    assert alpha["chosen"]["sku"] == "H100"
    assert alpha["chosen"]["machine_id"] == "a1"
    assert alpha["filter"] == {
        "accepted": True,
        "unfiltered": True,
        "n_history": 0,
        "currency": "USD",
    }
    assert alpha["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.05,
    }
    for sid in ("delta", "lium", "vast"):
        assert by_sid[sid]["status"] == "missing"
        assert by_sid[sid]["weight"] is None

    # Jump screen: no reference passed -> report only.
    assert payload["jump_screen"]["quarantined"] == []
    assert all(
        d["note"] == "no reference print"
        for d in payload["jump_screen"]["deltas"]
    )

    # Weight lane: genesis observation -> no cutoff -> fallback mode with
    # the config vector restricted to the eligible set; attendance at the
    # zero-scheduled genesis window is the 1.0 nothing-was-missed rule.
    wc = payload["weight_calc"]
    assert wc["mode"] == "fallback"
    assert wc["weights"] == {"alpha": 0.3, "bravo": 0.2, "charlie": 0.2}
    assert "switched_on" not in wc
    alpha_wc = wc["sources"]["alpha"]
    assert alpha_wc["Q"] is None
    assert alpha_wc["attendance_printed"] == 0
    assert alpha_wc["attendance_scheduled"] == 0
    assert alpha_wc["attendance_ratio"] == 1.0

    # calc_params embeds every grid/screen/statistic/dw param verbatim.
    cp = payload["calc_params"]
    assert cp["eligible_tiers"] == ["on-demand"]
    assert cp["record_sources"][0]["prefix"] == "index/raw_observatory"
    assert cp["slot_grids"][0]["slot_hours_utc"] == list(range(24))
    assert cp["jump_screen"]["quarantine_pct"] == 25.0
    assert cp["reject_tokens"] == sorted(
        ["PCIE", "NVL", "H800", "H20", "H100T", "GH200"]
    )
    assert cp["dynamic_weights"]["attendance_floor"] == 0.5
    assert len(cp["members"]) == 6
    assert cp["members"][0] == {
        "source_id": "alpha",
        "weight": 0.3,
        "skus": ["H100"],
        "variant": {"mode": "label", "require_tokens": ["SXM"]},
    }
    assert cp["statistic_params"]["vast_vwm_verified_us_ca_floor"] == {
        "min_population_hosts": 3,
        "min_population_machines": 5,
    }
    # The drift-scan bound is OPERATIONAL (top-level config key) and must
    # never ride calc_params/artifact bytes.
    assert "drift_scan_observations" not in cp
    json.dumps(payload)  # the artifact must be JSON-serializable as-is

    # State advanced: windows hold the prints, the weight series holds the
    # trusted prints at this stamp, the vector is stamp-keyed.
    assert state["window_history"] == {
        "alpha": [2.4], "bravo": [2.5], "charlie": [2.6],
    }
    assert state["weight_state"]["prices"]["alpha"][STAMP0] == {
        "usd": 2.4, "native": 2.4, "currency": "USD",
    }
    assert state["weight_state"]["vectors"][STAMP0] == {
        "alpha": 0.3, "bravo": 0.2, "charlie": 0.2,
    }
    assert state["weight_state"]["mode"] == "fallback"


def test_second_observation_advances_windows_and_attendance():
    cfg = _config()
    state = _state()
    _compute(cfg, _golden_snapshot(), state)
    payload = _compute(cfg, _golden_snapshot(), state, stamp=STAMP0 + 1)
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["alpha"]["filter"]["n_history"] == 1
    alpha_wc = payload["weight_calc"]["sources"]["alpha"]
    assert alpha_wc["attendance_printed"] == 1
    assert alpha_wc["attendance_scheduled"] == 1
    assert alpha_wc["attendance_ratio"] == 1.0
    assert state["window_history"]["alpha"] == [2.4, 2.4]
    assert set(state["weight_state"]["vectors"]) == {STAMP0, STAMP0 + 1}


def test_observation_missed_publishes_explicit_dark_artifact():
    cfg = _config()
    payload = _compute(cfg, None, _state())
    assert payload["observation_missed"] is True
    assert payload["panel_dark"] is True
    assert payload["index"] is None
    assert payload["snapshot_run_id"] is None
    assert all(s["status"] == "missing" for s in payload["sources"])
    assert payload["weight_calc"]["weights"] == {}


def test_claim_floor_publishes_dark_below_the_floor():
    cfg = _config()
    cfg["calc"]["min_sources_to_claim"] = 5
    validate_panel_config(cfg)
    payload = _compute(cfg, _golden_snapshot(), _state())
    assert payload["panel_dark"] is True
    assert payload["index"] is None
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["alpha"]["status"] == "ok"  # the prints still publish


def test_config_hour_scoped_exclusion_must_land_on_the_era_grid():
    """An hour-scoped exclusion at an off-grid hour would validate and then
    match no observation -- a silently inert incident record, the exact
    class the daily lanes' canonical-date rule refuses. Refused at load."""

    def two_era(c):
        c["slot_grids"] = [
            {"from_date": GENESIS, "slot_hours_utc": [4, 10, 16, 22]},
            {"from_date": "2026-08-25", "slot_hours_utc": list(range(24))},
        ]

    base = {"date": GENESIS, "source_id": "alpha", "reason": "incident"}

    # Off-grid hour inside the 4-slot era.
    def off_grid(c):
        two_era(c)
        c["calc"]["manual_exclusions"] = [dict(base, hour=15)]

    _reject("not a scheduled observation hour", off_grid)

    # Pre-genesis date: no hour of it is ever scheduled.
    def pre_genesis(c):
        c["calc"]["manual_exclusions"] = [dict(base, date="2026-08-22", hour=4)]

    _reject("not a scheduled observation hour", pre_genesis)

    # On-grid hours validate in both eras.
    cfg = _config()
    two_era(cfg)
    cfg["calc"]["manual_exclusions"] = [
        dict(base, hour=16),
        dict(base, date="2026-08-25", hour=15),
    ]
    validate_panel_config(cfg)


def test_manual_exclusions_hour_scoped_and_date_scoped():
    cfg = _config()
    cfg["calc"]["manual_exclusions"] = [
        {"date": GENESIS, "source_id": "alpha", "reason": "bad tap", "hour": 0},
        {"date": GENESIS, "source_id": "bravo", "reason": "whole day"},
    ]
    validate_panel_config(cfg)
    state = _state()
    at_h0 = _compute(cfg, _golden_snapshot(), state)
    by_sid = {s["source_id"]: s for s in at_h0["sources"]}
    assert by_sid["alpha"]["status"] == "manually_excluded"
    assert by_sid["alpha"]["excluded_reason"] == "bad tap"
    assert by_sid["bravo"]["status"] == "manually_excluded"
    assert state["window_history"].get("alpha") is None
    at_h1 = _compute(cfg, _golden_snapshot(), state, stamp=STAMP0 + 1)
    by_sid = {s["source_id"]: s for s in at_h1["sources"]}
    assert by_sid["alpha"]["status"] == "ok"  # hour scope: one observation
    assert by_sid["bravo"]["status"] == "manually_excluded"  # date scope


def test_quarantine_holds_the_member_out_of_index_windows_and_weights():
    cfg = _config()
    state = _state()
    first = _compute(cfg, _golden_snapshot(), state)
    reference = jump_reference_prints(first)
    assert set(reference) == {"alpha", "bravo", "charlie"}
    jumped = _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 3.12, machine_id="a1")]),
            _entry("bravo", [_obs("H100", "H100", 2.5)]),
            _entry("charlie", [_obs("H100", "H100 SXM5 80GB", 2.6)]),
        ]
    )
    payload = _compute(
        cfg, jumped, state, stamp=STAMP0 + 1,
        ref=reference, ref_label="2026-08-23T00",
    )
    assert payload["jump_screen"]["quarantined"] == [
        {"source_id": "alpha", "book_pct": 30.0, "corroborators": 0}
    ]
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["alpha"]["status"] == "uncorroborated_jump"
    assert by_sid["alpha"]["chosen"]["usd_per_gpu_hr"] == 3.12
    assert "filter" not in by_sid["alpha"]
    # Held out of the index...
    assert set(payload["index"]["renormalized_weights"]) == {"bravo", "charlie"}
    # ... out of the filter window (preserved, print never entered) ...
    assert state["window_history"]["alpha"] == [2.4]
    # ... out of the weight series and the eligible vector.
    assert (STAMP0 + 1) not in state["weight_state"]["prices"]["alpha"]
    assert set(payload["weight_calc"]["weights"]) == {"bravo", "charlie"}
    # A quarantined print never serves as the next reference.
    assert "alpha" not in jump_reference_prints(payload)


def test_extra_require_member_prints_secure_not_community():
    """FIX 2 end to end: with extra_require {cloud: secure} on the seat,
    the cheaper community row never prints, the mismatch is counted on
    the artifact, and the screen rides calc_params (member-shaping
    bytes)."""
    cfg = _config()
    cfg["members"][0]["extra_require"] = {"cloud": "secure"}
    validate_panel_config(cfg)
    snapshot = _snapshot(
        [
            _entry(
                "alpha",
                [
                    _obs("H100", "H100 SXM", 3.29, extra={"cloud": "secure"}),
                    _obs("H100", "H100 SXM", 2.69, extra={"cloud": "community"}),
                ],
            ),
            _entry("bravo", [_obs("H100", "H100", 2.5)]),
        ]
    )
    payload = _compute(cfg, snapshot, _state())
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["alpha"]["status"] == "ok"
    assert by_sid["alpha"]["chosen"]["usd_per_gpu_hr"] == 3.29
    assert by_sid["alpha"]["chosen"]["n_eligible_prints"] == 1
    assert by_sid["alpha"]["extra_require_mismatches"] == 1
    assert "extra_require_mismatches" not in by_sid["bravo"]
    cp = payload["calc_params"]
    assert cp["members"][0]["source_id"] == "alpha"
    assert cp["members"][0]["extra_require"] == {"cloud": "secure"}
    assert "extra_require" not in cp["members"][1]
    json.dumps(payload)


def test_statistic_members_resolve_inside_an_observation():
    cfg = _config()
    vast_rows = [
        _obs(
            "H100", "H100 SXM", 2.0 + i * 0.1, basis=8,
            machine_id=f"m{i}", host_id="h1",
            verification="verified", region="Oregon, US",
        )
        for i in range(3)
    ]
    lium_rows = [_lium_row("A", 1.0, 8, "mA")]
    snapshot = _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 2.4)]),
            _entry("bravo", [_obs("H100", "H100", 2.5)]),
            _entry(
                "vast", vast_rows,
                book_stats={"H100 SXM": {"verified_us_ca_machines": 3}},
            ),
            _entry("lium", lium_rows),
        ]
    )
    payload = _compute(cfg, snapshot, _state())
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["vast"]["status"] == "held_out"
    assert by_sid["vast"]["held_out"]["reason"] == "thin_book"
    assert by_sid["vast"]["held_out"]["population_hosts"] == 1
    assert by_sid["lium"]["status"] == "held_out"
    assert by_sid["lium"]["held_out"]["reason"] == "thin_book"
    # Held-out statistic seats are not weight-eligible and enter no window.
    assert "vast" not in payload["weight_calc"]["weights"]
    assert set(payload["index"]["renormalized_weights"]) == {"alpha", "bravo"}


def test_currency_change_fence_confirms_at_the_third_print():
    """Coverage stage (recipe 1, engine half): a member repricing USD ->
    EUR is held out fail-closed with pending_count 1..2 (window and
    weights see NO mixed-currency splice into the old window), the THIRD
    consecutive EUR print confirms (accepted, unfiltered, window
    RESEEDED from the pending prints), and the mismatch prints still
    enter the weight series (the fence holds a print out of the INDEX,
    never the weight series)."""
    cfg = _config()
    state = _state()

    def _snap(bravo_row):
        return _snapshot(
            [
                _entry("alpha", [_obs("H100", "H100 SXM", 2.4)]),
                _entry("bravo", [bravo_row]),
                _entry("charlie", [_obs("H100", "H100 SXM5 80GB", 2.6)]),
            ]
        )

    eur = lambda: _obs("H100", "H100", usd=None, native=2.0, currency="EUR")
    payloads = [
        _compute(cfg, _snap(_obs("H100", "H100", 2.5)), state, fx=FX),
        _compute(cfg, _snap(eur()), state, stamp=STAMP0 + 1, fx=FX),
        _compute(cfg, _snap(eur()), state, stamp=STAMP0 + 2, fx=FX),
        _compute(cfg, _snap(eur()), state, stamp=STAMP0 + 3, fx=FX),
    ]
    verdicts = [
        {s["source_id"]: s for s in p["sources"]}["bravo"].get("filter")
        for p in payloads
    ]
    assert verdicts[0] == {
        "accepted": True,
        "unfiltered": True,
        "n_history": 0,
        "currency": "USD",
    }
    assert verdicts[1] == {
        "accepted": False,
        "unfiltered": False,
        "currency_mismatch": True,
        "currency": "EUR",
        "window_currency": "USD",
        "filter_price": 2.0,
        "pending_count": 1,
        "confirm_after": CURRENCY_CONFIRM_DAYS,
        "n_history": 1,
    }
    assert verdicts[2]["pending_count"] == 2
    assert verdicts[2]["accepted"] is False
    assert verdicts[3] == {
        "accepted": True,
        "unfiltered": True,
        "currency_confirmed": True,
        "currency": "EUR",
        "window_currency": "USD",
        "filter_price": 2.0,
        "n_history": CURRENCY_CONFIRM_DAYS,
    }
    # Held out of the index while pending; back in at confirmation.
    assert set(payloads[1]["index"]["renormalized_weights"]) == {
        "alpha", "charlie",
    }
    assert set(payloads[3]["index"]["renormalized_weights"]) == {
        "alpha", "bravo", "charlie",
    }
    # Window RESEEDED from the pending prints (warm-up restarts); the
    # pending streak is consumed.
    assert state["window_history"]["bravo"] == [2.0, 2.0, 2.0]
    assert state["window_currencies"]["bravo"] == "EUR"
    assert state["pending_currencies"] == {}
    # Every real trusted print entered the weight series -- the USD one
    # AND the three EUR ones (mismatch verdicts included).
    bravo_prices = state["weight_state"]["prices"]["bravo"]
    assert bravo_prices[STAMP0] == {"usd": 2.5, "native": 2.5, "currency": "USD"}
    for offset in (1, 2, 3):
        assert bravo_prices[STAMP0 + offset] == {
            "usd": 2.3,  # EUR 2.0 * 1.15
            "native": 2.0,
            "currency": "EUR",
        }


def test_sigma_fence_holds_the_outlier_out_of_the_index_not_the_weights():
    """Coverage stage (recipe 2, engine half): after 20 stable prints a
    >3-sigma move that the jump screen does NOT quarantine (+8.3%, far
    under the 25% jump fence) is filter-rejected -- accepted false, the
    member absent from the index and its renormalized weights -- while
    the outlier still enters BOTH the filter window (the outlier-fence rule: the
    filter adapts) and the weight-state prices at this stamp (the fence
    holds a print out of the INDEX, never the weight series)."""
    cfg = _config()
    state = _state()
    state["window_history"].update(
        {"alpha": [2.4] * 20, "bravo": [2.5] * 20, "charlie": [2.6] * 20}
    )
    state["window_currencies"].update(
        {"alpha": "USD", "bravo": "USD", "charlie": "USD"}
    )
    reference = {
        sid: {"usd_per_gpu_hr": usd, "currency": "USD", "native_per_gpu_hr": usd}
        for sid, usd in (("alpha", 2.4), ("bravo", 2.5), ("charlie", 2.6))
    }
    snapshot = _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 2.6)]),  # the outlier
            _entry("bravo", [_obs("H100", "H100", 2.5)]),
            _entry("charlie", [_obs("H100", "H100 SXM5 80GB", 2.6)]),
        ]
    )
    payload = _compute(
        cfg, snapshot, state, ref=reference, ref_label="2026-08-22T23"
    )
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    alpha = by_sid["alpha"]
    # The jump screen saw the move and stood by: +8.33% < 25%.
    deltas = {d["source_id"]: d for d in payload["jump_screen"]["deltas"]}
    assert deltas["alpha"]["book_pct"] == 8.33
    assert payload["jump_screen"]["quarantined"] == []
    # Sigma fence: window all-2.4 (sigma 0, floored 0.05), band 0.15,
    # deviation 0.2 -> rejected.
    assert alpha["status"] == "ok"
    assert alpha["filter"]["accepted"] is False
    assert alpha["filter"]["unfiltered"] is False
    assert alpha["filter"]["n_history"] == 20
    # Out of the index and its renormalized weights...
    assert set(payload["index"]["renormalized_weights"]) == {"bravo", "charlie"}
    # ... but PRESENT in the filter window and the weight-state prices.
    assert state["window_history"]["alpha"] == [2.4] * 20 + [2.6]
    assert state["weight_state"]["prices"]["alpha"][STAMP0] == {
        "usd": 2.6, "native": 2.6, "currency": "USD",
    }


def test_r3_manual_verify_flags_the_far_from_mean_warmup_print():
    """Coverage stage (recipe 4): on a warm-up observation (unfiltered
    verdicts) a member far from the panel mean gains manual_verify true;
    near-mean members carry no flag at all."""
    cfg = _config()
    snapshot = _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 7.0)]),
            _entry("bravo", [_obs("H100", "H100", 7.2)]),
            _entry("charlie", [_obs("H100", "H100 SXM5 80GB", 10.0)]),
        ]
    )
    payload = _compute(cfg, snapshot, _state())
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    # Panel mean 8.066667: charlie deviates ~24% (> the 15% default),
    # alpha ~13.2% and bravo ~10.7% do not.
    assert by_sid["charlie"]["filter"]["manual_verify"] is True
    assert "manual_verify" not in by_sid["alpha"]["filter"]
    assert "manual_verify" not in by_sid["bravo"]["filter"]


def test_compute_observation_state_guards_raise():
    """Coverage stage (recipe 7): recorded-currency filtering without a
    window_currencies dict, or any panel observation without a
    weight_state, cannot replay deterministically -- both refuse."""
    cfg = _config()
    with pytest.raises(ValueError, match="window_currencies"):
        compute_observation(
            config=cfg,
            obs_stamp=STAMP0,
            snapshot=_golden_snapshot(),
            fx_records={},
            window_history={},
            weight_state=new_weight_state(),
        )
    with pytest.raises(ValueError, match="weight_state"):
        compute_observation(
            config=cfg,
            obs_stamp=STAMP0,
            snapshot=_golden_snapshot(),
            fx_records={},
            window_history={},
            window_currencies={},
            pending_currencies={},
        )


def test_unscheduled_stamp_refuses():
    cfg = _config()
    with pytest.raises(ValueError, match="not a scheduled observation"):
        _compute(cfg, _golden_snapshot(), _state(),
                 stamp=date_hour_to_stamp("2026-08-22", 5))
    coarse = _config()
    coarse["slot_grids"] = [
        {"from_date": GENESIS, "slot_hours_utc": [4, 10, 16, 22]}
    ]
    coarse["calc"]["dynamic_weights"]["lookback_horizons_hours"] = [6, 12]
    coarse["calc"]["dynamic_weights"]["forward_horizons_hours"] = [6, 12]
    validate_panel_config(coarse)
    with pytest.raises(ValueError, match="not a scheduled observation"):
        _compute(coarse, _golden_snapshot(), _state(),
                 stamp=date_hour_to_stamp(GENESIS, 5))


def test_record_source_for_cutover():
    sources = [
        {
            "kind": "basket",
            "prefix": "index/b300_basket",
            "from_date": "2026-08-10",
            "to_date": "2026-08-23",
        },
        {
            "kind": "observatory",
            "prefix": "index/raw_observatory",
            "from_date": "2026-08-24",
        },
    ]
    assert record_source_for(sources, "2026-08-10")["kind"] == "basket"
    assert record_source_for(sources, "2026-08-23")["kind"] == "basket"
    assert record_source_for(sources, "2026-08-24")["kind"] == "observatory"
    assert record_source_for(sources, "2027-01-01")["kind"] == "observatory"
    with pytest.raises(ValueError, match="no record source"):
        record_source_for(sources, "2026-08-09")


def test_compile_screens_require_none_vs_declared():
    params = panel_calc_params(_config())
    screens = compile_screens(params)
    assert screens["members"]["alpha"]["require"] is not None
    assert screens["members"]["bravo"]["require"] is None
    assert screens["members"]["bravo"]["declared"] is True
    assert screens["members"]["vast"]["statistic"] == "vast_vwm_verified_us_ca_floor"


def test_precomputed_calc_params_path_is_identical_and_fenced():
    """Hot-loop invariant (review perf stage): compute_observation with
    caller-precomputed calc_params + compiled screens prices EXACTLY what
    the derive-per-observation path prices, and a params set from another
    lane's law refuses loudly."""
    cfg = _config()
    params = panel_calc_params(cfg)
    screens = compile_screens(params)
    derived = _compute(cfg, _golden_snapshot(), _state())
    state = _state()
    amortized = compute_observation(
        config=cfg,
        obs_stamp=STAMP0,
        snapshot=_golden_snapshot(),
        fx_records={},
        window_history=state["window_history"],
        window_currencies=state["window_currencies"],
        pending_currencies=state["pending_currencies"],
        weight_state=state["weight_state"],
        calc_params=params,
        compiled_screens=screens,
    )
    assert json.dumps(amortized, sort_keys=True) == json.dumps(
        derived, sort_keys=True
    )
    # Foreign params refuse: the amortized path can never run another law.
    foreign = panel_calc_params(_config())
    foreign["methodology_id"] = "someone_elses_calc_v9"
    with pytest.raises(ValueError, match="methodology_id"):
        compute_observation(
            config=cfg,
            obs_stamp=STAMP0,
            snapshot=_golden_snapshot(),
            fx_records={},
            window_history={},
            weight_state=new_weight_state(),
            window_currencies={},
            pending_currencies={},
            calc_params=foreign,
        )
    # Screens without their params refuse: compiled from one resolution,
    # they must never run under another.
    with pytest.raises(ValueError, match="compiled_screens requires"):
        compute_observation(
            config=cfg,
            obs_stamp=STAMP0,
            snapshot=_golden_snapshot(),
            fx_records={},
            window_history={},
            weight_state=new_weight_state(),
            window_currencies={},
            pending_currencies={},
            compiled_screens=screens,
        )


# ------------------------------------------- availability disclosure


def test_availability_verified_sources_validation():
    cfg = _config()
    cfg["calc"]["availability_verified_sources"] = ["vast", "lium"]
    validate_panel_config(cfg)  # members -> accepted
    _reject("non-member id", lambda c: c["calc"].update(
        availability_verified_sources=["vast", "shadeform"]
    ))
    _reject("duplicate entries", lambda c: c["calc"].update(
        availability_verified_sources=["vast", "vast"]
    ))
    _reject("non-empty source_id strings", lambda c: c["calc"].update(
        availability_verified_sources=["vast", ""]
    ))
    _reject("non-empty source_id strings", lambda c: c["calc"].update(
        availability_verified_sources="vast"
    ))


def test_availability_verified_share_is_a_disclosure_aggregate():
    # bravo is verified and passes -> the share is EXACTLY bravo's
    # unrounded renormalized weight (0.2 / 0.7); vast is verified but has
    # no print this hour -> contributes nothing. The list is a CALC key
    # and MUST ride calc_params (sorted, canonical bytes) so a retune is
    # a versioned change and published-stamp recompute stays
    # byte-deterministic (adversarial review).
    cfg = _config()
    cfg["calc"]["availability_verified_sources"] = ["vast", "bravo"]
    payload = _compute(cfg, _golden_snapshot(), _state())
    assert payload["index"]["availability_verified_weight_share"] == 0.285714
    assert payload["calc_params"]["availability_verified_sources"] == [
        "bravo",
        "vast",
    ]


def test_availability_verified_share_sums_multiple_passers():
    # Two verified passing members: the share is the SUM of their
    # unrounded renormalized weights ((0.3 + 0.2) / 0.7) -- the shipped
    # prod shape on every H panel (vast + lium both passing). A
    # sum->max/single-pick mutant fails here (adversarial review).
    cfg = _config()
    cfg["calc"]["availability_verified_sources"] = ["alpha", "bravo"]
    payload = _compute(cfg, _golden_snapshot(), _state())
    assert payload["index"]["availability_verified_weight_share"] == 0.714286


def test_availability_verified_share_excludes_held_out_members():
    # bravo PRINTS but is HELD OUT (EUR print, no FX rate supplied): a
    # held-out verified member priced nothing and must contribute
    # nothing -- counting printed-but-not-passing members would overstate
    # the disclosure in exactly the hours the fences fired (adversarial
    # review mutant).
    cfg = _config()
    cfg["calc"]["availability_verified_sources"] = ["bravo"]
    snapshot = _snapshot(
        [
            _entry("alpha", [_obs("H100", "H100 SXM", 2.4, machine_id="a1")]),
            _entry(
                "bravo",
                [_obs("H100", "H100", usd=None, native=2.3, currency="EUR")],
            ),
            _entry("charlie", [_obs("H100", "H100 SXM5 80GB", 2.6)]),
        ]
    )
    payload = _compute(cfg, snapshot, _state())
    by_sid = {s["source_id"]: s for s in payload["sources"]}
    assert by_sid["bravo"]["status"] != "ok"
    assert payload["index"]["availability_verified_weight_share"] == 0.0


def test_availability_verified_share_all_passers_is_exactly_one():
    # Every passer verified -> exactly 1.0, never 1.000002: the share
    # derives from the UNROUNDED weights, not the published rounded
    # renormalized_weights (adversarial review: six 0.166667s sum past
    # 1.0).
    cfg = _config()
    cfg["calc"]["availability_verified_sources"] = [
        "alpha",
        "bravo",
        "charlie",
    ]
    payload = _compute(cfg, _golden_snapshot(), _state())
    assert payload["index"]["availability_verified_weight_share"] == 1.0


def test_availability_verified_share_zero_is_float_and_dark_absent():
    # No configured list -> an honest FLOAT 0.0 on the wire (a bare JSON
    # integer 0 would flip the field's wire type across observations).
    payload = _compute(_config(), _golden_snapshot(), _state())
    share = payload["index"]["availability_verified_weight_share"]
    assert share == 0.0
    assert isinstance(share, float)
    assert '"availability_verified_weight_share": 0.0' in json.dumps(
        payload["index"], indent=1
    )
    # Missed hour -> no index -> no share anywhere in the artifact.
    missed = _compute(_config(), None, _state())
    assert missed["index"] is None
