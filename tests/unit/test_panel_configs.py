# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""The six SHIPPED panel configs (stage 5 of
METHODOLOGY.md).

Engine behavior is pinned by test_panel_engine/test_panel_weights/
test_panel_cli over synthetic configs; THIS file pins the real lane
files in config/ -- the baked lane law the scheduled panel-index
job actually runs. Every assert here is a design-table fact (section 1)
or a migration invariant:

  - all six load through load_panel_config (so the full validator gate
    -- weights exact at 6dp summing to 1.0, contiguous record coverage,
    era grids, variant-rule shapes, dw feasibility -- has already run);
  - lane identity: methodology_id, prefix, genesis, claim floor, w_min,
    fx lane, and record stitching per the section 1 table;
  - prefixes never collide (pairwise distinct; nesting is refused by the
    loader itself);
  - the migrated B300/B200 lanes hold out the daily configs'
    manual_exclusions (date, source) pairs exactly, with every reason
    set to the ONE public-safe neutral string (the panel lanes are
    public-bound: reasons ride panel_calc_params into every published
    artifact's bytes);
  - membership matches design section 8 (counts, weights, statistics,
    and the fail-closed variant-rule posture: every SXM seat carries a
    variant rule; broad panels carry none and no reject_tokens);
  - no panel config carries a fallback_parity key (no parity with the
    frozen daily series is possible or claimed -- rule A1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.index.panel_config import load_panel_config

REPO_ROOT = Path(__file__).resolve().parents[2]

# The one public-safe manual-exclusion reason on public-bound panel
# lanes -- byte-exact (reasons ride panel_calc_params into every
# published artifact's bytes, so this exact string is the record).
NEUTRAL_EXCLUSION_REASON = (
    "capture defect: recorded print known wrong by rule; "
    "true print not captured"
)

# The section 1 lane table, verbatim.
LANES = {
    "config/index_panel_b300.json": {
        "panel_id": "b300",
        "methodology_id": "annex_a_v0_2_calc_v11",
        "prefix": "index/b300_basket",
        "genesis": "2026-08-10",
        "claim_floor": 5,
        "w_min": 0.025,
        "fx_lane": "ecb",
        "members": 8,
        "stitched": True,
        "quarter_hour": True,
    },
    "config/index_panel_b200.json": {
        "panel_id": "b200",
        "methodology_id": "annex_a2_v0_3_calc_v10",
        "prefix": "index/b200_basket",
        "genesis": "2026-08-16",
        "claim_floor": 5,
        "w_min": 0.025,
        "fx_lane": "none",
        "members": 9,
        "stitched": True,
        "quarter_hour": True,
    },
    "config/index_panel_h100_sxm.json": {
        "panel_id": "h100_sxm",
        "methodology_id": "h100_sxm_v1_calc_v5",
        "prefix": "index/h100_sxm",
        "genesis": "2026-08-23",
        "claim_floor": 5,
        "w_min": 0.025,
        "fx_lane": "ecb",  # scaleway bills EUR
        "members": 16,
        "stitched": False,
        "quarter_hour": True,
    },
    "config/index_panel_h200_sxm.json": {
        "panel_id": "h200_sxm",
        "methodology_id": "h200_sxm_v1_calc_v5",
        "prefix": "index/h200_sxm",
        "genesis": "2026-08-23",
        "claim_floor": 5,
        "w_min": 0.025,
        "fx_lane": "none",  # no EUR member seated
        "members": 13,
        "stitched": False,
        "quarter_hour": True,
    },
    "config/index_panel_h100_broad.json": {
        "panel_id": "h100_broad",
        "methodology_id": "h100_broad_v1_calc_v1",
        "prefix": "index/h100_broad",
        "genesis": "2026-08-23",
        "claim_floor": 8,
        "w_min": 0.0125,
        "fx_lane": "ecb",  # scaleway + ovh (FR subsidiary) bill EUR
        "members": 22,
        "stitched": False,
        "quarter_hour": False,
    },
    "config/index_panel_h200_broad.json": {
        "panel_id": "h200_broad",
        "methodology_id": "h200_broad_v1_calc_v1",
        "prefix": "index/h200_broad",
        "genesis": "2026-08-23",
        "claim_floor": 7,
        "w_min": 0.0125,
        "fx_lane": "ecb",  # ovh (FR subsidiary) bills EUR
        "members": 18,
        "stitched": False,
        "quarter_hour": False,
    },
}

OBS_PREFIX = "index/raw_observatory"
CUTOVER = "2026-08-24"
QUARTER_HOUR_CUTOVER = "2026-08-29"


@pytest.fixture(scope="module")
def configs():
    return {
        rel: load_panel_config(REPO_ROOT / rel) for rel in sorted(LANES)
    }


def test_all_six_load_and_match_the_lane_table(configs):
    for rel, expected in LANES.items():
        cfg = configs[rel]
        calc = cfg["calc"]
        assert cfg["panel_id"] == expected["panel_id"], rel
        assert calc["methodology_id"] == expected["methodology_id"], rel
        assert cfg["bucket_prefix"] == expected["prefix"], rel
        assert cfg["genesis_date"] == expected["genesis"], rel
        assert calc["min_sources_to_claim"] == expected["claim_floor"], rel
        assert (
            calc["dynamic_weights"]["weight_min"] == expected["w_min"]
        ), rel
        assert calc["fx_lane"] == expected["fx_lane"], rel
        assert len(cfg["members"]) == expected["members"], rel
        # The loader enforced the 6dp-exact sum-to-1.0 rule; re-assert the
        # arithmetic here so the invariant reads in THIS file too.
        assert (
            abs(sum(m["weight"] for m in cfg["members"]) - 1.0) <= 1e-9
        ), rel
        # A1: no parity leg exists on any panel lane.
        assert "fallback_parity_methodology_id" not in cfg, rel
        assert "fallback_parity_methodology_id" not in calc, rel
        # Tier screen (recorded in every calc.description): eligibility
        # is the ALLOW-LIST eligible_tiers, on-demand only (methodology
        # section 5) -- the old committed-tier limitation is
        # RECONCILED at these hourly mints (reserved/committed/serverless/
        # from_floor rows are ineligible by construction; the frozen
        # daily series kept committed-eligible). The retired exclusion
        # list must be gone: the loader refuses the key outright.
        assert calc["eligible_tiers"] == ["on-demand"], rel
        assert "interruptible_tiers" not in calc, rel
        # The drift-scan bound is a TOP-LEVEL operational key (amended
        # into the mints before any observation published) -- it gates an
        # ops sweep and must never ride calc_params/artifact bytes; the
        # loader refuses the retired calc location outright.
        assert cfg["drift_scan_observations"] == 48, rel
        assert "drift_scan_observations" not in calc, rel
        # Availability disclosure (marketplaces-first ruling 2026-08-24):
        # the verified-source list, pinned PER LANE so a config refactor
        # or merge-conflict resolution can never silently drop a
        # declared-verified seat (the share would quietly publish 0%
        # forever -- the validator accepts absence by design). vast on
        # all six lanes; lium on the four H panels only (not seated on
        # B300/B200). Sorted: calc_params embeds the list canonically.
        expected_verified = (
            ["vast"]
            if cfg["panel_id"] in ("b300", "b200")
            else ["lium", "vast"]
        )
        assert calc["availability_verified_sources"] == expected_verified, rel


def test_runpod_seats_are_secure_cloud_only(configs):
    """Annex A-2 2.1 excludes Community Cloud, and the observatory
    runpod collector records Secure AND Community rows under ONE
    sku_identifier distinguished only by extra.cloud -- every runpod
    seat must therefore carry the extra_require screen, or the
    lowest-eligible rule prices the cheaper community row. No other
    seat carries an extra_require today (a new one is a reviewed
    membership decision, not drift)."""
    for rel, cfg in configs.items():
        members = {m["source_id"]: m for m in cfg["members"]}
        assert members["runpod"]["extra_require"] == {"cloud": "secure"}, rel
        for sid, member in members.items():
            if sid != "runpod":
                assert "extra_require" not in member, (rel, sid)


def test_prefixes_are_pairwise_disjoint(configs):
    prefixes = [cfg["bucket_prefix"] for cfg in configs.values()]
    assert len(set(prefixes)) == len(prefixes)
    # Nesting is refused by the loader; equality across LANES entries
    # would collide composite keyspaces only if methodology ids also
    # collided -- assert those distinct too, one lane one series.
    mids = [cfg["calc"]["methodology_id"] for cfg in configs.values()]
    assert len(set(mids)) == len(mids)


def test_record_stitching_matches_the_design_table(configs):
    for rel, expected in LANES.items():
        cfg = configs[rel]
        sources = cfg["record_sources"]
        if expected["stitched"]:
            # Basket record through 2026-08-23, observatory from 08-24,
            # era grids cutting over on the same boundary.
            assert [s["kind"] for s in sources] == ["basket", "observatory"], rel
            assert sources[0]["prefix"] == expected["prefix"], rel
            assert sources[0]["from_date"] == expected["genesis"], rel
            assert sources[0]["to_date"] == "2026-08-23", rel
            assert sources[1]["prefix"] == OBS_PREFIX, rel
            assert sources[1]["from_date"] == CUTOVER, rel
            assert "to_date" not in sources[1], rel
            grids = cfg["slot_grids"]
            assert grids[0]["slot_hours_utc"] == [4, 10, 16, 22], rel
            assert grids[1]["from_date"] == CUTOVER, rel
            assert grids[1]["slot_hours_utc"] == list(range(24)), rel
        else:
            assert [s["kind"] for s in sources] == ["observatory"], rel
            assert sources[0]["prefix"] == OBS_PREFIX, rel
            assert sources[0]["from_date"] == expected["genesis"], rel
            assert "to_date" not in sources[0], rel
            grids = cfg["slot_grids"]
            assert grids[0]["slot_hours_utc"] == list(range(24)), rel
        if expected["quarter_hour"]:
            assert grids[-1]["from_date"] == QUARTER_HOUR_CUTOVER, rel
            assert grids[-1]["slot_minutes_utc"] == list(
                range(0, 1440, 15)
            ), rel
        else:
            assert len(grids) == 1, rel


def test_public_lane_configs_match_the_live_era3_calculation(configs):
    for rel in (
        "config/index_panel_b300.json",
        "config/index_panel_b200.json",
        "config/index_panel_h100_sxm.json",
        "config/index_panel_h200_sxm.json",
    ):
        calc = configs[rel]["calc"]
        assert calc["filter_sigma_floor_pct"] == 3.0, rel
        assert "filter_sigma_floor" not in calc, rel
        assert calc["iqm_alpha"] == 0.16666, rel
        assert calc["vote_sigma_source"] == "dw_history", rel
        assert calc["vote_sigma_floor_pct"] == 3.0, rel


def test_migrated_lanes_carry_daily_manual_exclusion_pairs_neutral_reasons(
    configs,
):
    """The daily lanes' manual_exclusions are pinned facts of published
    history: the hourly mints must hold out the same (date, source)
    pairs. The REASON strings ride panel_calc_params into every
    published artifact's bytes on a PUBLIC-BOUND lane, so each panel
    reason is the one neutral public-safe string, byte-exact --
    internal audit narrative (ticket ids, host ids, recorded prices)
    must never appear here."""
    for panel_rel, daily_rel in (
        ("config/index_panel_b300.json", "config/index_basket.json"),
        ("config/index_panel_b200.json", "config/index_basket_b200.json"),
    ):
        daily = json.loads((REPO_ROOT / daily_rel).read_text())
        panel = configs[panel_rel]
        panel_excl = panel["calc"]["manual_exclusions"]
        daily_excl = daily["calc"]["manual_exclusions"]
        assert [
            (e["date"], e["source_id"]) for e in panel_excl
        ] == [
            (e["date"], e["source_id"]) for e in daily_excl
        ], panel_rel
        assert [e["reason"] for e in panel_excl] == [
            NEUTRAL_EXCLUSION_REASON
        ] * len(panel_excl), panel_rel


def test_migrated_lane_statistics_per_rulings(configs):
    b300 = {m["source_id"]: m for m in configs["config/index_panel_b300.json"]["members"]}
    b200 = {m["source_id"]: m for m in configs["config/index_panel_b200.json"]["members"]}
    # Ruling A3: B300 vast stays lowest-eligible -- NO statistic named.
    assert "statistic" not in b300["vast"]
    # B200 vast prices via the population-gated v2 id (a NEW id: the
    # frozen daily vast_vwm_verified_us_ca is a different registry with a
    # different signature and must never be named by a panel).
    assert b200["vast"]["statistic"] == "vast_vwm_verified_us_ca_v2"
    for members in (b300, b200):
        for sid, member in members.items():
            assert "variant" not in member, sid  # single-sku lanes


def test_sxm_panels_fail_closed_variant_posture(configs):
    """Design section 8: EVERY seat on a form-factor panel carries a
    variant rule (require_tokens or declared-with-evidence) -- a seat
    without one would print unscreened generic rows. Statistic seats are
    exactly the two marketplaces."""
    for rel in (
        "config/index_panel_h100_sxm.json",
        "config/index_panel_h200_sxm.json",
    ):
        cfg = configs[rel]
        assert cfg.get("reject_tokens"), rel
        stats = {}
        for member in cfg["members"]:
            variant = member.get("variant")
            assert variant is not None, (rel, member["source_id"])
            if variant["mode"] == "declared":
                assert variant["evidence"].strip(), (rel, member["source_id"])
            if member.get("statistic"):
                stats[member["source_id"]] = member["statistic"]
        assert stats == {
            "vast": "vast_vwm_verified_us_ca_floor",
            "lium": "lium_vwm_book_floor",
        }, rel


def test_broad_panels_have_no_identity_screen(configs):
    """Composition is the instrument (design section 3): no panel-level
    reject tokens, no per-seat variant rules; the sku SET is the screen
    (H100T deliberately absent -- the Oracle seat stays inert pending the
    recorded rule)."""
    for rel, sku_set in (
        ("config/index_panel_h100_broad.json", {"H100", "H100_PCIE", "H100_NVL"}),
        ("config/index_panel_h200_broad.json", {"H200", "H200_NVL"}),
    ):
        cfg = configs[rel]
        assert "reject_tokens" not in cfg, rel
        for member in cfg["members"]:
            assert "variant" not in member, (rel, member["source_id"])
            assert set(member["skus"]) == sku_set, (rel, member["source_id"])
        assert "H100T" not in sku_set
