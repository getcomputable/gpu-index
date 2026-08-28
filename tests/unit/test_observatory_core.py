# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for the raw price observatory core (capture only).

Pins: (1) catalog normalization (boundary matching, ordering, unmapped
honesty), (2) config validation incl. the reserved-basket-prefix fence,
(3) snapshot normalization + rollups, (4) the budgeted collection loop,
(5) the append-only store discipline, and (6) that the snapshot document
derives NOTHING cross-source — the record-raw ruling this lane inherits.
"""

from __future__ import annotations

import importlib.util
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_index.observatory.catalog import (
    SkuCatalogError,
    load_sku_catalog,
    match_sku,
    normalize_label,
    plausible_band,
)
from gpu_index.observatory.collect import collect_all
from gpu_index.observatory.config import (
    ObservatoryConfigError,
    load_observatory_config,
)
from gpu_index.observatory.observation import observation, result
from gpu_index.observatory.snapshot import (
    SNAPSHOT_KIND,
    build_capture_snapshot,
    normalize_observation,
)
from gpu_index.observatory.sources import COLLECTORS
from gpu_index.observatory.store import upload_capture_snapshot
from gpu_index.common.store import BucketPublishError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("RAW_OBSERVATORY_CONFIG_PATH", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def catalog():
    return load_sku_catalog(REPO_ROOT / "config" / "gpu_sku_catalog.json")


@pytest.fixture(scope="module")
def real_config():
    return load_observatory_config(REPO_ROOT / "config" / "raw_observatory.json")


# ------------------------------------------------------------------ catalog


def test_real_catalog_loads(catalog):
    assert len(catalog["entries"]) >= 30
    skus = [e["sku"] for e in catalog["entries"]]
    assert len(skus) == len(set(skus))


@pytest.mark.parametrize(
    "label,expected",
    [
        # Boundary honesty: the generic token never matches inside the
        # superchip label — GB-class parts are first-class skus here, not
        # quarantine fodder.
        ("NVIDIA HGX B200", "B200"),
        ("NVIDIA GB200 NVL72", "GB200"),
        ("GB300 SXM6 288GB", "GB300"),
        ("B300 SXM6-x 8-0", "B300"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA H20", "H20"),
        ("H100 SXM5 80GB", "H100"),
        ("GH200 96GB", "GH200"),
        # A10 vs A100, L4 vs L40 vs L40S — boundary + ordering.
        ("NVIDIA A100-SXM4-80GB", "A100"),
        ("NVIDIA A10", "A10"),
        ("A10G", "A10"),
        ("NVIDIA L4", "L4"),
        ("NVIDIA L40", "L40"),
        ("NVIDIA L40S", "L40S"),
        # RTX family ordering: ADA/PRO variants above the bare token.
        ("RTX 6000 Ada Generation", "RTX_6000_ADA"),
        ("NVIDIA RTX PRO 6000 Blackwell", "RTX_PRO_6000"),
        ("Quadro RTX 6000", "RTX_6000_QUADRO"),
        # Bare 'RTX 6000' with no generation marker is deliberately its own
        # ambiguous bucket (DO's Ada-less label vs marketplace Quadros).
        ("NVIDIA RTX 6000", "RTX_6000"),
        ("RTX A6000", "RTX_A6000"),
        # Separator + compact variants.
        ("RTX-4090", "RTX_4090"),
        ("RTX4090", "RTX_4090"),
        ("GeForce RTX 4090", "RTX_4090"),
        ("AMD Instinct MI300X OAM", "MI300X"),
        ("MI325X", "MI325X"),
        ("Intel Gaudi 3", "GAUDI3"),
    ],
)
def test_catalog_normalization(catalog, label, expected):
    entry = match_sku(catalog, label)
    assert entry is not None, f"{label!r} should map"
    assert entry["sku"] == expected


@pytest.mark.parametrize(
    "label",
    [
        "",
        None,
        "CPU Optimized c2-standard-8",
        "TPU v5e",
        "Sapphire Rapids 8462Y+",
    ],
)
def test_catalog_unmapped_is_none(catalog, label):
    assert match_sku(catalog, label) is None


def test_normalize_label_collapses_separators():
    assert normalize_label(" rtx-4090 / 24GB ") == "RTX 4090 24 GB"


@pytest.mark.parametrize(
    "label,expected",
    [
        # Partial compactions must land on the SPECIFIC entry, not fall
        # through to a generic or wrong-generation bucket (digit-boundary
        # spacing normalizes label and token alike).
        ("RTX6000 Ada", "RTX_6000_ADA"),
        ("RTX5000 Ada", "RTX_5000_ADA"),
        ("Quadro RTX6000", "RTX_6000_QUADRO"),
        ("RTX4080 Super", "RTX_4080_SUPER"),
        ("RTX3090 Ti", "RTX_3090_TI"),
        # Suffix-letter variants: the variant entry sits ABOVE its parent,
        # so first-match-wins keeps them apart now that spacing exposes the
        # parent token inside the variant label.
        ("OCI - Compute - GPU - H100T", "H100T"),
        ("Tesla V100S", "V100S"),
    ],
)
def test_catalog_partial_compaction_and_variants(catalog, label, expected):
    entry = match_sku(catalog, label)
    assert entry is not None and entry["sku"] == expected, (
        f"{label!r} -> {entry and entry['sku']}"
    )


@pytest.mark.parametrize(
    "label,expected",
    [
        # H-series variant separation (hourly panel design section 7 /
        # METHODOLOGY.md section 5.1): NVL and PCIe form factors
        # trade at structurally different prices, so their entries sit
        # ABOVE the generic parts and first-match-wins peels them off.
        ("H200 NVL", "H200_NVL"),
        ("NVIDIA H200 NVL", "H200_NVL"),
        ("H200 NVL (141GB)", "H200_NVL"),
        ("H100 NVL", "H100_NVL"),
        ("H100 NVL (94GB) NVLink", "H100_NVL"),
        ("H100 PCIe", "H100_PCIE"),
        ("H100-PCIe-80GB", "H100_PCIE"),
        ("h100_pcie", "H100_PCIE"),
        ("H100 PCI-E", "H100_PCIE"),
        # The generic entries no longer swallow the variants, but still
        # catch everything else: SXM labels and bare parts stay put.
        ("H100 SXM", "H100"),
        ("H100 SXM5 80GB", "H100"),
        ("NVIDIA H100", "H100"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA H200 SXM", "H200"),
        # Boundary honesty: 'NVL' never fires inside a longer run, so
        # interconnect-marketing 'NVLink' labels are NOT the NVL part...
        ("NVIDIA H100 NVLink", "H100"),
        # ...and the GB-superchip entries above still win their labels.
        ("NVIDIA GB200 NVL72", "GB200"),
    ],
)
def test_hseries_variant_entries_win_above_generic(catalog, label, expected):
    entry = match_sku(catalog, label)
    assert entry is not None and entry["sku"] == expected, (
        f"{label!r} -> {entry and entry['sku']}"
    )


def test_hseries_variant_entries_are_hopper_nvidia(catalog):
    """The three new variant entries carry the H100/H200-band vendor,
    family and plausibility (design section 7)."""
    by_sku = {e["sku"]: e for e in catalog["entries"]}
    for sku, band in (
        ("H200_NVL", (0.3, 30.0)),
        ("H100_NVL", (0.2, 25.0)),
        ("H100_PCIE", (0.2, 25.0)),
    ):
        entry = by_sku[sku]
        assert entry["vendor"] == "NVIDIA"
        assert entry["family"] == "hopper"
        assert entry["plausible_usd_gpu_hr"] == band


def test_plausible_band_falls_back_to_default(catalog):
    assert plausible_band(catalog, None) == catalog["default_plausible_usd_gpu_hr"]
    assert plausible_band(catalog, "NOT_A_SKU") == catalog[
        "default_plausible_usd_gpu_hr"
    ]
    b300 = plausible_band(catalog, "B300")
    assert b300 != catalog["default_plausible_usd_gpu_hr"]


def test_catalog_rejects_duplicate_sku(tmp_path):
    bad = {
        "skus": [
            {"sku": "X", "match_tokens": ["X1"]},
            {"sku": "X", "match_tokens": ["X2"]},
        ]
    }
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SkuCatalogError, match="duplicate"):
        load_sku_catalog(p)


def test_catalog_rejects_bad_band(tmp_path):
    bad = {"skus": [{"sku": "X", "match_tokens": ["X"], "plausible_usd_gpu_hr": [5, 1]}]}
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SkuCatalogError, match="lo < hi"):
        load_sku_catalog(p)


# ------------------------------------------------------------------- config


def test_real_config_loads_and_validates(real_config):
    assert real_config["lane_id"] == "raw_observatory_v1"
    assert real_config["bucket_prefix"] == "index/raw_observatory"
    assert real_config["canonical_slot_utc"] in real_config["capture_slots_utc"]
    # Hourly cadence: all 24 marks, canonical 16 — changing the marks (or
    # moving to the minute vocabulary, which the loader now also accepts)
    # is a deliberate edit, and this pin makes it loud.
    assert real_config["capture_slots_utc"] == list(range(24))
    sids = [s["source_id"] for s in real_config["sources"]]
    assert len(sids) == len(set(sids))
    # Aggregator/reseller rows must be labeled as not-first-party — the
    # who-is-speaking disclosure the lane exists to preserve.
    for sid in ("shadeform", "computepulse"):
        src = next(s for s in real_config["sources"] if s["source_id"] == sid)
        assert src["first_party"] is False


def test_reserved_prefixes_cover_the_real_basket_lanes(real_config):
    """The keyspace fence's promise: RESERVED_LANE_PREFIXES must name the
    ACTUAL basket lanes' AND panel lanes' bucket prefixes — a new lane
    landing with a new prefix must grow the tuple (this is the reminder).
    The panel side pins full set-equality with its KNOWN_LANE_PREFIXES
    (test_panel_engine), so neither constant can drift alone."""
    from gpu_index.observatory.config import RESERVED_LANE_PREFIXES

    lane_prefixes = set()
    for name in (
        "index_basket.json",
        "index_basket_b200.json",
        "index_panel_b300.json",
        "index_panel_b200.json",
        "index_panel_h100_sxm.json",
        "index_panel_h200_sxm.json",
        "index_panel_h100_broad.json",
        "index_panel_h200_broad.json",
    ):
        cfg = json.loads((REPO_ROOT / "config" / name).read_text())
        lane_prefixes.add(cfg["bucket_prefix"])
    assert lane_prefixes == set(RESERVED_LANE_PREFIXES)
    # The observatory's own prefix must never appear here: the fence
    # refuses equality, so a self-entry would refuse the real config.
    assert real_config["bucket_prefix"] not in RESERVED_LANE_PREFIXES


def test_real_config_claim_floor_is_satisfiable(real_config):
    """min_sources_to_claim is validated against configured sources, but a
    slot can only ever be claimed by IMPLEMENTED ones — a floor above the
    implemented count would validate and then never claim a slot."""
    implemented = [
        s["source_id"]
        for s in real_config["sources"]
        if s["source_id"] in COLLECTORS
    ]
    assert len(implemented) >= int(real_config["min_sources_to_claim"])


def test_every_collector_is_a_configured_source(real_config):
    """An implemented collector missing from config would never run — a
    silently dead feed. (The reverse — configured but unimplemented — is
    allowed by design: it records a visible 'unimplemented' hole.)"""
    configured = {s["source_id"] for s in real_config["sources"]}
    orphans = set(COLLECTORS) - configured
    assert not orphans, f"collectors not in config: {sorted(orphans)}"


def _minimal_cfg(**overrides):
    cfg = {
        "lane_id": "t",
        "bucket_prefix": "index/t",
        "capture_slots_utc": [4, 16],
        "canonical_slot_utc": 16,
        "sku_catalog_path": "config/gpu_sku_catalog.json",
        "min_sources_to_claim": 1,
        "sources": [
            {
                "source_id": "a",
                "display_name": "A",
                "source_type": "direct_principal",
                "first_party": True,
            },
            {
                "source_id": "b",
                "display_name": "B",
                "source_type": "aggregator",
                "first_party": False,
            },
        ],
    }
    cfg.update(overrides)
    return cfg


def _load(tmp_path, cfg):
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(cfg))
    return load_observatory_config(p)


def test_minimal_config_loads(tmp_path):
    cfg = _load(tmp_path, _minimal_cfg())
    assert cfg["lane_id"] == "t"


@pytest.mark.parametrize(
    "prefix",
    [
        "curves/raw",
        "index/../curves/x",
        "index//x",
        "index/b300_basket",
        "index/b300_basket/raw",
        "index/b200_basket/sub",
        "index",
    ],
)
def test_config_rejects_bad_or_reserved_prefix(tmp_path, prefix):
    with pytest.raises(ObservatoryConfigError):
        _load(tmp_path, _minimal_cfg(bucket_prefix=prefix))


def test_config_rejects_prefix_that_would_swallow_a_basket_lane(tmp_path):
    """Nesting in the OTHER direction: an observatory prefix ABOVE a basket
    keyspace would LIST/PUT across the ruled lane separation."""
    with pytest.raises(ObservatoryConfigError, match="reserved"):
        _load(tmp_path, _minimal_cfg(bucket_prefix="index/b300_basket"))


@pytest.mark.parametrize(
    "slots",
    [[], [4] * 3, [24], [-1, 4], [4, 4, 16], "not-a-list", [True, 4]],
)
def test_config_rejects_bad_slots(tmp_path, slots):
    with pytest.raises(ObservatoryConfigError):
        _load(tmp_path, _minimal_cfg(capture_slots_utc=slots, canonical_slot_utc=None))


def test_config_accepts_hourly_slots(tmp_path):
    """Hourly cadence ruled 2026-08-23 — the full 24 distinct marks load."""
    cfg = _load(
        tmp_path,
        _minimal_cfg(capture_slots_utc=list(range(24)), canonical_slot_utc=16),
    )
    assert len(cfg["capture_slots_utc"]) == 24


def test_config_rejects_canonical_outside_slots(tmp_path):
    with pytest.raises(ObservatoryConfigError, match="canonical"):
        _load(tmp_path, _minimal_cfg(canonical_slot_utc=12))


def test_config_rejects_unknown_source_type(tmp_path):
    cfg = _minimal_cfg()
    cfg["sources"][0]["source_type"] = "friend-of-a-friend"
    with pytest.raises(ObservatoryConfigError, match="source_type"):
        _load(tmp_path, cfg)


def test_config_requires_first_party_flag(tmp_path):
    cfg = _minimal_cfg()
    del cfg["sources"][1]["first_party"]
    with pytest.raises(ObservatoryConfigError, match="first_party"):
        _load(tmp_path, cfg)


def test_config_rejects_min_claim_above_source_count(tmp_path):
    with pytest.raises(ObservatoryConfigError, match="min_sources_to_claim"):
        _load(tmp_path, _minimal_cfg(min_sources_to_claim=3))


def test_config_rejects_broken_catalog_path(tmp_path):
    with pytest.raises(ObservatoryConfigError, match="sku_catalog_path"):
        _load(tmp_path, _minimal_cfg(sku_catalog_path="config/does_not_exist.json"))


def test_explicit_config_arg_beats_env(tmp_path, monkeypatch):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_minimal_cfg()))
    decoy = tmp_path / "decoy.json"
    decoy.write_text("{}")
    monkeypatch.setenv("RAW_OBSERVATORY_CONFIG_PATH", str(decoy))
    cfg = load_observatory_config(good)
    assert cfg["_config_path"] == str(good)


# ----------------------------------------------------------------- snapshot


def _obs(**over):
    base = dict(
        sku_identifier="NVIDIA HGX B200",
        price_per_gpu_hr=6.5,
        raw_value="6.50",
    )
    base.update(over)
    return observation(**base)


def test_normalize_derives_sku_from_catalog(catalog):
    out = normalize_observation(_obs(), catalog)
    assert out["sku"] == "B200"
    assert out["sku_match"] == "catalog"
    assert out["vendor"] == "NVIDIA"
    assert out["price_usd_gpu_hr"] == 6.5
    assert out["implausible"] is False


def test_normalize_unmapped_is_honest(catalog):
    out = normalize_observation(
        _obs(sku_identifier="FrontierChip Z9000"), catalog
    )
    assert out["sku"] is None
    assert out["sku_match"] == "unmapped"
    assert out["sku_identifier"] == "FrontierChip Z9000"
    # Raw price still recorded; plausibility uses the default band.
    assert out["price_usd_gpu_hr"] == 6.5
    assert out["implausible"] is False


def test_normalize_currency_honesty(catalog):
    out = normalize_observation(
        _obs(currency="EUR", raw_unit="eur_per_node_hr"), catalog
    )
    assert out["price_usd_gpu_hr"] is None
    assert out["price_native_per_gpu_hr"] == 6.5
    assert out["currency"] == "EUR"


def test_normalize_plausibility_uses_per_sku_band(catalog):
    # $0.30/GPU-hr: implausible for a B300, fine for an L4.
    cheap_b300 = normalize_observation(
        _obs(sku_identifier="B300 SXM6", price_per_gpu_hr=0.30, raw_value="0.30"),
        catalog,
    )
    assert cheap_b300["implausible"] is True
    cheap_l4 = normalize_observation(
        _obs(sku_identifier="NVIDIA L4", price_per_gpu_hr=0.30, raw_value="0.30"),
        catalog,
    )
    assert cheap_l4["implausible"] is False


def test_normalize_passthrough_extra(catalog):
    original = {"verification_level": 0, "source_url": "https://x"}
    out = normalize_observation(_obs(extra=original), catalog)
    assert out["extra"] == original
    # Defensive copy: mutating the normalized dict must not reach back
    # into the collector's dict (or vice versa).
    out["extra"]["verification_level"] = 99
    assert original["verification_level"] == 0


def test_non_usd_prints_are_not_screened_against_usd_bands(catalog):
    """The plausibility bands are USD-denominated: an INR 373/hr print is a
    CORRECT figure that a USD band would false-flag forever. Non-USD
    prints record unscreened; a missing price always flags."""
    inr = normalize_observation(
        _obs(
            sku_identifier="NVIDIA H200",
            price_per_gpu_hr=373.0,
            raw_value="373",
            currency="INR",
        ),
        catalog,
    )
    assert inr["implausible"] is False
    assert inr["price_usd_gpu_hr"] is None
    no_price = normalize_observation(
        {"sku_identifier": "NVIDIA H200", "currency": "INR"}, catalog
    )
    assert no_price["implausible"] is True


def test_normalize_flags_non_finite_prices_implausible(catalog):
    """Finiteness fail-closed (harden review 2026-08-23,
    docs/adversarial-reviews.md): json admits NaN/Infinity,
    and the non-USD branch has no band to catch them -- a non-finite
    native must flag implausible in EVERY currency branch (flag-only,
    the capture convention; consumers screen on the flag)."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        usd = normalize_observation(
            {
                "sku_identifier": "NVIDIA H200",
                "price_native_per_gpu_hr": bad,
                "currency": "USD",
            },
            catalog,
        )
        assert usd["implausible"] is True
        eur = normalize_observation(
            {
                "sku_identifier": "NVIDIA H200",
                "price_native_per_gpu_hr": bad,
                "currency": "EUR",
            },
            catalog,
        )
        assert eur["implausible"] is True


def test_first_party_disclosure_comes_from_config(catalog):
    """A collector defaulting first_party=True must not be able to relabel
    a reseller — the validated config flag wins."""
    cfg = _cfg_for_snapshot()
    res = result(
        "gamma",
        method="api",
        url="https://gamma.example",
        observations=[_obs()],
        first_party=True,  # collector lies (or defaults)
    )
    res["status"] = "ok"
    snap = build_capture_snapshot(
        config=cfg,
        catalog=catalog,
        source_results=[res],
        captured_at=_utc(2026, 8, 22, 16, 4),
        run_id="20260822T160400Z-ffff",
        slot_date=_utc(2026, 8, 22, 16).date(),
        slot_hour_utc=16,
        canonical=True,
        capturer={"job": "test", "version": "t/0"},
    )
    gamma = next(s for s in snap["sources"] if s["source_id"] == "gamma")
    assert gamma["first_party_observation"] is False


def test_result_refuses_zero_observations():
    with pytest.raises(RuntimeError, match="zero"):
        result("x", method="html", url="https://x", observations=[])


def _cfg_for_snapshot():
    return {
        "lane_id": "raw_observatory_v1",
        "sku_catalog_path": "config/gpu_sku_catalog.json",
        "sources": [
            {
                "source_id": "alpha",
                "display_name": "Alpha",
                "source_type": "direct_principal",
                "first_party": True,
            },
            {
                "source_id": "beta",
                "display_name": "Beta",
                "source_type": "marketplace",
                "first_party": True,
            },
            {
                "source_id": "gamma",
                "display_name": "Gamma",
                "source_type": "aggregator",
                "first_party": False,
            },
        ],
    }


def _snapshot(catalog):
    results = [
        result(
            "alpha",
            method="html",
            url="https://alpha.example/pricing",
            observations=[
                _obs(),
                _obs(sku_identifier="NVIDIA H200", price_per_gpu_hr=3.5, raw_value="3.50"),
                _obs(sku_identifier="Mystery Meat 9000", price_per_gpu_hr=1.0, raw_value="1.00"),
            ],
        ),
        {
            "source_id": "beta",
            "status": "error",
            "error": "boom",
            "observations": [],
        },
        {
            "source_id": "gamma",
            "status": "unimplemented",
            "error": "collector not implemented yet",
            "observations": [],
        },
    ]
    results[0]["status"] = "ok"
    return build_capture_snapshot(
        config=_cfg_for_snapshot(),
        catalog=catalog,
        source_results=results,
        captured_at=_utc(2026, 8, 22, 16, 4),
        run_id="20260822T160400Z-abcd",
        slot_date=_utc(2026, 8, 22, 16).date(),
        slot_hour_utc=16,
        canonical=True,
        capturer={"job": "test", "version": "t/0"},
    )


def test_snapshot_shape_and_rollups(catalog):
    snap = _snapshot(catalog)
    assert snap["kind"] == SNAPSHOT_KIND
    assert snap["lane_id"] == "raw_observatory_v1"
    assert [s["source_id"] for s in snap["sources"]] == ["alpha", "beta", "gamma"]
    assert snap["sources_ok"] == ["alpha"]
    assert snap["sources_failed"] == ["beta"]
    assert snap["sources_unimplemented"] == ["gamma"]
    assert snap["observation_count"] == 3
    assert snap["skus_observed"] == ["B200", "H200"]
    assert snap["unmapped_identifiers"] == ["Mystery Meat 9000"]
    # Disclosure fields ride from config into the document.
    gamma = snap["sources"][2]
    assert gamma["source_type"] == "aggregator"


def test_snapshot_derives_nothing_cross_source(catalog):
    """The record-raw ruling: no composite, no index, no basis pairs — no
    key in the document may carry cross-source arithmetic."""
    snap = _snapshot(catalog)
    for forbidden in ("basis_pairs", "index", "composite", "basket_sources_ok"):
        assert forbidden not in snap


# ------------------------------------------------------------------ collect


def _collect_cfg(sources, **over):
    cfg = {
        "capture_budget_seconds": 30,
        "per_source_deadline_seconds": 5,
        "per_source_timeout_seconds": 1,
        "sources": sources,
    }
    cfg.update(over)
    return cfg


def test_collect_all_records_visible_holes():
    def ok_collector(timeout=None, options=None):
        return result(
            "wrong-id",  # registry key must win
            method="api",
            url="https://x",
            observations=[_obs()],
        )

    def broken_collector(timeout=None, options=None):
        raise RuntimeError("page reshaped")

    cfg = _collect_cfg(
        [
            {"source_id": "good"},
            {"source_id": "bad"},
            {"source_id": "missing"},
        ]
    )
    results = collect_all(
        cfg, {"good": ok_collector, "bad": broken_collector}
    )
    by_id = {r["source_id"]: r for r in results}
    assert by_id["good"]["status"] == "ok"
    assert by_id["good"]["source_id"] == "good"  # relabeled by registry key
    assert by_id["bad"]["status"] == "error"
    assert "page reshaped" in by_id["bad"]["error"]
    assert by_id["missing"]["status"] == "unimplemented"


def test_collect_all_passes_options_through():
    seen = {}

    def collector(timeout=None, options=None):
        seen["options"] = options
        return result("s", method="api", url="https://x", observations=[_obs()])

    cfg = _collect_cfg([{"source_id": "s", "options": {"gpus": ["a", "b"]}}])
    collect_all(cfg, {"s": collector})
    assert seen["options"] == {"gpus": ["a", "b"]}


def test_collect_all_budget_exhaustion_is_visible():
    """A hung source is abandoned at its budget share, and every source
    after the spent budget records the explicit exhausted error — visible
    holes, never a silently shorter list."""

    def slow(timeout=None, options=None):
        time.sleep(1)
        return result("s1", method="api", url="https://x", observations=[_obs()])

    cfg = _collect_cfg(
        [{"source_id": "s1"}, {"source_id": "s2"}],
        capture_budget_seconds=0.05,
        per_source_deadline_seconds=0.06,
    )
    results = collect_all(cfg, {"s1": slow, "s2": slow})
    assert results[0]["status"] == "error"
    assert "budget share" in results[0]["error"]
    assert results[1]["status"] == "error"
    assert "exhausted before this source ran" in results[1]["error"]


def test_collect_all_deadline_abandons_hung_source():
    def hung(timeout=None, options=None):
        time.sleep(5)

    cfg = _collect_cfg(
        [{"source_id": "s1"}], per_source_deadline_seconds=0.1
    )
    results = collect_all(cfg, {"s1": hung})
    assert results[0]["status"] == "error"
    assert "budget share" in results[0]["error"]


# -------------------------------------------------------------------- store


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


def test_upload_writes_snapshot_then_pointer_last(catalog):
    client = FakeS3()
    payload = _snapshot(catalog)
    out = upload_capture_snapshot(
        client,
        "curves",
        payload,
        prefix="index/raw_observatory",
        now=_utc(2026, 8, 22, 16, 5),
    )
    key = (
        "index/raw_observatory/snapshots/2026-08-22/"
        "slot16-20260822T160400Z-abcd.json"
    )
    assert out["snapshot_key"] == key
    assert client.put_order == [key, "index/raw_observatory/latest.json"]
    pointer = json.loads(client.objects["index/raw_observatory/latest.json"])
    assert pointer["snapshot_key"] == key
    assert pointer["lane_id"] == "raw_observatory_v1"
    assert pointer["skus_observed"] == ["B200", "H200"]
    assert pointer["observation_count"] == 3
    assert pointer["published_at"] == "2026-08-22T16:05:00Z"


def test_upload_refuses_to_overwrite_different_bytes(catalog):
    client = FakeS3()
    payload = _snapshot(catalog)
    upload_capture_snapshot(
        client, "curves", payload, prefix="index/raw_observatory"
    )
    mutated = json.loads(json.dumps(payload))
    mutated["sources"][0]["observations"][0]["price_usd_gpu_hr"] = 9.99
    with pytest.raises(BucketPublishError, match="append-only"):
        upload_capture_snapshot(
            client, "curves", mutated, prefix="index/raw_observatory"
        )


def test_pointer_never_regresses(catalog):
    client = FakeS3()
    newer = _snapshot(catalog)
    upload_capture_snapshot(
        client, "curves", newer, prefix="index/raw_observatory"
    )
    older = json.loads(json.dumps(newer))
    older["captured_at"] = "2026-08-22T10:04:00Z"
    older["capture_date"] = "2026-08-22"
    older["slot_hour_utc"] = 10
    older["run_id"] = "20260822T100400Z-eeee"
    out = upload_capture_snapshot(
        client, "curves", older, prefix="index/raw_observatory"
    )
    assert out["status"] == "published_pointer_kept"
    pointer = json.loads(client.objects["index/raw_observatory/latest.json"])
    assert pointer["slot_hour_utc"] == 16


# ------------------------------------------------------------------- runner


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "capture_raw_observatory",
        REPO_ROOT / "scripts" / "capture_raw_observatory.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_dry_run_records_nothing(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    cfg = _minimal_cfg(min_sources_to_claim=2)
    cfg_path = tmp_path / "obs.json"
    cfg_path.write_text(json.dumps(cfg))

    def fake_collect_all(config, collectors, only=None):
        return [
            {
                "source_id": "a",
                "status": "ok",
                "method": "api",
                "url": "https://a",
                "first_party_observation": True,
                "fetched_at": "2026-08-22T16:00:00Z",
                "observations": [_obs()],
            },
            {
                "source_id": "b",
                "status": "error",
                "error": "boom",
                "observations": [],
            },
        ]

    monkeypatch.setattr(runner, "collect_all", fake_collect_all)
    monkeypatch.setattr(
        "sys.argv",
        ["capture_raw_observatory.py", "--dry-run", "--config", str(cfg_path)],
    )
    # Dry run: below the claim floor (1 of 2 < 2) but records nothing, so
    # the gate is exempt — mirrors the basket CLI contract.
    assert runner.main() == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "B200" in out


def test_runner_rejects_unknown_only_source(monkeypatch, capsys):
    """A typo'd --only-source must fail loud at argparse time, not silently
    select nothing and exit with 'every implemented source failed'."""
    runner = _load_runner()
    monkeypatch.setattr(
        "sys.argv",
        ["capture_raw_observatory.py", "--only-source", "runpodd", "--dry-run"],
    )
    with pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert excinfo.value.code == 2
    assert "no such collector" in capsys.readouterr().err


def test_runner_fails_when_every_source_fails(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    cfg_path = tmp_path / "obs.json"
    cfg_path.write_text(json.dumps(_minimal_cfg()))

    def fake_collect_all(config, collectors, only=None):
        return [
            {"source_id": "a", "status": "error", "error": "x", "observations": []},
            {"source_id": "b", "status": "error", "error": "y", "observations": []},
        ]

    monkeypatch.setattr(runner, "collect_all", fake_collect_all)
    monkeypatch.setattr(
        "sys.argv",
        ["capture_raw_observatory.py", "--dry-run", "--config", str(cfg_path)],
    )
    assert runner.main() == 1
    assert "every implemented source failed" in capsys.readouterr().out
