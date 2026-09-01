# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Public API surface tripwire, modeled on numpy's test_public_api.py.

Downstream consumers (the composite pipeline and the MCP/JSON API server)
import these modules by name. The allowlists below enumerate what EXISTS
now — a tripwire against accidental surface change, not an aspirational
API doc: a new public module or top-level name must be added here
deliberately (reviewed as code), and a vanished one fails loudly before a
consumer finds out. Signature-level compatibility is a separate CI job
(griffe check); this test guards NAMES only.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import gpu_index

PUBLIC_MODULES = {
    "gpu_index",
    "gpu_index.common",
    "gpu_index.common.bucket",
    "gpu_index.common.http",
    "gpu_index.common.jsondiff",
    "gpu_index.common.slots",
    "gpu_index.common.store",
    "gpu_index.index",
    "gpu_index.index.composite",
    "gpu_index.index.config",
    "gpu_index.index.fx",
    "gpu_index.index.panel",
    "gpu_index.index.panel_config",
    "gpu_index.index.panel_schedule",
    "gpu_index.index.period_rate",
    "gpu_index.index.report",
    "gpu_index.index.screens",
    "gpu_index.index.snapshot",
    "gpu_index.index.sources",
    "gpu_index.index.weights",
    "gpu_index.published",
    "gpu_index.published.artifacts",
    "gpu_index.published.reader",
    "gpu_index.published.verify",
    "gpu_index.observatory",
    "gpu_index.observatory.catalog",
    "gpu_index.observatory.collect",
    "gpu_index.observatory.config",
    "gpu_index.observatory.observation",
    "gpu_index.observatory.snapshot",
    "gpu_index.observatory.sources",
    "gpu_index.observatory.store",
}

# Collector modules are public AS A NAMESPACE (one file per source_id,
# auto-discovered); individual source modules come and go by design.
PUBLIC_NAMESPACES = ("gpu_index.observatory.sources.",)

# Public top-level names (functions/classes defined in or re-exported by
# the module, name not starting with "_") for the most load-bearing
# modules. Re-exports are part of the surface consumers see, so they are
# listed where they exist today.
PUBLIC_NAMES = {
    "gpu_index.index.composite": {
        "FxUnavailableError", "advance_weight_state", "advance_window",
        "calc_params", "compute_day", "compute_dynamic_weights",
        "constituents", "daily_source_observation", "eur_to_usd",
        "evaluate_filter", "filter_observation", "median_stddev_composite",
        "resolve_daily_print", "resolve_slot_prints", "select_slot_snapshot",
        "series_print", "us_ca_verified_host", "vast_vwm_verified_us_ca",
        "vote_stddev", "weighted_composite", "window_incompatible",
    },
    "gpu_index.index.weights": {
        "advance_panel_weight_state", "advance_weight_state",
        "allocate_weights", "attendance", "build_samples",
        "compute_dynamic_weights", "compute_panel_weights", "fit_ridge",
        "in_sample_q", "loo_basket_return", "new_weight_state",
        "predict_ridge", "predictive_scores", "predictive_scores_obs",
        "recently_printed", "series_print", "solve_linear",
        "dw_vote_tail", "source_return", "stamp_to_hour_iso",
        "validate_attendance_floor",
    },
    "gpu_index.index.config": {
        "BasketConfigError", "load_basket_config", "sources_by_id",
    },
    "gpu_index.index.fx": {
        "FxUnavailableError", "ensure_rates", "eur_to_usd", "fetch",
        "fx_key", "get_object_bytes", "list_object_keys",
        "load_stored_rates", "lookup_rate", "parse_ecb_rates",
        "persist_rates", "put_json_bytes", "rates_cover", "rfc3339",
    },
    "gpu_index.index.screens": {
        # PRICEABLE_CURRENCIES is public surface too (a constant, so it is
        # asserted by test_declared_waiver_names_are_public below rather
        # than by the function/class walk).
        "apply_jump_screen", "lowest_eligible", "screen_params",
    },
    "gpu_index.observatory.catalog": {
        "SkuCatalogError", "boundary_pattern", "load_sku_catalog",
        "match_sku", "normalize_label", "plausible_band",
    },
    "gpu_index.common.store": {
        "BucketConfig", "BucketPublishError", "build_pointer",
        "composite_exists", "composite_key", "composite_pointer_key",
        "get_composite", "get_object_bytes", "latest_pointer_key",
        "list_object_keys", "make_client", "make_run_id",
        "move_pointer_no_regress", "previous_day_has_snapshots",
        "day_slot_keys", "get_panel_composite", "get_snapshot_by_key",
        "list_panel_observations", "panel_composite_exists",
        "panel_composite_key", "put_immutable", "put_json_bytes",
        "payload_slot_minute", "read_day_snapshots", "rfc3339",
        "slot_already_captured", "slot_hours_present",
        "slot_key_prefix", "slot_minutes_present", "slot_token",
        "snapshot_bytes",
        "snapshot_day_prefix", "snapshot_key", "upload_capture_snapshot",
        "upload_composite", "upload_panel_composite",
        "write_local_snapshot",
    },
    "gpu_index.common.bucket": {
        "BucketConfig", "BucketPublishError", "LocalStore",
        "PublicReadStore", "get_object_bytes", "list_object_keys",
        "make_client", "put_bytes", "put_json_bytes",
    },
    "gpu_index.common.jsondiff": {
        "field_diffs",
    },
    "gpu_index.observatory.collect": {
        "DeadlineExceeded", "TransportError", "call_with_deadline",
        "classify_failure", "collect_all",
    },
    "gpu_index.index.panel": {
        "FxUnavailableError", "advance_panel_weight_state",
        "advance_window", "apply_panel_jump_screen", "boundary_pattern",
        "compile_screens",
        "compute_observation", "compute_panel_weights",
        "embedded_calc_params", "eur_to_usd", "evaluate_filter",
        "exclusion_applies", "filter_observation",
        "jump_reference_prints", "lium_vwm_book_floor",
        "lowest_eligible_print", "median_stddev_composite",
        "member_eligible_rows", "normalize_label", "panel_calc_params",
        "record_exclusion_reason", "record_source_for",
        "dw_vote_tail", "resolve_member_print", "series_print",
        "stamp_to_date_hour", "stamp_to_date_minute",
        "stamp_to_hour_iso", "stamp_to_obs_key", "us_ca_verified_host",
        "vast_vwm_verified_us_ca_floor", "vast_vwm_verified_us_ca_v2",
        "vote_stddev", "weighted_composite", "window_incompatible",
    },
    "gpu_index.index.panel_config": {
        "PanelConfigError", "PanelSchedule", "PanelScheduleError",
        "date_hour_to_stamp", "date_minute_to_stamp",
        "load_panel_config", "normalize_label",
        "panel_schedule", "validate_panel_config",
    },
    "gpu_index.index.panel_schedule": {
        "PanelSchedule", "PanelScheduleError", "date_hour_to_stamp",
        "date_minute_to_stamp", "hour_iso_to_stamp", "obs_key_to_stamp",
        "stamp_to_date_hour", "stamp_to_date_minute",
        "stamp_to_hour_iso", "stamp_to_obs_key",
    },
    "gpu_index.index.period_rate": {
        "PanelSchedule", "PeriodRateError", "classify_artifact",
        "coverage_band", "fill_lookback_stamps", "find_gaps",
        "period_report", "stamp_to_hour_iso",
    },
    "gpu_index.published": {
        "ArtifactDigestError", "ObservationCheck", "PublishedRecordError",
        "PublishedRecordReader", "UnsupportedStatisticError",
        "canonical_compact_bytes", "day_key", "decode_and_verify_artifact",
        "latest_key", "payload_digest", "recompute_observation",
        "select_observations", "series_key",
    },
    "gpu_index.published.artifacts": {
        "ArtifactDigestError", "PublishedRecordError",
        "canonical_compact_bytes", "day_key", "decode_and_verify_artifact",
        "latest_key", "payload_digest", "series_key",
    },
    "gpu_index.published.reader": {
        "BucketConfig", "PublishedRecordReader", "day_key",
        "decode_and_verify_artifact", "get_object_bytes", "latest_key",
        "make_client", "series_key",
    },
    "gpu_index.published.verify": {
        "ObservationCheck", "PublishedRecordError", "UnsupportedStatisticError",
        "disclosure_window_warning", "field_diffs",
        "median_stddev_composite", "recompute_observation",
        "select_observations",
    },
}


def _walk_modules():
    yield "gpu_index"
    for info in pkgutil.walk_packages(
        gpu_index.__path__,
        prefix="gpu_index.",
        onerror=lambda name: pytest.fail(f"failed to walk/import {name}"),
    ):
        yield info.name


def test_module_surface_matches_allowlist():
    found = set(_walk_modules())
    namespaced = {m for m in found if m.startswith(PUBLIC_NAMESPACES)}
    unexpected = sorted(found - PUBLIC_MODULES - namespaced)
    missing = sorted(PUBLIC_MODULES - found)
    assert not unexpected, (
        f"Found unexpected public module(s): {unexpected} — a new module is "
        "new public API; add it here deliberately (or prefix it with _)"
    )
    assert not missing, (
        f"Missing expected public module(s): {missing} — removing or "
        "renaming a public module breaks downstream imports"
    )
    assert namespaced, "collector namespace is empty — the registry vanished"


def test_public_modules_import():
    for name in sorted(PUBLIC_MODULES):
        importlib.import_module(name)


def _observed_names(module_name):
    mod = importlib.import_module(module_name)
    return {
        name
        for name, obj in vars(mod).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", "").startswith("gpu_index")
    }


def test_declared_waiver_names_are_public():
    """The two cross-package waiver imports (ARCHITECTURE.md) ride
    PUBLIC names — the panel engine must never reach for underscore
    internals, and the names must stay importable as spelled."""
    from gpu_index.index.screens import PRICEABLE_CURRENCIES
    from gpu_index.observatory.catalog import boundary_pattern

    assert PRICEABLE_CURRENCIES == ("USD", "EUR")
    assert boundary_pattern("B200").search("GB200") is None
    assert boundary_pattern("B200").search("8x B200 SXM") is not None


@pytest.mark.parametrize("module_name", sorted(PUBLIC_NAMES))
def test_public_names_match_allowlist(module_name):
    observed = _observed_names(module_name)
    expected = PUBLIC_NAMES[module_name]
    unexpected = sorted(observed - expected)
    missing = sorted(expected - observed)
    assert not unexpected, (
        f"Found unexpected public name(s) in {module_name}: {unexpected} — "
        "new names are new public API; allowlist them deliberately (or "
        "prefix with _)"
    )
    assert not missing, (
        f"Missing expected public name(s) in {module_name}: {missing} — "
        "removing or renaming a public name breaks downstream consumers"
    )
