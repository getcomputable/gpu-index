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
    "gpu_index.common.slots",
    "gpu_index.common.store",
    "gpu_index.index",
    "gpu_index.index.composite",
    "gpu_index.index.config",
    "gpu_index.index.fx",
    "gpu_index.index.report",
    "gpu_index.index.screens",
    "gpu_index.index.snapshot",
    "gpu_index.index.sources",
    "gpu_index.index.weights",
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
        "advance_weight_state", "allocate_weights", "build_samples",
        "compute_dynamic_weights", "fit_ridge", "in_sample_q",
        "loo_basket_return", "new_weight_state", "predict_ridge",
        "predictive_scores", "recently_printed", "series_print",
        "solve_linear", "source_return",
    },
    "gpu_index.index.config": {
        "BasketConfigError", "load_basket_config", "sources_by_id",
    },
    "gpu_index.index.fx": {
        "FxUnavailableError", "ensure_rates", "eur_to_usd", "fetch",
        "fx_key", "get_object_bytes", "list_object_keys",
        "load_stored_rates", "lookup_rate", "parse_ecb_rates",
        "persist_rates", "put_json_bytes", "rfc3339",
    },
    "gpu_index.index.screens": {
        "apply_jump_screen", "lowest_eligible", "screen_params",
    },
    "gpu_index.common.store": {
        "BucketConfig", "BucketPublishError", "build_pointer",
        "composite_exists", "composite_key", "composite_pointer_key",
        "get_composite", "get_object_bytes", "latest_pointer_key",
        "list_object_keys", "make_client", "make_run_id",
        "move_pointer_no_regress", "previous_day_has_snapshots",
        "put_immutable", "put_json_bytes", "read_day_snapshots", "rfc3339",
        "slot_already_captured", "slot_hours_present", "slot_key_prefix",
        "snapshot_bytes", "snapshot_day_prefix", "snapshot_key",
        "upload_capture_snapshot", "upload_composite",
        "write_local_snapshot",
    },
    "gpu_index.common.bucket": {
        "BucketConfig", "BucketPublishError", "LocalStore",
        "PublicReadStore", "get_object_bytes", "list_object_keys",
        "make_client", "put_bytes", "put_json_bytes",
    },
    "gpu_index.observatory.collect": {
        "call_with_deadline", "collect_all",
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
