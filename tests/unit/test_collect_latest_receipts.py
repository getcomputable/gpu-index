# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Current receipt collection: state, policy, identity, and exit pins."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from threading import Event

import pytest

import gpu_index.common.http as common_http
from gpu_index.index.panel import compile_screens, panel_calc_params
from gpu_index.index.panel_config import load_panel_config
from gpu_index.observatory.catalog import load_sku_catalog
from gpu_index.observatory.collect import DeadlineExceeded, call_with_deadline
from gpu_index.observatory.config import (
    load_observatory_config,
    resolve_catalog_path,
)
from gpu_index.observatory.observation import observation as raw_observation

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "collect_latest_receipts.py"


@pytest.fixture(scope="module")
def collect_module():
    spec = importlib.util.spec_from_file_location(
        "collect_latest_receipts_for_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def configs():
    observatory = load_observatory_config()
    return (
        load_panel_config(REPO_ROOT / "config" / "index_panel_h100_sxm.json"),
        observatory,
        load_sku_catalog(resolve_catalog_path(observatory)),
    )


def _receipt(source_id: str, source_url: str, price: float, label: str) -> dict:
    return {
        "source_id": source_id,
        "source_url": source_url,
        "price": price,
        "currency": "USD",
        "sku_identifier": label,
        "status": "ok",
    }


def _latest(receipts: list[dict]) -> dict:
    return {
        "data": {
            "versions": [
                {
                    "sku": "H100",
                    "current_version": 2,
                    "methodology_id": "h100_sxm_v1_calc_v10",
                }
            ],
            "observations": [
                {
                    "sku": "H100",
                    "methodology_id": "h100_sxm_v1_calc_v6",
                    "observed_at": "2026-08-31T23:15:00.000Z",
                    "receipts": [],
                },
                {
                    "sku": "H100",
                    "methodology_id": "h100_sxm_v1_calc_v10",
                    "observed_at": "2026-09-01T08:45:00.000Z",
                    "receipts": receipts,
                },
            ],
        }
    }


class _Reader:
    def __init__(self, latest: dict, expected_user_agent: str) -> None:
        assert common_http.current_user_agent() == expected_user_agent
        self.latest = latest

    def read_latest(self) -> dict:
        assert common_http.current_user_agent().startswith("CGI-Verify/1.0")
        return self.latest


def _collector(
    *,
    source_id: str,
    source_url: str,
    label: str,
    price: float,
    expected_user_agent: str,
    calls: list[str],
):
    def collect(*, timeout, options=None):
        del timeout, options
        assert common_http.current_user_agent() == expected_user_agent
        calls.append(source_id)
        return {
            "source_id": source_id,
            "method": "fixture",
            "url": source_url,
            "observations": [
                raw_observation(
                    sku_identifier=label,
                    price_per_gpu_hr=price,
                    raw_value=str(price),
                )
            ],
        }

    return collect


def test_fixture_emits_all_four_states_and_restores_identity(
    collect_module, configs
):
    panel, observatory, catalog = configs
    urls = {
        "civo": "https://www.civo.com/pricing",
        "coreweave": "https://www.coreweave.com/pricing",
        "crusoe": "https://www.crusoe.ai/cloud/pricing",
        "lambda": "https://lambda.ai/pricing",
    }
    labels = {
        "civo": "NVIDIA H100 SXM",
        "coreweave": "NVIDIA HGX H100",
        "crusoe": "NVIDIA H100 80GB HGX",
        "lambda": "NVIDIA H100 SXM",
    }
    receipts = [
        _receipt("civo", urls["civo"], 2.99, labels["civo"]),
        _receipt("coreweave", urls["coreweave"], 6.155, labels["coreweave"]),
        _receipt("crusoe", urls["crusoe"], 3.9, labels["crusoe"]),
        _receipt("lambda", urls["lambda"], 3.99, labels["lambda"]),
    ]
    calls: list[str] = []
    verifier_ua = collect_module.VERIFY_USER_AGENT
    collectors = {
        "civo": _collector(
            source_id="civo",
            source_url=urls["civo"],
            label=labels["civo"],
            price=2.99,
            expected_user_agent=verifier_ua,
            calls=calls,
        ),
        "coreweave": _collector(
            source_id="coreweave",
            source_url=urls["coreweave"],
            label=labels["coreweave"],
            price=6.5,
            expected_user_agent=verifier_ua,
            calls=calls,
        ),
    }

    def unreachable(*, timeout, options=None):
        del timeout, options
        assert common_http.current_user_agent() == verifier_ua
        calls.append("crusoe")
        raise OSError("fixture host did not answer")

    collectors["crusoe"] = unreachable

    def forbidden(*, timeout, options=None):  # pragma: no cover - must not run
        del timeout, options
        raise AssertionError("key-required source was fetched")

    collectors["lambda"] = forbidden
    original_ua = common_http.current_user_agent()

    report = collect_module.collect_latest_receipts(
        "h100",
        reader_factory=lambda: _Reader(_latest(receipts), verifier_ua),
        panel_config=panel,
        observatory_config=observatory,
        catalog=catalog,
        collectors=collectors,
        skip_reasons={"lambda": collect_module.SKIP_KEY_REQUIRED},
        approved_urls={source_id: [url] for source_id, url in urls.items()},
    )

    assert common_http.current_user_agent() == original_ua
    assert [check.state for check in report.checks] == [
        collect_module.SAME,
        collect_module.MOVED,
        collect_module.UNREACHABLE,
        collect_module.SKIPPED,
    ]
    assert sorted(calls) == ["civo", "coreweave", "crusoe"]
    assert report.checks[2].reason == "fetch"
    assert report.checks[3].reason == "api-key-required"
    assert report.exit_code == 0

    lines = report.lines()
    assert len(lines) == len(receipts) + 2
    assert "price changed since capture" in lines[2]
    assert "discrep" not in lines[2].lower()
    assert "mismatch" not in lines[2].lower()
    assert lines[-1] == (
        "summary: 4 seat(s): 1 SAME, 1 MOVED, "
        "1 UNREACHABLE, 1 SKIPPED"
    )


def test_retrieval_restricted_source_is_not_called_and_exit_is_two(
    collect_module, configs
):
    panel, observatory, catalog = configs
    receipt = _receipt(
        "lambda", "https://lambda.ai/pricing", 3.99, "NVIDIA H100 SXM"
    )

    calls = []

    def forbidden(*, timeout, options=None):  # pragma: no cover - must not run
        del timeout, options
        calls.append("lambda")
        raise AssertionError("retrieval-restricted source was fetched")

    report = collect_module.collect_latest_receipts(
        "H100",
        reader_factory=lambda: _Reader(
            _latest([receipt]), collect_module.VERIFY_USER_AGENT
        ),
        panel_config=panel,
        observatory_config=observatory,
        catalog=catalog,
        collectors={"lambda": forbidden},
        skip_reasons={
            "lambda": collect_module.SKIP_RETRIEVAL_RESTRICTED
        },
        approved_urls={"lambda": ["https://lambda.ai/pricing"]},
    )

    assert report.checks[0].state == collect_module.SKIPPED
    assert report.checks[0].reason == "retrieval-restricted"
    assert calls == []
    assert report.comparison_count == 0
    assert report.exit_code == 2


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        (["SAME"], 0),
        (["MOVED"], 0),
        (["UNREACHABLE", "SKIPPED"], 2),
        ([], 2),
    ],
)
def test_exit_codes_are_checked_directly_without_a_pipeline(
    collect_module, checks, expected
):
    report = collect_module.CollectionReport(
        sku="H100",
        observed_at="2026-09-01T08:45:00.000Z",
        methodology_id="h100_sxm_v1_calc_v10",
        checks=tuple(
            collect_module.SeatCheck("H100", f"source-{i}", state)
            for i, state in enumerate(checks)
        ),
    )

    assert report.exit_code == expected


def test_current_pointer_selects_the_latest_methodology(collect_module):
    current = collect_module._select_latest_observation(  # noqa: SLF001
        _latest([]), "H100"
    )
    assert current["methodology_id"] == "h100_sxm_v1_calc_v10"
    assert current["observed_at"] == "2026-09-01T08:45:00.000Z"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda latest: latest["data"].update(versions=[]), "0 current-version"),
        (
            lambda latest: latest["data"]["versions"].append(
                dict(latest["data"]["versions"][0])
            ),
            "2 current-version",
        ),
        (lambda latest: latest["data"].update(observations=[]), "0 H100 prints"),
        (
            lambda latest: latest["data"]["observations"].append(
                dict(latest["data"]["observations"][-1])
            ),
            "2 H100 prints",
        ),
    ],
)
def test_current_pointer_rejects_missing_or_ambiguous_entries(
    collect_module, mutator, message
):
    latest = _latest([])
    mutator(latest)
    with pytest.raises(collect_module.CollectionError, match=message):
        collect_module._select_latest_observation(latest, "H100")  # noqa: SLF001


def test_current_pointer_rejects_a_missing_public_document(collect_module):
    with pytest.raises(collect_module.CollectionError, match="no latest.json"):
        collect_module._select_latest_observation(None, "H100")  # noqa: SLF001


def test_production_url_discovery_approves_only_declared_urls(
    collect_module, configs
):
    panel, observatory, catalog = configs
    source_url = "https://www.civo.com/pricing"
    calls: list[str] = []
    collector = _collector(
        source_id="civo",
        source_url=source_url,
        label="NVIDIA H100 SXM",
        price=2.99,
        expected_user_agent=collect_module.VERIFY_USER_AGENT,
        calls=calls,
    )
    module_name = "test_collect_latest_declared_urls"
    collector.__module__ = module_name
    source_module = types.ModuleType(module_name)
    source_module.URL = source_url
    source_module.URL_BASE = "https://api.example.test/v1"
    source_module.URLS = ("https://cdn.example.test/prices",)
    sys.modules[module_name] = source_module
    try:
        declared = collect_module._declared_urls(  # noqa: SLF001
            collector,
            {
                "options": {
                    "nested": ["https://options.example.test/prices", "ignored"]
                }
            },
        )
        assert declared == {
            source_url,
            "https://api.example.test/v1",
            "https://cdn.example.test/prices",
            "https://options.example.test/prices",
        }
        report = collect_module.collect_latest_receipts(
            "h100",
            reader_factory=lambda: _Reader(
                _latest(
                    [_receipt("civo", source_url, 2.99, "NVIDIA H100 SXM")]
                ),
                collect_module.VERIFY_USER_AGENT,
            ),
            panel_config=panel,
            observatory_config=observatory,
            catalog=catalog,
            collectors={"civo": collector},
        )
    finally:
        del sys.modules[module_name]

    assert calls == ["civo"]
    assert report.checks[0].state == collect_module.SAME


def test_eur_receipt_uses_its_captured_fx_rate(collect_module):
    panel = load_panel_config(REPO_ROOT / "config" / "index_panel_b300.json")
    observatory = load_observatory_config()
    catalog = load_sku_catalog(resolve_catalog_path(observatory))
    params = panel_calc_params(panel)
    screens = compile_screens(params)
    result = {
        "source_id": "scaleway",
        "status": "ok",
        "observations": [
            raw_observation(
                sku_identifier="B300-SXM",
                price_per_gpu_hr=9.48,
                currency="EUR",
                raw_value="9.48",
                raw_unit="eur_per_gpu_hr",
            )
        ],
    }

    price = collect_module._resolve_collected_price(  # noqa: SLF001
        receipt={"source_id": "scaleway", "currency": "EUR", "fx_rate": 1.1},
        result=result,
        catalog=catalog,
        params=params,
        screens=screens,
        observed_at="2026-09-01T08:45:00.000Z",
    )
    assert price == collect_module.Decimal("10.428")

    unavailable = collect_module._resolve_collected_price(  # noqa: SLF001
        receipt={"source_id": "scaleway", "currency": "EUR"},
        result=result,
        catalog=catalog,
        params=params,
        screens=screens,
        observed_at="2026-09-01T08:45:00.000Z",
    )
    assert unavailable is None


def test_verifier_identity_survives_an_abandoned_collector_thread(
    collect_module,
):
    started = Event()
    release = Event()
    finished = Event()
    seen: list[str] = []

    def delayed(*, timeout, options=None):
        del timeout, options
        started.set()
        release.wait(timeout=2)
        seen.append(common_http.current_user_agent())
        finished.set()

    original = common_http.current_user_agent()
    with collect_module._verifier_identity():  # noqa: SLF001
        with pytest.raises(DeadlineExceeded):
            call_with_deadline(delayed, timeout=1, deadline=0.01)
        assert started.is_set()
    assert common_http.current_user_agent() == original

    release.set()
    assert finished.wait(timeout=2)
    assert seen == [collect_module.VERIFY_USER_AGENT]


def test_unapproved_receipt_url_is_skipped_before_collection(
    collect_module, configs
):
    panel, observatory, catalog = configs
    receipt = _receipt(
        "civo",
        "https://unexpected.example/pricing",
        2.99,
        "NVIDIA H100 SXM",
    )

    def forbidden(*, timeout, options=None):  # pragma: no cover - must not run
        del timeout, options
        raise AssertionError("unapproved URL was fetched")

    report = collect_module.collect_latest_receipts(
        "h100",
        reader_factory=lambda: _Reader(
            _latest([receipt]), collect_module.VERIFY_USER_AGENT
        ),
        panel_config=panel,
        observatory_config=observatory,
        catalog=catalog,
        collectors={"civo": forbidden},
        approved_urls={"civo": ["https://www.civo.com/pricing"]},
    )

    assert report.checks[0].state == collect_module.SKIPPED
    assert report.checks[0].reason == "receipt-source-url-not-approved"
    assert report.exit_code == 2


def test_preflight_failures_are_typed_without_running_collectors(
    collect_module, configs
):
    panel, observatory, catalog = configs
    base = _latest([])["data"]["observations"][-1]

    cases = [
        (
            _receipt("civo", "https://www.civo.com/pricing", 0, "NVIDIA H100 SXM"),
            {"civo": lambda **kwargs: kwargs},
            "receipt-price-unavailable",
        ),
        (
            _receipt(
                "civo", "https://www.civo.com/pricing", 2.99, "NVIDIA H100 SXM"
            ),
            {},
            "collector-unavailable",
        ),
        (
            _receipt(
                "not-configured",
                "https://prices.example.test",
                2.99,
                "NVIDIA H100 SXM",
            ),
            {"not-configured": lambda **kwargs: kwargs},
            "source-configuration-unavailable",
        ),
    ]
    for receipt, collectors, expected_reason in cases:
        report = collect_module._compare_observation(  # noqa: SLF001
            {**base, "receipts": [receipt]},
            panel_config=panel,
            observatory_config=observatory,
            catalog=catalog,
            collectors=collectors,
            skip_reasons={},
            approved_urls={},
        )
        assert report.checks[0].state == collect_module.SKIPPED
        assert report.checks[0].reason == expected_reason

    with pytest.raises(collect_module.CollectionError, match="unknown source skip"):
        collect_module._compare_observation(  # noqa: SLF001
            base,
            panel_config=panel,
            observatory_config=observatory,
            catalog=catalog,
            collectors={},
            skip_reasons={"civo": "silent"},
        )


def test_missing_and_mismatched_results_are_typed(
    collect_module, configs, monkeypatch
):
    panel, observatory, catalog = configs
    source_url = "https://www.civo.com/pricing"
    observation = _latest(
        [_receipt("civo", source_url, 2.99, "NVIDIA H100 SXM")]
    )["data"]["observations"][-1]
    collectors = {"civo": lambda **kwargs: kwargs}
    options = {
        "panel_config": panel,
        "observatory_config": observatory,
        "catalog": catalog,
        "collectors": collectors,
        "skip_reasons": {},
        "approved_urls": {"civo": [source_url]},
    }

    monkeypatch.setattr(collect_module, "collect_all", lambda *args, **kwargs: [])
    missing = collect_module._compare_observation(observation, **options)  # noqa: SLF001
    assert missing.checks[0].reason == "collector-result-missing"

    monkeypatch.setattr(
        collect_module,
        "collect_all",
        lambda *args, **kwargs: [
            {
                "source_id": "civo",
                "status": "ok",
                "url": "https://other.example.test/pricing",
                "observations": [],
            }
        ],
    )
    mismatched = collect_module._compare_observation(  # noqa: SLF001
        observation, **options
    )
    assert mismatched.checks[0].reason == "collector-source-url-mismatch"


def test_resolution_failures_are_typed_and_do_not_compare(
    collect_module, configs, monkeypatch
):
    panel, observatory, catalog = configs
    source_url = "https://www.civo.com/pricing"
    observation = _latest(
        [_receipt("civo", source_url, 2.99, "NVIDIA H100 SXM")]
    )["data"]["observations"][-1]
    result = {
        "source_id": "civo",
        "status": "ok",
        "url": source_url,
        "observations": [],
    }
    monkeypatch.setattr(
        collect_module, "collect_all", lambda *args, **kwargs: [result]
    )
    options = {
        "panel_config": panel,
        "observatory_config": observatory,
        "catalog": catalog,
        "collectors": {"civo": lambda **kwargs: kwargs},
        "skip_reasons": {},
        "approved_urls": {"civo": [source_url]},
    }

    original = collect_module._resolve_collected_price  # noqa: SLF001

    def fail_resolution(**kwargs):
        del kwargs
        raise ValueError("fixture resolution failed")

    monkeypatch.setattr(collect_module, "_resolve_collected_price", fail_resolution)
    failed = collect_module._compare_observation(observation, **options)  # noqa: SLF001
    assert failed.checks[0].reason == "resolution-error"

    monkeypatch.setattr(collect_module, "_resolve_collected_price", original)
    empty = collect_module._compare_observation(observation, **options)  # noqa: SLF001
    assert empty.checks[0].reason == "no-current-price"
    assert empty.exit_code == 2
