#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Compare the latest public receipts with prices visible at their sources."""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_ROOT / "src"), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gpu_index.common.http as common_http  # noqa: E402
from gpu_index.common.bucket import (  # noqa: E402
    PUBLIC_BASE_URL_ENV,
    BucketConfig,
)
from gpu_index.index.panel import (  # noqa: E402
    compile_screens,
    member_eligible_rows,
    panel_calc_params,
    resolve_member_print,
)
from gpu_index.index.panel_config import load_panel_config  # noqa: E402
from gpu_index.observatory.catalog import load_sku_catalog  # noqa: E402
from gpu_index.observatory.collect import collect_all  # noqa: E402
from gpu_index.observatory.config import (  # noqa: E402
    load_observatory_config,
    resolve_catalog_path,
    sources_by_id,
)
from gpu_index.observatory.snapshot import normalize_observation  # noqa: E402
from gpu_index.observatory.sources import COLLECTORS  # noqa: E402
from gpu_index.published.reader import PublishedRecordReader  # noqa: E402

VERIFY_USER_AGENT = (
    "CGI-Verify/1.0 (+https://github.com/getcomputable/gpu-index)"
)
DEFAULT_PUBLIC_BASE_URL = "https://data.getcomputable.com"
PANEL_CONFIGS = {
    "B200": _ROOT / "config" / "index_panel_b200.json",
    "B300": _ROOT / "config" / "index_panel_b300.json",
    "H100": _ROOT / "config" / "index_panel_h100_sxm.json",
    "H200": _ROOT / "config" / "index_panel_h200_sxm.json",
}
PUBLIC_SKUS = tuple(sorted(PANEL_CONFIGS))

SAME = "SAME"
MOVED = "MOVED"
UNREACHABLE = "UNREACHABLE"
SKIPPED = "SKIPPED"

SKIP_KEY_REQUIRED = "api-key-required"
SKIP_RETRIEVAL_RESTRICTED = "retrieval-restricted"
VALID_POLICY_SKIP_REASONS = frozenset(
    {SKIP_KEY_REQUIRED, SKIP_RETRIEVAL_RESTRICTED}
)

# Add a source here only when its existing collector must not be invoked by
# an anonymous user-run check. An unregistered source also fails closed before
# any request because its receipt URL cannot be approved.
SOURCE_SKIP_REASONS: Mapping[str, str] = {}


class CollectionError(RuntimeError):
    """The latest print could not be selected or compared safely."""


def _one_line(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed > 0 else None


def _price(value: Decimal) -> str:
    return f"{value:.6f}"


@dataclass(frozen=True)
class SeatCheck:
    sku: str
    source_id: str
    state: str
    receipt_price: Optional[Decimal] = None
    collected_price: Optional[Decimal] = None
    reason: Optional[str] = None
    detail: Optional[str] = None

    def line(self) -> str:
        parts = [self.sku, f"source={self.source_id}", self.state]
        if self.receipt_price is not None:
            parts.append(f"receipt={_price(self.receipt_price)}")
        if self.collected_price is not None:
            parts.append(f"collected={_price(self.collected_price)}")
        if self.reason is not None:
            parts.append(f"reason={self.reason}")
        if self.detail:
            parts.append(f"detail={self.detail!r}")
        if self.state == MOVED:
            parts.append("price changed since capture")
        return " ".join(parts)


@dataclass(frozen=True)
class CollectionReport:
    sku: str
    observed_at: str
    methodology_id: str
    checks: tuple[SeatCheck, ...]

    @property
    def comparison_count(self) -> int:
        return sum(check.state in (SAME, MOVED) for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.comparison_count else 2

    def lines(self) -> list[str]:
        counts = Counter(check.state for check in self.checks)
        return [
            (
                f"latest print: {self.sku} {self.observed_at} "
                f"({self.methodology_id})"
            ),
            *(check.line() for check in self.checks),
            (
                f"summary: {len(self.checks)} seat(s): "
                f"{counts[SAME]} SAME, {counts[MOVED]} MOVED, "
                f"{counts[UNREACHABLE]} UNREACHABLE, "
                f"{counts[SKIPPED]} SKIPPED"
            ),
        ]


@contextmanager
def _verifier_identity():
    with common_http.user_agent_scope(VERIFY_USER_AGENT):
        yield


def _select_latest_observation(envelope: Optional[dict], sku: str) -> dict:
    if envelope is None:
        raise CollectionError("the public front has no latest.json")
    data = envelope.get("data") or {}
    versions = data.get("versions") or []
    pointers = [entry for entry in versions if entry.get("sku") == sku]
    if len(pointers) != 1:
        raise CollectionError(
            f"latest.json has {len(pointers)} current-version pointers for {sku}"
        )
    methodology = pointers[0].get("methodology_id")
    matches = [
        observation
        for observation in (data.get("observations") or [])
        if observation.get("sku") == sku
        and observation.get("methodology_id") == methodology
    ]
    if len(matches) != 1:
        raise CollectionError(
            f"latest.json has {len(matches)} {sku} prints for current "
            f"methodology {methodology!r}"
        )
    return matches[0]


def _https_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.startswith("https://"):
            yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _https_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _https_values(item)


def _declared_urls(
    collector: Callable[..., Any], source_config: Mapping[str, Any]
) -> frozenset[str]:
    module = importlib.import_module(collector.__module__)
    values: list[Any] = []
    for name in ("URL", "URL_BASE", "URLS"):
        if hasattr(module, name):
            values.append(getattr(module, name))
    values.append(source_config.get("options") or {})
    return frozenset(url for value in values for url in _https_values(value))


def _fx_records(receipt: Mapping[str, Any], observed_at: str) -> dict:
    if receipt.get("currency") != "EUR":
        return {}
    rate = _decimal(receipt.get("fx_rate"))
    if rate is None:
        return {}
    return {observed_at[:10]: {"rates": {"USD": float(rate)}}}


def _resolve_collected_price(
    *,
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
    catalog: Mapping[str, Any],
    params: Mapping[str, Any],
    screens: Mapping[str, Any],
    observed_at: str,
) -> Optional[Decimal]:
    source_id = str(receipt["source_id"])
    member = screens["members"].get(source_id)
    if member is None:
        return None
    entry = {
        "source_id": source_id,
        "status": "ok",
        "observations": [
            normalize_observation(row, catalog)
            for row in (result.get("observations") or [])
        ],
        "book_stats": result.get("book_stats"),
    }
    rows = member_eligible_rows(
        entry,
        skus=member["skus"],
        reject_patterns=screens["reject"],
        require_patterns=member["require"],
        eligible_tiers=params["eligible_tiers"],
        extra_require=member["extra_require"],
    )
    chosen = resolve_member_print(
        rows,
        source_entry=entry,
        statistic=member["statistic"],
        statistic_params=params["statistic_params"],
        obs_date=observed_at[:10],
        fx_records=_fx_records(receipt, observed_at),
        fx_max_staleness_days=params["fx_max_staleness_days"],
    )
    if not chosen or chosen.get("held_out") or chosen.get("fx_unavailable"):
        return None
    return _decimal(chosen.get("usd_per_gpu_hr"))


def _skip(
    sku: str,
    source_id: str,
    reason: str,
    *,
    receipt_price: Optional[Decimal] = None,
) -> SeatCheck:
    return SeatCheck(
        sku=sku,
        source_id=source_id,
        state=SKIPPED,
        receipt_price=receipt_price,
        reason=reason,
    )


def _compare_observation(
    observation: Mapping[str, Any],
    *,
    panel_config: Mapping[str, Any],
    observatory_config: Mapping[str, Any],
    catalog: Mapping[str, Any],
    collectors: Mapping[str, Callable[..., dict]],
    skip_reasons: Mapping[str, str],
    approved_urls: Optional[Mapping[str, Sequence[str]]] = None,
) -> CollectionReport:
    sku = str(observation["sku"])
    methodology = str(observation["methodology_id"])
    if panel_config["calc"]["methodology_id"] != methodology:
        raise CollectionError(
            f"this checkout has {panel_config['calc']['methodology_id']!r} for "
            f"{sku}, but the public front points to {methodology!r}; update the "
            "checkout before collecting"
        )
    invalid_reasons = sorted(set(skip_reasons.values()) - VALID_POLICY_SKIP_REASONS)
    if invalid_reasons:
        raise CollectionError(f"unknown source skip reasons: {invalid_reasons}")

    receipts = list(observation.get("receipts") or [])
    source_configs = sources_by_id(dict(observatory_config))
    checks: list[Optional[SeatCheck]] = [None] * len(receipts)
    runnable: set[str] = set()

    for index, receipt in enumerate(receipts):
        source_id = str(receipt.get("source_id") or "unknown")
        receipt_price = _decimal(receipt.get("price"))
        if receipt_price is None:
            checks[index] = _skip(sku, source_id, "receipt-price-unavailable")
            continue
        policy_reason = skip_reasons.get(source_id)
        if policy_reason is not None:
            checks[index] = _skip(
                sku, source_id, policy_reason, receipt_price=receipt_price
            )
            continue
        collector = collectors.get(source_id)
        if collector is None:
            checks[index] = _skip(
                sku,
                source_id,
                "collector-unavailable",
                receipt_price=receipt_price,
            )
            continue
        source_config = source_configs.get(source_id)
        if source_config is None:
            checks[index] = _skip(
                sku,
                source_id,
                "source-configuration-unavailable",
                receipt_price=receipt_price,
            )
            continue
        source_url = receipt.get("source_url")
        declared = (
            frozenset(approved_urls.get(source_id, ()))
            if approved_urls is not None
            else _declared_urls(collector, source_config)
        )
        if not isinstance(source_url, str) or source_url not in declared:
            checks[index] = _skip(
                sku,
                source_id,
                "receipt-source-url-not-approved",
                receipt_price=receipt_price,
            )
            continue
        runnable.add(source_id)

    collected = (
        collect_all(dict(observatory_config), collectors, only=runnable)
        if runnable
        else []
    )
    results = {
        result["source_id"]: result
        for result in collected
    }
    params = panel_calc_params(dict(panel_config))
    screens = compile_screens(params)
    observed_at = str(observation["observed_at"])

    for index, receipt in enumerate(receipts):
        if checks[index] is not None:
            continue
        source_id = str(receipt["source_id"])
        receipt_price = _decimal(receipt.get("price"))
        result = results.get(source_id)
        if result is None:
            checks[index] = SeatCheck(
                sku,
                source_id,
                UNREACHABLE,
                receipt_price=receipt_price,
                reason="collector-result-missing",
            )
            continue
        if result.get("status") != "ok":
            checks[index] = SeatCheck(
                sku,
                source_id,
                UNREACHABLE,
                receipt_price=receipt_price,
                reason=str(result.get("failure_kind") or "collector-error"),
                detail=_one_line(result.get("error") or result.get("status")),
            )
            continue
        if result.get("url") != receipt.get("source_url"):
            checks[index] = SeatCheck(
                sku,
                source_id,
                UNREACHABLE,
                receipt_price=receipt_price,
                reason="collector-source-url-mismatch",
                detail=_one_line(result.get("url")),
            )
            continue
        try:
            collected_price = _resolve_collected_price(
                receipt=receipt,
                result=result,
                catalog=catalog,
                params=params,
                screens=screens,
                observed_at=observed_at,
            )
        except Exception as exc:  # one source failure must not hide other receipts
            checks[index] = SeatCheck(
                sku,
                source_id,
                UNREACHABLE,
                receipt_price=receipt_price,
                reason="resolution-error",
                detail=_one_line(f"{type(exc).__name__}: {exc}"),
            )
            continue
        if collected_price is None:
            checks[index] = SeatCheck(
                sku,
                source_id,
                UNREACHABLE,
                receipt_price=receipt_price,
                reason="no-current-price",
            )
            continue
        state = SAME if collected_price == receipt_price else MOVED
        checks[index] = SeatCheck(
            sku,
            source_id,
            state,
            receipt_price=receipt_price,
            collected_price=collected_price,
        )

    if any(check is None for check in checks):
        raise CollectionError("a receipt produced no report line")
    return CollectionReport(
        sku=sku,
        observed_at=observed_at,
        methodology_id=methodology,
        checks=tuple(check for check in checks if check is not None),
    )


def collect_latest_receipts(
    sku: str,
    *,
    reader_factory: Optional[Callable[[], Any]] = None,
    panel_config: Optional[Mapping[str, Any]] = None,
    observatory_config: Optional[Mapping[str, Any]] = None,
    catalog: Optional[Mapping[str, Any]] = None,
    collectors: Optional[Mapping[str, Callable[..., dict]]] = None,
    skip_reasons: Optional[Mapping[str, str]] = None,
    approved_urls: Optional[Mapping[str, Sequence[str]]] = None,
) -> CollectionReport:
    """Collect and compare every receipt in one SKU's current public print."""
    normalized_sku = sku.upper()
    if normalized_sku not in PUBLIC_SKUS:
        raise CollectionError(
            f"unknown sku {sku!r} (expected b200, b300, h100, or h200)"
        )
    loaded_panel = (
        load_panel_config(PANEL_CONFIGS[normalized_sku])
        if panel_config is None
        else panel_config
    )
    loaded_observatory = (
        load_observatory_config()
        if observatory_config is None
        else observatory_config
    )
    loaded_catalog = (
        load_sku_catalog(resolve_catalog_path(dict(loaded_observatory)))
        if catalog is None
        else catalog
    )
    active_collectors = COLLECTORS if collectors is None else collectors
    active_skips = SOURCE_SKIP_REASONS if skip_reasons is None else skip_reasons

    with _verifier_identity():
        if reader_factory is None:
            config = BucketConfig.from_env(
                {PUBLIC_BASE_URL_ENV: DEFAULT_PUBLIC_BASE_URL}
            )
            reader = PublishedRecordReader(config)
        else:
            reader = reader_factory()
        latest = reader.read_latest()
        observation = _select_latest_observation(latest, normalized_sku)
        return _compare_observation(
            observation,
            panel_config=loaded_panel,
            observatory_config=loaded_observatory,
            catalog=loaded_catalog,
            collectors=active_collectors,
            skip_reasons=active_skips,
            approved_urls=approved_urls,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="collect current source prices for the latest public print"
    )
    parser.add_argument("--sku", required=True, choices=[s.lower() for s in PUBLIC_SKUS])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = collect_latest_receipts(args.sku)
    except Exception as exc:
        print(f"collect failed: {_one_line(f'{type(exc).__name__}: {exc}')}", file=sys.stderr)
        return 2
    for line in report.lines():
        print(line)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
