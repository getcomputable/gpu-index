# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Run observatory collectors serially under a hard wall-clock budget.

Same budget model as the basket capture (which this mirrors, not imports —
the basket's loop lives in its script, and the observatory's is module code
so it is unit-testable): urllib timeouts bound socket operations, not
requests, so the budget is the real fence. Each source gets
min(per-source deadline, remaining budget) enforced by joining a daemon
worker thread; once the budget is spent, remaining sources are recorded as
errors — visible holes, never a silently shorter list.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from gpu_index.common.http import TransportError

DEFAULT_CAPTURE_BUDGET_SECONDS = 1500.0
DEFAULT_PER_SOURCE_DEADLINE_SECONDS = 120.0
DEFAULT_PER_SOURCE_TIMEOUT_SECONDS = 30.0

# Failure classification (carry-forward design 2026-08-28): every error
# entry says WHICH WAY the source died, because the panel engine's
# carry-forward knob is scoped by failure class
# (calc.carry_forward_failure_kinds) and an error string is not a
# contract. The vocabulary is the capture lane's — basket.panel_config
# imports it, the QUARANTINE_REASON pattern.
FAILURE_KIND_FETCH = "fetch"  # the bytes never arrived whole (HTTP/socket/
#                               TLS error, non-https redirect, body cap,
#                               slow-drip guard)
FAILURE_KIND_PARSE = "parse"  # a 200 body arrived and the collector's
#                               fail-closed pins refused it
FAILURE_KIND_TIMEOUT = "timeout"  # deadline-abandoned; a fetch may still
#                                   be in flight (never retry these blind)
FAILURE_KIND_BUDGET = "budget"  # the capture budget was spent before the
#                                 source ran; nothing was attempted
VALID_FAILURE_KINDS = (
    FAILURE_KIND_BUDGET,
    FAILURE_KIND_FETCH,
    FAILURE_KIND_PARSE,
    FAILURE_KIND_TIMEOUT,
)


class DeadlineExceeded(RuntimeError):
    """A collector abandoned at its wall-clock deadline. RuntimeError
    subclass so every existing except/raise contract is unchanged; a
    dedicated type so failure classification never string-matches."""


def classify_failure(exc: BaseException) -> str:
    """Which way a collector died, from the exception TYPE — walking the
    explicit ``raise ... from`` cause chain.

    OSError covers the whole urllib fetch family (HTTPError and URLError
    subclass it) plus socket/TLS errors and socket-level timeouts (which
    therefore file as FETCH — the narrow 'timeout' kind is deadline
    abandonment only); TransportError is basket.http's transport-integrity
    refusal, also raised by collectors that wrap a multi-host fetch outage
    (shadeform). The __cause__ walk catches a future collector that wraps
    a transport failure in a domain exception via ``raise X from exc`` —
    without it the wrap files as 'parse' and a parse-only carry mint
    would re-cast a seat whose page merely stopped answering (Greptile
    P1, PR #75). Implicit __context__ is deliberately NOT walked: an
    error raised while HANDLING another says nothing about which layer
    refused. Everything else is the parse layer refusing a body that DID
    arrive — the fail-closed pins' RuntimeErrors, json/ValueError,
    KeyError on a reshaped payload."""
    seen = 0
    node: Optional[BaseException] = exc
    while node is not None and seen < 8:
        if isinstance(node, DeadlineExceeded):
            return FAILURE_KIND_TIMEOUT
        if isinstance(node, (OSError, TransportError)):
            return FAILURE_KIND_FETCH
        node = node.__cause__
        seen += 1
    return FAILURE_KIND_PARSE


def call_with_deadline(
    fn: Callable[..., Any],
    *,
    timeout: float,
    deadline: float,
    options: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run a collector with a HARD wall-clock bound.

    Between-source budget checks alone don't cap the source that is already
    in flight (a marketplace collector can make many sequential fetches).
    The worker is a daemon thread: an abandoned fetch dies with the
    process."""
    outcome: Dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn(timeout=timeout, options=options)
        except BaseException as exc:  # noqa: BLE001 — reported by the caller
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        raise DeadlineExceeded(
            f"source exceeded its {deadline:.0f}s budget share — abandoned mid-fetch"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def collect_all(
    config: Dict[str, Any],
    collectors: Mapping[str, Callable[..., Dict[str, Any]]],
    *,
    only: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """One result per configured source, in config order, always.

    A source with no registered collector records status 'unimplemented';
    a failed or budget-starved one records status 'error'. Per-source
    ``options`` from the config ride through to the collector as a shallow
    COPY — a collector must not be able to mutate the loaded config.

    Abandonment semantics (deliberate, bounded by the job timeout): when a
    collector exceeds its deadline, EVERY row it had already extracted is
    lost (its entry records status 'error' with empty observations — a
    partial multi-fetch record would be indistinguishable from a complete
    one), and its daemon worker thread may keep a fetch in flight until
    process exit, overlapping later collectors. The serial model is a
    politeness budget, not an isolation guarantee.
    """
    timeout = float(
        config.get("per_source_timeout_seconds", DEFAULT_PER_SOURCE_TIMEOUT_SECONDS)
    )
    budget = float(
        config.get("capture_budget_seconds", DEFAULT_CAPTURE_BUDGET_SECONDS)
    )
    per_source_deadline = float(
        config.get(
            "per_source_deadline_seconds", DEFAULT_PER_SOURCE_DEADLINE_SECONDS
        )
    )
    started_all = time.monotonic()
    results: List[Dict[str, Any]] = []
    for src in config["sources"]:
        sid = src["source_id"]
        if only and sid not in only:
            continue
        fn = collectors.get(sid)
        if fn is None:
            results.append(
                {
                    "source_id": sid,
                    "status": "unimplemented",
                    "error": "collector not implemented yet",
                    "observations": [],
                }
            )
            continue
        remaining = budget - (time.monotonic() - started_all)
        if remaining <= 0:
            results.append(
                {
                    "source_id": sid,
                    "status": "error",
                    "error": (
                        f"capture budget ({budget:.0f}s) exhausted before "
                        "this source ran"
                    ),
                    "failure_kind": FAILURE_KIND_BUDGET,
                    "observations": [],
                }
            )
            continue
        started = time.monotonic()
        try:
            options = src.get("options")
            result = call_with_deadline(
                fn,
                timeout=timeout,
                deadline=min(remaining, per_source_deadline),
                options=dict(options) if options is not None else None,
            )
            result["status"] = "ok"
            # The registry key is authoritative — a collector mislabeling
            # its own source_id must not relabel the snapshot entry.
            result["source_id"] = sid
        except Exception as exc:  # noqa: BLE001 — one dead feed must not kill the capture
            result = {
                "source_id": sid,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "failure_kind": classify_failure(exc),
                "observations": [],
            }
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
    return results
