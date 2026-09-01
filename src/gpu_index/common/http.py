# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Stdlib HTTP fetch + sku normalization for basket collectors.

These captures are audit evidence for a financial index, so transport
integrity is part of the contract:
  - redirects may only land on https — a downgraded hop surfaces as a source
    error, never a silent plaintext fetch;
  - response bodies are size-capped — a malicious/broken provider cannot
    balloon the job's memory or hold it open streaming.
"""

from __future__ import annotations

import contextvars
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from typing import Iterator, Optional

# One identity across every repository that collects for this index. The
# +URL is getcomputable.com and NOT this repository: a repo URL stops
# resolving if the repo is renamed or made private, and a 404 is no better
# than no contact at all.
UA = (
    "CGI-Collector/1.0 (+https://getcomputable.com; "
    "team@getcomputable.com)"
)
_USER_AGENT_OVERRIDE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "gpu_index_user_agent_override", default=None
)


def current_user_agent() -> str:
    """Return the request identity installed for the current execution context."""
    return _USER_AGENT_OVERRIDE.get() or UA


@contextmanager
def user_agent_scope(value: str) -> Iterator[None]:
    """Temporarily identify requests from this execution context as ``value``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("user agent must be a non-empty string")
    token = _USER_AGENT_OVERRIDE.set(value)
    try:
        yield
    finally:
        _USER_AGENT_OVERRIDE.reset(token)

DEFAULT_TIMEOUT = 30.0
# urllib's timeout bounds each socket OPERATION (connect / one recv), not the
# whole request — a provider trickling one chunk per 29s could hold a fetch
# open for many minutes. The wall-clock limit bounds the full body read.
DEFAULT_WALL_CLOCK_LIMIT = 90.0
# Largest legitimate page today is shadeform's ~700KB homepage; 8MB leaves
# an order of magnitude of headroom without letting a hostile body OOM the
# capture process.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_CHUNK = 1 << 16


class TransportError(RuntimeError):
    """A transport-integrity refusal (body cap, slow-drip guard): the bytes
    never arrived whole, so downstream failure classification must file it
    with the fetch failures, never the parse failures. RuntimeError subclass
    so every existing except/raise contract is unchanged."""


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Follow 308s like 302s (urllib skips 308 by default), and refuse any
    redirect target that is not https."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.HTTPError(
                newurl, code, "non-https redirect refused", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_308(self, req, fp, code, msg, headers):  # noqa: ANN001
        return self.http_error_302(req, fp, code, msg, headers)


# urllib verifies against the SYSTEM CA store by default, which can lack
# chains that certifi's bundle carries (observed live for www.ecb.europa.eu:
# CERTIFICATE_VERIFY_FAILED on a minimal container image). Prefer certifi
# when importable; verification itself is NEVER weakened.
try:
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover — fall back to the system CA store
    _SSL_CONTEXT = ssl.create_default_context()

_OPENER = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=_SSL_CONTEXT), _HttpsOnlyRedirect
)


def read_body_capped(
    resp,
    limit: int = MAX_RESPONSE_BYTES,
    wall_clock_limit: float = DEFAULT_WALL_CLOCK_LIMIT,
) -> bytes:
    """Read a response streamwise, refusing bodies past ``limit`` bytes or
    ``wall_clock_limit`` seconds (slow-drip guard)."""
    deadline = time.monotonic() + wall_clock_limit
    chunks = []
    total = 0
    while True:
        chunk = resp.read(_CHUNK)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise TransportError(
                f"response body exceeds {limit} bytes — refusing"
            )
        if time.monotonic() > deadline:
            raise TransportError(
                f"response body still streaming after {wall_clock_limit}s — refusing"
            )
        chunks.append(chunk)


def fetch(
    url: str,
    data: Optional[bytes] = None,
    headers: Optional[dict] = None,
    timeout: float = DEFAULT_TIMEOUT,
    wall_clock_limit: float = DEFAULT_WALL_CLOCK_LIMIT,
) -> str:
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": current_user_agent(), **(headers or {})},
    )
    with _OPENER.open(req, timeout=timeout) as resp:
        return read_body_capped(resp, wall_clock_limit=wall_clock_limit).decode(
            "utf-8", errors="replace"
        )


def norm_sku(name: Optional[str]) -> Optional[str]:
    """Map a marketing label to a bare sku; GB200/GB300 superchips are NOT
    B200/B300 and normalize to None."""
    n = (name or "").upper()
    if re.search(r"GB[23]00", n):
        return None
    for sku in ("B300", "B200", "H200", "H100"):
        if sku in n:
            return sku
    return None
