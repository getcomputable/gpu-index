# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Anonymous public HTTPS read backend (GPU_INDEX_PUBLIC_BASE_URL).

Covers the whole contract: env selection (and the mutual-exclusion /
https-only refusals), get-object semantics on an httpx mock transport
(200 bytes, 404 -> None, other statuses raise, redirects not followed),
the response-byte ceiling, the honest CGI User-Agent, and the structural
read-only posture (every write and the un-answerable list refuse loudly).
"""

from __future__ import annotations

import httpx
import pytest

from gpu_index.common.bucket import (
    BucketConfig,
    BucketPublishError,
    PublicReadStore,
    get_object_bytes,
    list_object_keys,
    make_client,
    put_json_bytes,
)
from gpu_index.common.http import MAX_RESPONSE_BYTES, UA

BASE_URL = "https://record.example.com/cgi"


def _store(handler, **kwargs) -> PublicReadStore:
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": UA},
        follow_redirects=False,
    )
    return PublicReadStore(BASE_URL, client=client, **kwargs)


# ------------------------------------------------------------ env selection


def test_from_env_selects_public_backend():
    config = BucketConfig.from_env({"GPU_INDEX_PUBLIC_BASE_URL": BASE_URL})
    assert config.backend == "public"
    assert config.public_base_url == BASE_URL
    assert isinstance(make_client(config), PublicReadStore)


def test_from_env_public_and_s3_together_refuse():
    with pytest.raises(BucketPublishError, match="mutually exclusive"):
        BucketConfig.from_env(
            {
                "GPU_INDEX_PUBLIC_BASE_URL": BASE_URL,
                "GPU_INDEX_S3_ENDPOINT": "https://gateway.example.com",
            }
        )


def test_from_env_public_requires_https():
    with pytest.raises(BucketPublishError, match="https"):
        BucketConfig.from_env(
            {"GPU_INDEX_PUBLIC_BASE_URL": "http://record.example.com/cgi"}
        )


def test_from_env_without_public_url_keeps_local_default(tmp_path):
    config = BucketConfig.from_env({"GPU_INDEX_DATA_DIR": str(tmp_path)})
    assert config.backend == "local"


# ------------------------------------------------------- get-object reads


def test_get_bytes_returns_body_with_cgi_user_agent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["ua"] = request.headers.get("User-Agent")
        return httpx.Response(200, content=b'{"ok": true}\n')

    store = _store(handler)
    body = store.get_bytes("index/basket/composites/m1/2026-08-16.json")
    assert body == b'{"ok": true}\n'
    assert seen["url"] == (
        f"{BASE_URL}/index/basket/composites/m1/2026-08-16.json"
    )
    assert seen["ua"] == UA
    assert "CGI-Collector/" in seen["ua"]
    # No browser prefix: the UA is a public identity, and the bot
    # protection on the collected hosts screens an unidentified client,
    # not the absence of a browser string.
    assert not seen["ua"].lower().startswith("mozilla/")


def test_get_bytes_missing_key_is_none_on_404():
    store = _store(lambda request: httpx.Response(404, content=b"not here"))
    assert store.get_bytes("index/basket/latest.json") is None


@pytest.mark.parametrize("status", [301, 302, 403, 500, 503])
def test_get_bytes_raises_on_non_200_non_404(status):
    store = _store(lambda request: httpx.Response(status))
    with pytest.raises(BucketPublishError, match=f"HTTP {status}"):
        store.get_bytes("index/basket/latest.json")


def test_get_bytes_enforces_the_byte_ceiling():
    store = _store(
        lambda request: httpx.Response(200, content=b"x" * 64), max_bytes=63
    )
    with pytest.raises(BucketPublishError, match="exceeds 63 bytes"):
        store.get_bytes("index/basket/latest.json")


def test_default_ceiling_matches_the_collector_transport():
    assert PublicReadStore(BASE_URL).max_bytes == MAX_RESPONSE_BYTES


def test_get_bytes_at_the_ceiling_is_accepted():
    store = _store(
        lambda request: httpx.Response(200, content=b"x" * 63), max_bytes=63
    )
    assert store.get_bytes("index/basket/latest.json") == b"x" * 63


@pytest.mark.parametrize(
    "key", ["/etc/passwd", "a/../../secrets.json", "../up.json"]
)
def test_unsafe_keys_refused_before_any_request(key):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        pytest.fail("unsafe key must never reach the transport")

    with pytest.raises(BucketPublishError, match="unsafe object key"):
        _store(handler).get_bytes(key)


def test_module_helper_get_object_bytes_dispatches():
    store = _store(lambda request: httpx.Response(200, content=b"body"))
    assert get_object_bytes(store, "public", "index/basket/latest.json") == b"body"


# ---------------------------------------------------- read-only structure


def _refusing_store() -> PublicReadStore:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        pytest.fail("a refused operation must never reach the transport")

    return _store(handler)


def test_put_refuses_with_a_clear_error():
    with pytest.raises(BucketPublishError, match="READ-ONLY"):
        _refusing_store().put_bytes("index/basket/latest.json", b"{}")


def test_delete_refuses_with_a_clear_error():
    with pytest.raises(BucketPublishError, match="READ-ONLY"):
        _refusing_store().delete("index/basket/latest.json")


def test_module_helper_put_json_bytes_refuses():
    with pytest.raises(BucketPublishError, match="READ-ONLY"):
        put_json_bytes(
            _refusing_store(), "public", "index/basket/latest.json", b"{}"
        )


def test_list_refuses_loudly_instead_of_faking_empty():
    with pytest.raises(BucketPublishError, match="cannot list keys"):
        _refusing_store().list_keys("index/basket/snapshots/")


def test_module_helper_list_object_keys_refuses():
    with pytest.raises(BucketPublishError, match="cannot list keys"):
        list_object_keys(_refusing_store(), "public", "index/basket/")
