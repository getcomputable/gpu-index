# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Object-store transport for the index lanes: local directory, S3, or
anonymous public HTTPS.

Three backends behind one get/put/list surface:

  - **Local directory** (the default): objects live as plain files under
    ``GPU_INDEX_DATA_DIR`` (default ``./data``), key == relative path.
    This is the backend for working against a downloaded copy of the
    published record, and for development.
  - **S3-compatible bucket** (optional, requires the ``publish`` extra):
    selected when ``GPU_INDEX_S3_ENDPOINT`` is set, with
    ``GPU_INDEX_S3_BUCKET`` / ``GPU_INDEX_S3_ACCESS_KEY`` /
    ``GPU_INDEX_S3_SECRET_KEY`` (and optional ``GPU_INDEX_S3_REGION``,
    default ``auto``).
  - **Public HTTPS, anonymous read-only**: selected when
    ``GPU_INDEX_PUBLIC_BASE_URL`` is set — plain httpx GETs of
    ``<base_url>/<key>`` against the published record's public front.
    Mutually exclusive with the S3 backend (setting both is a config
    error, never a silent pick). Every write REFUSES loudly, and so does
    key listing: a bare HTTPS object front has no list endpoint, and
    returning ``[]`` would make missing history look like recorded
    absence.

The append-only + verify-after-write + pointer-moved-last discipline lives
one layer up (``gpu_index.common.store``); everything here is transport.
S3 client tuning notes: botocore >= 1.36 enables default integrity
checksums (CRC + aws-chunked) that some S3-compatible gateways reject, so
request/response checksum calculation is pinned to ``when_required`` and
addressing is path-style. Missing keys are matched on HTTP 404 as well as
the S3 error code because some gateways return an empty error code.
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

DEFAULT_DATA_DIR = "data"

PUBLIC_BASE_URL_ENV = "GPU_INDEX_PUBLIC_BASE_URL"

S3_REQUIRED_ENV = (
    "GPU_INDEX_S3_ENDPOINT",
    "GPU_INDEX_S3_BUCKET",
    "GPU_INDEX_S3_ACCESS_KEY",
    "GPU_INDEX_S3_SECRET_KEY",
)


class BucketPublishError(RuntimeError):
    """A store publish failed a guard or transport step."""


@dataclass(frozen=True)
class BucketConfig:
    """Resolved backend selection.

    ``backend`` is ``"local"``, ``"s3"``, or ``"public"``. For the local
    and public backends, ``bucket`` is a label only (keys resolve under
    ``local_root`` / ``public_base_url``).
    """

    backend: str
    bucket: str
    local_root: Optional[Path] = None
    endpoint: Optional[str] = None
    region: str = "auto"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    public_base_url: Optional[str] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "BucketConfig":
        env = os.environ if env is None else env
        if env.get(PUBLIC_BASE_URL_ENV):
            if env.get("GPU_INDEX_S3_ENDPOINT"):
                # Two read backends configured at once — refuse rather
                # than silently prefer one: whichever the operator MEANT,
                # half their env is now dead config.
                raise BucketPublishError(
                    f"{PUBLIC_BASE_URL_ENV} and GPU_INDEX_S3_ENDPOINT are "
                    "both set — the public HTTPS backend and the S3 "
                    "backend are mutually exclusive; unset one"
                )
            base_url = env[PUBLIC_BASE_URL_ENV].strip()
            if urllib.parse.urlsplit(base_url).scheme != "https":
                # Same posture as the collectors' transport: the published
                # record is audit evidence and never rides plaintext.
                raise BucketPublishError(
                    f"{PUBLIC_BASE_URL_ENV} must be an https:// URL, "
                    f"got {base_url!r}"
                )
            return cls(
                backend="public", bucket="public", public_base_url=base_url
            )
        if env.get("GPU_INDEX_S3_ENDPOINT"):
            # CI systems expand missing secrets to "" — treat empty as missing.
            missing = [name for name in S3_REQUIRED_ENV if not env.get(name)]
            if missing:
                raise BucketPublishError(
                    f"S3 credentials missing/empty: {missing} — either export "
                    "the full GPU_INDEX_S3_* set or unset GPU_INDEX_S3_ENDPOINT "
                    "to use the local-directory backend"
                )
            return cls(
                backend="s3",
                bucket=env["GPU_INDEX_S3_BUCKET"],
                endpoint=env["GPU_INDEX_S3_ENDPOINT"],
                region=env.get("GPU_INDEX_S3_REGION") or "auto",
                access_key_id=env["GPU_INDEX_S3_ACCESS_KEY"],
                secret_access_key=env["GPU_INDEX_S3_SECRET_KEY"],
            )
        root = Path(env.get("GPU_INDEX_DATA_DIR") or DEFAULT_DATA_DIR)
        return cls(backend="local", bucket="local", local_root=root)


class LocalStore:
    """Filesystem backend exposing the same object surface as the S3 path.

    Keys are POSIX-style relative paths under ``root``. Writes are atomic
    (temp file + rename) so a crashed writer never leaves a truncated
    object where a verify-after-write read would find it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise BucketPublishError(f"refusing unsafe object key {key!r}")
        return self.root / key

    def get_bytes(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def list_keys(self, prefix: str) -> List[str]:
        # Keys are string-prefix matched (an S3 prefix need not end on a
        # path separator — e.g. ".../slot16-"), so walk the deepest
        # directory the prefix pins down and filter.
        dir_part = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
        start = self.root / dir_part if dir_part else self.root
        if not start.is_dir():
            return []
        keys: List[str] = []
        for path in start.rglob("*"):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                keys.append(key)
        return keys

    def delete(self, key: str) -> None:
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class PublicReadStore:
    """Anonymous read-only HTTPS backend over the published record.

    Same get-object semantics as the other backends — missing keys are
    ``None`` (matched on HTTP 404), any other non-200 raises — via plain
    httpx GETs carrying the collectors' honest CGI User-Agent, with the
    same response-byte ceiling ``gpu_index.common.http`` applies (a broken
    or hostile front cannot balloon a reader's memory). Redirects are NOT
    followed: a public object front serving the record redirects nothing,
    and following one silently could downgrade transport or swap hosts —
    a 3xx surfaces as an error instead.

    Read-only is structural: ``put_bytes``/``delete`` (and ``list_keys``,
    which a bare HTTPS front cannot answer) refuse loudly.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client=None,
        max_bytes: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        from gpu_index.common.http import (
            DEFAULT_TIMEOUT,
            MAX_RESPONSE_BYTES,
            current_user_agent,
        )

        self.base_url = base_url.rstrip("/")
        self.max_bytes = MAX_RESPONSE_BYTES if max_bytes is None else int(max_bytes)
        self._timeout = DEFAULT_TIMEOUT if timeout is None else float(timeout)
        self._user_agent = current_user_agent()
        self._client = client  # injectable for tests (httpx mock transports)

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
                follow_redirects=False,
            )
        return self._client

    def _url(self, key: str) -> str:
        if key.startswith("/") or ".." in key.split("/"):
            raise BucketPublishError(f"refusing unsafe object key {key!r}")
        return f"{self.base_url}/{urllib.parse.quote(key, safe='/')}"

    def get_bytes(self, key: str) -> Optional[bytes]:
        url = self._url(key)
        with self._http().stream("GET", url) as response:
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                raise BucketPublishError(
                    f"public read backend: GET {url} returned "
                    f"HTTP {response.status_code}"
                )
            chunks: List[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > self.max_bytes:
                    raise BucketPublishError(
                        f"public read backend: {url} body exceeds "
                        f"{self.max_bytes} bytes — refusing"
                    )
                chunks.append(chunk)
            return b"".join(chunks)

    def _refuse(self, operation: str) -> None:
        raise BucketPublishError(
            f"public read backend is READ-ONLY: refusing {operation} — "
            "publishing requires the S3 backend (GPU_INDEX_S3_*) or a "
            "local data dir (GPU_INDEX_DATA_DIR)"
        )

    def put_bytes(self, key: str, data: bytes) -> None:
        self._refuse(f"put of {key!r}")

    def delete(self, key: str) -> None:
        self._refuse(f"delete of {key!r}")

    def list_keys(self, prefix: str) -> List[str]:
        # NOT a write, but refused for the same fail-loud reason: a bare
        # HTTPS object front has no list endpoint, and returning [] would
        # make unreadable history look like recorded absence.
        raise BucketPublishError(
            "public read backend cannot list keys (no listing endpoint "
            f"for prefix {prefix!r}) — use the S3 backend or a downloaded "
            "copy of the record for prefix scans"
        )


def make_client(config: BucketConfig):
    """Client for the configured backend: a ``LocalStore``, a
    ``PublicReadStore``, or a boto3 S3 client tuned for S3-compatible
    gateways (see the module docstring)."""
    if config.backend == "local":
        if config.local_root is None:
            raise BucketPublishError("local backend selected without a data dir")
        return LocalStore(config.local_root)
    if config.backend == "public":
        if not config.public_base_url:
            raise BucketPublishError("public backend selected without a base URL")
        return PublicReadStore(config.public_base_url)
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=config.endpoint,
        region_name=config.region,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        config=BotoConfig(
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def get_object_bytes(client, bucket: str, key: str) -> Optional[bytes]:
    if isinstance(client, (LocalStore, PublicReadStore)):
        return client.get_bytes(key)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # boto surfaces missing keys as ClientError
        error_response = getattr(exc, "response", {}) or {}
        code = error_response.get("Error", {}).get("Code", "")
        status = error_response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        # Some S3-compatible gateways 404 missing keys with an EMPTY error
        # code — match on HTTP status as well as the S3 code.
        if code in ("NoSuchKey", "404", "NotFound") or status == 404:
            return None
        raise
    return response["Body"].read()


def list_object_keys(client, bucket: str, prefix: str) -> list:
    """All keys under a prefix (paginated on the S3 path)."""
    if isinstance(client, (LocalStore, PublicReadStore)):
        return client.list_keys(prefix)
    keys: list = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        keys.extend(entry["Key"] for entry in response.get("Contents", []))
        if not response.get("IsTruncated"):
            return keys
        token = response.get("NextContinuationToken")


def put_bytes(
    client,
    bucket: str,
    key: str,
    data: bytes,
    *,
    content_type: str,
    cache_control: Optional[str] = None,
) -> None:
    if isinstance(client, (LocalStore, PublicReadStore)):
        # content_type / cache_control are HTTP-serving concerns; a local
        # file has neither. (The public backend REFUSES here — read-only.)
        client.put_bytes(key, data)
        return
    kwargs = {
        "Bucket": bucket,
        "Key": key,
        "Body": data,
        "ContentType": content_type,
    }
    if cache_control:
        kwargs["CacheControl"] = cache_control
    client.put_object(**kwargs)


def put_json_bytes(
    client, bucket: str, key: str, data: bytes, *, cache_control: Optional[str] = None
) -> None:
    put_bytes(
        client,
        bucket,
        key,
        data,
        content_type="application/json",
        cache_control=cache_control,
    )
