# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Object-store transport for the index lanes: local directory or S3.

Two backends behind one get/put/list surface:

  - **Local directory** (the default): objects live as plain files under
    ``GPU_INDEX_DATA_DIR`` (default ``./data``), key == relative path.
    This is the backend for working against a downloaded copy of the
    published record, and for development.
  - **S3-compatible bucket** (optional, requires the ``publish`` extra):
    selected when ``GPU_INDEX_S3_ENDPOINT`` is set, with
    ``GPU_INDEX_S3_BUCKET`` / ``GPU_INDEX_S3_ACCESS_KEY`` /
    ``GPU_INDEX_S3_SECRET_KEY`` (and optional ``GPU_INDEX_S3_REGION``,
    default ``auto``).

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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

DEFAULT_DATA_DIR = "data"

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

    ``backend`` is ``"local"`` or ``"s3"``. For the local backend,
    ``bucket`` is a label only (keys resolve under ``local_root``).
    """

    backend: str
    bucket: str
    local_root: Optional[Path] = None
    endpoint: Optional[str] = None
    region: str = "auto"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "BucketConfig":
        env = os.environ if env is None else env
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


def make_client(config: BucketConfig):
    """Client for the configured backend: a ``LocalStore`` or a boto3 S3
    client tuned for S3-compatible gateways (see the module docstring)."""
    if config.backend == "local":
        if config.local_root is None:
            raise BucketPublishError("local backend selected without a data dir")
        return LocalStore(config.local_root)
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
    if isinstance(client, LocalStore):
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
    if isinstance(client, LocalStore):
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
    if isinstance(client, LocalStore):
        # content_type / cache_control are HTTP-serving concerns; a local
        # file has neither.
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
