# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Store-backed reader for the published record.

Rides the shared transport (``gpu_index.common.bucket``): a local
directory holding a downloaded copy of the record (``GPU_INDEX_DATA_DIR``,
default ``./data``), the anonymous public HTTPS front
(``GPU_INDEX_PUBLIC_BASE_URL``), or an S3-compatible bucket. Every read
returns a digest-verified envelope (``decode_and_verify_artifact``) —
there is no unverified read path.

``latest.json`` selects the current per-SKU integer version. Day and series
reads resolve through that pointer, while an explicit version can address a
prior frozen keyspace. During the pointer-last migration, a legacy pointer
keeps resolving the former flat paths; explicit versions can verify staged
objects before the pointer moves. A missing day remains an ordinary state —
not yet published, or absent from whichever copy of the record this reader is
pointed at — and reads return ``None`` for it rather than raising.
"""

from __future__ import annotations

from typing import Optional

from gpu_index.common.bucket import (
    BucketConfig,
    get_object_bytes,
    make_client,
)
from gpu_index.published.artifacts import (
    PublishedRecordError,
    day_key,
    decode_and_verify_artifact,
    latest_key,
    series_key,
)


class PublishedRecordReader:
    """Digest-verifying reader over the published layout."""

    def __init__(self, config: Optional[BucketConfig] = None) -> None:
        self.config = BucketConfig.from_env() if config is None else config
        self._client = make_client(self.config)

    def describe(self) -> str:
        """Human-readable source label for CLI banners."""
        if self.config.backend == "public":
            return f"public HTTPS front {self.config.public_base_url}"
        if self.config.backend == "local":
            return f"local record copy {self.config.local_root}"
        return f"s3 bucket {self.config.bucket}"

    def _read(self, key: str) -> Optional[dict]:
        raw = get_object_bytes(self._client, self.config.bucket, key)
        if raw is None:
            return None
        return decode_and_verify_artifact(raw)

    def read_latest(self) -> Optional[dict]:
        """``latest.json``: the newest observation per lane."""
        return self._read(latest_key())

    def resolve_day_key(
        self, date: str, *, sku: str, version: int | None = None
    ) -> str:
        """Resolve a SKU day through the pointer, or retain the flat key."""
        resolved = self._resolve_version(sku, version)
        if resolved is None:
            return day_key(date)
        return day_key(date, sku=sku, version=resolved)

    def read_day(
        self,
        date: str,
        *,
        sku: str | None = None,
        version: int | None = None,
    ) -> Optional[dict]:
        """Read one UTC day, resolving a SKU through ``latest.json``."""
        if sku is None:
            if version is not None:
                raise PublishedRecordError(
                    "reading an explicit version requires a SKU"
                )
            return self._read(day_key(date))
        return self._read(self.resolve_day_key(date, sku=sku, version=version))

    def resolve_series_key(
        self, series_range: str, *, sku: str, version: int | None = None
    ) -> str:
        """Resolve a SKU series through the pointer, or retain the flat key."""
        resolved = self._resolve_version(sku, version)
        if resolved is None:
            return series_key(series_range)
        return series_key(series_range, sku=sku, version=resolved)

    def read_series(
        self,
        series_range: str,
        *,
        sku: str | None = None,
        version: int | None = None,
    ) -> Optional[dict]:
        """Read a rolling series, resolving a SKU through ``latest.json``."""
        if sku is None:
            if version is not None:
                raise PublishedRecordError(
                    "reading an explicit version requires a SKU"
                )
            return self._read(series_key(series_range))
        return self._read(
            self.resolve_series_key(series_range, sku=sku, version=version)
        )

    def _resolve_version(self, sku: str, explicit: int | None) -> int | None:
        # An explicit version is useful during pointer-last migration: the
        # immutable versioned objects can be verified while latest.json still
        # advertises the legacy layout.
        if explicit is not None:
            day_key("2000-01-01", sku=sku, version=explicit)
            return explicit
        latest = self.read_latest()
        if latest is None:
            return None
        data = latest["data"]
        versions = data.get("versions")
        if versions is None:
            return None
        match = next((entry for entry in versions if entry["sku"] == sku), None)
        if match is None:
            raise PublishedRecordError(
                f"latest.json has no version pointer for SKU {sku}"
            )
        return match["current_version"]
