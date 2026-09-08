# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Store-backed reader for the published record.

Rides the shared transport (``gpu_index.common.bucket``): a local
directory holding a downloaded copy of the record (``GPU_INDEX_DATA_DIR``,
default ``./data``), the anonymous public HTTPS front
(``GPU_INDEX_PUBLIC_BASE_URL``), or an S3-compatible bucket. Every read
returns a digest-verified envelope (``decode_and_verify_artifact``) —
there is no unverified read path.

``latest.json`` advertises the per-SKU as-published history prefix. Default
SKU day and series reads use it when present, otherwise the current version;
an explicit version always addresses that version's re-derivation keyspace.
Version metadata is snapshotted on first resolution for this reader instance.
During the pointer-last migration, a legacy pointer
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
    PublishedRecordError as _PublishedRecordError,
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
        self._version_latest: tuple[dict | None] | None = None

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

    def version_pointer(self, sku: str) -> dict | None:
        """Return SKU metadata from one verified pointer snapshot per reader."""
        if self._version_latest is None:
            self._version_latest = (self.read_latest(),)
        latest = self._version_latest[0]
        versions = (latest or {}).get("data", {}).get("versions")
        if versions is None:
            return None
        match = next((entry for entry in versions if entry["sku"] == sku), None)
        if match is None:
            raise _PublishedRecordError(
                f"latest.json has no version pointer for SKU {sku}"
            )
        return match

    @staticmethod
    def _day_key(date: str, sku: str, resolved: int | str | None) -> str:
        if isinstance(resolved, str):
            return f"{resolved}/{day_key(date)}"
        return day_key(date) if resolved is None else day_key(date, sku=sku, version=resolved)

    def resolve_day_key(
        self, date: str, *, sku: str, version: int | None = None
    ) -> str:
        """Resolve a SKU day through the pointer, or retain the flat key."""
        resolved, _methodology = self._resolve_target(sku, version)
        return self._day_key(date, sku, resolved)

    def read_day(
        self,
        date: str,
        *,
        sku: str | None = None,
        version: int | None = None,
        resolved_key: str | None = None,
    ) -> Optional[dict]:
        """Read one UTC day, resolving a SKU through ``latest.json``.

        ``resolved_key`` pins a key returned by :meth:`resolve_day_key` so a
        caller can display and read the same object even if the mutable pointer
        moves between those operations.
        """
        if sku is None:
            if version is not None or resolved_key is not None:
                raise _PublishedRecordError(
                    "reading an explicit version or resolved key requires a SKU"
                )
            return self._read(day_key(date))
        if version is not None and resolved_key is not None:
            raise _PublishedRecordError(
                "read_day accepts either a version or a resolved key, not both"
            )
        if resolved_key is None:
            resolved, methodology = self._resolve_target(sku, version)
            key = self._day_key(date, sku, resolved)
        else:
            key = resolved_key
            flat_key = day_key(date)
            if key == flat_key:
                methodology = None
            elif key == f"{sku}/published/{flat_key}":
                resolved, methodology = self._resolve_target(sku, None)
                if resolved != f"{sku}/published":
                    raise _PublishedRecordError(
                        f"resolved day key {key!r} is not advertised by latest.json"
                    )
            else:
                parts = key.split("/")
                suffix_parts = flat_key.split("/")
                if (
                    len(parts) != len(suffix_parts) + 2
                    or parts[0] != sku
                    or not parts[1].startswith("v")
                ):
                    raise _PublishedRecordError(
                        f"resolved day key {key!r} does not address {sku} {date}"
                    )
                try:
                    resolved = int(parts[1].removeprefix("v"))
                    expected = day_key(date, sku=sku, version=resolved)
                except (IndexError, ValueError, _PublishedRecordError):
                    raise _PublishedRecordError(
                        f"resolved day key {key!r} does not address {sku} {date}"
                    ) from None
                if key != expected or parts[1] != f"v{resolved}":
                    raise _PublishedRecordError(
                        f"resolved day key {key!r} does not address {sku} {date}"
                    )
                _resolved, methodology = self._resolve_target(sku, resolved)
        envelope = self._read(key)
        if envelope is not None:
            data = envelope["data"]
            if data["kind"] != "gpu_index_observation_day" or data["date"] != date:
                raise _PublishedRecordError(
                    f"published day {key} does not describe {date}"
                )
            self._require_version_identity(data, sku, methodology, key)
        return envelope

    def resolve_series_key(
        self, series_range: str, *, sku: str, version: int | None = None
    ) -> str:
        """Resolve a SKU series through the pointer, or retain the flat key."""
        resolved, _methodology = self._resolve_target(sku, version)
        if isinstance(resolved, str):
            return f"{resolved}/{series_key(series_range)}"
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
                raise _PublishedRecordError(
                    "reading an explicit version requires a SKU"
                )
            return self._read(series_key(series_range))
        resolved, methodology = self._resolve_target(sku, version)
        if isinstance(resolved, str):
            key = f"{resolved}/{series_key(series_range)}"
        else:
            key = (
                series_key(series_range)
                if resolved is None
                else series_key(series_range, sku=sku, version=resolved)
            )
        envelope = self._read(key)
        if envelope is not None:
            data = envelope["data"]
            if (
                data["kind"] != "gpu_index_series"
                or data["range"] != series_range
            ):
                raise _PublishedRecordError(
                    f"published series {key} does not describe {series_range}"
                )
            self._require_version_identity(data, sku, methodology, key)
        return envelope

    def _resolve_target(
        self, sku: str, explicit: int | None
    ) -> tuple[int | str | None, str | frozenset[str] | None]:
        # An explicit version is useful during pointer-last migration: the
        # immutable versioned objects can be verified while latest.json still
        # advertises the legacy layout.
        if explicit is not None:
            day_key("2000-01-01", sku=sku, version=explicit)
        match = self.version_pointer(sku)
        if match is None:
            return explicit, None
        if explicit is None and "history_path" in match:
            return match["history_path"], frozenset(
                entry["methodology_id"] for entry in match["succession"]
            )
        if explicit is None:
            return match["current_version"], match["methodology_id"]
        version_entry = next(
            (entry for entry in match["succession"] if entry["version"] == explicit),
            None,
        )
        if version_entry is None:
            raise _PublishedRecordError(
                f"latest.json does not advertise {sku} version {explicit}"
            )
        return explicit, version_entry["methodology_id"]

    @staticmethod
    def _require_version_identity(
        data: dict,
        sku: str,
        methodology: str | frozenset[str] | None,
        key: str,
    ) -> None:
        if methodology is None:
            return
        methods = methodology if isinstance(methodology, frozenset) else {methodology}
        for observation in data["observations"]:
            if (
                observation.get("sku") != sku
                or observation.get("methodology_id") not in methods
            ):
                raise _PublishedRecordError(
                    f"published artifact {key} disagrees with its version identity"
                )
