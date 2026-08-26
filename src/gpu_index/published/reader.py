# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Store-backed reader for the published record.

Rides the shared transport (``gpu_index.common.bucket``): a local
directory holding a downloaded copy of the record (``GPU_INDEX_DATA_DIR``,
default ``./data``), the anonymous public HTTPS front
(``GPU_INDEX_PUBLIC_BASE_URL``), or an S3-compatible bucket. Every read
returns a digest-verified envelope (``decode_and_verify_artifact``) —
there is no unverified read path.

Day files are MUTABLE BY DESIGN inside the publisher's 90-day window
(the publisher re-reconciles them on every run and deletes files that
age out of the window), so a
missing day is an ordinary state — out of window, or not yet published —
and reads return ``None`` for it rather than raising.
"""

from __future__ import annotations

from typing import Optional

from gpu_index.common.bucket import (
    BucketConfig,
    get_object_bytes,
    make_client,
)
from gpu_index.published.artifacts import (
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

    def read_day(self, date: str) -> Optional[dict]:
        """``observations/YYYY/MM/DD.json``: one UTC day, all lanes."""
        return self._read(day_key(date))

    def read_series(self, series_range: str) -> Optional[dict]:
        """``series/{24h,7d,30d,90d}.json``: aggregate history rows."""
        return self._read(series_key(series_range))
