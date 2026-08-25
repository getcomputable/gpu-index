# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Append-only persistence for observatory capture snapshots.

Keyspace (consumed READ-ONLY by the hourly panel-index calc lanes since
2026-08-23 -- METHODOLOGY.md; nothing but this lane
ever writes under it):

  index/raw_observatory/snapshots/<YYYY-MM-DD>/slot<HH>-<run_id>.json  immutable
  index/raw_observatory/latest.json                                    pointer,
                                                                       moved LAST

Identical publish discipline to the basket lanes, IMPORTED rather than
forked (gpu_index.common.store): immutable objects are never overwritten with
different bytes, verify-after-write before the pointer moves, the pointer
never regresses, and its published_at is the heartbeat for the missed-day
alarm. Key layout helpers come from gpu_index.common.slots, so the slot-idempotency
and duplicate-slot (earliest-key-wins) semantics are the same machinery,
not a re-implementation.

Transport (BucketConfig/make_client) is the gpu_index.common.bucket stack
— the S3-gateway footguns it encodes must not be forked. Nothing here
reads or writes any basket-lane key: the prefix fence in
observatory.config makes that mechanical.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from gpu_index.common.slots import latest_pointer_key, rfc3339, snapshot_key
from gpu_index.common.store import (
    BucketConfig,
    BucketPublishError,
    make_client,
    make_run_id,
    move_pointer_no_regress,
    previous_day_has_snapshots,
    put_immutable,
    slot_already_captured,
    slot_hours_present,
    snapshot_bytes,
)

__all__ = [
    "BucketConfig",
    "BucketPublishError",
    "make_client",
    "make_run_id",
    "previous_day_has_snapshots",
    "slot_already_captured",
    "slot_hours_present",
    "write_local_snapshot",
    "upload_capture_snapshot",
]

POINTER_VERSION = 1

# Debug-mirror root (distinct from the local BACKEND root in
# gpu_index.common.bucket / GPU_INDEX_DATA_DIR).
DEFAULT_LOCAL_ROOT = Path(
    os.environ.get("RAW_OBSERVATORY_DATA_DIR")
    or Path(os.environ.get("GPU_INDEX_DATA_DIR") or "data") / "raw_observatory"
)


def write_local_snapshot(
    payload: Dict[str, Any], *, root: Path = DEFAULT_LOCAL_ROOT
) -> Path:
    """Dev/debug mirror of the bucket layout; the bucket copy is the record."""
    day = payload["capture_date"]
    slot = int(payload["slot_hour_utc"])
    out_dir = root / "snapshots" / day
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"slot{slot:02d}-{payload['run_id']}.json"
    out.write_bytes(snapshot_bytes(payload))
    return out


def build_pointer(
    payload: Dict[str, Any],
    *,
    key: str,
    sha256: str,
    byte_length: int,
    published_at: str,
) -> Dict[str, Any]:
    return {
        "pointer_version": POINTER_VERSION,
        "kind": payload["kind"],
        "lane_id": payload["lane_id"],
        "snapshot_key": key,
        "sha256": sha256,
        "byte_length": byte_length,
        "run_id": payload["run_id"],
        "capture_date": payload["capture_date"],
        "slot_hour_utc": payload["slot_hour_utc"],
        "canonical_slot": payload["canonical_slot"],
        "late_fill": payload.get("late_fill", False),
        "captured_at": payload["captured_at"],
        "sources_ok": payload["sources_ok"],
        "sources_failed": payload["sources_failed"],
        "observation_count": payload.get("observation_count", 0),
        "skus_observed": payload.get("skus_observed", []),
        "previous_day_empty": payload.get("previous_day_empty"),
        "published_at": published_at,
    }


def upload_capture_snapshot(
    client,
    bucket: str,
    payload: Dict[str, Any],
    *,
    prefix: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """PUT the immutable snapshot, verify it back, then move the pointer.

    Pointer never regresses: a late fill of yesterday's slot finishing
    after today's capture must not repoint latest at the older snapshot.
    """
    day = date.fromisoformat(payload["capture_date"])
    key = snapshot_key(
        prefix, day, int(payload["slot_hour_utc"]), payload["run_id"]
    )
    data = snapshot_bytes(payload)
    digest = put_immutable(client, bucket, key, data)
    pointer = build_pointer(
        payload,
        key=key,
        sha256=digest,
        byte_length=len(data),
        published_at=rfc3339(now),
    )
    moved = move_pointer_no_regress(
        client, bucket, latest_pointer_key(prefix), pointer, "captured_at"
    )
    return {
        "status": "published" if moved["moved"] else "published_pointer_kept",
        "snapshot_key": key,
        "pointer": moved["pointer"],
    }


def read_pointer(client, bucket: str, *, prefix: str) -> Optional[Dict[str, Any]]:
    from gpu_index.common.bucket import get_object_bytes

    raw = get_object_bytes(client, bucket, latest_pointer_key(prefix))
    return json.loads(raw) if raw is not None else None
