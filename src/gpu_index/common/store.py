# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Append-only persistence for basket capture snapshots.

Keyspace (per basket lane, under the lane's configured ``bucket_prefix``):

  <prefix>/snapshots/<YYYY-MM-DD>/slot<HH>-<run_id>.json  immutable
  <prefix>/latest.json                                    pointer,
                                                          moved LAST
  <prefix>/report/index.html                              MUTABLE —
    the ONE exception to append-only under the prefix: a derived HTML
    dashboard rewritten in place every firing, published warn-only by
    scripts/compute_index_composite.py. Nothing reads it back; the
    immutable snapshots/composites remain the record.

Publish discipline: immutable objects are never overwritten with different
bytes, verify-after-write before the pointer moves, and the pointer's
published_at is a heartbeat (its age is the missed-day alarm's cheapest
read).

Duplicate-slot rule: two schedulers can both pass the slot gate in a race
window (one scheduler's cron drifting into another's firing) and both
record — run_ids carry a random suffix so the keys never collide. Readers
take the EARLIEST key per slot prefix (lexicographic == chronological);
later duplicates are tolerated history, not corruption. The pointer never
moves backwards to an older captured_at.

Transport (BucketConfig/make_client/put/get/list) is IMPORTED from
gpu_index.common.bucket on purpose: those helpers encode S3-gateway
footguns (checksum pinning, path-style addressing, empty-code 404s) that
must not be forked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from gpu_index.common.bucket import (
    BucketConfig,
    BucketPublishError,
    get_object_bytes,
    list_object_keys,
    make_client,
    put_json_bytes,
)
from gpu_index.common.slots import (
    latest_pointer_key,
    rfc3339,
    slot_key_prefix,
    snapshot_day_prefix,
    snapshot_key,
)

__all__ = [
    "BucketConfig",
    "BucketPublishError",
    "make_client",
    "make_run_id",
    "snapshot_bytes",
    "write_local_snapshot",
    "slot_already_captured",
    "previous_day_has_snapshots",
    "slot_hours_present",
    "upload_capture_snapshot",
    "read_day_snapshots",
    "day_slot_keys",
    "get_snapshot_by_key",
    "composite_key",
    "composite_pointer_key",
    "composite_exists",
    "upload_composite",
    "panel_composite_key",
    "get_panel_composite",
    "panel_composite_exists",
    "list_panel_observations",
    "upload_panel_composite",
    "put_immutable",
    "move_pointer_no_regress",
]

POINTER_VERSION = 1

# Debug-mirror root for write_local_snapshot (distinct from the local
# BACKEND root, which lives in gpu_index.common.bucket / GPU_INDEX_DATA_DIR
# and lays keys out under the lane's bucket_prefix).
DEFAULT_LOCAL_ROOT = (
    Path(os.environ.get("GPU_INDEX_DATA_DIR") or "data") / "index_basket"
)


def make_run_id(when: Optional[datetime] = None, *, unique: bool = True) -> str:
    """Second-resolution stamp + random suffix. The suffix means two
    schedulers stamping the same second can never collide on a snapshot key
    (a read-then-PUT store has no conditional-PUT guarantee to lean on)."""
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(2)}" if unique else stamp


def snapshot_bytes(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")


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


def slot_already_captured(
    client, bucket: str, *, prefix: str, day: date, slot_hour: int
) -> bool:
    return bool(
        list_object_keys(client, bucket, slot_key_prefix(prefix, day, slot_hour))
    )


def previous_day_has_snapshots(
    client, bucket: str, *, prefix: str, day: date
) -> bool:
    """True iff <day> has at least one stored snapshot. The caller passes the
    PREVIOUS day; False is the missed-day alarm condition."""
    return bool(
        list_object_keys(client, bucket, snapshot_day_prefix(prefix, day) + "/")
    )


_SLOT_KEY_RE = re.compile(r"/slot(\d{2})-")


def slot_hours_present(client, bucket: str, *, prefix: str, day: date) -> Set[int]:
    """Which slot hours actually recorded on <day> — the partial-miss alarm's
    read (a day that captured 22:00 but missed the canonical 16:00 must not
    look healthy just because it is non-empty)."""
    hours: Set[int] = set()
    for key in list_object_keys(
        client, bucket, snapshot_day_prefix(prefix, day) + "/"
    ):
        m = _SLOT_KEY_RE.search(key)
        if m:
            hours.add(int(m.group(1)))
    return hours


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
        "basket_id": payload["basket_id"],
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
        "basket_sources_ok": payload.get("basket_sources_ok", []),
        "previous_day_empty": payload.get("previous_day_empty"),
        "published_at": published_at,
    }


def read_day_snapshots(
    client, bucket: str, *, prefix: str, day: date
) -> Dict[int, Dict[str, Any]]:
    """{slot_hour: payload} for a day, honoring the duplicate-slot rule:
    the EARLIEST key per slot wins (lexicographic == chronological — see the
    module docstring); later duplicates are tolerated history, never read."""
    chosen_key_per_slot: Dict[int, str] = {}
    for key in sorted(
        list_object_keys(client, bucket, snapshot_day_prefix(prefix, day) + "/")
    ):
        m = _SLOT_KEY_RE.search(key)
        if m is None:
            continue
        slot = int(m.group(1))
        if slot not in chosen_key_per_slot:  # sorted → first seen is earliest
            chosen_key_per_slot[slot] = key
    out: Dict[int, Dict[str, Any]] = {}
    for slot, key in chosen_key_per_slot.items():
        raw = get_object_bytes(client, bucket, key)
        if raw is not None:
            out[slot] = json.loads(raw)
    return out


def day_slot_keys(
    client, bucket: str, *, prefix: str, day: date
) -> Dict[int, str]:
    """{slot_hour: chosen snapshot key} for a day -- read_day_snapshots'
    duplicate-slot selection rule VERBATIM (EARLIEST key per slot wins;
    lexicographic == chronological) with the GETs left out, so an hourly
    reader (the panel CLI) can LIST a day once and fetch only the hours
    it actually needs. read_day_snapshots itself is untouched: the daily
    lanes read whole days and their bytes depend on it."""
    chosen_key_per_slot: Dict[int, str] = {}
    for key in sorted(
        list_object_keys(client, bucket, snapshot_day_prefix(prefix, day) + "/")
    ):
        m = _SLOT_KEY_RE.search(key)
        if m is None:
            continue
        slot = int(m.group(1))
        if slot not in chosen_key_per_slot:  # sorted → first seen is earliest
            chosen_key_per_slot[slot] = key
    return chosen_key_per_slot


def get_snapshot_by_key(client, bucket: str, key: str) -> Optional[Dict[str, Any]]:
    """One stored snapshot payload, parsed; None when the key vanished
    between the LIST and the GET (the caller treats that as slot-missing,
    the same posture as read_day_snapshots' silent raw-None skip)."""
    raw = get_object_bytes(client, bucket, key)
    return json.loads(raw) if raw is not None else None


# ------------------------------------------------------------- composites
#
# Same discipline as snapshots, one keyspace per methodology so recomputed
# series COEXIST rather than overwrite. The day key is DETERMINISTIC (no
# run_id): with a random suffix the append-only guard could never fire
# across runs, and two racing --sync writers could publish two "official"
# values for one date. One key per (methodology, day) makes first-write-wins
# real for TEMPORALLY SEPARATED runs — identical bytes are idempotent,
# divergent bytes raise loudly. Caveat: read-then-PUT is not atomic, so two
# OVERLAPPING writers can both pass the check and the last PUT silently
# wins; both candidates are valid computations of the same-day inputs and
# replay pins to the survivor, so the exposure is bounded (the run_id lives
# in the pointer for audit):
#   <prefix>/composites/<methodology_id>/<YYYY-MM-DD>.json
#   <prefix>/composites/<methodology_id>/latest.json   (pointer, moved last,
#                                                       never regresses by date)


def composite_key(prefix: str, methodology_id: str, day: str) -> str:
    return f"{prefix}/composites/{methodology_id}/{day}.json"


def composite_pointer_key(prefix: str, methodology_id: str) -> str:
    return f"{prefix}/composites/{methodology_id}/latest.json"


def composite_exists(
    client, bucket: str, *, prefix: str, methodology_id: str, day: str
) -> bool:
    return (
        get_object_bytes(client, bucket, composite_key(prefix, methodology_id, day))
        is not None
    )


def get_composite(
    client, bucket: str, *, prefix: str, methodology_id: str, day: str
) -> Optional[Dict[str, Any]]:
    raw = get_object_bytes(
        client, bucket, composite_key(prefix, methodology_id, day)
    )
    return json.loads(raw) if raw is not None else None


def _put_immutable(client, bucket: str, key: str, data: bytes) -> str:
    """Append-only PUT with verify-after-write; returns the sha256 digest."""
    digest = hashlib.sha256(data).hexdigest()
    existing = get_object_bytes(client, bucket, key)
    if existing is not None and hashlib.sha256(existing).hexdigest() != digest:
        raise BucketPublishError(
            f"{key} already exists with different bytes — this keyspace is "
            "append-only"
        )
    put_json_bytes(client, bucket, key, data)
    stored = get_object_bytes(client, bucket, key)
    if stored is None or hashlib.sha256(stored).hexdigest() != digest:
        raise BucketPublishError(
            f"Verify-after-write failed for {key} — latest NOT moved"
        )
    return digest


def _move_pointer_no_regress(
    client, bucket: str, pointer_key: str, pointer: Dict[str, Any], order_field: str
) -> Dict[str, Any]:
    """PUT the pointer unless the current one is newer on ``order_field``
    (RFC3339/ISO strings compare chronologically)."""
    current_raw = get_object_bytes(client, bucket, pointer_key)
    if current_raw is not None:
        try:
            current = json.loads(current_raw)
        except json.JSONDecodeError:
            current = {}
        if str(current.get(order_field, "")) > str(pointer.get(order_field, "")):
            return {"moved": False, "pointer": current}
    put_json_bytes(
        client, bucket, pointer_key, snapshot_bytes(pointer), cache_control="no-store"
    )
    return {"moved": True, "pointer": pointer}


# The raw-observatory lane (gpu_index/observatory/store.py) reuses this exact
# append-only + no-regress publish discipline; public aliases so a sibling
# capture lane imports it instead of forking it. Pure re-exports — no
# behavior change to this lane.
put_immutable = _put_immutable
move_pointer_no_regress = _move_pointer_no_regress


def upload_composite(
    client,
    bucket: str,
    payload: Dict[str, Any],
    *,
    prefix: str,
    run_id: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """PUT the immutable daily composite (one per methodology+day), verify it
    back, move the pointer last; the pointer never regresses by date.

    ``run_id`` goes in the POINTER only: the immutable object's bytes must be
    a pure function of the inputs so two racing writers with identical
    computations are byte-identical (idempotent re-PUT) and only genuinely
    divergent computations trip the append-only guard.
    """
    key = composite_key(prefix, payload["methodology_id"], payload["date"])
    digest = _put_immutable(client, bucket, key, snapshot_bytes(payload))
    pointer = {
        "pointer_version": POINTER_VERSION,
        "kind": payload["kind"],
        "basket_id": payload["basket_id"],
        "methodology_id": payload["methodology_id"],
        "composite_key": key,
        "sha256": digest,
        "date": payload["date"],
        "basket_dark": payload["basket_dark"],
        "index_value_usd_gpu_hr": (payload.get("index") or {}).get(
            "value_usd_gpu_hr"
        ),
        "sources_used_count": (payload.get("index") or {}).get(
            "sources_used_count", 0
        ),
        "run_id": run_id,
        "published_at": rfc3339(now),
    }
    moved = _move_pointer_no_regress(
        client, bucket, composite_pointer_key(prefix, payload["methodology_id"]),
        pointer, "date",
    )
    return {
        "status": "published" if moved["moved"] else "published_pointer_kept",
        "composite_key": key,
        "pointer": moved["pointer"],
    }


# ------------------------------------------------------ panel composites
#
# The hourly panel lanes (METHODOLOGY.md, stage 4)
# reuse the composite keyspace discipline VERBATIM at observation
# resolution: the key day segment is the fixed-width observation stamp
# 'YYYY-MM-DDTHH' (zero-padded hour), so lexicographic order == the
# chronological order the existing helpers already lean on --
# composite_key treats the segment as an opaque string, and the pointer's
# no-regress compare on the payload 'date' works unchanged. Everything
# below is ADDITIVE: the daily helpers' behavior (frozen series' bytes)
# is untouched.
#
#   <prefix>/composites/<methodology_id>/<YYYY-MM-DDTHH>.json  immutable
#   <prefix>/composites/<methodology_id>/latest.json           pointer,
#     moved LAST, never regresses by 'date' (the observation stamp)
#
# Panel lanes may share a prefix with a daily lane (the migrated B300/
# B200 hourly lanes) -- collision-safe because composites key per
# methodology_id, and every hourly lane is a fresh mint by construction.

_PANEL_OBSERVATION_RE = re.compile(r"\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3])")


def panel_composite_key(prefix: str, methodology_id: str, observation: str) -> str:
    """The deterministic observation key (no run_id -- the append-only
    first-write-wins guard depends on one key per (methodology,
    observation), exactly the daily composite rule). Loud on anything
    that is not a fixed-width zero-padded 'YYYY-MM-DDTHH' stamp: a
    day-keyed payload routed through the panel path would silently
    share a keyspace slot with a daily artifact. FULLMATCH on purpose
    (security review): re.match + '$' still accepts a trailing newline
    ('2026-08-12T05\\n'), which would mint a second, shadow key for the
    same observation and defeat first-write-wins."""
    text = str(observation)
    if not _PANEL_OBSERVATION_RE.fullmatch(text):
        raise ValueError(
            f"panel observation key must be fixed-width 'YYYY-MM-DDTHH' "
            f"(zero-padded hour 00-23), got {observation!r}"
        )
    return composite_key(prefix, methodology_id, text)


def get_panel_composite(
    client, bucket: str, *, prefix: str, methodology_id: str, observation: str
) -> Optional[Dict[str, Any]]:
    raw = get_object_bytes(
        client, bucket, panel_composite_key(prefix, methodology_id, observation)
    )
    return json.loads(raw) if raw is not None else None


def panel_composite_exists(
    client, bucket: str, *, prefix: str, methodology_id: str, observation: str
) -> bool:
    return (
        get_object_bytes(
            client,
            bucket,
            panel_composite_key(prefix, methodology_id, observation),
        )
        is not None
    )


def list_panel_observations(
    client, bucket: str, *, prefix: str, methodology_id: str
) -> Set[str]:
    """The published observation stamps of one panel series -- ONE
    (paginated) LIST of <prefix>/composites/<methodology_id>/, parsing
    the fixed-width 'YYYY-MM-DDTHH' key stems. latest.json (and any
    other non-observation key, defensively) is ignored. This is the
    bounded-replay frontier read (compute_panel_index.py): the published
    stamp SET is learned without a single per-artifact GET, so the CLI's
    steady-state cost stops scaling with the series' age."""
    base = f"{prefix}/composites/{methodology_id}/"
    out: Set[str] = set()
    for key in list_object_keys(client, bucket, base):
        name = key[len(base):]
        if not name.endswith(".json"):
            continue
        stem = name[: -len(".json")]
        if _PANEL_OBSERVATION_RE.fullmatch(stem):
            out.add(stem)
    return out


def upload_panel_composite(
    client,
    bucket: str,
    payload: Dict[str, Any],
    *,
    prefix: str,
    run_id: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """PUT the immutable observation composite (one per methodology +
    observation stamp), verify it back, move the pointer last; the
    pointer never regresses by 'date' (the 'YYYY-MM-DDTHH' stamp --
    lexicographic == chronological, same compare as the daily lane).

    ``run_id`` goes in the POINTER only, the upload_composite discipline
    verbatim: the immutable object's bytes must be a pure function of
    the inputs so two racing writers with identical computations are
    byte-identical (idempotent re-PUT) and only genuinely divergent
    computations trip the append-only guard.
    """
    key = panel_composite_key(prefix, payload["methodology_id"], payload["date"])
    digest = _put_immutable(client, bucket, key, snapshot_bytes(payload))
    pointer = {
        "pointer_version": POINTER_VERSION,
        "kind": payload["kind"],
        "panel_id": payload["panel_id"],
        "methodology_id": payload["methodology_id"],
        "composite_key": key,
        "sha256": digest,
        "date": payload["date"],
        "observation_date": payload["observation_date"],
        "observation_hour_utc": payload["observation_hour_utc"],
        "observation_missed": payload["observation_missed"],
        "panel_dark": payload["panel_dark"],
        "index_value_usd_gpu_hr": (payload.get("index") or {}).get(
            "value_usd_gpu_hr"
        ),
        "sources_used_count": (payload.get("index") or {}).get(
            "sources_used_count", 0
        ),
        "run_id": run_id,
        "published_at": rfc3339(now),
    }
    moved = _move_pointer_no_regress(
        client,
        bucket,
        composite_pointer_key(prefix, payload["methodology_id"]),
        pointer,
        "date",
    )
    return {
        "status": "published" if moved["moved"] else "published_pointer_kept",
        "composite_key": key,
        "pointer": moved["pointer"],
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
    digest = _put_immutable(client, bucket, key, data)
    pointer = build_pointer(
        payload,
        key=key,
        sha256=digest,
        byte_length=len(data),
        published_at=rfc3339(now),
    )
    moved = _move_pointer_no_regress(
        client, bucket, latest_pointer_key(prefix), pointer, "captured_at"
    )
    return {
        "status": "published" if moved["moved"] else "published_pointer_kept",
        "snapshot_key": key,
        "pointer": moved["pointer"],
    }
