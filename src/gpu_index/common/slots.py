# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Slot gating: which capture slot a run belongs to, and idempotency.

The capture job fires every 30 minutes (dozens of runs a day) but must
record only the configured 2-4 slots. A run
claims the LATEST slot mark at or before now (wrapping to the previous
day's last slot), then skips itself if any snapshot already exists for
that (date, slot) — so the first firing after each slot mark does the
capture, every later firing in the window is a cheap no-op, and a
scheduler dying for one firing self-heals on the next. A slot with no
firing at all before the NEXT mark stays visibly missing — there is no
backfill across marks. Fills recorded after the mark hour but inside the
window carry ``late_fill: true`` in the snapshot, so what "the 16:00 UTC
print" means (and any staleness rule for a multi-hour-late fill) is the
composite reader's call, made on honest data — the methodology's
promote-nearest rule (canonical-slot substitution) also belongs there, not
in the capture path.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

MINUTES_PER_DAY = 1440


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def rfc3339(ts: Optional[datetime] = None) -> str:
    """Single RFC3339 formatter for the whole lane (Z suffix, no micros)."""
    stamp = (ts or utc_now()).astimezone(timezone.utc).replace(microsecond=0)
    return stamp.isoformat().replace("+00:00", "Z")


def current_slot(
    now: datetime, slot_minutes_utc: Sequence[int]
) -> Tuple[date, int]:
    """(slot_date, minute_of_day) for the latest slot mark at or before
    ``now``.

    Before the day's first mark, the run belongs to the PREVIOUS day's
    last slot (e.g. marks [240, 600, 960, 1320] == hours [4,10,16,22]:
    an 02:24Z firing belongs to yesterday's 22:00 slot).
    """
    if not slot_minutes_utc:
        raise ValueError("capture slots are empty — nothing to gate on")
    slots: List[int] = sorted(set(int(m) for m in slot_minutes_utc))
    if slots[0] < 0 or slots[-1] > MINUTES_PER_DAY - 1:
        raise ValueError(f"capture slot minutes out of range: {slots}")
    now = now.astimezone(timezone.utc)
    now_minute = now.hour * 60 + now.minute
    todays = [m for m in slots if m <= now_minute]
    if todays:
        return now.date(), todays[-1]
    yesterday = (now - timedelta(days=1)).date()
    return yesterday, slots[-1]


def snapshot_day_prefix(bucket_prefix: str, day: date) -> str:
    return f"{bucket_prefix}/snapshots/{day.isoformat()}"


def slot_token(minute_of_day: int, *, minute_tokens: bool) -> str:
    """The key token for one mark. Hour-vocabulary lanes keep the legacy
    2-digit hour token (their marks are hour-aligned by construction --
    LOUD otherwise: writing an off-hour mark under an hour token would
    silently collapse two marks onto one key prefix); minute-vocabulary
    lanes write the 4-digit HHMM token for every mark."""
    minute_of_day = int(minute_of_day)
    hour, minute = divmod(minute_of_day, 60)
    if minute_tokens:
        return f"slot{hour:02d}{minute:02d}"
    if minute != 0:
        raise ValueError(
            f"hour-token lane cannot key sub-hour mark {minute_of_day} "
            f"(minute {minute})"
        )
    return f"slot{hour:02d}"


def slot_key_prefix(
    bucket_prefix: str,
    day: date,
    minute_of_day: int,
    *,
    minute_tokens: bool = False,
) -> str:
    token = slot_token(minute_of_day, minute_tokens=minute_tokens)
    return f"{snapshot_day_prefix(bucket_prefix, day)}/{token}-"


def snapshot_key(
    bucket_prefix: str,
    day: date,
    minute_of_day: int,
    run_id: str,
    *,
    minute_tokens: bool = False,
) -> str:
    return (
        f"{slot_key_prefix(bucket_prefix, day, minute_of_day, minute_tokens=minute_tokens)}"
        f"{run_id}.json"
    )


def latest_pointer_key(bucket_prefix: str) -> str:
    return f"{bucket_prefix}/latest.json"


def is_canonical(
    minute_of_day: int, canonical_minute_utc: Optional[int]
) -> bool:
    return canonical_minute_utc is not None and int(minute_of_day) == int(
        canonical_minute_utc
    )
