# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Era-aware scheduled-observation grid for the hourly panel lanes.

The hourly panel engine (METHODOLOGY.md) computes a lane at every SCHEDULED observation of an
era-scoped grid: the migrated B300/B200 lanes ride the basket record's
4-slot [4, 10, 16, 22] grid before their observatory cutover and the
hourly 0..23 grid after; the H-series lanes are hourly from genesis.
Three weight-engine quantities are defined ONLY against this grid:

  - the R-cutoff information boundary for the observation at stamp t is
    the PREVIOUS scheduled stamp (era-aware -- across the 4-slot ->
    hourly boundary the cutoff of the first hourly stamp is the last
    4-slot stamp, never a phantom hour);
  - the A2 attendance denominator is the count of scheduled stamps in
    the trailing window, counted PER ERA and clipped at the lane's
    genesis so pre-genesis hours never count as missed;
  - replay iterates every scheduled stamp in [genesis, now] -- a
    scheduled hour with no snapshot is an explicit observation_missed
    artifact, never a skipped loop index.

Everything here is a pure function of the constructed grid -- zero I/O,
zero clock reads, same posture as gpu_index.index.weights. Timestamps are the
weight lane's absolute integer hour stamps (day_ordinal * 24 + hour_utc,
proleptic-Gregorian day ordinals via datetime.date). The grid itself is
config (calc_params embeds it verbatim under the D2 refuse-to-extend
fence), so a cadence change is a MINT, never a silent shift of the
cutoff or the attendance denominator.

Construction is fail-closed: uniform slot spacing within each era
(including the wrap past midnight -- the same single-spacing rule the
basket config validator enforces, because horizon arithmetic has no
meaning on a lumpy grid), strictly increasing era from_dates, and a
first era that starts exactly at genesis. A malformed grid must refuse
at LOAD, never publish a wrong denominator.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


class PanelScheduleError(ValueError):
    """A malformed schedule config -- raised at construction (load time),
    never at the first observation that needs the grid."""


# ---------------------------------------------------------- stamp conversions


def date_hour_to_stamp(day: str, hour: int) -> int:
    """(ISO date, UTC hour) -> absolute hour stamp (day_ordinal*24 + hour)."""
    return date.fromisoformat(day).toordinal() * 24 + int(hour)


def stamp_to_date_hour(stamp: int) -> Tuple[str, int]:
    """Absolute hour stamp -> (ISO date, UTC hour)."""
    day_ordinal, hour = divmod(int(stamp), 24)
    return date.fromordinal(day_ordinal).isoformat(), hour


def stamp_to_hour_iso(stamp: int) -> str:
    """Absolute hour stamp -> the artifact observation key, a fixed-width
    'YYYY-MM-DDTHH' string (lexicographic == chronological)."""
    day_iso, hour = stamp_to_date_hour(stamp)
    return f"{day_iso}T{hour:02d}"


def hour_iso_to_stamp(value: str) -> int:
    """'YYYY-MM-DDTHH' -> absolute hour stamp. Loud on malformed input:
    every unparseable component raises PanelScheduleError (never a bare
    ValueError -- the CLI catches the schedule error class at its parse
    sites, and an uncaught int()/fromisoformat ValueError would traceback
    a typo'd --observation instead of printing the refusal). The hour is
    range-fenced too: '2026-08-12T99' would otherwise parse to a stamp on
    a DIFFERENT day (99 = day+4, hour 3) and silently target the wrong
    observation."""
    text = str(value)
    if len(text) != 13 or text[10] != "T":
        raise PanelScheduleError(
            f"observation stamp must be 'YYYY-MM-DDTHH', got {value!r}"
        )
    try:
        hour = int(text[11:13])
        if not 0 <= hour <= 23:
            raise ValueError(f"hour {hour} outside 0..23")
        return date_hour_to_stamp(text[:10], hour)
    except PanelScheduleError:
        raise
    except ValueError as exc:
        raise PanelScheduleError(
            f"observation stamp must be 'YYYY-MM-DDTHH' (valid date, hour "
            f"00-23), got {value!r}: {exc}"
        ) from exc


# ----------------------------------------------------------------- schedule


class PanelSchedule:
    """The era-scoped scheduled-observation grid of one panel lane.

    Built from {genesis_date, slot_grids: [{from_date, slot_hours_utc}]}.
    Era i covers every day from its from_date up to (exclusive) era
    i+1's from_date; the last era is open-ended. Within an era, the
    scheduled stamps of day d are d*24 + h for each configured slot
    hour. The first era starts AT genesis, so no scheduled stamp exists
    before the lane's genesis day -- the structural genesis clip.
    """

    def __init__(
        self,
        *,
        genesis_date: str,
        slot_grids: Sequence[Dict[str, Any]],
    ) -> None:
        try:
            genesis_day = date.fromisoformat(str(genesis_date)).toordinal()
        except ValueError as exc:
            raise PanelScheduleError(
                f"genesis_date must be an ISO date, got {genesis_date!r}"
            ) from exc
        if not isinstance(slot_grids, (list, tuple)) or not slot_grids:
            raise PanelScheduleError(
                f"slot_grids must be a non-empty list, got {slot_grids!r}"
            )
        eras: List[Tuple[int, Optional[int], Tuple[int, ...]]] = []
        start_days: List[int] = []
        prev_start: Optional[int] = None
        for i, grid in enumerate(slot_grids):
            if not isinstance(grid, dict):
                raise PanelScheduleError(
                    f"slot_grids[{i}] must be an object, got {grid!r}"
                )
            try:
                start_day = date.fromisoformat(
                    str(grid.get("from_date"))
                ).toordinal()
            except ValueError as exc:
                raise PanelScheduleError(
                    f"slot_grids[{i}].from_date must be an ISO date, got "
                    f"{grid.get('from_date')!r}"
                ) from exc
            if i == 0 and start_day != genesis_day:
                # The first era MUST start at genesis: attendance clips its
                # window at genesis, so an ungoverned gap between genesis
                # and the first era would be a denominator nobody defined.
                raise PanelScheduleError(
                    f"slot_grids[0].from_date {grid.get('from_date')!r} must "
                    f"equal genesis_date {genesis_date!r}"
                )
            if prev_start is not None and start_day <= prev_start:
                raise PanelScheduleError(
                    f"slot_grids from_dates must be strictly increasing, got "
                    f"{grid.get('from_date')!r} after era starting "
                    f"{date.fromordinal(prev_start).isoformat()!r}"
                )
            slots = self._validate_slots(i, grid.get("slot_hours_utc"))
            if eras:
                # Close the previous era at this era's start day.
                eras[-1] = (eras[-1][0], start_day, eras[-1][2])
            eras.append((start_day, None, slots))
            start_days.append(start_day)
            prev_start = start_day
        self.genesis_date = str(genesis_date)
        self._eras = eras
        self._start_days = start_days
        self._genesis_stamp = eras[0][0] * 24 + eras[0][2][0]

    @staticmethod
    def _validate_slots(index: int, raw: Any) -> Tuple[int, ...]:
        if (
            not isinstance(raw, (list, tuple))
            or not raw
            or not all(
                isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23
                for h in raw
            )
            or sorted(set(raw)) != list(raw)
        ):
            # Strictly increasing pins the grid's byte order in the embedded
            # calc_params -- two orderings of one slot set would be two
            # different byte streams for one methodology (the horizons rule).
            raise PanelScheduleError(
                f"slot_grids[{index}].slot_hours_utc must be a strictly "
                f"increasing list of ints in 0..23, got {raw!r}"
            )
        slots = tuple(int(h) for h in raw)
        gaps = [b - a for a, b in zip(slots, slots[1:])]
        gaps.append(24 - slots[-1] + slots[0])
        if len(set(gaps)) != 1:
            raise PanelScheduleError(
                f"slot_grids[{index}].slot_hours_utc requires UNIFORM spacing "
                f"including the midnight wrap, got {list(slots)!r} "
                f"(gaps {gaps!r})"
            )
        return slots

    # ------------------------------------------------------------- queries

    @property
    def genesis_stamp(self) -> int:
        """The lane's first scheduled stamp (genesis day, first slot)."""
        return self._genesis_stamp

    def _era_for_day(self, day_ordinal: int) -> Optional[int]:
        if day_ordinal < self._start_days[0]:
            return None
        return bisect_right(self._start_days, day_ordinal) - 1

    def is_scheduled(self, stamp: int) -> bool:
        """True iff `stamp` is a scheduled observation stamp of its era
        (always False before genesis)."""
        day_ordinal, hour = divmod(int(stamp), 24)
        idx = self._era_for_day(day_ordinal)
        if idx is None:
            return False
        return hour in self._eras[idx][2]

    def prev_scheduled_stamp(self, stamp: int) -> Optional[int]:
        """The last scheduled stamp STRICTLY before `stamp` -- the R-cutoff
        information boundary of the observation at `stamp`. Era-aware:
        across the 4-slot -> hourly boundary the first hourly stamp's
        cutoff is the last 4-slot stamp. None when nothing scheduled
        precedes (the genesis observation has no cutoff -- every score
        there is undefined, never zero)."""
        target = int(stamp) - 1
        for start_day, end_day, slots in reversed(self._eras):
            era_start = start_day * 24
            if era_start > target:
                continue
            top = target
            if end_day is not None:
                top = min(top, (end_day - 1) * 24 + slots[-1])
            day_ordinal, hour = divmod(top, 24)
            idx = bisect_right(slots, hour) - 1
            if idx >= 0:
                return day_ordinal * 24 + slots[idx]
            if day_ordinal - 1 >= start_day:
                return (day_ordinal - 1) * 24 + slots[-1]
            # This era holds no stamp <= target; walk into the prior era.
        return None

    def scheduled_stamps(self, window_start: int, window_end: int) -> List[int]:
        """Every scheduled stamp s with window_start <= s < window_end
        (half-open, matching the attendance window [t - H, t)), era-aware
        and structurally clipped at genesis (no era precedes it). Sorted
        ascending."""
        lo_bound = int(window_start)
        hi_bound = int(window_end)
        out: List[int] = []
        for start_day, end_day, slots in self._eras:
            lo = max(lo_bound, start_day * 24)
            hi = hi_bound if end_day is None else min(hi_bound, end_day * 24)
            if hi <= lo:
                continue
            for day_ordinal in range(lo // 24, (hi - 1) // 24 + 1):
                base = day_ordinal * 24
                for hour in slots:
                    s = base + hour
                    if lo <= s < hi:
                        out.append(s)
        return out

    def iter_scheduled(self, through_stamp: int) -> Iterator[int]:
        """Iterate every scheduled stamp in [genesis, through_stamp]
        INCLUSIVE -- the replay loop's domain (a scheduled stamp with no
        snapshot is an explicit observation_missed publish, never a
        skipped index)."""
        return iter(
            self.scheduled_stamps(self._genesis_stamp, int(through_stamp) + 1)
        )
