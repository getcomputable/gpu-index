# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Era-aware scheduled-observation grid for the panel lanes.

The panel engine (METHODOLOGY.md) computes a lane at every SCHEDULED
observation of an era-scoped grid: the migrated B300/B200 lanes ride the
basket record's 4-slot [4, 10, 16, 22] grid before their observatory
cutover and the hourly 0..23 grid after; the H-series lanes are hourly
from genesis; a 15-minute era is expressed as minute-of-day marks. Three
weight-engine quantities are defined ONLY against this grid:

  - the R-cutoff information boundary for the observation at stamp t is
    the PREVIOUS scheduled stamp (era-aware -- across an era boundary the
    cutoff of the first new-era stamp is the last old-era stamp, never a
    phantom mark);
  - the A2 attendance denominator is the count of scheduled stamps in
    the trailing window, counted PER ERA and clipped at the lane's
    genesis so pre-genesis marks never count as missed;
  - replay iterates every scheduled stamp in [genesis, now] -- a
    scheduled mark with no snapshot is an explicit observation_missed
    artifact, never a skipped loop index.

Everything here is a pure function of the constructed grid -- zero I/O,
zero clock reads, same posture as gpu_index.index.weights.

STAMP LATTICE (re-based 2026-08-27): timestamps are absolute integer
MINUTE stamps (day_ordinal * 1440 + minute_of_day_utc, proleptic-
Gregorian day ordinals via datetime.date). Hour-grid lanes simply
occupy the :00 minutes of that lattice; the lattice change alone moves
no lane's bytes (pinned by the re-base parity suite).

    day D-1                 day D                          day D+1
    ...|----|----|----|----||====|====|====|====|====||----|...
        stamps = D*1440 + m for each configured mark m (minute of day)

    era resolution: from_date bisect (day granularity) -> mark tuple
    key format:     one methodology id = ONE grain + key format forever
                    hour-grid lane  -> 'YYYY-MM-DDTHH'    (13 chars)
                    minute-grid lane-> 'YYYY-MM-DDTHHMM'  (15 chars)
                    (both fixed width => lexicographic == chronological
                    within a lane's keyspace, which is per-methodology)

The grid itself is config (calc_params embeds it verbatim under the D2
refuse-to-extend fence), so a cadence change is a MINT, never a silent
shift of the cutoff or the attendance denominator. An era declares its
marks as EITHER ``slot_hours_utc`` (ints 0..23, the legacy vocabulary)
OR ``slot_minutes_utc`` (ints 0..1439) -- never both. A lane with any
``slot_minutes_utc`` era is MINUTE-KEYED: every observation key it ever
writes (including replayed pre-cutover eras) carries the minute.

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

MINUTES_PER_DAY = 1440
MINUTES_PER_HOUR = 60


class PanelScheduleError(ValueError):
    """A malformed schedule config -- raised at construction (load time),
    never at the first observation that needs the grid."""


# ---------------------------------------------------------- stamp conversions


def date_minute_to_stamp(day: str, minute_of_day: int) -> int:
    """(ISO date, UTC minute-of-day) -> absolute minute stamp
    (day_ordinal*1440 + minute_of_day)."""
    return date.fromisoformat(day).toordinal() * MINUTES_PER_DAY + int(
        minute_of_day
    )


def date_hour_to_stamp(day: str, hour: int) -> int:
    """(ISO date, UTC hour) -> absolute minute stamp at that hour's :00.
    Convenience for the hour-grid lanes and their tests; the returned
    stamp lives on the SAME minute lattice as every other stamp."""
    return date_minute_to_stamp(day, int(hour) * MINUTES_PER_HOUR)


def stamp_to_date_minute(stamp: int) -> Tuple[str, int]:
    """Absolute minute stamp -> (ISO date, UTC minute-of-day)."""
    day_ordinal, minute = divmod(int(stamp), MINUTES_PER_DAY)
    return date.fromordinal(day_ordinal).isoformat(), minute


def stamp_to_obs_key(stamp: int, *, minute_keyed: bool) -> str:
    """Absolute minute stamp -> the artifact observation key.

    Hour-keyed lanes emit the legacy fixed-width 'YYYY-MM-DDTHH' and
    REFUSE a stamp that is not hour-aligned (an off-hour stamp under an
    hour-keyed lane is a unit bug upstream, never something to round);
    minute-keyed lanes emit fixed-width 'YYYY-MM-DDTHHMM'. Both formats
    are lexicographically chronological; they never share a keyspace
    (one methodology id = one format forever)."""
    day_iso, minute = stamp_to_date_minute(stamp)
    hour, rem = divmod(minute, MINUTES_PER_HOUR)
    if minute_keyed:
        return f"{day_iso}T{hour:02d}{rem:02d}"
    if rem != 0:
        raise PanelScheduleError(
            f"stamp {int(stamp)} ({day_iso} minute {minute}) is not "
            f"hour-aligned -- an hour-keyed lane can never write it"
        )
    return f"{day_iso}T{hour:02d}"


def obs_key_to_stamp(value: str) -> int:
    """'YYYY-MM-DDTHH' or 'YYYY-MM-DDTHHMM' -> absolute minute stamp.

    Loud on malformed input: every unparseable component raises
    PanelScheduleError (never a bare ValueError -- the CLI catches the
    schedule error class at its parse sites, and an uncaught
    int()/fromisoformat ValueError would traceback a typo'd
    --observation instead of printing the refusal). Hour and minute are
    range-fenced too: '2026-08-12T99' would otherwise parse to a stamp
    on a DIFFERENT day and silently target the wrong observation."""
    text = str(value)
    if len(text) not in (13, 15) or (len(text) > 10 and text[10] != "T"):
        raise PanelScheduleError(
            f"observation stamp must be 'YYYY-MM-DDTHH' or "
            f"'YYYY-MM-DDTHHMM', got {value!r}"
        )
    try:
        hour = int(text[11:13])
        if not 0 <= hour <= 23:
            raise ValueError(f"hour {hour} outside 0..23")
        minute = 0
        if len(text) == 15:
            minute = int(text[13:15])
            if not 0 <= minute <= 59:
                raise ValueError(f"minute {minute} outside 0..59")
        return date_minute_to_stamp(
            text[:10], hour * MINUTES_PER_HOUR + minute
        )
    except PanelScheduleError:
        raise
    except ValueError as exc:
        raise PanelScheduleError(
            f"observation stamp must be 'YYYY-MM-DDTHH' or "
            f"'YYYY-MM-DDTHHMM' (valid date, hour 00-23, minute 00-59), "
            f"got {value!r}: {exc}"
        ) from exc


# Hour-view conveniences over the minute lattice: each REFUSES a stamp
# that is not hour-aligned (an off-hour stamp reaching an hour-view call
# is a unit bug upstream, never something to round). The hour-grid lanes
# and their tests speak this vocabulary; minute-keyed paths use the
# minute/obs-key functions above.


def stamp_to_date_hour(stamp: int) -> Tuple[str, int]:
    """Hour-aligned minute stamp -> (ISO date, UTC hour). Loud otherwise."""
    day_iso, minute = stamp_to_date_minute(stamp)
    hour, rem = divmod(minute, MINUTES_PER_HOUR)
    if rem != 0:
        raise PanelScheduleError(
            f"stamp {int(stamp)} ({day_iso} minute {minute}) is not "
            f"hour-aligned -- hour-view callers cannot consume it"
        )
    return day_iso, hour


def stamp_to_hour_iso(stamp: int) -> str:
    """Hour-aligned minute stamp -> 'YYYY-MM-DDTHH'. Loud otherwise."""
    return stamp_to_obs_key(stamp, minute_keyed=False)


def hour_iso_to_stamp(value: str) -> int:
    """STRICT 'YYYY-MM-DDTHH' -> minute stamp (refuses the minute form --
    the name promises the hour vocabulary; format-agnostic callers use
    obs_key_to_stamp)."""
    if len(str(value)) != 13:
        raise PanelScheduleError(
            f"observation stamp must be 'YYYY-MM-DDTHH', got {value!r}"
        )
    return obs_key_to_stamp(value)


# ----------------------------------------------------------------- schedule


class PanelSchedule:
    """The era-scoped scheduled-observation grid of one panel lane.

    Built from {genesis_date, slot_grids: [{from_date, slot_hours_utc |
    slot_minutes_utc}]}. Era i covers every day from its from_date up to
    (exclusive) era i+1's from_date; the last era is open-ended. Within
    an era, the scheduled stamps of day d are d*1440 + m for each
    configured mark m (minute of day; the hour vocabulary normalizes to
    h*60). The first era starts AT genesis, so no scheduled stamp exists
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
        minute_keyed = False
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
            slots, era_minute_keyed = self._validate_slots(i, grid)
            minute_keyed = minute_keyed or era_minute_keyed
            if eras:
                # Close the previous era at this era's start day.
                eras[-1] = (eras[-1][0], start_day, eras[-1][2])
            eras.append((start_day, None, slots))
            start_days.append(start_day)
            prev_start = start_day
        self.genesis_date = str(genesis_date)
        self._eras = eras
        self._start_days = start_days
        self._genesis_stamp = eras[0][0] * MINUTES_PER_DAY + eras[0][2][0]
        self._minute_keyed = minute_keyed

    @staticmethod
    def _validate_slots(
        index: int, grid: Dict[str, Any]
    ) -> Tuple[Tuple[int, ...], bool]:
        """One era's marks -> (minute-of-day tuple, declared-in-minutes).

        Exactly one of slot_hours_utc / slot_minutes_utc must be present.
        Both vocabularies demand a strictly increasing list (this pins
        the grid's byte order in the embedded calc_params -- two
        orderings of one slot set would be two different byte streams
        for one methodology, the horizons rule) and UNIFORM spacing
        including the midnight wrap."""
        raw_hours = grid.get("slot_hours_utc")
        raw_minutes = grid.get("slot_minutes_utc")
        if (raw_hours is None) == (raw_minutes is None):
            raise PanelScheduleError(
                f"slot_grids[{index}] must declare exactly one of "
                f"slot_hours_utc / slot_minutes_utc"
            )
        if raw_minutes is None:
            raw, bound, label = raw_hours, 23, "slot_hours_utc"
        else:
            raw, bound, label = raw_minutes, 1439, "slot_minutes_utc"
        if (
            not isinstance(raw, (list, tuple))
            or not raw
            or not all(
                isinstance(m, int)
                and not isinstance(m, bool)
                and 0 <= m <= bound
                for m in raw
            )
            or sorted(set(raw)) != list(raw)
        ):
            raise PanelScheduleError(
                f"slot_grids[{index}].{label} must be a strictly "
                f"increasing list of ints in 0..{bound}, got {raw!r}"
            )
        if raw_minutes is None:
            slots = tuple(int(h) * MINUTES_PER_HOUR for h in raw)
        else:
            slots = tuple(int(m) for m in raw)
        gaps = [b - a for a, b in zip(slots, slots[1:])]
        gaps.append(MINUTES_PER_DAY - slots[-1] + slots[0])
        if len(set(gaps)) != 1:
            raise PanelScheduleError(
                f"slot_grids[{index}].{label} requires UNIFORM spacing "
                f"including the midnight wrap, got {raw!r} "
                f"(minute gaps {gaps!r})"
            )
        return slots, raw_minutes is not None

    # ------------------------------------------------------------- queries

    @property
    def genesis_stamp(self) -> int:
        """The lane's first scheduled stamp (genesis day, first mark)."""
        return self._genesis_stamp

    @property
    def minute_keyed(self) -> bool:
        """True iff any era declares slot_minutes_utc -- the lane writes
        'YYYY-MM-DDTHHMM' observation keys for its WHOLE history (one
        methodology id = one key format forever)."""
        return self._minute_keyed

    @property
    def slot_spacing_minutes(self) -> int:
        """The FINAL era's uniform mark spacing in minutes (the spacing
        the dynamic-weights horizon validator checks against)."""
        slots = self._eras[-1][2]
        if len(slots) > 1:
            return slots[1] - slots[0]
        return MINUTES_PER_DAY

    def stamp_key(self, stamp: int) -> str:
        """This lane's observation key for `stamp` (format follows
        minute_keyed)."""
        return stamp_to_obs_key(stamp, minute_keyed=self._minute_keyed)

    def _era_for_day(self, day_ordinal: int) -> Optional[int]:
        if day_ordinal < self._start_days[0]:
            return None
        return bisect_right(self._start_days, day_ordinal) - 1

    def is_scheduled(self, stamp: int) -> bool:
        """True iff `stamp` is a scheduled observation stamp of its era
        (always False before genesis)."""
        day_ordinal, minute = divmod(int(stamp), MINUTES_PER_DAY)
        idx = self._era_for_day(day_ordinal)
        if idx is None:
            return False
        return minute in self._eras[idx][2]

    def prev_scheduled_stamp(self, stamp: int) -> Optional[int]:
        """The last scheduled stamp STRICTLY before `stamp` -- the R-cutoff
        information boundary of the observation at `stamp`. Era-aware:
        across an era boundary the first new-era stamp's cutoff is the
        last old-era stamp. None when nothing scheduled precedes (the
        genesis observation has no cutoff -- every score there is
        undefined, never zero)."""
        target = int(stamp) - 1
        for start_day, end_day, slots in reversed(self._eras):
            era_start = start_day * MINUTES_PER_DAY
            if era_start > target:
                continue
            top = target
            if end_day is not None:
                top = min(top, (end_day - 1) * MINUTES_PER_DAY + slots[-1])
            day_ordinal, minute = divmod(top, MINUTES_PER_DAY)
            idx = bisect_right(slots, minute) - 1
            if idx >= 0:
                return day_ordinal * MINUTES_PER_DAY + slots[idx]
            if day_ordinal - 1 >= start_day:
                return (day_ordinal - 1) * MINUTES_PER_DAY + slots[-1]
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
            lo = max(lo_bound, start_day * MINUTES_PER_DAY)
            hi = (
                hi_bound
                if end_day is None
                else min(hi_bound, end_day * MINUTES_PER_DAY)
            )
            if hi <= lo:
                continue
            for day_ordinal in range(
                lo // MINUTES_PER_DAY, (hi - 1) // MINUTES_PER_DAY + 1
            ):
                base = day_ordinal * MINUTES_PER_DAY
                for minute in slots:
                    s = base + minute
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
