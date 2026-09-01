# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Minute-lattice re-base invariants (15-min cadence design 2026-08-27).

The re-base's governing claim is that hour-grid lanes are BYTE-UNCHANGED:
the lattice moved under them (stamps became day*1440 + minute), but every
schedule quantity, every artifact string, and every embedded calc_params
byte is identical. These tests pin that claim structurally — the
minute-lattice schedule of every REAL baked config must equal an
independently-computed lattice reference, the key formats must
round-trip and refuse cross-grain traffic, and the era-stitch across an
hourly -> 15-minute boundary must behave exactly like the 4-slot ->
hourly precedent one level denser.

The unit-audit rule these tests enforce mechanically: additive hour
literals (stamp + 73, < 24 guards) hide from grep-for-*24 sweeps, so the
schedule is verified against a reference built from first principles,
never against the implementation's own arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_index.index.panel_config import load_panel_config, panel_schedule
from gpu_index.index.panel_schedule import (
    MINUTES_PER_DAY,
    PanelSchedule,
    PanelScheduleError,
    date_hour_to_stamp,
    date_minute_to_stamp,
    hour_iso_to_stamp,
    obs_key_to_stamp,
    stamp_to_date_hour,
    stamp_to_date_minute,
    stamp_to_hour_iso,
    stamp_to_obs_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_CONFIG_PATHS = sorted(
    (REPO_ROOT / "config").glob("index_panel_*.json")
)
MINUTE_KEYED_PANEL_STEMS = {
    "index_panel_b200",
    "index_panel_b300",
    "index_panel_h100_sxm",
    "index_panel_h200_sxm",
}


# ------------------------------------------------- lattice equivalence


def _reference_stamps(config: dict, day_span: int) -> list[int]:
    """Scheduled stamps rebuilt from FIRST PRINCIPLES on the minute
    lattice: day ordinals walked by hand, era resolution by linear scan,
    hour marks x60 or minute marks verbatim — sharing no arithmetic with
    PanelSchedule."""
    from datetime import date, timedelta

    genesis = date.fromisoformat(config["genesis_date"])
    eras = []
    for grid in config["slot_grids"]:
        if "slot_hours_utc" in grid:
            marks = [int(hour) * 60 for hour in grid["slot_hours_utc"]]
        else:
            marks = [int(minute) for minute in grid["slot_minutes_utc"]]
        eras.append((date.fromisoformat(grid["from_date"]), marks))
    out = []
    for offset in range(day_span):
        day = genesis + timedelta(days=offset)
        grid = None
        for from_date, marks in eras:
            if from_date <= day:
                grid = marks
        if grid is None:
            continue
        for mark in grid:
            out.append(day.toordinal() * MINUTES_PER_DAY + mark)
    return out


@pytest.mark.parametrize(
    "config_path", PANEL_CONFIG_PATHS, ids=lambda p: p.stem
)
def test_real_config_schedule_matches_first_principles(config_path):
    config = load_panel_config(config_path)
    schedule = panel_schedule(config)
    span_days = 30
    expected = _reference_stamps(config, span_days)
    genesis = schedule.genesis_stamp
    got = schedule.scheduled_stamps(
        genesis, genesis - (genesis % MINUTES_PER_DAY) + span_days * MINUTES_PER_DAY
    )
    assert got == [s for s in expected if s >= genesis]
    expected_minute_keyed = config_path.stem in MINUTE_KEYED_PANEL_STEMS
    assert schedule.minute_keyed is expected_minute_keyed
    if expected_minute_keyed:
        assert any(s % 60 != 0 for s in got)
    else:
        assert all(s % 60 == 0 for s in got)
    # prev_scheduled_stamp is exactly the predecessor in the flat list.
    for idx in range(1, min(len(got), 200)):
        assert schedule.prev_scheduled_stamp(got[idx]) == got[idx - 1]
    assert schedule.prev_scheduled_stamp(got[0]) is None


@pytest.mark.parametrize(
    "config_path", PANEL_CONFIG_PATHS, ids=lambda p: p.stem
)
def test_real_config_calc_params_embed_grid_bytes_verbatim(config_path):
    """The D2 fence embeds slot_grids exactly as written, including each
    era's hour/minute vocabulary and values."""
    from gpu_index.index.panel import panel_calc_params

    config = load_panel_config(config_path)
    raw = json.loads(config_path.read_text())
    params = panel_calc_params(config)
    assert params["slot_grids"] == raw["slot_grids"]


# ------------------------------------------------------- era stitching


def _stitched_config_schedule() -> PanelSchedule:
    """A lane that lived 4-slot, walked to hourly, then to 15-minute —
    the b300 shape one era deeper."""
    return PanelSchedule(
        genesis_date="2026-08-10",
        slot_grids=[
            {"from_date": "2026-08-10", "slot_hours_utc": [4, 10, 16, 22]},
            {
                "from_date": "2026-08-24",
                "slot_hours_utc": list(range(24)),
            },
            {
                "from_date": "2026-09-01",
                "slot_minutes_utc": list(range(0, 1440, 15)),
            },
        ],
    )


def test_minute_era_boundary_cutoff_is_the_last_hourly_stamp():
    schedule = _stitched_config_schedule()
    assert schedule.minute_keyed
    assert schedule.slot_spacing_minutes == 15
    first_minute_stamp = date_minute_to_stamp("2026-09-01", 0)
    last_hourly_stamp = date_hour_to_stamp("2026-08-31", 23)
    assert schedule.is_scheduled(first_minute_stamp)
    assert schedule.prev_scheduled_stamp(first_minute_stamp) == last_hourly_stamp
    # The first sub-hour mark's cutoff is the day's :00.
    quarter = date_minute_to_stamp("2026-09-01", 15)
    assert schedule.prev_scheduled_stamp(quarter) == first_minute_stamp
    # :15 marks are NOT scheduled in the hourly era.
    assert not schedule.is_scheduled(date_minute_to_stamp("2026-08-25", 15))
    # 4-slot -> hourly boundary still stitches exactly as before.
    assert schedule.prev_scheduled_stamp(
        date_hour_to_stamp("2026-08-24", 0)
    ) == date_hour_to_stamp("2026-08-23", 22)


def test_minute_era_attendance_window_spans_the_boundary():
    schedule = _stitched_config_schedule()
    # Window [08-31T22:00, 09-01T01:00): two hourly stamps (22, 23) plus
    # four 15-min stamps (00:00..00:45) = 6 scheduled marks.
    window = schedule.scheduled_stamps(
        date_hour_to_stamp("2026-08-31", 22),
        date_minute_to_stamp("2026-09-01", 60),
    )
    assert window == [
        date_hour_to_stamp("2026-08-31", 22),
        date_hour_to_stamp("2026-08-31", 23),
        date_minute_to_stamp("2026-09-01", 0),
        date_minute_to_stamp("2026-09-01", 15),
        date_minute_to_stamp("2026-09-01", 30),
        date_minute_to_stamp("2026-09-01", 45),
    ]


def test_minute_grid_requires_uniform_spacing_and_one_vocabulary():
    with pytest.raises(PanelScheduleError, match="UNIFORM spacing"):
        PanelSchedule(
            genesis_date="2026-09-01",
            slot_grids=[
                {
                    "from_date": "2026-09-01",
                    "slot_minutes_utc": [0, 15, 40],
                }
            ],
        )
    with pytest.raises(PanelScheduleError, match="exactly one of"):
        PanelSchedule(
            genesis_date="2026-09-01",
            slot_grids=[
                {
                    "from_date": "2026-09-01",
                    "slot_hours_utc": [0],
                    "slot_minutes_utc": [0, 720],
                }
            ],
        )
    with pytest.raises(PanelScheduleError, match="0..1439"):
        PanelSchedule(
            genesis_date="2026-09-01",
            slot_grids=[
                {"from_date": "2026-09-01", "slot_minutes_utc": [0, 1440]}
            ],
        )


# ----------------------------------------------------- key format rules


def test_obs_key_round_trips_both_formats():
    hour_stamp = date_hour_to_stamp("2026-09-01", 16)
    assert stamp_to_obs_key(hour_stamp, minute_keyed=False) == "2026-09-01T16"
    assert stamp_to_obs_key(hour_stamp, minute_keyed=True) == "2026-09-01T1600"
    assert obs_key_to_stamp("2026-09-01T16") == hour_stamp
    assert obs_key_to_stamp("2026-09-01T1600") == hour_stamp
    quarter = date_minute_to_stamp("2026-09-01", 16 * 60 + 45)
    assert stamp_to_obs_key(quarter, minute_keyed=True) == "2026-09-01T1645"
    assert obs_key_to_stamp("2026-09-01T1645") == quarter


def test_hour_keyed_lane_refuses_sub_hour_stamps():
    quarter = date_minute_to_stamp("2026-09-01", 16 * 60 + 15)
    with pytest.raises(PanelScheduleError, match="not\\s+hour-aligned"):
        stamp_to_obs_key(quarter, minute_keyed=False)
    with pytest.raises(PanelScheduleError, match="not\\s+hour-aligned"):
        stamp_to_hour_iso(quarter)
    with pytest.raises(PanelScheduleError, match="not\\s+hour-aligned"):
        stamp_to_date_hour(quarter)


def test_obs_key_parser_is_loud_on_malformed_stamps():
    for bad in (
        "2026-09-01",  # day-grained
        "2026-09-01T160",  # 14 chars
        "2026-09-01T1660",  # minute 60
        "2026-09-01T24",  # hour 24
        "2026-09-01T1600\n",  # trailing newline
        "2026-09-01 16",  # no T
    ):
        with pytest.raises(PanelScheduleError):
            obs_key_to_stamp(bad)
    # The strict hour-name helper refuses the minute form by contract.
    with pytest.raises(PanelScheduleError):
        hour_iso_to_stamp("2026-09-01T1600")


def test_stamp_key_follows_the_lane_format():
    schedule = _stitched_config_schedule()
    assert schedule.stamp_key(date_hour_to_stamp("2026-08-25", 5)) == (
        "2026-08-25T0500"
    )
    hourly_only = PanelSchedule(
        genesis_date="2026-08-23",
        slot_grids=[
            {"from_date": "2026-08-23", "slot_hours_utc": list(range(24))}
        ],
    )
    assert hourly_only.stamp_key(date_hour_to_stamp("2026-08-25", 5)) == (
        "2026-08-25T05"
    )


# ------------------------------------------- observatory token dual-era


def test_slot_token_formats_and_hour_refusal():
    from gpu_index.common.slots import slot_token

    assert slot_token(960, minute_tokens=False) == "slot16"
    assert slot_token(960, minute_tokens=True) == "slot1600"
    assert slot_token(975, minute_tokens=True) == "slot1615"
    with pytest.raises(ValueError, match="sub-hour"):
        slot_token(975, minute_tokens=False)


def test_day_slot_keys_normalizes_both_token_eras():
    from datetime import date as date_cls

    from gpu_index.common.store import day_slot_keys

    class _NoSuchKey(Exception):
        pass

    class FakeS3:
        def __init__(self, keys):
            self.keys = keys

        def list_objects_v2(self, Bucket, Prefix, **kwargs):
            keys = sorted(k for k in self.keys if k.startswith(Prefix))
            return {
                "Contents": [{"Key": k} for k in keys],
                "IsTruncated": False,
            }

    day_prefix = "index/raw_observatory/snapshots/2026-09-01"
    client = FakeS3(
        [
            # The format-cutover shape: the pre-roll writer's hour token
            # and a post-roll self-heal's minute token name the SAME
            # 14:00 mark; the hour token is the earlier capture (the old
            # writer cannot fire after the roll) AND sorts first.
            f"{day_prefix}/slot14-20260901T140200Z-aa.json",
            f"{day_prefix}/slot1400-20260901T140800Z-bb.json",
            f"{day_prefix}/slot1415-20260901T141700Z-cc.json",
            f"{day_prefix}/slot16-20260901T160200Z-dd.json",
            # Out-of-range tokens are skipped like non-slot keys.
            f"{day_prefix}/slot2460-20260901T000000Z-ee.json",
            f"{day_prefix}/slot1499-20260901T000000Z-ff.json",
        ]
    )
    keys = day_slot_keys(
        client,
        "curves",
        prefix="index/raw_observatory",
        day=date_cls(2026, 9, 1),
    )
    assert keys == {
        14 * 60: f"{day_prefix}/slot14-20260901T140200Z-aa.json",
        14 * 60 + 15: f"{day_prefix}/slot1415-20260901T141700Z-cc.json",
        16 * 60: f"{day_prefix}/slot16-20260901T160200Z-dd.json",
    }


def test_read_day_snapshots_refuses_minute_tokens():
    from datetime import date as date_cls

    from gpu_index.common.store import read_day_snapshots

    class FakeS3:
        def list_objects_v2(self, Bucket, Prefix, **kwargs):
            return {
                "Contents": [
                    {"Key": Prefix + "slot1415-20260901T141700Z-cc.json"}
                ],
                "IsTruncated": False,
            }

    with pytest.raises(ValueError, match="minute-grain slot key"):
        read_day_snapshots(
            FakeS3(),
            "curves",
            prefix="index/b300_basket",
            day=date_cls(2026, 9, 1),
        )


# ---------------------------------------------- capture grid vocabulary


def test_observatory_capture_grid_normalizes_both_vocabularies():
    from gpu_index.observatory.config import capture_grid

    hours_cfg = {
        "capture_slots_utc": [4, 10, 16, 22],
        "canonical_slot_utc": 16,
    }
    slots, canonical, minute_tokens = capture_grid(hours_cfg)
    assert slots == [240, 600, 960, 1320]
    assert canonical == 960
    assert minute_tokens is False

    minutes_cfg = {
        "capture_slot_minutes_utc": list(range(0, 1440, 15)),
        "canonical_slot_minute_utc": 960,
    }
    slots, canonical, minute_tokens = capture_grid(minutes_cfg)
    assert len(slots) == 96
    assert canonical == 960
    assert minute_tokens is True


def test_shipped_observatory_config_is_still_the_hour_vocabulary():
    """The SHIPPED capture grid is unchanged by the minute re-base.

    The lattice port lands the CAPABILITY, not a cadence change: densifying
    the capture grid multiplies the request volume this project sends to
    third-party providers. So the flip is a deliberate, separate edit --
    this pin fails loudly if one arrives without it.
    """
    from gpu_index.observatory.config import capture_grid, load_observatory_config

    cfg = load_observatory_config()
    assert "capture_slot_minutes_utc" not in cfg
    assert len(cfg["capture_slots_utc"]) == 24
    assert cfg["canonical_slot_utc"] == 16
    # The hour vocabulary still resolves onto the minute lattice.
    slots, canonical, minute_tokens = capture_grid(cfg)
    assert slots == [h * 60 for h in range(24)]
    assert canonical == 960
    assert minute_tokens is False


def test_a_96_mark_minute_config_validates_through_the_real_loader(tmp_path):
    """The capability half: a 15-minute capture config passes the SHIPPED
    validator end to end, so flipping the grid later is a config edit and
    never a code change."""
    from gpu_index.observatory.config import capture_grid, load_observatory_config

    base = json.loads(
        (REPO_ROOT / "config" / "raw_observatory.json").read_text()
    )
    del base["capture_slots_utc"]
    del base["canonical_slot_utc"]
    base["capture_slot_minutes_utc"] = list(range(0, 1440, 15))
    base["canonical_slot_minute_utc"] = 960
    base["per_source_deadline_seconds"] = 90
    base["capture_budget_seconds"] = 360
    cfg_path = tmp_path / "raw_observatory_15min.json"
    cfg_path.write_text(json.dumps(base))

    cfg = load_observatory_config(cfg_path)
    assert len(cfg["capture_slot_minutes_utc"]) == 96
    slots, canonical, minute_tokens = capture_grid(cfg)
    assert len(slots) == 96
    assert canonical == 960
    assert minute_tokens is True


def test_stamp_helpers_agree_on_the_minute_lattice():
    stamp = date_minute_to_stamp("2026-09-01", 1439)
    assert stamp_to_date_minute(stamp) == ("2026-09-01", 1439)
    assert date_hour_to_stamp("2026-09-01", 5) == date_minute_to_stamp(
        "2026-09-01", 300
    )


# ------------------------------------------------ fill-lookback ruling


def test_fill_lookback_is_72_wall_clock_hours_on_every_grid():
    """Founder ruling 2026-08-27: the section 6.1 L parameter stays 72
    WALL-CLOCK hours -- 72 stamps on an hourly lane (byte-identical to
    the pre-re-base behavior and the cross-repo vectors), 288 on a
    15-minute era, 12 on a pure 6-hourly grid."""
    from gpu_index.index.period_rate import FILL_LOOKBACK_HOURS, fill_lookback_stamps

    assert FILL_LOOKBACK_HOURS == 72
    hourly = PanelSchedule(
        genesis_date="2026-08-23",
        slot_grids=[
            {"from_date": "2026-08-23", "slot_hours_utc": list(range(24))}
        ],
    )
    assert fill_lookback_stamps(hourly) == 72
    assert fill_lookback_stamps(_stitched_config_schedule()) == 288
    four_slot = PanelSchedule(
        genesis_date="2026-08-10",
        slot_grids=[
            {"from_date": "2026-08-10", "slot_hours_utc": [4, 10, 16, 22]}
        ],
    )
    assert fill_lookback_stamps(four_slot) == 12


# ------------------------------------------- minute-lattice slot selection


class _ListOnlyS3:
    def __init__(self, keys):
        self.keys = list(keys)

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        keys = sorted(k for k in self.keys if k.startswith(Prefix))
        return {
            "Contents": [{"Key": k} for k in keys],
            "IsTruncated": False,
        }


def test_earliest_capture_wins_by_run_id_not_key_bytes():
    """The rollback inversion: a LATER hour-token
    capture must not beat an EARLIER minute-token capture just because
    'slot16-' sorts before 'slot1600-'."""
    from datetime import date as date_cls

    from gpu_index.common.store import day_slot_keys

    day_prefix = "index/raw_observatory/snapshots/2026-09-02"
    later_hour_token = f"{day_prefix}/slot16-20260902T171000Z-bb.json"
    earlier_minute_token = f"{day_prefix}/slot1600-20260902T160200Z-aa.json"
    keys = day_slot_keys(
        _ListOnlyS3([later_hour_token, earlier_minute_token]),
        "curves",
        prefix="index/raw_observatory",
        day=date_cls(2026, 9, 2),
    )
    assert keys == {960: earlier_minute_token}


def test_slot_already_captured_probes_both_tokens_both_directions():
    from datetime import date as date_cls

    from gpu_index.common.store import slot_already_captured

    day_prefix = "index/raw_observatory/snapshots/2026-09-02"
    minute_key = f"{day_prefix}/slot1600-20260902T160200Z-aa.json"
    hour_key = f"{day_prefix}/slot16-20260902T160200Z-aa.json"
    # Hour-vocab writer (config rollback) finds the minute-token capture.
    assert slot_already_captured(
        _ListOnlyS3([minute_key]),
        "curves",
        prefix="index/raw_observatory",
        day=date_cls(2026, 9, 2),
        minute_of_day=960,
        minute_tokens=False,
    )
    # Minute-vocab writer (cutover day) finds the hour-token capture.
    assert slot_already_captured(
        _ListOnlyS3([hour_key]),
        "curves",
        prefix="index/raw_observatory",
        day=date_cls(2026, 9, 2),
        minute_of_day=960,
        minute_tokens=True,
    )
    # A sub-hour mark has one representable token: one probe, no false hit.
    assert not slot_already_captured(
        _ListOnlyS3([minute_key]),
        "curves",
        prefix="index/raw_observatory",
        day=date_cls(2026, 9, 2),
        minute_of_day=975,
        minute_tokens=True,
    )


def test_period_boundary_minute_form_refused_on_hour_lane_at_parse():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compute_period_rate",
        REPO_ROOT / "scripts" / "compute_period_rate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(PanelScheduleError, match="hour-keyed"):
        mod.parse_period_stamp("2026-08-01T0030", minute_keyed=False)
    # The aligned minute form is refused too -- the hour lane's grammar
    # is 13 chars, full stop (silent acceptance would train bad habits).
    with pytest.raises(PanelScheduleError, match="hour-keyed"):
        mod.parse_period_stamp("2026-08-01T1600", minute_keyed=False)
    # A minute-keyed lane takes both forms.
    assert mod.parse_period_stamp(
        "2026-08-01T0030", minute_keyed=True
    ) == date_minute_to_stamp("2026-08-01", 30)
    assert mod.parse_period_stamp(
        "2026-08-01T16", minute_keyed=True
    ) == date_hour_to_stamp("2026-08-01", 16)


def test_minute_keyed_lane_refuses_without_live_lever(
    tmp_path, monkeypatch, capsys
):
    """PANEL_MINUTE_LANES_LIVE is the mechanical prerequisite gate,
    scoped to the hazard it names -- DUAL-PUBLISHING.

    A minute-keyed config validates offline (--check-config), replays
    read-only without any lever (--dry-run; --verify-published likewise,
    returning before the fence), and refuses to PUBLISH until the lever
    is set. The read-only modes write nothing, so there is no keyspace
    for them to dual-publish into -- and refusing them would fence off
    replaying the serving generation of a lane whose grid is
    minute-keyed, which is the one thing this repository exists for."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compute_panel_index_fence_test",
        REPO_ROOT / "scripts" / "compute_panel_index.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    base = json.loads(
        (REPO_ROOT / "config" / "index_panel_h100_sxm.json").read_text()
    )
    base["slot_grids"] = base["slot_grids"] + [
        {
            "from_date": "2026-09-01",
            "slot_minutes_utc": list(range(0, 1440, 15)),
        }
    ]
    base["calc"]["methodology_id"] = "h100_sxm_v1_calc_v99"
    cfg_path = tmp_path / "minute_lane.json"
    cfg_path.write_text(json.dumps(base))

    monkeypatch.delenv("PANEL_MINUTE_LANES_LIVE", raising=False)
    # Offline validation stays fence-free (the config-only PR gate).
    monkeypatch.setattr(
        "sys.argv",
        ["compute_panel_index.py", "--config", str(cfg_path), "--check-config"],
    )
    assert mod.main() == 0
    # A PUBLISHING run refuses loudly before touching the bucket.
    monkeypatch.setattr(
        "sys.argv",
        ["compute_panel_index.py", "--config", str(cfg_path), "--sync"],
    )
    assert mod.main() == 1
    # ... and says so in terms of publishing, so an operator reading the
    # refusal knows a read-only replay is available without the lever.
    assert "PUBLISHING" in capsys.readouterr().out

    # A READ-ONLY replay of the same lane runs with no lever at all. An
    # empty record yields observation_missed artifacts and writes
    # nothing -- the point is that the fence does not fire.
    monkeypatch.setenv("GPU_INDEX_DATA_DIR", str(tmp_path / "empty-record"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "compute_panel_index.py",
            "--config",
            str(cfg_path),
            "--observation",
            "2026-09-01T0015",
            "--dry-run",
        ],
    )
    assert mod.main() == 0
    assert "minute-keyed lane refused" not in capsys.readouterr().out
    assert not (tmp_path / "empty-record").exists() or not list(
        (tmp_path / "empty-record").rglob("*.json")
    )
