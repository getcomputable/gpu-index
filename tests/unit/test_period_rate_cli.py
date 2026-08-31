# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for scripts/compute_period_rate.py (the read-only section
6.1 report CLI).

Pins: the CLI is READ-ONLY (zero puts, ever); JSON goes to stdout and
notices to stderr so the report pipes clean; unpublished stamps never
GET (the LIST-then-targeted-GET read budget); the backward context walk
stops once enough filled stamps precede the period; the frontier clip
records period.clipped_at and refuses an all-future period; day-form
period boundaries expand to T00; refusals exit 2 with an error line,
never a traceback.

Conventions follow test_panel_cli.py: FakeS3 in-memory client, the CLI
loaded via importlib and wired with monkeypatched BucketConfig /
make_client / utc_now -- no wall-clock reads anywhere.
"""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

GENESIS = "2026-08-10"
PREFIX = "index/h100_sxm"
MID = "h100_sxm_wt_calc_v1"
NOW = datetime(2026, 8, 12, 23, 30, tzinfo=timezone.utc)


class _NoSuchKey(Exception):
    def __init__(self):
        super().__init__("missing")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.put_order = []
        self.get_keys = []
        self.list_prefixes = []

    def get_object(self, Bucket, Key):
        self.get_keys.append(Key)
        if Key not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
        self.put_order.append(Key)

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        self.list_prefixes.append(Prefix)
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


def _config():
    """The migrated-lane shape from test_panel_cli: 4-slot basket era
    2026-08-10..11, hourly observatory era from 2026-08-12."""
    return {
        "panel_id": "h100_sxm_wt",
        "bucket_prefix": PREFIX,
        "genesis_date": GENESIS,
        "record_sources": [
            {
                "kind": "basket",
                "prefix": "index/b300_basket",
                "from_date": GENESIS,
                "to_date": "2026-08-11",
            },
            {
                "kind": "observatory",
                "prefix": "index/raw_observatory",
                "from_date": "2026-08-12",
            },
        ],
        "slot_grids": [
            {"from_date": GENESIS, "slot_hours_utc": [4, 10, 16, 22]},
            {"from_date": "2026-08-12", "slot_hours_utc": list(range(24))},
        ],
        "drift_scan_observations": 48,
        "members": [
            {"source_id": "alpha", "weight": 0.5, "skus": ["H100"]},
            {"source_id": "bravo", "weight": 0.3, "skus": ["H100"]},
            {"source_id": "charlie", "weight": 0.2, "skus": ["H100"]},
        ],
        "calc": {
            "methodology_id": MID,
            "min_sources_to_claim": 2,
            "eligible_tiers": ["on-demand"],
            "filter_window": 20,
            "filter_sigma": 3.0,
            "filter_warmup": 10,
            "filter_sigma_floor": 0.05,
            "filter_terms": "recorded_currency",
            "composite_statistic": "median_ci_votes",
            "fx_lane": "ecb",
            "manual_exclusions": [],
            "jump_screen": {
                "quarantine_pct": 25.0,
                "corroborate_pct": 10.0,
                "min_corroborators": 2,
                "reference_max_lookback": 24,
            },
            "dynamic_weights": {
                "scheme": "predictive_v1",
                "lookback_horizons_hours": [1, 2],
                "forward_horizons_hours": [1, 2],
                "history_days": 2,
                "half_life_days": 1,
                "ridge_lambda": 0.001,
                "gamma": 4.0,
                "weight_min": 0.025,
                "weight_max": 0.5,
                "min_train_samples": 3,
                "target_variance_floor": 1e-12,
                "switch_min_eligible": 3,
                "max_abs_log_return": 0.5,
                "source_weight_caps": {},
                "attendance_floor": 0.5,
            },
        },
    }


def _write_config(tmp_path):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(_config()))
    return path


def _key(stamp):
    return f"{PREFIX}/composites/{MID}/{stamp}.json"


def _artifact(stamp, value=None, missed=False, quarantined=None):
    index = None if value is None else {"value_usd_gpu_hr": value}
    return {
        "date": stamp,
        "methodology_id": MID,
        "index": index,
        "observation_missed": missed,
        "record_quarantined": quarantined,
        "panel_dark": index is None,
    }


def _seed(client, stamp, **kwargs):
    client.objects[_key(stamp)] = json.dumps(_artifact(stamp, **kwargs)).encode()


def _seed_world(client):
    """Basket-era slots published with values; hourly 08-12: hours 00-05
    published (03 an explicit missed artifact, 04 dark), 06 UNPUBLISHED,
    07-21 published, 22 (closed at NOW 23:30) unpublished."""
    ramp = {4: 7.0, 10: 7.2, 16: 7.4, 22: 7.6}
    for day in ("2026-08-10", "2026-08-11"):
        for hour, value in ramp.items():
            _seed(client, f"{day}T{hour:02d}", value=value + 0.0)
    for hour in (0, 1, 2, 5):
        _seed(client, f"2026-08-12T{hour:02d}", value=8.0)
    _seed(client, "2026-08-12T03", missed=True)
    _seed(client, "2026-08-12T04")  # dark: published, index null
    for hour in range(7, 22):
        _seed(client, f"2026-08-12T{hour:02d}", value=8.0)


_CLI_CACHE = {}


def _load_cli():
    if "mod" not in _CLI_CACHE:
        spec = importlib.util.spec_from_file_location(
            "compute_period_rate",
            REPO_ROOT / "scripts" / "compute_period_rate.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _run(monkeypatch, client, argv, now=NOW):
    cli = _load_cli()

    class StubConfig:
        bucket = "curves"

    monkeypatch.setattr(
        cli.BucketConfig, "from_env", staticmethod(lambda: StubConfig())
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: client)
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    return cli.main(argv)


def _report(capsys):
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


def test_check_config_offline(monkeypatch, capsys, tmp_path):
    cli = _load_cli()

    def _boom():
        raise AssertionError("check-config must not touch bucket creds")

    monkeypatch.setattr(cli.BucketConfig, "from_env", staticmethod(_boom))
    assert cli.main(["--config", str(_write_config(tmp_path)), "--check-config"]) == 0
    assert "config ok" in capsys.readouterr().err


def test_report_happy_path_reads_and_shape(monkeypatch, capsys, tmp_path):
    client = FakeS3()
    _seed_world(client)
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T00",
            "--end", "2026-08-12T12",
            "--series",
        ],
    )
    assert rc == 0
    report, err = _report(capsys)
    assert "NOTICE" in err or "::notice::" in err

    # 12 scheduled; missing = T03 (missed), T04 (dark), T06 (unpublished).
    assert report["coverage"]["scheduled"] == 12
    assert report["coverage"]["filled"] == 9
    assert report["coverage"]["causes"] == {
        "missed": 1, "dark": 1, "quarantined": 0, "unpublished": 1
    }
    gaps = report["coverage"]["gaps"]
    assert [(g["start"], g["length"]) for g in gaps] == [
        ("2026-08-12T03", 2), ("2026-08-12T06", 1)
    ]
    # Gap 1 (G=2): last two filled = T01, T02 (both 8.0).
    assert gaps[0]["fill"] == {
        "value_usd_gpu_hr": 8.0, "window_stamps": 2,
        "window_start": "2026-08-12T01", "window_end": "2026-08-12T02",
    }
    assert report["band"] == "determination"
    assert report["period_rate"]["value_usd_gpu_hr"] == 8.0
    assert report["period"]["clipped_at"] is None
    assert len(report["series"]) == 12

    # Read budget: ONE list of the composite keyspace; the unpublished
    # stamp T06 never GETs. Published context stamps before the period
    # may GET (backward walk), unpublished ones never.
    assert client.list_prefixes == [f"{PREFIX}/composites/{MID}/"]
    assert _key("2026-08-12T06") not in client.get_keys
    assert not client.put_order  # READ-ONLY, always


def test_backward_walk_fills_from_before_period(monkeypatch, capsys, tmp_path):
    client = FakeS3()
    _seed_world(client)
    # Period starts at the unpublished T06: the gap's window must come
    # from stamps BEFORE the period (T05=8.0, T02=8.0 ... plus basket-era
    # stamps if needed).
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T06",
            "--end", "2026-08-12T09",
        ],
    )
    assert rc == 0
    report, _ = _report(capsys)
    gap = report["coverage"]["gaps"][0]
    assert gap["start"] == "2026-08-12T06"
    # G=1 -> the single nearest preceding filled stamp, T05.
    assert gap["fill"]["window_start"] == "2026-08-12T05"
    assert gap["fill"]["window_stamps"] == 1
    assert report["period_rate"]["stamps_carried"] == 1


def test_frontier_clip_and_day_form_boundaries(monkeypatch, capsys, tmp_path):
    client = FakeS3()
    _seed_world(client)
    # Day-form boundaries expand to T00; end is beyond NOW (23:30 ->
    # last closed stamp is T22, clip at T23).
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12",
            "--end", "2026-08-13",
        ],
    )
    assert rc == 0
    report, err = _report(capsys)
    assert report["period"]["clipped_at"] == "2026-08-12T23"
    assert report["period"]["end"] == "2026-08-12T23"
    assert report["coverage"]["scheduled"] == 23
    assert "clipping" in err


def test_all_future_period_refuses(monkeypatch, capsys, tmp_path):
    client = FakeS3()
    _seed_world(client)
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-13",
            "--end", "2026-08-14",
        ],
    )
    assert rc == 2
    assert "empty after the frontier clip" in capsys.readouterr().err


def test_bad_stamp_refuses_without_traceback(monkeypatch, capsys, tmp_path):
    client = FakeS3()
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T99",
            "--end", "2026-08-13",
        ],
    )
    assert rc == 2
    assert "YYYY-MM-DDTHH" in capsys.readouterr().err


def test_out_writes_file(monkeypatch, capsys, tmp_path):
    client = FakeS3()
    _seed_world(client)
    out_path = tmp_path / "report.json"
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T00",
            "--end", "2026-08-12T12",
            "--out", str(out_path),
        ],
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["coverage"]["scheduled"] == 12
    # Nothing on stdout when --out is set.
    assert capsys.readouterr().out == ""


def test_era_stitched_period_spans_grids(monkeypatch, capsys, tmp_path):
    client = FakeS3()
    _seed_world(client)
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-11T00",
            "--end", "2026-08-12T06",
        ],
    )
    assert rc == 0
    report, _ = _report(capsys)
    # 08-11: four 4-slot stamps; 08-12: six hourly stamps.
    assert report["coverage"]["scheduled"] == 10
    assert report["coverage"]["filled"] == 8
    assert report["band"] == "determination"


def test_backward_walk_collects_enough_filled_context(monkeypatch, capsys, tmp_path):
    """A two-stamp gap at the period START needs TWO filled context
    stamps."""
    client = FakeS3()
    _seed_world(client)
    # Period starts at the missed/dark pair: [T03, T06).
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T03",
            "--end", "2026-08-12T06",
        ],
    )
    assert rc == 0
    report, _ = _report(capsys)
    gap = report["coverage"]["gaps"][0]
    assert gap["start"] == "2026-08-12T03"
    assert gap["length"] == 2
    # G=2 -> the last TWO filled stamps before the period (T01, T02).
    assert gap["fill"] == {
        "value_usd_gpu_hr": 8.0,
        "window_stamps": 2,
        "window_start": "2026-08-12T01",
        "window_end": "2026-08-12T02",
    }


def test_sparse_era_slot_stays_open_until_its_next_mark(
    monkeypatch, capsys, tmp_path
):
    """On the 4-slot era a slot's window spans six hours: at 14:30Z the
    10Z slot is still OPEN (its next mark is 16Z) and must not be
    counted as missing."""
    client = FakeS3()
    _seed_world(client)
    now = datetime(2026, 8, 11, 14, 30, tzinfo=timezone.utc)
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-11",
            "--end", "2026-08-12",
        ],
        now=now,
    )
    assert rc == 0
    report, err = _report(capsys)
    # Last CLOSED stamp is the 04Z slot; the 10Z slot is open. The clip
    # label is the first NOT-closed scheduled mark (grid-aware frontier,
    # 15-min re-base 2026-08-27) -- the pre-re-base label was the hour
    # after last_closed, an hour-lattice artifact this sparse-grid test
    # existed to expose.
    assert report["period"]["clipped_at"] == "2026-08-11T10"
    assert report["coverage"]["scheduled"] == 1
    assert report["coverage"]["filled"] == 1
    assert "clipping" in err


def test_malformed_artifact_refuses_exit_two(monkeypatch, capsys, tmp_path):
    """A published artifact violating its own invariants refuses with
    an ERROR line and exit 2 -- never a traceback."""
    client = FakeS3()
    _seed_world(client)
    client.objects[_key("2026-08-12T09")] = json.dumps(
        {
            "date": "2026-08-12T09",
            "methodology_id": MID,
            "index": {"value_usd_gpu_hr": 8.0},
            "observation_missed": True,
            "record_quarantined": None,
            "panel_dark": False,
        }
    ).encode()
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T08",
            "--end", "2026-08-12T11",
        ],
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "report refused" in err
    assert "Traceback" not in err


def test_missing_bucket_env_refuses_exit_two(monkeypatch, capsys, tmp_path):
    cli = _load_cli()

    class Boom(Exception):
        pass

    def _raise():
        raise Boom("Bucket credentials missing/empty: ['GPU_INDEX_S3_BUCKET']")

    monkeypatch.setattr(cli.BucketConfig, "from_env", staticmethod(_raise))
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    rc = cli.main(
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T00",
            "--end", "2026-08-12T06",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "READ-ONLY" in err
    assert "Traceback" not in err


def test_malformed_json_artifact_refuses_exit_two(monkeypatch, capsys, tmp_path):
    """A listed artifact with unparseable bytes refuses (exit 2), never
    tracebacks -- JSONDecodeError used to bypass the refusal handler."""
    client = FakeS3()
    _seed_world(client)
    client.objects[_key("2026-08-12T09")] = b"{not json"
    rc = _run(
        monkeypatch,
        client,
        [
            "--config", str(_write_config(tmp_path)),
            "--start", "2026-08-12T08",
            "--end", "2026-08-12T11",
        ],
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "report refused" in err
    assert "Traceback" not in err
