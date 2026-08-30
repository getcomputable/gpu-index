# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Unit tests for the hourly panel CLI and the panel composite store
helpers.

Pins: the observation-keyed store discipline (fixed-width YYYY-MM-DDTHH
keys, append-only first-write-wins, run_id-in-pointer-only, pointer
no-regress across hour stamps); the era-stitched replay loop (basket-era
4-slot days then observatory-era hourly days publishing artifacts for
EXACTLY the scheduled stamps); the closure rule (amended for early compose:
a stamp whose slot snapshot EXISTS on the record LIST composes on the
next firing, before its window closes at the next mark, byte-identical
to a late compose; a stamp with NO snapshot stays open until its mark,
and its observation_missed / record_quarantined artifact publishes only
once next mark + MISSED_PUBLISH_GRACE_MINUTES has passed -- a
within-grace firing defers, publishes nothing at or after the deferred
stamp, and exits 0);
publish-in-order refusal; the D2 calc_params-drift refusal; hour-scoped
and day-scoped exclusion pinning with conflict refusal; the
--max-observations valve resuming byte-identically; replay determinism
(idempotent re-sync, chunked-now restarts reconstructing state from
published artifacts byte-for-byte); the warn-only bounded drift scan,
GATED to the 16:00Z firing / --drift-scan; the BOUNDED replay (perf
stage): one LIST of the composite keyspace finds the publish frontier,
state rebuilds from the trailing state window with a single pre-window
seed GET, and bounded-vs-full artifacts are BYTE-IDENTICAL on a world
whose genesis predates the window; FX bounded loading + the
conditional (skip-if-covered) feed fetch; and --check-config running
with no bucket credentials at all.

Conventions follow test_index_composite.py: FakeS3 in-memory client,
the CLI loaded via importlib and wired with monkeypatched BucketConfig /
make_client / load_stored_rates / ensure_rates / utc_now -- no
wall-clock reads anywhere.
"""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_index.common.store import (
    BucketPublishError,
    panel_composite_exists,
    panel_composite_key,
    upload_panel_composite,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

GENESIS = "2026-08-10"
PREFIX = "index/h100_sxm"
BASKET_PREFIX = "index/b300_basket"
OBS_PREFIX = "index/raw_observatory"
MID = "h100_sxm_wt_calc_v1"

FX = {
    GENESIS: {
        "source": "ecb_reference_rate",
        "as_of": GENESIS,
        "rates": {"USD": 1.15},
    }
}

# The seeded world's clock marks (all UTC): at NOW1 the hourly grid is
# closed through T05 by marks, and T06 -- seeded but with its 07:00 mark
# unpassed -- closes EARLY on its existing snapshot (early compose), so the
# whole seeded world is computable at NOW1; NOW2 merely passes T06's
# mark (T07 has no snapshot and stays open at both clocks).
NOW1 = datetime(2026, 8, 12, 6, 30, tzinfo=timezone.utc)
NOW2 = datetime(2026, 8, 12, 7, 35, tzinfo=timezone.utc)

BASKET_STAMPS = [
    f"2026-08-{day}T{hour:02d}" for day in ("10", "11") for hour in (4, 10, 16, 22)
]
# T04 missed; T06 early-closes on its snapshot at NOW1.
HOURLY_STAMPS_NOW1 = [f"2026-08-12T{h:02d}" for h in range(7)]
COMPUTABLE_AT_NOW1 = BASKET_STAMPS + HOURLY_STAMPS_NOW1


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    # GITHUB_ACTIONS flips warn() to '::warning::' -- without the delenv
    # the WARNING string asserts fail ON CI ITSELF. The bucket vars must
    # be absent so --check-config provably needs no creds.
    for var in (
        "GITHUB_ACTIONS",
        "BASKET_CONFIG_PATH",
        "RAW_OBSERVATORY_CONFIG_PATH",
        "GPU_INDEX_S3_ENDPOINT",
        "GPU_INDEX_S3_REGION",
        "GPU_INDEX_S3_ACCESS_KEY",
        "GPU_INDEX_S3_SECRET_KEY",
        "GPU_INDEX_S3_BUCKET",
    ):
        monkeypatch.delenv(var, raising=False)


# ------------------------------------------------------------------ fakes


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
        # Read instrumentation for the bounded-replay pins: every GET key
        # (hit or miss) and every LIST prefix, in call order.
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

    def reset_reads(self):
        self.get_keys = []
        self.list_prefixes = []


# ---------------------------------------------------------------- fixtures


def _config():
    """A migrated-lane-shaped panel config: basket-era 4-slot record for
    2026-08-10..11, observatory-era hourly record from 2026-08-12."""
    return {
        "panel_id": "h100_sxm_wt",
        "bucket_prefix": PREFIX,
        "genesis_date": GENESIS,
        "record_sources": [
            {
                "kind": "basket",
                "prefix": BASKET_PREFIX,
                "from_date": GENESIS,
                "to_date": "2026-08-11",
            },
            {
                "kind": "observatory",
                "prefix": OBS_PREFIX,
                "from_date": "2026-08-12",
            },
        ],
        "slot_grids": [
            {"from_date": GENESIS, "slot_hours_utc": [4, 10, 16, 22]},
            {"from_date": "2026-08-12", "slot_hours_utc": list(range(24))},
        ],
        # TOP-LEVEL operational knob (never in calc/calc_params).
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


def _write_config(tmp_path, cfg=None, name="panel.json"):
    path = tmp_path / name
    path.write_text(json.dumps(cfg if cfg is not None else _config()))
    return path


def _row(usd=2.4, currency="USD"):
    return {
        "sku": "H100",
        "sku_identifier": "H100 SXM",
        "price_usd_gpu_hr": usd,
        "price_native_per_gpu_hr": usd,
        "currency": currency,
        "tier": "on-demand",
        "gpu_count_basis": 1,
        "raw_value": str(usd),
        "raw_unit": "usd_per_gpu_hr",
        "implausible": False,
        "notes": "",
    }


def _seed_slot(client, prefix, day, hour, prices=None, run_id=None):
    prices = prices or {"alpha": 2.4, "bravo": 2.5, "charlie": 2.6}
    run_id = run_id or f"{day.replace('-', '')}T{hour:02d}0500Z-aaaa"
    payload = {
        "sources": [
            {
                "source_id": sid,
                "status": "ok",
                "observations": [_row(usd) if not isinstance(usd, dict) else usd],
            }
            for sid, usd in prices.items()
        ],
        "run_id": run_id,
        "late_fill": False,
    }
    key = f"{prefix}/snapshots/{day}/slot{hour:02d}-{run_id}.json"
    client.objects[key] = json.dumps(payload).encode()


def _seed_world(client):
    """Basket-era days 08-10/08-11 (all four slots), observatory-era
    08-12 hours 0-3 and 5-6 (hour 4 deliberately MISSING; hour 6's mark
    has not passed at NOW1 but its snapshot exists, so it closes EARLY
    for early compose)."""
    for day in ("2026-08-10", "2026-08-11"):
        for hour in (4, 10, 16, 22):
            _seed_slot(client, BASKET_PREFIX, day, hour)
    for hour in (0, 1, 2, 3, 5, 6):
        _seed_slot(client, OBS_PREFIX, "2026-08-12", hour)


def _artifacts(client):
    # Day artifacts only: the latest.json pointer legitimately embeds
    # run_id/published_at, which differ across replay schedules.
    return {
        k: v
        for k, v in client.objects.items()
        if "/composites/" in k and not k.endswith("latest.json")
    }


# -------------------------------------------------------------------- CLI

_CLI_CACHE = {}


def _load_cli():
    if "mod" not in _CLI_CACHE:
        spec = importlib.util.spec_from_file_location(
            "compute_panel_index",
            REPO_ROOT / "scripts" / "compute_panel_index.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _wire_cli(monkeypatch, client, now, argv, fx_calls=None, stored_fx=None):
    cli = _load_cli()

    class StubConfig:
        bucket = "curves"

    monkeypatch.setattr(
        cli.BucketConfig, "from_env", staticmethod(lambda: StubConfig())
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: client)

    def _stored(client_, bucket_, *, prefix, from_day=None):
        if fx_calls is not None:
            fx_calls.append(("stored", prefix, from_day))
        return dict(stored_fx if stored_fx is not None else FX)

    def _feed(client_, bucket_, *, prefix, persist=True, from_day=None):
        if fx_calls is not None:
            fx_calls.append(("feed", prefix, persist, from_day))
        return dict(FX)

    monkeypatch.setattr(cli, "load_stored_rates", _stored)
    monkeypatch.setattr(cli, "ensure_rates", _feed)
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr("sys.argv", ["compute_panel_index.py", *argv])
    return cli


def _key(stamp):
    return f"{PREFIX}/composites/{MID}/{stamp}.json"


# ------------------------------------------------------ store discipline


def _panel_payload(observation="2026-08-12T05", value=2.45):
    return {
        "schema_version": 1,
        "kind": "index_panel_composite",
        "panel_id": "h100_sxm_wt",
        "methodology_id": MID,
        "date": observation,
        "observation_date": observation[:10],
        "observation_hour_utc": int(observation[11:]),
        "observation_missed": False,
        "panel_dark": False,
        "index": {"value_usd_gpu_hr": value, "sources_used_count": 3},
        "sources": [],
    }


def test_panel_composite_store_discipline():
    client = FakeS3()
    out = upload_panel_composite(
        client,
        "curves",
        _panel_payload(),
        prefix=PREFIX,
        run_id="20260812T061000Z-aaaa",
        now=datetime(2026, 8, 12, 6, 10, tzinfo=timezone.utc),
    )
    key = out["composite_key"]
    # Deterministic observation key -- fixed-width hour, no run_id.
    assert key == f"{PREFIX}/composites/{MID}/2026-08-12T05.json"
    assert client.put_order[-1].endswith("latest.json")  # pointer moved last
    assert panel_composite_exists(
        client, "curves", prefix=PREFIX, methodology_id=MID,
        observation="2026-08-12T05",
    )
    # run_id lives in the POINTER only -- artifact bytes are pure inputs.
    assert b"run_id" not in client.objects[key]
    pointer = json.loads(client.objects[f"{PREFIX}/composites/{MID}/latest.json"])
    assert pointer["run_id"] == "20260812T061000Z-aaaa"
    assert pointer["panel_id"] == "h100_sxm_wt"
    assert pointer["date"] == "2026-08-12T05"
    # Idempotent identical re-PUT under a different run...
    again = upload_panel_composite(
        client, "curves", _panel_payload(), prefix=PREFIX,
        run_id="20260812T061030Z-bbbb",
    )
    assert again["composite_key"] == key
    # ...but divergent bytes for the same observation fail loudly.
    with pytest.raises(BucketPublishError, match="append-only"):
        upload_panel_composite(
            client, "curves", _panel_payload(value=9.99), prefix=PREFIX,
            run_id="20260812T061100Z-cccc",
        )
    # Pointer never regresses across hour stamps: T06 then T04 keeps T06.
    upload_panel_composite(
        client, "curves", _panel_payload("2026-08-12T06", 2.46),
        prefix=PREFIX, run_id="r1",
    )
    kept = upload_panel_composite(
        client, "curves", _panel_payload("2026-08-12T04", 2.44),
        prefix=PREFIX, run_id="r2",
    )
    assert kept["status"] == "published_pointer_kept"
    pointer = json.loads(client.objects[f"{PREFIX}/composites/{MID}/latest.json"])
    assert pointer["date"] == "2026-08-12T06"


def test_panel_composite_key_refuses_non_observation_stamps():
    # A day-keyed payload routed through the panel path would silently
    # share a keyspace slot with a daily artifact -- refuse loudly. The
    # trailing-newline case is the security-review fullmatch pin:
    # re.match + '$' accepts '...T05\n' and would mint a SECOND, shadow
    # key for the same observation, defeating first-write-wins.
    for bad in (
        "2026-08-12",
        "2026-08-12T5",
        "2026-08-12T24",
        "20260812T05",
        "2026-08-12T05\n",
        "2026-08-12T05.json\n2026-08-12T05",
    ):
        with pytest.raises(ValueError, match="YYYY-MM-DDTHH"):
            panel_composite_key(PREFIX, MID, bad)
    assert panel_composite_key(PREFIX, MID, "2026-08-12T00").endswith(
        "/2026-08-12T00.json"
    )


# ----------------------------------------------------------- check-config


def test_cli_check_config_needs_no_creds(monkeypatch, capsys, tmp_path):
    """--check-config validates offline: BucketConfig is never touched
    (the autouse fixture guarantees no GPU_INDEX_S3_* env, so no real
    bucket is ever addressed)."""
    cli = _load_cli()
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["compute_panel_index.py", "--config", str(cfg_path), "--check-config"],
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "config OK" in out
    assert MID in out
    assert "no bucket access" in out
    # A malformed config refuses loudly (weights must sum to 1.0).
    bad = _config()
    bad["members"][0]["weight"] = 0.4
    bad_path = _write_config(tmp_path, bad, name="bad.json")
    monkeypatch.setattr(
        "sys.argv",
        ["compute_panel_index.py", "--config", str(bad_path), "--check-config"],
    )
    assert cli.main() == 1
    assert "panel config refused" in capsys.readouterr().out


# ------------------------------------------------------------ era replay


def test_cli_era_stitched_sync_publishes_exactly_the_scheduled_stamps(
    monkeypatch, capsys, tmp_path
):
    client = FakeS3()
    _seed_world(client)
    cfg_path = _write_config(tmp_path)
    fx_calls = []
    cli = _wire_cli(
        monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"], fx_calls
    )
    assert cli.main() == 0
    out = capsys.readouterr().out

    # Exactly the computable scheduled stamps -- basket-era 4-slot days
    # then hourly days; T06's mark (07:00) has not passed but its
    # snapshot exists, so it composes EARLY (early compose).
    assert set(_artifacts(client)) == {_key(s) for s in COMPUTABLE_AT_NOW1}
    assert "observations written: 15" in out
    t06 = json.loads(client.objects[_key("2026-08-12T06")])
    assert t06["observation_missed"] is False
    assert t06["panel_dark"] is False

    # The missing hour published an explicit dark artifact after close.
    missed = json.loads(client.objects[_key("2026-08-12T04")])
    assert missed["observation_missed"] is True
    assert missed["panel_dark"] is True
    assert missed["index"] is None
    assert "2026-08-12T04: PANEL_DARK [observation_missed]" in out

    # Era stitch: each artifact names the record it was priced from.
    assert json.loads(client.objects[_key("2026-08-10T04")])["record_kind"] == "basket"
    assert (
        json.loads(client.objects[_key("2026-08-12T00")])["record_kind"]
        == "observatory"
    )
    # Jump-screen reference walk-back: the genesis observation has none;
    # the first hourly observation references the LAST basket-era one
    # (era-stitched); T05 skips the missed T04 back to T03.
    genesis = json.loads(client.objects[_key("2026-08-10T04")])
    assert genesis["jump_screen"]["reference"] is None
    t00 = json.loads(client.objects[_key("2026-08-12T00")])
    assert t00["jump_screen"]["reference"] == "2026-08-11T22"
    t05 = json.loads(client.objects[_key("2026-08-12T05")])
    assert t05["jump_screen"]["reference"] == "2026-08-12T03"

    # FX read under the LANE's own prefix, never the record's; the stored
    # rates covered every unpublished day (walk-back), so the live feed
    # was never fetched (the conditional-fetch rule).
    assert fx_calls and all(call[1] == PREFIX for call in fx_calls)
    assert all(call[0] == "stored" for call in fx_calls)
    # Pointer sits at the newest observation (the early-composed T06).
    pointer = json.loads(client.objects[f"{PREFIX}/composites/{MID}/latest.json"])
    assert pointer["date"] == "2026-08-12T06"

    # NOW2 passes T06's mark: nothing new (it already composed early),
    # replay silent about the rest.
    cli = _wire_cli(monkeypatch, client, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 0" in out
    assert "DRIFT" not in out
    assert _key("2026-08-12T06") in client.objects

    # Idempotent re-sync: nothing written, bytes stand.
    before = _artifacts(client)
    cli = _wire_cli(monkeypatch, client, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 0" in out
    assert "DRIFT" not in out
    assert _artifacts(client) == before


def test_cli_replay_determinism_across_restarts(monkeypatch, capsys, tmp_path):
    """One-shot world vs a world synced in four chunks with a FRESH
    main() each time (fresh in-memory state -- the container-restart
    contract): byte-identical artifacts, including the first artifact
    computed AFTER a restart purely from replayed published state."""
    cfg_path = _write_config(tmp_path)
    world_a = FakeS3()
    _seed_world(world_a)
    cli = _wire_cli(monkeypatch, world_a, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    artifacts_a = _artifacts(world_a)
    assert len(artifacts_a) == 15

    world_b = FakeS3()
    _seed_world(world_b)
    for run_now in (
        datetime(2026, 8, 10, 17, 0, tzinfo=timezone.utc),  # T04,T10 by mark; T16 early
        datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        NOW1,
        NOW2,
    ):
        cli = _wire_cli(
            monkeypatch, world_b, run_now, ["--config", str(cfg_path), "--sync"]
        )
        assert cli.main() == 0
        capsys.readouterr()
    artifacts_b = _artifacts(world_b)
    assert set(artifacts_a) == set(artifacts_b)
    for key, blob in artifacts_a.items():
        assert artifacts_b[key] == blob, key


def _fence_world_config():
    """A 4-member single-era hourly lane for the fence/quarantine/holdout
    restart world: three lowest-eligible members plus a vast statistic
    seat whose record never carries population accounting (permanently
    held out -- the per-status replay gate under test). fx_lane none
    keeps the world FX-free."""
    cfg = _config()
    cfg["members"] = [
        {"source_id": "alpha", "weight": 0.4, "skus": ["H100"]},
        {"source_id": "bravo", "weight": 0.3, "skus": ["H100"]},
        {"source_id": "charlie", "weight": 0.2, "skus": ["H100"]},
        {
            "source_id": "vast",
            "weight": 0.1,
            "skus": ["H100"],
            "statistic": "vast_vwm_verified_us_ca_v2",
        },
    ]
    cfg["record_sources"] = [
        {"kind": "observatory", "prefix": OBS_PREFIX, "from_date": GENESIS}
    ]
    cfg["slot_grids"] = [{"from_date": GENESIS, "slot_hours_utc": list(range(24))}]
    cfg["calc"]["fx_lane"] = "none"
    return cfg


def _seed_fence_world(client):
    """Hours 0..14 of genesis day: stable prints through hour 11 (filter
    warm-up 10 completes), a sigma-fenced-but-not-jumped alpha outlier at
    hour 12 (+8.3% -- inside the 25% jump fence, outside the 3-sigma
    band over an all-2.4 window), an uncorroborated bravo jump at hour
    13 (+32%, quarantined), and a normal hour 14."""
    for hour in range(15):
        prices = {"alpha": 2.4, "bravo": 2.5, "charlie": 2.6, "vast": 2.0}
        if hour == 12:
            prices["alpha"] = 2.6
        if hour == 13:
            prices["bravo"] = 3.3
        _seed_slot(client, OBS_PREFIX, GENESIS, hour, prices=prices)


def test_cli_restarts_cross_fenced_quarantined_and_held_out_artifacts(
    monkeypatch, capsys, tmp_path
):
    """Coverage stage (recipe 2, CLI half): a world containing a
    sigma-FENCED print, a jump-QUARANTINED hour, and a permanently
    HELD-OUT statistic seat replays byte-identically when synced in four
    chunks with fresh main() state -- the chunk boundaries land exactly
    after the fenced artifact and after the quarantined one, so
    advance_panel_state_from_published's per-status gates (fenced:
    chosen+filter -> advances window AND weight series; quarantined:
    chosen without filter -> advances nothing; held_out: no chosen ->
    advances nothing) are all load-bearing for the parity."""
    cfg_path = _write_config(tmp_path, _fence_world_config(), name="fence.json")

    incremental = FakeS3()
    _seed_fence_world(incremental)
    # Every hour is seeded, so each chunk's LAST in-range hour closes
    # EARLY on its snapshot (early compose) -- the chunk clocks sit just past
    # each boundary hour's own stamp, keeping the boundaries exactly
    # after the fenced artifact (12) and the quarantined one (13).
    for run_now, written in (
        (datetime(2026, 8, 10, 4, 30, tzinfo=timezone.utc), 5),  # hours 0-4
        (datetime(2026, 8, 10, 12, 30, tzinfo=timezone.utc), 8),  # ..fenced 12
        (datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc), 1),  # quarantine 13
        (datetime(2026, 8, 10, 14, 30, tzinfo=timezone.utc), 1),  # hour 14
    ):
        cli = _wire_cli(
            monkeypatch, incremental, run_now, ["--config", str(cfg_path), "--sync"]
        )
        assert cli.main() == 0
        out = capsys.readouterr().out
        assert f"observations written: {written}" in out

    control = FakeS3()
    _seed_fence_world(control)
    cli = _wire_cli(
        monkeypatch, control,
        datetime(2026, 8, 10, 15, 30, tzinfo=timezone.utc),
        ["--config", str(cfg_path), "--sync"],
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "JUMP QUARANTINED" in out

    art_control = _artifacts(control)
    art_incremental = _artifacts(incremental)
    assert len(art_control) == 15
    assert set(art_control) == set(art_incremental)
    for key, blob in art_control.items():
        assert art_incremental[key] == blob, key

    # The world genuinely contains all three per-status shapes.
    t12 = json.loads(art_control[_key("2026-08-10T12")])
    by_sid = {s["source_id"]: s for s in t12["sources"]}
    assert by_sid["alpha"]["status"] == "ok"
    assert by_sid["alpha"]["filter"]["accepted"] is False  # sigma-fenced
    assert by_sid["alpha"]["filter"]["unfiltered"] is False
    assert t12["jump_screen"]["quarantined"] == []  # +8.3% < the 25% fence
    assert "alpha" not in t12["index"]["renormalized_weights"]
    t13 = json.loads(art_control[_key("2026-08-10T13")])
    by_sid = {s["source_id"]: s for s in t13["sources"]}
    assert by_sid["bravo"]["status"] == "uncorroborated_jump"
    assert by_sid["bravo"]["chosen"]["usd_per_gpu_hr"] == 3.3
    assert [q["source_id"] for q in t13["jump_screen"]["quarantined"]] == ["bravo"]
    for key in sorted(art_control):
        artifact = json.loads(art_control[key])
        vast = {s["source_id"]: s for s in artifact["sources"]}["vast"]
        assert vast["status"] == "held_out", key
        assert vast["held_out"]["reason"] == "no_population_accounting", key
    # The quarantined seat re-entered at hour 14 (one-observation cost).
    t14 = json.loads(art_control[_key("2026-08-10T14")])
    by_sid = {s["source_id"]: s for s in t14["sources"]}
    assert by_sid["bravo"]["status"] == "ok"
    assert by_sid["bravo"]["filter"]["accepted"] is True


def _dw_history_world_config():
    """The fence/quarantine/holdout world REARMED with the live-lane vote
    rule (ruling 2026-08-27): vote_sigma_source dw_history over the
    percent floor (the six-mint posture), fx_lane ecb so a currency
    change can CONFIRM across a restart boundary."""
    cfg = _fence_world_config()
    del cfg["calc"]["filter_sigma_floor"]
    cfg["calc"]["filter_sigma_floor_pct"] = 3.0
    # Floor split (ruling 2026-08-27): the armed world carries the vote
    # floor knob too, so the restart byte-identity pin exercises the
    # fence/vote floor SPLIT exactly as the six live mints run it.
    cfg["calc"]["vote_sigma_floor_pct"] = 3.0
    cfg["calc"]["vote_sigma_source"] = "dw_history"
    cfg["calc"]["fx_lane"] = "ecb"
    return cfg


def _seed_dw_history_world(client, hours=range(18)):
    """Hours 0..17 of genesis day: stable prints through hour 11 (warm-up
    10 completes), a pct-fence-held alpha outlier at hour 12 (+12.5% —
    inside the 25% jump fence, dev 0.3 > band 3.0 * (3% of 2.4) = 0.216),
    an uncorroborated bravo jump at hour 13 (+32%, quarantined), charlie
    repricing USD -> EUR over hours 14-16 (pending 1/3, 2/3, CONFIRMED at
    the third print), and a normal hour 17 (the first post-confirmation
    dw vote). ``hours`` narrows the seed to a sub-range so a chunked
    replay can land each capture at its own clock (capture-time
    seeding)."""
    for hour in hours:
        prices = {"alpha": 2.4, "bravo": 2.5, "charlie": 2.6, "vast": 2.0}
        if hour == 12:
            prices["alpha"] = 2.7
        if hour == 13:
            prices["bravo"] = 3.3
        if hour >= 14:
            prices["charlie"] = dict(
                _row(2.0), price_usd_gpu_hr=None, currency="EUR"
            )
        _seed_slot(client, OBS_PREFIX, GENESIS, hour, prices=prices)


def test_cli_replay_determinism_across_restarts_dw_history(
    monkeypatch, capsys, tmp_path
):
    """The restart-determinism pin with vote_sigma_source dw_history
    ARMED (ruling 2026-08-27): the vote tail now reads the weight-state
    PRICES series, so a restart must rebuild that series from published
    artifacts byte-exactly or the first post-restart vote forks. The
    world holds every per-status replay shape — a pct-fence-held print
    (in the series, held out of the index), a jump-quarantined hour
    (advances nothing), the permanently held-out vast seat — plus a
    currency change whose CONFIRMATION lands AFTER a restart boundary
    (the pending streak and the mismatch prints' series entries must
    both survive the rebuild). One-shot world A vs world B synced in
    four chunks with a fresh main() each and each capture landing at
    its own chunk's clock: identical artifact sets, every artifact
    byte-identical."""
    cfg_path = _write_config(
        tmp_path, _dw_history_world_config(), name="dw_history.json"
    )

    # Captures land AT THEIR CHUNK CLOCKS (capture-time seeding), so the
    # chunk counts are invariant to the closure rule: a mark that has
    # passed is closed under any rule, and the next hour's snapshot has
    # not landed yet, so it cannot close early either (early compose's
    # snapshot-exists closure on main merges cleanly into this lane and
    # would otherwise pull one extra stamp into the first chunk).
    incremental = FakeS3()
    for seed_hours, run_now, written, code in (
        # hours 0-12: warm-up + the fenced print (fence is not red).
        (range(0, 13), datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc), 13, 0),
        # hour 13: quarantine (loud, not red).
        (range(13, 14), datetime(2026, 8, 10, 14, 30, tzinfo=timezone.utc), 1, 0),
        # hours 14-15: currency mismatches (red, still published).
        (range(14, 16), datetime(2026, 8, 10, 16, 30, tzinfo=timezone.utc), 2, 1),
        # hours 16-17: confirmation (green) after the restart + normal.
        (range(16, 18), datetime(2026, 8, 10, 18, 30, tzinfo=timezone.utc), 2, 0),
    ):
        _seed_dw_history_world(incremental, seed_hours)
        cli = _wire_cli(
            monkeypatch, incremental, run_now,
            ["--config", str(cfg_path), "--sync"],
        )
        assert cli.main() == code
        out = capsys.readouterr().out
        assert f"observations written: {written}" in out

    control = FakeS3()
    _seed_dw_history_world(control)
    cli = _wire_cli(
        monkeypatch, control,
        datetime(2026, 8, 10, 18, 30, tzinfo=timezone.utc),
        ["--config", str(cfg_path), "--sync"],
    )
    assert cli.main() == 1  # the mismatch firings redden the one-shot too
    out = capsys.readouterr().out
    assert "JUMP QUARANTINED" in out
    assert "currency change CONFIRMED USD -> EUR" in out

    art_control = _artifacts(control)
    art_incremental = _artifacts(incremental)
    assert len(art_control) == 18
    assert set(art_control) == set(art_incremental)
    for key, blob in art_control.items():
        assert art_incremental[key] == blob, key

    # The world genuinely contains every shape the pin rides on.
    t12 = json.loads(art_control[_key("2026-08-10T12")])
    assert t12["calc_params"]["vote_sigma_source"] == "dw_history"
    assert t12["calc_params"]["filter_sigma_floor_pct"] == 3.0
    assert t12["calc_params"]["vote_sigma_floor_pct"] == 3.0
    assert "filter_sigma_floor" not in t12["calc_params"]
    by_sid = {s["source_id"]: s for s in t12["sources"]}
    assert by_sid["alpha"]["filter"]["accepted"] is False  # pct-fenced
    assert by_sid["alpha"]["filter"]["unfiltered"] is False
    t13 = json.loads(art_control[_key("2026-08-10T13")])
    by_sid = {s["source_id"]: s for s in t13["sources"]}
    assert by_sid["bravo"]["status"] == "uncorroborated_jump"
    t16 = json.loads(art_control[_key("2026-08-10T16")])
    by_sid = {s["source_id"]: s for s in t16["sources"]}
    assert by_sid["charlie"]["filter"]["currency_confirmed"] is True
    # The confirmed vote reads the EUR-scoped dw slice (the two mismatch
    # prints, entered at their own stamps): frozen -> floored at 3% of
    # the 2.0 NATIVE print, converted at the day's 1.15 rate.
    assert by_sid["charlie"]["vote"] == {
        "sigma": 0.0,
        "sigma_floored": True,
        "conf_usd_gpu_hr": 0.069,
    }
    for key in sorted(art_control):
        artifact = json.loads(art_control[key])
        vast = {s["source_id"]: s for s in artifact["sources"]}["vast"]
        assert vast["status"] == "held_out", key
    # The last chunk's FIRST artifact (hour 16, computed fresh after the
    # restart): alpha's vote must carry the REAL dw sigma — the fenced
    # 2.7 lives in the REBUILT series ([2.4]*12 + 2.7 + [2.4]*3 ->
    # pstdev 0.07261843774138912 -> 0.072618, a hair over the 0.072
    # floor) — so a state-rebuild fork would show here first. (By hour
    # 17 one more flat print dilutes the sigma back under the floor.)
    t16i = json.loads(art_incremental[_key("2026-08-10T16")])
    by_sid = {s["source_id"]: s for s in t16i["sources"]}
    assert by_sid["alpha"]["vote"] == {
        "sigma": 0.072618,
        "sigma_floored": False,
        "conf_usd_gpu_hr": 0.072618,
    }


# ------------------------------------------------- targets, order, valve


def test_cli_target_guards_and_publish_in_order(monkeypatch, capsys, tmp_path):
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)

    # An unscheduled stamp is refused outright (era-aware: hour 5 is not
    # on the basket-era grid).
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--observation", "2026-08-10T05"],
    )
    assert cli.main() == 1
    assert "not a scheduled observation" in capsys.readouterr().out

    # Beyond now: a typo'd backfill must never look like a no-op.
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--observation", "2026-08-12T07"],
    )
    assert cli.main() == 1
    assert "outside the replayable range" in capsys.readouterr().out

    # Scheduled, in range, next mark unpassed AND no snapshot: still
    # open (early closure needs the snapshot to exist -- early compose).
    t06_keys = [k for k in client.objects if "/snapshots/2026-08-12/slot06" in k]
    assert len(t06_keys) == 1
    t06_snapshot = client.objects.pop(t06_keys[0])
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--observation", "2026-08-12T06"],
    )
    assert cli.main() == 1
    assert "not yet closed" in capsys.readouterr().out
    assert _key("2026-08-12T06") not in client.objects

    # With the snapshot back, T06 closes EARLY -- but publish-in-order
    # still governs: every earlier stamp is unpublished, so the target
    # refuses on ORDER now, not on closure.
    client.objects[t06_keys[0]] = t06_snapshot
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--observation", "2026-08-12T06"],
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "not yet closed" not in out
    assert "earlier unpublished" in out
    assert _key("2026-08-12T06") not in client.objects

    # Publish-in-order: targeting the second stamp while the first is
    # unpublished refuses (and writes nothing).
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--observation", "2026-08-10T10"],
    )
    assert cli.main() == 1
    assert "earlier unpublished" in capsys.readouterr().out
    assert _key("2026-08-10T10") not in client.objects

    # A --from/--to range publishes in order from genesis...
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        [
            "--config", str(cfg_path),
            "--from", "2026-08-10T04", "--to", "2026-08-10T10",
        ],
    )
    assert cli.main() == 0
    assert "observations written: 2" in capsys.readouterr().out
    # ...after which the next single observation is publishable.
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--observation", "2026-08-10T16"],
    )
    assert cli.main() == 0
    assert _key("2026-08-10T16") in client.objects


def test_cli_malformed_observation_stamps_refuse_loudly(
    monkeypatch, capsys, tmp_path
):
    """'2026-08-12Txx' (unparseable hour) and '2026-13-01T05'
    (unparseable date) must print the refusal and exit 1 -- never
    traceback out of main() on an uncaught ValueError."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    for bad in ("2026-08-12Txx", "2026-13-01T05"):
        cli = _wire_cli(
            monkeypatch, client, NOW1,
            ["--config", str(cfg_path), "--observation", bad],
        )
        assert cli.main() == 1
        out = capsys.readouterr().out
        assert "ERROR" in out
        assert "YYYY-MM-DDTHH" in out
    # Same parse site guards --from/--to.
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        [
            "--config", str(cfg_path),
            "--from", "2026-08-10Tzz", "--to", "2026-08-10T10",
        ],
    )
    assert cli.main() == 1
    assert "YYYY-MM-DDTHH" in capsys.readouterr().out


def test_cli_dry_run_writes_nothing(monkeypatch, capsys, tmp_path):
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)
    keys_before = set(client.objects)
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--sync", "--dry-run"],
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 0" in out
    assert set(client.objects) == keys_before


def test_cli_max_observations_valve_resumes_byte_identically(
    monkeypatch, capsys, tmp_path
):
    cfg_path = _write_config(tmp_path)
    control = FakeS3()
    _seed_world(control)
    cli = _wire_cli(monkeypatch, control, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    valved = FakeS3()
    _seed_world(valved)
    cli = _wire_cli(
        monkeypatch, valved, NOW2,
        ["--config", str(cfg_path), "--sync", "--max-observations", "6"],
    )
    assert cli.main() == 0  # the valve is a notice, never a failure
    out = capsys.readouterr().out
    assert "observations written: 6" in out
    assert "max-observations" in out and "next run continues" in out
    assert len(_artifacts(valved)) == 6

    cli = _wire_cli(
        monkeypatch, valved, NOW2,
        ["--config", str(cfg_path), "--sync", "--max-observations", "6"],
    )
    assert cli.main() == 0
    assert "observations written: 6" in capsys.readouterr().out

    cli = _wire_cli(monkeypatch, valved, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    assert "observations written: 3" in capsys.readouterr().out

    # The resumed series is byte-identical to the one-shot series.
    assert _artifacts(valved) == _artifacts(control)


# ------------------------------------------------------------------ fences


def test_cli_refuses_to_extend_on_calc_params_drift(monkeypatch, capsys, tmp_path):
    """Rule D2 at observation grain: after observations publish, a live
    config whose calc_params drift errors loudly naming the key and
    refuses to publish new observations under the same methodology_id."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    # A fresh hour-7 snapshot: T07 closes early at NOW2 (early compose), so
    # the drifted config has a NEW observation to refuse.
    _seed_slot(client, OBS_PREFIX, "2026-08-12", 7)
    drifted = _config()
    drifted["calc"]["filter_sigma"] = 2.5
    drifted_path = _write_config(tmp_path, drifted, name="drifted.json")
    cli = _wire_cli(
        monkeypatch, client, NOW2, ["--config", str(drifted_path), "--sync"]
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "calc_params drift" in out
    assert "'filter_sigma'" in out
    assert "mint a new methodology_id" in out
    assert "NOT published" in out
    assert _key("2026-08-12T07") not in client.objects

    # The unchanged config still extends the series cleanly.
    cli = _wire_cli(monkeypatch, client, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    assert _key("2026-08-12T07") in client.objects


def test_cli_refuses_a_floor_key_swap_without_a_mint(monkeypatch, capsys, tmp_path):
    """Ruling 2026-08-26 under the D2 fence: swapping the floor SEMANTICS
    (absolute filter_sigma_floor -> percent filter_sigma_floor_pct) against
    published observations is calc_params drift on BOTH keys and refuses
    to extend — the swap is a mint, never an edit. No adoption grace may
    ever grow around the floor pair: the two keys are different RULES, not
    a vocabulary addition."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    # A fresh hour-7 snapshot: T07 closes early at NOW2 (early compose), so
    # the swapped config has a NEW observation to refuse.
    _seed_slot(client, OBS_PREFIX, "2026-08-12", 7)
    swapped = _config()
    del swapped["calc"]["filter_sigma_floor"]
    swapped["calc"]["filter_sigma_floor_pct"] = 3.0
    # Floor split (ruling 2026-08-27): a percent-regime median-votes lane
    # must carry its own vote floor to load at all.
    swapped["calc"]["vote_sigma_floor_pct"] = 3.0
    swapped_path = _write_config(tmp_path, swapped, name="swapped.json")
    cli = _wire_cli(
        monkeypatch, client, NOW2, ["--config", str(swapped_path), "--sync"]
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "calc_params drift" in out
    assert "'filter_sigma_floor'" in out
    assert "'filter_sigma_floor_pct'" in out
    assert "mint a new methodology_id" in out
    assert "NOT published" in out
    assert _key("2026-08-12T07") not in client.objects


def test_cli_refuses_a_vote_sigma_floor_pct_flip_without_a_mint(
    monkeypatch, capsys, tmp_path
):
    """Floor split (ruling 2026-08-27) under the D2 fence: ADDING
    calc.vote_sigma_floor_pct against published observations is
    calc_params drift (the fence symmetric-diffs the key sets, so a NEW
    key is auto-covered) and refuses to extend — the floor split is a
    mint, never an edit. The smallest VALID config carrying the key is the
    full percent-regime flip (the vote floor refuses alongside the
    absolute floor at load), so the drift names all three floor keys. No
    adoption grace may ever grow around this key: absent-vs-present is a
    different vote conviction RULE, not a vocabulary addition (the
    floor-pair doctrine)."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    published = set(_artifacts(client))

    # A fresh hour-7 capture: NOW2 has a NEW frontier observation to
    # refuse under EITHER closure rule (T06's mark has passed at NOW2;
    # under main's snapshot-exists closure T06 already composed
    # at NOW1 and T07 closes early on this snapshot instead).
    _seed_slot(client, OBS_PREFIX, "2026-08-12", 7)
    flipped = _config()
    del flipped["calc"]["filter_sigma_floor"]
    flipped["calc"]["filter_sigma_floor_pct"] = 3.0
    flipped["calc"]["vote_sigma_floor_pct"] = 3.0
    flipped_path = _write_config(tmp_path, flipped, name="flipped.json")
    cli = _wire_cli(
        monkeypatch, client, NOW2, ["--config", str(flipped_path), "--sync"]
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "calc_params drift" in out
    assert "'vote_sigma_floor_pct'" in out
    assert "mint a new methodology_id" in out
    assert "NOT published" in out
    assert _key("2026-08-12T07") not in client.objects
    assert set(_artifacts(client)) == published  # refusal wrote NOTHING


def test_cli_refuses_a_vote_sigma_source_flip_without_a_mint(
    monkeypatch, capsys, tmp_path
):
    """Ruling 2026-08-27 under the D2 fence: ADDING calc.vote_sigma_source
    against published observations is calc_params drift (the fence
    symmetric-diffs the key sets, so a NEW key is auto-covered) and
    refuses to extend — the vote-sigma decouple is a mint, never an edit.
    No adoption grace may ever grow around this key: absent-vs-present is
    a different vote RULE, not a vocabulary addition (the floor-pair
    doctrine)."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    published = set(_artifacts(client))

    # A fresh hour-7 capture: NOW2 has a NEW frontier observation to
    # refuse under EITHER closure rule (T06's mark has passed at NOW2;
    # under main's snapshot-exists closure T06 already composed
    # at NOW1 and T07 closes early on this snapshot instead).
    _seed_slot(client, OBS_PREFIX, "2026-08-12", 7)
    flipped = _config()
    flipped["calc"]["vote_sigma_source"] = "dw_history"
    flipped_path = _write_config(tmp_path, flipped, name="flipped.json")
    cli = _wire_cli(
        monkeypatch, client, NOW2, ["--config", str(flipped_path), "--sync"]
    )
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "calc_params drift" in out
    assert "'vote_sigma_source'" in out
    assert "mint a new methodology_id" in out
    assert "NOT published" in out
    assert _key("2026-08-12T07") not in client.objects
    assert set(_artifacts(client)) == published  # refusal wrote NOTHING


def test_cli_exclusion_pinning_hour_and_day_scoped(monkeypatch, capsys, tmp_path):
    """Section 3 item 9 at the CLI: an hour-scoped exclusion pins ONLY
    its observation, a date-scoped one pins every observation of its
    date, and editing either against published history refuses."""
    cfg = _config()
    cfg["calc"]["manual_exclusions"] = [
        {"date": GENESIS, "source_id": "alpha", "reason": "bad tap", "hour": 4},
        {"date": GENESIS, "source_id": "bravo", "reason": "whole day"},
    ]
    cfg_path = _write_config(tmp_path, cfg)
    client = FakeS3()
    _seed_world(client)
    # 03:59: 08-10's T22 early-closes on its snapshot (early compose) but
    # 08-11's T04 stamp is not yet scheduled -- exactly the 08-10 slots.
    now = datetime(2026, 8, 11, 3, 59, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    t04 = json.loads(client.objects[_key("2026-08-10T04")])
    by_sid = {s["source_id"]: s for s in t04["sources"]}
    assert by_sid["alpha"]["status"] == "manually_excluded"
    assert by_sid["alpha"]["excluded_reason"] == "bad tap"
    assert by_sid["bravo"]["status"] == "manually_excluded"
    assert t04["panel_dark"] is True  # one passer < claim floor 2
    t10 = json.loads(client.objects[_key("2026-08-10T10")])
    by_sid = {s["source_id"]: s for s in t10["sources"]}
    assert by_sid["alpha"]["status"] == "ok"  # hour scope: one observation
    assert by_sid["bravo"]["status"] == "manually_excluded"  # date scope

    # REMOVING the hour-scoped entry contradicts the published T04.
    edited = _config()
    edited["calc"]["manual_exclusions"] = [
        {"date": GENESIS, "source_id": "bravo", "reason": "whole day"},
    ]
    edited_path = _write_config(tmp_path, edited, name="edited.json")
    later = datetime(2026, 8, 11, 11, 0, tzinfo=timezone.utc)  # closes 08-11T04
    cli = _wire_cli(monkeypatch, client, later, ["--config", str(edited_path), "--sync"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "2026-08-10T04: manual_exclusions contradict" in out
    assert "NOT published" in out
    assert _key("2026-08-11T04") not in client.objects

    # ADDING an hour-scoped entry for a published observation refuses too.
    added = _config()
    added["calc"]["manual_exclusions"] = cfg["calc"]["manual_exclusions"] + [
        {"date": GENESIS, "source_id": "charlie", "reason": "late edit", "hour": 10},
    ]
    added_path = _write_config(tmp_path, added, name="added.json")
    cli = _wire_cli(monkeypatch, client, later, ["--config", str(added_path), "--sync"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "2026-08-10T10: manual_exclusions contradict" in out

    # The pinned config extends the series cleanly.
    cli = _wire_cli(monkeypatch, client, later, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    assert _key("2026-08-11T04") in client.objects


def test_cli_record_quarantine_unwedges_a_poisoned_snapshot(
    monkeypatch, capsys, tmp_path
):
    """Record quarantine end to end: a malformed record object at an
    unpublished stamp crashes every firing forever (publish-in-order
    blocks the lane; earliest-key-wins means a later good snapshot can
    never shadow it) -- and the record_exclusions escape hatch publishes
    an explicit record_quarantined artifact WITHOUT reading the poison,
    the lane continues, and the verdict pins against later edits."""
    client = FakeS3()
    # 2026-08-10: slot 4 POISONED (earliest key -- a later good duplicate
    # cannot win), slot 10 healthy.
    poison_key = f"{BASKET_PREFIX}/snapshots/2026-08-10/slot04-20260810T040500Z-dead.json"
    client.objects[poison_key] = b"{ this is not json"
    _seed_slot(
        client, BASKET_PREFIX, "2026-08-10", 4,
        run_id="20260810T041500Z-good",  # later key: never read
    )
    _seed_slot(client, BASKET_PREFIX, "2026-08-10", 10)
    now = datetime(2026, 8, 10, 17, 0, tzinfo=timezone.utc)  # closes T04+T10

    # Without the hatch the lane wedges: the poisoned parse raises out of
    # every firing and NOTHING publishes.
    cfg_path = _write_config(tmp_path)
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    with pytest.raises(json.JSONDecodeError):
        cli.main()
    capsys.readouterr()
    assert not _artifacts(client)

    # With the quarantine: T04 publishes as record_quarantined (index
    # null, observation_missed FALSE), the poison is never GET, and the
    # lane continues to T10.
    cfg = _config()
    cfg["record_exclusions"] = [
        {"date": GENESIS, "hour": 4, "reason": "poisoned snapshot object"}
    ]
    q_path = _write_config(tmp_path, cfg, name="quarantined.json")
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(q_path), "--sync"])
    client.reset_reads()
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "record quarantined by config" in out
    assert "[record_quarantined]" in out
    assert poison_key not in client.get_keys  # never read
    t04 = json.loads(client.objects[_key("2026-08-10T04")])
    assert t04["record_quarantined"] == "poisoned snapshot object"
    assert t04["observation_missed"] is False
    assert t04["panel_dark"] is True
    assert t04["index"] is None
    assert t04["calc_params"]["record_exclusions"] == cfg["record_exclusions"]
    t10 = json.loads(client.objects[_key("2026-08-10T10")])
    assert t10["record_quarantined"] is None
    assert t10["panel_dark"] is False  # the lane CONTINUED past the poison
    # The jump reference walked past the quarantined stamp (no prints
    # there to reference) -- T10 has no reference at all on this world.
    assert t10["jump_screen"]["reference"] is None

    # The published verdict PINS: removing the entry contradicts T04.
    later = datetime(2026, 8, 10, 23, 30, tzinfo=timezone.utc)  # closes T16
    cli = _wire_cli(monkeypatch, client, later, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "record_exclusions contradict" in out
    assert "NOT published" in out
    assert _key("2026-08-10T16") not in client.objects
    # The pinned config extends the series cleanly.
    _seed_slot(client, BASKET_PREFIX, "2026-08-10", 16)
    cli = _wire_cli(monkeypatch, client, later, ["--config", str(q_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    assert _key("2026-08-10T16") in client.objects


class BlippyS3(FakeS3):
    """FakeS3 whose FIRST snapshot-keyspace LIST returns empty Contents --
    the transient gateway blip the false-missed guard exists for."""

    def __init__(self, blip_prefix):
        super().__init__()
        self._blip_prefix = blip_prefix
        self.blipped = False

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        if not self.blipped and Prefix.startswith(self._blip_prefix):
            self.blipped = True
            self.list_prefixes.append(Prefix)
            return {"Contents": [], "IsTruncated": False}
        return super().list_objects_v2(Bucket, Prefix, **kwargs)


def test_cli_false_missed_guard_re_lists_before_publishing_missed(
    monkeypatch, capsys, tmp_path
):
    """False-missed guard: a transient empty-Contents LIST must not pin an
    immutable false observation_missed artifact -- the missed verdict is
    confirmed against ONE fresh re-LIST, and when the key appears the
    observation computes normally."""
    cfg_path = _write_config(tmp_path)
    client = BlippyS3(f"{BASKET_PREFIX}/snapshots/2026-08-10/")
    _seed_slot(client, BASKET_PREFIX, "2026-08-10", 4)
    now = datetime(2026, 8, 10, 11, 0, tzinfo=timezone.utc)  # closes T04 only
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert client.blipped  # the blip genuinely fired
    assert "slot key appeared on the confirming re-LIST" in out
    artifact = json.loads(client.objects[_key("2026-08-10T04")])
    assert artifact["observation_missed"] is False
    assert artifact["panel_dark"] is False

    # Control: a GENUINELY missing slot still publishes missed -- after
    # its confirming re-LIST (two LISTs of the day keyspace, not one).
    control = FakeS3()
    _seed_slot(control, BASKET_PREFIX, "2026-08-10", 4)
    later = datetime(2026, 8, 10, 17, 0, tzinfo=timezone.utc)  # closes T10 too
    cli = _wire_cli(monkeypatch, control, later, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    missed = json.loads(control.objects[_key("2026-08-10T10")])
    assert missed["observation_missed"] is True
    day_prefix = f"{BASKET_PREFIX}/snapshots/2026-08-10/"
    assert control.list_prefixes.count(day_prefix) == 2  # cache + confirm


# ------------------------------------- early closure + drain grace


def _hourly_config():
    """A single-era hourly observatory lane from GENESIS, fx none -- the
    minimal world for the closure-predicate pins."""
    cfg = _config()
    cfg["record_sources"] = [
        {"kind": "observatory", "prefix": OBS_PREFIX, "from_date": GENESIS}
    ]
    cfg["slot_grids"] = [{"from_date": GENESIS, "slot_hours_utc": list(range(24))}]
    cfg["calc"]["fx_lane"] = "none"
    return cfg


def test_cli_early_compose_is_byte_identical_to_late_compose(
    monkeypatch, capsys, tmp_path
):
    """Early compose's load-bearing property: an observation composed EARLY
    (snapshot exists, closing mark unpassed) is BYTE-IDENTICAL to the
    same observation composed LATE (after its mark) -- composite bytes
    are a pure function of the record, so the closure predicate cannot
    move a single published byte."""
    cfg_path = _write_config(tmp_path)
    early = FakeS3()
    _seed_world(early)
    cli = _wire_cli(monkeypatch, early, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 15" in out  # T06 composed at NOW1

    late = FakeS3()
    _seed_world(late)
    t06_keys = [k for k in late.objects if "/snapshots/2026-08-12/slot06" in k]
    assert len(t06_keys) == 1
    t06_snapshot = late.objects.pop(t06_keys[0])
    cli = _wire_cli(monkeypatch, late, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 14" in out  # no snapshot: T06 stays open
    assert _key("2026-08-12T06") not in late.objects
    late.objects[t06_keys[0]] = t06_snapshot  # the capture lands late
    cli = _wire_cli(monkeypatch, late, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    assert "observations written: 1" in capsys.readouterr().out

    art_early = _artifacts(early)
    art_late = _artifacts(late)
    assert set(art_early) == set(art_late)
    for key, blob in art_early.items():
        assert art_late[key] == blob, key


def test_cli_missed_waits_for_drain_grace_then_self_heals(
    monkeypatch, capsys, tmp_path
):
    """Early compose's missingness half: a closed stamp with NO snapshot does
    not publish observation_missed inside next mark + grace (the firing
    defers, exits 0, publishes nothing at or after the deferred stamp);
    a snapshot landing INSIDE the grace composes values on the next
    firing -- the late self-heal case the grace exists for."""
    cfg_path = _write_config(tmp_path, _hourly_config(), name="hourly.json")
    client = FakeS3()
    for hour in range(4):
        _seed_slot(client, OBS_PREFIX, GENESIS, hour)
    # Hour 4 has NO snapshot; its mark (05:00) has passed at 05:10 but
    # the 20-minute drain grace (until 05:20) has not.
    within_grace = datetime(2026, 8, 10, 5, 10, tzinfo=timezone.utc)
    cli = _wire_cli(
        monkeypatch, client, within_grace, ["--config", str(cfg_path), "--sync"]
    )
    assert cli.main() == 0  # routine deferral, never a red firing
    out = capsys.readouterr().out
    assert "observations written: 4" in out  # hours 0-3
    assert "drain grace" in out and "deferring" in out
    assert _key("2026-08-10T04") not in client.objects

    # The self-heal lands inside the grace: the next firing composes
    # VALUES immediately (composition is never grace-gated once the
    # snapshot exists) -- no false missed print was ever pinned.
    _seed_slot(client, OBS_PREFIX, GENESIS, 4)
    heal_now = datetime(2026, 8, 10, 5, 15, tzinfo=timezone.utc)
    cli = _wire_cli(
        monkeypatch, client, heal_now, ["--config", str(cfg_path), "--sync"]
    )
    assert cli.main() == 0
    assert "observations written: 1" in capsys.readouterr().out
    t04 = json.loads(client.objects[_key("2026-08-10T04")])
    assert t04["observation_missed"] is False
    assert t04["panel_dark"] is False


def test_cli_missed_publishes_at_next_mark_plus_grace(
    monkeypatch, capsys, tmp_path
):
    """Early-compose boundary: with no snapshot ever landing, observation_missed
    publishes on the first firing with utc_now >= next mark +
    MISSED_PUBLISH_GRACE_MINUTES -- and not one minute earlier."""
    cli_mod = _load_cli()
    assert cli_mod.MISSED_PUBLISH_GRACE_MINUTES == 20
    cfg_path = _write_config(tmp_path, _hourly_config(), name="hourly.json")
    client = FakeS3()
    for hour in range(4):
        _seed_slot(client, OBS_PREFIX, GENESIS, hour)
    just_before = datetime(2026, 8, 10, 5, 19, tzinfo=timezone.utc)
    cli = _wire_cli(
        monkeypatch, client, just_before, ["--config", str(cfg_path), "--sync"]
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "deferring" in out
    assert _key("2026-08-10T04") not in client.objects

    at_deadline = datetime(2026, 8, 10, 5, 20, tzinfo=timezone.utc)
    cli = _wire_cli(
        monkeypatch, client, at_deadline, ["--config", str(cfg_path), "--sync"]
    )
    assert cli.main() == 0
    capsys.readouterr()
    missed = json.loads(client.objects[_key("2026-08-10T04")])
    assert missed["observation_missed"] is True
    assert missed["panel_dark"] is True
    assert missed["index"] is None


def test_cli_publish_in_order_holds_across_a_deferred_missed_stamp(
    monkeypatch, capsys, tmp_path
):
    """Early compose meets the existing publish-in-order guard: an
    early-composable H (snapshot exists) must NOT publish while H-1 is a
    missed candidate still inside the drain grace -- the firing stops AT
    the deferred stamp (H-1's verdict could still flip to a value via a
    draining self-heal, and H's provenance must replay after H-1's).
    Once the grace passes, H-1's missed artifact and H's composite
    publish in stamp order on one firing."""
    cfg_path = _write_config(tmp_path, _hourly_config(), name="hourly.json")
    client = FakeS3()
    for hour in (0, 1, 2, 4):  # hour 3 missing; hour 4 captured
        _seed_slot(client, OBS_PREFIX, GENESIS, hour)
    # 04:10: hour 3 closed at its 04:00 mark but inside the grace
    # (until 04:20); hour 4 is early-composable on its snapshot.
    now = datetime(2026, 8, 10, 4, 10, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 3" in out  # hours 0-2 only
    assert "deferring" in out
    assert _key("2026-08-10T03") not in client.objects
    assert _key("2026-08-10T04") not in client.objects  # in-order held

    # Targeting the early-composable H while H-1 defers is LOUD -- a
    # targeted stamp that cannot publish must never look like a no-op.
    cli = _wire_cli(
        monkeypatch, client, now,
        ["--config", str(cfg_path), "--observation", "2026-08-10T04"],
    )
    assert cli.main() == 1
    assert "deferring" in capsys.readouterr().out
    assert _key("2026-08-10T04") not in client.objects

    # Grace passed: both publish, in stamp order, on one firing.
    later = datetime(2026, 8, 10, 4, 25, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, later, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    assert "observations written: 2" in capsys.readouterr().out
    missed = json.loads(client.objects[_key("2026-08-10T03")])
    assert missed["observation_missed"] is True
    composed = json.loads(client.objects[_key("2026-08-10T04")])
    assert composed["observation_missed"] is False
    order = [
        k
        for k in client.put_order
        if "/composites/" in k and not k.endswith("latest.json")
    ]
    assert order.index(_key("2026-08-10T03")) < order.index(_key("2026-08-10T04"))


def test_cli_quarantine_publish_waits_for_drain_grace(
    monkeypatch, capsys, tmp_path
):
    """Early compose: record_quarantined artifacts are grace-gated exactly
    like observation_missed -- both are immutable verdicts about the
    record's readable bytes at the stamp, and neither may pin before the
    drain settles. No premature quarantine warn either: the verdict line
    prints only when the artifact actually publishes."""
    cfg = _hourly_config()
    cfg["record_exclusions"] = [
        {"date": GENESIS, "hour": 3, "reason": "poisoned snapshot object"}
    ]
    cfg_path = _write_config(tmp_path, cfg, name="quarantine_grace.json")
    client = FakeS3()
    for hour in (0, 1, 2, 3):
        _seed_slot(client, OBS_PREFIX, GENESIS, hour)
    within = datetime(2026, 8, 10, 4, 10, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, within, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 3" in out
    assert "deferring" in out
    assert "record quarantined by config" not in out
    assert _key("2026-08-10T03") not in client.objects

    after = datetime(2026, 8, 10, 4, 25, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, after, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "record quarantined by config" in out
    t03 = json.loads(client.objects[_key("2026-08-10T03")])
    assert t03["record_quarantined"] == "poisoned snapshot object"
    assert t03["observation_missed"] is False


# -------------------------------------------------------------- drift scan


def test_cli_drift_scan_gated_and_warns_without_failing(
    monkeypatch, capsys, tmp_path
):
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    published = _artifacts(client)

    # A late-landing snapshot with an EARLIER run_id becomes the record's
    # earliest key for 08-12 hour 3 -- with a different alpha print.
    _seed_slot(
        client, OBS_PREFIX, "2026-08-12", 3,
        prices={"alpha": 2.9, "bravo": 2.5, "charlie": 2.6},
        run_id="20260812T030100Z-0000",
    )
    # And the published-as-missed hour 4 now has record evidence.
    _seed_slot(
        client, OBS_PREFIX, "2026-08-12", 4,
        run_id="20260812T040900Z-9999",
    )
    # GATED (perf stage): NOW1 is 06:30Z, outside the DRIFT_SCAN_WINDOW_MINUTES
    # arming window and no --drift-scan -- the scan does not run and no record
    # is read for published observations.
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    client.reset_reads()
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DRIFT" not in out
    assert not [k for k in client.get_keys if "/snapshots/" in k]

    # --drift-scan forces the sweep at any hour.
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--sync", "--drift-scan"],
    )
    assert cli.main() == 0  # warn-only: the published series stands
    out = capsys.readouterr().out
    assert "DRIFT 2026-08-12T03" in out
    assert "selection changed" in out
    assert "alpha: published print 2.4 vs record 2.9" in out
    assert "DRIFT 2026-08-12T04" in out
    assert "published as observation_missed" in out
    assert "observations written: 0" in out
    assert _artifacts(client) == published  # artifacts untouched

    # A firing inside [16:00Z, 16:30Z) runs the sweep on its own (module
    # constant DRIFT_SCAN_WINDOW_MINUTES). --dry-run keeps the newly-closed
    # hours from publishing so only the scan's output is under test.
    cli = _load_cli()
    assert cli.DRIFT_SCAN_WINDOW_MINUTES == (16 * 60, 16 * 60 + 30)
    at_16 = datetime(2026, 8, 12, 16, 15, tzinfo=timezone.utc)
    cli = _wire_cli(
        monkeypatch, client, at_16,
        ["--config", str(cfg_path), "--sync", "--dry-run"],
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DRIFT 2026-08-12T03" in out
    assert _artifacts(client) == published

    # The scan is BOUNDED in observations: with drift_scan_observations 1
    # (the TOP-LEVEL ops knob) only the newest published observation (T05)
    # is re-resolved, so the T03/T04 divergence goes unscanned. (Nothing
    # new is computable at NOW1, so the D2 fence stays out of the way --
    # same as the daily lane's horizon test.)
    bounded = _config()
    bounded["drift_scan_observations"] = 1
    bounded_path = _write_config(tmp_path, bounded, name="bounded.json")
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(bounded_path), "--sync", "--drift-scan"],
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DRIFT" not in out


def test_cli_drift_scan_vanished_record_warns(monkeypatch, capsys, tmp_path):
    """Coverage stage (recipe 3): a published NON-missed observation whose
    record snapshot keys were later deleted fires the vanished-record
    tripwire -- warn-only, exit 0, artifacts untouched."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_slot(client, BASKET_PREFIX, "2026-08-10", 4)
    _seed_slot(client, BASKET_PREFIX, "2026-08-10", 10)
    now = datetime(2026, 8, 10, 17, 0, tzinfo=timezone.utc)  # closes T04+T10
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    published = _artifacts(client)

    key10 = [
        k for k in client.objects if "/snapshots/2026-08-10/slot10" in k
    ]
    assert len(key10) == 1
    del client.objects[key10[0]]
    cli = _wire_cli(
        monkeypatch, client, now,
        ["--config", str(cfg_path), "--sync", "--drift-scan"],
    )
    assert cli.main() == 0  # warn-only: the published series stands
    out = capsys.readouterr().out
    assert "DRIFT 2026-08-10T10" in out
    assert "record now holds NO snapshot for this published observation" in out
    assert "DRIFT 2026-08-10T04" not in out  # the intact stamp is silent
    assert _artifacts(client) == published


def test_cli_drift_scan_compares_eur_prints_in_native_terms(
    monkeypatch, capsys, tmp_path
):
    """Coverage stage (recipe 3): an EUR seat whose record-native price
    changed under a published observation fires the NATIVE-terms compare
    (never the USD one -- a late-landing real ECB rate must not page)."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    eur_row = dict(_row(2.0), price_usd_gpu_hr=None, currency="EUR")
    _seed_slot(
        client, BASKET_PREFIX, "2026-08-10", 4,
        prices={"alpha": 2.4, "bravo": eur_row, "charlie": 2.6},
    )
    now = datetime(2026, 8, 10, 11, 0, tzinfo=timezone.utc)  # closes T04
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    published = json.loads(client.objects[_key("2026-08-10T04")])
    bravo = {s["source_id"]: s for s in published["sources"]}["bravo"]
    assert bravo["chosen"]["fx_rate"] == 1.15  # a genuinely FX-priced print

    # A late snapshot with an EARLIER run_id becomes the record's earliest
    # key -- same slot, different native EUR print.
    eur_moved = dict(_row(2.1), price_usd_gpu_hr=None, currency="EUR")
    _seed_slot(
        client, BASKET_PREFIX, "2026-08-10", 4,
        prices={"alpha": 2.4, "bravo": eur_moved, "charlie": 2.6},
        run_id="20260810T040100Z-0000",
    )
    cli = _wire_cli(
        monkeypatch, client, now,
        ["--config", str(cfg_path), "--sync", "--drift-scan"],
    )
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "DRIFT 2026-08-10T04" in out
    assert "selection changed" in out
    assert "bravo: published native print (2.0, 'EUR') vs record (2.1, 'EUR')" in out
    assert "alpha: published" not in out  # the unchanged USD seat is silent


def test_cli_drift_scan_missing_member_is_loud_not_a_crash(
    monkeypatch, capsys, tmp_path
):
    """Coverage stage (recipe 3): a live config MISSING a published
    member cannot re-resolve that seat -- the scan says so loudly and
    keeps walking instead of crashing (the D2 fence governs publishes;
    with nothing new computable it never fires here)."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_slot(client, BASKET_PREFIX, "2026-08-10", 4)
    now = datetime(2026, 8, 10, 11, 0, tzinfo=timezone.utc)  # closes T04
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    dropped = _config()
    dropped["members"] = [
        {"source_id": "alpha", "weight": 0.7, "skus": ["H100"]},
        {"source_id": "bravo", "weight": 0.3, "skus": ["H100"]},
    ]
    # switch_min_eligible must stay <= the member count to load; the D2
    # fence never fires here (nothing new is computable), so the scan's
    # missing-member branch is what's under test.
    dropped["calc"]["dynamic_weights"]["switch_min_eligible"] = 2
    dropped_path = _write_config(tmp_path, dropped, name="dropped.json")
    cli = _wire_cli(
        monkeypatch, client, now,
        ["--config", str(dropped_path), "--sync", "--drift-scan"],
    )
    assert cli.main() == 0  # warn-only; nothing new to publish
    out = capsys.readouterr().out
    assert "DRIFT 2026-08-10T04" in out
    assert "charlie: published seat is not a member of the live config" in out
    assert "observations written: 0" in out


# ------------------------------------------------------------ FX bounding


def test_cli_fx_bounded_load_and_conditional_feed(monkeypatch, capsys, tmp_path):
    """Perf stage FX rules: stored rates load bounded to
    [window_start_date - fx_max_staleness_days, ...]; the live ECB feed
    fetches ONLY when some unpublished observation's day cannot resolve
    a stored rate after walk-back; fx_lane none touches neither."""
    cfg_path = _write_config(tmp_path)

    # (1) Stored rates cover every unpublished day (GENESIS rate within
    # the 7d walk-back of 08-10..08-12): feed skipped, load bounded to
    # genesis - 7d (the window starts at genesis on this small world).
    client = FakeS3()
    _seed_world(client)
    fx_calls = []
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--sync", "--dry-run"], fx_calls,
    )
    assert cli.main() == 0
    capsys.readouterr()
    assert fx_calls == [("stored", PREFIX, "2026-08-03")]

    # (2) Stored rates too stale for the frontier days: the feed fetches,
    # same bound, honoring dry-run persist=False...
    stale = {
        "2026-08-01": {
            "source": "ecb_reference_rate",
            "as_of": "2026-08-01",
            "rates": {"USD": 1.1},
        }
    }
    fx_calls = []
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--sync", "--dry-run"],
        fx_calls, stored_fx=stale,
    )
    assert cli.main() == 0
    capsys.readouterr()
    assert fx_calls == [
        ("stored", PREFIX, "2026-08-03"),
        ("feed", PREFIX, False, "2026-08-03"),
    ]

    # ... and persist=True on a real publish run.
    fx_calls = []
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--sync"], fx_calls, stored_fx=stale,
    )
    assert cli.main() == 0
    capsys.readouterr()
    assert fx_calls[-1] == ("feed", PREFIX, True, "2026-08-03")

    # (3) With everything published and nothing new computable, no day
    # needs a rate: stored loads (drift-scan armor) but the feed skips.
    fx_calls = []
    cli = _wire_cli(
        monkeypatch, client, NOW1,
        ["--config", str(cfg_path), "--sync"], fx_calls, stored_fx=stale,
    )
    assert cli.main() == 0
    capsys.readouterr()
    assert fx_calls == [("stored", PREFIX, "2026-08-03")]

    # (4) fx_lane none: zero FX calls of either kind (USD-only rule), and
    # no fx/ key of any shape ever lands in the bucket (recipe 9: a
    # USD-only lane's keyspace carries no FX record at all).
    none_cfg = _config()
    none_cfg["calc"]["fx_lane"] = "none"
    none_path = _write_config(tmp_path, none_cfg, name="none.json")
    none_client = FakeS3()
    _seed_world(none_client)
    fx_calls = []
    cli = _wire_cli(
        monkeypatch, none_client, NOW1,
        ["--config", str(none_path), "--sync"], fx_calls,
    )
    assert cli.main() == 0
    capsys.readouterr()
    assert fx_calls == []
    assert not [k for k in none_client.objects if "/fx/" in k]


# ---------------------------------------- bounded replay (LIST frontier)

LONG_GENESIS = "2026-08-01"
LONG_SLOTS = (4, 10, 16, 22)
LONG_LAST_DAY = 17  # record seeded 2026-08-01 .. 2026-08-17


def _long_config():
    """A 4-slot single-era lane with tiny dw params: the state window =
    history 48h + max forward 6h + PRUNE_MARGIN_MINUTES (168h) = 222h in
    minute stamps, small enough that a two-week world's genesis predates
    the window."""
    cfg = _config()
    cfg["genesis_date"] = LONG_GENESIS
    cfg["record_sources"] = [
        {"kind": "observatory", "prefix": OBS_PREFIX, "from_date": LONG_GENESIS}
    ]
    cfg["slot_grids"] = [
        {"from_date": LONG_GENESIS, "slot_hours_utc": list(LONG_SLOTS)}
    ]
    cfg["calc"]["fx_lane"] = "none"  # keep the read-count pins FX-free
    cfg["calc"]["dynamic_weights"].update(
        {
            "lookback_horizons_hours": [6],
            "forward_horizons_hours": [6],
            "history_days": 2,
            "min_train_samples": 3,
            "switch_min_eligible": 3,
        }
    )
    return cfg


def _seed_long_world(client):
    """Deterministically wiggling USD prints at every slot of every day --
    small moves (well inside the sigma band and the jump fence) so every
    source prints trusted everywhere and the dynamic switch fires early."""
    i = 0
    for day in range(1, LONG_LAST_DAY + 1):
        day_str = f"2026-08-{day:02d}"
        for hour in LONG_SLOTS:
            prices = {
                "alpha": round(2.4 + 0.02 * ((i * 7) % 5), 6),
                "bravo": round(2.5 + 0.01 * ((i * 3) % 7), 6),
                "charlie": round(2.6 + 0.01 * ((i * 5) % 4), 6),
            }
            _seed_slot(client, OBS_PREFIX, day_str, hour, prices=prices)
            i += 1


def test_cli_bounded_replay_is_byte_identical_to_full_replay(
    monkeypatch, capsys, tmp_path
):
    """THE perf-stage parity proof: a series grown incrementally (each
    later run replaying only the bounded state window behind the LIST
    frontier, seeded from one pre-window artifact) is BYTE-IDENTICAL to
    the same world synced in one full replay from genesis -- across the
    dynamic-weight switch, which lands BEFORE the final run's window so
    the seed GET (mode latch) is genuinely load-bearing."""
    from gpu_index.index.panel_schedule import hour_iso_to_stamp
    from gpu_index.index.weights import PRUNE_MARGIN_MINUTES

    cfg_path = tmp_path / "long.json"
    cfg_path.write_text(json.dumps(_long_config()))

    run_clocks = [
        datetime(2026, 8, 12, 1, 0, tzinfo=timezone.utc),  # full (backlog)
        datetime(2026, 8, 15, 1, 0, tzinfo=timezone.utc),  # bounded
        datetime(2026, 8, 18, 1, 0, tzinfo=timezone.utc),  # bounded + seed
    ]

    incremental = FakeS3()
    _seed_long_world(incremental)
    for run_now in run_clocks:
        cli = _wire_cli(
            monkeypatch, incremental, run_now, ["--config", str(cfg_path), "--sync"]
        )
        assert cli.main() == 0
        capsys.readouterr()

    control = FakeS3()
    _seed_long_world(control)
    cli = _wire_cli(
        monkeypatch, control, run_clocks[-1], ["--config", str(cfg_path), "--sync"]
    )
    assert cli.main() == 0
    capsys.readouterr()

    art_incremental = _artifacts(incremental)
    art_control = _artifacts(control)
    assert set(art_incremental) == set(art_control)
    for key, blob in art_control.items():
        assert art_incremental[key] == blob, key

    # The fixture is not vacuous: the final run's window truncated real
    # history (genesis predates window_start), the switch fired BEFORE
    # that window (the seed carried the latch), and the last artifact is
    # dynamic-mode. (Run 2's clock, 08-15T01, early-closes 08-14T22 on
    # its snapshot -- early compose -- so run 3's frontier is 08-15T04.)
    frontier = hour_iso_to_stamp("2026-08-15T04")  # first unpublished at run 3
    window_start = frontier - (2 * 24 * 60 + 6 * 60 + PRUNE_MARGIN_MINUTES)
    assert window_start > hour_iso_to_stamp(f"{LONG_GENESIS}T04")
    switched = [
        json.loads(blob)["weight_calc"]["switched_on"]
        for blob in art_control.values()
        if b"switched_on" in blob
    ]
    assert len(switched) == 1
    assert hour_iso_to_stamp(switched[0]) < window_start
    last = json.loads(art_control[_long_key("2026-08-17T16")])
    assert last["weight_calc"]["mode"] == "dynamic"


def _long_key(stamp):
    return f"{PREFIX}/composites/{MID}/{stamp}.json"


def test_cli_list_frontier_bounds_the_reads(monkeypatch, capsys, tmp_path):
    """The read-count pins for the bounded replay: ONE composite-keyspace
    LIST learns the published set; composite GETs stay inside the state
    window except the single pre-window seed; record reads are
    slot-granular (a day's keys LIST once, only unpublished hours GET --
    the drift scan is gated off at this hour)."""
    from gpu_index.index.panel_schedule import hour_iso_to_stamp, stamp_to_hour_iso
    from gpu_index.index.weights import PRUNE_MARGIN_MINUTES

    cfg_path = tmp_path / "long.json"
    cfg_path.write_text(json.dumps(_long_config()))
    client = FakeS3()
    _seed_long_world(client)
    # Grow the series so the final run has a deep published prefix.
    for run_now in (
        datetime(2026, 8, 12, 1, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 15, 1, 0, tzinfo=timezone.utc),
    ):
        cli = _wire_cli(
            monkeypatch, client, run_now, ["--config", str(cfg_path), "--sync"]
        )
        assert cli.main() == 0
        capsys.readouterr()

    final_now = datetime(2026, 8, 18, 1, 0, tzinfo=timezone.utc)
    cli = _wire_cli(
        monkeypatch, client, final_now, ["--config", str(cfg_path), "--sync"]
    )
    client.reset_reads()
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "observations written: 12" in out

    # Run 2's clock (08-15T01) early-closed 08-14T22 on its snapshot
    # (early compose), so the final run's frontier is 08-15T04 -- and the
    # final run itself early-closes 08-17T22.
    frontier = hour_iso_to_stamp("2026-08-15T04")
    window_start = frontier - (2 * 24 * 60 + 6 * 60 + PRUNE_MARGIN_MINUTES)
    window_start_iso = stamp_to_hour_iso(window_start)  # 2026-08-05T22
    composites_base = f"{PREFIX}/composites/{MID}/"

    # Exactly ONE LIST of the composite keyspace (the frontier read).
    assert client.list_prefixes.count(composites_base) == 1

    # Composite artifact GETs: every in-window key as needed, plus
    # EXACTLY ONE pre-window GET -- the seed (the latest published
    # artifact strictly before window_start). latest.json is pointer
    # machinery, not an artifact read.
    stems = [
        k[len(composites_base):-len(".json")]
        for k in client.get_keys
        if k.startswith(composites_base) and not k.endswith("latest.json")
    ]
    pre_window = [s for s in stems if s < window_start_iso]
    assert pre_window == ["2026-08-05T16"]  # the seed, exactly once

    # Slot-granular record reads: only the 12 unpublished hours' chosen
    # snapshot keys GET (nothing for published stamps -- drift gated off),
    # and each needed day's snapshot keyspace LISTs exactly once (the
    # early-closure probe's LIST of 08-17 seeds the cache the walk uses).
    snapshot_gets = sorted(k for k in client.get_keys if "/snapshots/" in k)
    expected_stamps = [
        f"2026-08-{day:02d}T{hour:02d}"
        for day in (14, 15, 16, 17)
        for hour in LONG_SLOTS
        if f"2026-08-{day:02d}T{hour:02d}" > "2026-08-14T22"
        and f"2026-08-{day:02d}T{hour:02d}" <= "2026-08-17T22"
    ]
    assert len(expected_stamps) == 12
    expected_days = sorted({s[:10] for s in expected_stamps})
    assert sorted({k.split("/snapshots/")[1][:10] for k in snapshot_gets}) == (
        expected_days
    )
    assert len(snapshot_gets) == len(set(snapshot_gets)) == 12
    snapshot_lists = [
        p for p in client.list_prefixes if "/snapshots/" in p
    ]
    assert sorted(p.split("/snapshots/")[1].rstrip("/") for p in snapshot_lists) == (
        expected_days
    )


def test_snapshot_strip_is_byte_neutral():
    """Change 2's fence: the slot-granular cache strips stored snapshots
    to member entries + in-sku-union rows BEFORE caching, and the artifact
    computed from the stripped payload is BYTE-IDENTICAL to one computed
    from the full stored payload -- on a snapshot that genuinely carries
    a non-member source, foreign-sku rows on a member, and extra top-level
    keys."""
    from gpu_index.index.panel import compute_observation, panel_calc_params
    from gpu_index.index.panel_schedule import hour_iso_to_stamp
    from gpu_index.index.weights import new_weight_state

    cli = _load_cli()
    cfg = _config()
    full = {
        "sources": [
            {
                "source_id": "alpha",
                "status": "ok",
                "observations": [
                    _row(2.4),
                    dict(_row(9.9), sku="B200"),  # foreign sku: never prices
                ],
                "book_stats": {"H100 SXM": {"verified_us_ca_machines": 2}},
            },
            {
                "source_id": "zulu",  # not a member: never consulted
                "status": "ok",
                "observations": [_row(1.0)],
            },
            {"source_id": "bravo", "status": "ok", "observations": [_row(2.5)]},
            {"source_id": "charlie", "status": "ok", "observations": [_row(2.6)]},
        ],
        "run_id": "20260810T040500Z-aaaa",
        "late_fill": False,
        "capture_note": "top-level noise the panel never reads",
    }
    stripped = cli.strip_snapshot(
        full,
        member_ids={"alpha", "bravo", "charlie"},
        sku_union={"H100"},
    )
    assert [s["source_id"] for s in stripped["sources"]] == [
        "alpha", "bravo", "charlie",
    ]
    assert [o["sku"] for o in stripped["sources"][0]["observations"]] == ["H100"]
    assert stripped["sources"][0]["book_stats"]  # entry fields survive
    assert "capture_note" not in stripped

    def _compute(snapshot):
        return compute_observation(
            config=cfg,
            obs_stamp=hour_iso_to_stamp("2026-08-10T04"),
            snapshot=snapshot,
            fx_records=dict(FX),
            window_history={},
            window_currencies={},
            pending_currencies={},
            weight_state=new_weight_state(),
        )

    assert json.dumps(_compute(full), sort_keys=True) == json.dumps(
        _compute(stripped), sort_keys=True
    )
    params = panel_calc_params(cfg)
    scan_full = cli.detect_drift(
        _compute(full),
        full,
        obs_date="2026-08-10",
        params=params,
        screens=cli.compile_screens(params),
        fx_records=dict(FX),
    )
    scan_stripped = cli.detect_drift(
        _compute(full),
        stripped,
        obs_date="2026-08-10",
        params=params,
        screens=cli.compile_screens(params),
        fx_records=dict(FX),
    )
    assert scan_full == scan_stripped == []


# ---------------------------------------------------------- loud verdicts


def test_cli_untrusted_currency_is_loud_and_red(monkeypatch, capsys, tmp_path):
    """Rule D1 at the CLI: an untrusted currency label reddens the
    firing (exit 1) but the observation still publishes with the print
    held out fail-closed."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_slot(
        client, BASKET_PREFIX, "2026-08-10", 4,
        prices={
            "alpha": _row(2.4, currency="??"),
            "bravo": _row(2.5),
            "charlie": _row(2.6),
        },
    )
    now = datetime(2026, 8, 10, 11, 0, tzinfo=timezone.utc)  # closes T04 only
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "untrusted currency" in out
    assert "held out fail-closed" in out
    artifact = json.loads(client.objects[_key("2026-08-10T04")])
    by_sid = {s["source_id"]: s for s in artifact["sources"]}
    assert by_sid["alpha"]["filter"]["untrusted_currency"] is True
    assert artifact["panel_dark"] is False  # bravo + charlie still claim


def test_cli_currency_change_is_loud_publishes_and_replays_identically(
    monkeypatch, capsys, tmp_path
):
    """Coverage stage (recipe 1, CLI half): a member repricing USD -> EUR
    reddens each mismatch firing (exit 1) while the artifacts STILL
    publish (verdicts pinned: pending_count 1..2, confirmation at the
    third print), and a fresh replay through
    advance_panel_state_from_published over the published artifacts
    alone rebuilds the exact state the engine test pins for the live
    path -- window reseeded in EUR, pending consumed, every trusted
    print (mismatch ones included) in the weight series."""
    from gpu_index.index.panel_schedule import hour_iso_to_stamp
    from gpu_index.index.weights import new_weight_state

    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    eur_row = dict(_row(2.0), price_usd_gpu_hr=None, currency="EUR")
    _seed_slot(client, BASKET_PREFIX, "2026-08-10", 4)  # bravo USD 2.5
    for hour in (10, 16, 22):
        _seed_slot(
            client, BASKET_PREFIX, "2026-08-10", hour,
            prices={"alpha": 2.4, "bravo": eur_row, "charlie": 2.6},
        )
    now = datetime(2026, 8, 11, 5, 0, tzinfo=timezone.utc)  # closes 08-10
    cli = _wire_cli(monkeypatch, client, now, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 1  # the mismatch firings are RED...
    out = capsys.readouterr().out
    assert "recorded currency changed USD -> EUR" in out
    assert "(1/3 toward confirmation)" in out
    assert "(2/3 toward confirmation)" in out
    assert "currency change CONFIRMED USD -> EUR" in out
    assert "window reseeded" in out
    assert "observations written: 4" in out  # ... and the artifacts stand

    t10 = json.loads(client.objects[_key("2026-08-10T10")])
    verdict = {s["source_id"]: s for s in t10["sources"]}["bravo"]["filter"]
    assert verdict["currency_mismatch"] is True
    assert verdict["pending_count"] == 1
    t16 = json.loads(client.objects[_key("2026-08-10T16")])
    verdict = {s["source_id"]: s for s in t16["sources"]}["bravo"]["filter"]
    assert verdict["pending_count"] == 2
    t22 = json.loads(client.objects[_key("2026-08-10T22")])
    verdict = {s["source_id"]: s for s in t22["sources"]}["bravo"]["filter"]
    assert verdict["currency_confirmed"] is True
    assert verdict["unfiltered"] is True

    # Replay from published artifacts alone (the container-restart path).
    window_history: dict = {}
    window_currencies: dict = {}
    pending_currencies: dict = {}
    weight_state = new_weight_state()
    stamps = ["2026-08-10T04", "2026-08-10T10", "2026-08-10T16", "2026-08-10T22"]
    for stamp in stamps:
        cli.advance_panel_state_from_published(
            json.loads(client.objects[_key(stamp)]),
            window_history,
            window_currencies,
            pending_currencies,
            weight_state,
        )
    assert window_history == {
        "alpha": [2.4] * 4,
        "bravo": [2.0, 2.0, 2.0],  # reseeded from the pending prints
        "charlie": [2.6] * 4,
    }
    assert window_currencies == {
        "alpha": "USD", "bravo": "EUR", "charlie": "USD",
    }
    assert pending_currencies == {}  # consumed at confirmation
    t04_stamp = hour_iso_to_stamp("2026-08-10T04")
    assert weight_state["prices"]["bravo"] == {
        t04_stamp: {"usd": 2.5, "native": 2.5, "currency": "USD"},
        t04_stamp + 6 * 60: {"usd": 2.3, "native": 2.0, "currency": "EUR"},
        t04_stamp + 12 * 60: {"usd": 2.3, "native": 2.0, "currency": "EUR"},
        t04_stamp + 18 * 60: {"usd": 2.3, "native": 2.0, "currency": "EUR"},
    }
    assert set(weight_state["vectors"]) == {
        t04_stamp, t04_stamp + 6 * 60, t04_stamp + 12 * 60, t04_stamp + 18 * 60,
    }
    # The mismatch prints stayed weight-eligible: every vector holds all
    # three members at the config fallback weights.
    assert weight_state["vectors"][t04_stamp + 6 * 60] == {
        "alpha": 0.5, "bravo": 0.3, "charlie": 0.2,
    }
    assert weight_state["mode"] == "fallback"


# ------------------------------------------------------- config required


def test_cli_config_is_required(monkeypatch, capsys):
    """Exactly one config source exists in the open pipeline (the baked
    file): a run without --config is an argparse-level refusal (exit 2),
    before any config or bucket read."""
    client = FakeS3()
    cli = _wire_cli(monkeypatch, client, NOW1, ["--sync"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2
    capsys.readouterr()  # drop argparse's usage text


# --------------------------------- availability adoption grace + GET-miss


def test_d2_adoption_grace_then_full_ownership(monkeypatch, capsys, tmp_path):
    """Deploy transition: every live artifact predates
    calc.availability_verified_sources, so the D2 compare skips the key
    while the baseline lacks it (extending must not dark six lanes) --
    but once a keyed artifact is the baseline, a retune refuses like any
    other param drift."""
    cfg = _config()
    cfg["calc"]["availability_verified_sources"] = ["bravo"]
    cfg_path = _write_config(tmp_path, cfg)
    client = FakeS3()
    _seed_world(client)
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    # Doctor the NEWEST artifact into a pre-field one (published before
    # the key shipped): drop the key from its embedded calc_params.
    newest = max(
        k for k in client.objects if "/composites/" in k and not k.endswith("latest.json")
    )
    doctored = json.loads(client.objects[newest])
    assert doctored["calc_params"].pop("availability_verified_sources") == ["bravo"]
    client.objects[newest] = json.dumps(doctored).encode()

    # Extending under the keyed live config must NOT refuse (grace) --
    # a fresh hour-7 snapshot makes T07 computable at NOW2 (early
    # closure) so the extension genuinely publishes.
    _seed_slot(client, OBS_PREFIX, "2026-08-12", 7)
    cli = _wire_cli(monkeypatch, client, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "calc_params drift" not in out
    assert "observations written: 1" in out

    # The newest artifact (T07) now carries the key -> a retune refuses
    # at the next computable observation (T08 goes missed-computable
    # once its 09:00 mark plus the drain grace has passed).
    cfg["calc"]["availability_verified_sources"] = []
    cfg_path2 = _write_config(tmp_path, cfg, name="retuned.json")
    later = datetime(2026, 8, 12, 9, 25, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, later, ["--config", str(cfg_path2), "--sync"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "calc_params drift" in out
    assert "availability_verified_sources" in out


def test_listed_but_missing_artifact_refuses_the_firing(
    monkeypatch, capsys, tmp_path
):
    """LIST says published, GET (twice) says gone -> the firing refuses
    instead of recomputing a published stamp."""
    cfg_path = _write_config(tmp_path)
    client = FakeS3()
    _seed_world(client)
    cli = _wire_cli(monkeypatch, client, NOW1, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 0
    capsys.readouterr()

    target = min(
        k for k in client.objects if "/composites/" in k and not k.endswith("latest.json")
    )
    real_get = client.get_object

    def flaky_get(Bucket, Key):
        if Key == target:
            raise _NoSuchKey()
        return real_get(Bucket=Bucket, Key=Key)

    client.get_object = flaky_get
    cli = _wire_cli(monkeypatch, client, NOW2, ["--config", str(cfg_path), "--sync"])
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "GET returned nothing twice" in out
    assert "refusing this firing" in out
