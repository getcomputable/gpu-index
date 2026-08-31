# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""--verify-published: byte-for-byte recompute of a PUBLISHED day.

The mode's contract: recompute one published day deterministically from
the record — prior published artifacts advance the replay state, the
day's snapshot is the artifact's own pinned choice fetched by exact key,
dynamic-weight slot prints come from the artifact's pinned block — write
NOTHING, canonicalize, and byte-compare against the stored artifact.
MATCH exits 0 (with the sha256); MISMATCH exits 1 with field-path diffs.

Runs on the B200 lane (fx_lane none) so the FX machinery is provably
untouched. The fakes are local copies of test_b200_composite.py's —
tests/unit is not a package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_index.common.store import snapshot_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
B200_CONFIG_PATH = REPO_ROOT / "config" / "index_basket_b200.json"
FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "b200_snapshot_2026-08-16_slot22-20260816T221613Z-1fdf.json"
)
SNAPSHOT_KEY = (
    "index/b200_basket/snapshots/2026-08-16/slot22-20260816T221613Z-1fdf.json"
)
DAY2_SNAPSHOT_KEY = (
    "index/b200_basket/snapshots/2026-08-17/slot16-20260817T161001Z-2222.json"
)
COMPOSITE_KEY_DAY1 = (
    "index/b200_basket/composites/annex_a2_v0_3_calc_v5/2026-08-16.json"
)
COMPOSITE_KEY_DAY2 = (
    "index/b200_basket/composites/annex_a2_v0_3_calc_v5/2026-08-17.json"
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)


def _fixture_snapshot() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _day2_snapshot() -> dict:
    snap = _fixture_snapshot()
    snap["capture_date"] = "2026-08-17"
    snap["run_id"] = "20260817T161001Z-2222"
    snap["slot_hour_utc"] = 16
    for entry in snap["sources"]:
        if entry["source_id"] == "nebius":
            for obs in entry["observations"]:
                if obs.get("sku") == "B200":
                    obs["price_usd_gpu_hr"] = 7.2
                    obs["price_native_per_gpu_hr"] = 7.2
    return snap


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

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
        self.put_order.append(Key)

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


_CLI_CACHE = {}


def _load_cli():
    if "mod" not in _CLI_CACHE:
        spec = importlib.util.spec_from_file_location(
            "compute_index_composite_verify",
            REPO_ROOT / "scripts" / "compute_index_composite.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _wire_cli(monkeypatch, client, now, argv):
    cli = _load_cli()

    class StubConfig:
        bucket = "record"

    monkeypatch.setattr(
        cli.BucketConfig, "from_env", staticmethod(lambda: StubConfig())
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: client)
    # fx_lane none: neither the feed NOR the stored-rates read may fire.
    monkeypatch.setattr(
        cli,
        "ensure_rates",
        lambda *a, **k: pytest.fail("USD-only basket reached the FX feed"),
    )
    monkeypatch.setattr(
        cli,
        "load_stored_rates",
        lambda *a, **k: pytest.fail("USD-only basket read stored FX records"),
    )
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(
        "sys.argv",
        ["compute_index_composite.py", "--config", str(B200_CONFIG_PATH), *argv],
    )
    return cli


def _publish_day1(monkeypatch, capsys) -> FakeS3:
    client = FakeS3()
    client.objects[SNAPSHOT_KEY] = FIXTURE_PATH.read_bytes()
    now = datetime(2026, 8, 17, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    assert COMPOSITE_KEY_DAY1 in client.objects
    return client


def _publish_two_days(monkeypatch, capsys) -> FakeS3:
    client = FakeS3()
    client.objects[SNAPSHOT_KEY] = FIXTURE_PATH.read_bytes()
    client.objects[DAY2_SNAPSHOT_KEY] = snapshot_bytes(_day2_snapshot())
    now = datetime(2026, 8, 18, 5, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--sync"])
    assert cli.main() == 0
    capsys.readouterr()
    assert COMPOSITE_KEY_DAY2 in client.objects
    return client


def _verify(monkeypatch, client, day):
    now = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)
    return _wire_cli(monkeypatch, client, now, ["--verify-published", day])


# ------------------------------------------------------------------- MATCH


def test_published_day_verifies_match_with_sha256(monkeypatch, capsys):
    client = _publish_day1(monkeypatch, capsys)
    stored_sha = hashlib.sha256(client.objects[COMPOSITE_KEY_DAY1]).hexdigest()
    before = dict(client.objects)

    cli = _verify(monkeypatch, client, "2026-08-16")
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert f"MATCH sha256={stored_sha}" in out
    assert "MISMATCH" not in out
    # Reads only: the mode wrote NOTHING (published days are never revised).
    assert client.objects == before


def test_verify_pins_to_the_record_when_the_raw_store_grows(
    monkeypatch, capsys
):
    """Legit raw growth after publication — an EARLIER-run_id duplicate
    landing in day 2's canonical slot. Fresh selection would now pick the
    earlier key (earliest-per-slot rule); verify pins the artifact's own
    recorded snapshot by exact key and still byte-matches."""
    client = _publish_two_days(monkeypatch, capsys)
    late_arrival = _day2_snapshot()
    late_arrival["run_id"] = "20260817T155900Z-0000"
    for entry in late_arrival["sources"]:
        if entry["source_id"] == "nebius":
            for obs in entry["observations"]:
                if obs.get("sku") == "B200":
                    obs["price_usd_gpu_hr"] = 9.9
                    obs["price_native_per_gpu_hr"] = 9.9
    client.objects[
        "index/b200_basket/snapshots/2026-08-17/"
        "slot16-20260817T155900Z-0000.json"
    ] = snapshot_bytes(late_arrival)

    cli = _verify(monkeypatch, client, "2026-08-17")
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "MATCH sha256=" in out


# ---------------------------------------------------------------- MISMATCH


def test_tampered_artifact_mismatches_with_field_paths(monkeypatch, capsys):
    client = _publish_day1(monkeypatch, capsys)
    stored = json.loads(client.objects[COMPOSITE_KEY_DAY1])
    original_sha = hashlib.sha256(
        client.objects[COMPOSITE_KEY_DAY1]
    ).hexdigest()
    # Tamper a published value in place (canonically re-serialized, so the
    # mismatch is CONTENT, not formatting).
    stored["index"]["value_usd_gpu_hr"] = 1.23
    for entry in stored["sources"]:
        if entry["source_id"] == "nebius":
            entry["chosen"]["usd_per_gpu_hr"] = 1.11
    client.objects[COMPOSITE_KEY_DAY1] = snapshot_bytes(stored)
    tampered_sha = hashlib.sha256(
        client.objects[COMPOSITE_KEY_DAY1]
    ).hexdigest()

    cli = _verify(monkeypatch, client, "2026-08-16")
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert f"MISMATCH 2026-08-16: published sha256={tampered_sha}" in out
    # The honest recompute lands back on the original bytes.
    assert f"recomputed sha256={original_sha}" in out
    # Field-path diffs name exactly what was tampered.
    assert "index.value_usd_gpu_hr: published 1.23" in out
    assert ".chosen.usd_per_gpu_hr: published 1.11" in out


def test_rewritten_bytes_mismatch_even_with_identical_content(
    monkeypatch, capsys
):
    """Re-serializing the artifact (same JSON content, different bytes) is
    still a MISMATCH — the record is bytes — reported as such."""
    client = _publish_day1(monkeypatch, capsys)
    stored = json.loads(client.objects[COMPOSITE_KEY_DAY1])
    client.objects[COMPOSITE_KEY_DAY1] = json.dumps(stored).encode("utf-8")

    cli = _verify(monkeypatch, client, "2026-08-16")
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "MISMATCH 2026-08-16" in out
    assert "not the canonical serialization" in out


# ------------------------------------------------------------- error paths


def test_unpublished_day_is_an_error_not_a_verdict(monkeypatch, capsys):
    client = _publish_day1(monkeypatch, capsys)
    cli = _verify(monkeypatch, client, "2026-08-17")
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "not published" in out
    assert "MATCH" not in out


def test_missing_pinned_snapshot_is_a_loud_mismatch(monkeypatch, capsys):
    client = _publish_day1(monkeypatch, capsys)
    del client.objects[SNAPSHOT_KEY]
    cli = _verify(monkeypatch, client, "2026-08-16")
    assert cli.main() == 1
    out = capsys.readouterr().out
    assert "MISMATCH 2026-08-16" in out
    assert "MISSING from the raw store" in out


def test_verify_is_its_own_mode(monkeypatch, capsys):
    client = FakeS3()
    now = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)
    cli = _wire_cli(
        monkeypatch, client, now, ["--verify-published", "2026-08-16", "--sync"]
    )
    with pytest.raises(SystemExit):
        cli.main()
    assert "its own mode" in capsys.readouterr().err


def test_malformed_date_is_a_clean_error(monkeypatch, capsys):
    client = FakeS3()
    now = datetime(2026, 8, 18, 6, 0, tzinfo=timezone.utc)
    cli = _wire_cli(monkeypatch, client, now, ["--verify-published", "yesterday"])
    assert cli.main() == 1
    assert "not a YYYY-MM-DD date" in capsys.readouterr().out
