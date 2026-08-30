# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""PublishedRecordReader backends + the verify_published_record CLI.

The reader rides the shared transport (local record copy, the anonymous
public HTTPS front, or S3) and every read digest-verifies before
returning. The CLI is the public face ./reproduce execs by default:
per observation it prints recomputed vs published with MATCH/MISMATCH
and the file digest verdict; exit 0 all-match, 1 any mismatch or digest
FAIL, 2 could not verify (nothing published for the request, or the
record source is unreachable); withheld-degraded observations get a
distinct message without failing the run.

The CLI is loaded via importlib (tests/unit is not a package), the same
pattern the panel verify CLI tests use.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest

from gpu_index.common.bucket import BucketConfig
from gpu_index.published.artifacts import payload_digest
from gpu_index.published.reader import PublishedRecordReader

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "published"


@pytest.fixture()
def cli():
    spec = importlib.util.spec_from_file_location(
        "verify_published_record",
        REPO_ROOT / "scripts" / "verify_published_record.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def record_env(monkeypatch):
    """Point the shared transport at the fixture record copy."""
    monkeypatch.setenv("GPU_INDEX_DATA_DIR", str(FIXTURES))
    monkeypatch.delenv("GPU_INDEX_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("GPU_INDEX_S3_ENDPOINT", raising=False)


def _local_reader(root: Path) -> PublishedRecordReader:
    return PublishedRecordReader(
        BucketConfig(backend="local", bucket="local", local_root=root)
    )


def _tampered_record(tmp_path: Path, key: str, mutate, redigest=True) -> Path:
    root = tmp_path / "record"
    shutil.copytree(FIXTURES, root)
    target = root / key
    document = json.loads(target.read_text())
    mutate(document)
    if redigest:
        payload = {k: document[k] for k in ("data", "meta", "license")}
        document["artifact_sha256"] = payload_digest(payload)
    target.write_text(json.dumps(document, indent=2) + "\n")
    return root


def _versioned_record(tmp_path: Path) -> Path:
    root = tmp_path / "versioned-record"
    shutil.copytree(FIXTURES, root)
    latest_path = root / "latest.json"
    latest = json.loads(latest_path.read_text())
    current_methodology = next(
        row["methodology_id"]
        for row in latest["data"]["observations"]
        if row["sku"] == "H100"
    )
    latest["data"]["versions"] = [
        {
            "sku": "H100",
            "current_version": 2,
            "methodology_id": current_methodology,
            "effective_from": "2026-08-29T00:00:00Z",
            "succession": [
                {
                    "version": 1,
                    "methodology_id": "annex_h100_v1",
                    "effective_from": "2026-08-10T00:00:00Z",
                },
                {
                    "version": 2,
                    "methodology_id": current_methodology,
                    "effective_from": "2026-08-29T00:00:00Z",
                },
            ],
        }
    ]
    payload = {key: latest[key] for key in ("data", "meta", "license")}
    latest["artifact_sha256"] = payload_digest(payload)
    latest_path.write_text(json.dumps(latest, indent=2) + "\n")
    for source in [
        root / "observations" / "2026" / "08" / "25.json",
        root / "series" / "24h.json",
    ]:
        suffix = source.relative_to(root)
        for version in (1, 2):
            target = root / "H100" / f"v{version}" / suffix
            target.parent.mkdir(parents=True, exist_ok=True)
            document = json.loads(source.read_text())
            if version == 1:
                for observation in document["data"]["observations"]:
                    observation["methodology_id"] = "annex_h100_v1"
                    if "calc_params" in observation:
                        observation["calc_params"][
                            "methodology_id"
                        ] = "annex_h100_v1"
                payload = {
                    key: document[key] for key in ("data", "meta", "license")
                }
                document["artifact_sha256"] = payload_digest(payload)
            target.write_text(json.dumps(document, indent=2) + "\n")
        source.unlink()
    return root


# -------------------------------------------------------------------- reader


def test_local_reader_reads_all_layout_keys():
    reader = _local_reader(FIXTURES)
    assert reader.read_latest()["data"]["kind"] == "gpu_index_latest"
    day = reader.read_day("2026-08-25")
    assert day["data"]["kind"] == "gpu_index_observation_day"
    assert day["data"]["date"] == "2026-08-25"
    series = reader.read_series("24h")
    assert series["data"]["range"] == "24h"
    assert "local record copy" in reader.describe()


def test_local_reader_missing_day_is_none_not_an_error():
    # Day files are mutable/deletable by design inside the publisher's
    # trailing window; a missing day is an ordinary state.
    assert _local_reader(FIXTURES).read_day("2020-01-01") is None


def test_reader_dual_resolves_current_and_explicit_versions(tmp_path):
    reader = _local_reader(_versioned_record(tmp_path))
    assert (
        reader.resolve_day_key("2026-08-25", sku="H100")
        == "H100/v2/observations/2026/08/25.json"
    )
    assert reader.read_day("2026-08-25", sku="H100")["data"]["date"] == "2026-08-25"
    assert reader.read_day("2026-08-25", sku="H100", version=1)["data"]["date"] == "2026-08-25"
    assert reader.read_series("24h", sku="H100")["data"]["range"] == "24h"
    assert (
        reader.resolve_series_key("24h", sku="H100", version=1)
        == "H100/v1/series/24h.json"
    )


def test_reader_explicit_version_can_verify_staged_objects_before_pointer_swap(tmp_path):
    root = tmp_path / "staged-record"
    shutil.copytree(FIXTURES, root)
    target = root / "H100" / "v1" / "observations" / "2026" / "08" / "25.json"
    target.parent.mkdir(parents=True)
    shutil.copyfile(root / "observations" / "2026" / "08" / "25.json", target)
    reader = _local_reader(root)
    assert reader.read_day("2026-08-25", sku="H100", version=1) is not None


def test_reader_rejects_content_cross_wired_between_versions(tmp_path):
    root = _versioned_record(tmp_path)
    v1_day = root / "H100" / "v1" / "observations" / "2026" / "08" / "25.json"
    v2_day = root / "H100" / "v2" / "observations" / "2026" / "08" / "25.json"
    shutil.copyfile(v2_day, v1_day)

    with pytest.raises(ValueError, match="disagrees with its version identity"):
        _local_reader(root).read_day("2026-08-25", sku="H100", version=1)


def test_public_https_reader_digest_verifies(monkeypatch):
    base = "https://record.example.com/cgi"

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path.removeprefix("/cgi/")
        path = FIXTURES / key
        if not path.is_file():
            return httpx.Response(404)
        return httpx.Response(200, content=path.read_bytes())

    monkeypatch.setenv("GPU_INDEX_PUBLIC_BASE_URL", base)
    monkeypatch.delenv("GPU_INDEX_S3_ENDPOINT", raising=False)
    reader = PublishedRecordReader()
    # Inject the mock transport through the store's test seam.
    reader._client._client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )
    day = reader.read_day("2026-08-20")
    assert day["data"]["date"] == "2026-08-20"
    assert reader.read_day("2020-01-01") is None
    assert base in reader.describe()


# ----------------------------------------------------------------------- CLI


def _run(monkeypatch, cli, *argv) -> int:
    monkeypatch.setattr(
        sys, "argv", ["verify_published_record.py", *argv]
    )
    return cli.main()


def test_cli_full_match_exits_zero(record_env, monkeypatch, cli, capsys):
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 0
    out = capsys.readouterr().out
    assert "digest OK" in out
    assert out.count(" MATCH digest OK") == 2
    assert " MISMATCH digest OK" not in out
    assert "recomputed 2.067501" in out
    assert "published 2.067501" in out
    assert "2 MATCH, 0 MISMATCH, 0 degraded" in out


def test_cli_full_match_through_versioned_pointer_exits_zero(
    tmp_path, monkeypatch, cli, capsys
):
    root = _versioned_record(tmp_path)
    monkeypatch.setenv("GPU_INDEX_DATA_DIR", str(root))
    monkeypatch.delenv("GPU_INDEX_PUBLIC_BASE_URL", raising=False)
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 0
    out = capsys.readouterr().out
    assert "2 MATCH, 0 MISMATCH, 0 degraded" in out


def test_cli_announces_the_versioned_day_key_it_reads(
    tmp_path, monkeypatch, cli, capsys
):
    root = _versioned_record(tmp_path)
    base = "https://record.example.com/cgi"
    requested_keys = []
    latest_reads = 0
    moved_latest = json.loads((root / "latest.json").read_text())
    pointer = moved_latest["data"]["versions"][0]
    pointer["current_version"] = 1
    pointer["methodology_id"] = "annex_h100_v1"
    pointer["effective_from"] = pointer["succession"][0]["effective_from"]
    payload = {
        key: moved_latest[key] for key in ("data", "meta", "license")
    }
    moved_latest["artifact_sha256"] = payload_digest(payload)
    moved_latest_bytes = (json.dumps(moved_latest, indent=2) + "\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal latest_reads
        key = request.url.path.removeprefix("/cgi/")
        requested_keys.append(key)
        if key == "latest.json":
            latest_reads += 1
            if latest_reads > 1:
                return httpx.Response(200, content=moved_latest_bytes)
        path = root / key
        if not path.is_file():
            return httpx.Response(404)
        return httpx.Response(200, content=path.read_bytes())

    reader = PublishedRecordReader(
        BucketConfig(
            backend="public",
            bucket="public",
            public_base_url=base,
        )
    )
    reader._client._client = httpx.Client(
        transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(cli, "PublishedRecordReader", lambda: reader)

    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 0
    output = capsys.readouterr().out
    banner = next(
        line for line in output.splitlines() if line.startswith("published record: ")
    )
    announced_key = banner.removeprefix("published record: ").split(" via ", 1)[0]
    (read_key,) = {
        key
        for key in requested_keys
        if key.endswith("observations/2026/08/25.json")
    }
    assert announced_key == read_key


def test_cli_single_stamp_targets_one_observation(
    record_env, monkeypatch, cli, capsys
):
    assert (
        _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25T15")
        == 0
    )
    out = capsys.readouterr().out
    assert "1 observation(s): 1 MATCH" in out


def test_cli_slot_era_day_verifies_ok_and_no_print(
    record_env, monkeypatch, cli, capsys
):
    assert _run(monkeypatch, cli, "--sku", "B200", "--date", "2026-08-20") == 0
    out = capsys.readouterr().out
    assert "NO-PRINT (insufficient_coverage)" in out
    assert "2 MATCH" in out


def test_cli_value_tamper_exits_one_naming_the_field(
    tmp_path, monkeypatch, cli, capsys
):
    def mutate(document):
        document["data"]["observations"][0]["value_usd_gpu_hr"] += 0.01

    root = _tampered_record(tmp_path, "observations/2026/08/25.json", mutate)
    monkeypatch.setenv("GPU_INDEX_DATA_DIR", str(root))
    monkeypatch.delenv("GPU_INDEX_PUBLIC_BASE_URL", raising=False)
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out
    assert "value_usd_gpu_hr" in out


def test_cli_digest_tamper_exits_one_with_digest_fail(
    tmp_path, monkeypatch, cli, capsys
):
    def mutate(document):
        document["data"]["observations"][0]["value_usd_gpu_hr"] += 0.01

    root = _tampered_record(
        tmp_path, "observations/2026/08/25.json", mutate, redigest=False
    )
    monkeypatch.setenv("GPU_INDEX_DATA_DIR", str(root))
    monkeypatch.delenv("GPU_INDEX_PUBLIC_BASE_URL", raising=False)
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 1
    err = capsys.readouterr().err
    assert "digest FAIL" in err


def test_cli_withheld_degrades_with_distinct_message_and_exit_zero(
    record_env, monkeypatch, cli, capsys
):
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-23") == 0
    out = capsys.readouterr().out
    assert "DEGRADED digest-only" in out
    assert "withheld: charlie" in out
    assert "1 degraded" in out
    assert "NOTE:" in out
    assert "0 MISMATCH" in out
    assert " MISMATCH digest OK" not in out


def test_cli_unreachable_front_exits_two_with_one_actionable_line(
    record_env, monkeypatch, cli, capsys
):
    # An unreachable record source must be "could not verify" (exit 2)
    # with an actionable message -- never a traceback, and never exit 0
    # having verified nothing.
    class _DeadFrontReader:
        def describe(self):
            return "public HTTPS front https://data.getcomputable.com"

        def resolve_day_key(self, date, *, sku):
            return f"observations/{date.replace('-', '/')}.json"

        def read_day(self, date, *, sku=None, resolved_key=None):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        cli, "PublishedRecordReader", lambda *a, **k: _DeadFrontReader()
    )
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 2
    err = capsys.readouterr().err
    assert "could not verify" in err
    assert "unreachable" in err
    assert "GPU_INDEX_PUBLIC_BASE_URL" in err
    assert "GPU_INDEX_DATA_DIR" in err
    assert "Traceback" not in err


def test_cli_broken_front_non_200_exits_two_not_a_traceback(
    record_env, monkeypatch, cli, capsys
):
    from gpu_index.common.bucket import BucketPublishError

    class _BrokenFrontReader:
        def describe(self):
            return "public HTTPS front https://data.getcomputable.com"

        def resolve_day_key(self, date, *, sku):
            return f"observations/{date.replace('-', '/')}.json"

        def read_day(self, date, *, sku=None, resolved_key=None):
            raise BucketPublishError(
                "public read backend: GET .../25.json returned HTTP 503"
            )

    monkeypatch.setattr(
        cli, "PublishedRecordReader", lambda *a, **k: _BrokenFrontReader()
    )
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 2
    err = capsys.readouterr().err
    assert "could not verify" in err
    assert "HTTP 503" in err


def test_cli_conflicting_backend_env_exits_two(monkeypatch, cli, capsys):
    # Two read backends configured at once is a config refusal in the
    # transport; the CLI turns it into exit 2, not a traceback.
    monkeypatch.setenv(
        "GPU_INDEX_PUBLIC_BASE_URL", "https://record.example.com/cgi"
    )
    monkeypatch.setenv("GPU_INDEX_S3_ENDPOINT", "https://s3.example.com")
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-08-25") == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_cli_missing_day_file_exits_two_pointing_at_producer(
    record_env, monkeypatch, cli, capsys
):
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2020-01-01") == 2
    err = capsys.readouterr().err
    assert "no published day file" in err
    assert "--producer" in err


def test_cli_unknown_sku_in_day_exits_two(
    record_env, monkeypatch, cli, capsys
):
    assert _run(monkeypatch, cli, "--sku", "B300", "--date", "2026-08-25") == 2
    err = capsys.readouterr().err
    assert "no B300 observation" in err


def test_cli_bad_date_exits_two(record_env, monkeypatch, cli, capsys):
    assert _run(monkeypatch, cli, "--sku", "H100", "--date", "2026-8-25") == 2
    assert "bad date" in capsys.readouterr().err
