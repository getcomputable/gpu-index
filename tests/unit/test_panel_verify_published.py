# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""--verify-published on the PANEL CLI: byte-for-byte recompute of one
PUBLISHED hourly observation.

The mode's contract (the daily CLI's verify_published re-minted per
observation): prior published artifacts advance the replay state, the
observation's snapshot is the artifact's own pinned choice fetched by
exact key, write NOTHING, canonicalize, and byte-compare against the
stored artifact. MATCH exits 0 with the sha256; MISMATCH exits 1 with
field-path diffs.

Runs on the frozen B200 v6 parameters reconstructed from the shipping
config (fx_lane none, so the FX
machinery is provably untouched) over a REAL local-directory record in
tmp_path, built by the CLI's own --sync from the SAME committed snapshot
fixture the panel golden uses -- so the MATCH case is pinned to the
committed golden sha256, and every path (record build, verify replay,
pinned-snapshot GET) crosses the real LocalStore backend. The CLI is
loaded via importlib (tests/unit is not a package).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_index.common.store import snapshot_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_CONFIG_PATH = REPO_ROOT / "config" / "index_panel_b200.json"
FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "b200_snapshot_2026-08-16_slot22-20260816T221613Z-1fdf.json"
)
GOLDEN_SHA_PATH = (
    REPO_ROOT
    / "tests"
    / "golden"
    / "b200_panel_annex_a2_v0_3_calc_v6_2026-08-16T22.json.sha256"
)
SNAPSHOT_REL = (
    "index/b200_basket/snapshots/2026-08-16/slot22-20260816T221613Z-1fdf.json"
)
ARTIFACT_REL = (
    "index/b200_basket/composites/annex_a2_v0_3_calc_v6/2026-08-16T22.json"
)

_CLI_CACHE = {}


def _load_cli():
    if "mod" not in _CLI_CACHE:
        spec = importlib.util.spec_from_file_location(
            "compute_panel_index_verify",
            REPO_ROOT / "scripts" / "compute_panel_index.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _wire(monkeypatch):
    """Real LocalStore backend from env; the USD-only lane must never
    touch FX (neither the live feed nor the stored-rates read)."""
    cli = _load_cli()
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GPU_INDEX_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("GPU_INDEX_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setattr(
        cli,
        "ensure_rates",
        lambda *a, **k: pytest.fail("USD-only panel reached the FX feed"),
    )
    monkeypatch.setattr(
        cli,
        "load_stored_rates",
        lambda *a, **k: pytest.fail("USD-only panel read stored FX records"),
    )
    return cli


def _run(monkeypatch, cli, *argv) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        ["compute_panel_index.py", "--config", str(PANEL_CONFIG_PATH), *argv],
    )
    return cli.main()


def _frozen_v6_config(tmp_path: Path) -> Path:
    """Reconstruct the immutable v6 config after the shipping file mints on."""
    config = json.loads(PANEL_CONFIG_PATH.read_text())
    config["slot_grids"] = config["slot_grids"][:2]
    calc = config["calc"]
    calc["methodology_id"] = "annex_a2_v0_3_calc_v6"
    calc["description"] = (
        "calc_v6 hourly panel mint (annex_a2_v0_3_calc_v6); "
        "METHODOLOGY.md is the binding methodology document"
    )
    for key in (
        "filter_sigma_floor_pct",
        "iqm_alpha",
        "vote_sigma_source",
        "vote_sigma_floor_pct",
    ):
        calc.pop(key)
    calc["filter_sigma_floor"] = 0.05
    path = tmp_path / "index_panel_b200_calc_v6.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


@pytest.fixture()
def record_dir(tmp_path, monkeypatch, capsys):
    """A real local-directory record holding the committed fixture
    snapshot plus the four genesis-day published artifacts, built by one
    real --sync (utc_now pinned past T22's closing mark, exactly the
    panel golden test's world)."""
    data = tmp_path / "data"
    snapshot_path = data / SNAPSHOT_REL
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(FIXTURE_PATH.read_bytes())
    monkeypatch.setenv("GPU_INDEX_DATA_DIR", str(data))
    monkeypatch.setattr(
        sys.modules[__name__], "PANEL_CONFIG_PATH", _frozen_v6_config(tmp_path)
    )
    cli = _wire(monkeypatch)
    monkeypatch.setattr(
        cli, "utc_now", lambda: datetime(2026, 8, 17, 5, 0, tzinfo=timezone.utc)
    )
    assert _run(monkeypatch, cli, "--sync") == 0
    assert (data / ARTIFACT_REL).exists()
    capsys.readouterr()
    return data


def test_verify_published_match_pins_golden_sha(record_dir, monkeypatch, capsys):
    cli = _wire(monkeypatch)
    rc = _run(monkeypatch, cli, "--verify-published", "2026-08-16T22")
    out = capsys.readouterr().out
    assert rc == 0
    committed_sha = GOLDEN_SHA_PATH.read_text().split()[0]
    assert f"MATCH sha256={committed_sha}" in out
    assert "verify-published 2026-08-16T22" in out
    assert "MISMATCH" not in out


def test_verify_published_missed_observation_matches(
    record_dir, monkeypatch, capsys
):
    # The explicit observation_missed artifacts verify byte-for-byte too:
    # the missed verdict is a published fact of the series, not a gap.
    cli = _wire(monkeypatch)
    rc = _run(monkeypatch, cli, "--verify-published", "2026-08-16T10")
    out = capsys.readouterr().out
    assert rc == 0
    assert "MATCH sha256=" in out
    assert "[observation_missed]" in out


def test_verify_published_tampered_value_mismatches(
    record_dir, monkeypatch, capsys
):
    artifact = record_dir / ARTIFACT_REL
    doc = json.loads(artifact.read_bytes())
    doc["index"]["value_usd_gpu_hr"] = doc["index"]["value_usd_gpu_hr"] + 1.0
    # Canonical re-serialization, so the ONLY divergence is the tampered
    # value and the diff must name its exact field path.
    tampered = snapshot_bytes(doc)
    artifact.write_bytes(tampered)
    tampered_sha = hashlib.sha256(tampered).hexdigest()

    cli = _wire(monkeypatch)
    rc = _run(monkeypatch, cli, "--verify-published", "2026-08-16T22")
    out = capsys.readouterr().out
    assert rc == 1
    assert f"published sha256={tampered_sha} vs recomputed sha256=" in out
    assert "MISMATCH 2026-08-16T22" in out
    assert "index.value_usd_gpu_hr" in out
    # The recompute reproduces the ORIGINAL golden bytes.
    committed_sha = GOLDEN_SHA_PATH.read_text().split()[0]
    assert f"recomputed sha256={committed_sha}" in out


def test_verify_published_rewritten_bytes_mismatch(
    record_dir, monkeypatch, capsys
):
    # Same JSON content, non-canonical serialization: the verdict must
    # say the bytes were rewritten rather than show field diffs.
    artifact = record_dir / ARTIFACT_REL
    doc = json.loads(artifact.read_bytes())
    artifact.write_text(json.dumps(doc, sort_keys=True, indent=4))
    cli = _wire(monkeypatch)
    rc = _run(monkeypatch, cli, "--verify-published", "2026-08-16T22")
    out = capsys.readouterr().out
    assert rc == 1
    assert "MISMATCH 2026-08-16T22" in out
    assert "the artifact bytes were rewritten" in out


def test_verify_published_pinned_snapshot_missing_is_loud(
    record_dir, monkeypatch, capsys
):
    (record_dir / SNAPSHOT_REL).unlink()
    cli = _wire(monkeypatch)
    rc = _run(monkeypatch, cli, "--verify-published", "2026-08-16T22")
    out = capsys.readouterr().out
    assert rc == 1
    assert "pinned snapshot" in out
    assert "MISSING" in out


def test_verify_published_unpublished_refuses(record_dir, monkeypatch, capsys):
    cli = _wire(monkeypatch)
    rc = _run(monkeypatch, cli, "--verify-published", "2026-08-17T04")
    out = capsys.readouterr().out
    assert rc == 1
    assert "not published under annex_a2_v0_3_calc_v6" in out
    assert "--observation 2026-08-17T04 --dry-run" in out


def test_verify_published_unscheduled_stamp_refuses(
    record_dir, monkeypatch, capsys
):
    # 2026-08-16 sits in the 4-slot era: hour 05 is not on the grid.
    cli = _wire(monkeypatch)
    rc = _run(monkeypatch, cli, "--verify-published", "2026-08-16T05")
    out = capsys.readouterr().out
    assert rc == 1
    assert "not a scheduled observation" in out


def test_verify_published_is_its_own_mode(record_dir, monkeypatch, capsys):
    cli = _wire(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, cli, "--verify-published", "2026-08-16T22", "--sync")
    assert excinfo.value.code == 2
