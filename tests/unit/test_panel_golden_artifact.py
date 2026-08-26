# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""THE panel-engine golden-artifact test (the daily golden's sibling: ONE
golden per engine, reviewed as code).

Rebuilds the 2026-08-16 B200 record from the SAME committed snapshot
fixture the daily golden uses, computes the basket-era (stitched) T22
observation with the SHIPPING panel config (config/index_panel_b200.json,
methodology annex_a2_v0_3_calc_v6) through the real panel CLI, and asserts
the published hourly artifact is BYTE-identical to tests/golden/
b200_panel_annex_a2_v0_3_calc_v6_2026-08-16T22.json (sha256 committed
alongside). 2026-08-16 is the lane's genesis day inside its 4-slot
basket-record era, so this pins the record-stitching read path, the
embedded calc_params (the D2 fence's bytes, manual-exclusion pin
included), the explicit observation_missed artifacts never publishing in
the golden's place, and the observation composite itself.

Any diff in these bytes is a change to what the panel pipeline would
publish. Policy and regeneration discipline are the daily golden's
verbatim (see RELEASING.md + GOVERNANCE.md): a byte-changing PR must
regenerate the golden IN THE SAME DIFF, published-value changes must mint
a new methodology version, and regeneration is explicit and local ONLY
(GPU_INDEX_UPDATE_GOLDEN=1 with env CI unset) — never implicit, never on
CI.

The fakes are local copies of test_golden_artifact.py's — tests/unit is
not a package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_CONFIG_PATH = REPO_ROOT / "config" / "index_panel_b200.json"
FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "b200_snapshot_2026-08-16_slot22-20260816T221613Z-1fdf.json"
)
SNAPSHOT_KEY = (
    "index/b200_basket/snapshots/2026-08-16/slot22-20260816T221613Z-1fdf.json"
)
OBSERVATION_KEY = (
    "index/b200_basket/composites/annex_a2_v0_3_calc_v6/2026-08-16T22.json"
)
GOLDEN_PATH = (
    REPO_ROOT
    / "tests"
    / "golden"
    / "b200_panel_annex_a2_v0_3_calc_v6_2026-08-16T22.json"
)
SHA_PATH = GOLDEN_PATH.with_name(GOLDEN_PATH.name + ".sha256")

MINT_POLICY = (
    "the panel golden is the hourly pipeline's published bytes for "
    "2026-08-16T22 — a byte-changing PR must update the golden in the same "
    "diff for review (regenerate locally with GPU_INDEX_UPDATE_GOLDEN=1, CI "
    "unset) AND, if it alters published-value behavior, mint a new "
    "methodology version (see RELEASING.md and GOVERNANCE.md; published "
    "series pin their params)"
)


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

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}


_CLI_CACHE = {}


def _load_cli():
    if "mod" not in _CLI_CACHE:
        spec = importlib.util.spec_from_file_location(
            "compute_panel_index_golden",
            REPO_ROOT / "scripts" / "compute_panel_index.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _recompute_observation(monkeypatch, capsys) -> bytes:
    """One real --sync of the shipping B200 panel config over a record
    holding exactly the committed fixture snapshot: the genesis day's
    T04/T10/T16 publish as explicit observation_missed artifacts and T22
    computes from the fixture (utc_now sits past T22's closing mark, the
    next scheduled stamp 2026-08-17T04)."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    cli = _load_cli()
    client = FakeS3()
    client.objects[SNAPSHOT_KEY] = FIXTURE_PATH.read_bytes()

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
        lambda *a, **k: pytest.fail("USD-only panel reached the FX feed"),
    )
    monkeypatch.setattr(
        cli,
        "load_stored_rates",
        lambda *a, **k: pytest.fail("USD-only panel read stored FX records"),
    )
    monkeypatch.setattr(
        cli, "utc_now", lambda: datetime(2026, 8, 17, 5, 0, tzinfo=timezone.utc)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "compute_panel_index.py",
            "--config",
            str(PANEL_CONFIG_PATH),
            "--sync",
        ],
    )
    assert cli.main() == 0
    capsys.readouterr()
    # The genesis day's four scheduled stamps all published: three explicit
    # missed artifacts and the computed T22 — never a skipped loop index.
    for stamp in ("2026-08-16T04", "2026-08-16T10", "2026-08-16T16"):
        key = (
            "index/b200_basket/composites/annex_a2_v0_3_calc_v6/"
            f"{stamp}.json"
        )
        assert key in client.objects
        assert b'"observation_missed": true' in client.objects[key]
    assert OBSERVATION_KEY in client.objects
    return client.objects[OBSERVATION_KEY]


def test_panel_golden_artifact_bytes_and_sha256(monkeypatch, capsys):
    recomputed = _recompute_observation(monkeypatch, capsys)
    recomputed_sha = hashlib.sha256(recomputed).hexdigest()

    on_ci = bool(os.environ.get("CI"))
    update = os.environ.get("GPU_INDEX_UPDATE_GOLDEN") == "1"
    if update and not on_ci:
        # The ONLY write path: explicit, local, and reviewed via git diff.
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_bytes(recomputed)
        SHA_PATH.write_text(recomputed_sha + "\n")

    if not GOLDEN_PATH.exists() or not SHA_PATH.exists():
        pytest.fail(
            "panel golden artifact or its .sha256 is missing — it is never "
            f"written implicitly; {MINT_POLICY}"
        )
    golden = GOLDEN_PATH.read_bytes()
    committed_sha = SHA_PATH.read_text().split()[0]
    assert hashlib.sha256(golden).hexdigest() == committed_sha, (
        "the committed panel golden and its .sha256 disagree — regenerate "
        f"both together; {MINT_POLICY}"
    )
    if recomputed != golden:
        pytest.fail(
            f"recomputed 2026-08-16T22 artifact (sha256={recomputed_sha}) "
            "does not byte-match the committed panel golden "
            f"(sha256={committed_sha}) — " + MINT_POLICY
        )
    assert recomputed_sha == committed_sha
