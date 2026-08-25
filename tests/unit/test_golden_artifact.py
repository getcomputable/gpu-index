# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""THE golden-artifact test (gofmt-style: ONE golden, reviewed as code).

Rebuilds the 2026-08-16 B200 record from the committed snapshot fixture,
computes the day with the SHIPPING config (config/index_basket_b200.json,
methodology annex_a2_v0_3_calc_v5) through the real CLI, and asserts the
published artifact is BYTE-identical to tests/golden/
b200_annex_a2_v0_3_calc_v5_2026-08-16.json (sha256 committed alongside).

Any diff in these bytes is a change to what the pipeline would publish.
The policy (see RELEASING.md + GOVERNANCE.md):

  - a byte-changing PR must regenerate the golden IN THE SAME DIFF so the
    change is reviewed as code, and
  - if it alters published-value behavior it must also mint a new
    methodology version (the calculator refuses param drift mechanically).

Regeneration is explicit and local ONLY: GPU_INDEX_UPDATE_GOLDEN=1 with
env CI unset. The golden is never written implicitly and never on CI.

The fakes are local copies of test_verify_published.py's — tests/unit is
not a package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

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
COMPOSITE_KEY = (
    "index/b200_basket/composites/annex_a2_v0_3_calc_v5/2026-08-16.json"
)
GOLDEN_PATH = (
    REPO_ROOT / "tests" / "golden" / "b200_annex_a2_v0_3_calc_v5_2026-08-16.json"
)
SHA_PATH = GOLDEN_PATH.with_name(GOLDEN_PATH.name + ".sha256")

MINT_POLICY = (
    "the golden artifact is the pipeline's published bytes for 2026-08-16 — "
    "a byte-changing PR must update the golden in the same diff for review "
    "(regenerate locally with GPU_INDEX_UPDATE_GOLDEN=1, CI unset) AND, if "
    "it alters published-value behavior, mint a new methodology version "
    "(see RELEASING.md and GOVERNANCE.md; published series pin their params)"
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
            "compute_index_composite_golden",
            REPO_ROOT / "scripts" / "compute_index_composite.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLI_CACHE["mod"] = mod
    return _CLI_CACHE["mod"]


def _recompute_artifact(monkeypatch, capsys) -> bytes:
    """The full pipeline run test_verify_published.py's _publish_day1 does:
    record dir = the fixture snapshot at its exact key; --sync with the
    shipping config; the day's published artifact bytes come back."""
    monkeypatch.delenv("BASKET_CONFIG_PATH", raising=False)
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
        lambda *a, **k: pytest.fail("USD-only basket reached the FX feed"),
    )
    monkeypatch.setattr(
        cli,
        "load_stored_rates",
        lambda *a, **k: pytest.fail("USD-only basket read stored FX records"),
    )
    monkeypatch.setattr(
        cli, "utc_now", lambda: datetime(2026, 8, 17, 5, 0, tzinfo=timezone.utc)
    )
    monkeypatch.setattr(
        "sys.argv",
        ["compute_index_composite.py", "--config", str(B200_CONFIG_PATH), "--sync"],
    )
    assert cli.main() == 0
    capsys.readouterr()
    assert COMPOSITE_KEY in client.objects
    return client.objects[COMPOSITE_KEY]


def test_golden_artifact_bytes_and_sha256(monkeypatch, capsys):
    recomputed = _recompute_artifact(monkeypatch, capsys)
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
            "golden artifact or its .sha256 is missing — it is never "
            f"written implicitly; {MINT_POLICY}"
        )
    golden = GOLDEN_PATH.read_bytes()
    committed_sha = SHA_PATH.read_text().split()[0]
    assert hashlib.sha256(golden).hexdigest() == committed_sha, (
        "the committed golden and its .sha256 disagree — regenerate both "
        f"together; {MINT_POLICY}"
    )
    if recomputed != golden:
        pytest.fail(
            f"recomputed 2026-08-16 artifact (sha256={recomputed_sha}) does "
            f"not byte-match the committed golden (sha256={committed_sha}) — "
            + MINT_POLICY
        )
    assert recomputed_sha == committed_sha
