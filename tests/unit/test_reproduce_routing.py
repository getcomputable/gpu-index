# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""./reproduce routing: published record first, producer lanes behind it.

The launch promise is `./reproduce h100 <date>`. The DEFAULT always
consumes the PUBLISHED record, routing to the recompute-and-match
verifier (verify_published_record.py): a local copy holding the
requested day file, else the public front -- GPU_INDEX_PUBLIC_BASE_URL
when set, defaulting to the official record host
https://data.getcomputable.com -- so a clean clone with no env verifies
the live record instead of silently replaying nothing. Only under
--producer does the public SKU map to its live hourly panel lane:
h100 -> h100_sxm, h200 -> h200_sxm, b300 -> b300, b200 -> b200, routed
to the PANEL engine (compute_panel_index.py). A date without an hour
covers the whole UTC day; with THH it targets one observation; anything
already in the local producer record auto-routes to --verify-published.
The broad lanes exist but are NOT public SKUs and require the explicit
--lane flag; --frozen keeps the retired daily-lane behavior.

These tests pin the ROUTING, not the engine (the engine's own suites do
that): PYTHON is pointed at a shim that passes `-c` config reads through
to the real interpreter and prints any other invocation's argv instead
of executing it, so no replay, no network, and no clock dependency
beyond "the fixture dates are in the past".
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCE = REPO_ROOT / "reproduce"


def _lane_meta(config_name: str) -> tuple[str, str]:
    config = json.loads((REPO_ROOT / "config" / config_name).read_text())
    return config["bucket_prefix"], config["calc"]["methodology_id"]


@pytest.fixture()
def shim(tmp_path) -> Path:
    # SHIM: pins the routed argv; SHIMENV: pins the public-front env the
    # routed process would see (empty = local record copy / producer).
    path = tmp_path / "python-shim"
    path.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-c" ]; then exec /usr/bin/env python3 "$@"; fi\n'
        "printf 'SHIMENV:%s\\n' \"${GPU_INDEX_PUBLIC_BASE_URL:-}\"\n"
        "printf 'SHIM:%s\\n' \"$*\"\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture()
def data_dir(tmp_path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    return data


def _run(shim, data_dir, *args, base_url=None):
    env = dict(os.environ)
    env["PYTHON"] = str(shim)
    env["GPU_INDEX_DATA_DIR"] = str(data_dir)
    # Hermetic: a developer's exported public front must not flip these
    # routing assertions into published mode.
    env.pop("GPU_INDEX_PUBLIC_BASE_URL", None)
    if base_url is not None:
        env["GPU_INDEX_PUBLIC_BASE_URL"] = base_url
    return subprocess.run(
        [str(REPRODUCE), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _shim_lines(result) -> list:
    return [
        line for line in result.stdout.splitlines() if line.startswith("SHIM:")
    ]


def _shim_env(result) -> str:
    (line,) = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("SHIMENV:")
    ]
    return line.removeprefix("SHIMENV:")


def _publish(data_dir: Path, config_name: str, stamp: str) -> None:
    prefix, methodology = _lane_meta(config_name)
    artifact = data_dir / prefix / "composites" / methodology / f"{stamp}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n")


# --------------------------------------- published record: default mode


def _publish_public_day(data_dir: Path, date: str) -> None:
    year, month, day = date.split("-")
    target = data_dir / "observations" / year / month / f"{day}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n")


def _publish_versioned_public_day(
    data_dir: Path, sku: str, date: str, version: int = 2
) -> None:
    latest = {
        "data": {
            "versions": [
                {"sku": sku, "current_version": version},
            ]
        }
    }
    (data_dir / "latest.json").write_text(json.dumps(latest) + "\n")
    year, month, day = date.split("-")
    target = (
        data_dir
        / sku
        / f"v{version}"
        / "observations"
        / year
        / month
        / f"{day}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n")


def test_default_with_no_env_verifies_against_the_official_front(
    shim, data_dir
):
    # THE flagship command: a clean clone, no env, empty data dir must
    # verify the live published record via the official host -- never
    # fall through to the producer replay having verified nothing.
    result = _run(shim, data_dir, "h100", "2026-08-24")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert "--sku H100 --date 2026-08-24" in line
    assert _shim_env(result) == "https://data.getcomputable.com"


def test_local_published_day_routes_to_the_published_verifier(
    shim, data_dir
):
    _publish_public_day(data_dir, "2026-08-25")
    result = _run(shim, data_dir, "h100", "2026-08-25T14")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert "--sku H100 --date 2026-08-25T14" in line
    # A local copy holding the day keeps the local backend: the official
    # front is a fallback, not an override of the downloaded record.
    assert _shim_env(result) == ""


def test_public_base_url_routes_to_the_published_verifier(shim, data_dir):
    # A configured public front is intent: published mode even with an
    # empty local data dir, and the operator's URL wins over the default.
    result = _run(
        shim,
        data_dir,
        "b200",
        "2026-08-20",
        base_url="https://record.example.com/cgi",
    )
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert "--sku B200 --date 2026-08-20" in line
    assert _shim_env(result) == "https://record.example.com/cgi"


def test_full_mode_routes_to_the_raw_only_public_derivation(shim, data_dir):
    result = _run(shim, data_dir, "--full", "h100", "2026-09-01")

    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert "--sku H100 --date 2026-09-01 --full" in line
    assert _shim_env(result) == "https://data.getcomputable.com"


def test_producer_flag_forces_the_internal_replay(shim, data_dir):
    # Even with a published day file present, --producer keeps the
    # previous producer-record semantics verbatim.
    _publish_public_day(data_dir, "2026-08-24")
    result = _run(shim, data_dir, "--producer", "h100", "2026-08-24T05")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "compute_panel_index.py" in line
    assert "index_panel_h100_sxm.json" in line
    assert "--observation 2026-08-24T05 --dry-run" in line


def test_published_day_for_another_date_falls_to_the_official_front(
    shim, data_dir
):
    # The probe is day-specific: a local copy of some OTHER day does not
    # cover this date, so the run verifies via the public front instead
    # (default host; still the published verifier, never the producer).
    _publish_public_day(data_dir, "2026-08-25")
    result = _run(shim, data_dir, "h100", "2026-08-24T05")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert _shim_env(result) == "https://data.getcomputable.com"


def test_local_versioned_public_day_keeps_the_local_backend(shim, data_dir):
    _publish_versioned_public_day(data_dir, "H100", "2026-08-25")
    result = _run(shim, data_dir, "h100", "2026-08-25T14")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert "--sku H100 --date 2026-08-25T14" in line
    assert _shim_env(result) == ""


def test_versioned_public_day_for_another_date_falls_to_official_front(
    shim, data_dir
):
    _publish_versioned_public_day(data_dir, "H100", "2026-08-25")
    result = _run(shim, data_dir, "h100", "2026-08-24T05")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert _shim_env(result) == "https://data.getcomputable.com"


def test_version_pointer_for_another_sku_falls_to_official_front(
    shim, data_dir
):
    _publish_versioned_public_day(data_dir, "B200", "2026-08-25")
    result = _run(shim, data_dir, "h100", "2026-08-25")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "verify_published_record.py" in line
    assert _shim_env(result) == "https://data.getcomputable.com"


def test_producer_flag_still_refuses_unknown_skus(shim, data_dir):
    result = _run(shim, data_dir, "--producer", "h300", "2026-08-24")
    assert result.returncode == 2
    assert "unknown sku 'h300'" in result.stderr


# ------------------------------------------- sku -> lane map (producer)


def test_h100_resolves_to_h100_sxm_panel_lane(shim, data_dir):
    result = _run(shim, data_dir, "--producer", "h100", "2026-08-24T05")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "compute_panel_index.py" in line
    assert "index_panel_h100_sxm.json" in line
    assert "--observation 2026-08-24T05 --dry-run" in line


def test_h200_b300_b200_resolve_to_their_panel_lanes(shim, data_dir):
    for sku, config in (
        ("h200", "index_panel_h200_sxm.json"),
        ("b300", "index_panel_b300.json"),
        ("b200", "index_panel_b200.json"),
    ):
        result = _run(shim, data_dir, "--producer", sku, "2026-08-24T05")
        assert result.returncode == 0, result.stderr
        (line,) = _shim_lines(result)
        assert "compute_panel_index.py" in line
        assert config in line


def test_unknown_sku_refuses(shim, data_dir):
    result = _run(shim, data_dir, "h300", "2026-08-24")
    assert result.returncode == 2
    assert "unknown sku 'h300'" in result.stderr


# -------------------------------- published -> verification (producer)


def test_published_observation_auto_routes_to_verify(shim, data_dir):
    _publish(data_dir, "index_panel_b200.json", "2026-08-16T22")
    result = _run(shim, data_dir, "--producer", "b200", "2026-08-16T22")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "compute_panel_index.py" in line
    assert "--verify-published 2026-08-16T22" in line


def test_day_mode_verifies_published_then_derives_the_rest(shim, data_dir):
    _publish(data_dir, "index_panel_h200_sxm.json", "2026-08-20T03")
    _publish(data_dir, "index_panel_h200_sxm.json", "2026-08-20T07")
    result = _run(shim, data_dir, "--producer", "h200", "2026-08-20")
    assert result.returncode == 0, result.stderr
    lines = _shim_lines(result)
    assert len(lines) == 3
    assert "--verify-published 2026-08-20T03" in lines[0]
    assert "--verify-published 2026-08-20T07" in lines[1]
    assert "--from 2026-08-20T00 --to 2026-08-20T23 --dry-run" in lines[2]


def test_day_mode_on_a_clean_tree_is_one_derive_pass(shim, data_dir):
    result = _run(shim, data_dir, "--producer", "h100", "2026-08-24")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "index_panel_h100_sxm.json" in line
    assert "--from 2026-08-24T00 --to 2026-08-24T23 --dry-run" in line


def test_day_mode_future_date_refuses(shim, data_dir):
    result = _run(shim, data_dir, "--producer", "h100", "2099-01-01")
    assert result.returncode == 1
    assert "in the future" in result.stderr


def test_bad_date_refuses(shim, data_dir):
    result = _run(shim, data_dir, "h100", "2026-8-24")
    assert result.returncode == 2
    assert "bad date" in result.stderr


# ------------------------------------------------- broad lanes: --lane


def test_broad_lane_requires_the_explicit_flag(shim, data_dir):
    result = _run(shim, data_dir, "h100_broad", "2026-08-24")
    assert result.returncode == 2
    assert "not a public SKU" in result.stderr
    assert "--lane h100_broad" in result.stderr
    assert not _shim_lines(result)


def test_lane_flag_reaches_the_broad_config(shim, data_dir):
    result = _run(shim, data_dir, "--lane", "h100_broad", "2026-08-24T05")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "index_panel_h100_broad.json" in line
    assert "--observation 2026-08-24T05 --dry-run" in line


def test_lane_flag_unknown_lane_lists_configured_lanes(shim, data_dir):
    result = _run(shim, data_dir, "--lane", "h100_sxm5", "2026-08-24")
    assert result.returncode == 2
    assert "no panel lane 'h100_sxm5'" in result.stderr
    assert "h100_broad" in result.stderr


# --------------------------------------------- frozen daily escape hatch


def test_frozen_routes_to_the_daily_engine_with_notice(shim, data_dir):
    result = _run(shim, data_dir, "--frozen", "b300", "2026-08-22")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "compute_index_composite.py" in line
    assert "index_basket.json" in line
    assert "--date 2026-08-22 --dry-run" in line
    assert "frozen" in result.stdout
    assert "no longer extended" in result.stdout


def test_frozen_published_day_auto_routes_to_daily_verify(shim, data_dir):
    prefix, methodology = _lane_meta("index_basket_b200.json")
    artifact = (
        data_dir / prefix / "composites" / methodology / "2026-08-16.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    result = _run(shim, data_dir, "--frozen", "b200", "2026-08-16")
    assert result.returncode == 0, result.stderr
    (line,) = _shim_lines(result)
    assert "compute_index_composite.py" in line
    assert "--verify-published 2026-08-16" in line


def test_frozen_refuses_non_daily_skus_and_hour_stamps(shim, data_dir):
    result = _run(shim, data_dir, "--frozen", "h100", "2026-08-22")
    assert result.returncode == 2
    assert "only b300 and b200" in result.stderr

    result = _run(shim, data_dir, "--frozen", "b300", "2026-08-22T05")
    assert result.returncode == 2
    assert "day-keyed" in result.stderr
