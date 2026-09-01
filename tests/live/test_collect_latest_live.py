# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Live acceptance for current collection of the latest H100 receipts."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCE = REPO_ROOT / "reproduce"
SUMMARY = re.compile(
    r"^summary: (\d+) seat\(s\): (\d+) SAME, (\d+) MOVED, "
    r"(\d+) UNREACHABLE, (\d+) SKIPPED$",
    re.MULTILINE,
)


def test_current_h100_receipts_include_at_least_one_live_comparison():
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    env.pop("GPU_INDEX_PUBLIC_BASE_URL", None)
    result = subprocess.run(
        [str(REPRODUCE), "--collect", "h100"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    report = (
        f"./reproduce --collect h100 exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.returncode == 0, report
    summary = SUMMARY.search(result.stdout)
    assert summary is not None, report
    total, same, moved, unreachable, skipped = (
        int(value) for value in summary.groups()
    )
    assert total > 0, report
    assert same + moved > 0, report
    assert same + moved + unreachable + skipped == total, report
