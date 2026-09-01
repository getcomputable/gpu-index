# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Pytest path bootstrap — prefer editable src/ layout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
