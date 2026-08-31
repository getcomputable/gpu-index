# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Auto-discovering collector registry.

Each source lives in its own module (gpu_index/observatory/sources/<source_id>.py)
defining SOURCE_ID (== its module name) and collect(timeout=..., options=None).
Discovery walks the package at import time, so ADDING A SOURCE IS ADDING ONE
FILE — no shared registry file to edit, no merge conflicts between parallel
source lanes.

The strictness here is deliberate: a module that forgot SOURCE_ID or
collect, or whose SOURCE_ID drifted from its filename, fails the WHOLE
import loudly rather than silently dropping out of the registry — a
collector that vanishes from COLLECTORS would record as 'unimplemented'
forever and nobody would notice the hole until the history was already
lost. The config file remains the operational contract: only sources listed
in config/raw_observatory.json are ever invoked; only implemented ones may
succeed.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Dict

COLLECTORS: Dict[str, Callable[..., Dict[str, Any]]] = {}

for _mod_info in pkgutil.iter_modules(__path__):
    if _mod_info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__name__}.{_mod_info.name}")
    _sid = getattr(_mod, "SOURCE_ID", None)
    _fn = getattr(_mod, "collect", None)
    if not isinstance(_sid, str) or not callable(_fn):
        raise RuntimeError(
            f"observatory source module {_mod_info.name!r} must define "
            "SOURCE_ID (str) and collect(timeout=..., options=None)"
        )
    if _sid != _mod_info.name:
        raise RuntimeError(
            f"observatory source module {_mod_info.name!r} declares "
            f"SOURCE_ID {_sid!r} — the two must match so greps stay honest"
        )
    if _sid in COLLECTORS:
        raise RuntimeError(f"duplicate observatory SOURCE_ID {_sid!r}")
    COLLECTORS[_sid] = _fn
