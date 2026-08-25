# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Field-path diffs between two parsed JSON documents.

One home for the --verify-published MISMATCH explainer shared by the
daily CLI (scripts/compute_index_composite.py) and the hourly panel CLI
(scripts/compute_panel_index.py): a byte mismatch is only actionable when
the operator can see WHICH fields moved, and two forked walkers could
disagree on exactly that. Pure function, zero I/O.
"""

from __future__ import annotations

from typing import Any, List


def field_diffs(
    published: Any, recomputed: Any, path: str = "", limit: int = 40
) -> List[str]:
    """Field-path differences between two parsed JSON documents, e.g.
    ``sources[3].chosen.usd_per_gpu_hr: published 6.0 vs recomputed 5.0``."""
    diffs: List[str] = []

    def walk(a, b, at):
        if len(diffs) > limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                sub = f"{at}.{key}" if at else str(key)
                if key not in a:
                    diffs.append(f"{sub}: only in recomputed ({b[key]!r})")
                elif key not in b:
                    diffs.append(f"{sub}: only in published ({a[key]!r})")
                else:
                    walk(a[key], b[key], sub)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(
                    f"{at}: published has {len(a)} entries vs "
                    f"recomputed {len(b)}"
                )
            for i, (ai, bi) in enumerate(zip(a, b)):
                walk(ai, bi, f"{at}[{i}]")
            return
        if a != b or type(a) is not type(b):
            diffs.append(f"{at}: published {a!r} vs recomputed {b!r}")

    walk(published, recomputed, path)
    if len(diffs) > limit:
        return diffs[:limit] + [f"... and more (stopped at {limit} diffs)"]
    return diffs
