# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""CI fence for the package dependency arrows (docs/architecture.md):
common <- observatory <- index <- published, plus exactly two declared
waivers. A new cross-package import fails here until it is either
removed or documented as a third waiver in docs/architecture.md."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "gpu_index"

# (importing package, imported package) pairs; same-package is always fine.
ALLOWED_EDGES = {
    ("observatory", "common"),
    ("index", "common"),
    ("published", "common"),
    ("published", "index"),
}
# The two declared waivers (docs/architecture.md): file -> module prefixes.
WAIVERS = {
    "observatory/sources/vast.py": ("gpu_index.index.composite", "gpu_index.index.sources"),
    "index/": ("gpu_index.observatory.catalog",),
}


def _gpu_index_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names if a.name.startswith("gpu_index"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module and node.module.startswith("gpu_index"):
                yield node.module


def test_only_declared_cross_package_edges():
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        src_pkg = rel.split("/", 1)[0].removesuffix(".py")
        for module in _gpu_index_imports(ast.parse(path.read_text())):
            parts = module.split(".")
            dst_pkg = parts[1] if len(parts) > 1 else ""
            if dst_pkg in ("", src_pkg):
                continue  # root or same-package: not a cross-package edge
            if (src_pkg, dst_pkg) in ALLOWED_EDGES:
                continue
            if any(rel.startswith(f) and module.startswith(p) for f, ps in WAIVERS.items() for p in ps):
                continue
            violations.append(f"{rel}: import of {module}")
    assert not violations, (
        "Undeclared cross-package import(s) — the dependency arrows are\n"
        "common <- observatory <- index <- published with two waivers\n"
        "(docs/architecture.md). Remove the import or document a waiver:\n"
        + "\n".join(violations)
    )
