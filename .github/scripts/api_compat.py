#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Signature-compatibility gate: griffe's breakages, minus value-only drift.

`griffe check` exits 1 on ANY breakage it finds, with no way to scope
which kinds are fatal. One of the kinds it reports is
`ATTRIBUTE_CHANGED_VALUE` -- a module-level constant whose *value*
changed. That is not a signature break: no downstream caller stops
working because a constant's contents changed, and it is what the
package's own tests are for. A gate that fires on every edit to every
constant is a gate reviewers learn to click past, which is worse than no
gate at all.

So this wrapper runs griffe's own comparison through griffe's own API and
drops exactly that one breakage kind, keyed on the `BreakageKind` enum
member rather than on the text of the message (a text filter would be
brittle and would swallow real breakages whose message happened to
match). Ignored breakages are still printed, so nothing goes silent.

Every signature check stays fatal, including:

  * OBJECT_REMOVED            public name removed or renamed
  * OBJECT_CHANGED_KIND       name now points at a different kind of object
  * PARAMETER_REMOVED         parameter removed
  * PARAMETER_MOVED           positional parameters reordered
  * PARAMETER_CHANGED_DEFAULT default value changed
  * PARAMETER_CHANGED_KIND / _CHANGED_REQUIRED / _ADDED_REQUIRED
  * RETURN_CHANGED_TYPE, ATTRIBUTE_CHANGED_TYPE, CLASS_REMOVED_BASE

Worth knowing what this does and does not give up. Griffe's attribute
comparison is value-only: `_attribute_incompatibilities` compares
`.value` and carries a `# TODO: Support annotation breaking changes`, so
`AttributeChangedTypeBreakage` is defined but never constructed
(griffe 2.2). An attribute's *annotation* changing is therefore not
caught by griffe, before this change or after it -- the check being
narrowed here never distinguished a retyped constant from a retyped
string, because griffe cannot see the difference. It is kept in the
fatal set above so it takes effect if griffe implements it.

Usage (same shape as the `griffe check` call it replaces):

  api_compat.py gpu_index --search src --against origin/main --format github

Exit codes: 0 no fatal breakage, 1 at least one fatal breakage,
2 usage or git/loading error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from griffe import (
    BreakageKind,
    ExplanationStyle,
    find_breaking_changes,
    load,
    load_git,
)

# The one breakage kind this gate does not fail on: an attribute's VALUE.
# Every other kind griffe knows about -- ATTRIBUTE_CHANGED_TYPE included --
# is absent here and therefore fatal.
IGNORED_KINDS = frozenset({BreakageKind.ATTRIBUTE_CHANGED_VALUE})


def _repo_root() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="Package to load and check, by name.")
    parser.add_argument(
        "-s",
        "--search",
        action="append",
        default=[],
        dest="search",
        metavar="PATH",
        help="Path to search packages into (repeatable).",
    )
    parser.add_argument(
        "-a",
        "--against",
        required=True,
        metavar="REF",
        help="Older git reference (commit, branch, tag) to check against.",
    )
    parser.add_argument(
        "-f",
        "--format",
        default=ExplanationStyle.ONE_LINE.value,
        choices=[style.value for style in ExplanationStyle],
        dest="style",
        help="Output format for reported breakages.",
    )
    args = parser.parse_args(argv)
    style = ExplanationStyle(args.style)

    try:
        repo = _repo_root()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"api_compat: error: not a git repository ({error})", file=sys.stderr)
        return 2

    try:
        # The base ref, from git history; and the working tree as it stands.
        # Both loaded exactly as `griffe check` loads them.
        old_package = load_git(
            args.package,
            ref=args.against,
            repo=repo,
            search_paths=args.search,
            resolve_aliases=True,
            resolve_external=None,
        )
        new_package = load(
            args.package,
            try_relative_path=True,
            search_paths=args.search,
            resolve_aliases=True,
            resolve_external=None,
        )
    except Exception as error:  # noqa: BLE001 - any load failure is a gate error
        print(f"api_compat: error: could not load package ({error})", file=sys.stderr)
        return 2

    fatal = 0
    for breakage in find_breaking_changes(old_package, new_package):
        if breakage.kind in IGNORED_KINDS:
            print(
                "api_compat: not a signature break, ignored: "
                f"{breakage.explain(style=ExplanationStyle.ONE_LINE)}"
            )
            continue
        fatal += 1
        print(breakage.explain(style=style), file=sys.stderr)

    if fatal:
        print(
            f"api_compat: {fatal} breaking change(s) against {args.against}",
            file=sys.stderr,
        )
        return 1
    print(f"api_compat: no breaking changes against {args.against}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
