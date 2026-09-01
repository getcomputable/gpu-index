# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Canonical-serializer determinism guards.

The record is BYTES: every immutable artifact and pointer serializes
through gpu_index.common.store.snapshot_bytes — json.dumps with
sort_keys=True, indent=2, a trailing newline, UTF-8. --verify-published
and the append-only guard both compare bytes, so the serializer must be a
pure function of the logical document:

  - sorted keys => dict insertion order can never change bytes;
  - ensure_ascii (the json default) + repr-based float formatting =>
    no locale or platform dependence (CPython's float repr is the
    shortest round-trip form, pinned since 3.1);
  - zero clock reads: timestamps only appear where a caller put them.

Audit note (2026-08-25): the one theoretical hole found is allow_nan
(json's default), under which float('nan') would serialize as the
non-JSON token NaN — but non-finite values already raise upstream in the
calculators (median_stddev_composite / compute_dynamic_weights fail
closed), so no behavior was changed. No real nondeterminism was found.
"""

from __future__ import annotations

import json
from pathlib import Path

from gpu_index.common.store import snapshot_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_ARTIFACT_PATH = (
    REPO_ROOT / "tests" / "golden" / "b200_annex_a2_v0_3_calc_v5_2026-08-16.json"
)
FIXTURE_SNAPSHOT_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "b200_snapshot_2026-08-16_slot22-20260816T221613Z-1fdf.json"
)


def _reinserted_reversed(doc):
    """The same logical document rebuilt with every dict's keys inserted in
    REVERSED order, recursively — worst-case insertion-order shuffle."""
    if isinstance(doc, dict):
        return {k: _reinserted_reversed(doc[k]) for k in reversed(list(doc))}
    if isinstance(doc, list):
        return [_reinserted_reversed(v) for v in doc]
    return doc


def test_insertion_order_never_changes_bytes():
    # A hand-built artifact-shaped dict assembled in two different orders.
    a = {
        "date": "2026-08-16",
        "index": {"value_usd_gpu_hr": 6.74, "sources_used_count": 8},
        "sources": [{"source_id": "nebius", "weight": 0.12}],
        "basket_dark": False,
    }
    b = {
        "basket_dark": False,
        "sources": [{"weight": 0.12, "source_id": "nebius"}],
        "index": {"sources_used_count": 8, "value_usd_gpu_hr": 6.74},
        "date": "2026-08-16",
    }
    assert snapshot_bytes(a) == snapshot_bytes(b)


def test_real_artifact_insertion_order_never_changes_bytes():
    # The full golden artifact, every dict re-inserted in reversed key
    # order, must serialize to the exact committed bytes.
    raw = GOLDEN_ARTIFACT_PATH.read_bytes()
    doc = json.loads(raw)
    assert snapshot_bytes(_reinserted_reversed(doc)) == raw


def test_serialize_parse_serialize_round_trips_byte_identically():
    # The committed golden artifact AND the raw snapshot fixture are both
    # stored in canonical form: parse -> re-serialize is the identity.
    for path in (GOLDEN_ARTIFACT_PATH, FIXTURE_SNAPSHOT_PATH):
        raw = path.read_bytes()
        assert snapshot_bytes(json.loads(raw)) == raw, path.name


def test_float_formatting_is_stable():
    # 6.74 and 6.740000 are the SAME float; the serializer must emit one
    # spelling (repr shortest form), never a padded variant.
    assert snapshot_bytes({"v": 6.74}) == snapshot_bytes({"v": 6.740000})
    assert b'"v": 6.74\n' in snapshot_bytes({"v": 6.74})
    # Parse -> serialize preserves the spelling (floats round-trip by repr).
    assert snapshot_bytes(json.loads('{"v": 6.74}')) == snapshot_bytes(
        {"v": 6.74}
    )
    # Non-terminating binary fractions have ONE canonical spelling too.
    assert b'"v": 0.30000000000000004\n' in snapshot_bytes({"v": 0.1 + 0.2})
    # Integer-valued floats keep their float spelling: 1.0 is not 1.
    assert b'"v": 1.0\n' in snapshot_bytes({"v": 1.0})
    assert snapshot_bytes({"v": 1.0}) != snapshot_bytes({"v": 1})
