# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""The COM-1570 carried-vote CROSS-REPO WIRE-SHAPE PIN.

``tests/fixtures/cross_repo/carried-vote-pin.observation.json`` is the verbatim
production artifact ``h100_sxm_v1_calc_v16`` 2026-09-14T18:30Z — the first
smoothing-armed generation, whose ``vast`` seat is a live fence-reject carry
(fresh print 5.8006 rejected by the sigma fence; the engine cast the smoothed
booked vote 4.60857 carried from 17:45) — exactly as the index publisher's
projector published it.

The load-bearing wire shape is FLAT: the publisher flattens the engine's
``carried_vote`` block to the receipt-level ``carried_vote_from`` key, with
``carry_basis`` and the cast (``smoothed_vote_usd``) beside it on a
``status: ok`` + ``filter_verdict: rejected`` receipt. The publisher's own
suite pins the same projection over the same bytes; this suite feeds them to
the reproducer and requires the published value EXACTLY. Whichever side
reshapes the wire goes red in its own suite instead of against the public
reproduce guarantee — and a nested ``carried_vote`` object is pinned REFUSED,
not silently admitted.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gpu_index.published.verify import (
    VERDICT_MATCH,
    VERDICT_MISMATCH,
    PublishedRecordError,
    _carried_vote_disclosed,
    recompute_observation,
)

PIN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cross_repo"
    / "carried-vote-pin.observation.json"
)

PUBLISHED_VALUE = 3.52045
PUBLISHED_BAND = 0.54375
VAST_CAST = 4.60857


def _pinned() -> dict:
    return json.loads(PIN.read_bytes())


def _vast(observation: dict) -> dict:
    (receipt,) = [
        receipt
        for receipt in observation["receipts"]
        if receipt["source_id"] == "vast"
    ]
    return receipt


def test_pin_bytes_carry_the_flat_fence_reject_disclosure():
    """The wire-shape pin itself: the projected receipt discloses the FLAT
    keys — carried_vote_from + carry_basis + the cast — beside the real
    rejected print, and no nested object."""
    vast = _vast(_pinned())
    assert vast["status"] == "ok"
    assert vast["upstream_status"] == "ok"
    assert vast["filter_verdict"] == "rejected"
    assert vast["price"] == 5.8006
    assert vast["carried_vote_from"] == "2026-09-14T17:45:00.000Z"
    assert vast["carry_basis"] == "no_price"
    assert vast["smoothed_vote_usd"] == VAST_CAST
    assert "carried_vote" not in vast
    assert _carried_vote_disclosed(vast)


def test_pin_reproduces_the_published_value_exactly():
    """Happy path, no synthetic edits: the smoothing-armed ballot over the
    disclosed casts lands EXACTLY on the engine's published value and band.
    A nudged cast misses, so the match is the vote itself, not a plateau."""
    check = recompute_observation(_pinned())
    assert check.verdict == VERDICT_MATCH
    assert check.recomputed_value == PUBLISHED_VALUE
    assert check.recomputed_band == PUBLISHED_BAND

    nudged = _pinned()
    _vast(nudged)["smoothed_vote_usd"] = VAST_CAST + 1.5
    assert recompute_observation(nudged).verdict == VERDICT_MISMATCH


def test_dropping_the_flat_disclosure_drops_the_seat_and_reds_the_reproduce():
    """The contrapositive this pin exists for: a projector that stops
    flattening the disclosure makes the row read as a plain sigma reject —
    the seat's vote drops and the recompute walks away from the published
    value. Loud red, never a silently different ballot admitted."""
    stripped = _pinned()
    vast = _vast(stripped)
    del vast["carried_vote_from"]
    del vast["carry_basis"]
    del vast["smoothed_vote_usd"]
    check = recompute_observation(stripped)
    assert check.verdict == VERDICT_MISMATCH


def test_nested_object_shape_reads_as_absent():
    """A nested carried_vote object is not the wire shape and is REFUSED as
    a disclosure: the presence fence reads it absent, the seat stops voting,
    and the reproduce reds — pinning that publisher and reproducer cannot
    silently diverge on shape."""
    nested = _pinned()
    vast = _vast(nested)
    vast["carried_vote"] = {
        "from": vast.pop("carried_vote_from"),
        "carry_basis": vast.pop("carry_basis"),
        "age_minutes": 45,
    }
    del vast["smoothed_vote_usd"]
    assert not _carried_vote_disclosed(vast)
    check = recompute_observation(nested)
    assert check.verdict == VERDICT_MISMATCH


def test_armed_pin_refuses_without_a_cast():
    """Admission proof: the flat disclosure ADMITS the vast vote, so a
    missing cast price refuses loudly naming the seat — never a silent
    recompute without it."""
    torn = _pinned()
    del _vast(torn)["smoothed_vote_usd"]
    with pytest.raises(PublishedRecordError, match="vast") as caught:
        recompute_observation(torn)
    assert "smoothed_vote_usd" in str(caught.value)


def test_dressing_on_the_flat_key_reads_as_absent():
    """Number/object/array dressing on carried_vote_from is not a
    disclosure: the fence reads it absent and the seat drops back onto the
    plain law, which fails closed on the rejected verdict (mismatch, never
    admission)."""
    for dressing in (45, {"from": "2026-09-14T17:45:00.000Z"}, ["17:45"], ""):
        dressed = copy.deepcopy(_pinned())
        vast = _vast(dressed)
        vast["carried_vote_from"] = dressing
        del vast["smoothed_vote_usd"]
        assert not _carried_vote_disclosed(vast)
        assert recompute_observation(dressed).verdict == VERDICT_MISMATCH
