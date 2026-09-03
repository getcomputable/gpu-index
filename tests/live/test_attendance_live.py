# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""LIVE seam: attendance weighting as the published record actually runs it.

`./reproduce` recomputes each observation's index from its own receipts,
consuming the published weights AS PUBLISHED -- so it verifies the
aggregation and says nothing at all about how current weights were
allocated. This file closes that gap from the other side: every input to
the current allocation is itself disclosed (`liveness_score` = Q_i,
`attendance_factor` = A_i, and `calc_params.liveness` = gamma / eta /
w_min / w_max), so that allocation can be RE-DERIVED from the public
record and matched exactly against `gpu_index.index.weights.allocate_weights`.

A collection-failure carry is deliberately outside that current vector:
it re-casts its prior accepted vote, including the booked historical
weight. The raw-history reproduction checks those rows. A provider-side
no-price carry is inside the current vector and receives a current,
attendance-faded weight; `carry_basis` distinguishes the two.

That makes this the acceptance check for the attendance port: it fails
if the ported allocator and the producer disagree by one part in 1e-6 on
any seat of any observation, on lanes where attendance is currently
biting (a seat at A_i = 0.04 is pinned far below the 2.5% floor, and a
seat past the hard cutoff carries no weight row at all).

Like its sibling in this directory it refuses fixtures and refuses to
record one: a self-recorded golden would freeze whatever this engine
currently believes and re-assert it forever, which is the bug, not the
check. Marked `live`, deselected by default, run with `pytest -m live`.
"""

from __future__ import annotations

import datetime
import json
import os

import httpx
import pytest

from gpu_index.index.panel import CARRIED_STATUS, CARRY_BASIS_NO_PRICE
from gpu_index.index.weights import allocate_weights
from gpu_index.published.artifacts import day_key

pytestmark = pytest.mark.live

DEFAULT_PUBLIC_BASE_URL = "https://data.getcomputable.com"
PUBLIC_BASE_URL = (
    os.environ.get("GPU_INDEX_PUBLIC_BASE_URL") or DEFAULT_PUBLIC_BASE_URL
)
PUBLIC_SKUS = ("H100", "H200", "B300", "B200")
LOOKBACK_DAYS = 7

# The published receipt fields the attendance disclosure adds.
ATTENDANCE_FIELDS = ("attendance_factor", "no_price_streak",
                     "no_price_excluded")


def _get_json(key: str):
    response = httpx.get(
        f"{PUBLIC_BASE_URL.rstrip('/')}/{key}", timeout=30.0,
        follow_redirects=True,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return json.loads(response.content)


@pytest.fixture(scope="module")
def live_days():
    """Newest published day per SKU as {sku: day document}."""
    latest = _get_json("latest.json")
    versions = {
        entry["sku"]: entry["current_version"]
        for entry in latest["data"]["versions"]
    }
    today = datetime.datetime.now(datetime.timezone.utc).date()
    out = {}
    for sku in PUBLIC_SKUS:
        for back in range(0, LOOKBACK_DAYS + 1):
            date = (today - datetime.timedelta(days=back)).isoformat()
            document = _get_json(
                day_key(date, sku=sku, version=versions[sku])
            )
            if document is not None and document["data"]["observations"]:
                out[sku] = document
                break
        else:
            pytest.fail(
                f"the public front {PUBLIC_BASE_URL} served no {sku} "
                f"observation day file in the last {LOOKBACK_DAYS} days"
            )
    return out


def _attendance_observations(document):
    """The day's observations whose params carry the attendance knobs."""
    return [
        observation
        for observation in document["data"]["observations"]
        if "attendance_eta" in (observation["calc_params"].get("liveness") or {})
    ]


@pytest.mark.parametrize("sku", PUBLIC_SKUS)
def test_attendance_disclosure_rides_every_receipt(sku, live_days):
    """The three per-source fields the methodology promises, on every
    seat of every observation of a minted lane -- present, typed, and in
    range. Their absence is the failure this catches: a disclosure that
    quietly stops publishing looks exactly like a healthy panel."""
    observations = _attendance_observations(live_days[sku])
    if not observations:
        pytest.skip(f"{sku}: the serving generation is not attendance-minted")
    for observation in observations:
        for receipt in observation["receipts"]:
            where = f"{sku} {observation['observed_at']} {receipt['source_id']}"
            for field in ATTENDANCE_FIELDS:
                assert field in receipt, f"{where}: missing {field}"
            factor = receipt["attendance_factor"]
            assert isinstance(factor, (int, float)) and not isinstance(
                factor, bool
            ), f"{where}: attendance_factor {factor!r} is not a number"
            assert 0.0 <= factor <= 1.0, f"{where}: A_i {factor} outside [0, 1]"
            streak = receipt["no_price_streak"]
            assert isinstance(streak, int) and not isinstance(streak, bool)
            assert streak >= 0, f"{where}: negative no_price_streak {streak}"
            assert isinstance(receipt["no_price_excluded"], bool)


@pytest.mark.parametrize("sku", PUBLIC_SKUS)
def test_published_weights_re_derive_from_the_published_inputs(sku, live_days):
    """The port's acceptance check: softmax(gamma*Q + eta*ln A) with
    attendance-scaled ceilings and collapsing floors, run over the
    published Q and A under the published liveness params, must
    reproduce every published weight EXACTLY at the published 6dp.

    The domain is the producer's CURRENT allocation set: ordinary rows
    plus provider-side no-price carries. A collection-failure carry
    instead re-casts its booked historical weight; the full-history live
    test re-derives that row from the prior accepted observation. A hard
    cutoff removes the seat from both sets."""
    observations = _attendance_observations(live_days[sku])
    if not observations:
        pytest.skip(f"{sku}: the serving generation is not attendance-minted")
    checked = 0
    for observation in observations:
        liveness = observation["calc_params"]["liveness"]
        domain = {
            receipt["source_id"]: receipt
            for receipt in observation["receipts"]
            if receipt.get("weight") is not None
            and not (
                receipt.get("upstream_status") == CARRIED_STATUS
                and receipt.get("carry_basis") != CARRY_BASIS_NO_PRICE
            )
        }
        if not domain:
            continue  # a dark observation carries no vector to re-derive
        scores = {
            sid: (receipt["liveness_score"] or 0.0)
            for sid, receipt in domain.items()
        }
        factors = {
            sid: receipt["attendance_factor"]
            for sid, receipt in domain.items()
        }
        weights, _ = allocate_weights(
            scores,
            gamma=float(liveness["gamma"]),
            weight_min=float(liveness["weight_min"]),
            weight_max=float(liveness["weight_max"]),
            source_caps={},
            attendance_factors=factors,
            attendance_eta=float(liveness["attendance_eta"]),
        )
        for sid, receipt in domain.items():
            assert round(weights[sid], 6) == receipt["weight"], (
                f"{sku} {observation['observed_at']} {sid}: published weight "
                f"{receipt['weight']} but re-derived {round(weights[sid], 6)} "
                f"from Q={scores[sid]} A={factors[sid]} under "
                f"gamma={liveness['gamma']} eta={liveness['attendance_eta']}"
            )
            checked += 1
    assert checked, f"{sku}: no weight rows were re-derived"


@pytest.mark.parametrize("sku", PUBLIC_SKUS)
def test_an_excluded_seat_carries_no_weight_or_vote(sku, live_days):
    """The hard cutoff is a pre-advance verdict: the seat has no weight
    or vote while excluded. Its first fresh accepted recovery price may
    remain visible as receipt evidence; that print advances state and
    re-admits the seat at the next observation."""
    observations = _attendance_observations(live_days[sku])
    if not observations:
        pytest.skip(f"{sku}: the serving generation is not attendance-minted")
    for observation in observations:
        for receipt in observation["receipts"]:
            if not receipt["no_price_excluded"]:
                continue
            where = f"{sku} {observation['observed_at']} {receipt['source_id']}"
            assert receipt.get("weight") is None, (
                f"{where}: excluded but published a weight row "
                f"{receipt.get('weight')}"
            )
            assert receipt.get("sd") is None, (
                f"{where}: excluded but published a vote dispersion "
                f"{receipt.get('sd')}"
            )
            if receipt.get("price") is None:
                continue
            assert receipt.get("upstream_status") == "ok", (
                f"{where}: excluded price is not a fresh upstream print"
            )
            assert receipt.get("filter_verdict") == "accepted", (
                f"{where}: excluded recovery price was not accepted"
            )
            assert receipt.get("last_seen") == observation["observed_at"], (
                f"{where}: excluded price is stale receipt evidence from "
                f"{receipt.get('last_seen')}"
            )
