# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Live H100/v7 public receipt pin, distinct from producer fixtures.

Fetched 2026-09-19 from:
https://data.getcomputable.com/H100/v7/observations/2026/09/16.json
The envelope was digest-verified before extracting its 00:00 observation:
7e44977acf4bcce3b9587e8e2d256ad21f320e4237d28291d2fc8a8044d16fc0.
The captured row predates population-credit disclosure; never synthesize it
into the live pin. Full-history credit restoration has separate replay tests.
"""
import json
from pathlib import Path

from gpu_index.published.full import _first_divergence, public_weight_print
from gpu_index.published.verify import recompute_observation

PIN = Path(__file__).resolve().parents[1] / "fixtures/cross_repo/h100-v16-public.observation.json"


def test_live_public_h100_pin_and_full_mode_diagnostic():
    row = json.loads(PIN.read_text())
    assert row["methodology_id"] == "h100_sxm_v1_calc_v16"
    assert row["observed_at"] == "2026-09-16T00:00:00.000Z"
    check = recompute_observation(row)
    assert check.verdict == "match"
    assert (check.recomputed_value, check.recomputed_band) == (3.476907, 0.555356)
    weights = {r["source_id"]: r["weight"] for r in row["receipts"]}
    sources = {r["source_id"]: {"attendance_factor": r["attendance_factor"],
                              "Q": r["liveness_score"]} for r in row["receipts"]}
    for receipt in row["receipts"]:
        assert "population_scale" not in receipt
        assert "credit" not in public_weight_print(receipt, observed_at=row["observed_at"])
    weights["vast"] = 0.047940
    divergence = _first_divergence(
        row["receipts"], {"sources": sources}, weights,
        derived_value=3.478314, published_value=row["value_usd_gpu_hr"],
        derived_band=0.578014, published_band=row["stability_band_usd_gpu_hr"],
    )
    assert (divergence.quantity, divergence.source_id) == ("weight", "vast")
    assert (divergence.derived, divergence.published) == (0.047940, 0.042571)
