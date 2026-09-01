# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""METHODOLOGY.md section 7.2's worked example, executed.

Runs the five-provider example through the code's own
median_stddev_composite and pins the live IQM calculation:

    index 6.6425, stability band 0.2425
    (vote p25 6.4, p75 6.8; point median diagnostic 6.6)
"""

from __future__ import annotations

from gpu_index.index.composite import median_stddev_composite

# The section 7.2 table: (provider, liveness weight, price), sd per provider.
PASSING = [
    ("A", 0.30, 6.60),
    ("B", 0.25, 6.75),
    ("C", 0.20, 6.50),
    ("D", 0.15, 7.20),
    ("E", 0.10, 5.80),
]
VOTE_SD = {"A": 0.20, "B": 0.21, "C": 0.20, "D": 0.25, "E": 0.45}
IQM_ALPHA = 0.16666


def test_section_7_2_worked_example_executes_live_iqm():
    out = median_stddev_composite(PASSING, VOTE_SD, iqm_alpha=IQM_ALPHA)
    assert out["value_usd_gpu_hr"] == 6.6425
    assert out["confidence_usd_gpu_hr"] == 0.2425
    assert out["vote_p25_usd_gpu_hr"] == 6.4
    assert out["vote_p75_usd_gpu_hr"] == 6.8
    assert out["vote_median_usd_gpu_hr"] == 6.6
    assert out["iqm_alpha"] == IQM_ALPHA
    # The diagnostic weighted mean of the same table, for orientation.
    assert out["weighted_mean_usd_gpu_hr"] == 6.6275
    assert out["sources_used_count"] == 5


def test_section_7_2_reprice_d_robustness_point():
    """Repricing the high-tail D vote leaves the central IQM unchanged."""
    repriced = [(s, w, 9.36 if s == "D" else p) for s, w, p in PASSING]
    base = median_stddev_composite(PASSING, VOTE_SD, iqm_alpha=IQM_ALPHA)
    moved = median_stddev_composite(
        repriced, VOTE_SD, iqm_alpha=IQM_ALPHA
    )
    assert moved["value_usd_gpu_hr"] == 6.6425
    assert moved["value_usd_gpu_hr"] == base["value_usd_gpu_hr"]
    assert moved["weighted_mean_usd_gpu_hr"] == 6.9515
    assert moved["weighted_mean_usd_gpu_hr"] > base["weighted_mean_usd_gpu_hr"]
