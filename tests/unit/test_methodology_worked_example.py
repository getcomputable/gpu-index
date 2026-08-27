# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""METHODOLOGY.md section 7.2's worked example, executed.

Runs the five-provider example through the code's own
median_stddev_composite and pins what the CODE actually produces under
its exact-boundary straddle-averaging quantile convention:

    index 6.64, stability band 0.11 (vote p25 6.54, p75 6.75)

METHODOLOGY.md's printed walkthrough says 6.62 / 0.13 (and 6.66 for the
repriced-E case, where the code produces 6.68): its prose accumulates to
the vote AT cumulative weight 1.5 instead of averaging the exact-boundary
straddle. The two therefore disagree, and the implementation is what the
index publishes -- this test pins the implementation.
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
VOTE_SD = {"A": 0.06, "B": 0.05, "C": 0.12, "D": 0.25, "E": 0.45}


def test_section_7_2_worked_example_executes_to_6_64_band_0_11():
    out = median_stddev_composite(PASSING, VOTE_SD)
    assert out["value_usd_gpu_hr"] == 6.64
    assert out["confidence_usd_gpu_hr"] == 0.11
    assert out["vote_p25_usd_gpu_hr"] == 6.54
    assert out["vote_p75_usd_gpu_hr"] == 6.75
    # The diagnostic weighted mean of the same table, for orientation.
    assert out["weighted_mean_usd_gpu_hr"] == 6.6275
    assert out["sources_used_count"] == 5


def test_section_7_2_reprice_e_robustness_point():
    """Section 7.2's second act: reprice E (cheapest, most volatile)
    up 30% to 7.54. The median of votes barely moves while the weighted
    average jumps — the outlier-resistance the section demonstrates."""
    repriced = [(s, w, 7.54 if s == "E" else p) for s, w, p in PASSING]
    base = median_stddev_composite(PASSING, VOTE_SD)
    moved = median_stddev_composite(repriced, VOTE_SD)
    assert moved["value_usd_gpu_hr"] == 6.68  # the code's number (doc: 6.66)
    median_move = moved["value_usd_gpu_hr"] - base["value_usd_gpu_hr"]
    mean_move = (
        moved["weighted_mean_usd_gpu_hr"] - base["weighted_mean_usd_gpu_hr"]
    )
    assert 0 < median_move < mean_move
