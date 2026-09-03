# Methodology change log

The methodology change log referenced by [GOVERNANCE.md](GOVERNANCE.md).
Entries record lane mints, retirements, and adopted methodology changes,
with effective dates. Every published record embeds the full parameter set
that produced it; any parameter change touching a published day mints a
new methodology_id, and prior versions stay frozen and readable under
their own keyspaces. Newest first.

## 2026-09-03

- **Hyperbolic seated on the H100-SXM and H200-SXM panels.** New lane
  mints `h100_sxm_v1_calc_v10` (17 seats, was 16) and `h200_sxm_v1_calc_v10`
  (14 seats, was 13), each adding `hyperbolic` as a label-screened SXM seat
  (`require_tokens: ["SXM"]`; Hyperbolic's own `gpuFormFactor` labels the
  rows). The seat prices under a new panel statistic, `book_median`: the
  plain median of its screened SXM book, weight 1 per row, no population
  floor, no volume weighting. The population is every row the seat's
  screens admit, including options Hyperbolic lists as disabled or without
  availability; one row prints as itself and two print as their midpoint
  (a two-row book at 3.99 and 4.92 prints 4.455). The statistic is named
  in the SKU document each lane runs from; the published `calc_params` do
  not list per-seat statistics. No other member, screen, or threshold
  changed. Measured on the seat alone over the recorded 15-minute slots
  from 2026-08-29 to 2026-09-03, the median print moves on 9% (H100) and
  13% (H200) of slot transitions with mean moves of 1.6% and 1.4%, against
  4% of transitions with mean moves of 0.7% and 0.6% under the
  lowest-eligible rule, with the same largest single move (52% H100, 23%
  H200).
- Hyperbolic is classed `marketplace` in the observatory config, with
  `first_party: true`, the same disclosure Vast.ai and Lium carry: its
  terms describe a marketplace over third-party suppliers. The class is a
  disclosure field; nothing in the calculation branches on it.
- Fallback member weights on these two lanes are now derived (uniform 1/n)
  rather than authored. The fallback vector prices the panel only until it
  latches to dynamic weighting, which both lanes did on 2026-08-28, so the
  new series differs from the old one over 2026-08-23 to 2026-08-27 and
  agrees with it from 2026-08-28 until Hyperbolic's first collection on
  2026-08-29. Measured stamp by stamp against the previous
  methodology: H100-SXM differs by 1.0% to 1.6% on average per day in that
  window, at most 3.07% (2026-08-27 12:00Z, 3.493 versus 3.386); H200-SXM
  by at most 1.23%. From 2026-08-28 onward, before Hyperbolic collected,
  the two series agree to 0.02%. The prior series stays published under its
  own keyspace.
- Effective from the production promotion timestamp recorded in each
  panel's `versions.succession` (`effective_from`) in `latest.json`.
  Prior prints under earlier methodology ids remain published, unchanged,
  under their own keyspaces. A consumer plotting across the boundary will
  see more frequent intra-day movement from the new seat. Measured on the
  published composite over every 15-minute stamp from 2026-08-29 to
  2026-09-03 against the previous methodology: the level moves by −0.22%
  (H100-SXM) and +0.25% (H200-SXM) on average, +0.34% and +0.91% over the
  last 24 hours of that window; the standard deviation of stamp-to-stamp
  changes rises from 0.21% to 0.45% (H100) and from 0.20% to 0.36% (H200);
  stamps moving more than 1% rise from 9 to 44 (H100) and from 3 to 17
  (H200), of which 34 and 13 coincide with a change in Hyperbolic's own
  print, whose book median steps between 3.19, 4.02 and 4.85 as options
  come and go; the largest single stamp-to-stamp move rises from 1.94% to
  3.29% (H100). No stamp went dark that was priced before. This is a
  property of the new methodology, not a correction to the old one.
- Hyperbolic's receipt field `provider_class` publishes as null until the
  publisher is rebuilt with the seat's disclosure table. Its `gpu_variant`
  and `vram_gb` publish as null for the same reason Vast.ai's and Lium's
  do: a statistic over several rows has no single row to project them
  from. Its price and weight are complete.

## 2026-09-01

- Change log restarted at the public launch of this repository. Earlier
  internal iterations are superseded by the published record itself:
  every published observation names its `methodology_id` and embeds the
  complete calculation parameters that produced it, so any historical
  print remains verifiable with `./reproduce` regardless of when its
  methodology version was minted. Changes from this date onward are
  recorded here.
