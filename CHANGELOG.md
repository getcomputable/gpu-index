# Methodology change log

The methodology change log referenced by [GOVERNANCE.md](GOVERNANCE.md).
Entries record lane mints, retirements, and adopted methodology changes,
with effective dates. Every published record embeds the full parameter set
that produced it; any parameter change touching a published day mints a
new methodology_id, and prior versions stay frozen and readable under
their own keyspaces. Newest first.

## 2026-09-14

New methodology versions on every public lane: `annex_a_v0_2_calc_v17`,
`annex_a2_v0_3_calc_v17`, `h100_sxm_v1_calc_v16`, `h200_sxm_v1_calc_v16`. Each
provider's vote is now a one-hour exponentially weighted average of its own
accepted prints, a provider whose print is rejected by the outlier fence keeps
voting its last accepted value as a carried vote, and every receipt discloses the
price it cast (`smoothed_vote_usd`). `./reproduce` verifies these versions from the
disclosed cast prices.

## 2026-09-08

`./reproduce` now re-derives the as-published history end to end under the version
live at each stamp, using each version's own lookback history. `--receipts` offers
the fast per-observation check, and `--version` selects a single version.

Published history now serves the version that was live at each observation. A
methodology version's keyspace holds that version's full re-derivation from its
first observation, including rows before its effective time, for verification;
`./reproduce --version` labels those rows back-calculated.

## 2026-09-03

Hyperbolic seated on the H100-SXM and H200-SXM panels, effective
2026-09-03T18:59:44Z (`h100_sxm_v1_calc_v10`, `h200_sxm_v1_calc_v10`). The
seat prices as the plain median of its asks (`book_median`).

## 2026-09-01

Attendance weighting armed at 0.5 on every lane, effective
2026-09-01T00:13:39Z (`h100_sxm_v1_calc_v8`, `h200_sxm_v1_calc_v8`,
`annex_a_v0_2_calc_v14`, `annex_a2_v0_3_calc_v14`).

Change log restarted at the public launch of this repository. Earlier
versions remain verifiable from the published record, which names the
`methodology_id` and embeds the parameters of every observation.
