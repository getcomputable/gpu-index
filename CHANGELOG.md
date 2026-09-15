# Methodology change log

The methodology change log referenced by [GOVERNANCE.md](GOVERNANCE.md).
Entries record lane mints, retirements, and adopted methodology changes,
with effective dates. Every published record embeds the full parameter set
that produced it; any parameter change touching a published day mints a
new methodology_id, and prior versions stay frozen and readable under
their own keyspaces. Newest first.

## 2026-09-14

EWMA vote pre-smoothing (half-life 1 hour) and fence-reject carry armed on
every public lane (`annex_a_v0_2_calc_v17`, `annex_a2_v0_3_calc_v17`,
`h100_sxm_v1_calc_v16`, `h200_sxm_v1_calc_v16`), effective at the production
promotion recorded in latest.json's `versions.succession`. Each voting
provider's cast price is now the time-based EWMA of its own accepted prints
(`calc_params.pre_smoothing_half_life_hours`); an outlier-rejected print's
provider re-casts its booked smoothed vote as a carried vote
(`calc_params.liveness.fence_reject_carry`). Every voting receipt disclosed its
exact cast price as `smoothed_vote_usd`; fence-reject carried votes carry a
`carried_vote_from` marker beside the unchanged rejected print. Carried votes never
satisfy the minimum passing panel.

`./reproduce` prices each voting seat at its disclosed cast price on these
generations — it never reruns the smoothing state — and refuses loudly when a
voting receipt omits the disclosure, never falling back to the raw print.
Pre-smoothing versions re-derive byte-identically under the frozen raw-vote
rule. The local producer replay (`--producer`, `--lane`) refuses
smoothing-armed lanes outright: this mirror carries no EWMA vote state, and
pricing raw votes under a smoothed methodology_id would be silently wrong.
Provider-statistic parameters (the population floors of section 6.2 and their
`calc.statistic_params` overrides) act upstream of the recorded prints — they
shape each provider's chosen print before it is published — and are outside
the scope of reproduction from the published record, which starts from the
disclosed prints.

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
