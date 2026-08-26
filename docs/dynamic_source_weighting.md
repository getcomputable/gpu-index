# Dynamic Source Weighting

**Methodology for the index-basket constituent weights, as minted for
`annex_a_v0_2_calc_v6` (B300) and `annex_a2_v0_3_calc_v5` (B200); the
calc_v5 / a2 calc_v4 predecessors froze at those mints
(caps removed, count-based quorum) after publishing fallback-mode days
only.**

Every design rule below (R-*) is a deliberate, recorded choice pinned at
mint time. Code:
`src/gpu_index/index/weights.py`, integrated in
`src/gpu_index/index/composite.py::compute_day`
and the composite CLI. See also METHODOLOGY.md section 8.

---

## 1. The schema

For the eligible sources on day *t*, the index prices as before (the median
of standard-deviation votes; the weighted mean rides as a diagnostic) but
each source's weight is derived, not configured:

1. **Features** — for lookbacks Δ ∈ D: the source's excess movement
   x = r_i^Δ − r_{−i}^Δ, where r_i is the source's own log return and r_{−i}
   the leave-one-out basket's.
2. **Targets** — for forward horizons h ∈ H: y = the leave-one-out basket's
   forward log return over [τ, τ+h].
3. **Regression** — per (source, h): one exponentially weighted ridge
   (half-life T_half, window L, penalty λ) of y on X, fitted at day t.
4. **Score** — q = max(0, weighted **in-sample** R² of that fit); Q = mean
   over H (defined only when every horizon's q is).
5. **Allocation** — s = softmax(γ·Q); w = w_min + (1 − N·w_min)·s; the
   global w_max applied by iterative cap-and-redistribute (per-source risk
   caps were removed at the calc_v6 mint; the mechanism
   remains in the allocator, unused).

## 2. Parameters (pinned in `calc.dynamic_weights`, embedded in every artifact)

| Param | Value | Note |
|---|---|---|
| D (lookbacks) | 6h, 1d, 2d | HOURS on the capture slot grid (R-slots); 1h joins when capture walks to hourly snapshots |
| H (forwards) | 6h, 1d, 2d | same grid |
| `slot_hours_utc` | 4, 10, 16, 22 | the grid itself, embedded — a cadence change is a mint |
| L (`history_days`) | 90d | |
| T_half (`half_life_days`) | 30d | decay runs in hours (× 24) toward the cutoff |
| λ (`ridge_lambda`) | 1.0 | pinned PRIOR on weighted z-scored features; the counterweight to in-sample optimism |
| γ (`gamma`) | 4.0 | pinned PRIOR; maps plausible Q onto ~[w_min, w_max] |
| `weight_min` | 2.5% | |
| `weight_max` | 30% | |
| `min_train_samples` | 10 | THE sample gate (R-insample): ~2.5 days of slots |
| `target_variance_floor` | 1e-12 | weighted var(y) below it ⇒ q undefined (0/0 must never mint a score) |
| `switch_min_eligible` | 5 | R-quorum-v2: the switch needs this many eligible-AND-defined sources (= the capture claim floor) |
| `max_abs_log_return` | 0.5 | R-winsor's per-return clamp, far above any genuine repricing captured |
| `source_weight_caps` | — (removed, calc_v6) | no source-level manual intervention; the allocator's cap mechanism remains in code, unused |

λ and γ are priors, not validated values. Retuning either is a mint; the
scheduled revisit is ≥ 60 daily prints.

## 3. Rulings (each a deliberate, recorded choice)

- **R-slots** — the estimator runs on the
  capture SLOT grid: timestamps are integer hours (day_ordinal × 24 +
  slot_hour), the 6h cadence yields four sample anchors per day, and the
  horizon sets are {6h, 1d, 2d} in both directions. 1h was ruled IN for the
  future — it becomes expressible when capture itself walks to hourly
  snapshots (a cadence renegotiation; the validator requires every horizon
  to be a multiple of the uniform slot spacing so an inexpressible horizon
  refuses at load rather than sitting silently inert). Each day's artifact
  pins the just-closed prior day's per-slot trusted prints
  (`weight_calc.slot_prints`) so replay stays a pure function of published
  artifacts; slot prints resolve with the SAME machinery as the daily print
  (R1/statistic, FX at the day's ECB record, filter_observation trust
  rules, manual_exclusions) but are NOT adjudicated by the daily σ-fence —
  R-winsor is the weight lane's bound.
- **R-insample** — q scores the single EW ridge
  fit **in-sample**; no out-of-sample evaluation points are required. This
  supersedes an earlier walk-forward OOS protocol
  (retired before any day published). Recorded trade:
  in-sample R² is optimistic — under ridge with an unpenalized intercept it
  is mathematically ≥ 0 (the intercept-only fit is always feasible), so
  max(0,·) guards only float dust and the counterweight to noise-minted
  scores is the shrinkage itself (λ=1.0 on z-scored features) plus the
  softmax/cap fences. A score defines at `min_train_samples` realized
  samples — ~2.5 days of slots.
- **R-terminology** — "confidence interval"
  language is retired from code identifiers and docs in favor of "standard
  deviation" (the vote band IS the floored trailing-window σ; the published
  aggregate dispersion remains the 25th/75th weighted-vote-percentile
  distance). WIRE FORMAT IS PINNED: the artifact keys `conf_usd_gpu_hr` /
  `confidence_usd_gpu_hr` and the `composite_statistic` value
  `median_ci_votes` are frozen-series bytes read by downstream consumers —
  renaming them is a separate cross-consumer decision.
- **R-native** — source returns r_i run on the RECORDED-currency print (the
  same `filter_observation` value the σ-filter uses — one trust rule, one
  home); the LOO basket, r_{−i}, and all targets stay in USD. A return whose
  endpoints differ in currency is undefined, never FX-spliced. (An
  FX move must never masquerade as a source price move.)
- **R-series** — the weight series holds every real TRUSTED print: accepted
  and held-out both enter (the filter-window membership rule); untrusted-
  currency prints never enter; manual exclusions are gaps. No carry-forward
  anywhere: a return needs real prints at both exact endpoints.
- **R-fixed** — LOO returns hold weights AND composition fixed at the sample
  time τ: the pinned vector of τ's day at both endpoints, over sources
  printing at both endpoints. **Divergence from the schema-literal
  P_{−i,t}**: weight drift and composition churn must never masquerade as
  basket movement.
- **R-cutoff** — every sample endpoint (features and target alike) is
  realized by the prior day's LAST capture slot (yesterday 22:00Z). Nothing
  captured on day t can move day t's own weights — the anti-manipulation
  property; scores are announceable before the day's first snapshot exists.
- **R-winsor** — every log return entering a feature or LOO leg is clamped
  at ±`max_abs_log_return` (0.5 ≈ a ±65% move, far above the largest
  genuine repricing ever captured, ~10.9%): the series deliberately admits
  held-out prints, so without a bound one absurd trusted print the σ-fence
  correctly held OUT of the index would poison every rival's scoring
  windows at unbounded magnitude.
- **R-quorum-v2** (supersedes R-quorum) — the
  permanent switch fires on the first day at least `switch_min_eligible`
  sources are BOTH eligible today AND have a defined Q: enough scored
  providers to carry an index. A sparse or late source (vast, at the time
  of the ruling) no longer holds the switch; it scores 0 from the switch
  day, published as an auditable `Q: null`. Recorded trade: the original
  all-recently-printed rule also prevented a one-day print suppression
  from firing the switch before the suppressed source was ever scored —
  deliberately given up; the eligible-count leg and R-cutoff remain.
- **R-undefined / R-audit-null** — q exists only past the sample gate and
  the variance floor; a non-finite R² (numerical pathology) publishes as an
  auditable `q: null`, never laundered into a plausible q=0. Q exists only
  when every horizon's q does; pre-switch an undefined Q keeps the lane in
  fallback, post-switch it scores 0. q/Q are stored at 9dp (weights at 6dp)
  so the softmax→floor→cap chain is recomputable from the artifact alone.
- **R-fallback** — the series starts in `fallback` mode: the opening
  config weights RESTRICTED to the day's eligible sources, deliberately
  unnormalized (every consumer is scale-invariant — the composite
  renormalizes over passers exactly as the prior series did — so
  fallback-mode index math is byte-identical to the frozen fixed-weight
  series). The switch to `dynamic` is day-level, permanent, and pinned
  (`weight_calc.mode`; the switch day carries `switched_on`).
- **R-eligible** — the weight domain is the sources with a trusted DAILY
  print today. No LOCF ghosts: a stale price is not a price.
- **R-rounded** — allocation runs in full precision and rounds ONCE to 6dp;
  the rounded vector is the ONLY weight state — pinned, consumed by the
  votes, carried for LOO reconstruction. Σw is approximately 1 after
  rounding; consumers renormalize; no source is ever nudged.
- **R-determinism** — pure-Python fixed-order solver (never numpy/BLAS);
  sorted iteration everywhere; any NaN/inf reaching a weight raises and
  kills the publish.
- **R-infeasible** — degenerate bounds publish flagged
  (`degenerate_allocation`) with a CLI WARNING, not red. Cap mass < 1
  publishes **cap-proportional** weights w = cap/Σcaps (the negotiated
  RELATIVE haircuts survive on exactly the thin days they exist for);
  N·w_min > 1 or a cap below the floor publishes uniform; caps summing to
  exactly 1 publish the caps. Config-level feasibility (floors, caps ≥
  w_min, 6dp-exact weights, finite numerics, grid-expressible horizons)
  refuses at load.

## 4. Expected behavior on current data (stated so nobody is surprised)

With in-sample scoring on the slot grid, a source with clean attendance
clears the gates in **8 data-days** (max lookback 2d + max forward 2d + 10
slot-samples + the yesterday-22:00Z cutoff). Under R-quorum-v2 the switch
needs only `switch_min_eligible` (5) eligible-and-defined sources, so the
**calc_v6 genesis replay switches retroactively from the first historical
day that held — 2026-08-17 on B300** (seven sources defined that day; vast
scores 0 with an auditable null until its own samples mature). B200's five
defined sources land ~Aug 24-25. Pre-switch days publish fallback
(byte-identical to the frozen predecessor, machine-checked by the parity
tripwire pointed at calc_v5 / a2 calc_v4); post-switch days diverge by
design — that divergence IS the mint's ruling, measured and recorded at
mint time. Steady state: movers trend toward the global
`weight_max`, frozen list prices toward the floor — with no per-source
caps, softmax spread + w_max + w_min + R-winsor are the only fences, and
the in-sample optimism means even static sources earn small nonzero Q from
noise (accepted under R-insample; λ is the dial).

## 5. Open rulings (the mint ships with these RECORDED, not resolved)

1. **Switchover — RESOLVED (at the calc_v6 mint).**
   Count-based quorum (R-quorum-v2); the v6 genesis replay switches from the
   first historical day the quorum held (2026-08-17 on B300), so v6 diverges
   from the frozen fixed-weight values from that day onward — the ruling
   accepted that divergence explicitly.
2. **Risk caps — RESOLVED: REMOVED (at the calc_v6 mint).** The index
   must not need source-level manual intervention; the softmax spread, the
   global `weight_max`, the `weight_min` floor, and R-winsor are the only
   fences. The manipulation finding (thin-book churn earning weight)
   is now bounded by those global fences alone — recorded, accepted. The
   allocator's per-source-cap mechanism remains in code, unused.
3. **γ / λ retune** — pinned priors; revisit at ≥ 60 daily prints. Under
   R-insample, λ is also the anti-noise dial — raising it is the remedy if
   noise-minted Q spreads weights too aggressively in practice.
4. **Turnover posture** — the schema forbids separate smoothing of final
   weights; a per-day |Δw| cap was proposed and NOT implemented. If live
   turnover proves unacceptable, that is a param change (mint).
5. **Disclosures** — FX residue in USD targets (scaleway's calc-time EURUSD
   sits inside P_{−i} for every other source); the switch day is a
   step-change in weights with zero price movement (flagged `switched_on`).
   Additional recorded disclosures:
   - **Noise-leveling channel**: a source's prints ride inside every other
     source's LOO target at its pinned weight, so sustained fence-passing
     but unpredictable churn depresses rivals' scores and flattens the
     softmax — bounded by R-winsor, the median composite, and [w_min, cap].
   - **In-sample noise floor** (new under R-insample): every source with
     enough samples earns a small positive q from noise, compressing
     relative differences; λ=1.0 is the counterweight and the retune
     checkpoint the remedy.
   - **Thin-day cap posture**: on cap-infeasible days the published weights
     exceed the caps proportionally — vast at ~12.5% on a 4-eligible B300
     day, vs 25% under uniform.
   - **Ops**: deploy code and config atomically (one image); a stale image
     with the new config publishes v5-keyspace artifacts without
     `dynamic_weights`, which the D2 fence then refuses to extend
     (recoverable only by minting v6). `math.exp/log` are not bit-identical
     across platforms; never backfill a series from a different machine
     than the lane's runner.
   - **Fallback-parity tripwire**: `fallback_parity_methodology_id`
     (top-level config, deliberately NOT a calc param) machine-compares
     every computed fallback-mode day against the frozen artifact for the
     same date; a mismatch warns loudly and reddens the firing while still
     publishing. Run the genesis replay as `--dry-run` and rule on any
     mismatch before the first real publish.
