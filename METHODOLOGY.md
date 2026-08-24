# Computable GPU Index (CGI) — Methodology

> Draft: a finalize pass is pending (methodology version sync and cadence wording).

**Collection, screening, and hourly index calculation for GPU rental rates.**

Published for public review. Current version: **v5** (B300) / **v4** (B200).

---

## 1. Scope

A USD price per GPU-hour for a specified accelerator, computed from published on-demand rental rates of a fixed panel of providers. Collection is hourly, and the index and its weights are recomputed at **every hourly observation**.

Live panels: **B200**, **B300**. The calculation is chip-generic — panel, currency treatment and per-provider statistic are configuration, not code. **H100, H200, A100** are the intended next panels; their price history is already being collected.

**Design commitments**

| | |
|---|---|
| Collect first, decide later | Price pages cannot be read retroactively. Collection is unfiltered and runs ahead of any methodology consuming it. |
| Per-source series is the record | Snapshots contain no index value and no cross-source arithmetic. Everything downstream is re-derivable. |
| Deterministic replay | The index history is a pure function of the published record plus published parameters. |
| No revisions | A published observation stands. Corrections publish forward under a new version; prior series stay readable. |

---

# PART I — COLLECTION

## 2. Coverage

Hourly, **28 providers**, **66 chip models** (NVIDIA, AMD, Intel; datacenter and consumer parts). Collection is not filtered to current panels, chips, or eligible tiers — a page listing eleven chips across four tiers is recorded in full.

Every panel is a *selection* over this record, not a separate collection effort.

## 3. Source classes

Each provider carries two disclosure fields: `source_type` and `first_party`.

| Class | Meaning | Examples |
|---|---|---|
| `direct_principal` | Owns or operates the hardware | Verda, Nebius, CoreWeave, Lambda |
| `direct_partnered` | Sells capacity under direct arrangement | RunPod (Secure), Latitude.sh |
| `marketplace` | Third-party hosts; price includes host spread | Vast.ai, Lium |
| `reseller` | Republishes another cloud's capacity | Shadeform |
| `aggregator` | Publishes its *observation* of others' prices | Compute Pulse |
| `hyperscaler` | Separate pricing regime | Oracle |

**Why aggregators and resellers are collected.** Breadth of history, and cross-checking. An aggregator's figure against the provider's own page is how a wrong extraction gets caught — aggregator tables have been observed carrying spot rates as on-demand, and prices for providers that publish none at all.

**What is panel-eligible.** Direct principals, direct-partnered providers, and **marketplaces**. Excluded: **aggregators**, which publish no price of their own, and **resellers**, whose price is another panel member's capacity under a second brand.

Marketplaces are included deliberately. Vast.ai is the only venue in either panel where the published price is a live transactable ask rather than a posted list rate, which makes it the panel's only direct read on clearing prices. Two disclosures follow: its price embeds a host and platform spread, and it is the only member where manipulation would be transactional rather than reputational — hence the specific prohibition in §14.

Exclusion is by panel construction, not by filter. The panel is an explicit enumerated list (§C.2); nothing in the calculation branches on `source_type` or `first_party`, which are disclosure fields.

**Double-counting** is prevented at panel selection, not at calculation: no two panel members may resolve to the same underlying capacity. Shadeform was excluded from the B300 panel because its only B300 listing republished Verda's capacity; Prime Intellect was excluded because its API returns the upstream provider name directly. See §15 for the residual gap.

## 4. What is recorded

| Field | Meaning |
|---|---|
| Provider's own label | The listing string as shown — **this is the record** |
| Chip | Canonical model, derived from that label |
| Price per GPU-hour | Normalized, in the provider's quoted currency |
| USD price | Populated only for USD-listed prices |
| Raw value and unit | The published figure and what it priced |
| GPU count basis | The divisor, from the provider's own page |
| Tier | on-demand / spot / preemptible / reserved / committed / serverless |
| Region, notes | As published |
| Quality flags | Plausibility, rejection reasons |
| Marketplace identity | Offer, machine, host ids; verification status |

**Normalization uses the provider's stated GPU count** — never a third party's per-GPU figure, never an inferred count. $68.80 for an 8-GPU instance → $8.60/GPU-hour. A missing or implausible count skips the listing rather than dividing by an assumed 1.

**Currency is recorded, never assumed.** A euro-billing provider is recorded in euros with the USD field empty. Conversion happens at calculation, against a published rate.

**Chip identity is derived from the label**, so a later revision to identification re-applies across all history. Matching is boundary-aware: `B200` never matches inside `GB200`, `H20` never inside `H200`, `A10` never inside `A100`. An unmatched label records no chip and is surfaced for review — never guessed.

## 5. Index eligibility

Collection records everything; the index uses a screened subset. All screens **flag**; none delete.

| Screen | Rule |
|---|---|
| Panel | Fixed enumerated list per chip (§C.2) |
| Tier | On-demand only; spot and preemptible recorded but never eligible |
| Product identity | Reject `NVL`, `GB200`, `GB300`, `Grace`, `B200A` in the structured label; stated memory 186 or 189 GB; GPU count 36 or 72 |
| Plausibility | Per-chip price band; out-of-band prices flagged, not deleted |
| Jump | ≥25% move with fewer than 2 other members moving ≥10% is flagged |

**Product identity** exists because NVIDIA's GB200 is a different product from the B200 — different power envelope, sold in different quantities, priced 2–4× higher — yet two providers market it under the string "B200". Collection files it as its own chip; the index rejects it as a B200. Matching is on the structured label only, since descriptive text legitimately mentions "NVLink". A count of 4 is deliberately not rejected: 4-GPU slices of genuine 8-GPU servers are ordinary listings.

**The jump screen's corroboration requirement** is what distinguishes a market move from a single-provider glitch — B200 fell ~30% over three weeks in mid-2026 and passed. Where too few providers are comparable for corroboration to be decidable, the screen stands down. A paired report also shows the move in the *specific machine* the previous price came from: a machine at an unchanged price beside a book that moved 59% is a selection artifact, not a repricing.

Collection mechanics: **Appendix A**. Per-provider extraction: **Appendix B**.

---

# PART II — INDEX CALCULATION

## 6. Observation cadence

An index value and a weight vector are published for **each hourly observation**. There is no designated fixing hour: the series is the hourly series, and every value carries the parameter set used to compute it.

An hour with no usable record publishes as an explicit missing-observation entry — never skipped, never interpolated.

**Settlement** over a calculation period is the arithmetic average of the published observations within it. Averaging over many observations, rather than reading one, is what removes any single hour's significance and with it the incentive to influence a particular moment.

## 7. Each provider's observed price

**Default: the lowest eligible listing** — matching chip, eligible tier, not flagged, usable price. Foreign-currency prices convert before the minimum is taken.

**A provider publishing an order book needs a different statistic.** The distinction is not the source class but what the page publishes:

| Publishes | Statistic | Why |
|---|---|---|
| One list price | Lowest eligible | The minimum selects among that provider's own tiers and configurations |
| A book of third-party asks | Median over the eligible host population | The lowest ask is one host's offer — frequently unverified, sometimes a single machine — and is not representative of the venue |

Vast.ai is the only order book in either panel. It prices as the **volume-weighted median of rentable on-demand per-GPU asks across verified US and Canadian hosts**, weighted by each offer's GPU count. On one real observation the lowest-price rule would have returned an unverified host 29% below the median.

Every other panel member — including those classed `marketplace` — publishes a price list, so the lowest-eligible rule applies to them unchanged.

**Currency conversion** uses the ECB reference rate — public, citable, archived. Each rate is stored on first use and reused permanently, so replays convert at the original rate. Non-publication days walk back to the last published rate, recording its actual date. No rate within seven days → the provider is held out, never converted at a guess.

The ECB publishes once per business day, so every hourly observation within a day converts at the same rate. Intraday movement in a non-USD provider's index contribution therefore reflects its own price only, not exchange-rate drift.

**Ambiguous currency labels fail closed.** Foreign-currency treatment requires a well-formed three-letter code *and* a native figure; anything else is held out. An apparent billing-currency switch is held out until **three consecutive** observations confirm it, then old history is discarded and new-currency history starts fresh.

## 8. Per-provider outlier check

Judged against the provider's **own** recent history, never cross-sectionally — providers sit at structurally different price levels, so a cross-sectional test would flag normal marketplace pricing at every observation.

```
accept if |price - mean(last 20)| <= 3.0 * max(sd, $0.05)
```

| Property | Reason |
|---|---|
| Rejected prices still enter the history | A genuine repricing then costs one day, not perpetuity |
| The test runs in the quoted currency | On 20 Aug 2026 a ~1% EUR/USD move ejected a provider whose price sat unchanged at €7.50 |
| Minimum band ±$0.15 | A frozen list price has sd 0; without a floor any repricing is rejected |
| Threshold widened 2.5 → 3.0 sd | At 2.5 it rejected a genuine $5.95 → $6.60 repricing by 1.5¢. The aggregation in §9 is now the primary outlier defense; this is a gross-error screen |

First ten observations pass untested. Such prices are flagged for review if more than 15% from that observation's cross-provider average, but counted.

**Documented exclusions.** Where a recorded price is known wrong by rule and the true price was never captured, that (observation, provider) pair is excluded by hand with a written reason — substituting a value would fabricate unobserved data. Exclusions publish with the observation and are then fixed permanently.

## 9. The index: median of standard-deviation votes

Not a weighted average. A weighted median over votes, adapted from Pyth Network's price-aggregation design.

Each passing provider casts **its full weight three times** — at its price and its price ± its own standard deviation:

```
for each passing provider i:
    vote (price_i - sd_i)  weight w_i
    vote (price_i)         weight w_i
    vote (price_i + sd_i)  weight w_i

index      = weighted median of all votes
dispersion = larger distance from the index to the
             25th / 75th weighted vote percentiles
```

sd is the provider's own recent price variability — the §8 window — with the same $0.05 floor.

A stable provider votes tightly and concentrates its influence; a volatile one spreads its votes and dilutes its own. Because the result is a median, no single provider can pull the index past where the vote mass sits. The floor stops staleness impersonating conviction: a frozen price would otherwise cast three identical votes claiming certainty it never demonstrated.

Dispersion widens when providers disagree, giving a usable read on how much confidence the published value deserves.

**Robustness** — eight providers, one repricing +30%, all else fixed:

```
median of votes    6.740000 → 6.740000   (0.000000)
weighted average   6.825054 → 7.161576   (+0.336522)
```

The weighted average is still published as a diagnostic. It is not the index.

**Insufficient providers.** Below the minimum passing panel members (5 of 9 on B200), the observation publishes explicitly as *no index*.

---

# PART III — WEIGHTING

## 10. Principle

Weights are **derived from measured behaviour**. Membership is negotiated; allocation across members is computed.

**A provider earns weight to the extent its recent price movements anticipate subsequent movement in the rest of the panel.** Two constraints:

- **Price level is not an input.** Only movement is scored. A cheap marketplace is not penalized for being cheap.
- **No provider is scored against itself.** Evaluation runs against a leave-one-out panel excluding the provider scored.

## 11. Scoring

### 11.1 Returns

The provider's own return uses its quoted currency; the panel's return is in USD and excludes the provider being scored.

```
r_i(a,b)  = clamp log( p_i(b) / p_i(a) )

r^-i(a,b) = clamp log(  Σ_{j≠i} w_j · p_j(b)
                      / Σ_{j≠i} w_j · p_j(a) )
```

Both require real observations at both endpoints; a return spanning a currency change is undefined, never spliced across an exchange rate.

`w_j` is the weight vector pinned at the sample's day, held fixed at both endpoints and summed over providers with observations at both — so weight drift and membership churn cannot register as panel movement. The denominators cancel, so the vector needs no normalization.

`clamp` bounds every return at ±0.5 (≈ ±65%; the largest genuine repricing observed is ~11%). This is load-bearing because scoring history deliberately includes prices the §8 check rejected — without a bound, one absurd-but-real observation would distort every *other* provider's window without limit.

### 11.2 Signal and outcome

Lookbacks `Δ ∈ {6h, 1d, 2d}`, forwards `h ∈ {6h, 1d, 2d}`. At each sample time τ:

```
signal    X_i(τ)   = [ r_i(τ-Δ, τ) - r^-i(τ-Δ, τ) ]  over Δ

outcome   y_i^h(τ) = r^-i(τ, τ+h)
```

The signal is movement *in excess of* the panel. Moving with the panel carries no information; moving ahead of it is the whole quantity being measured.

### 11.3 Admissible samples

With `T` = the last observation strictly before the one being priced and `L` = 90 days:

```
τ ≥ T - L      inside the history window
τ + h ≤ T      outcome realized by the cutoff
```

plus a pinned weight vector for τ's day, and every leg of `X` and `y` computable from real observations — no carry-forward, since a stale price is not a price.

`τ + h ≤ T` is the information boundary of §13: `T` is the last observation strictly before the one being priced, so nothing observed at an hour can enter that hour's own weights.

Samples decay exponentially toward the cutoff, half-life 30 days:

```
a(τ) = 2^( -(T - τ) / 30d )
```

### 11.4 Fit and score

Per provider and forward horizon, features are standardized by their `a`-weighted moments over the window, then fitted by ridge with an unpenalized intercept:

```
min over α, β:
   Σ a(τ) · ( y(τ) - α - βᵀz(τ) )²  +  λ‖β‖²
   λ = 1.0
```

Standardizing makes `λ` mean the same thing for a 6h feature and a 2d one. A feature with no variation over the window is neutralized to zero — the routine case for a frozen list price.

The score is that fit measured in-sample, against the same weighted measure:

```
R²      = 1 - Σ a(τ)(y - ŷ)² / Σ a(τ)(y - ȳ)²

q_{i,h} = max(0, R²)

Q_i     = mean of q_{i,h} over the three forwards
```

`q_{i,h}` is **undefined — not zero** — with fewer than 10 samples, if the weighted variance of `y` falls below `1e-12`, or if the fit is singular or `R²` non-finite. `Q_i` requires every `q_{i,h}` to be defined, so the score cannot lurch when a longer forward window comes online after a shorter one.

## 12. Allocation

```
s_i = exp(γ Q_i) / Σ_j exp(γ Q_j)

w_i = w_min + (1 - N·w_min) · s_i

then, while any w_i > w_max:
    set the violators to w_max and redistribute their
    excess over the remainder in proportion to s_i
```

Every provider receives the **floor `w_min` = 2.5%** first; the remainder distributes by share. **Both bounds are uniform — no per-provider values, and no weight requires a human decision.** Allocation runs in full precision and rounds once; the rounded vector is the published weight.

**γ controls how sharply score differences become weight differences.** Four providers scoring 0.20 / 0.10 / 0.05 / 0.00:

| γ | Weights |
|---|---|
| 0 | 25.0 / 25.0 / 25.0 / 25.0% |
| 1 | 27.6 / 25.2 / 24.1 / 23.1% |
| **4** | **36.2 / 25.1 / 21.0 / 17.7%** |
| 10 | 54.6 / 21.7 / 14.1 / 9.6% |
| 25 | 83.4 / 9.1 / 4.4 / 3.0% |

γ = 4 is a chosen prior, not a fitted value: predictiveness matters visibly, a good score cannot run away with the panel. Revisiting it is a versioned change.

## 13. Safeguards

**A uniform ceiling suffices because the aggregation bounds influence.** A manipulated provider that moves and drags others looks identical to one that genuinely leads. Take a provider at the full 30% ceiling and reprice it +30%:

```
index  6.970000 → 6.970000   (0.000000)
```

A median over standard-deviation votes does not move. Ceiling and aggregation together bound single-provider influence without per-provider judgment.

**An observation cannot move its own weights.** Every weight input is realized strictly before the observation being priced. A party listing capacity minutes before an observation cannot move its own weight for it, and each hour's weights are determinable before that hour's prices are read.

**Before scores exist.** `Q_i` requires 10 samples per forward window, so a new panel cannot compute weights on its first observations. Until it can, the index uses the panel's opening weights — the membership weights agreed when the panel was seated, published in each record like any other parameter.

The panel switches to computed weights **once and permanently**, on the first day both conditions hold:

```
every provider meeting the attendance floor
    has a defined Q_i
and at least 5 providers reported at that observation
```

The first condition stops a brief outage from triggering the switch before that provider has ever been scored. The second stops a thin observation from locking it in. In practice a panel clears both within roughly eight days of collection, so opening weights govern only the opening window. The switch is flagged permanently in the record, since it is a step change in weights with no underlying price movement.

**Degenerate cases** (too few providers for the floors to fit) publish an even split, flagged.

**Attendance floor.** The first switch condition applies only to providers with a usable price at **50% or more of scheduled observations** over the trailing 90 days. Without that qualifier, a provider publishing rarely — often enough to stay inside the window, never often enough to reach 10 samples — would be permanently unscorable and would block the switch indefinitely.

The floor governs the switch test alone. A provider below it stays index-eligible on any day it prints, and post-switch its undefined score allocates zero, so it receives the weight floor. A provider chronically below it keeps receiving the weight floor; removing it from the panel is a membership decision taken at panel review, not by the calculation.

50% is deliberately loose. A defined score needs 10 samples per forward window, and each sample requires the provider present at four offsets (τ−48h, τ−24h, τ−6h, τ) — at 50% attendance that yields roughly 68 samples, at 25% roughly two, below 10% effectively none. A single contiguous two-week outage leaves attendance near 84%. The floor therefore excuses only providers that could not be scored under any pattern, not a provider having a bad month.

---

## 14. Governance

| | |
|---|---|
| **Versioning** | The full parameter set publishes inside every record. Any parameter change requires a new version; the calculation refuses to extend a series under altered parameters. Prior versions stay frozen and readable. |
| **Replay** | Each observation advances from the *published record* of prior observations, not from re-reading raw observations — raw records legitimately grow (late uploads, backfilled FX), and re-deriving would rewrite history the series was built on. |
| **Reconciliation** | Published days are re-compared against the raw record on a rolling basis; divergence is reported. The published index stands — the report is for review, not revision. |
| **Retention** | Every observation kept for life of trade + 2 years, available to either party: source URL, timestamp, figure as published, collecting process identity. |
| **Independence** | The calculation agent must be independent of both parties. Neither party nor an affiliate may be a panel provider, nor list capacity on a marketplace panel member so as to influence the price at or near any observation. |

## 15. Known limitations

**Committed-tier pricing.** Eligibility excludes interruptible tiers but not committed ones, so where a provider publishes both, a committed-term rate can win the lowest-price rule. The contractual definition excludes committed pricing. Unreconciled.

**Scoring is in-sample.** Scores are measured on the data the relationship was fitted to. Shrinkage, the score floor and the weight bounds are the counterweights, but every provider with sufficient data earns a small positive score from noise, compressing differences between them.

**Window lengths are counted in observations and are short in wall-clock terms.** The outlier window (20), warm-up (10), scoring sample gate (10) and currency-change confirmation (3) span 20, 10, 10 and 3 hours respectively. The outlier test therefore judges a price against roughly the preceding day rather than a longer history, and three hours is a weak guard against a mislabelled billing currency. Sizing these to the hourly grid is a versioned change and is outstanding.

**γ and shrinkage are unvalidated priors**, scheduled for review after ~60 days of prints.

**Manipulation resistance is structural, not targeted.** No per-provider limits exist, by design — the panel requires no source-level intervention. The defenses are the uniform ceiling, the median aggregation (§13), the same-day information boundary, and the movement cap. A cheap-to-move provider cannot shift the index materially but can accumulate weight to the ceiling if its moves happen to lead. Whether liquidity should map to a rule-based ceiling is open.

**Cross-influence.** Each provider's prices sit inside every other's evaluation panel, so a provider moving constantly but unpredictably depresses others' scores. Bounded by the movement cap, the median and the weight floor; not zero.

**Currency residue.** A euro-billing provider's converted price sits inside the USD panel used to evaluate every other provider.

**Residual double-count exposure.** Panel selection prevents two members resolving to the same capacity, but a reseller that does not disclose its upstream providers cannot be verified as distinct. Such sources are excluded from panels for that reason; the check is judgment at selection, not a runtime test.

**Quoted, not transacted.** On any day the index may average prices for capacity that is not obtainable — providers have been observed publishing a price while showing out of stock. An availability screen remains open.

---

# APPENDIX A — Collection mechanics

| | |
|---|---|
| **Idempotency** | Collection runs more frequently than the recording interval and attributes each run to the most recent scheduled observation. If that observation exists, the run exits. First run after each interval collects; later runs are free; a failed run self-heals on the next. |
| **No backfill** | An interval missed before the following one is permanently missing. Late-recorded observations are marked. |
| **Time limits** | Three bounds per provider: network timeout, hard per-provider limit (a network timeout bounds one operation, not a sequence), overall run budget. Exhaustion records an error. |
| **Visible holes** | Every configured provider appears in every snapshot — success, error, or unimplemented. Never a silently shorter list. |
| **Transport** | HTTPS-only including on redirect (a redirect to HTTP raises). Response bodies size-capped, with a wall-clock read limit bounding a slow-drip response a network timeout cannot see. Certificate verification never weakened. |
| **Extraction contract** | Record the published figure alongside the normalization; state currency explicitly; **fail loudly on reading nothing** — a page that silently changed shape must error, not present as a healthy provider with no prices. No recipe aggregates across providers. |
| **Thin-observation gate** | Fewer than five panel members read successfully → the observation is discarded and the interval left unclaimed so the next run retries. Recording a thin snapshot would both stop retries and become the reading the calculation uses. |
| **Alarms** | A missing previous day, or a previous day missing intervals, is reported on every run until resolved. |

---

# APPENDIX B — Per-provider extraction

Every recipe fails loudly rather than guessing. Each pin below exists because of an observed failure.

## B.1 Panel providers

| Provider | Published as | Normalization | Extraction pin |
|---|---|---|---|
| Verda | per-GPU, leading count | ÷ stated count | Currency read from within each offer's own record so it cannot bleed from an adjacent offer; unstated currency marked unknown, not assumed USD. Serverless and cluster rows excluded (priced per node) |
| Nebius | per GPU-hour | none | Complete column header row verified before any read — a price column inserted ahead of the known ones would relabel one tier as another. Rows with an unexpected extra price column refused |
| Hyperstack | per GPU-hour | none | Exactly three numeric specification fields required per row; this is what separates on-demand from reserved rows, which lack them |
| Scaleway | **EUR** per node-hour | ÷ stated GPU count | Public instance-catalog API. Non-standard memory configs noted, not dropped. Pagination bounded with an explicit warning rather than silent truncation |
| RunPod | per GPU-hour | none | Secure Cloud price only. Community Cloud is excluded from panels and its price field is never requested, so it cannot be recorded by accident |
| Vast.ai | per-instance total | ÷ stated GPU count | See below |
| Massed Compute | per node-hour | ÷ stated count | Each row's price window bounded at the next row's key, so a "contact us" row cannot capture a neighbour's price — with differing GPU counts that error is wrong by an integer factor. Exactly one price in window or skip |
| Latitude.sh | per node-hour **and** node-month | hourly ÷ count; monthly ÷ 730 ÷ count | Both tiers recorded, each labelled |
| Lambda | per GPU-hour | none | Read starts after the instance-pricing heading and fails if committed cluster pricing appears in scope. The 8-GPU row is pinned on full specification (180 GB / 208 vCPU / 2900 GiB / 22 TiB) because the 4/2/1-GPU tabs price the same product name at $6.79 / $6.89 / $6.99. Exactly one match required |
| CoreWeave | per 8-GPU instance | ÷ 8 | Section narrowed to on-demand North America, must contain exactly one B200 heading, fails if GB200 NVL72 content appears. GPU count must read 8 and memory 180 GB — Grace-coupled rows state their count as literal text "4^1" and fail this check |
| Together AI | per GPU-hour | none | Read confined to the GPU Clusters section: the Dedicated Inference section contains a **character-identical** B200 row at a different price. Four identity checks, one binding the first column to the on-demand tier so a column reorder fails |
| E2E Networks | per GPU-hour (USD table) | none | The specification triple must immediately precede the price cell, and the next price cell must be ~730× it (the monthly column), so a column reorder fails rather than promoting the wrong cell |
| Shadeform | **cents** per instance-hour | ÷ 100 ÷ stated count | Reseller; recorded not-first-party with the underlying provider attached to every row |

**Vast.ai** (order book). Queried without a GPU-count preference and without the API's `verified` filter, which is broken in a way that silently drops the cheapest hosts. Results ordered by instance total truncate the largest totals first — exactly the cheap multi-GPU machines — so a full result page triggers a second query from the other end. Per offer: an integer GPU count 1–16 required (a missing count is never treated as 1, which would record a whole-instance price as per-GPU); an arithmetic check requires per-GPU price × count to reproduce the published total. Offers deduplicated to one per physical machine, keeping the cheapest, ranked by per-GPU price rather than instance total — ranking by total had buried $6.25/GPU 8-GPU machines beneath $10.94/GPU single-GPU slices of one expensive host.

## B.2 Additional collected providers

Not in any panel (§3). Civo, Compute Pulse, Crusoe, Deep Infra, DigitalOcean, fal, GMI Cloud, GPU.ai, Hot Aisle, Lium, Oracle Cloud, OVHcloud, Sesterce, TensorPool, Voltage Park.

| Provider | Extraction pin |
|---|---|
| Civo | 23 tables share one style; a section fence keeps Kubernetes, database and storage rows out of the GPU series |
| Compute Pulse | Aggregator — every row is its observation of another provider's price. Not-first-party, recorded as leads |
| Deep Infra | Page dominated by ~30 per-token serverless tables plus two lookalike "Tier" tables; only the dedicated-GPU table is read |
| DigitalOcean | Documentation page used over the marketing page — prices present in the raw response |
| fal | Three tables in near-identical markup with no identifiers; position pinned explicitly |
| Hot Aisle | One of few AMD principals publishing a public per-GPU-hour rate |
| Lium | Bittensor-subnet marketplace; entire machine book in one response |
| Oracle | Public price-list API behind the pricing page, whose cells are empty until scripted. Licence and on-premises rows excluded by policy and accounted for |
| OVHcloud | Two billing subsidiaries queried separately; wrong currency or duplicate plan code fails the read |
| Sesterce | Page self-declares a five-minute cache; embedded render timestamp recorded so staleness is visible |
| TensorPool | Page renders the same table twice and includes competitor columns; only the provider's own column is read |

---

# APPENDIX C — Parameters

Every value publishes inside each record. Changes require a new version.

## C.1 Collection

| Parameter | Value |
|---|---|
| Collection interval | hourly |
| Index / weight recomputation | every hourly observation |
| Providers collected | 28 |
| Chip models identified | 66 |
| Minimum panel members to record | 5 |
| Series start | 10 Aug 2026 (B300) · 16 Aug 2026 (B200) |
| Retention | life of trade + 2 years |
| Jump threshold | 25% |
| Corroboration threshold | 10% |
| Corroborators required | 2 |

## C.2 Index calculation

| Parameter | B300 | B200 | Controls |
|---|---|---|---|
| Aggregation | median of votes | median of votes | How prices combine |
| Excluded tiers | spot, preemptible | spot, preemptible | Interruptible capacity is not the underlying |
| History window | 20 | 20 | Observations in the outlier test |
| Threshold | 3.0 sd | 3.0 sd | Acceptance band width |
| Minimum variability | $0.05 | $0.05 | Floor under a frozen price's band |
| Warm-up | 10 | 10 | Observations before the test applies |
| Test currency | as quoted | as quoted | Prevents FX moves ejecting a provider |
| Review flag | 15% | 15% | Distance from panel average flagging review |
| Minimum panel | — | 5 | Passing providers needed to publish |
| FX source | ECB | none (USD only) | Conversion reference |
| FX staleness limit | 7 days | — | Beyond this, hold the provider out |
| Currency-change confirmation | 3 | 3 | Consecutive observations confirming a switch |

Counts above — history window, warm-up, currency confirmation — are in **observations**, so their wall-clock span follows the collection cadence: at hourly collection the outlier window spans 20 hours and warm-up 10 hours. See §15.
| Order-book statistic | — (lowest eligible) | volume-weighted median | Replaces lowest-price for Vast.ai |

**Panel membership.** Membership is the parameter; weights are computed (Part III).

| B300 (8) | B200 (9) |
|---|---|
| Verda | Verda |
| Nebius | Nebius |
| Hyperstack | Hyperstack |
| Scaleway | Lambda |
| RunPod | CoreWeave |
| Vast.ai | Together AI |
| Massed Compute | Massed Compute |
| Latitude.sh | RunPod |
| | Vast.ai |

## C.3 Weighting

Identical on both panels. No per-provider values.

| Parameter | Value | Controls |
|---|---|---|
| Lookback windows | 6 h, 1 d, 2 d | Spans over which a provider's own move is measured |
| Forward windows | 6 h, 1 d, 2 d | Spans over which the panel's response is measured |
| History | 90 days | How far back the relationship is estimated |
| Half-life | 30 days | How fast old observations lose influence |
| Shrinkage penalty | 1.0 | How hard the fit is pulled toward "no relationship" — the main defence against scoring noise as skill |
| **Sensitivity (γ)** | **4.0** | **How sharply score differences become weight differences** (§12) |
| Weight floor | 2.5% | Minimum weight for an eligible provider |
| Weight ceiling | 30% | Maximum weight — uniform, no exceptions |
| **Minimum observations** | **10** | **Per provider, per window: observations needed before that provider can be scored.** Below it the score is undefined |
| **Minimum variation** | **1e-12** | **The panel must have moved for prediction to be a meaningful question.** Below this variance in what is predicted, the score is undefined rather than computed from nothing. Set above price-rounding noise (~1e-14), below any real movement |
| **Minimum panel for transition** | **5** | **Whole-panel, one time: providers that must report at the observation where the index permanently switches to derived weights.** Distinct from minimum observations — that asks "does this provider have enough history?", this asks "is this observation broad enough to change the methodology?" |
| Attendance floor | 50% | Share of scheduled observations a provider must reach to gate the transition (§13) |
| Movement cap | ±65% | Bounds any single observation's influence; largest real move observed ~11% |

γ and shrinkage are **initial priors, not fitted values** — review scheduled at ~60 days of prints; revisiting either is a versioned change.

---

*Prices, weights and parameters here are current as of publication and are configuration, not fixed properties of the methodology. Every published record carries the parameter set used to compute it.*
