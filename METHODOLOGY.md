# Computable GPU Index (CGI) Methodology

The Computable GPU Index (CGI) is the first open-source price index for GPU compute, with a mathematically robust methodology anyone can verify and reproduce.

CGI is a USD price per GPU-hour for a specified accelerator (e.g. H100, H200, B200, B300), computed from the published on-demand rental rates of a fixed panel of providers.

## Motivation

What is the price of a GPU?

Nobody agrees. Every provider quotes a different number, and existing indexes are closed black boxes: a figure you are asked to trust without seeing the data, the method, or the code behind it. Despite being rented, resold, and financed at commodity scale, compute lacks the reference rate every mature commodity market has.

CGI is built to be that number, and to earn trust in the open rather than by authority.

## The five properties

Five properties are the spine of the design. Every rule in this document enforces at least one, and each section is tagged with the properties it serves.

| Property | Meaning | Enforced by |
| --- | --- | --- |
| Verifiable | Every data point is sourced, timestamped, and kept | Collection record (section 3), retention (section 10) |
| Reproducible | Every published value is a pure function of the public record plus published parameters | Replay and versioning (section 10), parameters inside every record (section 12), open code |
| Fault-tolerant | A source can break or vanish; the number stands | Fail-loud extraction (section 3.5), explicit missing observations and the period-rate fill rule (section 9), undefined-not-zero scores (section 8.5), attendance (section 8.6) |
| Outlier-resistant | A source can lie or glitch; the number stands | Screens (section 5), self-history outlier check (section 6.4), interquantile mean of votes (section 7), movement cap (section 8.2) |
| Transparent | No black box: the method, every parameter, and every human decision are published | This document, versioned changes (section 10), published exclusions (section 6.5) |

These properties highlight our design principles directly: the data is verifiable, the method is transparent, the prints are reproducible via code, and the result is a robust price surviving data sources that disagree, err, or break.

Fault-tolerant and outlier-resistant together are what "mathematically robust" means: the index never needs to know why an input went wrong. The aggregation makes a wrong input irrelevant whether the cause was breakage or manipulation, so no per-provider judgment exists anywhere in the calculation. Sections 7.3 and 8.9 demonstrate the bound.

## About this document

This is the general methodology, chip-generic: panel membership, per-SKU parameters, and the per-provider statistic are configuration, not code. Each SKU binds it through its own SKU document, a sub-page of this page: B300, B200, H100-SXM, and H200-SXM. Normative rules live in the numbered sections; the reasoning behind a rule sits under it in a **Why** block you can skip without losing the spec. Everything needed to re-implement the index and reproduce every published value is here, in the SKU documents, or in the published records.

## Contents

- [Motivation](#motivation)
- [The five properties](#the-five-properties)
- [About this document](#about-this-document)
- [1. Overview](#1-overview)
  - [1.1 Summary](#11-summary)
  - [1.2 Design commitments](#12-design-commitments)
  - [1.3 Conventions](#13-conventions)
- [2. Definitions](#2-definitions)
- [3. Collection](#3-collection)
  - [3.1 Coverage](#31-coverage)
  - [3.2 Source classes](#32-source-classes)
  - [3.3 What is recorded](#33-what-is-recorded)
  - [3.4 Normalization rules](#34-normalization-rules)
  - [3.5 Collection mechanics](#35-collection-mechanics)
- [4. Panel construction](#4-panel-construction)
  - [4.1 Eligibility by source class](#41-eligibility-by-source-class)
  - [4.2 Double counting](#42-double-counting)
  - [4.3 Membership](#43-membership)
- [5. Screening](#5-screening)
  - [5.1 Product identity](#51-product-identity)
  - [5.2 Jump screen](#52-jump-screen)
- [6. Each provider's price](#6-each-providers-price)
  - [6.1 Default: the lowest eligible listing](#61-default-the-lowest-eligible-listing)
  - [6.2 Order books](#62-order-books)
  - [6.3 Currency conversion](#63-currency-conversion)
  - [6.4 Outlier check](#64-outlier-check)
  - [6.5 Documented exclusions](#65-documented-exclusions)
- [7. Aggregation](#7-aggregation)
  - [7.1 Interquantile mean of standard-deviation votes](#71-interquantile-mean-of-standard-deviation-votes)
  - [7.2 Worked example](#72-worked-example)
  - [7.3 Robustness](#73-robustness)
  - [7.4 Stability band](#74-stability-band)
  - [7.5 Insufficient providers](#75-insufficient-providers)
- [8. Liveness weights](#8-liveness-weights)
  - [8.1 Principle](#81-principle)
  - [8.2 Returns](#82-returns)
  - [8.3 Signal and outcome](#83-signal-and-outcome)
  - [8.4 Admissible samples](#84-admissible-samples)
  - [8.5 Fit and liveness score](#85-fit-and-liveness-score)
  - [8.6 Attendance](#86-attendance)
  - [8.7 Allocation](#87-allocation)
  - [8.8 Opening weights and the switch](#88-opening-weights-and-the-switch)
  - [8.9 Safeguards](#89-safeguards)
- [9. Publication and settlement](#9-publication-and-settlement)
  - [9.1 Cadence](#91-cadence)
  - [9.2 Missing observations](#92-missing-observations)
  - [9.3 Settlement and the period rate](#93-settlement-and-the-period-rate)
  - [9.4 Coverage](#94-coverage)
- [10. Governance and reproducibility](#10-governance-and-reproducibility)
- [11. Providers](#11-providers)
- [12. Parameters](#12-parameters)
  - [12.1 Collection](#121-collection)
  - [12.2 Period rate](#122-period-rate)
  - [12.3 Index calculation](#123-index-calculation)
  - [12.4 Liveness weighting](#124-liveness-weighting)
- [SKU documents](#sku-documents)

---

## 1. Overview

### 1.1 Summary

What CGI delivers: one reference price for GPU compute that no single provider can materially move, that both sides of a trade can settle against, and that anyone can recompute from the public record. Each claim below is checkable against that record.

- Prices are collected every 15 minutes from 28 providers, unfiltered.
- An index value, a stability band, and a liveness weight vector publish for every observation, every 15 minutes. There is no designated fixing time.
- Each panel provider contributes one number per observation: its lowest eligible on-demand listing, except order books, which contribute a volume-weighted median.
- The index is an interquantile mean over votes, not a weighted average: the weighted mean of the central third of the vote mass. Each provider votes at its price and its price plus and minus its own standard deviation.
- Liveness weights are computed from measured behavior: whether a provider's recent moves anticipated the rest of the panel. No weight requires a per-provider human decision.
- The index history is a pure function of the published record plus published parameters. Published observations are never revised.
- A period rate is the time-average of the index values within it, missing observations filled from the observations preceding them; whether it is fit to settle on is governed by a published coverage record.

### 1.2 Design commitments

Four commitments shape everything downstream.

| Goal | Meaning |
| --- | --- |
| Collect first, decide later | Price pages cannot be read retroactively. Collection is unfiltered and runs ahead of any methodology consuming it. |
| Per-source series is the record | Snapshots contain no index value and no cross-source arithmetic. Everything downstream is re-derivable. |
| Deterministic replay | The index history is a pure function of the published record plus published parameters. |
| No revisions | A published observation stands. Corrections publish forward under a new version; prior series stay readable. |

### 1.3 Conventions

Window lengths named in observations (history window, warm-up, currency confirmation) follow the collection cadence: at 15-minute collection, a 20-observation window spans 5 hours.

Parameters identical across all SKUs live in section 12. Parameters bound per SKU (panel membership, minimum panel, FX treatment, order-book statistic, weight floor, product identity screen, opening weights, series start, version) live in each SKU document.

---

## 2. Definitions

| Term | Meaning |
| --- | --- |
| SKU | One accelerator model with its own panel, index series, and SKU document |
| Observation | One scheduled reading of all configured providers, every 15 minutes |
| Listing | One published price for one configuration on one provider's page |
| Panel | The fixed, enumerated list of providers whose prices enter a SKU's index (SKU document) |
| Eligible | A listing that passes every screen in section 5 |
| Passing | A provider whose observed price passes the outlier check in section 6.4 |
| Index value | The interquantile mean of votes at one observation (section 7) |
| Stability band | The published uncertainty band around the index value (section 7.4) |
| Liveness weight | A provider's computed share of influence, `w_i` (section 8) |
| Liveness score | A provider's measured predictiveness, `Q_i`, from which weights derive (section 8.5) |
| Settlement | The time-average of index values over a calculation period, missing observations filled from preceding observations (section 9.3) |
| Coverage | Filled observations divided by scheduled observations over a period (section 9.4) |

Symbols: `p_i` is provider i's observed price, `sd_i` its standard deviation over its recent history, `w_i` its liveness weight. `τ` is a sample time, `T` the last observation strictly before the one being priced, `L` the history window, `Δ` a lookback span, `h` a forward span.

---

## 3. Collection

*Verifiable. Fault-tolerant.*

### 3.1 Coverage

Every 15 minutes, 28 providers, every chip model they price (NVIDIA, AMD, Intel; datacenter and consumer parts). Collection is not filtered to current panels, chips, or eligible tiers. A page listing eleven chips across four tiers is recorded in full.

Every panel is a selection over this record, not a separate collection effort.

### 3.2 Source classes

Each provider carries two disclosure fields: `source_type` and `first_party`.

| Class | Meaning | Examples |
| --- | --- | --- |
| `direct_principal` | Owns or operates the hardware | Verda, Nebius, CoreWeave, Lambda |
| `direct_partnered` | Sells capacity under direct arrangement | RunPod (Secure), Latitude.sh |
| `marketplace` | Third-party hosts; price includes host spread | Vast.ai, Lium |
| `reseller` | Republishes another cloud's capacity | Shadeform |
| `aggregator` | Publishes its observation of others' prices | Compute Pulse |
| `hyperscaler` | Separate pricing regime | Oracle |

Aggregators and resellers are collected but never panel-eligible (section 4.1).

> **Why collect them at all?** Breadth of history, and cross-checking. An aggregator's figure against the provider's own page is how a wrong extraction gets caught. Aggregator tables have been observed carrying spot rates as on-demand, and prices for providers that publish none at all.

### 3.3 What is recorded

| Field | Meaning |
| --- | --- |
| Provider's own label | The listing string as shown. This is the record |
| Chip | Canonical model, derived from that label |
| Price per GPU-hour | Normalized, in the provider's quoted currency |
| USD price | Populated only for USD-listed prices |
| Raw value and unit | The published figure and what it priced |
| GPU count basis | The divisor, from the provider's own page |
| Tier | on-demand / spot / preemptible / reserved / committed / monthly-commit / serverless / from-floor |
| Region, notes | As published |
| Quality flags | Plausibility, rejection reasons |
| Marketplace identity | Offer, machine, host ids; verification status |
| Book population accounting | On order-book sources, whether the stored book is the full eligible population or a truncated read (section 6.2) |

### 3.4 Normalization rules

**GPU count.** Normalization uses the provider's stated GPU count. Never a third party's per-GPU figure, never an inferred count. $68.80 for an 8-GPU instance normalizes to $8.60/GPU-hour. A missing or implausible count skips the listing rather than dividing by an assumed 1.

**Currency.** Recorded, never assumed. A euro-billing provider is recorded in euros with the USD field empty. Conversion happens at calculation, against a published rate (section 6.3).

**Chip identity.** Derived from the label, so a later revision to identification re-applies across all history. Matching is boundary-aware: `B200` never matches inside `GB200`, `H20` never inside `H200`, `A10` never inside `A100`. An unmatched label records no chip and is surfaced for review, never guessed.

### 3.5 Collection mechanics

| Idempotency | Collection runs more frequently than the recording interval and attributes each run to the most recent scheduled observation. If that observation exists, the run exits. First run after each interval collects; later runs are free; a failed run self-heals on the next. |
| --- | --- |
| No backfill | An interval missed before the following one is permanently missing. Late-recorded observations are marked. |
| Time limits | Three bounds per provider: network timeout, hard per-provider limit (a network timeout bounds one operation, not a sequence), overall run budget. Exhaustion records an error. |
| Visible holes | Every configured provider appears in every snapshot: success, error, or unimplemented. Never a silently shorter list. |
| Transport | HTTPS-only including on redirect (a redirect to HTTP raises). Response bodies size-capped, with a wall-clock read limit bounding a slow-drip response a network timeout cannot see. Certificate verification never weakened. |
| Extraction contract | Record the published figure alongside the normalization; state currency explicitly; fail loudly on reading nothing. A page that silently changed shape must error, not present as a healthy provider with no prices. No recipe aggregates across providers. |
| Thin-observation gate | Fewer than 8 of the 28 collected providers read successfully: the observation is discarded and the interval left unclaimed so the next run retries. The gate counts collected providers, not panel members; panel sufficiency is the calculation's minimum-panel rule (section 7.5). Recording a thin snapshot would both stop retries and become the reading the calculation uses. |
| Alarms | A missing previous day, or a previous day missing intervals, is reported on every run until resolved. |

The provider roster: section 11. Per-provider extraction recipes live in the open source collector.

---

## 4. Panel construction

*Outlier-resistant. Transparent.*

### 4.1 Eligibility by source class

Panel-eligible: direct principals, direct-partnered providers, and marketplaces.

Excluded: aggregators, which publish no price of their own, and resellers, whose price is another panel member's capacity under a second brand.

Exclusion is by panel construction, not by filter. The panel is an explicit enumerated list in the SKU document; nothing in the calculation branches on `source_type` or `first_party`, which are disclosure fields.

> **Why marketplaces are in.** Vast.ai and Lium are the only venues on any live panel where the published price is a live transactable ask rather than a posted list rate, which makes them each panel's only direct read on clearing prices. Two disclosures follow: their prices embed a host and platform spread, and they are the only members where manipulation would be transactional rather than reputational, hence the specific prohibition in section 10 (Independence).

### 4.2 Double counting

No two panel members may resolve to the same underlying capacity. This is enforced at panel selection, not at calculation: a reseller that does not disclose its upstream providers cannot be verified as distinct, so such sources are excluded from panels at selection.

Shadeform was excluded from the B300 panel because its only B300 listing republished Verda's capacity. Prime Intellect was excluded because its API returns the upstream provider name directly.

### 4.3 Membership

Membership is a parameter and is negotiated; allocation of liveness weight across members is computed (section 8). Current panels are enumerated in each SKU document. Membership changes, like any parameter change, publish under a new version.

---

## 5. Screening

*Outlier-resistant.*

Collection records everything; the index uses a screened subset. All screens flag; none delete.

| Screen | Rule |
| --- | --- |
| Panel | Fixed enumerated list per SKU (SKU document) |
| Tier | Allow-list: on-demand only. Every other tier (spot, preemptible, reserved, committed, monthly-commit, serverless, from-floor) is recorded but never eligible |
| Product identity | Reject listings whose structured label or stated specifications identify a different product marketed under the SKU's name. The reject set is per-SKU configuration (SKU document) |
| Plausibility | Per-chip price band; out-of-band prices flagged, not deleted |
| Jump | A move of 25% or more, with fewer than 2 other members moving 10% or more, is flagged |

A from-floor listing is a "from $X" starting price rather than a rate for a specific configuration, which is why it is ineligible.

### 5.1 Product identity

Vendors sell materially different products under the same chip string. The screen matches on the structured label and stated specifications only, since descriptive text legitimately mentions related product names. Each SKU document states its reject set and the observed failure that motivated it.

### 5.2 Jump screen

Corroboration distinguishes a market move from a single-provider glitch. B200 fell roughly 30% over three weeks in mid-2026 and passed, because other members moved with it. Where too few providers are comparable for corroboration to be decidable, the screen stands down.

A paired report also shows the move in the specific machine the previous price came from: a machine at an unchanged price beside a book that moved 59% is a selection artifact, not a repricing.

---

## 6. Each provider's price

*Outlier-resistant. Fault-tolerant.*

### 6.1 Default: the lowest eligible listing

A provider's observed price is its lowest listing with a matching chip, an eligible tier, no flags, and a usable price. Foreign-currency prices convert before the minimum is taken.

### 6.2 Order books

A provider publishing an order book needs a different statistic. The distinction is not the source class but what the page publishes.

| Publishes | Statistic | Why |
| --- | --- | --- |
| One list price | Lowest eligible | The minimum selects among that provider's own tiers and configurations |
| A book of third-party asks | Median over the eligible host population | The lowest ask is one host's offer, frequently unverified, sometimes a single machine. It is not representative of the venue |

Two order books are seated across the live panels: Vast.ai and Lium. Which statistic each seat uses is per-SKU configuration (SKU documents). Where the order-book statistic applies, it prices as the volume-weighted median of rentable on-demand per-GPU asks, weighted by each offer's GPU count, under two population conditions the SKU document binds. Population accounting: the stored book must prove the full eligible population was recorded, else the seat is held out rather than pricing a truncated, one-sided-low book; fail closed, deterministic on replay. Population floors, on the H-series panels: a book below the minimum distinct machines, hosts, or sellers holds the seat out with the counts recorded.

> **Why not the lowest ask?** On one real observation the lowest-price rule would have returned an unverified host 29% below the median.

Every other panel member, including those classed `marketplace`, publishes a price list, so the lowest-eligible rule applies to them unchanged.

### 6.3 Currency conversion

Conversion uses the ECB reference rate: public, citable, archived. Each rate is stored on first use and reused permanently, so replays convert at the original rate. Non-publication days walk back to the last published rate, recording its actual date. No rate within seven days: the provider is held out, never converted at a guess.

The ECB publishes once per business day, so every observation within a day converts at the same rate. Intraday movement in a non-USD provider's index contribution therefore reflects its own price only, not exchange-rate drift.

**Ambiguous currency labels fail closed.** Foreign-currency treatment requires a well-formed three-letter code and a native figure; anything else is held out. An apparent billing-currency switch is held out until three consecutive observations confirm it, then old history is discarded and new-currency history starts fresh.

### 6.4 Outlier check

Each observed price is judged against the provider's own recent history, never cross-sectionally. The sigma here is computed over the last 20 observations; it is deliberately separate from the 90-day sigma that prices the votes (section 7.1).

```javascript
accept if |price - mean(last 20)| <= 3.0 * max(sd, 3% of mean(last 20))
```

> **Why never cross-sectionally?** Providers sit at structurally different price levels. A cross-sectional test would flag normal marketplace pricing at every observation.

| Property | Reason |
| --- | --- |
| Rejected prices still enter the history | A genuine repricing then costs one day, not perpetuity |
| The test runs in the quoted currency | On 20 Aug 2026 a roughly 1% EUR/USD move ejected a provider whose price sat unchanged at 7.50 EUR |
| Minimum sigma of 3% of price | A frozen list price has sd 0; without a floor any repricing is rejected |
| Threshold widened from 2.5 to 3.0 sd | At 2.5 it rejected a genuine $5.95 to $6.60 repricing by 1.5 cents. The aggregation in section 7 is now the primary outlier defense; this is a gross-error screen |

The first ten observations pass untested. Such prices are flagged for review if more than 15% from that observation's cross-provider average, but counted.

### 6.5 Documented exclusions

Where a recorded price is known wrong by rule and the true price was never captured, that (observation, provider) pair is excluded by hand with a written reason. Substituting a value would fabricate unobserved data. Exclusions publish with the observation and are then fixed permanently.

---

## 7. Aggregation

*Outlier-resistant.*

### 7.1 Interquantile mean of standard-deviation votes

The index is not a weighted average. It is an interquantile mean over votes: the weighted mean of the central third of the vote mass.

Each passing provider casts its full liveness weight three times: at its price and at its price plus and minus its own standard deviation.

```javascript
for each passing provider i:
    vote (price_i - sd_i)  weight w_i
    vote (price_i)         weight w_i
    vote (price_i + sd_i)  weight w_i

index          = weighted mean of the votes between the 1/3 and
                 2/3 quantiles of cumulative vote weight, with
                 fractional weight at the band edges
stability band = larger distance from the index to the
                 25th / 75th weighted vote percentiles
```

`sd_i` is the provider's own price variability over a trailing 90-day window, with the same 3% floor. This is a longer window than the outlier check's 20 observations, deliberately: the outlier sigma is a fast gross-error screen, while the vote sigma sets how much conviction a provider's votes carry and reflects its longer record.

A stable provider votes tightly and concentrates its influence. A volatile one spreads its votes and dilutes its own. Because only the central third of the vote mass is averaged, votes in the outer thirds have no direct effect on the value, so no small group of providers can drag the index from the tails. The floor stops staleness impersonating conviction: a frozen price would otherwise cast three identical votes claiming certainty it never demonstrated.

> **Why a band instead of the median?** A median depends only on the vote at the midpoint of the mass: nearby votes can move without moving the index until they cross it. Averaging the central third responds to movement throughout the band, so the index tracks the market continuously while the outer thirds still cannot touch it. Shrink the band to nothing and the statistic is the weighted median; widen it to everything and it is the weighted average. One third is the published parameter (section 12.3).

### 7.2 Worked example

Five providers, sigmas floored at 3% of price. Liveness weights sum to 1; each casts them three times, so total vote weight is 3 and the central third is the vote mass between cumulative weight 1 and 2.

| Provider | Weight | Price | sd | Votes |
| --- | --- | --- | --- | --- |
| A | 0.30 | 6.60 | 0.20 | 6.40, 6.60, 6.80 |
| B | 0.25 | 6.75 | 0.21 | 6.54, 6.75, 6.96 |
| C | 0.20 | 6.50 | 0.20 | 6.30, 6.50, 6.70 |
| D | 0.15 | 7.20 | 0.25 | 6.95, 7.20, 7.45 |
| E | 0.10 | 5.80 | 0.45 | 5.35, 5.80, 6.25 |

Sorting all fifteen votes and accumulating weight, the central third runs from cumulative weight 1 to 2 and holds exactly four votes: 6.54 (weight 0.25), 6.60 (0.30), 6.70 (0.20), and 6.75 (0.25). Their weighted mean is the index, **6.6425**. The 25th percentile of vote weight falls at 6.40 and the 75th at 6.80, so the stability band is max(6.6425 - 6.40, 6.80 - 6.6425) = **0.2425**.

Now reprice D, the most expensive member, up 30% to 9.36. The weighted average moves from 6.63 to 6.95. The interquantile mean does not move: D's vote mass already sat in the top third, and pushing it further up changes nothing inside the band.

### 7.3 Robustness

The same experiment as the calculation engine runs it, on the worked example panel (D repriced 7.20 to 9.36, all else fixed):

```javascript
interquantile mean   6.642500 -> 6.642500   (0.000000)
median of votes      6.600000 -> 6.600000   (0.000000)
weighted average     6.627500 -> 6.951500   (+0.324000)
```

The weighted average is still published as a diagnostic. It is not the index.

### 7.4 Stability band

The stability band widens when providers disagree, giving a usable read on how much confidence the published value deserves.

### 7.5 Insufficient providers

Below the minimum passing panel members (a per-SKU parameter; see the SKU documents), the observation publishes explicitly as no index.

---

## 8. Liveness weights

*Outlier-resistant. Transparent.*

### 8.1 Principle

Liveness weights are derived from measured behavior. Membership is negotiated; allocation across members is computed.

A provider earns weight to the extent its recent price movements anticipate subsequent movement in the rest of the panel. Two constraints:

- **Price level is not an input.** Only movement is scored. A cheap marketplace is not penalized for being cheap.
- **No provider is scored against itself.** Evaluation runs against a leave-one-out panel excluding the provider scored.

### 8.2 Returns

The provider's own return uses its quoted currency; the panel's return is in USD and excludes the provider being scored.

```javascript
r_i(a,b)  = clamp log( p_i(b) / p_i(a) )

r^-i(a,b) = clamp log(  Σ_{j≠i} w_j · p_j(b)
                      / Σ_{j≠i} w_j · p_j(a) )
```

Both require real observations at both endpoints; a return spanning a currency change is undefined, never spliced across an exchange rate.

`w_j` is the liveness weight vector published at the sample's own observation τ, held fixed at both endpoints and summed over providers with observations at both, so weight drift and membership churn cannot register as panel movement. The denominators cancel, so the vector needs no normalization. Pinning to the sample's own observation keeps one vector per sample, since weights recompute at every observation.

`clamp` bounds every return at plus or minus 0.5 log (roughly +65% / -39%; the largest genuine repricing observed is around 11%).

> **Why the clamp is load-bearing.** Scoring history deliberately includes prices the section 6.4 check rejected. Without a bound, one absurd-but-real observation would distort every other provider's window without limit.

### 8.3 Signal and outcome

Lookbacks `Δ ∈ {6h, 1d, 2d}`, forwards `h ∈ {6h, 1d, 2d}`. At each sample time τ:

```javascript
signal    X_i(τ)   = [ r_i(τ-Δ, τ) - r^-i(τ-Δ, τ) ]  over Δ

outcome   y_i^h(τ) = r^-i(τ, τ+h)
```

The signal is movement in excess of the panel. Moving with the panel carries no information; moving ahead of it is the whole quantity being measured.

### 8.4 Admissible samples

With `T` = the last observation strictly before the one being priced and `L` = 90 days:

```javascript
τ ≥ T - L      inside the history window
τ + h ≤ T      outcome realized by the cutoff
```

plus a published weight vector at τ itself, and every leg of `X` and `y` computable from real observations. No carry-forward: a stale price is not a price.

`τ + h ≤ T` is the information boundary of section 8.9: `T` is the last observation strictly before the one being priced, so nothing observed at an observation can enter that observation's own weights.

Samples decay exponentially toward the cutoff, half-life 30 days:

```javascript
a(τ) = 2^( -(T - τ) / 30d )
```

### 8.5 Fit and liveness score

Per provider and forward horizon, features are standardized by their `a`-weighted moments over the window, then fitted by ridge regression with an unpenalized intercept:

```javascript
min over α, β:
   Σ a(τ) · ( y(τ) - α - βᵀz(τ) )²  +  λ‖β‖²
   λ = 1.0
```

Standardizing makes `λ` mean the same thing for a 6h feature and a 2d one. A feature with no variation over the window is neutralized to zero, the routine case for a frozen list price.

The liveness score is that fit measured in-sample, against the same weighted measure:

```javascript
R²      = 1 - Σ a(τ)(y - ŷ)² / Σ a(τ)(y - ȳ)²

q_{i,h} = max(0, R²)

Q_i     = mean of q_{i,h} over the three forwards
```

`q_{i,h}` is undefined, not zero, with fewer than 10 samples, if the weighted variance of `y` falls below 1e-12, or if the fit is singular or `R²` non-finite. `Q_i` requires every `q_{i,h}` to be defined, so the score cannot lurch when a longer forward window comes online after a shorter one.

### 8.6 Attendance

A provider's weight also reflects whether it shows up. Each scheduled observation marks every provider: 1 if it was read successfully and produced a usable price, 0 if it was read successfully and produced none (a price rejected by the outlier check of section 6.4 counts as none), and unchanged if our own collection or parsing failed, since a provider is never penalized for our failure.

The attendance factor `A_i` is the exponentially weighted average of this series over the 90-day regression window, with its own attendance half-life, normalized so a provider present throughout has `A_i` = 1. A newly seated provider's scheduled observations before it joined count as 0, so its first print starts near zero; at the 6-hour half-life, sustained printing reaches full attendance in about two days.

The missing print itself is handled by cause:

- Our own collection or parsing failure: the provider's last usable price is carried forward into the observation, and attendance is unchanged. A carried price never advances the provider's own price series, so it enters neither the liveness regression nor the vote sigma, and it never counts toward the minimum passing panel.
- Provider read, no usable price: attendance falls and the consecutive no-price count advances. The provider's last usable price is carried forward and fades as attendance falls.
- Hard cutoff: past 96 consecutive observations without a usable price (24 hours), the provider is excluded entirely until it produces a new usable price. Our own failures never advance the count.

> **Why attendance?** A new provider should not receive full weight from its first print, and a provider that stops publishing should fade rather than vanish instantly or linger stale. The half-life sets the smooth fade during a temporary absence; the hard cutoff removes persistently absent sources. Entry, fade-out, and recovery all happen without per-provider judgment.

Attendance is computed and published with every observation. Attendance weighting is armed: the current version runs at attendance sensitivity η = 0.5, disclosed with every observation as `calc_params.liveness.attendance_eta`. It was armed as a versioned parameter change effective 2026-09-01 (section 12.4).

### 8.7 Allocation

```javascript
s_i = A_i^η · exp(γ Q_i) / Σ_j A_j^η · exp(γ Q_j)
    = softmax_i( γ Q_i + η · log A_i )

w_i = w_min + (1 - N·w_min) · s_i

then, while any w_i > w_max:
    set the violators to w_max and redistribute their
    excess over the remainder in proportion to s_i
```

At η = 0 the attendance term vanishes and the rule is the plain softmax of γQ; the current version runs at η = 0.5.

Every provider receives the floor `w_min` = 2.5% first; the remainder distributes by share. Both bounds are uniform: no per-provider values, and no weight requires a human decision. Allocation runs in full precision and rounds once; the rounded vector is the published liveness weight.

With attendance weighting armed, each provider's ceiling also scales with its attendance: the base ceiling times `A_i`, normalized so the ceilings together can still hold the full weight; if panel attendance is low enough that they cannot, all ceilings expand proportionally just enough to make the allocation feasible. The base floor and ceiling values are unchanged.

`γ` controls how sharply score differences become weight differences. Four providers scoring 0.20 / 0.10 / 0.05 / 0.00:

| γ | Weights |
| --- | --- |
| 0 | 25.0 / 25.0 / 25.0 / 25.0% |
| 1 | 27.6 / 25.2 / 24.1 / 23.1% |
| **4** | **36.2 / 25.1 / 21.0 / 17.7%** |
| 10 | 54.6 / 21.7 / 14.1 / 9.6% |
| 25 | 83.4 / 9.1 / 4.4 / 3.0% |

γ = 4 is a chosen prior, not a fitted value: predictiveness matters visibly, and a good score cannot run away with the panel. Revisiting it is a versioned change.

### 8.8 Opening weights and the switch

`Q_i` requires 10 samples per forward window, so a new panel cannot compute weights on its first observations. Until it can, the index uses the panel's opening weights: the membership weights agreed when the panel was seated, published in each record like any other parameter.

The panel switches to computed weights once and permanently, on the first day both conditions hold:

```javascript
every provider meeting the attendance floor
    has a defined Q_i
and at least 5 providers reported at that observation
```

The first condition stops a brief outage from triggering the switch before that provider has ever been scored. The second stops a thin observation from locking it in. In practice a panel clears both within roughly eight days of collection, so opening weights govern only the opening window. The switch is flagged permanently in the record, since it is a step change in weights with no underlying price movement.

Degenerate cases (too few providers for the floors to fit) publish an even split, flagged.

**Attendance floor.** The first switch condition applies only to providers with a usable price at 50% or more of scheduled observations over the trailing 90 days. Without that qualifier, a provider publishing rarely, often enough to stay inside the window but never often enough to reach 10 samples, would be permanently unscorable and would block the switch indefinitely.

The floor governs the switch test alone. A provider below it stays index-eligible on any day it prints, and post-switch its undefined score allocates zero, so it receives the weight floor. A provider chronically below it keeps receiving the weight floor; removing it from the panel is a membership decision taken at panel review, not by the calculation.

> **Why 50% is deliberately loose.** A defined score needs 10 samples per forward window, and each sample requires the provider present at four offsets (τ-48h, τ-24h, τ-6h, τ). At 50% attendance and a 15-minute cadence that yields roughly 270 samples, at 25% roughly eight, below 10% effectively none. A single contiguous two-week outage leaves attendance near 84%. The floor therefore excuses only providers that could not be scored under any pattern, not a provider having a bad month.

### 8.9 Safeguards

**A uniform ceiling suffices because the aggregation bounds influence.** A manipulated provider that moves and drags others looks identical to one that genuinely leads. The ceiling and the aggregation are sized together: the averaging band is the central third of the vote mass, and the ceiling caps any provider at 30% of it, so even at maximum weight no single provider's votes can constitute the whole band, and a provider whose vote mass sits in an outer third cannot touch the value at all. Ceiling and aggregation together bound single-provider influence without per-provider judgment.

**An observation cannot move its own weights.** Every weight input is realized strictly before the observation being priced. A party listing capacity minutes before an observation cannot move its own weight for it, and each observation's weights are determinable before its prices are read.

---

## 9. Publication and settlement

*Fault-tolerant. Verifiable.*

### 9.1 Cadence

An index value, a stability band, and a liveness weight vector publish for each observation, every 15 minutes. There is no designated fixing time: the series is the 15-minute series, and every value carries the parameter set used to compute it.

### 9.2 Missing observations

An observation with no usable record publishes as an explicit missing-observation entry. Never skipped, never interpolated. An observation carries no index value for one of two reasons: collection failed, or too few providers passed the screens and the observation published dark.

That rule scopes to the index value: the index never invents a value for a missing observation. A missing provider is handled at the provider level (section 8.6), and a carried price is that provider's own last published price, never an invented one, and never counted toward the minimum passing panel.

### 9.3 Settlement and the period rate

Two distinct operations run at two levels. The interquantile mean runs across providers within one observation and produces that observation's index value; it is what stops one provider moving the price. A time-average runs across observations within a period and produces a single figure for that period; it is what stops one print moving a settlement. The index publishes only the 15-minute series; any period aggregation belongs to the contract that references it, and no period length is specified here.

Each of the N observations in a period enters the average at weight 1/N, so a print wrong by D moves the period rate by D/N. At a 15-minute cadence a month is N of roughly 2,920, so a single print at twice its true level moves the period rate by about 0.034%; against a single-point fixing, the same bad print moves settlement by the full 100%.

The period rate is the time-average of the index values within the period, with any missing observation filled from the observations preceding it. Every observation in a gap of G missing observations takes the mean of the last min(G, L) filled observations immediately preceding it, where L = 72 hours (288 observations): the window scales with the gap and caps at three days. The rule applies always, not past a threshold; it draws only on preceding observations (at a panel's genesis, observations with no prior value are dropped from the average instead); and it never feeds the weighting, which requires real observations and handles its own gaps by returning undefined.

> **Why fill from preceding observations?** It invents no price movement. Averaging only the filled observations silently assigns each gap the whole-period average, and interpolating fabricates a path through it.

> **Why average instead of fix?** Averaging over many observations removes any single print's significance, and with it the incentive to influence a particular moment.

### 9.4 Coverage

Whether a period rate is fit to settle on is a separate question, governed by coverage: filled observations divided by scheduled observations. Two thresholds, three bands:

| Band | Coverage | Longest gap | Consequence |
| --- | --- | --- | --- |
| Settles | 98% or more | and 2% of period or less | Period rate stands as computed |
| Review | 90% to 98% | and 2% of period or less | Calculation agent certifies before settlement, recording whether the filled values materially affected the result. The rate is not recomputed |
| Determination | below 90% | or above 2% of period | Calculation agent determines the rate in good faith from the filled series and the coverage report, reasons recorded |

Bands are tested in order and the strictest match governs: any period meeting a Determination condition is a Determination regardless of the other test.

In hours, the 98% and 2% lines are 3 h weekly, 15 h monthly, 44 h quarterly; the 90% line is 17 h, 73 h, 219 h. Ordinary operation settles without review.

The thresholds are recommended contract defaults, not index parameters: the index publishes the 15-minute series and the coverage record, and where a counterparty draws these lines is a contract term. They are stated so a contract has a default to adopt rather than a blank, and so any deviation is visible as one.

The coverage report publishes every period, passing or failing: scheduled and filled observations, coverage, and every gap with its timestamps and cause. A figure that appears only when something has gone wrong is a dispute on first sight. Published values are never revised: this procedure produces a period rate, and the 15-minute series stands as published, holes and all.

---

## 10. Governance and reproducibility

*Reproducible. Verifiable. Transparent.*

| **Versioning** | The full parameter set publishes inside every record. Any parameter change requires a new version; the calculation refuses to extend a series under altered parameters. Prior versions stay frozen and readable. |
| --- | --- |
| **Replay** | Each observation advances from the published record of prior observations, not from re-reading raw observations. Raw records legitimately grow (late uploads, backfilled FX), and re-deriving would rewrite history the series was built on. |
| **Reconciliation** | Published days are re-compared against the raw record on a rolling basis; divergence is reported. The published index stands. The report is for review, not revision. |
| **Retention** | Every observation kept for life of trade + 2 years, available to either party: source URL, timestamp, figure as published, collecting process identity. |
| **Independence** | The calculation agent must be independent of both parties. Neither party nor an affiliate may be a panel provider, nor list capacity on a marketplace panel member so as to influence the price at or near any observation. |

---

## 11. Providers

*Transparent.*

The collector is open source, and each provider's extraction recipe is the authoritative documentation of how that provider is read. Every recipe obeys the extraction contract of section 3.5 and fails loudly rather than guessing; the reasoning behind each pin lives as comments beside the code it protects.

The roster is what the methodology needs from the code: who is collected, its source class, and where it sits. Adding a provider adds a row; a removed provider's row is marked retired, never deleted, so the record behind old observations stays readable.

| Provider | Source class | Panels | Status |
| --- | --- | --- | --- |
| Civo | see published records | H100-SXM, H200-SXM | active |
| Compute Pulse | aggregator | none | active |
| CoreWeave | direct_principal | B200, H100-SXM, H200-SXM | active |
| Crusoe | see published records | H100-SXM, H200-SXM | active |
| Deep Infra | see published records | none | active |
| DigitalOcean | see published records | H100-SXM, H200-SXM | active |
| E2E Networks | see published records | none | active |
| fal | see published records | none | active |
| GMI Cloud | see published records | none | active |
| GPU.ai | see published records | none | active |
| Hot Aisle | direct_principal | none | active |
| Hyperstack | see published records | B300, B200, H100-SXM, H200-SXM | active |
| Lambda | direct_principal | B200, H100-SXM | active |
| Latitude.sh | direct_partnered | B300 | active |
| Lium | marketplace | H100-SXM, H200-SXM | active |
| Massed Compute | see published records | B300, B200, H100-SXM | active |
| Nebius | direct_principal | B300, B200, H100-SXM, H200-SXM | active |
| Oracle Cloud | hyperscaler | none | active |
| OVHcloud | see published records | none | active |
| RunPod | direct_partnered (Secure) | B300, B200, H100-SXM, H200-SXM | active |
| Scaleway | see published records | B300, H100-SXM | active |
| Sesterce | see published records | H200-SXM | active |
| Shadeform | reseller | none | active |
| TensorPool | see published records | H100-SXM, H200-SXM | active |
| Together AI | see published records | B200, H100-SXM, H200-SXM | active |
| Vast.ai | marketplace | B300, B200, H100-SXM, H200-SXM | active |
| Verda | direct_principal | B300, B200, H100-SXM, H200-SXM | active |
| Voltage Park | see published records | H100-SXM | active |

Panel eligibility is by source class (section 4.1); membership is enumerated per SKU (SKU documents). "see published records" means the class is carried as a disclosure field in the published records rather than stated here.

**Vast.ai** is the one provider whose extraction detail is part of the calculation spec, since it feeds the order-book statistic of section 6.2. It is queried without a GPU-count preference and without the API's `verified` filter, which is broken in a way that silently drops the cheapest hosts. Results ordered by instance total truncate the largest totals first, exactly the cheap multi-GPU machines, so a full result page triggers a second query from the other end. Per offer: an integer GPU count 1-16 required (a missing count is never treated as 1, which would record a whole-instance price as per-GPU); an arithmetic check requires per-GPU price times count to reproduce the published total. Offers deduplicated to one per physical machine, keeping the cheapest, ranked by per-GPU price rather than instance total. Ranking by total had buried $6.25/GPU 8-GPU machines beneath $10.94/GPU single-GPU slices of one expensive host.

**Lium** is the second seated order book: a Bittensor-subnet marketplace publishing its entire machine book in one response. It publishes no host-verification field, so no verification screen exists there and none is applied (geography likewise); both are recorded open decisions, and its full extraction record is being documented.

---

## 12. Parameters

*Reproducible. Transparent.*

Every value publishes inside each record. Changes require a new version. Parameters bound per SKU live in the SKU documents.

### 12.1 Collection

| Parameter | Value |
| --- | --- |
| Collection interval | every 15 minutes |
| Index / weight recomputation | every observation |
| Providers collected | 28 |
| Minimum providers to record an observation | 8 of 28 collected (panel-agnostic; section 3.5) |
| Retention | life of trade + 2 years |
| Jump threshold | 25% |
| Corroboration threshold | 10% |
| Corroborators required | 2 |

### 12.2 Period rate

Applies to any period a referencing contract defines; no period length is specified by the index (section 9.3).

| Parameter | Value | Controls |
| --- | --- | --- |
| Fill lookback (L) | 72 h (288 observations) | Cap on the averaging window used to fill a gap; window = min(gap, L) |
| Review threshold, coverage | 98% of scheduled observations | Below it, the period rate is reviewed before certification |
| Escalation threshold, coverage | 90% of scheduled observations | Below it, calculation-agent determination |
| Escalation threshold, longest gap | 2% of period | Above it, calculation-agent determination |

The three rows below the fill lookback are recommended contract defaults, not index parameters; the index computes the same period rate either way (section 9.4).

### 12.3 Index calculation

Values identical across the live SKUs. SKU-bound parameters (minimum panel, FX source and staleness, order-book statistic, product identity reject set, panel membership, series start) are in the SKU documents.

| Parameter | Value | Controls |
| --- | --- | --- |
| Aggregation | interquantile mean of votes | How prices combine |
| Aggregation band | central third of vote mass (1/3 to 2/3 quantiles) | Votes averaged; outer thirds have no direct effect |
| Eligible tiers | on-demand (allow-list) | Only the on-demand rate is the underlying; every other tier is recorded but never eligible |
| History window | 20 | Observations in the outlier test |
| Threshold | 3.0 sd | Acceptance band width |
| Minimum variability | 3% of price | Floor under both sigmas; a frozen price otherwise has sd 0 |
| Warm-up | 10 | Observations before the test applies |
| Vote sigma window | 90 days | Window for the sigma used in votes and the stability band (section 7.1) |
| Test currency | as quoted | Prevents FX moves ejecting a provider |
| Review flag | 15% | Distance from panel average flagging review |
| Currency-change confirmation | 3 | Consecutive observations confirming a switch |

Counts above (history window, warm-up, currency confirmation) are in observations, so their wall-clock span follows the collection cadence: at 15-minute collection the outlier window spans 5 hours, warm-up 2.5 hours, and currency confirmation 45 minutes.

### 12.4 Liveness weighting

Identical on all panels. No per-provider values.

| Parameter | Value | Controls |
| --- | --- | --- |
| Lookback windows | 6 h, 1 d, 2 d | Spans over which a provider's own move is measured |
| Forward windows | 6 h, 1 d, 2 d | Spans over which the panel's response is measured |
| History | 90 days | How far back the relationship is estimated |
| Half-life | 30 days | How fast old observations lose influence |
| Shrinkage penalty | 1.0 | How hard the fit is pulled toward "no relationship"; the main defense against scoring noise as skill |
| Sensitivity (γ) | 4.0 | How sharply score differences become weight differences (section 8.7) |
| Weight floor | 2.5% on every live panel | Minimum weight for an eligible provider, uniform within a panel; set per panel so that N times the floor leaves room for the computed allocation |
| Weight ceiling | 30% | Maximum weight; uniform, no exceptions |
| Attendance half-life | 6 hours | How quickly provider-side missing prints lose influence, and returning providers regain it (section 8.6) |
| Attendance sensitivity (η) | 0.5 | Strength of attendance in allocation; at 0 the softmax is unchanged. Armed at 0.5 by the versioned change effective 2026-09-01 |
| Consecutive no-price limit | 96 observations (24 h) | Hard exclusion after sustained provider-side absence; our own collection failures never count (section 8.6) |
| Minimum observations | 10 | Per provider, per window: observations needed before that provider can be scored. Below it the score is undefined |
| Minimum variation | 1e-12 | The panel must have moved for prediction to be a meaningful question. Below this variance in what is predicted, the score is undefined rather than computed from nothing. Set above price-rounding noise (roughly 1e-14), below any real movement |
| Minimum panel for transition | 5 | Whole-panel, one time: providers that must report at the observation where the index permanently switches to derived weights. Distinct from minimum observations: that asks "does this provider have enough history?", this asks "is this observation broad enough to change the methodology?" |
| Attendance floor | 50% | Share of scheduled observations a provider must reach to gate the transition (section 8.8); distinct from the attendance factor of section 8.6 |
| Movement cap | 0.5 log | Bounds any single observation's influence; largest real move observed roughly 11% |

γ and shrinkage are initial priors, not fitted values. Review is scheduled at roughly 60 days of prints; revisiting either is a versioned change.

---

*Prices, weights, and parameters here are current as of publication and are configuration, not fixed properties of the methodology. Every published record carries the parameter set used to compute it.*

## SKU documents

- SKU: B300
- SKU: B200
- SKU: H100-SXM
- SKU: H200-SXM
- [Methodology Change Log](CHANGELOG.md)
