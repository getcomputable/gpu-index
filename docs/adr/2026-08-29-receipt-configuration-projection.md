# ADR: Preserve receipt configuration as non-calculation evidence

- Date: 2026-08-29
- Status: Accepted
- Issue: COM-1437
- Upstream reference: `getcomputable/forward-curve-pricing` at `c7fa766789bdc21db2bac6d0b4ab6b0b8b2a83d5`

## Context

The vendored hourly panel engine selects an exact provider observation but its
`sources[].chosen` block dropped two retained dimensions: `sku_identifier` and
`region`. It already retained `gpu_count_basis` and `currency`. Public CGI
receipts need this evidence to disclose which configuration underlies a
selected price.

This is a projection defect, not a capture or methodology defect. The public
receipt writer lives outside gpu-index. This repository mirrors the calculation
engine and verifies published values; it must preserve the upstream evidence
while keeping every configuration field outside the calculation.

COM-1437 permits a metadata-only re-projection of pre-launch history before
2026-09-01. A missing historical dimension must remain null. Current provider
pages cannot supply evidence for a historical row.

## Decision

Port the upstream selected-row projection exactly, adapted only for the
`gpu_index` package namespace and repository documentation layout:

- `lowest_eligible_print` copies `sku_identifier` and `region` from the exact
  row that wins the existing minimum-price comparison.
- The existing `gpu_count_basis` and `currency` behavior is unchanged.
- Configuration evidence is forbidden as an input to eligibility, filtering,
  weighting, vote construction, or the weighted-median aggregate.
- Missing evidence stays null. No row may donate configuration to another row.
- The public writer may derive variant or VRAM only from unambiguous recorded
  identifier evidence and may derive provider class only from an explicit
  source lookup.
- Before any historical public write, compare the full ordered vector of
  `(sku, observed_at, status, value, methodology_id)` and abort if any element
  changes.

No new `methodology_id` is minted. The aggregator remains three votes per
source at price minus standard deviation, price, and price plus standard
deviation, followed by the existing weighted median.

## Alternatives considered

1. Re-scrape historical provider pages. Rejected because present inventory is
   not evidence for an earlier observation.
2. Put configuration in `calc_params`. Rejected because descriptive evidence
   does not define the calculation and would create false methodology drift.
3. Join configuration from another row. Rejected because it can synthesize a
   historical configuration that was never observed for the winning price.
4. Change only the public receipt writer. Rejected because new panel artifacts
   would continue dropping the evidence before it reaches that boundary.

## Consequences

- New mirrored panel artifacts preserve the upstream selected-row evidence.
- Existing index values, filters, weights, vote dispersion, and methodology
  identifiers remain unchanged.
- Historical public projection, roster publication, and production backfill
  remain work for the external receipt writer; this mirror commit performs no
  public write.
- Older records may legitimately disclose null dimensions.

## Reversal conditions

Revisit this decision if configuration becomes calculation-bearing, the public
writer no longer receives retained observation evidence, or gpu-index stops
vendoring the upstream panel engine. Calculation-bearing configuration requires
a separately reviewed methodology change.
