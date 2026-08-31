# Methodology change log

The methodology change log referenced by [GOVERNANCE.md](GOVERNANCE.md).
Entries record lane mints, retirements, and adopted methodology changes,
with effective dates. Every published record embeds the full parameter set
that produced it; any parameter change touching a published day mints a
new methodology_id, and prior versions stay frozen and readable under
their own keyspaces. Newest first.

<!-- DRAFT, NOT ADOPTED: this entry is the governance record for the
15-minute mint and is submitted for founder approval at PR review
(COM-1455). Delete this comment when it is approved; drop the entry if it
is not. -->

## 2026-08-30 (DRAFT — awaiting founder approval)

- All four public lanes minted a third published version, effective
  `2026-08-30T23:07:28Z`: B300 `annex_a_v0_2_calc_v11`, B200
  `annex_a2_v0_3_calc_v10`, H100-SXM `h100_sxm_v1_calc_v5`, H200-SXM
  `h200_sxm_v1_calc_v5`. Versions 1 and 2 stay frozen and readable under
  their own keyspaces.
- Collection cadence: the observation grid adds a 15-minute era from
  2026-08-28 (96 marks, `slot_minutes_utc`). Every earlier era keeps its
  own marks and its own published bytes; the lanes are minute-keyed from
  this mint on.
- Minimum variability: the absolute 0.05 USD/GPU-hr sigma floor is
  replaced by the percent pair of the 2026-08-27 floor split —
  `filter_sigma_floor_pct` 3% fences outliers, `vote_sigma_floor_pct` 3%
  floors the median-vote band — and the vote sigma is sourced from the
  trailing 90-day dynamic-weights history (`vote_sigma_source`
  `dw_history`) rather than the 20-print outlier window. This changes
  published index values and stability bands, which is why it mints.
- No membership, opening weight, liveness parameter, tier allow-list,
  filter window/sigma/warm-up/terms, manual-verify percent, FX lane,
  manual exclusion or claim floor changed at this mint; each was verified
  identical to the live published record field by field.
- NOT RECORDED HERE, because it is not disclosed upstream and this repo
  will not invent it: versions 2 and 3 also changed the AGGREGATE. Under
  version 1 the published index value is always one of the published
  votes (the weighted median METHODOLOGY.md section 6 describes); under
  versions 2 and 3 it never is, on any observation of any lane. See the
  open blocker on COM-1455 — until the aggregate is disclosed, this log
  cannot record what versions 2 and 3 actually compute.

## 2026-08-29

- Disclosure-only adoption: selected panel rows now retain
  `sku_identifier` and `region` beside the existing `gpu_count_basis` and
  `currency` evidence for downstream receipt projection. No calculation
  input or aggregation changed, no methodology ID was minted, and no
  published index value changed.

## 2026-08-26

- Additive adoption: availability_verified_sources joined calc_params and
  index.availability_verified_weight_share joined published observations
  on all live lanes — a one-time additive adoption under the documented
  grace; disclosure only, never a calculation input.

## 2026-08-23

- B300: hourly panel lane minted under `annex_a_v0_2_calc_v7`
  (config/index_panel_b300.json). The hourly series replays from the
  lane's 2026-08-10 genesis; the record stitches basket-era 4-slot
  snapshots through 2026-08-23 and raw-observatory hourly snapshots from
  2026-08-24T00Z. The daily series is frozen at its final version
  `annex_a_v0_2_calc_v6` (config/index_basket.json) and no longer
  extended; `annex_a_v0_2_calc_v1` through `_v5` are its earlier frozen
  versions.
- B200: hourly panel lane minted under `annex_a2_v0_3_calc_v6`
  (config/index_panel_b200.json), same record stitching cutover, replaying
  from the lane's 2026-08-16 genesis. The daily series is frozen at its
  final version `annex_a2_v0_3_calc_v5` (config/index_basket_b200.json)
  and no longer extended; `annex_a2_v0_3_calc_v1` through `_v4` are its
  earlier frozen versions.
- H100-SXM: panel lane genesis under `h100_sxm_v1_calc_v1`
  (config/index_panel_h100_sxm.json), genesis date 2026-08-23. No
  predecessor series exists for this instrument.
- H200-SXM: panel lane genesis under `h200_sxm_v1_calc_v1`
  (config/index_panel_h200_sxm.json), genesis date 2026-08-23. No
  predecessor series exists for this instrument.
- The broad lanes h100_broad (`h100_broad_v1_calc_v1`) and h200_broad
  (`h200_broad_v1_calc_v1`) share the 2026-08-23 genesis date; they are
  configured lanes, not public SKUs.
