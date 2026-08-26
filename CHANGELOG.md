# Methodology change log

The methodology change log referenced by [GOVERNANCE.md](GOVERNANCE.md).
Entries record lane mints, retirements, and adopted methodology changes,
with effective dates. Every published record embeds the full parameter set
that produced it; any parameter change touching a published day mints a
new methodology_id, and prior versions stay frozen and readable under
their own keyspaces. Newest first.

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
