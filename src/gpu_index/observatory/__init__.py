# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Raw price observatory — wide-net GPU price snapshotting (capture only).

The observatory records raw per-source GPU rental prices for AS MANY chips
and sources as have parseable public surfaces. No composite or index value
is derived here — the point is that capture gaps can never be backfilled
(a price page cannot be read retroactively), so the historical record
starts accumulating before any methodology consuming it exists.

Relationship to the basket lanes (gpu_index.index):
  - same lane family, deliberately shared machinery: transport hardening
    (gpu_index.common.http), slot gating (gpu_index.common.slots), and the
    append-only publish discipline (gpu_index.common.store) are IMPORTED,
    never forked;
  - fully separate data: own config, own collectors, own keyspace
    (index/raw_observatory/), own schedule. Nothing here reads or writes a
    basket key, and neither basket lane reads this prefix;
  - the basket collectors stay byte-untouched — an observatory recipe
    change can never move an index print.
"""
