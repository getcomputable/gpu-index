# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Raw price observatory — wide-net GPU price snapshotting (capture only).

The observatory records raw per-source GPU rental prices for AS MANY chips
and sources as have parseable public surfaces. Since the hourly panel mint
(2026-08-23, METHODOLOGY.md) the stored record IS consumed, READ-ONLY, by
the six panel-index calc lanes -- so a collector or recipe change here CAN
move a contractual panel print and must be treated with basket-lane care.
The original founding point stands: capture gaps can never be backfilled
(a price page cannot be read retroactively), so the historical record
started accumulating before any methodology existed.

Relationship to the basket lanes (gpu_index.index):
  - same lane family, deliberately shared machinery: transport hardening
    (gpu_index.common.http), slot gating (gpu_index.common.slots), and the
    append-only publish discipline (gpu_index.common.store) are IMPORTED,
    never forked;
  - fully separate data: own config, own collectors, own keyspace
    (index/raw_observatory/), own schedule. Nothing here reads or writes a
    basket key, and neither basket lane reads this prefix;
  - the basket collectors stay byte-untouched. NOTE (since 2026-08-23):
    an observatory recipe change CAN move a contractual print -- the
    hourly panel lanes select over this record (read-only; nothing else
    writes this prefix, mechanically fenced on both sides).
"""
