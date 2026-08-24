# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Computable
"""Index basket lanes: per-source price capture and the daily composite.

Collectors snapshot each basket source's own price print (raw + normalized
per-GPU) into an append-only keyspace 2-4x daily; the composite CLI then
derives one published index value per day per methodology. Methodology
parameters are pinned per methodology_id — see METHODOLOGY.md."""
