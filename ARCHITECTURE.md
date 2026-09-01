# Architecture

Four packages under `src/gpu_index/`, layered bottom-up:

| Package | Role |
|---|---|
| `common/` | Shared primitives: HTTP transport with the project's honest User-Agent (`http.py`), the S3-compatible/local/HTTPS store (`bucket.py`, `store.py`), slot grids (`slots.py`), JSON diffing (`jsondiff.py`) |
| `observatory/` | Collection: one collector per provider (`sources/<source_id>.py`, auto-discovered), the chip catalog and label normalization (`catalog.py`), the capture loop and snapshot recorder (`collect.py`, `snapshot.py`, `store.py`) |
| `index/` | Calculation: screens and per-provider statistics (`screens.py`, `panel.py`), the vote-IQM aggregate (`composite.py`), liveness weighting (`weights.py`), panel config loading (`panel_config.py`), period rates and reports |
| `published/` | The public record: file layout + envelope digest (`artifacts.py`), the digest-verifying reader (`reader.py`), recompute-and-match verification (`verify.py`) |

## Dependency arrows

```
common  <--  observatory  <--  index  <--  published
```

Every package may import `common`. `index` consumes the observatory's
record and its label machinery; `published` re-derives published values
with the index engine's own vote math. Three edges are declared waivers
rather than layering accidents:

1. `observatory/sources/vast.py` imports `gpu_index.index.composite`
   (`us_ca_verified_host`) and `gpu_index.index.sources`
   (`parse_vast_offers`, `VAST_POPULATION_LIMIT`): the capture side runs
   the same verified-host screen and order-book parser the calc side
   prices with.
2. `index` imports `gpu_index.observatory.catalog` (label normalization
   and the token-boundary matcher) so a token means the same thing to the
   catalog and to the panel screens.
3. `index/panel_config.py` imports `gpu_index.observatory.collect`
   (`VALID_FAILURE_KINDS`) so the carry-forward failure scope
   (METHODOLOGY.md section 8.6) is validated against the ONE
   classification vocabulary the capture lane records with — an error
   string is not a contract, and a second copy of the kinds list would
   drift silently.

The observatory and index packages deliberately share single-home
machinery in both directions (the catalog's label normalization feeds the
panel screens; the vast order-book parser feeds both capture lanes): we
accept one documented bidirectional package edge over forked copies of
identity-critical code.

These are the ONLY sanctioned cross-package edges;
`tests/unit/test_import_boundaries.py` fails on any new one.

## Contributor seams

1. **Add a collector** — one new file in
   `src/gpu_index/observatory/sources/` plus a source entry in
   `config/raw_observatory.json` and a fixture-based test; see
   CONTRIBUTING.md ("Add a collector in 4 steps"). Merging a collector
   does not seat a provider on a panel (GOVERNANCE.md).
2. **Panel membership and parameters** —
   `config/index_panel_<lane>.json`, loaded and validated by
   `src/gpu_index/index/panel_config.py`; parameters are configuration
   under the refuse-to-extend fence. Changes here are governance
   decisions, not ordinary code review.
3. **Verify the published record** — `./reproduce` drives
   `scripts/verify_published_record.py` over `src/gpu_index/published/`;
   anything that touches the published contract must keep the
   recompute-and-match and digest checks green against the fixtures in
   `tests/fixtures/published/`. The verifier owns fail-closed statistic
   dispatch: an unsupported declared aggregate raises the public
   `UnsupportedStatisticError` before any degraded verdict can return.
