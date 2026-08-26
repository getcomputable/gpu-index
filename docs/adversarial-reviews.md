# Adversarial review findings (F-number map)

The pre-publish reviews of the hourly panel lane left numbered findings
(F1..F12) cited throughout the code as `adversarial review F<n>`. This file
is the map from each number to its finding, reconstructed from the code
comments, tests, and config prose that cite it — so a citation like
"adversarial review F6" is a link, not session residue. The accepted
amendments were folded into the panel mints before any observation
published; the full amendment prose lives in the per-lane mint records
(`docs/mints/<lane_id>.md`, the "perf review 2026-08-23" and
"harden review 2026-08-23" blocks).

Numbers with no surviving in-tree citation are marked as such: the finding
either required no code marker or was resolved without one. Nothing is
reconstructed beyond what the tree itself says.

## Panel-lane review series (cited as `adversarial review F<n>`)

| # | Finding (one line) | Cited from |
|---|---|---|
| F1 | Dead-seat visibility: an ok-status seat whose rows ALL screen out was indistinguishable from one with no rows at all — silent ok seats now pin `eligible_rows` plus per-screen `screen_counts`, so a dead seat can never hide again (the vast-H200 label rule was such a provably dead seat; its amendment to `declared` is the OPEN RATIFICATION CALL on that seat's notes). | `src/gpu_index/index/panel.py`, `tests/unit/test_panel_engine.py`, `config/index_panel_h200_sxm.json` (vast seat notes) |
| F2 | Fallback→dynamic switch quorum (A2 amendment): with zero attendance-passers the every-passer-has-Q clause is vacuously true, so one post-outage observation could flip the PERMANENT weight-mode latch on zero attendance information — the quorum now additionally requires a non-empty attendance-passer set (zero passers HOLD the switch). | `src/gpu_index/index/weights.py`, `tests/unit/test_panel_weights.py` |
| F3 | No in-tree citation under this series. (A daily-lane review also numbered a finding F3 — see the daily-lane section below.) | — |
| F4 | No in-tree citation. | — |
| F5 | No in-tree citation. | — |
| F6 | Record quarantine: a poisoned/unparseable record object at an unpublished stamp crashes every firing forever (publish-in-order blocks the lane; earliest-key-wins means a later good snapshot can never shadow it) — the top-level `record_exclusions` key quarantines ONE scheduled (date, hour), which publishes an explicit `record_quarantined` artifact (index null, `observation_missed` false) WITHOUT reading the record; entries ride calc_params and pin per published observation exactly like manual exclusions. | `src/gpu_index/index/panel_config.py`, `src/gpu_index/index/panel.py`, `src/gpu_index/index/period_rate.py`, `scripts/compute_panel_index.py`, `tests/unit/test_panel_cli.py`, `tests/unit/test_panel_engine.py` |
| F7 | False-missed guard: an `observation_missed` artifact is immutable, so the missed verdict is confirmed against ONE fresh re-LIST of the slot keys (never the run cache) before publishing — a transient empty-Contents gateway blip must never pin a permanent false missed record. | `scripts/compute_panel_index.py`, `tests/unit/test_panel_cli.py` |
| F8 | No in-tree citation. | — |
| F9 | Finiteness fail-closed: `json.loads` happily admits NaN/Infinity literals (and bools are numbers to `isinstance` but never prices) — a candidate row must carry a finite native price (and a finite USD price when one is present) or it is excluded and counted (`screen_counts.non_finite_price`); capture-side normalization flags non-finite natives implausible in every currency branch (the harden-review capture sibling). | `src/gpu_index/index/panel.py`, `tests/unit/test_panel_engine.py`, `tests/unit/test_observatory_core.py` |
| F10 | No in-tree citation. | — |
| F11 | No in-tree citation. | — |
| F12 | Unknown-key rejection: a config key the engine does not read would validate and silently do nothing — the exact silently-inert-config class the loader refuses everywhere else (a typo'd `rejected_tokens` would ship a panel with NO identity screen) — so the loader validates against documented schema allowlists and refuses unknown keys loudly. | `src/gpu_index/index/panel_config.py`, `tests/unit/test_panel_engine.py` |

## Named reviews folded into the mints

| Review | Findings (one line each) | Recorded in |
|---|---|---|
| perf review 2026-08-23 | (a) `drift_scan_observations` moved out of calc to a TOP-LEVEL operational key (the bound gates an ops sweep, never the series bytes) and the scan runs on the 16:00Z firing only; (b) replay is BOUNDED to the trailing state window behind the publish frontier, seeded by the latest pre-window artifact, with the manual-exclusion pin check covering published artifacts inside that window only; (c) the ECB feed is fetched only when an unpublished observation day cannot resolve a stored rate after walk-back. | `docs/mints/*.md` (every lane's mint record) |
| harden review 2026-08-23 | (d) the fallback→dynamic switch quorum requires a NON-EMPTY attendance-passer set (= F2); (e) row eligibility fails closed on non-finite prices (= F9); (f) the `record_exclusions` record-quarantine escape hatch (= F6); plus the artifact-visibility hardening — silent ok seats pin `eligible_rows`/`screen_counts` (= F1) and `observation_missed` publishes only after a confirming fresh re-LIST (= F7); and on H200-SXM (g) the vast seat's variant rule amended label→declared (the F1 dead seat), open ratification recorded on the seat's notes. | `docs/mints/*.md`, `config/index_panel_h200_sxm.json` |

## Daily-lane review series (older; cited from the frozen daily lane's tests)

A separate, earlier review of the daily composite lane used its own numbers.
They are NOT the panel series above; the tree cites two of them, plus one
ruling cited by name:

| # | Finding (one line) | Cited from |
|---|---|---|
| F1 (daily) | The raw store can grow AFTER publication (a late upload with an earlier run_id) — the published composite stays the replay authority: history unchanged, loud DRIFT warning, artifact bytes untouched. | `tests/unit/test_index_composite.py` |
| F3 (daily) | The capture coverage line reads the top-level claim-floor knob while the composite floors on the calc knob — the loader must enforce ONE claim floor so the two can never disagree. | `tests/unit/test_b200_composite.py` |
| F1 fail-loudly ruling | An unreachable record source must be "could not verify" (exit 2) with an actionable message — never a traceback, and never exit 0 having verified nothing. | `tests/unit/test_published_reader_cli.py` |

## Unnumbered review rulings still cited in code

Cited as plain `(adversarial review: ...)`; listed here so the idiom stays
linkable:

- Published-stamp recompute refusal: LIST says published, GET (twice) says
  gone → the firing refuses instead of recomputing a published stamp (a
  recompute under any evolved byte-shaping input collides with the
  immutable original and wedges the lane). — `scripts/compute_panel_index.py`,
  `tests/unit/test_panel_cli.py`
- `availability_verified_sources` is a CALC key riding calc_params (an
  unpinned live key turned the store's idempotent re-PUT into
  BucketPublishError after any retune), and the verified share derives from
  the UNROUNDED passing weights (six published 0.166667s sum past 1.0). —
  `src/gpu_index/index/panel.py`, `tests/unit/test_panel_engine.py`
- Period-rate gap accounting: off-grid statuses keys never enter a fill
  window; a multi-stamp boundary gap needs a full filled-context walk; an
  OPEN slot mid-window is not missing; a malformed artifact refuses with
  ERROR + exit 2, never a traceback. — `tests/unit/test_period_rate.py`,
  `tests/unit/test_period_rate_cli.py`
