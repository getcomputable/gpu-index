<h1 align="center">Computable GPU Index (CGI)</h1>

<p align="center">
  <a href="https://github.com/getcomputable/gpu-index/actions/workflows/ci.yml"><img src="https://github.com/getcomputable/gpu-index/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-0E6B4F" alt="license: Apache-2.0"></a>
  <a href="LICENSE-DATA.md"><img src="https://img.shields.io/badge/data%20license-CC%20BY--NC%204.0-0E6B4F" alt="data license: CC BY-NC 4.0"></a>
</p>

<p align="center">
An open price index for GPU compute: verifiable, reproducible,
fault-tolerant, outlier-resistant, transparent.
</p>

<p align="center">
  <a href="METHODOLOGY.md">Methodology</a> ·
  <a href="GOVERNANCE.md">Governance</a> ·
  <a href="LICENSE-DATA.md">Data license</a> ·
  <a href="ARCHITECTURE.md">Architecture</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

The Computable GPU Index (CGI) is a USD price per GPU-hour for a specified
accelerator, computed from the published on-demand rental rates of a fixed
panel of providers. Live today: H100, H200, B200, B300.

This repository is the collector and the calculation: the code behind the
published index. Clone it and recompute any print since inception; you will get
the same number we published. The methodology, the collection code, the panel
inputs, and every published value live in the open.

## Where the index is published

- **Live index page**: <https://getcomputable.com/gpu-index>
- **REST API**: `https://api.getcomputable.com/v1/index/`. The latest value, its
  source receipts, and price history. Access is anonymous and read-only: no
  account or API key is required.
  [Quickstart](https://docs.getcomputable.com/quickstart) ·
  [Latest observation](https://docs.getcomputable.com/api-reference/index/latest-observation) ·
  [Price history](https://docs.getcomputable.com/api-reference/index/price-history)
- **Flat-file corpus**: `https://data.getcomputable.com/`. The published
  record as flat files: `latest.json` and the dated day archives.
  A version keyspace holds that version's full re-derivation from its first
  observation, including rows before its effective time; the as-published
  history is the record of what was live.
- **MCP server**: `https://mcp.getcomputable.com/mcp`. The same index through
  three read-only tools, for Claude and other AI clients.
  [Connect with MCP](https://docs.getcomputable.com/mcp-server)

## Reproduce a published value

Every published value can be re-derived end to end from the public record
alone: the attendance factors, the liveness weights, the votes, and the value
itself, never consuming a published intermediate.

```
git clone https://github.com/getcomputable/gpu-index
cd gpu-index
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
./reproduce h100 "$(date -u +%F)"
```

(httpx is the only runtime dependency. The virtual environment keeps the
install local; on systems that allow bare `pip install`, the venv lines are
optional.)

That re-derives the current UTC day's as-published index end to end, including
attendance events and factors, liveness scores, weights, votes, IQM, value, and
stability band. It reads the history advertised by `latest.json` as
`data.versions[].history_path`, selects the version live at each stamp by its
succession `effective_from`, and uses that version's own `v<n>/` lookback history.
Before public launch, it selects the launch version. Each result reports the
version and `methodology_id`; observations before that version's effective time
are labeled `back-calculated`.

The raw inputs are disclosed prices and dispersions, recorded currency and FX,
upstream status, carry basis, filter verdicts, timing, top-level flags, and
`calc_params`. On vote-pre-smoothing generations
(`calc_params.pre_smoothing_half_life_hours`, effective 2026-09-14) each voting
receipt also disclosed `smoothed_vote_usd` — the exact cast price the engine
aggregated — and a fence-rejected row that still voted its booked price carries
a `carried_vote_from` marker; the reproduction votes those disclosed cast prices
rather than rerunning the engine's smoothing state, keeps the raw print as
evidence, and refuses loudly if a voting row omits the disclosure (never a
raw-price fallback). Published attendance factors, liveness scores, and
weights are comparison outputs, never derivation inputs. Every artifact read
is digest verified. Missing required history or withheld raw inputs cause a
full-reproduction refusal. If `history_path` is absent, the command prints a notice and uses
`current_version` (or the legacy flat record when no version pointer exists).

Exit 0: every verifiable value matched. Exit 1 means a mismatch, invalid artifact,
or digest failure; exit 2 means verification could not run or the arguments were
invalid. To check another accelerator or time, run
`./reproduce <h100|h200|b300|b200> YYYY-MM-DD`; add `THH` to check one UTC
hour. A full day takes a few minutes.

`--full` explicitly selects the default. `--receipts` is the fast opt-in: it
recomputes each value and band from that observation's own published receipts,
without re-deriving attendance or weights. In receipts mode only, withheld
contributing receipts degrade to digest-only verification with a notice.

To verify one version's re-derivation throughout, pass `--version <n>`:

```
./reproduce --version 5 h100 2026-09-01
./reproduce --version 6 h100 2026-09-01
./reproduce --receipts --version 5 h100 2026-09-01
```

Rows before the selected version's `effective_from` are labeled
`back-calculated` in both modes.

Recorded 2026-09-08 against https://data.getcomputable.com from a checkout with
no `./data`. This day spans two versions: 76 observations under version 5 and
20 under version 6. Repeated lines and derived source weight vectors are elided.

```
$ ./reproduce h100 2026-09-03
published record: full history via public HTTPS front https://data.getcomputable.com
raw-only full reproduction: prices, dispersions, upstream status, carry basis, filter verdicts, timing, top-level flags, and calc_params are inputs; published derived intermediates are not
H100 2026-09-03T00 derived 3.519725 (band 0.520902) published 3.519725 (band 0.520902) MATCH public digests OK version 5 methodology_id h100_sxm_v1_calc_v8
H100 2026-09-03T00:15 derived 3.519749 (band 0.520978) published 3.519749 (band 0.520978) MATCH public digests OK version 5 methodology_id h100_sxm_v1_calc_v8
[... 73 more MATCH lines under version 5 ...]
H100 2026-09-03T18:45 derived 3.505789 (band 0.605489) published 3.505789 (band 0.605489) MATCH public digests OK version 5 methodology_id h100_sxm_v1_calc_v8
H100 2026-09-03T19 derived 3.508224 (band 0.518224) published 3.508224 (band 0.518224) MATCH public digests OK version 6 methodology_id h100_sxm_v1_calc_v10
[... 18 more MATCH lines under version 6 ...]
H100 2026-09-03T23:45 derived 3.547149 (band 0.646849) published 3.547149 (band 0.646849) MATCH public digests OK version 6 methodology_id h100_sxm_v1_calc_v10
summary: 96 observation(s): 96 MATCH, 0 MISMATCH, 0 degraded
```

`--producer` and `--lane` replay a local collection record.

To compare the latest print with prices visible at its sources now, run
`./reproduce --collect <h100|h200|b300|b200>`. It reports `SAME`, `MOVED`,
`UNREACHABLE`, or `SKIPPED` for every receipt and summarizes the counts.
`MOVED` means the provider price changed since capture, not that the published
record disagrees; marketplace prices move often, so that result is expected.
Only the latest print can be checked this way because past source inputs are not
retroactively observable.

## What is here

| Path | What it is |
|---|---|
| `src/gpu_index/observatory/` | Collection: per-provider price collectors and the snapshot recorder |
| `src/gpu_index/index/` | Calculation: screens, per-provider statistics, the vote-IQM aggregation, liveness weighting |
| `src/gpu_index/published/` | The public record: layout, envelope digests, and the recompute-and-match verifier `./reproduce` runs |
| `src/gpu_index/common/` | Shared primitives: HTTP transport, the object store, slot grids, JSON diffing |
| `scripts/` | The operational entry points: capture, panel compute, period rates, published-record verification |
| `reproduce` | One-command verification of the published record (see above) |
| `tests/` | The suite: unit tests, live-captured fixtures, and the golden artifacts that pin published bytes |
| `config/` | Panels, parameters, and the chip catalog. Parameters are configuration, not code; every published record embeds the parameter set that produced it |
| `METHODOLOGY.md` | The full methodology specification |
| `ARCHITECTURE.md` | The package map, the dependency arrows, and the contributor seams |

Note: `src/gpu_index/index/sources.py` and `composite.py` are the FROZEN
daily lane, retained so the retired daily series stays replayable, not a
live duplication of the observatory collectors.

## Architecture

Four packages, layered `common <- observatory <- index <- published`, with
three declared waiver edges (the vast collector shares the calc lane's
order-book parser; the panel screens share the catalog's label machinery;
the panel config validates carry-forward against the collector's failure
vocabulary).
The package map, the dependency arrows, the frozen-lane rationale, and the
three contributor seams are in [ARCHITECTURE.md](ARCHITECTURE.md);
`tests/unit/test_import_boundaries.py` enforces the arrows in CI.

## Methodology and governance

The full specification, covering the provider panel, the screens, the
aggregation, the liveness weighting, and the versioning rules, is
[METHODOLOGY.md](METHODOLOGY.md). Published values are never revised, and
corrections publish forward under a new methodology version while prior
series stay frozen and readable. How this repository is governed, from code
license permanence to methodology change control and panel membership, is
[GOVERNANCE.md](GOVERNANCE.md).

## Licensing

- Code in this repository: Apache License 2.0 (see [LICENSE](LICENSE)).
- Published index values and data artifacts: CC BY-NC 4.0
  (see [LICENSE-DATA.md](LICENSE-DATA.md)).
- The CGI name and logo are trademarks of Computable and are not licensed by
  either of the above (see [TRADEMARKS.md](TRADEMARKS.md)).

## Contributing

Contributions are welcome under the Developer Certificate of Origin (see
[CONTRIBUTING.md](CONTRIBUTING.md)).

Found a problem? A published value that looks wrong, a misread provider, a
methodology surprise: [open an issue](https://github.com/getcomputable/gpu-index/issues)
or write to team@getcomputable.com.
