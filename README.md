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

That verifies the current UTC day's published receipts and artifact digests.
By default it reads the as-published keyspace advertised by `latest.json` as
`data.versions[].history_path`. Until that field is available, it prints a
notice and falls back to `current_version`. Each result names the version
and `methodology_id` verified.

Exit 0: every verifiable value matched; withheld receipts degrade to digest-only
verification with a notice. Exit 1 means a mismatch, invalid artifact, or digest
failure; exit 2 means verification could not run or the arguments were invalid.
To check another accelerator or time, run
`./reproduce <h100|h200|b300|b200> YYYY-MM-DD`; add `THH` to check one UTC
hour. `--receipts` explicitly selects this default receipts check.

To verify a particular version's re-derivation, pass `--version <n>`:

```
./reproduce --receipts --version 5 h100 2026-09-01
./reproduce --receipts --version 6 h100 2026-09-01
./reproduce --full --version 5 h100 2026-09-01
```

Explicit-version observations before that version's `effective_from` are labeled
`back-calculated`. `--full` derives attendance events and factors, liveness
scores, weights, votes, and values from raw disclosed history without consuming
published derived intermediates. It stays within one version and takes a few
minutes; pass `--version <n>`. During the cutover, `--full` without a version
retains the current-version fallback only when `history_path` is absent.

Recorded 2026-09-01 under version 5; reproduce it with
`./reproduce --full --version 5 h100 2026-09-01`. Taken live against
https://data.getcomputable.com at 07:29 UTC and left exactly as recorded; the
repeated middle lines and each observation's derived 16-source weight vector
are elided.

```
$ ./reproduce h100 "$(date -u +%F)"
published record: full history via public HTTPS front https://data.getcomputable.com
raw-only full reproduction: prices, dispersions, upstream status, carry basis, filter verdicts, timing, top-level flags, and calc_params are inputs; published derived intermediates are not
H100 2026-09-01T00 derived 3.456577 (band 0.556277) published 3.456577 (band 0.556277) MATCH public digests OK
H100 2026-09-01T00:15 derived 3.457619 (band 0.557319) published 3.457619 (band 0.557319) MATCH public digests OK
[... 26 more MATCH lines, each followed by its derived weight vector ...]
H100 2026-09-01T07 derived 3.465561 (band 0.565261) published 3.465561 (band 0.565261) MATCH public digests OK
H100 2026-09-01T07:15 derived 3.465393 (band 0.565093) published 3.465393 (band 0.565093) MATCH public digests OK
summary: 30 observation(s): 30 MATCH, 0 MISMATCH
```

Run `./reproduce --receipts` for the receipts-only value check; `--producer`
and `--lane` replay a local collection record rather than the published one.

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
