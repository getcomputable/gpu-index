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
- **MCP server**: `https://mcp.getcomputable.com/mcp`. The same index through
  three read-only tools, for Claude and other AI clients.
  [Connect with MCP](https://docs.getcomputable.com/mcp-server)

## Reproduce a published value

Every published value can be recomputed from its own receipts.

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

That reproduces the current UTC day: every H100 observation published so far,
recomputed from the published per-provider receipts and matched against the
published value, with every artifact digest verified. Exit 0 means every value
and digest matched, 1 means a mismatch, and 2 means nothing could be verified.
To check another accelerator or time, run
`./reproduce <h100|h200|b300|b200> YYYY-MM-DD`; add `THH` to check one UTC hour.

A successful run looks like this, recorded live against
https://data.getcomputable.com on 2026-08-31 at 18:31 UTC. The transcript is
left exactly as recorded, with the repeated middle lines elided.

```
$ ./reproduce h100 "$(date -u +%F)"
published record: H100/v4/observations/2026/08/31.json via public HTTPS front https://data.getcomputable.com
digest OK: 2e406eef6a178179cd260c24a2d17706bad7280daa8e9e4b203710e14a532626
H100 2026-08-31T00 recomputed 3.463378 (band 0.563078) published 3.463378 (band 0.563078) MATCH digest OK
H100 2026-08-31T00:15 recomputed 3.463379 (band 0.563079) published 3.463379 (band 0.563079) MATCH digest OK
[... 70 more MATCH lines ...]
H100 2026-08-31T18 recomputed 3.530037 (band 0.519443) published 3.530037 (band 0.519443) MATCH digest OK
H100 2026-08-31T18:15 recomputed 3.530067 (band 0.519526) published 3.530067 (band 0.519526) MATCH digest OK
summary: 74 observation(s): 74 MATCH, 0 MISMATCH, 0 degraded
[... non-fatal weight-window warning; the recompute-and-match above is unaffected ...]
```

Run `./reproduce` with no arguments for the other modes, which replay a local
collection record rather than the published one.

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
two declared waiver edges (the vast collector shares the calc lane's
order-book parser; the panel screens share the catalog's label machinery).
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
