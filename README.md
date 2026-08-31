<h1 align="center">Computable GPU Index (CGI)</h1>

<p align="center">
  <a href="https://github.com/getcomputable/gpu-index/actions/workflows/ci.yml"><img src="https://github.com/getcomputable/gpu-index/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-0E6B4F" alt="license: Apache-2.0"></a>
  <a href="LICENSE-DATA.md"><img src="https://img.shields.io/badge/data%20license-CC%20BY--NC%204.0-0E6B4F" alt="data license: CC BY-NC 4.0"></a>
</p>

<p align="center">
An open price index for GPU compute -- verifiable, reproducible,
fault-tolerant, outlier-resistant, transparent.
</p>

<p align="center">
  <a href="METHODOLOGY.md">Methodology</a> ·
  <a href="GOVERNANCE.md">Governance</a> ·
  <a href="LICENSE-DATA.md">Data license</a> ·
  <a href="ARCHITECTURE.md">Architecture</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

The Computable GPU Index (CGI) publishes a USD price per GPU-hour for
specified accelerators, computed from published on-demand rental rates of a
fixed, disclosed panel of providers. The methodology, the collection code, the
panel inputs, and every published value live in the open.

Live panels: B300 (since 2026-08-10), B200 (since
2026-08-16), H100-SXM and H200-SXM (since 2026-08-23). Additional accelerators
are collected ahead of panel seating.

## Reproduce a published value

```
git clone https://github.com/getcomputable/gpu-index
cd gpu-index
pip install -e .
./reproduce h100 2026-08-25
```

(httpx is the only runtime dependency.)

A successful run looks like this, captured live against
https://data.getcomputable.com on 2026-08-25, when the record published one
observation per hour. The record now publishes one every 15 minutes, so a
current day prints 96 observations rather than 24. The transcript is left
exactly as recorded.

```
$ ./reproduce h100 2026-08-25
published record: observations/2026/08/25.json via public HTTPS front https://data.getcomputable.com
digest OK: 5e92cc88c42c0b6c8c58ad497add5ef3aadcfe6792706faeb1a9f8404b203590
H100 2026-08-25T00 recomputed 3.645759 (band 0.505759) published 3.645759 (band 0.505759) MATCH digest OK
H100 2026-08-25T01 recomputed 3.645759 (band 0.505759) published 3.645759 (band 0.505759) MATCH digest OK
H100 2026-08-25T02 recomputed 3.645759 (band 0.505759) published 3.645759 (band 0.505759) MATCH digest OK
[... 20 more hourly MATCH lines ...]
H100 2026-08-25T23 recomputed 3.704254 (band 0.564254) published 3.704254 (band 0.564254) MATCH digest OK
summary: 24 observation(s): 24 MATCH, 0 MISMATCH, 0 degraded
```

(While a lane's published history is shorter than 100 days the run also
prints the non-fatal weight-window warning described below.)

## What you can verify

`./reproduce <h100|h200|b300|b200> <date>` verifies the published record:
recompute the index from the published per-provider inputs and weights and
match it exactly; verify every file's digest. A UTC day (YYYY-MM-DD) covers all of that
day's observations, and YYYY-MM-DDTHH narrows to the observations inside
that hour (four, at the 15-minute cadence). It reads the record
from a local downloaded copy (GPU_INDEX_DATA_DIR, default ./data) when it holds
the requested day, otherwise straight from the public front: the official
record host https://data.getcomputable.com by default; set
GPU_INDEX_PUBLIC_BASE_URL to point at another copy. `latest.json` is the
version-free pointer: it names the newest published observation per lane. Day
files live at `observations/YYYY/MM/DD.json` and rolling windows at
`series/{24h,7d,30d,90d}.json`. The reader also accepts a per-SKU versioned
layout (`<sku>/v<n>/...`) so a future methodology succession can publish
alongside the current series without breaking existing readers. For every
observation the verifier prints the recomputed value next to the published value with a
MATCH or MISMATCH verdict: each published observation carries the per-provider
receipts (price, standard deviation, liveness weight, status) its value and
stability band were computed from, and the verifier rebuilds the same
weighted-median-of-votes aggregate from exactly those inputs. Every file also embeds a digest of its own canonical content,
which is recomputed and checked on every read. Exit 0 means everything
matched; exit 1 means a mismatch or a digest failure; exit 2 means it could
not verify (the record source is unreachable, or nothing is published for the
requested date). It never exits 0 without verifying. Where the published
disclosure policy withholds a provider's recent prices, the affected
observation says so and is verified by digest only.

Three further modes replay a LOCAL collection record rather than the
published one, so they need a populated `./data` directory and do nothing from
a fresh clone: `./reproduce --producer <sku> <date>`, the retired daily series
via `./reproduce --frozen b300|b200 <date>`, and configured non-SKU panel lanes
via `./reproduce --lane <panel_id>`.

One bound on what the public record alone can re-derive: liveness weights are
fitted over a 90-day history of samples whose forward outcomes extend up to 2
days past each sample, so re-deriving a day's weight vector from published
observations needs at least 100 days of published history before it (90 + 2,
plus slack for window edges). Per-observation recompute-and-match is
unaffected, because it consumes only the observation's own receipts, which
embed the weights as published. But while a lane's observable published
window is shorter than 100 days, the weight vector itself is verifiable
against the record only as far back as the window reaches; the day-mode
verifier prints a non-fatal warning when that is the case.

## What is here

| Path | What it is |
|---|---|
| `src/gpu_index/observatory/` | Collection: per-provider price collectors and the snapshot recorder |
| `src/gpu_index/index/` | Calculation: screens, per-provider statistics, the median-of-votes aggregation, liveness weighting |
| `src/gpu_index/published/` | The public record: layout, envelope digests, and the recompute-and-match verifier `./reproduce` runs |
| `src/gpu_index/common/` | Shared primitives: HTTP transport, the object store, slot grids, JSON diffing |
| `scripts/` | The operational entry points: capture, panel compute, period rates, published-record verification |
| `reproduce` | One-command verification of the published record (see above) |
| `tests/` | The suite: unit tests, live-captured fixtures, and the golden artifacts that pin published bytes |
| `config/` | Panels, parameters, and the chip catalog. Parameters are configuration, not code; every published record embeds the parameter set that produced it |
| `METHODOLOGY.md` | The full methodology specification |
| `ARCHITECTURE.md` | The package map, the dependency arrows, and the contributor seams |

Note: `src/gpu_index/index/sources.py` and `composite.py` are the FROZEN
daily lane, retained so the retired daily series stays replayable — not a
live duplication of the observatory collectors.

## Architecture

Four packages, layered `common <- observatory <- index <- published`, with
two declared waiver edges (the vast collector shares the calc lane's
order-book parser; the panel screens share the catalog's label machinery).
The package map, the dependency arrows, the frozen-lane rationale, and the
three contributor seams are in [ARCHITECTURE.md](ARCHITECTURE.md);
`tests/unit/test_import_boundaries.py` enforces the arrows in CI.

## Methodology and governance

The full specification -- the provider panel, the screens, the aggregation,
the liveness weighting, the versioning rules -- is
[METHODOLOGY.md](METHODOLOGY.md). Published values are never revised, and
corrections publish forward under a new methodology version while prior
series stay frozen and readable. How this repository is governed -- code
license permanence, methodology change control, panel membership -- is
[GOVERNANCE.md](GOVERNANCE.md).

## Conflict of interest statement

CGI is published by Computable, which operates a GPU compute marketplace. That is
a conflict we manage by construction rather than by asking for trust: the panel,
weights, parameters, and inputs are all published; the calculation is
deterministic and replayable; methodology changes require a new version and the
calculation refuses to extend a series under altered parameters. Computable's own
venue prices are not panel inputs.

## Licensing

- Code in this repository: Apache License 2.0 (see [LICENSE](LICENSE)).
- Published index values and data artifacts: CC BY-NC 4.0 plus the CGI Derived
  Index Grant (see [LICENSE-DATA.md](LICENSE-DATA.md)).
- The CGI name and logo are trademarks of Computable and are not licensed by
  either of the above (see [TRADEMARKS.md](TRADEMARKS.md)).

## Contributing

Contributions are welcome under the Developer Certificate of Origin (see
[CONTRIBUTING.md](CONTRIBUTING.md)).

Found a problem? A published value that looks wrong, a misread provider, a
methodology surprise: [open an issue](https://github.com/getcomputable/gpu-index/issues)
or write to team@getcomputable.com.
