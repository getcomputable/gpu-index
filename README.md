# Computable GPU Index (CGI)

An open price index for GPU compute, built on five properties: verifiable,
reproducible, fault-tolerant, outlier-resistant, transparent. Open collection
code, open aggregate values, open methodology, reproducible by anyone.

CGI publishes a USD price per GPU-hour for specified accelerators, computed from
published on-demand rental rates of a fixed, disclosed panel of providers. The
methodology, the collection code, the panel inputs, and every published value live
in the open. See [METHODOLOGY.md](METHODOLOGY.md) for the full specification.

**Status: pre-launch.** Live panels: B300 (since 2026-08-10), B200 (since
2026-08-16), H100-SXM and H200-SXM (since 2026-08-23). Additional accelerators
are collected ahead of panel seating.

## Reproduce a published value

```
git clone https://github.com/getcomputable/gpu-index
cd gpu-index
pip install -e .
./reproduce h100 2026-08-24
```

(httpx is the only runtime dependency.)

`./reproduce <h100|h200|b300|b200> <date>` verifies the published record:
recompute the index from the published per-provider inputs and weights and
match it exactly; verify every file's digest. A UTC day (YYYY-MM-DD) covers all of that
day's observations, and YYYY-MM-DDTHH targets a single one. It reads the record
from a local downloaded copy (GPU_INDEX_DATA_DIR, default ./data) when it holds
the requested day, otherwise straight from the public front: the official
record host https://data.getcomputable.com by default; set
GPU_INDEX_PUBLIC_BASE_URL to point at another copy. The record's layout at
that host: latest.json (the newest observation per lane),
observations/YYYY/MM/DD.json (one UTC day, all lanes), and
series/{24h,7d,30d,90d}.json (aggregate history rows). For every
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

A successful run looks like this (captured live against
https://data.getcomputable.com):

```
$ ./reproduce h100 2026-08-25
published record: observations/2026/08/25.json via public HTTPS front https://data.getcomputable.com
digest OK: 5f573fbb2dfaa01479172177d10bf8c77f82582f0d71457dc1de8442b6c4cbe9
H100 2026-08-25T00 recomputed 3.645759 (band 0.505759) published 3.645759 (band 0.505759) MATCH digest OK
H100 2026-08-25T01 recomputed 3.645759 (band 0.505759) published 3.645759 (band 0.505759) MATCH digest OK
H100 2026-08-25T02 recomputed 3.645759 (band 0.505759) published 3.645759 (band 0.505759) MATCH digest OK
[... 20 more hourly MATCH lines ...]
H100 2026-08-25T23 recomputed 3.704254 (band 0.564254) published 3.704254 (band 0.564254) MATCH digest OK
summary: 24 observation(s): 24 MATCH, 0 MISMATCH, 0 degraded
```

(While a lane's published history is shorter than 100 days the run also
prints the non-fatal weight-window warning described below.)

Published values are never revised, and corrections publish forward under a
new methodology version while prior series stay frozen and readable. The
internal producer-record replay (deriving unpublished observations from the
raw collection record) remains available via `./reproduce --producer <sku>
<date>`; the retired daily series via `./reproduce --frozen b300|b200 <date>`;
and configured non-SKU panel lanes via `./reproduce --lane <panel_id>`.

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

## Conflict of interest statement

CGI is published by Computable, which operates a GPU compute marketplace. That is
a conflict we manage by construction rather than by asking for trust: the panel,
weights, parameters, and inputs are all published; the calculation is
deterministic and replayable; methodology changes require a new version and the
calculation refuses to extend a series under altered parameters. Computable's own
venue prices are not panel inputs.

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
| `docs/` | Supporting design documents, per-lane mint records (`docs/mints/`), and [docs/architecture.md](docs/architecture.md) |

Note: `src/gpu_index/index/sources.py` and `composite.py` are the FROZEN
daily lane, retained so the retired daily series stays replayable — not a
live duplication of the observatory collectors.

## Architecture

Four packages, layered `common <- observatory <- index <- published`, with
two declared waiver edges (the vast collector shares the calc lane's
order-book parser; the panel screens share the catalog's label machinery).
The package map, the dependency arrows, the frozen-lane rationale, and the
three contributor seams are in [docs/architecture.md](docs/architecture.md);
`tests/unit/test_import_boundaries.py` enforces the arrows in CI.

## Licensing

- Code in this repository: Apache License 2.0 (see [LICENSE](LICENSE)).
- Published index values and data artifacts: CC BY-NC 4.0 plus the CGI Derived
  Index Grant (see [LICENSE-DATA.md](LICENSE-DATA.md)).
- The CGI name and logo are trademarks of Computable and are not licensed by
  either of the above (see [TRADEMARKS.md](TRADEMARKS.md)).

## Contributing

Contributions are welcome under the Developer Certificate of Origin (see
[CONTRIBUTING.md](CONTRIBUTING.md)). The code in this repository is Apache-2.0
permanently; see [GOVERNANCE.md](GOVERNANCE.md).

Found a problem? A published value that looks wrong, a misread provider, a
methodology surprise: [open an issue](https://github.com/getcomputable/gpu-index/issues)
or write to index@getcomputable.com.

Contact: index@getcomputable.com
