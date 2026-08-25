# Computable GPU Index (CGI)

An open price index for GPU compute: open collection code, open aggregate values,
open methodology, reproducible by anyone.

CGI publishes a USD price per GPU-hour for specified accelerators, computed from
published on-demand rental rates of a fixed, disclosed panel of providers. The
methodology, the collection code, the panel inputs, and every published value live
in the open. See [METHODOLOGY.md](METHODOLOGY.md) for the full specification.

**Status: pre-launch.** Live panels: B300 (since 2026-08-10), B200 (since
2026-08-16), H100 and H200 (since 2026-08-23). Additional accelerators are
collected ahead of panel seating.

## Reproduce a published value

```
git clone https://github.com/getcomputable/gpu-index
./reproduce h100 2026-08-24
```

`./reproduce <h100|h200|b300|b200> <date>` verifies the published record:
recompute the index and weights from the published inputs and match them
exactly; verify every file's digest. A UTC day (YYYY-MM-DD) covers all of that
day's observations, and YYYY-MM-DDTHH targets a single one. It reads the record
from a local downloaded copy (GPU_INDEX_DATA_DIR, default ./data) or straight
from the public front (set GPU_INDEX_PUBLIC_BASE_URL), and for every
observation prints the recomputed value next to the published value with a
MATCH or MISMATCH verdict: each published observation carries the per-provider
receipts (price, dispersion, weight, status) its value was computed from, and
the verifier rebuilds the same weighted-median-of-votes aggregate from exactly
those inputs. Every file also embeds a digest of its own canonical content,
which is recomputed and checked on every read. Exit 0 means everything
matched; exit 1 means a mismatch or a digest failure. Where the published
disclosure policy withholds a provider's recent prices, the affected
observation says so and is verified by digest only.

Published values are never revised, and corrections publish forward under a
new methodology version while prior series stay frozen and readable. The
internal producer-record replay (deriving unpublished observations from the
raw collection record) remains available via `./reproduce --producer <sku>
<date>`; the retired daily series via `./reproduce --frozen b300|b200 <date>`;
and configured non-SKU panel lanes via `./reproduce --lane <panel_id>`.

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
| `src/gpu_index/index/` | Calculation: screens, per-provider statistics, the median-of-votes aggregation, dynamic weighting |
| `config/` | Panels, parameters, and the chip catalog. Parameters are configuration, not code; every published record embeds the parameter set that produced it |
| `METHODOLOGY.md` | The full methodology specification |
| `docs/` | Supporting design documents |

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

Contact: index@getcomputable.com
