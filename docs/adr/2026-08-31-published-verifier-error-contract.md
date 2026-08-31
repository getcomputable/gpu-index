# ADR: Published verifier unsupported-statistic error contract

- Status: Accepted
- Date: 2026-08-31
- Ticket: COM-1451

## Context

The public verifier exposes `recompute_observation(observation)` through the `gpu_index.published` package. Until COM-1451 it did not inspect the observation's declared `calc_params.aggregation`; it always ran the one aggregate it happened to implement. A future public statistic could therefore be reported as recomputed even though the verifier had used different math.

COM-1451 requires the verifier to refuse loudly with a named, tested error when a published observation declares a statistic the verifier does not implement. The repository also pins package and module interfaces in `tests/unit/test_public_api.py`, so naming and exporting an exception is a deliberate public-interface change.

## Decision

Add `UnsupportedStatisticError` as a subclass of `PublishedRecordError` in the `gpu_index.published.verify` module and re-export it from `gpu_index.published`.

`recompute_observation` validates the public wire value `calc_params.aggregation == "median_stddev_votes"` before inspecting receipts or returning a digest-only degradation. Callers that already catch `PublishedRecordError` remain compatible; callers that need to distinguish an unsupported calculation can catch the narrower class. No function signature changes.

This keeps the seam deep and local: callers learn one additional error mode, while the verifier owns statistic dispatch, validation order, and engine adaptation behind its existing function interface.

## Alternatives considered

1. Raise the existing `PublishedRecordError` with a distinctive message. Rejected: callers would have to parse prose, and it would not satisfy the ticket's named-error contract.
2. Return a new `ObservationCheck` verdict such as `unsupported`. Rejected: unsupported math is not a verification result, and adding a fourth verdict would invite callers to treat it as a nonfatal mismatch or degradation.
3. Put the subclass in `gpu_index.published.artifacts`. Rejected: that module owns envelope/key/digest validity. The unsupported-statistic condition belongs to the recomputation module; placing it there preserves locality.
4. Silently map unknown names to the current engine. Rejected: this is the failure mode COM-1451 closes.

## Consequences

- Existing broad catches continue to work because the new error subclasses `PublishedRecordError`.
- The package gains one public name, pinned by the public-interface test.
- Validation must precede withheld-source degradation so disclosure policy cannot bypass the declared-calculation check.
- The one currently supported public wire name is explicit and distinct from the engine's internal `median_ci_votes` name.

## Verification

- A fixture with an unknown aggregation raises the exact subclass and names the unsupported value.
- The same assertion holds when a contributing source is withheld.
- Existing v1 fixtures using `median_stddev_votes` remain all-MATCH.
- The public-interface allowlist includes the new error at both module and package seams.

## Reversal conditions

Revisit the one-error design if the public projection adds multiple calculation families whose configuration validation or result types materially differ. At that point, introduce a small statistic-adapter registry behind `recompute_observation`; keep the exception as the stable unsupported-dispatch outcome. Remove the public exception only in a separately versioned breaking release.
