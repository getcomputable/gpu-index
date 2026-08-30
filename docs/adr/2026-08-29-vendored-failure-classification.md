# ADR: Port the vendored failure-classification interface

- Date: 2026-08-29
- Status: Accepted
- Issue: COM-1430
- Upstream reference: `getcomputable/forward-curve-pricing` at `03cd4ee89ebf8414fe37a37f05f017ad80c029d9`

## Context

gpu-index vendors fcp's observatory collection engine but had none of the machine-readable failure vocabulary introduced upstream. Every failure was only an `error` string, so consumers could not distinguish parse refusals, fetch failures, deadline abandonment, and sources skipped after the total capture budget was spent. COM-1425 exposed the drift when its Hyperbolic acceptance test could not retain the upstream `failure_kind == "parse"` assertion.

This change adds public names to `gpu_index.observatory.collect`, which triggers the repository's architecture-review gate. The HTTP body-cap and slow-drip guards also need a typed exception: without it their existing `RuntimeError` is indistinguishable from a parser's fail-closed refusal.

## Decision

Port fcp's classification interface and implementation exactly. Adapt only the package seam from `basket.http.TransportError` to `gpu_index.common.http.TransportError`, retain gpu-index's SPDX header, and retain local multiline formatting.

`gpu_index.common.http` remains the transport seam. Its body-cap and slow-drip refusals raise `TransportError`, a `RuntimeError` subclass that preserves existing catch behavior. Shadeform's both-hosts-down fallback raises the same type from its final fetch error. `gpu_index.observatory.collect` owns classification because it already owns every collection error envelope, and `gpu_index.observatory.snapshot` conditionally persists that classification. The new class and function names are intentionally public and are added to the public-surface tripwire.

## Alternatives considered

1. Parse error strings in consumers. Rejected because prose is not a stable contract and cannot reliably separate transport from parsing.
2. Classify inside each collector. Rejected because it would duplicate policy across every source and allow collectors to drift independently.
3. Add a new classification module or external dependency. Rejected because the existing HTTP and collection modules are the natural seams; a third module would add interface with no leverage.
4. Port unrelated current fcp HTTP changes at the same time. Rejected by COM-1430's instruction not to improve either copy while porting.

## Full `collect.py` drift inventory

The pre-port full-file diff found:

1. All semantic drift is the failure-classification change: the `TransportError` import, four `FAILURE_KIND_*` constants, `VALID_FAILURE_KINDS`, `DeadlineExceeded`, `classify_failure`, the typed deadline raise, and `failure_kind` on budget/exception envelopes.
2. gpu-index has an Apache-2.0 SPDX/copyright header that upstream does not.
3. gpu-index wraps the `call_with_deadline` signature while upstream currently formats it on one line.
4. There is no other drift in control flow, defaults, documentation, or output shape.

After this port, the remaining full-file differences are the repository-local header, formatting, and the required import namespace adaptation. They are non-behavioral.

The review also found two classification companion hunks outside `collect.py`: fcp types Shadeform's both-hosts-down wrapper as `TransportError`, and its snapshot builder conditionally preserves `failure_kind`. Both are ported so the classification remains correct and reaches the persisted record.

## Consequences

- Persisted capture failures gain a stable `parse | fetch | timeout | budget` field; successful results remain byte-shape compatible because they do not gain the key.
- Existing callers catching `RuntimeError` continue to work because both new exception types subclass it.
- The bounded explicit-cause walk recognizes domain wrappers without treating unrelated implicit exception context as a transport failure.
- gpu-index still relies on manual vendoring. Automating the sync policy is intentionally deferred to separate architecture work.

## Reversal conditions

Revisit this decision if gpu-index stops vendoring the observatory engine, if the repositories adopt a generated shared contract, or if failure classification moves to a versioned external schema. Until then, changes to this vocabulary must be synchronized deliberately and checked with a full-file diff.
