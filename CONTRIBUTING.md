# Contributing to CGI

Thank you for contributing. Two rules keep this simple:

## 1. Developer Certificate of Origin (DCO)

Every commit must be signed off:

```
git commit -s
```

The sign-off certifies the Developer Certificate of Origin v1.1
(https://developercertificate.org/): that you wrote the contribution or have
the right to submit it under this repository's license. All submissions are
accepted under Apache-2.0 with DCO sign-off and no other terms. Pull requests
with unsigned commits fail the DCO check and cannot merge.

We do not use a CLA. For an unusually large contribution (for example a whole
subsystem) we may ask for a one-time signed statement of provenance.

## 2. What contributions can and cannot change

- Code improvements, new provider collectors, tests, and documentation are
  welcome.
- Merging a collector does NOT add that provider to a published panel. Panel
  membership is a governance decision with published criteria, separate from
  code review (see GOVERNANCE.md and METHODOLOGY.md).
- Changes that would alter any published value require a new methodology
  version. The calculation refuses to extend a series under altered
  parameters; do not try to work around that.

## Practicalities

- Collectors must follow the extraction contract in METHODOLOGY.md Appendix A:
  fail loudly rather than guess, record the published figure alongside the
  normalization, state currency explicitly, honest User-Agent, HTTPS only.
- Run the test suite before opening a PR. Every collector has a fixture-based
  test; new collectors need one.
- Keep fixtures minimal: the smallest page excerpt that exercises the recipe.
