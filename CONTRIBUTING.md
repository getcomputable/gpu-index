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

## Add a collector in 4 steps

1. **One file**: `src/gpu_index/observatory/sources/<source_id>.py`
   defining `SOURCE_ID` (must equal the module name) and
   `collect(timeout=..., options=None)`. Discovery is automatic — adding
   a source is adding one file, no registry to edit.
2. **One config entry**: add the source to `config/raw_observatory.json`'s
   `sources` list (`source_id`, `display_name`, `source_type`,
   `first_party`, `notes`). Only listed sources are ever invoked.
3. **Fixture + test**: record a trimmed live response under
   `tests/fixtures/observatory/<source_id>/` and write
   `tests/unit/test_observatory_source_<source_id>.py` following the
   runpod exemplar's house style: parse the recorded fixture, pin exact
   prints for a few known rows including this source's edge cases, prove
   the framework normalization maps its real labels, and prove
   present-but-unusable values are counted, never silently dropped.
4. **Run the suite**: `pip install -e ".[dev]" && pytest`.

Remember: merging a collector does NOT seat the provider on a published
panel (see rule 2 above).

## Where things live

The README's "What is here" table maps the tree, and
[docs/architecture.md](docs/architecture.md) has the package layering,
the sanctioned cross-package edges, and the three contributor seams.

## Practicalities

- Collectors must follow the extraction contract in METHODOLOGY.md section 3.5:
  fail loudly rather than guess, record the published figure alongside the
  normalization, state currency explicitly, HTTPS only. Collectors also send
  the project's honest User-Agent (a repo convention, not a methodology rule).
- Run the test suite before opening a PR. Every collector has a fixture-based
  test; new collectors need one.
- Keep fixtures minimal: the smallest page excerpt that exercises the recipe.
