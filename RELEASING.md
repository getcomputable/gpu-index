# Releasing

Checklist for cutting a release of `gpu-index`. Every item must hold before
tagging.

1. **Full test suite green**, including the contract guards:
   `pytest` passes with the golden-artifact tests
   (`tests/unit/test_golden_artifact.py` for the daily engine and
   `tests/unit/test_panel_golden_artifact.py` for the hourly panel
   engine, byte-identity against `tests/golden/`) and the public-API
   surface test
   (`tests/unit/test_public_api.py`) untouched or deliberately updated in
   the same diff.
2. **Signature compatibility**: `griffe check gpu_index --search src
   --against <previous release tag>` reports no breakages, or the release
   is a major version bump and the breakages are listed in the release
   notes. (CI runs the same check against the PR base on every pull
   request.)
3. **Value-affecting changes are minted**: any change that alters what the
   pipeline would publish requires a new methodology version. See
   GOVERNANCE.md ("Methodology changes"): every artifact embeds its full
   parameter set, and the calculation mechanically refuses to extend a
   series under drifted parameters. A minted change ships with the
   regenerated golden artifact(s) in the same diff.
4. **Upstream duty (maintainers)**: before any breaking release, run the
   private downstream consumer's test suite against the release candidate.
   A breaking release does not ship until that suite is green or the
   breakage is coordinated with its owners. The engine and collector
   sources were last re-synced to the private upstream at pin
   a783285e48a5 (2026-08-26; previous pin 071b0049).
5. **Golden regeneration is explicit and local only**: regenerate with
   `GPU_INDEX_UPDATE_GOLDEN=1 pytest tests/unit/test_golden_artifact.py
   tests/unit/test_panel_golden_artifact.py` (with `CI` unset), then
   review the golden diff like code. A golden is never written implicitly
   and never regenerated on CI.
6. **Tag and signed release notes**: tag the release commit, and publish
   release notes signed by a maintainer covering API changes, methodology
   mints, and data-license notes (LICENSE-DATA.md) where relevant.
7. **Disclosure window covers the weighting lookback**: the published
   window must expose at least `MIN_DISCLOSURE_WINDOW_DAYS` (100 = 90d
   lookback + 2d forwards + slack; `src/gpu_index/published/verify.py`)
   of observable history per lane, or the README's stated bound and the
   day-mode verifier warning must remain in place saying exactly what a
   shorter window withholds.
