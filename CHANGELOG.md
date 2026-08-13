# Changelog

## 2026-08-13 — 0.5.0 release-candidate audit

- Audited the stable-release gate as a synthetic research harness release. The
  candidate is **not stable**: the required locked CPU environment could not be
  constructed because the package-index tunnel was unavailable and the locked
  `pygments==2.20.0` wheel was not cached. No failed gate is treated as passing.
- Kept package version `0.5.0`. This audit changes release documentation only;
  it does not change the package API, dataset contract, simulator, model, or
  evaluator. Advancing the version would misrepresent an unverified candidate.
- CUDA and Apple MPS were not tested. Historical accelerator results are not
  evidence for this candidate.
- Release scope remains limited to a synthetic research harness. This audit
  makes no selected-candidate, privacy-certification, external-validity, or
  production-readiness claim.

The candidate record and unresolved gates are documented in the
[verification ledger](docs/VERIFICATION.md#release-candidate-audit-2026-08-13-utc).
