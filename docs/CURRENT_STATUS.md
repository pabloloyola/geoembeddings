# Current status

## Release

- Package/repository version: `0.5.0`
- Behavioral lineage: tested v0.4.x simulator and modeling code
- v0.5.0 changes: agent handoff, research protocol, roadmap, command reference,
  task backlog, literature guide, and locked dependencies
- Dataset contract: `geoembeddings-dataset/1.0`

## Working capabilities

- Semi-synthetic Kanto simulation across five controlled scenarios
- Structural and deep simulator validation
- Observed/truth information boundary
- Leakage-safe global temporal preparation
- Statistical history-vector baseline
- MPS-safe single-vector GRU training
- Learned embedding export at three cutoffs
- Observed-only dense timestamped export for both the statistical baseline and
  learned encoder, with configurable event stride, history counts, explicit
  field-order metadata, and no protected truth labels
- Protected episode evaluation that joins dense timestamps to half-open truth
  intervals and reports within/adjacent-episode coherence, boundary change,
  response curves, drift/recovery, a held-out intent probe, and collapse checks
- Versioned deterministic GPS, timestamp, leave-one-service-out, and recent-truncation robustness views with matched baseline/learned masks, coverage, cosine drift, realized corruption, and frozen-probe degradation
- Protected latent probes, next-event evaluation, and fair frozen
  baseline-versus-learned comparison

The historical release verification recorded in `docs/VERIFICATION.md` reports
13 passing tests and a 50-user, 7-day learned pipeline plus baseline comparison
completed end to end. That is a dated verification record, not the current test
inventory. It marks the **execution/contract smoke as passed**; it does not mark
the **500-user scientific baseline as archived**.

## Phase 0 evidence milestones

| Evidence state | Status | Meaning |
|---|---|---|
| 50-user execution/contract smoke passed | verified | The learned and baseline paths completed under one preparation contract, as recorded in `docs/VERIFICATION.md`; the small cohort is not scientific evidence. |
| Complete smoke artifact manifest registered | pending | Record immutable paths or external-storage identifiers, source commit, dataset manifest hash, and the shared-preparation-contract statement; large generated binaries need not be committed. |
| 500-user scientific baseline archived | pending — immediate next task | `docs/artifacts/t0.2-reference500.json` is registered, but its audit records no immutable location or durable external identifier and marks every required artifact missing. Run, archive, and re-index the matched reference. |
| T0.2a post-reference action | pending | `docs/decisions/t0.2a-reference-decision.md` selects exactly `finish the evaluator gate`, but every per-axis conclusion remains pending because T0.2 provenance and reports are absent. It declares no aggregate winner and does not authorize factorization. |

## Requirement status

Status describes evaluation coverage, not whether a model has satisfied the
scientific requirement. `partial` means at least one required axis is
executable while other named axes remain absent. Artifacts are relative to
`EXPERIMENT_DIR` unless stated otherwise.

| ID | Status | Responsible command | Artifact or present limitation |
|---|---|---|---|
| R1 | partial | `evaluate`; `evaluate --episodes`; `compare` | `evaluation.json`, `episode_response.json`, and `comparison/embedding_comparison.json`; factorized persistent/context outputs and matched change scenarios pending |
| R2 | partial | `train`; `evaluate` | `model/training_report.json` and `evaluation.json` contain next-geohash metrics; distance retrieval and boundary-pair evaluation pending |
| R3 | partial | `train`; `evaluate` | `model/training_report.json` and `evaluation.json` cover future-time next-event prediction; routine, periodicity, and duration probes pending |
| R4 | partial | `export-dense`; `evaluate --episodes`; `compare` | `dense_embeddings.npz`, `episode_response.json`, and matched episode deltas in `comparison/embedding_comparison.json`; factorized persistent/context separation and matched change scenarios pending |
| R5 | pending | — | Matched counterfactual exposure/opportunity evaluator and artifacts pending |
| R6 | partial | `robustness`; `compare` | Leave-one-service-out drift, coverage, and frozen-probe degradation are executable; candidate-aware cross-service transfer remains pending |
| R7 | partial | `robustness`; `compare` | `robustness/{kind}_robustness.json` covers deterministic GPS, timestamp, and recent-truncation views; real-noise calibration and causal invariance remain unmeasurable |
| R8 | partial | `prepare`; `evaluate` | Chronological split metadata in `prepared/prepared_metadata.json` and future-time metrics in `evaluation.json`; explicit geographic/POI holdouts pending |
| R9 | blocked | — | Observable recommendation request, candidate, impression, and interaction contract is absent |
| R10 | pending | — | Uncertainty evaluator and artifact pending |
| R11 | pending | — | Matched transient/sustained change scenarios and adaptation evaluator pending |
| R12 | pending | — | Privacy audit harness and artifact pending |
| R13 | pending | — | Efficiency benchmark harness and artifact pending |

The base `evaluate` command writes the backward-compatible three-cutoff report
and does not run the R4 or R7 supplemental evaluators. Run `export-dense`
followed by `evaluate --episodes` for episode evidence, and run `robustness` for
event-removal evidence. The base report names these supplemental commands and
artifacts rather than incorrectly marking their implemented axes as pending.

## Pending scientific capabilities

- Factorized persistent/context and routine/context representation and metrics
- Matched temporary-versus-sustained change scenarios
- Explicit geographic holdout experiments
- Matched counterfactual exposure/opportunity invariance
- Recommendation request/impression/interaction data contract
- Candidate-aware ranking and Tokyo-to-Hakone evaluation
- Representation uncertainty, nonstationarity, privacy, and efficiency audits

## Evidence limitations

Episode response, drift, recovery, intent prediction, or coherence from a
single vector is **not evidence of persistent/context disentanglement**. R4 is
partial because its episode metrics are executable, not because the current GRU
has separated persistent preference from episode state. Matched baseline and
learned reports, collapse diagnostics, and future factorized outputs are still
required for that claim.

The historical 50-user, 7-day smoke comparison remains an execution and
contract check only. Its small held-out sample and incomplete fine-geohash
coverage support no scientific model-quality, episode-coherence, robustness,
or disentanglement claim.

## Immediate instruction

Complete T0.2 before treating the reference as evidence. Generate the 500-user
matched baseline-versus-learned run, preserve same-run cutoff and dense exports,
episode and robustness reports, and comparison JSON/Markdown, then replace each
missing entry in `docs/artifacts/t0.2-reference500.json` with an immutable local
location or durable external identifier and hash. Record source commit,
simulator manifest hash, seeds, cohort, cutoffs, observed-source hashes,
categorical field order, users, and preparation identity; the comparability
audit must pass. Only then populate the T0.2a axes and reconsider its checkbox.
