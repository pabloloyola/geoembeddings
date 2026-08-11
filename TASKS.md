# Executable backlog

Select one coherent task at a time. Each completed task must satisfy the
definition of done in `AGENTS.md`.

## P0 — Reproduction and baselines

- [x] **T0.1 — Execution/contract smoke verified.** The 50-user, 7-day
  learned pipeline, statistical baseline, and matched comparison completed
  successfully. This milestone verifies execution and cross-stage contracts,
  not scientific model quality; the commands, seed, coverage, and limitations
  are recorded in `docs/VERIFICATION.md`.
- [ ] **T0.1a — Register the complete smoke artifact manifest.** Preserve an
  index of every artifact produced by T0.1 using immutable repository-relative
  paths or external-storage identifiers. Generated binaries do not have to be
  committed to Git. The index must record the source commit and dataset
  manifest hash and state that the baseline and learned reports use the same
  preparation contract.
- [ ] **T0.2 — Immediate next task: archive the 500-user scientific reference.**
  Run the baseline-versus-learned comparison at the fixed 500-user reference
  scale and preserve or externally register all of the following outputs:
  - resolved simulation and embedding configurations;
  - source hashes, simulator/model seeds, and evaluation cutoffs;
  - learned checkpoint and training report;
  - baseline and learned cutoff exports;
  - baseline and learned episode reports;
  - baseline and learned event-removal reports;
  - comparison JSON and Markdown.

  The reference must include an artifact index with immutable paths or external
  storage identifiers, the source commit, the dataset manifest hash, and an
  explicit statement that the baseline and learned reports share the same
  preparation contract. Completion of this milestone, rather than the smoke
  run, establishes the archived scientific baseline for subsequent work.
- [ ] **T0.2a (R1, R3, R4, R7) — Record the post-reference model decision.**
  Immediately after archiving T0.2, write a decision record that compares the
  statistical baseline and single-vector GRU **separately** on every axis below:
  - persistent-trait probe R² and category-preference probe R²;
  - incremental preference information beyond geography and event volume;
  - same-user stability and different-user separation;
  - temporal retrieval and centered effective rank;
  - episode coherence, boundary response, and post-episode recovery;
  - event-removal drift and probe degradation;
  - next-event performance and known-label coverage.

  Draw a conclusion for each axis; do not name or imply an aggregate winner.
  Based on those per-axis conclusions, the record must choose exactly one next
  action: (1) repair or ablate the single-vector baseline, (2) finish the
  remaining evaluator gate, or (3) begin the persistent/context factorized
  model. Record the artifact paths, artifact and source hashes, seed, cohort
  size, cutoffs, limitations, and rationale in `docs/CURRENT_STATUS.md` or a
  dedicated versioned decision document.
- [ ] **T0.3** Add environment/runtime metadata to training and comparison reports.
- [ ] **T0.4** Add majority/popularity next-event baselines and class-balance metrics.

## P1 — Evaluator foundations

- [x] **T1.1 (R1, R4, R11)** Add dense timestamped embedding export with no truth labels.
- [x] **T1.2 (R1, R4)** Add evaluator-side episode-boundary joins and response curves.
- [x] **T1.3 (R7)** Add deterministic event-removal robustness views. The
  configuration-driven `robustness` command uses matched learned/baseline masks,
  reports sparse coverage and frozen-probe degradation, and keeps GPS and
  missing-service robustness explicitly pending under T1.4/T1.5.
- [ ] **T1.4 (R7)** Add GPS-noise and timestamp-jitter robustness views.
- [ ] **T1.5 (R6, R7)** Add leave-one-service-out encoding/evaluation.
- [ ] **T1.6 (R8)** Add explicit held-out-region and unseen-cell slices.
- [ ] **T1.7 (R2)** Add distance-aware retrieval and geohash-boundary pairs.
- [ ] **T1.8 (R3)** Add hour/day, duration, routine, and periodicity probes.
- [ ] **T1.9 (R10)** Add window-resampling stability and reliability diagnostics.
- [ ] **T1.10 (R13)** Add runtime, throughput, memory, and artifact-size benchmarks.

## P1 — Simulator counterfactual support

- [ ] **T1.11 (R5)** Split simulator randomness into named independent streams.
- [ ] **T1.12 (R5)** Emit pair manifests for matched counterfactual runs.
- [ ] **T1.13 (R5)** Preserve users/world/episodes while intervening on exposure.
- [ ] **T1.14 (R5, R7)** Preserve latents while intervening on opportunity/observation.
- [ ] **T1.15 (R11)** Add temporary-trip and sustained-preference-change scenarios.

## P2 — Factorized models

- [ ] **T2.1 (R1)** Define a versioned multi-component embedding export schema.
- [ ] **T2.2 (R1)** Implement a capacity-matched persistent/context encoder.
- [ ] **T2.3 (R1)** Add persistent-only, context-only, and fusion ablations.
- [ ] **T2.4 (R1, R4)** Add branch-specific training objectives.
- [ ] **T2.5 (R3, R4)** Implement routine branch only after routine tests exist.
- [ ] **T2.6** Add configuration-driven model registry without breaking current CLI.

## P3 — Recommendation contract and ranking

- [ ] **T3.1 (R9)** Define POI catalog schema and version bump/migration.
- [ ] **T3.2 (R9)** Define request, availability, impression, and interaction schemas.
- [ ] **T3.3 (R9)** Extend Hakone POIs and request-time attributes.
- [ ] **T3.4 (R9)** Implement popularity, nearest, and category-preference rankers.
- [ ] **T3.5 (R9)** Implement frozen-embedding candidate ranker.
- [ ] **T3.6 (R5, R9)** Add exposure-aware training and counterfactual evaluation.
- [ ] **T3.7 (R8, R9)** Add unseen-region and unseen-POI recommendation slices.

## P4 — Responsible deployment evidence

- [ ] **T4.1 (R10)** Calibrate representation uncertainty.
- [ ] **T4.2 (R11)** Quantify adaptation and forgetting under change points.
- [ ] **T4.3 (R12)** Add membership and sensitive-attribute inference audits.
- [ ] **T4.4 (R13)** Add online update/export latency benchmarks.
- [ ] **T4.5** Document simulator calibration and external-validity limits.

## Task template

Copy this into a work note or pull-request description:

```text
Task:
Requirements affected:
Hypothesis:
Baseline artifact:
Changed contract/modules:
Information-boundary review:
Tests added:
Commands run:
Artifacts produced:
Metric deltas:
Regression axes:
Limitations:
Next decision:
```
