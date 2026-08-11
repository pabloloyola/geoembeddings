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
  missing-service robustness explicitly pending under P1A.

### P1A — Complete the deterministic robustness framework (R6, R7)

- [ ] Extend the existing `src/geoembeddings/robustness.py` implementation and
  `geoembed robustness` command with GPS perturbation, timestamp jitter,
  leave-one-service-out views, and recent-history truncation. Use shared,
  deterministic view identifiers so learned and baseline reports have matched
  coverage, and report representation-drift and frozen-downstream-degradation
  curves.

### P1B — Spatial and transfer evaluation (R2, R8)

- [ ] Add held-out-region, unseen-geohash/cell, and later-time slices; add
  distance-aware retrieval and geohash-boundary pair consistency; and report
  explicit label and user coverage for every slice and metric.

### P1C — Temporal and routine diagnostics (R3, R4)

- [ ] Add hour/day probes, duration-related tasks, periodic retrieval, and
  repeated-routine-versus-one-off-episode tests. If the current simulator cannot
  support a valid schedule-shift test, declare that test blocked rather than
  substituting an observational proxy.

### P1D — Reliability and offline efficiency (R10, R13)

- [ ] Add window/event resampling variance, reliability-error curves, and
  coverage-risk curves; benchmark training throughput, batch latency, peak
  memory, export throughput, and artifact size.

## P1 — Simulator counterfactual support

- [ ] **T1.11 (R5, R7) — Staged matched-counterfactual program.** There is no
  valid matched-counterfactual baseline artifact yet; the current independent
  scenario runs do not establish identity preservation. This program touches
  the simulator, dataset manifest/contract/layout, truth-side validation, and
  evaluator, but must not change the observed-data inputs available to modeling
  commands. Complete the stages in order:
  1. Introduce named, independent random streams in
     `src/geoembeddings/simulator.py` for world/POI generation, user latents,
     episodes, choice noise, and observation noise.
  2. Through the centralized `contract.py` and `layout.py` path/schema
     contract, record every stream seed and stable object identity in each
     run's `manifest.json`.
  3. Define and version a pair-manifest schema that identifies the reference
     run, intervention run, invariant objects, changed parameters, source
     hashes, and user/time/object matching keys.
  4. Add pair-integrity validation proving that users, latent preferences,
     world objects, and required episodes are byte-identical or identical on
     explicitly enumerated fields across an exposure-only pair. Report precise
     mismatches and fail before calculating representation metrics.
  5. Add exposure, opportunity, and observation interventions one at a time.
     Each intervention must declare the exact fields and objects allowed to
     change; all other declared invariants must pass pair-integrity validation.
  6. Implement evaluator-side user/time matching and report match coverage,
     persistent-trait invariance, representation drift, and downstream-task
     degradation separately for each intervention. Compare baseline and learned
     exports made from the paired runs with matching source hashes, cutoffs,
     users, and matching keys.

  Keep latent values, invariant/changed-field declarations, intervention truth,
  and pair-integrity evidence under `truth/`; `prepare`, `baseline`, `train`,
  and `export` must continue to consume only `observed/`. **R5 remains
  non-executable until both identity-preservation validation and the paired
  evaluator pass their contract, boundary, and integration tests.**
- [ ] **T1.15 (R11)** Add temporary-trip and sustained-preference-change scenarios.

## P2 — Factorized models

P2 must not begin until all of the following entry gates are satisfied:

- the 500-user reference is archived and the post-reference decision is recorded
  under T0.2/T0.2a;
- the R1/R4 episode metrics are executable;
- the selected R5/R6/R7 invariance tests required to test the documented model
  hypothesis are executable; and
- collapse diagnostics are present, including separation, temporal retrieval,
  centered effective rank, and task-information reporting.

Once the entry gates pass, complete the work in this order:

- [ ] **T2.1 (R1, R4)** Define a typed multi-component encoder output contract
  with explicit `persistent`, `context`, and `combined` components.
- [ ] **T2.2 (R1, R4)** Add a configuration-driven model registry behind the
  existing CLI while preserving `SingleVectorEncoder` and its current behavior.
- [ ] **T2.3 (R1, R4)** Define a versioned multi-component export schema with
  explicit component names, dimensions, field order, source hashes, and
  backward compatibility with existing single-vector exports.
- [ ] **T2.4 (R1, R4)** Implement a capacity-matched persistent/context encoder.
- [ ] **T2.5 (R1, R4)** Add a capacity-matched single-vector control and
  persistent-only, context-only, fusion, and loss ablations.
- [ ] **T2.6 (R1, R4, R5, R6, R7)** Add branch-specific objectives and
  branch-specific evaluation reporting.
- [ ] **T2.7 (R1, R4, R5, R6, R7)** Require a matched comparison demonstrating
  that the persistent and context branches improve their intended requirement
  axes without collapse or unacceptable regression on the other reported axes.

The routine branch remains blocked until P1C can distinguish recurring routine
from both persistent identity and temporary episode state. Do not add it to the
P2 encoder merely because the persistent/context path is complete.

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
- [ ] **T4.4 (R13)** Add online incremental update latency benchmarks.
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
