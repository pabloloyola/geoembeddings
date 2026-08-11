# Executable backlog

Select one coherent task at a time. Each completed task must satisfy the
definition of done in `AGENTS.md`.

## P0 — Reproduction and baselines

- [ ] **T0.1** Run and preserve a 50-user learned/baseline comparison.
- [ ] **T0.2** Run and preserve the 500-user reference comparison.
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
