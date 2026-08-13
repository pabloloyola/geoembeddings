# Current status

External-validity claim boundaries, accepted/unavailable evidence distinctions,
and the evidence needed to change every requirement status are maintained in
[`EXTERNAL_VALIDITY.md`](EXTERNAL_VALIDITY.md).

## Release and contracts

- Package/repository version: `0.5.0`
- Current dataset contract: `geoembeddings-dataset/2.0`
- Explicit compatibility: event-only `geoembeddings-dataset/1.0` runs remain
  readable by modeling commands. Readers do not invent the recommendation
  tables that only exist in 2.0.
- Current learned surfaces: the legacy single-vector GRU, a typed component
  export contract, and the implemented `factorized_pc` plus capacity/branch
  ablation variants. Implementation is not a scientific factorization result.

Contract 2.0 adds the observed POI catalog, recommendation requests,
impressions, and interactions. Protected utilities, probabilities, episode
state, and counterfactual outcomes remain under `truth/`. The compatibility
rule is intentionally one-way: new code can consume an event-only 1.0 run for
the old modeling path, but recommendation consumers must require 2.0 tables.

## Working capabilities

- Semi-synthetic Kanto simulation, structural/deep validation, and a strict
  observed/truth boundary.
- Leakage-safe preparation, a statistical baseline, the MPS-safe single-vector
  GRU, and configuration-selected persistent/context model variants.
- Versioned cutoff and dense component exports with named `persistent`,
  `context`, and `combined` branches; legacy single-vector exports have an
  explicit reader compatibility path.
- Protected episode, temporal/routine, spatial-transfer, robustness, and
  train-only next-event diagnostics, with matched baseline/learned comparison.
- Immutable matched simulation for exposure, opportunity, observation,
  temporary-trip, sustained-preference, and schedule-shift interventions;
  pair manifests, field-level pair integrity, paired representation evaluation,
  and protected change evaluation are executable.
- A 2.0 observed recommendation contract populated with a synthetic Hakone POI
  catalog, requests, availability/impressions, interactions, and request-time
  attributes, plus observed-only popularity, nearest-POI, and
  category-preference rankers with versioned predictions and metric reports.
- T3.5 frozen-embedding candidate ranking, T3.6 observed-only exposure-aware
  training with integrity-gated protected regret/probability-recovery
  evaluation, and T3.7 observed-only frozen seen/unseen region/POI and
  early/late slices. T3.7 has no truth input and cannot report utility regret.
- Seeded three-cutoff reliability/repeatability evaluation plus T4.1a
  held-out-user window-bootstrap calibration for immutable diagnostic controls,
  and an observed-only frozen-export/offline-evaluation benchmark. No selected
  candidate exists, so selection-dependent calibrated uncertainty and
  hardware-normalized online/training benchmarks remain unavailable.
- The T4.2 nonstationarity audit authenticates matched no-change, temporary,
  and sustained change reports and reports adaptation, recovery, forgetting,
  permanent drift, uncertainty, censoring, exclusions, and coverage separately.
  It currently provides diagnostic-control evidence only because no
  representation has the `selected_candidate` role.

The historical 50-user, 7-day verification in `docs/VERIFICATION.md` remains a
dated execution smoke, not scientific evidence. Current authoritative
completion and gate records live in `TASKS.md`.

## Phase 0 evidence milestones

| Evidence state | Status | Meaning |
|---|---|---|
| 50-user execution/contract smoke | verified | The original learned and baseline paths completed under one preparation contract; the small cohort supports no scientific claim. |
| Artifact-index workflow | complete | The indexer authenticates current base and supplemental reports, exports, robustness views, benchmark, lineage, and comparability fields. |
| Historical T0.2 reference | closed: evidence lost/unverifiable | `docs/artifacts/t0.2-reference500.json` preserves identity and hashes, but the indexed bytes/source commit are unavailable; it is not recovered evidence. |
| T0.2a absence decision | complete | The record selected `finish the evaluator gate`, made no aggregate decision, and drew no scientific conclusion from missing bytes. |
| T0.4 replacement reference | complete and accepted for its stated diagnostic scope | `docs/artifacts/t0.4-r2-r3-reference-20260811.json` authenticates the immutable 500-user, 14-day, seed-20260811 replacement lineage and passes its comparability audit. `docs/decisions/t0.4-r2-r3-reference-20260811.md` records coverage, per-axis deltas, unsupported claims, and no aggregate winner. It accepts the R2/R3 next-event diagnostic surface, not factorization, causal invariance, or external validity. |

## Requirement status

Statuses describe executable coverage, not scientific success. `partial` means
that at least one named axis or contract is implemented while important axes,
accepted matched evidence, or external validity remain absent.

| ID | Status | Responsible command/surface | Executable coverage and limitation |
|---|---|---|---|
| R1 | partial | `export-dense`; `evaluate --episodes`; `evaluate-change`; component exports | Episode and matched temporary/sustained-change curves plus persistent/context branches are executable. The matched T2.7 gate rejected factorization advancement; persistent/context recovery or disentanglement was not established. |
| R2 | partial | `train`; `evaluate`; `evaluate --transfer`; `compare` | T0.4 durably authenticates coverage-aware learned-versus-naive diagnostics and synthetic transfer slices. Fine-local coverage is weak and real geographic calibration/external validity are absent. |
| R3 | partial | `evaluate --temporal-routine`; `simulate-pair --intervention schedule-shift`; `evaluate-pair` | Observational temporal/routine axes and controlled simulator schedule response are executable. There is no accepted routine component or real-world schedule-shift evidence. |
| R4 | partial | `evaluate --episodes`; `evaluate --temporal-routine`; `evaluate-change`; component reports | Episode response, repeated-versus-one-off behavior, change curves, and branch outputs are measurable. Branch semantics and routine/context separation have not passed T2.7. |
| R5 | partial | `simulate-pair`; `validate-pair`; `evaluate-pair`; exposure-aware `rank` | Exposure/opportunity pairs and T3.6 protected ranking metrics have strict integrity and identity gates. Regret/probability recovery are available only after authentication. No complete reference-scale paired artifact is archived, and simulator control does not establish external causal invariance. |
| R6 | partial | `robustness`; component evaluation; `compare` | Leave-one-service-out drift/degradation and per-component diagnostics are executable. Semantic cross-service recommendation transfer remains pending. |
| R7 | partial | `robustness`; `simulate-pair --intervention observation`; `evaluate-pair` | Deterministic corruptions and controlled observation pairs are executable. They are not calibration to real GPS, timestamp, or missingness processes. |
| R8 | partial | `evaluate --transfer`; `evaluate-ranking` | Held-out-region/geohash diagnostics and T3.7 frozen seen/unseen region/POI plus early/late observed slices report explicit coverage. T3.7 cannot report utility regret, and reference-scale and external geographic evidence remain absent. |
| R9 | partial | dataset 2.0 simulation/validation; `rank`; `evaluate-ranking`; protected `evaluate-pair` | T3.4 controls, T3.5 frozen ranking, T3.6 exposure-aware training/protected evaluation, and T3.7 observed transfer slices are executable. Protected regret/probability recovery require pair authentication; T3.7 cannot report regret. One small synthetic T3.5 lineage is archived, but reference-scale, real-world causal, and external-validity evidence are absent. |
| R10 | partial | `evaluate --reliability`; `calibrate-reliability` | Seeded cutoff repeatability remains distinct from T4.1a's held-out-user window bootstrap. Raw/calibrated reliability bins and coverage-risk curves are executable for authenticated diagnostic controls, but no selected candidate exists and selection-dependent T4.1 remains pending. |
| R11 | partial | `simulate-pair --intervention {temporary-trip,sustained-preference}`; `evaluate-change`; `audit-nonstationarity` | T4.2 is complete: authenticated matched-control adaptation, recovery, forgetting, permanent drift, uncertainty, coverage, and censoring are executable in the simulator. Because no representation has the `selected_candidate` role, the audit currently provides diagnostic-control evidence only and no selection-dependent or external nonstationarity conclusion follows. |
| R12 | pending | — | Membership and sensitive-attribute/privacy audit harnesses remain unimplemented. |
| R13 | partial | `benchmark` | Frozen artifact work plus atomic cold-start, steady single-event, frozen batch, and serialization online workloads are executable for baseline and learned diagnostic controls with mandatory recomputation. Training and hardware-normalized benchmarks remain pending. |

No row above is an aggregate model verdict. In particular, a simulator metric
becoming executable does not show that a representation satisfies the
requirement, and no simulator-only result establishes real-world validity.

## Current gate and open tasks

The T2.7 matched-factorization gate is complete on the new immutable 50-user,
14-day, seed-20260812 pilot. All six variants share one preparation, population,
and cutoff identity. The persistent and combined task-information/collapse
gates failed against the capacity-matched control, so the decision is **do not
advance to a routine branch**. See `docs/FACTORIZATION_DECISION.md` and the
registered T2.7 evidence index. Component names, next-event accuracy, and
covariance diagnostics remain insufficient evidence of semantics.

T3.4--T3.7 are complete over the 2.0 recommendation contract. T3.5 supplies the
frozen-embedding candidate head; T3.6 supplies observed-only exposure-aware
training and a separately integrity-gated protected evaluator; T3.7 supplies
observed-only frozen transfer slices and therefore no utility regret.

T4.2 adaptation/forgetting auditing and T4.5 external-validity claim boundaries
are complete. Because no representation has the `selected_candidate` role,
T4.2 currently supplies diagnostic-control evidence only; the
selection-dependent R11 conclusion remains unavailable. The remaining open
tasks are **T4.1 uncertainty calibration** and **T4.3 privacy audits**. T4.4
online incremental-update benchmarking is implemented as diagnostic-control
evidence and does not relax the negative T2.7
routine-branch gate.

## Evidence limitations

- The accepted T0.4 reference supports only its explicitly reported synthetic,
  coverage-qualified axes; it selects no aggregate winner.
- Paired interventions establish internal simulator control after integrity
  validation, not that the intervention is a faithful real-world causal model.
- Reliability is repeatability, and the benchmark is host-specific offline
  instrumentation; neither completes its deployment requirement.
- Synthetic Hakone catalog attributes are hypotheses, not facts about Hakone or
  Kanto, and a data contract is not recommendation quality.
