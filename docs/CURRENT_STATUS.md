# Current status

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
- Seeded three-cutoff reliability/repeatability evaluation and an observed-only
  frozen-export/offline-evaluation benchmark. These are not calibrated
  uncertainty or hardware-normalized online/training benchmarks.

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
| R5 | partial | `simulate-pair`; `validate-pair`; `evaluate-pair` | Exposure/opportunity pairs have strict identity/integrity gates and separate representation metrics. No complete reference-scale paired artifact is archived, and simulator control does not establish external causal invariance. |
| R6 | partial | `robustness`; component evaluation; `compare` | Leave-one-service-out drift/degradation and per-component diagnostics are executable. Semantic cross-service recommendation transfer remains pending. |
| R7 | partial | `robustness`; `simulate-pair --intervention observation`; `evaluate-pair` | Deterministic corruptions and controlled observation pairs are executable. They are not calibration to real GPS, timestamp, or missingness processes. |
| R8 | partial | `evaluate --transfer`; recommendation contract | Held-out-region and seen/unseen-geohash slices report coverage; the POI contract now removes the schema blocker. Unseen-POI ranking evaluation (T3.7) is still unimplemented and synthetic geography is not external validity. |
| R9 | partial | dataset 2.0 simulation/validation; `rank` | Public POI/request/impression/interaction tables and Hakone request-time attributes are implemented and leakage-tested. Observed-only naive ranking predictions, Recall@K, NDCG@K, MRR, coverage, and shared set hashes are executable; embedding-aware ranking remains pending. |
| R10 | partial | `evaluate --reliability` | Seeded cutoff-bootstrap variance, reliability-error bins, and coverage-risk diagnostics are executable. This is repeatability over three cutoffs, not calibrated uncertainty; T4.1 remains pending. |
| R11 | partial | `simulate-pair --intervention {temporary-trip,sustained-preference}`; `evaluate-change` | Matched-control adaptation, recovery, forgetting, permanent drift, coverage, and censoring are executable in the simulator. No external nonstationarity claim follows. |
| R12 | pending | — | Membership and sensitive-attribute/privacy audit harnesses remain unimplemented. |
| R13 | partial | `benchmark` | Frozen artifact load/export serialization and reliability-evaluation latency, throughput, allocation, and RSS are executable. Training, online update, and hardware-normalized benchmarks remain pending. |

No row above is an aggregate model verdict. In particular, a simulator metric
becoming executable does not show that a representation satisfies the
requirement, and no simulator-only result establishes real-world validity.

## Current gate and next implementation task

The T2.7 matched-factorization gate is complete on the new immutable 50-user,
14-day, seed-20260812 pilot. All six variants share one preparation, population,
and cutoff identity. The persistent and combined task-information/collapse
gates failed against the capacity-matched control, so the decision is **do not
advance to a routine branch**. See `docs/FACTORIZATION_DECISION.md` and the
registered T2.7 evidence index. Component names, next-event accuracy, and
covariance diagnostics remain insufficient evidence of semantics.

T3.4 observable naive rankers are complete over the 2.0 recommendation
contract. Popularity, nearest, and category-preference controls consume only
observed data and authenticate shared immutable request and available-candidate
sets. **T3.5 frozen-embedding candidate ranking** is the next recommendation
implementation task; it must retain those causal and identity contracts.

## Evidence limitations

- The accepted T0.4 reference supports only its explicitly reported synthetic,
  coverage-qualified axes; it selects no aggregate winner.
- Paired interventions establish internal simulator control after integrity
  validation, not that the intervention is a faithful real-world causal model.
- Reliability is repeatability, and the benchmark is host-specific offline
  instrumentation; neither completes its deployment requirement.
- Synthetic Hakone catalog attributes are hypotheses, not facts about Hakone or
  Kanto, and a data contract is not recommendation quality.
