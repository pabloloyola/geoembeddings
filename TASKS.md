# Executable backlog

Select one coherent task at a time. Completion still requires the definition of
done in `AGENTS.md`. Commands marked **proposed** are acceptance contracts for
commands that do not exist yet; completing the task includes implementing and
documenting that command. Artifact paths are relative to `RUN_DIR` or
`EXPERIMENT_DIR` unless another root is shown.

## Completed and verified

These milestones are complete. Their evidence verifies execution and evaluator
contracts, not scientific superiority or disentanglement.

| Task | Requirements | Verification command | Verified artifact/report location |
|---|---|---|---|
| **T1.1 — Dense timestamped export** | R1, R4, R11 | `uv run pytest tests/test_dense_export.py tests/test_cli_paths.py`; `uv run geoembed export-dense --kind {baseline,learned} --event-stride 1 --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` | `dense_statistical_baseline.npz`; `dense_embeddings.npz`; verification notes in `docs/VERIFICATION.md` |
| **T1.2 — Episode-boundary evaluation** | R1, R4 | `uv run pytest tests/test_episode_evaluation.py tests/test_dense_export.py tests/test_cli_paths.py`; `uv run geoembed evaluate --episodes --kind {baseline,learned} --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR`; `uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` | `baseline_episode_response.json`; `episode_response.json`; matched deltas in `comparison/embedding_comparison.{json,md}` |
| **T1.4 — Complete deterministic robustness views** | R6, R7 | `uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind {baseline,learned} --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR`; `uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` | `robustness/{kind}/{view_id}.npz`; `robustness/{kind}_robustness.json`; separate matched R6/R7 axes |
| **T1.3 — Event-removal robustness** | R7 | `uv run geoembed robustness --kind {baseline,learned} --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR`; `uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` | `robustness/{kind}/removal_RATE.npz`; `robustness/{kind}_event_removal.json`; matched R7 axes in `comparison/embedding_comparison.{json,md}` |
| **T1.7 — Reliability and offline efficiency** | R10, R13 | `uv run pytest tests/test_reliability.py tests/test_benchmark.py tests/test_cli_paths.py`; both `evaluate --reliability` kinds; `uv run geoembed benchmark --warmup 1 --iterations 5 ...` | `baseline_reliability.json`; `reliability.json`; `benchmarks/offline.json`; matched smoke evidence in `docs/VERIFICATION.md` |

## Now

T0.2 is **closed as evidence lost/unverifiable**. Its registered evidence index
preserves the original hashes and scientific identity, but the indexed local
roots are absent, the recorded source commit is unavailable in this clone, and no durable
artifact identifier was registered. T0.2a is complete as an absence decision:
it selects only `finish the evaluator gate`, leaves every scientific axis
without an authenticated conclusion, and declares no aggregate winner. Closing
these evidence-disposition tasks does not make the missing reference scientific
evidence.

## Later/gated

- **Counterfactual simulation:** T1.11a–T1.11f and T1.15 are gated by the
  reference decision and strict truth-side pair integrity.
- **Factorized model:** T2.1–T2.7 are gated by T0.2/T0.2a and the evaluator
  gates stated below.
- **Routine branch:** no implementation task is opened until T1.6 can
  distinguish recurring routine from persistent identity and one-off context.
- **Recommendation contract/ranking:** T3.1–T3.7 are gated by an observable,
  versioned recommendation contract with leakage tests.
- **Deployment audits:** T4.1–T4.5 follow scientific evaluator/model evidence;
  simulator-only results must not be presented as external validation.

## P0 — Reproduction, reference, and status

- [x] **T0.1 — Execution/contract smoke verified.** The 50-user, 7-day learned
  pipeline, statistical baseline, and matched comparison are recorded in
  `docs/VERIFICATION.md`; this is plumbing evidence only.

- [x] **T0.1a — Register the complete artifact manifest workflow.**
  - **Requirement IDs:** R1, R3, R4, R7 (evidence provenance only).
  - **Prerequisites:** T0.1.
  - **Affected layer:** documentation.
  - **Baseline artifact required:** the existing T0.1 smoke `RUN_DIR` and
    `EXPERIMENT_DIR`; if unavailable, record that loss rather than regenerating
    and calling it the same artifact.
  - **Command → expected artifact:** `uv run python scripts/index_artifacts.py --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR --output docs/artifacts/t0.1-smoke.json` → immutable artifact index (script is proposed if absent).
  - **Minimum coverage:** unit tests for stable hashing/path normalization;
    integration test rejecting mismatched source hashes or preparation metadata.
  - **Completion evidence:** index records source commit, dataset manifest hash,
    all report/export identifiers, and a same-preparation-contract assertion.
  - **Known limitation/blocker:** generated binaries may live in external
    storage; the smoke cohort is not scientific evidence.
  - **Replacement completion evidence (2026-08-11):** the unavailable T0.1
    smoke was not regenerated under its old identity. The stronger, newly named
    `t0.4-r2-r3-reference-20260811` lineage exercises the completed indexer and
    is authenticated by `docs/artifacts/t0.4-r2-r3-reference-20260811.json`,
    including all current supplemental reports and the offline benchmark.

- [x] **T0.2 — Record the 500-user reference disposition — closed as evidence
  lost/unverifiable.**
  - **Requirement IDs:** R1–R8 (reference coverage; absent axes remain pending).
  - **Prerequisites:** T0.1; clean immutable output roots and sufficient compute.
  - **Affected layer:** model, evaluator, documentation.
  - **Baseline artifact required:** statistical and learned exports made from
    the same 500-user preparation metadata and observed-source hashes.
  - **Command → expected artifact:** run `uv run geoembed pipeline --run-dir runs/reference500 --experiment-dir experiments/reference500 --mode learned --seed 20260811`, then the matched `baseline`, both `export-dense`, both episode `evaluate`, both `robustness`, and `compare` commands documented in `docs/COMMAND_REFERENCE.md` → resolved configs, checkpoint/training report, cutoff and dense exports, evaluation/episode/robustness reports, and `comparison/embedding_comparison.{json,md}`; run the T0.1a indexer → `docs/artifacts/t0.2-reference500.json`.
  - **Minimum coverage:** existing full unit/integration suite plus artifact-index
    integration checks for common users, cutoffs, field order, finiteness,
    source hashes, masks, and preparation contract.
  - **Completion evidence:** immutable index gives source commit, manifest hash,
    seeds, cohort, cutoffs, hashes/locations for every required artifact, and
    explicitly confirms one shared preparation contract.
  - **Known limitation/blocker:** the run is semi-synthetic and compute-heavy;
    missing users or labels must be explained, never silently dropped.
  - **Evidence-loss audit (2026-08-11):** an exhaustive search of the available
    filesystem found neither indexed root nor an archive, no additional artifact
    filesystem is mounted, no Git remote or tag identifies an archive, and the
    recorded source commit is not present in this clone. The index retains the
    original hashes, seeds, cutoffs, field order, and user-set identity, but the
    bytes cannot be authenticated. T0.2 is therefore closed as evidence
    lost/unverifiable, not completed as a scientific reference. Any replacement
    must use a new run/experiment identity and a new evidence-index lineage.

- [x] **T0.2a — Reconcile status and record the post-reference decision.**
  - **Requirement IDs:** R1, R3, R4, R7.
  - **Prerequisites:** T0.2 and its complete artifact index.
  - **Affected layer:** evaluator, documentation.
  - **Baseline artifact required:** `docs/artifacts/t0.2-reference500.json` and
    its indexed baseline/learned reports.
  - **Command → expected artifact:** `uv run python scripts/reconcile_status.py --artifact-index docs/artifacts/t0.2-reference500.json --output docs/decisions/t0.2a-reference-decision.md` → per-axis decision record and reconciled `docs/CURRENT_STATUS.md`/`TASKS.md` statuses (implemented; reconciliation aborts if indexed evidence is unavailable or no longer authentic).
  - **Minimum coverage:** unit tests for status derivation and missing-axis
    handling; integration test that mismatched hashes/cutoffs abort reconciliation.
  - **Completion evidence:** separate conclusions for persistent/preference
    probes, incremental information, geometry/collapse checks, episode response,
    removal robustness, and next-event performance/coverage; exactly one next
    action is selected: repair/ablate, finish evaluator gate, or factorize.
  - **Known limitation/blocker:** no aggregate winner; observational and
    single-vector metrics cannot establish causal invariance or disentanglement.
    The completed decision is an absence audit: all scientific axes remain
    without conclusions until a recovered or replacement reference supplies
    durable artifacts and passes the comparability audit.
  - **Current checkout blocker (2026-08-11):** the committed index says its
    audit passed, but its gitignored `runs/reference500` and
    `experiments/reference500` local roots are absent from this checkout and no
    durable external identifier is recorded. The reconciler correctly aborts
    on the first unavailable indexed report and therefore does not derive
    report-based scientific conclusions. T0.2a is
    complete only as the recorded evidence-disposition decision.

- [x] **T0.3 — Add runtime metadata to reports.**
  - **Requirement IDs:** R13.
  - **Prerequisites:** report schemas and T0.2 report inventory.
  - **Affected layer:** model, evaluator.
  - **Baseline artifact required:** indexed T0.2 reports without standardized
    runtime metadata.
  - **Command → expected artifact:** `uv run geoembed train ...` and `uv run geoembed compare ...` → reports containing schema version, Python/package/torch versions, OS, device, source commit, wall time, and seed.
  - **Minimum coverage:** unit serialization tests; train/compare integration
    tests asserting required finite fields on CPU and tolerant optional device data.
  - **Completion evidence:** regenerated reports validate against the documented
    metadata schema and preserve scientific metric fields.
  - **Known limitation/blocker:** hardware metadata aids reproducibility but is
    not a cross-device performance benchmark.
  - **Completion note (2026-08-11):** versioned, typed runtime provenance is
    present in CPU train and matched compare reports and in compatible base,
    episode, robustness, transfer, and temporal-routine evaluation reports.
    Unit and CPU integration tests validate serialization, null device/source
    handling, finite duration, integer seed, JSON round trips, and preservation
    of scientific metrics and lineage fields. Acceptance uses the newly named
    `t0.3-cpu-smoke-20260811` smoke lineage; it is not the unavailable T0.2
    reference. Proceed next to T1.7 reliability and offline-efficiency work.

- [x] **T0.4 — Add naive next-event baselines and balance metrics.**
  - **Requirement IDs:** R2, R3.
  - **Prerequisites:** prepared train-only vocabularies and current evaluation.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** newly generated replacement reference
    `t0.4-r2-r3-reference-20260811`, with its learned next-event report, training
    label counts, and durable evidence index at
    `docs/artifacts/t0.4-r2-r3-reference-20260811.json`. This is a new lineage,
    not a recovery or continuation of the unavailable T0.2 bytes.
  - **Command → expected artifact:** `uv run geoembed evaluate --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` → `evaluation.json` with train-fitted majority/popularity accuracy, macro-F1 or balanced accuracy, class counts, and known-label coverage.
  - **Minimum coverage:** unit tests for imbalance, unknown labels, and zero
    coverage; integration test proving all baseline statistics fit train only.
  - **Implemented diagnostic surface (static confirmation, 2026-08-11):**
    `src/geoembeddings/training.py` fits the majority/popularity control and
    class counts from `EventWindowDataset(..., "train", ...)` targets, then
    reports learned and naive macro-F1, balanced accuracy, class counts, and
    known-label coverage. `tests/test_next_event_diagnostics.py` exercises an
    imbalanced majority, unknown and zero-known-label coverage, finite balance
    metrics, and invariance of fitted counts/majority to changed evaluation
    frequencies. `docs/COMMAND_REFERENCE.md` documents the same train-only
    report contract. This confirms implementation and tests, not scientific
    acceptance.
  - **Completion evidence:** the durably indexed replacement reference reports
    learned-versus-naive, coverage-aware deltas per target without changing
    frozen embeddings.
  - **Known limitation/blocker:** next-event prediction does not prove embedding
    quality, spatial transfer, or disentanglement. T0.4 remains scientifically
    incomplete until a new immutable run and experiment lineage is durably
    indexed at `docs/artifacts/t0.4-r2-r3-reference-20260811.json` or a newly
    named successor. The lost/unverifiable T0.2 artifacts are not recovered
    evidence and cannot satisfy this gate.
  - **Acceptance evidence (2026-08-11):** the immutable 500-user, seed-20260811
    replacement lineage and per-target learned-versus-naive diagnostics are
    indexed at `docs/artifacts/t0.4-r2-r3-reference-20260811.json`; the
    non-composite interpretation, coverage, missingness, and unsupported claims
    are recorded in `docs/decisions/t0.4-r2-r3-reference-20260811.md`.

## P1 — Evaluator foundations

- [x] **T1.4 (P1A) — Complete deterministic robustness views.**
  - **Requirement IDs:** R6, R7.
  - **Prerequisites:** T1.3, T0.2/T0.2a; selected unless T0.2a explicitly chooses
    another permitted action.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** matched T0.2 baseline/learned exports and
    event-removal reports.
  - **Command → expected artifact:** `uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind {baseline,learned} --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` → versioned view exports and `robustness/{kind}_robustness.json`; `compare` → matched R6/R7 deltas (implemented).
  - **Minimum coverage:** deterministic/view-ID unit tests, boundary cases and
    row-order independence; integration tests for matched masks/coverage,
    observed-only encoding, truth opened only for evaluation, and mismatch rejection.
  - **Completion evidence:** drift and frozen-downstream-degradation curves with
    realized perturbations, matched coverage, hashes, and explicit unencodable rows.
  - **Known limitation/blocker:** deterministic corruption is a sensitivity
    analysis, not proof of real-world noise or causal invariance.

- [x] **T1.5 (P1B) — Add spatial and transfer evaluation.**
  - **Requirement IDs:** R2, R8.
  - **Prerequisites:** T0.2, explicit train/test geography definitions.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** matched T0.2 exports and prepared metadata.
  - **Command → expected artifact:** `uv run geoembed evaluate --transfer --kind {baseline,learned} --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` → `{kind}_transfer_evaluation.json`; `compare` → matched slice deltas (proposed).
  - **Minimum coverage:** unit tests for held-out region/cell, later-time,
    distance, boundary pairs, unknown labels, and zero coverage; integration test
    proving split construction and normalization use no held-out information.
  - **Completion evidence:** held-out-region, unseen-cell/geohash, later-time,
    distance-retrieval, and boundary-consistency metrics each report user/label coverage.
  - **Completion evidence:** versioned `evaluation.transfer` definitions now drive
    train-fitted distance retrieval, boundary pairs, held-out regions, and
    seen/unseen geohash slices. Matched reports are produced with the documented
    baseline/learned commands and rejected by `compare` when identity, fitting,
    definitions, or coverage differ.
  - **Known limitation/blocker:** unseen-POI transfer remains gated by T3 contract;
    observed noisy coordinates are synthetic and these slices do not prove causal transfer.

- [x] **T1.6 (P1C) — Add temporal and routine diagnostics.**
  - **Requirement IDs:** R3, R4.
  - **Prerequisites:** T1.2 and simulator audit of routine/schedule truth validity.
  - **Affected layer:** evaluator, documentation.
  - **Baseline artifact required:** matched T0.2 dense exports and episode reports.
  - **Command → expected artifact:** `uv run geoembed evaluate --temporal-routine --kind {baseline,learned} --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` → `{kind}_temporal_routine.json` (proposed).
  - **Minimum coverage:** unit tests for cyclic hour/day labels, durations,
    periodic retrieval, and one-off/repeated episode selection; integration test
    for evaluator-only truth joins and matched baseline/learned populations.
  - **Completion evidence:** hour/day probes, duration task, periodic retrieval,
    and repeated-routine-versus-one-off results with coverage and collapse checks.
  - **Known limitation/blocker:** declare schedule shift `blocked` if the simulator
    lacks a controlled intervention; do not substitute an observational proxy.
  - **Completion note (2026-08-11):** evaluator-only dense/truth joins now emit
    matched cyclic, duration, periodic retrieval, repeated-versus-one-off,
    coverage, separation, and collapse diagnostics. Simulator audit found no
    controlled schedule-shift scenario, so that evaluation remains blocked.

- [x] **T1.6a — Controlled schedule-shift and protected routine labels.**
  - **Requirement IDs:** R3, R4.
  - **Affected layers:** simulator and protected evaluator; the observed contract is unchanged.
  - **Baseline artifact:** the fixed-seed T1.11 paired-simulation smoke contract; no new model result is claimed.
  - **Completion note (2026-08-11):** `schedule-shift` moves only recurring routine clocks while preserving user latents, episode selection, and one-off timing. Pair integrity authenticates the allowed timestamp/trajectory/event changes. Temporal reports expose protected weekday/weekend, periodic retrieval, and repeated-routine-versus-one-off labels; paired reports expose schedule response beside persistent probes, retrieval, separation, and effective rank.
  - **Routine-model gate:** **not opened.** The completed T2.7 matched gate
    rejected routine expansion because the factorized persistent and combined
    branches failed task-information and collapse checks against the
    capacity-matched control. Post-trip recovery remains available through
    T1.15, but it does not override that negative decision.

- [x] **T1.7 (P1D) — Add reliability and offline-efficiency evaluation.**
  - **Requirement IDs:** R10, R13.
  - **Prerequisites:** T0.3 runtime schema and explicitly named replacement lineage `t0.3-cpu-smoke-20260811`; the lost T0.2 reference is not treated as continuous evidence.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** matched frozen exports/checkpoint and runtime metadata from `t0.3-cpu-smoke-20260811`.
  - **Command → expected artifact:** `uv run geoembed evaluate --reliability ...` → `{kind}_reliability.json`; `uv run geoembed benchmark --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR` → `benchmarks/offline.json`.
  - **Minimum coverage:** seeded resampling and calibration-bin unit tests;
    integration tests for finite variance/coverage-risk plus benchmark schema and
    CPU execution.
  - **Completion evidence:** resampling variance, reliability-error and
    coverage-risk curves, throughput/latency/peak-memory/export/artifact-size results.
  - **Completion note (2026-08-11):** versioned reports validate observed-source/preparation/cutoff identity, finite metrics, seeded cutoff resampling, explicit sparse bins/users, and overwrite protection. The observed-only CPU harness records frozen-export read/validation and reliability-evaluator cost separately for both representations.
  - **Known limitation/blocker:** cutoff bootstrap is a temporal-repeatability diagnostic, not calibrated uncertainty or event/window bootstrap. Offline timing is hardware-specific and does not measure training or online incremental-update latency (T4.4).

## P1 — Matched counterfactual and change support

T1.11 retains its historical program ID; suffixes separate auditable acceptance
steps without losing PR traceability.

- [x] **T1.11a — Independent simulator random streams.**
  - **Requirement IDs:** R5, R7.
  - **Prerequisites:** T0.2a authorizes the counterfactual path.
  - **Affected layer:** simulator.
  - **Baseline artifact required:** fixed-seed T0.2 run/config for behavioral comparison.
  - **Command → expected artifact:** `uv run geoembed simulate --seed 20260811 --run-dir RUN_DIR` → unchanged run contract plus reproducible named stream seeds.
  - **Minimum coverage:** unit tests for stream independence/repeatability;
    fixed-seed simulation/`validate` integration regression.
  - **Completion evidence:** world, user, episode, choice, and observation streams
    can be varied independently and resolved seeds are recorded.
  - **Completion note (2026-08-11):** versioned SHA-256 derivation now resolves
    five named streams from the root seed with optional per-stream overrides.
    Fixed-seed integration coverage verifies stable protected identities and
    truth tables when only observation randomness changes, manifest/config seed
    provenance, validation, and an unchanged observed information boundary.
  - **Known limitation/blocker:** RNG refactoring can change historical draws;
    document lineage rather than claiming bitwise compatibility.

- [x] **T1.11b — Stable identities and stream manifest.**
  - **Requirement IDs:** R5, R7.
  - **Prerequisites:** T1.11a.
  - **Affected layer:** simulator, observed contract.
  - **Baseline artifact required:** T1.11a fixed-seed run.
  - **Command → expected artifact:** `uv run geoembed simulate ...` → versioned `manifest.json` with stream seeds and stable object identities.
  - **Minimum coverage:** manifest schema/hash unit tests; simulate/validate
    integration and migration/contract tests for any version bump.
  - **Completion evidence:** all required identities and seeds validate and
    modeling commands still receive only `observed/`.
  - **Completion note (2026-08-11):** root `manifest.json` now carries
    `geoembeddings-simulation-identity/1.0` with the derivation/identity/hash
    algorithms, all resolved seeds, six entity counts, and canonical set
    hashes. Semantic SHA-256 IDs make users, POIs, episodes, decisions, and
    trajectories independent of CSV row order and unrelated RNG consumption;
    fixed-seed observation-stream integration retains all identity hashes.
    Deep validation rejects missing, malformed, duplicate, incomplete, or
    table-inconsistent declarations. The additions are run-level/truth-side
    metadata, so `geoembeddings-dataset/1.0` and its two observed tables remain
    unchanged; legacy dataset/1.0 modeling remains readable.
  - **Known limitation/blocker:** contract changes require explicit versioning;
    truth declarations must not enter observed tables.

- [x] **T1.11c — Versioned pair-manifest contract.**
  - **Requirement IDs:** R5, R7.
  - **Prerequisites:** T1.11b.
  - **Affected layer:** simulator, observed contract, evaluator.
  - **Baseline artifact required:** two identity-compatible fixed-seed runs.
  - **Command → expected artifact:** `uv run geoembed pair-manifest --reference-run-dir REF --intervention-run-dir INT --output PAIR_DIR/pair_manifest.json` → versioned pair manifest (proposed).
  - **Minimum coverage:** schema unit tests for invariant/changed fields and
    keys; integration tests rejecting missing hashes, incompatible versions, or
    overlapping invariant/change declarations.
  - **Completion evidence:** manifest identifies both runs, intervention,
    hashes, invariant objects, allowed changes, and user/time/object matching keys.
  - **Known limitation/blocker:** declarations are protected truth and cannot be
    inputs to prepare/train/export.
  - **Completion note (2026-08-11):** typed
    `geoembeddings-pair-manifest/1.0` declarations now bind reference and
    intervention run identities, manifest/config/source/entity hashes,
    intervention parameters, invariants, allowed changes, semantic matching
    keys, stream lineage, and creation provenance. The canonical protected
    pair-root CLI rejects incompatible/malformed/ambiguous declarations and
    permits overwrite only after validating the exact existing pair artifact.
    Fixed-seed observation-stream integration coverage preserves all six
    identity classes. Pair field equality and representation metrics remain
    explicitly gated on T1.11d/T1.11f.

- [x] **T1.11d — Pair-integrity validator.**
  - **Requirement IDs:** R5, R7.
  - **Prerequisites:** T1.11c.
  - **Affected layer:** simulator, evaluator.
  - **Baseline artifact required:** declared exposure-only pair.
  - **Command → expected artifact:** `uv run geoembed validate-pair --pair-manifest PAIR_DIR/pair_manifest.json` → `PAIR_DIR/pair_integrity.json` (proposed).
  - **Minimum coverage:** field-level match/mismatch unit tests; integration
    tests proving exact failures occur before representation metrics execute.
  - **Completion evidence:** users, preferences, world objects, and required
    episodes are identical under enumerated invariants with precise diagnostics.
  - **Known limitation/blocker:** R5 remains non-executable until this validator
    and T1.11f both pass.
  - **Completion note (2026-08-11):** `validate-pair` now authenticates the
    canonical pair/run layouts, schemas, source/config/manifest/entity hashes,
    matching keys, and stream lineage before comparing all observed and truth
    tables at field granularity. Its versioned report includes row/entity
    coverage, missing/duplicate keys, invariant and allowed-change outcomes,
    and bounded exact mismatch samples. A reusable hard gate rejects missing,
    failing, stale, or input-stale reports before future representation metrics.
    This proves controlled simulator artifact integrity, not external causal
    validity; T1.11e/T1.11f remain required for executable R5/R7 deltas.

- [x] **T1.11e — Exposure, opportunity, and observation interventions.**
  - **Requirement IDs:** R5, R7.
  - **Prerequisites:** T1.11d passes for exposure-only pairs.
  - **Affected layer:** simulator.
  - **Baseline artifact required:** validated reference run/pair manifest.
  - **Command → expected artifact:** `uv run geoembed simulate-pair --intervention {exposure,opportunity,observation} ...` → immutable paired runs, pair manifests, and passing `pair_integrity.json` reports (proposed).
  - **Minimum coverage:** intervention-specific unit tests for allowed changes;
    fixed-seed simulate/validate/validate-pair integration per intervention.
  - **Completion evidence:** each intervention changes only declared fields and
    passes structural plus behavioral diagnostics.
  - **Known limitation/blocker:** constants are experimental hypotheses, not
    calibrated facts about Tokyo/Kanto.
  - **Completion note (2026-08-11):** simulation config v2 now declares each
    intervention's override, invariant entities, permitted fields, affected
    independent stream, and expected diagnostic. `simulate-pair` preflights and
    refuses existing roots, creates matched stable-identity runs, preserves
    unrelated user latent/world/episode state, runs structural validation,
    writes the protected manifest and passing field-level integrity report, and
    checks fixed-seed behavioral directions. Parameterized integration tests
    cover exposure, opportunity, and observation plus immutability. These are
    experimental simulator assumptions only; T1.11f is still required before
    any representation-level R5/R7 conclusion.

- [x] **T1.11f — Matched counterfactual evaluator.**
  - **Requirement IDs:** R5, R7.
  - **Prerequisites:** T1.11e and baseline/learned exports from both paired runs.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** validated pair plus statistical and learned
    exports with matching source lineage, cutoffs, users, and keys.
  - **Command → expected artifact:** `uv run geoembed evaluate-pair --pair-manifest PAIR_DIR/pair_manifest.json --baseline-experiment-dir ... --learned-experiment-dir ...` → `PAIR_DIR/counterfactual_comparison.{json,md}` (proposed).
  - **Minimum coverage:** matching/coverage and metric unit tests; integration
    tests for truth boundary, pair-integrity prerequisite, mismatch rejection,
    and complete baseline/learned execution.
  - **Completion evidence:** per-intervention match coverage, persistent-trait
    invariance, embedding drift, and downstream degradation reported separately.
  - **Known limitation/blocker:** paired simulator evidence supports controlled
    causal claims only within the simulator, not external causal validity.
  - **Completion note (2026-08-11):** `evaluate-pair` hard-gates on the supported,
    hash-current passing integrity report, authenticates both representation kinds
    and all four source/preparation/export contracts, and rejects mismatched keys,
    fields, cutoffs, users, lineage, dimensions, or invariant labels before metrics.
    Versioned JSON and Markdown keep each intervention and representation separate
    and report coverage/exclusions, frozen persistent/preference probes, drift,
    separation, retrieval, effective rank, task information, hashes, seeds, and
    limitations without an aggregate winner. No complete reference-scale paired
    artifact is archived; external causal validity and real-noise calibration remain blocked.

- [x] **T1.15 — Temporary-trip and sustained-preference-change scenarios.**
  - **Requirement IDs:** R1, R11.
  - **Prerequisites:** T1.11a–T1.11f.
  - **Affected layer:** simulator, evaluator.
  - **Baseline artifact required:** matched no-change pair and T0.2 representations.
  - **Command → expected artifact:** `uv run geoembed simulate-pair --intervention {temporary-trip,sustained-preference} ...`; `uv run geoembed evaluate-change ...` → paired runs and `change_evaluation.{json,md}` (proposed).
  - **Minimum coverage:** change-point/duration unit tests; integration tests for
    invariant identities, evaluator-only change truth, adaptation and recovery.
  - **Completion evidence:** temporary and sustained curves report adaptation,
    forgetting, permanent drift, coverage, and baseline/learned deltas.
  - **Known limitation/blocker:** validity depends on distinguishable simulator
    interventions; one trip cannot be treated as long-term preference evidence.
  - **Completion note (2026-08-11):** versioned choice-stream interventions preserve stable identities and persistent latent truth, expose protected half-open change points, and pass fixed-seed utility diagnostics. `evaluate-change` hard-gates on current pair integrity and emits matched-control baseline/learned adaptation, recovery, forgetting, and permanent-drift curves with explicit coverage and censoring. These synthetic single-vector diagnostics do not establish real-world causality or disentanglement.

## P2 — Two-way factorized model

P2 begins only after T0.2/T0.2a, executable episode metrics, the R5/R6/R7 tests
selected for the model hypothesis, and collapse diagnostics (separation,
retrieval, centered effective rank, task information). The routine branch stays
gated by T1.6.

- [x] **T2.1 — Typed multi-component encoder output.**
  - **Requirement IDs:** R1, R4.
  - **Prerequisites:** P2 entry gate.
  - **Affected layer:** model.
  - **Baseline artifact required:** T0.2 single-vector checkpoint/export.
  - **Command → expected artifact:** `uv run pytest tests/test_model.py` → typed
    `persistent`, `context`, and `combined` output contract.
  - **Minimum coverage:** unit tests for names, shapes, finiteness, gradients,
    device movement, and single-vector adapter; model smoke integration.
  - **Completion evidence:** public APIs use the typed boundary rather than new
    positional tuples, with no training behavior change yet.
  - **Known limitation/blocker:** an interface alone is not factorization evidence.
  - **Completion note (2026-08-11):** a named, device-movable output boundary
    exposes persistent, context, and combined tensors; the explicit legacy
    adapter preserves the single vector as persistent/combined with zero context.

- [x] **T2.2 — Configuration-driven model registry.**
  - **Requirement IDs:** R1, R4.
  - **Prerequisites:** T2.1.
  - **Affected layer:** model.
  - **Baseline artifact required:** T0.2 config/checkpoint behavior.
  - **Command → expected artifact:** `uv run geoembed train --config CONFIG ...` → checkpoint naming the registered model variant.
  - **Minimum coverage:** registry/config validation unit tests; integration test
    reproducing `SingleVectorEncoder` behavior and rejecting unknown variants.
  - **Completion evidence:** existing CLI selects variants without truth inputs
    and the legacy default remains compatible.
  - **Known limitation/blocker:** registry flexibility does not justify model complexity.
  - **Completion note (2026-08-11):** YAML selects `single_vector` while omitted
    variants retain that default. Unknown variants fail before data or output access.

- [x] **T2.3 — Multi-component export schema.**
  - **Requirement IDs:** R1, R4.
  - **Prerequisites:** T2.1–T2.2.
  - **Affected layer:** model, observed contract, evaluator.
  - **Baseline artifact required:** existing single-vector cutoff/dense exports.
  - **Command → expected artifact:** `uv run geoembed export ...` and
    `export-dense ...` → versioned component exports with names, dimensions,
    field order, and source hashes.
  - **Minimum coverage:** schema/migration unit tests; cross-stage integration
    for legacy single-vector and multi-component readers plus mismatch rejection.
  - **Completion evidence:** each component is independently addressable and
    legacy exports remain readable under an explicit compatibility rule.
  - **Known limitation/blocker:** exported branch names do not establish semantics.
  - **Completion note (2026-08-11):** checkpoints and learned exports now carry
    versioned components, ordered fields, hashes, cutoffs, variant, and compatibility
    metadata. Readers explicitly migrate legacy vectors and reject malformed schemas.

- [x] **T2.4 — Capacity-matched persistent/context encoder.**
  - **Requirement IDs:** R1, R4.
  - **Prerequisites:** T2.1–T2.3 and approved hypothesis/evaluator set.
  - **Affected layer:** model.
  - **Baseline artifact required:** T0.2 GRU and capacity specification.
  - **Command → expected artifact:** `uv run geoembed train --config configs/embedding/factorized_pc.yaml ...` → factorized checkpoint/training report.
  - **Minimum coverage:** branch/update/masking unit tests including MPS-safe
    final-state behavior; learned pipeline integration without truth access.
  - **Completion evidence:** parameter budget is documented/matched and all
    branches train/export with finite outputs.
  - **Known limitation/blocker:** the completed T2.7 comparison failed the
    scientific advancement gate; implementation completion is not evidence of
    factorization.
  - **Implementation note (2026-08-11):** `factorized_pc` now provides shared
    event features, padded long/recent GRUs, conservative persistent updates,
    gated fusion, and component exports. The later matched T2.7 run supplied the
    capacity control and recorded the negative gate decision.

- [x] **T2.5 — Capacity controls and ablations.**
  - **Requirement IDs:** R1, R4.
  - **Prerequisites:** T2.4.
  - **Affected layer:** model, evaluator.
  - **Baseline artifact required:** T0.2 and T2.4 checkpoints/exports.
  - **Command → expected artifact:** train/export configured `capacity_matched_single`, `persistent_only`, `context_only`, fusion, and loss ablations → separate immutable experiment reports.
  - **Minimum coverage:** config/parameter-count unit tests; integration smoke
    for every ablation and artifact provenance.
  - **Completion evidence:** artifact index reports parameter counts, seeds,
    hashes, and matched evaluation inputs for all controls.
  - **Known limitation/blocker:** exhaustive ablations may be compute-limited;
    omissions must be explicit.
  - **Implementation note (2026-08-11):** versioned configs cover a dynamically
    parameter-matched single GRU, persistent-only, context-only, fusion, and both
    loss-routing removals. The completed T2.7 matrix now authenticates their
    matched artifacts; its failed gate supports no factorization claim.

- [x] **T2.6 — Branch-specific objectives and reports.**
  - **Requirement IDs:** R1, R4, R5, R6, R7.
  - **Prerequisites:** T2.4–T2.5 and relevant executable evaluators.
  - **Affected layer:** model, evaluator.
  - **Baseline artifact required:** capacity-matched single and branch ablations.
  - **Command → expected artifact:** `train`, component `export`, and relevant
    `evaluate`/`compare` commands → branch-specific losses and per-component reports.
  - **Minimum coverage:** loss-routing/no-truth unit tests; cross-stage
    integration proving each branch is evaluated on intended and failure axes.
  - **Completion evidence:** reports separate persistent, context, and combined
    task information, invariance, response, and collapse diagnostics.
  - **Known limitation/blocker:** covariance/orthogonality is diagnostic only,
    not semantic disentanglement.
  - **Implementation note (2026-08-11):** named objective and consistency routes
    are observed-only; base evaluation reports every component with persistent
    probes, stability, separation, retrieval, and centered effective rank.

- [x] **T2.7 — Matched factorization decision.**
  - **Requirement IDs:** R1, R4, R5, R6, R7.
  - **Prerequisites:** T2.5–T2.6.
  - **Affected layer:** evaluator, documentation.
  - **Baseline artifact required:** indexed capacity-matched single-vector and
    factorized/ablation exports on identical data and cutoffs.
  - **Command → expected artifact:** `uv run geoembed compare ...` →
    `comparison/factorized_comparison.{json,md}` and decision record.
  - **Minimum coverage:** comparison/mismatch unit tests; full matched
    baseline/factorized integration across selected R1/R4/R5/R6/R7 axes.
  - **Completion evidence:** intended branch improvements, regression axes,
    retrieval/separation/effective-rank/task-information checks, hashes, and
    limitations are reported separately.
  - **Known limitation/blocker:** no aggregate winner; failure to beat controls
    blocks routine expansion rather than being hidden by next-event accuracy.
  - **Superseded planning note (2026-08-11):** the gate protocol was defined
    before evidence generation. The completed replacement evidence and decision
    below supersede that pre-run status without changing the required per-axis
    protocol.
  - **Completion note (2026-08-12):** the new immutable 50-user/14-day seed
    20260812 lineage trained all six controls from one preparation and produced
    matched cutoff/dense/base/episode/robustness/temporal reports plus the
    strict factorized comparison. Persistent and combined task-information and
    collapse gates failed versus the capacity control, so the evidence-backed
    decision is **do not advance** and routine work remains closed.

## P3 — Recommendation contract and ranking

- [x] **T3.1 — POI catalog schema and migration.**
  - **Requirement IDs:** R9.
  - **Prerequisites:** approved observable/truth field review.
  - **Affected layer:** simulator, observed contract, documentation.
  - **Baseline artifact required:** current dataset contract and POI truth tables.
  - **Command → expected artifact:** `uv run geoembed simulate ...` → versioned
    `observed/poi_catalog.csv.gz`, manifest/schema update, migration notes.
  - **Minimum coverage:** schema, leakage, ID, timestamp, and migration unit
    tests; simulate/validate/prepare integration.
  - **Completion evidence:** catalog exposes only request-time public metadata
    and old contract handling is explicit.
  - **Known limitation/blocker:** synthetic POI attributes are not real Kanto facts.
  - **Completion note (2026-08-11):** contract 2.0 adds an ordered public catalog;
    event-only 1.0 runs remain explicitly readable without fabricated tables.

- [x] **T3.2 — Request, availability, impression, interaction schemas.**
  - **Requirement IDs:** R9.
  - **Prerequisites:** T3.1.
  - **Affected layer:** simulator, observed contract.
  - **Baseline artifact required:** T3.1 catalog contract.
  - **Command → expected artifact:** `simulate` → versioned
    `observed/recommendation_requests.csv.gz`, `impressions.csv.gz`, and
    `interactions.csv.gz` with manifest entries.
  - **Minimum coverage:** referential/schema/leakage unit tests; end-to-end
    simulate/validate and observed-only consumer integration.
  - **Completion evidence:** available, shown, rank, response, and request-time
    metadata semantics are documented and validated.
  - **Known limitation/blocker:** utility, probabilities, latent intent, and
    counterfactual outcomes must remain under `truth/`.
  - **Completion note (2026-08-11):** schemas validate keys, timestamps,
    availability-before-display, field order, and protected-field exclusion.

- [x] **T3.3 — Hakone catalog and request-time attributes.**
  - **Requirement IDs:** R9.
  - **Prerequisites:** T3.1–T3.2.
  - **Affected layer:** simulator, observed contract.
  - **Baseline artifact required:** valid empty/minimal recommendation contract.
  - **Command → expected artifact:** fixed-seed `simulate`/`validate` → populated
    public Hakone catalog/requests and behavioral validation report.
  - **Minimum coverage:** availability/open-hours/travel-time/category unit
    tests; fixed-seed behavioral integration with coverage diagnostics.
  - **Completion evidence:** onsen/restaurant/cafe/shop/hotel/attraction
    requests have usable candidates and plausible documented diagnostics.
  - **Known limitation/blocker:** attributes are hypotheses, not calibrated reality.
  - **Completion note (2026-08-11):** fixed-seed runs populate all six target
    categories and record observed-only naive-ranker readiness diagnostics.

- [x] **T3.4 — Observable naive rankers.**
  - **Requirement IDs:** R9.
  - **Prerequisites:** T3.1–T3.3.
  - **Affected layer:** model, evaluator.
  - **Baseline artifact required:** frozen request/candidate sets.
  - **Command → expected artifact:** `uv run geoembed rank --model {popularity,nearest,category_preference} ...` → `ranking/{model}.{npz,json}`.
  - **Minimum coverage:** scoring/tie/availability unit tests; integration test
    proving rankers consume observed data only and share candidate sets.
  - **Completion evidence:** Recall/NDCG/MRR/coverage per naive ranker with
    immutable request/candidate hashes.
  - **Known limitation/blocker:** popularity and proximity are controls, not personalization.
  - **Completion note (2026-08-12):** `src/geoembeddings/ranking.py` and the
    `geoembed rank` command implement the `popularity`, `nearest`, and
    `category_preference` controls. `tests/test_ranking.py` covers scoring,
    deterministic ties, causal cutoffs, metrics, shared sets, dataset-version
    rejection, and the observed-only boundary. Versioned
    `geoembeddings-ranking-predictions/1.0` NPZ and
    `geoembeddings-ranking-report/1.0` JSON schemas report Recall@K, NDCG@K,
    MRR, and coverage, and carry identical canonical request and available-
    candidate SHA-256 hashes across models. Ranking reads only the observed POI,
    request, impression, interaction, and event tables; protected truth is not
    an input.

- [x] **T3.5 — Frozen-embedding candidate ranker.**
  - **Requirement IDs:** R9.
  - **Prerequisites:** T3.4 and a selected frozen embedding.
  - **Affected layer:** model, evaluator.
  - **Baseline artifact required:** T3.4 results and frozen embedding export.
  - **Command → expected artifact:** `uv run geoembed rank --model frozen_embedding ...` → checkpoint, predictions, and `ranking/frozen_embedding.json` (proposed).
  - **Minimum coverage:** causal cutoff/candidate scoring unit tests; integration
    for observed-only training and frozen candidate comparison.
  - **Completion evidence:** first-arrival/local-action ranking metrics beat or
    contextualize naive baselines with coverage and no end-to-end tuning.
  - **Known limitation/blocker:** stochastic-choice accuracy alone is insufficient.
  - **Implementation record (2026-08-12):** the existing `rank` command now
    consumes the canonical timestamped `export-dense` artifact, authenticates
    its schema, source/preparation hashes, component identity, and timestamps,
    and selects the latest user vector no later than each request. It rejects
    empty scorable training data before preprocessing, fits preprocessing and
    only a deterministic candidate interaction head on explicitly disjoint
    training requests, and writes the centralized checkpoint plus
    versioned prediction/report artifacts. Reports preserve T3.4 candidate
    identity and give separate control deltas, per-split requested/scorable/
    positive/candidate counts, request identities, and explicit exclusions.
    Seed is `20260812`. The immutable same-run evidence lineage is indexed in
    `docs/artifacts/t3.5-ranking-20260812.json` and interpreted in
    `docs/VERIFICATION.md`: all four reports share request/candidate hashes;
    frozen Recall@1/5/10 is 0.94/0.98/0.98, NDCG@1/5/10 is
    0.94/0.962619/0.962619, and MRR is 0.956667. Every metric exceeds each
    control separately. Training/validation/test request and user coverage is
    36/36, 8/8, and 5/6, with the one causal-history exclusion named in the
    index. This satisfies the T3.5 gate on this small synthetic fixed-seed
    lineage only; it is not causal or utility evidence and does not authorize
    end-to-end encoder tuning.

- [x] **T3.6 — Exposure-aware ranking and counterfactual evaluation.**
  - **Requirement IDs:** R5, R9.
  - **Prerequisites:** T1.11f, T3.5.
  - **Affected layer:** model, evaluator.
  - **Baseline artifact required:** T3.4/T3.5 rankers on validated paired candidate sets.
  - **Command → expected artifact:** train exposure-aware ranker and
    `evaluate-pair --ranking ...` → `ranking/exposure_counterfactual.{json,md}` (proposed).
  - **Minimum coverage:** propensity/weighting and clipping unit tests;
    paired-run integration with truth restricted to evaluation.
  - **Completion evidence:** observed ranking metrics, utility regret, and
    probability recovery are separated with sensitivity diagnostics.
  - **Known limitation/blocker:** exposure adjustment depends on simulator
    identification assumptions and must not imply real-world causality.
  - **Implementation record (2026-08-12):** after the indexed T3.5 evidence and
    executable T3.7 transfer slices became available, the frozen candidate head
    gained an observed-only exposure-aware variant. It estimates the logging
    propensity as a Laplace-smoothed shown rate by public candidate position,
    fitted exclusively on training requests, with clipping and threshold
    sensitivity configured by `configs/ranking/exposure_v1.yaml`. Reports expose
    ESS, weight quantiles, clipping rate, threshold sensitivity, coverage, and
    observed ranking metrics while retaining all T3.4 controls and the unweighted
    T3.5 head. The protected paired evaluator first re-authenticates a current
    passing `pair_integrity.json`, prediction schemas, source hashes, and exact
    request/candidate identities before opening evaluator-only recommendation
    utility and choice probabilities. Its regret and probability-recovery output
    is simulator-identification evidence only, never real-world causal evidence.

- [x] **T3.7 — Unseen-region and unseen-POI ranking slices.**
  - **Requirement IDs:** R8, R9.
  - **Prerequisites:** T3.5 and explicit frozen split contract.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** identical T3.4/T3.5 predictions/candidate sets.
  - **Command → expected artifact:** `uv run geoembed evaluate-ranking --slices unseen-region,unseen-poi ...` → `ranking/transfer_slices.json` (proposed).
  - **Minimum coverage:** split/coverage/unknown-POI unit tests; integration test
    preventing train leakage and candidate-set mismatch.
  - **Completion evidence:** seen/unseen region and POI metrics, regret, coverage,
    and early/late trip results are reported separately.
  - **Known limitation/blocker:** synthetic transfer is not external validity.
  - **Completion note (2026-08-12):** `evaluate-ranking` now authenticates all
    four T3.4/T3.5 reports and prediction surfaces, fits a versioned identity
    contract only from requests at or before the frozen ranker's `train_end`,
    and writes separate seen/unseen region/POI intersections and observable
    early/late-stage metrics with four-dimensional coverage. Utility regret is
    explicitly unavailable because this evaluator never receives protected
    truth. Unit and cross-stage integration tests cover equality, leakage,
    unknown identities, empty slices, duplicates, and hash mismatch. Synthetic
    geography and implicit interaction labels remain the principal limitations.

## P4 — Responsible deployment evidence

- [ ] **T4.1 — Calibrate representation uncertainty.**
  - **Requirement IDs:** R10.
  - **Prerequisites:** T1.7 and a `selected_candidate` under the
    representation-selection policy in `docs/EXPERIMENT_PROTOCOL.md`.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** T1.7 resampling/reliability reports.
  - **Command → expected artifact:** `evaluate --reliability --calibrate ...` →
    `reliability/calibration.json` (proposed).
  - **Minimum coverage:** calibration/coverage-risk unit tests; held-out-user
    integration preventing calibration/test reuse.
  - **Completion evidence:** reliability-error and coverage-risk improve over
    uncalibrated variance with uncertainty method/config recorded.
  - **Known limitation/blocker:** simulator calibration may not transfer to real users.
  - **Selection status:** after the negative T2.7 decision, no current export is
    a `selected_candidate`, so selection-dependent calibration conclusions are
    unavailable. Explicitly labeled `diagnostic_control` comparisons may still
    audit the statistical baseline, capacity-matched single vector, and
    factorized diagnostic variants under the protocol's provenance rules.

  - [ ] **T4.1a — Diagnostic-control uncertainty calibration (non-selection
    evidence).**
    - **Scope:** calibrate on held-out users for the statistical baseline, the
      capacity-matched single-vector control, and each factorized diagnostic
      variant. Every representation remains immutably labeled
      `diagnostic_control`; this subtask must neither select a candidate nor
      establish the hypothesized semantics of a factorized branch.
    - **Split and uncertainty protocol:** calibration users and test users must
      be disjoint, frozen before fitting, and shared across controls. Estimate
      uncertainty with a seeded event- or window-bootstrap whose sampling unit,
      replicate count, replacement policy, and sparse-history handling are
      explicit. Do not reuse T1.7's three-cutoff temporal-repeatability estimate
      as though it were calibrated uncertainty.
    - **Artifact contract:** `reliability/calibration.json` must record the
      immutable representation role and identity for every control, all input
      source hashes, preparation identity, calibration-user and test-user
      hashes, resampling method and seed, and every fitted calibration parameter.
      Reject role changes, preparation/source mismatches, split overlap, or
      post-hoc user-set changes.
    - **Required reporting:** report raw and calibrated reliability-error bins
      separately, and raw and calibrated coverage-risk curves separately, for
      every diagnostic control; include bin/coverage counts and exclusions and
      emit no aggregate winner.
    - **Minimum coverage:** unit tests for bootstrap determinism, calibration
      fitting, sparse bins, and coverage-risk; integration tests for held-out-
      user isolation plus role, source, preparation, and split-hash mismatch
      rejection.
    - **Completion boundary:** completing this diagnostic subtask does not
      complete T4.1. Only a future gate-passing representation with the
      immutable `selected_candidate` role can provide T4.1's
      selection-dependent completion evidence.

- [x] **T4.2 — Adaptation and forgetting audit.**
  - **Requirement IDs:** R11.
  - **Prerequisites:** T1.15 and a `selected_candidate` under the
    representation-selection policy in `docs/EXPERIMENT_PROTOCOL.md`.
  - **Affected layer:** evaluator.
  - **Baseline artifact required:** no-change, temporary, and sustained paired reports.
  - Run `evaluate-change` separately to generate each authenticated no-change,
    temporary, and sustained scenario input report.
  - **Command → expected artifact:**
    `uv run geoembed audit-nonstationarity --no-change-report pairs/no-change/change_evaluation.json --temporary-report pairs/temporary/change_evaluation.json --sustained-report pairs/sustained/change_evaluation.json --output-dir experiments/r11-audit`
    → `audits/nonstationarity.{json,md}`.
  - **Minimum coverage:** time-to-threshold/censoring unit tests; matched-scenario
    integration for adaptation, forgetting, drift, and regret.
  - **Completion evidence:** temporary decay and sustained update are compared
    per component/control with coverage and uncertainty.
  - **Known limitation/blocker:** the simulator defines the change semantics.
  - **Selection status:** after the negative T2.7 decision, no current export is
    a `selected_candidate`, so selection-dependent adaptation/forgetting
    conclusions are unavailable. The protocol still permits explicitly labeled
    `diagnostic_control` comparisons of the statistical baseline,
    capacity-matched single vector, and factorized diagnostic variants.
  - **Completion note (2026-08-12):** `audit-nonstationarity` writes the
    versioned canonical `audits/nonstationarity.{json,md}` artifacts only after
    authenticating no-change, temporary-trip, and sustained-preference report
    schemas and exact user, cutoff, preparation, source-lineage, component,
    relative-day, and censoring identities. It reports matched-control
    time-to-adaptation, recovery, temporary forgetting, sustained permanent
    drift, confidence intervals, censoring, exclusions, and coverage separately
    for the statistical baseline, capacity-matched single vector, and each
    factorized diagnostic component; it explicitly emits no aggregate winner.
    The current role is `diagnostic_control`, so the selection-dependent R11
    conclusion remains unavailable. Simulator-defined change semantics,
    thresholds, sparse post-change observations, and unestablished factorized
    branch semantics remain limitations. **Next decision:** preserve the T2.7
    do-not-advance decision and proceed to T4.3 without selecting a least-bad
    representation; rerun this audit only when immutable, indexed three-scenario
    reports for a gate-passing candidate exist.

- [ ] **T4.3 — Privacy audits.**
  - **Requirement IDs:** R12.
  - **Design contract:** [`docs/PRIVACY_THREAT_MODEL.md`](docs/PRIVACY_THREAT_MODEL.md)
    fixes membership and sensitive-attribute units, attacker access, held-out
    splits, imbalance/attack controls, uncertainty, immutable report provenance,
    and prohibited interpretations before implementation.
  - **Prerequisites:** a `selected_candidate` under the
    representation-selection policy in `docs/EXPERIMENT_PROTOCOL.md` and a
    threat-model document.
  - **Affected layer:** evaluator, documentation.
  - **Baseline artifact required:** matched statistical/single/factorized exports
    and utility reports.
  - **Command → expected artifact:** `uv run geoembed audit-privacy ...` →
    `audits/privacy.{json,md}` (proposed).
  - **Minimum coverage:** split/attack/imbalance unit tests; integration test for
    held-out membership and sensitive-attribute protocols.
  - **Completion evidence:** attack AUC, sensitive probes, utility/privacy curves,
    confidence intervals, and threat model are reported.
  - **Known limitation/blocker:** simulator privacy attacks do not certify real deployment.
  - **Selection status:** after the negative T2.7 decision, no current export is
    a `selected_candidate`, so selection-dependent privacy conclusions are
    unavailable. Threat-model-scoped comparative attacks may still be reported
    for the statistical baseline, capacity-matched single vector, and
    factorized variants only when each is labeled `diagnostic_control` and its
    required audit provenance is recorded.

- [x] **T4.4 — Online incremental-update benchmarks.**
  - **Requirement IDs:** R13.
  - **Design contract:** [`docs/INCREMENTAL_UPDATE_CONTRACT.md`](docs/INCREMENTAL_UPDATE_CONTRACT.md)
    fixes atomic batch-append semantics, per-component mutation, edge-case and
    recomputation requirements, immutable workloads, measurement provenance,
    and Apple MPS constraints before implementation.
  - **Prerequisites:** stable update/export API and T0.3 metadata schema.
  - **Affected layer:** model, evaluator.
  - **Baseline artifact required:** T1.7 offline benchmark and selected checkpoint.
  - **Command → expected artifact:** `uv run geoembed benchmark ...` → immutable
    `benchmarks/online_workload.json` and canonical `benchmarks/online.json`.
  - **Implemented evidence:** typed atomic state/result/workload boundaries,
    deterministic frozen cold/steady/batched workloads, baseline and learned
    diagnostic controls, full-recomputation gating, rollback/idempotency tests,
    tail latency, throughput, memory, lineage, runtime metadata, and exclusions.
    CPU is required; CUDA/MPS remain optional and must reuse workload identity.
  - **Minimum coverage:** warmup/iteration/statistics unit tests; CPU integration
    plus optional CUDA/MPS regression with identical workload metadata.
  - **Completion evidence:** cold-start, steady single-event, frozen batched,
    and export-serialization workloads report p50/p95 update latency,
    throughput, peak memory, batch size, device/software metadata, hashes,
    artifact size, and passing full-recomputation correctness checks.
  - **Known limitation/blocker:** hardware-specific results require comparable environments.

- [x] **T4.5 — Calibration and external-validity limits.**
  - **Requirement IDs:** R1–R13 (claim boundaries).
  - **Prerequisites:** current simulator/evaluator evidence inventory.
  - **Affected layer:** documentation.
  - **Baseline artifact required:** latest indexed reports and any licensed
    aggregate calibration sources.
  - **Command → expected artifact:** `uv run python scripts/check_evidence_links.py docs/EXTERNAL_VALIDITY.md` → validated evidence/limitation document (proposed).
  - **Minimum coverage:** documentation link/artifact-hash check; integration
    check that every quantitative claim names cohort, seed, source, and scope.
  - **Completion evidence:** calibrated and uncalibrated assumptions, licensing,
    representativeness, missing real-data tests, and prohibited claims are explicit.
  - **Known limitation/blocker:** without appropriate real de-identified data,
    external validity remains unmeasurable rather than proxied.
  - **Completion note (2026-08-12):** `docs/EXTERNAL_VALIDITY.md` now separates
    simulator assumptions, accepted indexed evidence, unsupported claims,
    licensing/data constraints, synthetic Hakone/Kanto limits, and the evidence
    needed to change each R1--R13 status. The static evidence registry and
    `scripts/check_evidence_links.py` authenticate local paths and hashes,
    validate claim metadata and identifiers, preserve T0.2 as unavailable
    history, and distinguish T0.4's narrow acceptance from scientific success.

## Task/PR work-note template

```text
Task and requirement IDs:
Prerequisite tasks:
Affected layer(s):
Hypothesis:
Baseline artifact and hashes:
Information-boundary/contract review:
Unit and integration tests:
Executable command:
Expected and produced artifacts:
Completion evidence and metric deltas:
Regression axes:
Known limitation or blocker:
Next decision:
```
