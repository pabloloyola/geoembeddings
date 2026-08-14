# Command reference

Run commands from the repository root with `uv run geoembed`. Relative default
configuration paths are resolved from the current working directory.

This reference is intentionally organized one-to-one with the subcommands
registered in `src/geoembeddings/cli.py`. In the tables below, `RUN_DIR` is a
dataset root, `EXPERIMENT_DIR` is a modeling root, and `PAIR_DIR` is the parent
of the canonical `pair_manifest.json`. A protected evaluator may read `truth/`;
an **observed-only** command is structurally denied that input. Simulator
commands create protected truth but do not expose it to modeling code.

## Exit status and path contract

Successful commands print a JSON summary to stdout. Reported scientific
`unavailable` status is also a successful result. Failures use exit code `2`
for schema/identity errors, `3` for immutable-output conflicts, and `4` for
missing prerequisites; unexpected failures use `1`.

Public dataset commands accept the parent `RUN_DIR`, never `RUN_DIR/observed`
or `RUN_DIR/truth`. Modeling commands accept `EXPERIMENT_DIR`. Internal names
are resolved by `layout.py`; paths below are relative to those roots.

## Path model

The following trees are an index of canonical path resolvers, not a promise that
one command creates every artifact. Bracketed annotations name the owning
command and whether the artifact is always or conditionally written. Dataset
table names come from `OBSERVED_FILES` and `TRUTH_FILES` in `contract.py`; the
remaining paths come from `DatasetLayout`, `PairLayout`, `ExperimentLayout`,
`PrivacyEvidenceLayout`, or `UtilityReportLayout` in `layout.py`. Commands may
consume artifacts owned by another command, but they must resolve the same path.

`RUN_DIR` is the root accepted by `--run-dir`:

```text
RUN_DIR/
├── config.resolved.yaml                         [simulate; always]
├── manifest.json                                [simulate; always]
├── validation_report.json                       [simulate; always; simulator-owned structural report]
├── deep_validation_report.json                  [validate; conditional: default output only]
├── observed/
│   ├── users_observed.csv.gz                    [simulate; always in contracts 1.0 and 2.0]
│   ├── observed_events.csv.gz                   [simulate; always in contracts 1.0 and 2.0]
│   ├── poi_catalog.csv.gz                       [simulate; contract 2.0 only]
│   ├── recommendation_requests.csv.gz           [simulate; contract 2.0 only]
│   ├── impressions.csv.gz                       [simulate; contract 2.0 only]
│   └── interactions.csv.gz                      [simulate; contract 2.0 only]
└── truth/                                       [simulate; protected, evaluator-only consumption]
    ├── user_latents.csv.gz                      [simulate; always]
    ├── episodes_truth.csv.gz                    [simulate; always]
    ├── candidate_sets.csv.gz                    [simulate; always]
    ├── choices_truth.csv.gz                     [simulate; always]
    ├── trajectories_truth.csv.gz                [simulate; always]
    └── observation_process.csv.gz               [simulate; always]
```

Contract 1.0 compatibility is event-only: its public contract contains only
`users_observed.csv.gz` and `observed_events.csv.gz`. Readers do not synthesize
the four recommendation tables. Change-intervention simulations may additionally
write protected change-point truth; that simulator extension is not one of the
six `TRUTH_FILES` entries in the dataset contract.

`EXPERIMENT_DIR` is the root accepted by `--experiment-dir`. `KIND` is
`baseline` or `learned`, `MODEL` is the selected ranking model, and `VIEW_ID` is
the configured robustness view identifier:

```text
EXPERIMENT_DIR/
├── prepared/
│   ├── config.resolved.yaml                     [prepare; always]
│   ├── prepared_metadata.json                   [prepare; always]
│   └── vocabularies.json                        [prepare; always]
├── model/
│   ├── best_model.pt                            [train; learned only]
│   ├── training_report.json                     [train; learned only]
│   └── training_participation.json              [train; learned only]
├── statistical_baseline.npz                     [baseline; baseline only]
├── embeddings.npz                               [export; learned only]
├── dense_statistical_baseline.npz               [export-dense; conditional: baseline]
├── dense_embeddings.npz                         [export-dense; conditional: learned]
├── baseline_evaluation.json                     [evaluate; conditional: default baseline]
├── evaluation.json                              [evaluate; conditional: default learned]
├── baseline_episode_response.json               [evaluate; conditional: episodes baseline]
├── episode_response.json                        [evaluate; conditional: episodes learned]
├── baseline_transfer_evaluation.json            [evaluate; conditional: transfer baseline]
├── learned_transfer_evaluation.json             [evaluate; conditional: transfer learned]
├── baseline_temporal_routine.json                [evaluate; conditional: temporal-routine baseline]
├── learned_temporal_routine.json                 [evaluate; conditional: temporal-routine learned]
├── baseline_reliability.json                    [evaluate; conditional: reliability baseline]
├── reliability.json                             [evaluate; conditional: reliability learned]
├── reliability/
│   └── calibration.json                         [calibrate-reliability; conditional: this root is OUTPUT_DIR]
├── robustness/
│   ├── baseline/
│   │   └── VIEW_ID.npz                          [robustness; conditional: each selected baseline view]
│   ├── learned/
│   │   └── VIEW_ID.npz                          [robustness; conditional: each selected learned view]
│   ├── baseline_robustness.json                 [robustness; conditional: baseline kind]
│   └── learned_robustness.json                  [robustness; conditional: learned kind]
├── ranking/
│   ├── MODEL.npz                                [rank; conditional: selected model predictions]
│   ├── MODEL.json                               [rank; conditional: selected model report]
│   ├── frozen_embedding_checkpoint.npz          [rank; conditional: frozen_embedding model]
│   ├── exposure_aware_checkpoint.npz            [rank; conditional: exposure_aware model]
│   ├── transfer_slices.json                     [evaluate-ranking; conditional]
│   └── visualization/                           [visualize-ranking; conditional]
│       ├── metadata.json
│       └── ranking_explanation.html
├── benchmarks/
│   ├── offline.json                             [benchmark; always when benchmark runs]
│   ├── online.json                              [benchmark; always when benchmark runs]
│   └── online_workload.json                     [benchmark; always when benchmark runs]
├── comparison/
│   ├── embedding_comparison.json                [compare; standard comparison only]
│   ├── embedding_comparison.md                  [compare; standard comparison only]
│   ├── factorized_comparison.json               [compare; conditional: factorized matrix]
│   └── factorized_comparison.md                 [compare; conditional: factorized matrix]
├── visualization/
    ├── baseline/                                [visualize-embeddings; conditional: sparse baseline]
    ├── baseline_dense/                          [visualize-embeddings; conditional: dense baseline]
    ├── learned/                                 [visualize-embeddings; conditional: sparse learned]
│   └── learned_dense/                           [visualize-embeddings; conditional: dense learned]
└── user_journey/
    ├── USER_ID.html                             [user-journey; always]
    └── USER_ID.metadata.json                    [user-journey; always]
```

Each visualization leaf contains the command-owned projection metadata, CSV and
NPZ coordinates, small-multiples image, and trajectories image described under
`visualize-embeddings`. A normal `pipeline` writes only the mode-specific core
preparation, representation, and default-evaluation subset; it does not create
the dense, supplemental evaluation, calibration, robustness, ranking,
benchmark, comparison, or visualization branches.

`PAIR_DIR` is the root resolved by `PairLayout` (and the parent of the canonical
pair manifest):

```text
PAIR_DIR/
├── pair_manifest.json                           [simulate-pair or pair-manifest; always for a declared pair]
├── pair_integrity.json                          [simulate-pair or validate-pair; conditional on validation]
├── counterfactual_comparison.json               [evaluate-pair; conditional]
├── counterfactual_comparison.md                 [evaluate-pair; conditional]
├── change_evaluation.json                       [evaluate-change; conditional]
├── change_evaluation.md                         [evaluate-change; conditional]
├── ranking/
│   └── exposure_counterfactual.json             [evaluate-pair; conditional: ranking inputs supplied]
└── audits/
    ├── nonstationarity.json                     [audit-nonstationarity; conditional: PAIR_DIR used as OUTPUT_DIR]
    ├── nonstationarity.md                       [audit-nonstationarity; conditional: PAIR_DIR used as OUTPUT_DIR]
    ├── privacy.json                              [audit-privacy; conditional: PAIR_DIR used as OUTPUT_DIR]
    └── privacy.md                               [audit-privacy; conditional: PAIR_DIR used as OUTPUT_DIR]
```

The privacy command's two input roots use their own authoritative layouts;
`REPORT_NAME` must be a single safe filename stem:

```text
PRIVACY_EVIDENCE_DIR/
└── evidence_index.json                          [privacy evidence producer; required audit-privacy input]

UTILITY_REPORT_DIR/
└── REPORT_NAME.json                             [upstream utility evaluator; named audit-privacy input]
```

Calibration and audits accept generic `--output-dir` roots, which are resolved
with `ExperimentLayout` and `PairLayout` respectively even when the root is not
a modeling experiment or pair used by another command:

```text
CALIBRATION_OUTPUT_DIR/
└── reliability/
    └── calibration.json                         [calibrate-reliability; always when command runs]

AUDIT_OUTPUT_DIR/
└── audits/
    ├── nonstationarity.json                     [audit-nonstationarity; command-specific]
    ├── nonstationarity.md                       [audit-nonstationarity; command-specific]
    ├── privacy.json                              [audit-privacy; command-specific]
    └── privacy.md                               [audit-privacy; command-specific]
```

Only the two files owned by the selected audit command are written; the audit
tree does not imply that either audit invokes the other.


## Command and artifact quick reference

Paths in both tables are relative to the root named in **Storage root**. “Protected”
means evaluator-only truth access; “derived protected” means the command reads an
authenticated protected report rather than opening raw `truth/`. Follow the command
links for complete prerequisites and collision rules.

| Command | Reads from | Primary artifacts produced | Storage root | Truth access | Overwrite behavior | Typical next command |
|---|---|---|---|---|---|---|
| [`inspect-evidence`](#inspect-evidence) | `INDEX_DIR/**/*.json` | stdout only | none | None (documentation only) | No output and no `--overwrite` | None; review inspection result |
| [`simulate`](#simulate) | Simulation YAML | `manifest.json`, `validation_report.json`, `observed/*.csv.gz`, `truth/*.csv.gz` | `RUN_DIR` | Creates truth as trusted simulator | Existing nonempty run fails unless `--overwrite` | [`validate`](#validate) |
| [`simulate-pair`](#simulate-pair) | Simulation YAML | Two complete run trees; `pair_manifest.json`; `pair_integrity.json` | `REFERENCE_RUN_DIR`, `INTERVENTION_RUN_DIR`, `PAIR_DIR` | Creates and validates truth | No overwrite; all three roots must be new | [`prepare`](#prepare) for both runs, then [`evaluate-pair`](#evaluate-pair) or [`evaluate-change`](#evaluate-change) |
| [`validate`](#validate) | Complete `RUN_DIR`, including `observed/` and `truth/` | `deep_validation_report.json` (or `OUTPUT`) | `RUN_DIR` by default | Protected | Selected report is replaceable; no `--overwrite` | [`prepare`](#prepare) |
| [`pair-manifest`](#pair-manifest) | Two complete `RUN_DIR` trees | `pair_manifest.json` | `PAIR_DIR` | Protected contract authentication | Existing manifest fails unless `--overwrite` | [`validate-pair`](#validate-pair) |
| [`validate-pair`](#validate-pair) | `pair_manifest.json` and both complete run trees | `pair_integrity.json` | `PAIR_DIR` | Protected | Canonical report is refreshed; no `--overwrite` | [`evaluate-pair`](#evaluate-pair) or [`evaluate-change`](#evaluate-change) |
| [`evaluate-pair`](#evaluate-pair) | Pair manifest/integrity, four experiment exports; optional ranking predictions/reports | [Pair reports](#pair-and-audit-report-artifacts), optionally `ranking/exposure_counterfactual.json` | `PAIR_DIR` | Protected | Selected existing output fails unless `--overwrite` | [`compare`](#compare), evidence review, or [`audit-privacy`](#audit-privacy) |
| [`evaluate-change`](#evaluate-change) | Pair manifest/integrity, four dense-export experiments, change-point truth | `change_evaluation.{json,md}` | `PAIR_DIR` | Protected | Either report fails unless `--overwrite`; pair replaced together | [`audit-nonstationarity`](#audit-nonstationarity) |
| [`audit-nonstationarity`](#audit-nonstationarity) | Three compatible `change_evaluation.json` reports | [`audits/nonstationarity.{json,md}`](#pair-and-audit-report-artifacts) | `OUTPUT_DIR` | Derived protected | Existing reports fail unless `--overwrite` | Evidence review |
| [`audit-privacy`](#audit-privacy) | Run identity, named experiment exports, evidence index, utility reports | [`audits/privacy.{json,md}`](#pair-and-audit-report-artifacts) | `OUTPUT_DIR` | Protected | Existing reports fail unless `--overwrite` | Evidence review |
| [`calibrate-reliability`](#calibrate-reliability) | Observed events and named dense-export experiments | `reliability/calibration.json` | `OUTPUT_DIR` | None (observed-only) | Existing report fails unless `--overwrite` | Evidence review |
| [`prepare`](#prepare) | `RUN_DIR/manifest.json`, `RUN_DIR/observed/`, embedding YAML | `prepared/config.resolved.yaml`, `prepared/prepared_metadata.json`, `prepared/vocabularies.json` | `EXPERIMENT_DIR` | None (observed-only) | Immutable; no `--overwrite`; use a new experiment | [`baseline`](#baseline) or [`train`](#train) |
| [`train`](#train) | Observed events and `prepared/*` | `model/best_model.pt`, `model/training_report.json`, `model/training_participation.json` | `EXPERIMENT_DIR` | None (observed-only) | Conflicting participation fails; no `--overwrite` | [`export`](#export) and [`export-dense`](#export-dense) |
| [`baseline`](#baseline) | Observed events and `prepared/*` | `statistical_baseline.npz` | `EXPERIMENT_DIR` | None (observed-only) | Canonical export is replaceable; no `--overwrite` | [`evaluate`](#evaluate) |
| [`export`](#export) | Observed events, `prepared/*`, `model/best_model.pt` | `embeddings.npz` | `EXPERIMENT_DIR` | None (observed-only) | Canonical export is replaceable; no `--overwrite` | [`evaluate`](#evaluate) |
| [`export-dense`](#export-dense) | Observed events, `prepared/*`; learned checkpoint when selected | [`dense_embeddings.npz` or `dense_statistical_baseline.npz`](#dense-export-artifacts) | `EXPERIMENT_DIR` | None (observed-only) | Regenerates only selected kind; no `--overwrite` | [`visualize-embeddings`](#visualize-embeddings), an [evaluation mode](#evaluation-mode-artifacts), or [`rank`](#rank) |
| [`visualize-embeddings`](#visualize-embeddings) | Selected cutoff or dense export | `visualization/KIND[_dense]/{projection_metadata.json,projections.csv,projections.npz,small_multiples.FORMAT,trajectories.FORMAT}` | `EXPERIMENT_DIR` | None (observed-only) | Any target fails unless `--overwrite`; selected leaf only | Inspect plots (terminal artifact) |
| `user-journey` | Observed tables, authenticated dense export, selected ranking reports; episodes only with `--evaluator-truth` | `user_journey/USER_ID.{html,metadata.json}` | `EXPERIMENT_DIR` | Optional, explicit protected episode access | Existing pair fails unless `--overwrite` | Inspect the deterministic journey report |
| [`evaluate`](#evaluate) | Selected exports and prepared metadata; mode-dependent observed/truth inputs | [Mode-specific evaluation report](#evaluation-mode-artifacts) | `EXPERIMENT_DIR` | Mode-dependent: protected except transfer/reliability | Reliability requires `--overwrite`; other selected reports are replaceable | [`compare`](#compare), [`benchmark`](#benchmark), or evidence review |
| [`benchmark`](#benchmark) | Observed events, prepared metadata, available exports/checkpoint | `benchmarks/{offline.json,online_workload.json,online.json}` | `EXPERIMENT_DIR` | None (observed-only) | Existing reports fail unless `--overwrite` | Evidence review |
| [`rank`](#rank) | Dataset-2.0 observed tables; dense learned export for learned models | [Model-specific ranking predictions, report, and optional checkpoint](#ranking-model-artifacts) | `EXPERIMENT_DIR` | None (observed-only) | Selected model outputs fail unless `--overwrite` | [`evaluate-ranking`](#evaluate-ranking) or protected [`evaluate-pair`](#evaluate-pair) |
| [`evaluate-ranking`](#evaluate-ranking) | Observed recommendation tables and selected ranking model outputs | `ranking/transfer_slices.json` | `EXPERIMENT_DIR` | None (observed-only) | Existing report fails unless `--overwrite` | Evidence review |
| [`visualize-ranking`](#visualize-ranking) | Observed recommendation tables and four authenticated ranking controls | `ranking/visualization/{metadata.json,ranking_explanation.html}` | `EXPERIMENT_DIR` | None (observed-only) | Existing output fails unless `--overwrite` | R9 inspection |
| [`robustness`](#robustness) | Observed events, prepared metadata, original export; learned checkpoint if applicable | `robustness/KIND/VIEW_ID.npz`, `robustness/KIND_robustness.json` | `EXPERIMENT_DIR` | Protected after observed-only view construction | Immutable writer checks; no `--overwrite` | [`compare`](#compare) |
| [`compare`](#compare) | Matched baseline/learned experiments, optional supplemental reports and factorized experiments | [Standard or factorized comparison reports](#comparison-report-artifacts) | `OUTPUT_DIR` | Protected | Immutable; no `--overwrite` | Evidence review/decision record |
| [`pipeline`](#pipeline) | Simulation and embedding YAML | Fresh run plus prepared artifacts, one representation, and its default evaluation | `RUN_DIR` and `EXPERIMENT_DIR` | Mixed by stage | Run target honors `--overwrite`; downstream immutable rules remain | Run the other representation, then [`compare`](#compare) |

### Dense export artifacts

| Selection | Artifact | Meaning |
|---|---|---|
| `export-dense --kind learned` | `EXPERIMENT_DIR/dense_embeddings.npz` | Learned components at observed event timestamps. |
| `export-dense --kind baseline` | `EXPERIMENT_DIR/dense_statistical_baseline.npz` | Statistical vectors at observed event timestamps. |

### Evaluation mode artifacts

| `evaluate` selection | Artifact | Boundary |
|---|---|---|
| default, `--kind learned` | `EXPERIMENT_DIR/evaluation.json` | Protected |
| default, `--kind baseline` | `EXPERIMENT_DIR/baseline_evaluation.json` | Protected |
| `--episodes` | `EXPERIMENT_DIR/episode_response.json` or `baseline_episode_response.json` | Protected |
| `--transfer` | `EXPERIMENT_DIR/KIND_transfer_evaluation.json` | Observed-only |
| `--temporal-routine` | `EXPERIMENT_DIR/KIND_temporal_routine.json` | Protected |
| `--reliability` | `EXPERIMENT_DIR/reliability.json` or `baseline_reliability.json` | Observed-only |

### Ranking model artifacts

Every model writes `EXPERIMENT_DIR/ranking/MODEL.npz` predictions and
`EXPERIMENT_DIR/ranking/MODEL.json` observed metrics.

| `--model` | Additional principal input | Additional artifact |
|---|---|---|
| `popularity` | Observed catalog/request/interaction contract | None |
| `nearest` | Observed request and candidate coordinates | None |
| `category_preference` | Observed histories and candidate categories | None |
| `frozen_embedding` | Learned dense export and observed controls | `EXPERIMENT_DIR/ranking/frozen_embedding_checkpoint.npz` |
| `exposure_aware` | Learned dense export, observed impressions/interactions, and ranking YAML | `EXPERIMENT_DIR/ranking/exposure_aware_checkpoint.npz` |

See the detailed [`rank`](#rank) section for shared prerequisites, identities,
and collision behavior.

### Comparison report artifacts

A standard comparison writes `OUTPUT_DIR/embedding_comparison.{json,md}`. A
factorized matrix instead writes `OUTPUT_DIR/factorized_comparison.{json,md}`
when `--factorized-experiment` is supplied.

### Pair and audit report artifacts

Paired representation evaluation writes
`PAIR_DIR/counterfactual_comparison.{json,md}` and may write
`PAIR_DIR/ranking/exposure_counterfactual.json`; change evaluation writes
`PAIR_DIR/change_evaluation.{json,md}`. Audits write
`OUTPUT_DIR/audits/nonstationarity.{json,md}` or
`OUTPUT_DIR/audits/privacy.{json,md}`.

## Where did this file come from?

Use the owning command to determine which root and identity checks apply. A
filename alone is not evidence that artifacts from different roots are
comparable.

| Common artifact | Produced by | Location | Usually consumed by / next step |
|---|---|---|---|
| `manifest.json` | [`simulate`](#simulate) or [`simulate-pair`](#simulate-pair) | `RUN_DIR/manifest.json` | `validate`, `prepare`, and identity-authenticating evaluators |
| `deep_validation_report.json` | [`validate`](#validate) (also a [`pipeline`](#pipeline) stage) | `RUN_DIR/deep_validation_report.json` | Integrity review, then `prepare` |
| `prepared_metadata.json` | [`prepare`](#prepare) (also a [`pipeline`](#pipeline) stage) | `EXPERIMENT_DIR/prepared/prepared_metadata.json` | All representation/export/evaluation stages |
| `best_model.pt` | [`train`](#train) (also learned [`pipeline`](#pipeline)) | `EXPERIMENT_DIR/model/best_model.pt` | `export`, learned `export-dense`, robustness, benchmark, or learned rankers |
| `embeddings.npz` | [`export`](#export) (also learned [`pipeline`](#pipeline)) | `EXPERIMENT_DIR/embeddings.npz` | Learned evaluation, visualization, robustness, benchmark, and compare |
| `statistical_baseline.npz` | [`baseline`](#baseline) (also baseline [`pipeline`](#pipeline)) | `EXPERIMENT_DIR/statistical_baseline.npz` | Baseline evaluation, visualization, robustness, benchmark, and compare |
| Dense exports | [`export-dense`](#export-dense) | [`EXPERIMENT_DIR/dense_embeddings.npz` or `dense_statistical_baseline.npz`](#dense-export-artifacts) | Dense visualization, temporal/episode/change evaluation, calibration, or ranking |
| Evaluation reports | [`evaluate`](#evaluate) | [Mode-specific files in `EXPERIMENT_DIR`](#evaluation-mode-artifacts) | Compare, benchmark, audits, or evidence review depending on mode |
| Ranking reports | [`rank`](#rank) and [`evaluate-ranking`](#evaluate-ranking) | [`EXPERIMENT_DIR/ranking/MODEL.{npz,json}` and `transfer_slices.json`](#ranking-model-artifacts) | Ranking transfer evaluation, optional protected pair evaluation, or evidence review |
| Comparison reports | [`compare`](#compare) | [`OUTPUT_DIR/*_comparison.{json,md}`](#comparison-report-artifacts) | Terminal decision/evidence record |
| Pair reports | [`evaluate-pair`](#evaluate-pair) or [`evaluate-change`](#evaluate-change) | [`PAIR_DIR/{counterfactual_comparison,change_evaluation}.{json,md}` plus optional ranking report](#pair-and-audit-report-artifacts) | Compare, nonstationarity/privacy audit, or evidence review |
| Audit reports | [`audit-nonstationarity`](#audit-nonstationarity) or [`audit-privacy`](#audit-privacy) | [`OUTPUT_DIR/audits/{nonstationarity,privacy}.{json,md}`](#pair-and-audit-report-artifacts) | Terminal evidence review |

## `inspect-evidence`

### Purpose and information boundary

Read-only verification of documentation evidence indexes. **Documentation-only:** it reads neither `observed/` nor `truth/`.

### Arguments

`--index-dir` is optional and defaults to `docs/artifacts`.

### Prerequisites consumed

Evidence index JSON files below the explicitly named `INDEX_DIR`; no run, experiment, or pair artifact is consumed.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `None` | — | Prints inspection results to stdout only. | Never |

### Existing output and overwrite

No artifact is written, so there is no collision and no `--overwrite` option.

### Minimal example

```bash
uv run geoembed inspect-evidence --index-dir docs/artifacts
```

### Follow-up consumers

None; use the stdout result to verify documentation evidence.

## `simulate`

### Purpose and information boundary

Generate one synthetic dataset. **Trusted simulator producer:** it writes both public observations and protected truth; it is not an observed-only model command or a protected evaluator.

### Arguments

Required: `--run-dir RUN_DIR`. Important options: `--config`, `--users`, `--days`, `--start-date`, `--seed`, `--scenario`, `--[no-]full-kanto`, and `--overwrite`.

### Prerequisites consumed

Simulation YAML (default `configs/simulation/kanto_v1.yaml`); no prior run artifacts.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `config.resolved.yaml` | YAML | Resolved generator configuration and seeds. | Always |
| `manifest.json` | JSON | Dataset contract, counts, hashes, and identity lineage. | Always |
| `validation_report.json` | JSON | Simulator structural validation summary. | Always |
| `observed/users_observed.csv.gz` | gzip CSV | Public user attributes. | Always |
| `observed/observed_events.csv.gz` | gzip CSV | Public event history. | Always |
| `observed/poi_catalog.csv.gz` | gzip CSV | Public recommendation catalog (contract 2.0 only). | Always for contract 2.0 |
| `observed/recommendation_requests.csv.gz` | gzip CSV | Public request contexts (contract 2.0 only). | Always for contract 2.0 |
| `observed/impressions.csv.gz` | gzip CSV | Public exposures (contract 2.0 only). | Always for contract 2.0 |
| `observed/interactions.csv.gz` | gzip CSV | Public responses (contract 2.0 only). | Always for contract 2.0 |
| `truth/user_latents.csv.gz` | gzip CSV | Protected persistent traits. | Always |
| `truth/episodes_truth.csv.gz` | gzip CSV | Protected episode state. | Always |
| `truth/candidate_sets.csv.gz` | gzip CSV | Protected candidate/utility records. | Always |
| `truth/choices_truth.csv.gz` | gzip CSV | Protected choices. | Always |
| `truth/trajectories_truth.csv.gz` | gzip CSV | Protected noiseless trajectories. | Always |
| `truth/observation_process.csv.gz` | gzip CSV | Protected observation mechanism. | Always |
| `truth/change_points_truth.csv.gz` | gzip CSV | Protected intervention change points. | Conditional: change intervention only |

The four recommendation tables are contract-2.0-only. Event-only contract 1.0
retains only the two public user/event tables, and compatible readers do not
fabricate recommendation data.

### Existing output and overwrite

A nonempty target is immutable unless `--overwrite` is supplied; overwrite applies only to the validated target run.

### Minimal example

```bash
uv run geoembed simulate --run-dir runs/smoke --users 10 --days 2 --seed 1729
```

### Follow-up consumers

`validate` consumes the run; `prepare` and other observed-only stages consume `observed/`; protected evaluators consume `truth/`.

## `simulate-pair`

### Purpose and information boundary

Generate two matched runs, declare their relationship, and validate field-level integrity. **Trusted simulator plus protected validation:** it creates and authenticates truth on both sides.

### Arguments

Required: `--intervention`, `--reference-run-dir`, `--intervention-run-dir`, and `--pair-dir`. Important options: `--config`, `--users`, `--days`, and `--seed`.

### Prerequisites consumed

Simulation YAML; all three output roots must be new. No existing run artifact is a prerequisite.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `REFERENCE_RUN_DIR/*` | mixed YAML/JSON/gzip CSV | Complete run tree listed under `simulate`. | Always |
| `INTERVENTION_RUN_DIR/*` | mixed YAML/JSON/gzip CSV | Matched complete run tree, including conditional change-point truth. | Always |
| `PAIR_DIR/pair_manifest.json` | JSON | Authenticated pair declaration and allowed changes. | Always |
| `PAIR_DIR/pair_integrity.json` | JSON | Field-level matching and integrity report. | Always |

### Existing output and overwrite

There is no `--overwrite`; any conflicting immutable run or pair output fails. Choose three new roots.

### Minimal example

```bash
uv run geoembed simulate-pair --intervention exposure --reference-run-dir runs/ref --intervention-run-dir runs/exposed --pair-dir pairs/exposure --users 10 --days 2 --seed 1729
```

### Follow-up consumers

Each run can feed `prepare`; the manifest feeds `validate-pair`, `evaluate-pair`, or `evaluate-change`; the integrity report gates paired evaluators.

## `validate`

### Purpose and information boundary

Deep-validate one simulation. **Protected evaluator:** it opens the complete run, including `truth/`.

### Arguments

Required: `--run-dir RUN_DIR`. `--output` optionally names the JSON destination.

### Prerequisites consumed

The complete `RUN_DIR` contract: `manifest.json`, every required `observed/*.csv.gz`, and every required `truth/*.csv.gz`.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `deep_validation_report.json` | JSON | Deep integrity and behavioral diagnostics at the default destination. | Always when `--output` is omitted |
| `OUTPUT` | JSON | Same report at the explicit destination. | Conditional: `--output` supplied |

### Existing output and overwrite

No `--overwrite` option is exposed; the report writer replaces the selected report path after validation. It never deletes source artifacts.

### Minimal example

```bash
uv run geoembed validate --run-dir runs/smoke
```

### Follow-up consumers

A passing report is a prerequisite/evidence for `prepare`, manual workflows, and interpreting later evaluator results.

## `pair-manifest`

### Purpose and information boundary

Declare two existing simulator runs as a protected matched pair. **Protected contract command:** it authenticates run-level observed and truth hashes.

### Arguments

Required: `--reference-run-dir`, `--intervention-run-dir`, and canonical `--output PAIR_DIR/pair_manifest.json`; optional `--overwrite`.

### Prerequisites consumed

Both complete run roots, including resolved configs, manifests, observed tables, truth tables, and identity/source hashes.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `PAIR_DIR/pair_manifest.json` | JSON | Immutable pair identities, intervention, matching keys, lineage, and allowed changes. | Always |

### Existing output and overwrite

An existing manifest fails unless `--overwrite` is explicit; overwrite replaces only the authenticated manifest.

### Minimal example

```bash
uv run geoembed pair-manifest --reference-run-dir runs/ref --intervention-run-dir runs/exposed --output pairs/exposure/pair_manifest.json
```

### Follow-up consumers

`validate-pair`, `evaluate-pair`, and `evaluate-change` consume the manifest.

## `validate-pair`

### Purpose and information boundary

Perform field-level validation of a declared pair. **Protected evaluator:** it reads and compares both runs, including protected truth.

### Arguments

Required: `--pair-manifest PAIR_DIR/pair_manifest.json`.

### Prerequisites consumed

The pair manifest and the complete referenced run trees with hashes matching the declaration.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `PAIR_DIR/pair_integrity.json` | JSON | Invariant matches, allowed differences, identity coverage, and pass/fail status. | Always on completed validation |

### Existing output and overwrite

No `--overwrite` flag is exposed; rerunning refreshes the canonical integrity report but never modifies either run or the manifest.

### Minimal example

```bash
uv run geoembed validate-pair --pair-manifest pairs/exposure/pair_manifest.json
```

### Follow-up consumers

A current passing report gates `evaluate-pair` and `evaluate-change`.

## `evaluate-pair`

### Purpose and information boundary

Evaluate R5/R7 representation response across an authenticated pair and optionally exposure-aware ranking. **Protected evaluator:** it reads protected paired truth only after integrity checks.

### Arguments

Required: `--pair-manifest`, two roots after `--baseline-experiment-dir`, and two after `--learned-experiment-dir`. Important options: `--config`, `--overwrite`, plus paired `--ranking-predictions` and `--ranking-reports`.

### Prerequisites consumed

`PAIR_DIR/pair_manifest.json`, `PAIR_DIR/pair_integrity.json`, and cutoff exports plus prepared metadata from all four experiment roots. Optional ranking inputs require two prediction NPZs and two report JSONs.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `PAIR_DIR/counterfactual_comparison.json` | JSON | Machine-readable matched representation effects. | Always |
| `PAIR_DIR/counterfactual_comparison.md` | Markdown | Human-readable matched representation report. | Always |
| `PAIR_DIR/ranking/exposure_counterfactual.json` | JSON | Protected utility-regret/probability recovery comparison. | Conditional: both ranking option pairs supplied |

### Existing output and overwrite

Any selected existing output fails unless `--overwrite`; optional ranking arguments must be supplied together.

### Minimal example

```bash
uv run geoembed evaluate-pair --pair-manifest pairs/exposure/pair_manifest.json --baseline-experiment-dir experiments/ref-b experiments/int-b --learned-experiment-dir experiments/ref-l experiments/int-l
```

### Follow-up consumers

`compare` may incorporate paired reports; evidence indexing and scientific review consume the JSON/Markdown outputs. The optional ranking output is a utility input for `audit-privacy` only when named by its config/report root.

## `evaluate-change`

### Purpose and information boundary

Evaluate adaptation, recovery, forgetting, and drift on an authenticated change pair. **Protected evaluator:** it reads `truth/change_points_truth.csv.gz`.

### Arguments

Required: `--pair-manifest`, two baseline experiment roots, and two learned experiment roots; optional `--overwrite`.

### Prerequisites consumed

Current passing pair manifest/integrity report; dense baseline and learned exports and prepared metadata for reference and intervention experiments; protected change-point truth in both runs.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `PAIR_DIR/change_evaluation.json` | JSON | Relative-day matched-control curves, metrics, coverage, and censoring. | Always |
| `PAIR_DIR/change_evaluation.md` | Markdown | Human-readable change evaluation. | Always |

### Existing output and overwrite

Either existing output causes failure unless `--overwrite`, which replaces the two regular outputs together.

### Minimal example

```bash
uv run geoembed evaluate-change --pair-manifest pairs/trip/pair_manifest.json --baseline-experiment-dir experiments/ref-b experiments/trip-b --learned-experiment-dir experiments/ref-l experiments/trip-l
```

### Follow-up consumers

Three compatible reports (no-change, temporary, sustained) feed `audit-nonstationarity`.

## `audit-nonstationarity`

### Purpose and information boundary

Synthesize the three authenticated R11 change conditions. **Protected evaluator-derived command:** it consumes protected reports, not raw `truth/`, and retains their protected classification.

### Arguments

Required: `--no-change-report`, `--temporary-report`, `--sustained-report`, and `--output-dir`. Important options: thresholds and `--overwrite`.

### Prerequisites consumed

Three compatible `geoembeddings-change-evaluation/2.0` JSON reports with identical users, cutoffs, source/preparation lineage, components, relative-day definition, and censoring rules.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `OUTPUT_DIR/audits/nonstationarity.json` | JSON | Canonical metrics, gates, coverage, and limitations. | Always |
| `OUTPUT_DIR/audits/nonstationarity.md` | Markdown | Human-readable R11 audit. | Always |

### Existing output and overwrite

Existing outputs fail unless `--overwrite`; both are replaced as one validated report set.

### Minimal example

```bash
uv run geoembed audit-nonstationarity --no-change-report pairs/no/change_evaluation.json --temporary-report pairs/trip/change_evaluation.json --sustained-report pairs/sustained/change_evaluation.json --output-dir experiments/r11
```

### Follow-up consumers

Evidence indexes and scientific review consume both audit renderings; no CLI stage consumes them automatically.

## `audit-privacy`

### Purpose and information boundary

Run the authenticated R12 diagnostic-control privacy audit. **Protected evaluator:** it authenticates protected evidence and utility reports against the run and frozen exports; it is not a training command.

### Arguments

Required: `--run-dir`, repeated `--experiment-dir NAME=ROOT`, `--evidence-dir`, `--utility-report-dir`, and `--output-dir`. Important options: `--config` and `--overwrite`.

### Prerequisites consumed

The run manifest/source identities; each experiment’s prepared metadata and dense export; `EVIDENCE_DIR/evidence_index.json` and referenced attack evidence; configured named JSON reports in `UTILITY_REPORT_DIR`.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `OUTPUT_DIR/audits/privacy.json` | JSON | Authoritative diagnostic-control privacy/utility audit. | Always |
| `OUTPUT_DIR/audits/privacy.md` | Markdown | Human-readable R12 audit and limitations. | Always |

### Existing output and overwrite

Existing outputs fail unless `--overwrite`; replacement occurs only after all identities authenticate.

### Minimal example

```bash
uv run geoembed audit-privacy --run-dir runs/smoke --experiment-dir baseline=experiments/base --evidence-dir evidence/privacy --utility-report-dir evidence/utility --output-dir experiments/r12
```

### Follow-up consumers

Documentation evidence indexes and scientific review consume the reports; no model stage may consume them.

## `calibrate-reliability`

### Purpose and information boundary

Fit and test diagnostic-control reliability calibration on held-out users. **Observed-only:** it reads observed histories and authenticated exports, never `truth/`.

### Arguments

Required: `--run-dir`, repeated `--experiment-dir NAME=ROOT`, and `--output-dir`. Important options: `--config` and `--overwrite`.

### Prerequisites consumed

`RUN_DIR/manifest.json` and `RUN_DIR/observed/observed_events.csv.gz`; for each named experiment, prepared metadata and dense export with matching observed-source/preparation identities.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `OUTPUT_DIR/reliability/calibration.json` | JSON | Frozen split, bootstrap uncertainty, calibration fits, bins, and coverage-risk curves. | Always |

### Existing output and overwrite

The report is immutable unless `--overwrite`; mismatched controls or identities fail before replacement.

### Minimal example

```bash
uv run geoembed calibrate-reliability --run-dir runs/smoke --experiment-dir baseline=experiments/base --experiment-dir learned=experiments/learned --output-dir experiments/calibration
```

### Follow-up consumers

Evidence review consumes the report; it is not consumed by `evaluate` or model training.

## `prepare`

### Purpose and information boundary

Fit vocabularies, normalization, cutoffs, and window metadata on training events. **Observed-only:** no truth path is passed.

### Arguments

Required: `--run-dir RUN_DIR` and `--experiment-dir EXPERIMENT_DIR`; important optional `--config`.

### Prerequisites consumed

`RUN_DIR/manifest.json`, public users and observed events under `RUN_DIR/observed/`; the embedding YAML.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/prepared/config.resolved.yaml` | YAML | Resolved preprocessing/model configuration. | Always |
| `EXPERIMENT_DIR/prepared/prepared_metadata.json` | JSON | Source hashes, splits, cutoffs, field order, statistics, and row counts. | Always |
| `EXPERIMENT_DIR/prepared/vocabularies.json` | JSON | Training-only categorical vocabularies in explicit field order. | Always |

### Existing output and overwrite

There is no `--overwrite`; immutable protocol metadata rejects an existing preparation target. Use a new experiment for a changed preparation.

### Minimal example

```bash
uv run geoembed prepare --run-dir runs/smoke --experiment-dir experiments/smoke
```

### Follow-up consumers

`baseline`, `train`, `export`, `export-dense`, all evaluation variants, `rank`, `benchmark`, `robustness`, and `compare` authenticate these files.

## `train`

### Purpose and information boundary

Train the configured sequence encoder. **Observed-only:** training receives public events and prepared artifacts only.

### Arguments

Required: `--run-dir` and `--experiment-dir`; important optional `--config`.

### Prerequisites consumed

Public observed events; all `EXPERIMENT_DIR/prepared/*` files and matching source hashes.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/model/best_model.pt` | PyTorch checkpoint | Best learned weights plus schema/field-order identity. | Always after successful training |
| `EXPERIMENT_DIR/model/training_report.json` | JSON | Epoch metrics, losses, configuration, and checkpoint provenance. | Always |
| `EXPERIMENT_DIR/model/training_participation.json` | JSON | Immutable user/window participation lineage. | Always |

### Existing output and overwrite

No CLI `--overwrite` exists. Training refuses conflicting immutable participation output; use a new experiment rather than silently changing a trained attempt.

### Minimal example

```bash
uv run geoembed train --run-dir runs/smoke --experiment-dir experiments/learned
```

### Follow-up consumers

`export`, learned `export-dense`, learned evaluation/robustness, `benchmark`, and exposure-aware/frozen ranking consume the checkpoint or its exports.

## `baseline`

### Purpose and information boundary

Create the statistical cutoff comparator. **Observed-only:** it uses observed histories and training-fitted preparation.

### Arguments

Required: `--run-dir` and `--experiment-dir`; important optional `--config`.

### Prerequisites consumed

Public observed events and all prepared artifacts with matching source identity.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/statistical_baseline.npz` | compressed NPZ | Row-aligned baseline vectors at train/validation/test cutoffs with schema metadata. | Always |

### Existing output and overwrite

No `--overwrite` flag is exposed; the selected baseline file is regenerated atomically/replaceably while preparation remains untouched.

### Minimal example

```bash
uv run geoembed baseline --run-dir runs/smoke --experiment-dir experiments/base
```

### Follow-up consumers

Baseline `evaluate`, `visualize-embeddings`, `robustness`, `benchmark`, and `compare` consume it; `pipeline --mode baseline` evaluates it immediately.

## `export`

### Purpose and information boundary

Export learned cutoff embeddings from a checkpoint. **Observed-only:** it reads no protected labels.

### Arguments

Required: `--run-dir` and `--experiment-dir`; important optional `--config`.

### Prerequisites consumed

Public observed events, prepared artifacts, and `EXPERIMENT_DIR/model/best_model.pt`.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/embeddings.npz` | compressed NPZ | Learned component vectors at frozen cutoffs with source/checkpoint/schema metadata. | Always |

### Existing output and overwrite

No `--overwrite` option is exposed; rerunning regenerates the canonical export but never changes preparation or checkpoint.

### Minimal example

```bash
uv run geoembed export --run-dir runs/smoke --experiment-dir experiments/learned
```

### Follow-up consumers

Learned `evaluate`, `visualize-embeddings`, `robustness`, `benchmark`, `compare`, and reliability calibration consume it.

## `export-dense`

### Purpose and information boundary

Export baseline or learned embeddings at observed event timestamps. **Observed-only:** all cutoffs derive from public event history.

### Arguments

Required: `--run-dir` and `--experiment-dir`. Important options: `--kind learned|baseline`, `--event-stride`, and `--config`.

### Prerequisites consumed

Public observed events and prepared artifacts; learned mode additionally requires the checkpoint.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/dense_embeddings.npz` | compressed NPZ | Timestamped learned components, always including each user’s first/last event. | Conditional: `--kind learned` |
| `EXPERIMENT_DIR/dense_statistical_baseline.npz` | compressed NPZ | Timestamped statistical vectors. | Conditional: `--kind baseline` |

### Existing output and overwrite

No `--overwrite` option is exposed; rerunning regenerates only the selected dense export. `--event-stride` must be at least one.

### Minimal example

```bash
uv run geoembed export-dense --run-dir runs/smoke --experiment-dir experiments/learned --kind learned --event-stride 1
```

### Follow-up consumers

Dense visualization consumes either output. Episode/temporal evaluation and `evaluate-change` consume dense exports; ranking uses the learned dense export; calibration uses named dense exports.

## `visualize-embeddings`

### Purpose and information boundary

Project and plot a frozen export. **Observed-only:** the command accepts only an experiment root and cannot open dataset truth.

### Arguments

Required: `--experiment-dir`. Important options: `--kind`, `--dense`, `--reference-cutoff`, `--normalization`, `--reducer`, `--seed`, `--format`, UMAP controls, and `--overwrite`.

### Prerequisites consumed

The selected cutoff or dense NPZ export in `EXPERIMENT_DIR`; UMAP additionally requires the visualization extra.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/visualization/KIND[_dense]/projection_metadata.json` | JSON | Reducer, normalization, reference population, hashes, and provenance. | Always |
| `EXPERIMENT_DIR/visualization/KIND[_dense]/projections.csv` | CSV | Human-readable row-aligned 2-D coordinates. | Always |
| `EXPERIMENT_DIR/visualization/KIND[_dense]/projections.npz` | compressed NPZ | Machine-readable row-aligned 2-D coordinates. | Always |
| `EXPERIMENT_DIR/visualization/KIND[_dense]/small_multiples.FORMAT` | PNG or SVG | Per-component/cutoff projection panels. | Always |
| `EXPERIMENT_DIR/visualization/KIND[_dense]/trajectories.FORMAT` | PNG or SVG | Per-user temporal paths. | Always |

### Existing output and overwrite

If any target exists, the command fails unless `--overwrite`; replacement is limited to the selected kind/density directory.

### Minimal example

```bash
uv run --extra viz geoembed visualize-embeddings --experiment-dir experiments/learned --kind learned --reference-cutoff train
```

### Follow-up consumers

The files are terminal exploratory artifacts; no evaluator or training command consumes them.

## `evaluate`

### Purpose and information boundary

Evaluate one frozen representation. **Boundary depends on mode:** default, `--episodes`, and `--temporal-routine` are protected evaluators; `--transfer` and `--reliability` are observed-only. Only one supplemental flag may be selected.

### Arguments

Required: `--run-dir` and `--experiment-dir`. Important options: `--kind`, exactly zero or one of `--episodes`, `--transfer`, `--temporal-routine`, `--reliability`, plus `--config` and supplemental `--overwrite`.

### Prerequisites consumed

Selected cutoff or dense export and prepared metadata. Default also needs truth and, for learned kind, the checkpoint. Episode mode needs protected episodes; temporal-routine needs protected temporal/routine truth. Transfer and reliability consume only observed artifacts.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/evaluation.json` | JSON | Protected main learned probes and geometry diagnostics. | Conditional: default learned |
| `EXPERIMENT_DIR/baseline_evaluation.json` | JSON | Protected main baseline probes. | Conditional: default baseline |
| `EXPERIMENT_DIR/episode_response.json` | JSON | Protected learned episode response. | Conditional: `--episodes --kind learned` |
| `EXPERIMENT_DIR/baseline_episode_response.json` | JSON | Protected baseline episode response. | Conditional: `--episodes --kind baseline` |
| `EXPERIMENT_DIR/KIND_transfer_evaluation.json` | JSON | Observed-only spatial transfer slices. | Conditional: `--transfer` |
| `EXPERIMENT_DIR/KIND_temporal_routine.json` | JSON | Protected temporal/routine diagnostics. | Conditional: `--temporal-routine` |
| `EXPERIMENT_DIR/reliability.json` | JSON | Observed-only learned reliability diagnostics. | Conditional: `--reliability --kind learned` |
| `EXPERIMENT_DIR/baseline_reliability.json` | JSON | Observed-only baseline reliability diagnostics. | Conditional: `--reliability --kind baseline` |

### Existing output and overwrite

Supplemental modes select separate files. `--overwrite` is enforced for reliability; other evaluator writers replace their selected canonical report and do not alter source exports.

### Minimal example

```bash
uv run geoembed evaluate --run-dir runs/smoke --experiment-dir experiments/learned --kind learned
```

### Follow-up consumers

`compare` consumes main reports and may merge episode/transfer supplemental reports; `benchmark` consumes reliability reports when available; audits/evidence review consume specialized reports.

## `benchmark`

### Purpose and information boundary

Measure frozen-export offline work and atomic online-update workloads. **Observed-only:** it never accepts or reads truth.

### Arguments

Required: `--run-dir` and `--experiment-dir`. Important options: `--config`, `--warmup`, `--iterations`, and `--overwrite`.

### Prerequisites consumed

Public observed events; prepared metadata; existing baseline and/or learned cutoff exports for offline measurements; learned checkpoint for online measurements.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/benchmarks/offline.json` | JSON | Latency, throughput, memory, bytes/hashes, workload, and missing-artifact status. | Always |
| `EXPERIMENT_DIR/benchmarks/online_workload.json` | JSON | Frozen seeded cold-start/single-event/batch workload identity. | Always |
| `EXPERIMENT_DIR/benchmarks/online.json` | JSON | Oracle-checked online latency, throughput, memory, and environment metadata. | Always |

### Existing output and overwrite

Existing reports fail unless `--overwrite`. Overwrite may refresh measurements for the identical named workload; a changed frozen workload belongs in a new experiment.

### Minimal example

```bash
uv run geoembed benchmark --run-dir runs/smoke --experiment-dir experiments/learned --warmup 1 --iterations 5
```

### Follow-up consumers

These are terminal R13 evidence artifacts consumed by documentation/evidence indexing, not by training or comparison.

## `rank`

### Purpose and information boundary

Train/run one observable dataset-2.0 recommendation control. **Observed-only:** ranking never reads utility, latent intent, chosen flags, or other truth.

### Arguments

Required: `--run-dir`, `--experiment-dir`, and `--model`. Important options: `--ranking-config` (exposure-aware only), `--k`, and `--overwrite`.

### Prerequisites consumed

Dataset-2.0 public catalog, requests, impressions, interactions, users/events, and manifest. Frozen/exposure-aware models require the learned dense export; learned models may require baseline ranking reports as controls.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/ranking/MODEL.npz` | compressed NPZ | Per-request candidate scores/ranks and authenticated identities. | Always |
| `EXPERIMENT_DIR/ranking/MODEL.json` | JSON | Observed ranking metrics, coverage, configuration, and provenance. | Always |
| `EXPERIMENT_DIR/ranking/frozen_embedding_checkpoint.npz` | compressed NPZ | Frozen-embedding ranker weights. | Conditional: `--model frozen_embedding` |
| `EXPERIMENT_DIR/ranking/exposure_aware_checkpoint.npz` | compressed NPZ | Exposure-aware ranker weights/configuration. | Conditional: `--model exposure_aware` |

### Existing output and overwrite

Any selected existing prediction/report/checkpoint fails unless `--overwrite`; replacement is scoped to the named model after identity checks.

### Minimal example

```bash
uv run geoembed rank --run-dir runs/smoke --experiment-dir experiments/learned --model popularity --k 1 5 10
```

### Follow-up consumers

`evaluate-ranking` consumes model NPZ/JSON pairs. Optional protected `evaluate-pair` ranking arguments consume matched prediction/report pairs.

## `evaluate-ranking`

### Purpose and information boundary

Evaluate frozen seen/unseen region/POI and early/late ranking slices. **Observed-only:** utility regret is unavailable because truth is never opened.

### Arguments

Required: `--run-dir` and `--experiment-dir`. Important options: `--models`, `--k`, and `--overwrite`.

### Prerequisites consumed

Dataset-2.0 observed request/catalog/interaction tables and each selected `EXPERIMENT_DIR/ranking/MODEL.{npz,json}` pair.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/ranking/transfer_slices.json` | JSON | Frozen slice identities, coverage, and ranking metrics by model/k. | Always |

### Existing output and overwrite

Existing output fails unless `--overwrite`; selected model names must be supported and their identities must match.

### Minimal example

```bash
uv run geoembed evaluate-ranking --run-dir runs/smoke --experiment-dir experiments/learned --models popularity nearest --k 1 5 10
```

### Follow-up consumers

The report is terminal observed-only R2/R8 evidence; no model stage consumes it.

## `visualize-ranking`

### Purpose and information boundary

Render one deterministically selected request across `popularity`, `nearest`,
`category_preference`, and `frozen_embedding`. **Observed-only:** the command
accepts the dataset root but resolves only its canonical `observed/` directory;
protected utility is unavailable rather than inferred.

The renderer authenticates each prediction/report pair, current observed source
hashes, model identity, and the common request and available-candidate hashes.
It displays request-time candidate fields, score/rank, availability and
impression state, and any recorded interaction. Its “what changed?” section is
a descriptive rank-order comparison, not a causal explanation, and no feature
attribution is fabricated.

```bash
uv run geoembed visualize-ranking --run-dir runs/smoke --experiment-dir experiments/learned
```

The versioned metadata and HTML are written beneath
`EXPERIMENT_DIR/ranking/visualization/`; either existing target requires
`--overwrite`.

## `robustness`

### Purpose and information boundary

Re-encode deterministic perturbation views and evaluate sensitivity for R6/R7. **Mixed protected evaluator:** view construction is observed-only, then the evaluator opens truth for authenticated reporting.

### Arguments

Required: `--run-dir` and `--experiment-dir`. Important options: `--kind`, `--views` (comma-separated), and `--config`.

### Prerequisites consumed

Public events, prepared metadata, original selected cutoff export, and learned checkpoint when applicable; protected truth is opened only after view construction succeeds.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `EXPERIMENT_DIR/robustness/KIND/VIEW_ID.npz` | compressed NPZ | Embeddings for configured removal/GPS/timestamp/service/truncation views. | Conditional: one per selected/configured view |
| `EXPERIMENT_DIR/robustness/KIND_robustness.json` | JSON | Authenticated sensitivity metrics, coverage, and view manifest. | Always |

### Existing output and overwrite

There is no `--overwrite`; existing view exports/report follow immutable robustness writer checks. Use a fresh experiment or remove only explicitly disposable derived outputs outside this command.

### Minimal example

```bash
uv run geoembed robustness --run-dir runs/smoke --experiment-dir experiments/learned --kind learned --views gps,timestamp
```

### Follow-up consumers

`compare` may merge matching robustness reports; evidence review consumes the individual view report.

## `compare`

### Purpose and information boundary

Compare baseline and learned exports with common frozen probes and optional factorized matrix. **Protected evaluator:** it authenticates and uses run truth; it never trains either representation.

### Arguments

Required: `--run-dir` plus either shared `--experiment-dir` or both separate experiment-root options. Important: `--output-dir`, `--config`, and repeated `--factorized-experiment NAME=PATH`.

### Prerequisites consumed

Prepared metadata and baseline/learned cutoff exports (and main evaluations as applicable) from matched roots; protected run truth. Compatible episode, robustness, and transfer reports are merged when present. Factorized mode consumes every named immutable experiment.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `OUTPUT_DIR/embedding_comparison.json` | JSON | Machine-readable fair baseline-versus-learned axes and supplemental merges. | Always without factorized matrix |
| `OUTPUT_DIR/embedding_comparison.md` | Markdown | Human-readable comparison without aggregate winner. | Always without factorized matrix |
| `OUTPUT_DIR/factorized_comparison.json` | JSON | Authenticated T2.7 named-control matrix and gates. | Conditional: `--factorized-experiment` supplied |
| `OUTPUT_DIR/factorized_comparison.md` | Markdown | Human-readable factorized matrix decision. | Conditional: `--factorized-experiment` supplied |

### Existing output and overwrite

No `--overwrite` exists; comparison outputs are immutable and an existing target fails. Default `OUTPUT_DIR` is the shared experiment’s `comparison/` (or the explicitly resolved comparison destination).

### Minimal example

```bash
uv run geoembed compare --run-dir runs/smoke --baseline-experiment-dir experiments/base --learned-experiment-dir experiments/learned --output-dir experiments/comparison/comparison
```

### Follow-up consumers

The reports are terminal decision/evidence artifacts; documentation evidence indexes may reference them.

## `pipeline`

### Purpose and information boundary

Run simulation, deep validation, preparation, one representation path, and its protected default evaluation. **Mixed boundary:** simulator creates observed/truth; preparation and representation are observed-only; validation/evaluation are protected. It is a fresh pipeline, not resume.

### Arguments

Required: `--run-dir` and `--experiment-dir`. Important options: simulation arguments, `--embedding-config`, `--mode baseline|learned`, and `--overwrite` for the new simulation target.

### Prerequisites consumed

Simulation and embedding YAML only. The command creates the run before consuming its observed and protected artifacts in later stages.

### Produces

| Artifact path | Format | Meaning | Written |
|---|---|---|---|
| `RUN_DIR/config.resolved.yaml, manifest.json, validation_report.json` | YAML + JSON | Resolved simulation, dataset identity, and structural report. | Always: both modes |
| `RUN_DIR/observed/*.csv.gz` | gzip CSV | All six dataset-2.0 public tables listed under `simulate`. | Always: both modes |
| `RUN_DIR/truth/*.csv.gz` | gzip CSV | All six core truth tables; change-points only for configured change interventions. | Always core; change points conditional |
| `RUN_DIR/deep_validation_report.json` | JSON | Protected deep validation. | Always: both modes |
| `EXPERIMENT_DIR/prepared/config.resolved.yaml` | YAML | Resolved embedding/preparation config. | Always: both modes |
| `EXPERIMENT_DIR/prepared/prepared_metadata.json` | JSON | Splits, source hashes, statistics, cutoffs, and field order. | Always: both modes |
| `EXPERIMENT_DIR/prepared/vocabularies.json` | JSON | Training-only explicit-order vocabularies. | Always: both modes |
| `EXPERIMENT_DIR/statistical_baseline.npz` | compressed NPZ | Baseline cutoff representation. | Baseline mode only |
| `EXPERIMENT_DIR/baseline_evaluation.json` | JSON | Protected default baseline evaluation. | Baseline mode only |
| `EXPERIMENT_DIR/model/best_model.pt` | PyTorch checkpoint | Best learned weights and schema. | Learned mode only |
| `EXPERIMENT_DIR/model/training_report.json` | JSON | Learned training metrics/provenance. | Learned mode only |
| `EXPERIMENT_DIR/model/training_participation.json` | JSON | Learned participation lineage. | Learned mode only |
| `EXPERIMENT_DIR/embeddings.npz` | compressed NPZ | Learned cutoff export. | Learned mode only |
| `EXPERIMENT_DIR/evaluation.json` | JSON | Protected default learned evaluation. | Learned mode only |

### Existing output and overwrite

The simulation target obeys validated `--overwrite`; downstream immutable preparation/training rules still apply. Therefore pipeline should target new run and experiment roots and must not be used as resume.

### Minimal example

```bash
uv run geoembed pipeline --run-dir runs/pipeline --experiment-dir experiments/pipeline --mode baseline --users 10 --days 2 --seed 1729
```

### Follow-up consumers

Baseline output can feed baseline visualization/robustness/benchmark; learned output can feed learned visualization, dense export, robustness, benchmark, and ranking. Run the other representation explicitly and then `compare`. **Pipeline does not produce dense exports, robustness reports, ranking artifacts, baseline-versus-learned comparisons, or paired-evaluation artifacts.**
