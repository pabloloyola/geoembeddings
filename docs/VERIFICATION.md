# Release verification

This document is an evidence ledger, not a claim that the current checkout has
been verified. Commands and results below are historical records for the source
revision named by each entry. Where an entry does not name a revision, its
source revision was not recorded and its results must not be attributed to the
current checkout.

The opening environment, package/test, learned-pipeline, and baseline/comparison
records were captured on 2026-08-11 UTC for source commit
`bbc44216c60d8790c2bb93fc5f0b052216f103df` (`init`). The `13 passed` result
below applies only to that revision; it is not the result of the current test
suite.

For pipeline and ranking smoke commands, follow
[`docs/LOCAL_EXPLORATION.md`](LOCAL_EXPLORATION.md).

## Bounded CLI integration smoke

Run the complete public-command contract against a disposable supported smoke
cohort with deterministic CPU training:

```bash
uv run python scripts/run_cli_integration_smoke.py
```

The script uses a temporary root by default (or a new path supplied with
`--work-dir`), 50 synthetic users over seven days, one training epoch, and one
measured benchmark iteration. It covers simulation, deep validation,
preparation, baseline and learned representations, sparse and dense export,
both evaluations, comparison, the dataset-2.0 nearest ranking control,
offline/online benchmarking, and the authenticated privacy audit. It verifies
canonical artifacts, explicit fields, preparation/source identities, finite
arrays, unchanged observed/truth bytes, default output immutability, and
rejection of a protected subdirectory as `--run-dir`. The intentionally small
cohort does not support scientific membership inference; the smoke requires the
audit to record membership as unavailable rather than treating that limitation
as privacy evidence. This CI-only contract check is not reference-scale
scientific evidence and must not be indexed or cited as such.

## Verify this checkout

```bash
uv sync --locked --extra dev
uv lock --check
uv run pytest
uv run geoembed --version
```

## Stable-release checklist

This is a **blocking release gate**. A release may be called stable only as a
**synthetic research harness** after every checkbox below has current-revision
evidence. “Stable” in this repository does **not** mean selected-candidate
success, external validity, privacy certification, or production readiness;
all four are explicitly outside this gate. A diagnostic result that is
scientifically unavailable can satisfy an execution check only when the report
honestly records its applicability, support, exclusions, and limitations. It
cannot satisfy or imply the excluded scientific claims.

- [ ] **Locked CPU suite is clean.** In a clean CPU-only environment, record
  the source commit, OS/architecture, Python and uv versions, lockfile hash, and
  the passing result of `uv sync --locked --extra dev`, `uv lock --check`, and
  `uv run pytest`. A repaired, partially cached, unlocked, GPU-backed, or
  failing environment does not pass.
- [ ] **Package and CLI smokes pass.** Record a successful `uv build`, the
  produced wheel and source-distribution paths/hashes, `uv run geoembed
  --version`, and `uv run geoembed --help`. The reported CLI/package version
  must agree with the release decision.
- [ ] **One complete small baseline/learned workflow passes.** Run the bounded
  disposable CPU workflow with `uv run python
  scripts/run_cli_integration_smoke.py --work-dir NEW_IMMUTABLE_PATH`, record
  its seed and exact run/experiment roots, and link its baseline and learned
  exports/evaluations. Do not reuse or overwrite a prior release's evidence.
- [ ] **One comparison report is valid.** Link the canonical comparison JSON
  from that same source hash, preparation identity, users, cutoffs, and
  candidate sets. Record the exact `uv run geoembed compare ...` command and
  confirm that validation passed; do not name an aggregate winner.
- [ ] **One online benchmark report is valid.** Link both immutable
  `benchmarks/online_workload.json` and `benchmarks/online.json`, produced by
  `uv run geoembed benchmark --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
  --warmup 10 --iterations 100`. Record CPU/software/hardware identity and
  confirm every full-recomputation oracle check passed.
- [ ] **One T4.3a privacy report is valid and candid.** Link the authenticated
  `audits/privacy.json` and `audits/privacy.md`, the frozen threat-model config,
  input/evidence identities, and the exact `uv run geoembed audit-privacy ...`
  command. Confirm that membership and each sensitive probe are labeled
  `applicable`, `unavailable`, or `not_applicable` as warranted, with support,
  exclusions, and uncertainty. The required
  `selection_dependent_privacy_conclusion` remains unavailable while there is
  no `selected_candidate`; neither successful execution nor AUC near 0.5 is a
  privacy guarantee or certification.
- [ ] **Status and evidence-link checks pass.** Record successful runs of `uv
  run python scripts/check_status_consistency.py` and `uv run python
  scripts/check_evidence_links.py docs/EXTERNAL_VALIDITY.md`. Every linked
  release artifact must exist, authenticate, and identify the revision being
  released.
- [ ] **No quantitative claim is unindexed.** Review the changelog, release
  notes, decision documents, and verification additions. Every quantitative
  claim must point to an evidence-registry entry with cohort, seed, source and
  preparation identities, cutoff/candidate scope where applicable, artifact
  hashes, and limitations. Otherwise remove the number or explicitly label it
  unverified history rather than release evidence.
- [ ] **No accidental generated source is tracked.** Review `git status
  --short`, `git diff --cached --stat`, and `git ls-files runs experiments
  notebook_artifacts`. Generated run, experiment, benchmark, audit, build, and
  notebook outputs must remain untracked unless the release intentionally
  indexes a small evidence/fixture file; each intentional exception needs a
  documented purpose, scope, and provenance.
- [ ] **Observed/truth leakage protections pass.** In addition to the full
  suite, record passing targeted boundary tests for observed-only preparation,
  training, baseline, export, ranking, and benchmarking; protected evaluator
  access; rejected truth-like observed fields; and authentication before
  protected-label access. A workflow that merely happens not to expose truth
  is insufficient without these negative tests.
- [ ] **Output immutability protections pass.** Record passing tests that
  existing run, experiment, pair, comparison, benchmark, privacy, and other
  report targets fail by default, that validated `--overwrite` behavior is
  limited to commands which explicitly support it, and that failed writes leave
  prior bytes unchanged.
- [ ] **Accelerator status is explicit.** State separately whether CUDA and
  Apple MPS were tested on this exact revision, with device/runtime identity
  and test commands when they were. “Not tested” is acceptable for the CPU
  gate; silence, CPU fallback described as an accelerator pass, or inference
  from historical MPS results is not. Preserve the padded-GRU/floating-mask MPS
  regression boundary.
- [ ] **Changelog and version decision are recorded.** Add a dated changelog
  entry that links the release evidence and limitations, and explicitly decide
  whether the package version changes. If it changes, update all canonical
  version declarations and verify wheel, sdist, and CLI agreement; if it does
  not, document why the accumulated changes do not require a bump.

The release sign-off must name the reviewer, UTC date, commit and tag, package
version decision, evidence-index path, workflow/comparison/benchmark/privacy
artifact paths, and unresolved limitations. Checking this list records software
and contract readiness at that immutable revision only; it does not reverse a
model-selection decision or expand the evidence scope.

## Continuous integration scope

`.github/workflows/ci.yml` runs the locked development environment and test
suite on the supported Python 3.11--3.14 matrix using CPU-only GitHub-hosted
runners. It also checks the CLI version, privacy command and imports,
documentation status, and external-validity evidence links. The uv download
cache is keyed from `uv.lock`; generated `runs/`, `experiments/`, and
`notebook_artifacts/` outputs are neither checked out nor cached.

CUDA and Apple MPS regressions remain separate, optional checks on appropriate
hardware. A passing CPU workflow does not establish CUDA or MPS compatibility,
and accelerator results must continue to report their actual device and avoid
counting device fallback as an accelerator pass.

## Integration verification: `c64611e` (2026-08-13 UTC)

This run verified source commit
`c64611e5e5bebf03ef306e37fdd0c284019857e9` on
`Linux-6.18.35-x86_64-with-glibc2.39` with `uv 0.7.22`. The lockfile identity
was SHA-256
`b95e1b4e89d7a1c3296f5c059888f0ee42371202b4a2ad1ff0e0b4a15da0415f`
(`uv.lock`); the accompanying `pyproject.toml` identity was
`37220d156e98b653d06d73e32f87666116f943a9f32b168bbccceadbeceb0634`.

Runtime identity before the failed sync was Python `3.14.4` (GCC `13.3.0`),
PyTorch `2.13.0+cu130`, NumPy `2.5.2`, pandas `2.3.3`, and CPU device. CUDA was
not available to PyTorch in this container.

The required `uv sync --locked --extra dev` did **not** complete: the package
index tunnel failed after three retries while fetching `packaging==26.3`.
An offline retry also failed because the locked `pygments==2.20.0` artifact was
not in uv's cache. Consequently, the following pytest result is an integration
diagnostic from a locally repaired environment using pytest `9.0.3`, **not** a
successful locked-environment verification. The complete suite reported `185
passed, 13 failed, 0 skipped, 1 warning in 49.27s` (53.526 seconds measured
wall-clock). The warning was an unregistered `integration` marker. The failed
tests were:

- `tests/test_artifact_index.py::test_index_rejects_mismatched_preparation_source_metadata`
- `tests/test_dense_export.py::test_dense_dataset_uses_only_observed_prepared_contract`
- both cases of `tests/test_episode_evaluation.py::test_rejects_invalid_embedding_values_or_dimensions`
- `tests/test_pair_evaluation.py::test_local_ridge_probe_fits_an_intercept_and_variance_weighted_r2`
- `tests/test_pair_evaluation.py::test_full_paired_run_evaluation`
- `tests/test_pair_manifest.py::test_pair_manifest_rejects_incompatible_contract`
- `tests/test_privacy.py::test_membership_population_is_deterministic_under_export_and_mapping_order`
- `tests/test_privacy.py::test_attack_is_unavailable_for_imbalanced_or_missing_split_classes`
- `tests/test_privacy.py::test_bootstrap_reports_degenerate_replicates_instead_of_hiding_them`
- `tests/test_privacy_protected_labels.py::test_continuous_bins_fit_train_only_and_emit_aggregate_metadata`
- `tests/test_ranking.py::test_rankers_share_sets_reject_v1_and_never_open_truth`
- `tests/test_simulate_pair.py::test_configured_intervention_changes_only_declared_fields[opportunity]`

The initial collection attempt after the interrupted sync additionally had 35
dependency import errors; those are an environment consequence and are not
counted as a separate software-suite result. The CLI version and privacy help
commands passed. Both documentation checks passed, reporting 35 completed
tasks across two status documents and a valid external-validity evidence file.

This record is software-integration verification only. It is not evidence that
T2.7 or any scientific gate passed. The T2.7 **do not advance** decision remains
binding, and no evaluator or model feature was added.

## Failure reconciliation and repair (2026-08-13 UTC)

The full suite was reproduced before repair in the available core environment
with `PYTHONPATH=.venv/lib/python3.14/site-packages:src pytest`. It reported
`205 passed, 13 failed, 1 warning in 31.55s`. The locked development sync could
not be completed because the package-index tunnel could not download
`pygments==2.20.0`; `uv sync --locked --offline` restored the cached locked core
dependencies, and the pre-existing host pytest runner was used only as an
environment workaround. This is a dependency/platform limitation, not one of
the executed software failures below.

Compared with the 13 failures recorded for `c64611e`, 12 test identities
recurred. The earlier local-ridge intercept test was already resolved and passed
in the reproduced suite. A privacy orchestration integration fixture was the
one newly exposed failure. Every reproduced failure is classified and resolved
as follows:

| Reproduced failure | Classification | Resolution |
|---|---|---|
| artifact-index preparation/source mismatch rejection | **Code defect** | Source authentication now verifies exactly the observed inputs declared by preparation, rejects unknown declared observed filenames, and no longer requires dataset/2.0 recommendation files from a legacy event-only preparation. |
| dense observed-only preparation fixture | **Stale fixture** | Added the required public dataset-contract manifest; no truth input was introduced. |
| episode invalid non-finite export case | **Code defect** | Component validation now reports non-finite values distinctly from shape and row-alignment failures. |
| episode invalid dimensionality case | **Code defect** | Component validation now reports the required 2-D shape failure distinctly. |
| full paired-run evaluation | **Code defect** | Added authenticated explicit wall-clock preparation cutoffs for matched interventions. A representation whose independently fitted observed-only feature spaces have unequal dimensions is now reported as unavailable with dimensions and coverage instead of aborting the entire report; Markdown preserves that state. |
| pair-manifest incompatible-contract rejection | **Stale fixture** | The fixture now changes dataset/2.0 to the genuinely incompatible legacy 1.0 value instead of assigning the already-current version. Authentication was not weakened. |
| privacy population ordering | **Stale fixture** | Synthetic vector values are now keyed by user/cutoff rather than NPZ row position, and assertions use the documented participation-record field names. |
| privacy imbalanced/missing split classes | **Stale fixture** | Updated the fixture to the documented `vector_features`, `provenance_covariates`, and `matching_stratum` record contract; fail-closed support behavior is unchanged. |
| privacy bootstrap degeneracy accounting | **Intentional contract change** | The expectation now reflects the frozen protocol in which class is part of every effective stratum, so singleton classes remain present in every stratified replicate. |
| privacy protected continuous bins | **Stale fixture** | Protected synthetic values now cover every frozen split's train-fitted bins; minimum class-by-split support remains enforced. |
| privacy audit orchestration | **Stale fixture** | The fixture now supplies valid observed schemas, role-eligible export populations, authentic participation coverage, and explicitly expects a scientifically unavailable sensitive attack when the frozen cell-support gate fails. |
| ranking observed/truth-boundary integration | **Stale fixture** | Dense NPZ identity arrays now use non-object strings, and the test expects the public CLI's authenticated `SystemExit(2)` failure contract for a tampered legacy manifest. Truth remains unopened. |
| opportunity intervention pair integrity | **Intentional contract change** | Opportunity interventions now declare all newly public observed recommendation tables as permitted-to-change outputs; protected truth permissions and pair-integrity checks remain intact. |

Focused module tests were rerun after each repair group. The final focused run
for all affected modules passed, followed by a complete run of `218 passed in
22.00s` with no failures, collection errors, skips, or warnings. The previously
unregistered integration marker is now registered. There are **no remaining
software-suite failures** in this environment. The locked-extra sync remains an
environment limitation until the exact dev artifact can be fetched; it does not
alter the passing executed-suite result or constitute scientific evidence.

## Locked CPU verification attempt: `0bc2781` (2026-08-13 UTC)

This is a fresh verification attempt of exact source commit
`0bc27811d2bd9c7b8aa9d7dc2ea277f947602b82`; it does not reuse the earlier
repaired-environment result. The checkout was clean before verification. The
environment was Ubuntu 24.04.4 LTS on Linux `6.18.35`, `x86_64`, with three
online Intel Xeon Platinum 8370C CPU cores. No `/dev/nvidia0` or `/dev/dri`
device was present. CUDA and Apple MPS were not tested and this entry records
no accelerator pass.

Runtime metadata was captured at `2026-08-13T05:27:08+00:00`: Python `3.14.4`,
uv `0.7.22`, package/CLI version `0.5.0`, and `uv.lock` SHA-256
`b95e1b4e89d7a1c3296f5c059888f0ee42371202b4a2ad1ff0e0b4a15da0415f`.
Durations below are wall-clock seconds measured around each process; command
output and exit status, rather than duration, determine the result.

| Command | Exit | Duration | Result |
|---|---:|---:|---|
| `uv sync --locked --extra dev` | nonzero | 0.578 s | **Failed.** uv resolved 145 packages, then the package-index tunnel failed after three retries while downloading locked `iniconfig==2.3.0`. |
| `uv lock --check` | 0 | 2.291 s | Passed; uv resolved 145 packages in 4 ms. |
| `uv run pytest` | 2 | 3.949 s | **Failed during collection:** 40 collection errors in pytest's reported 1.17 s; no tests ran, so totals are **0 passed, 0 failed tests, 40 collection errors**. |
| `uv run geoembed --version` | 0 | 1.059 s | Passed: `geoembed 0.5.0`. |
| `uv run geoembed --help` | 0 | 0.965 s | Passed; the public command list included `audit-privacy` and `calibrate-reliability`. |
| `uv run geoembed audit-privacy --help` | 0 | 0.917 s | Passed; canonical run, named experiment, evidence, utility-report, config, output, and overwrite arguments were shown. |
| `uv run geoembed calibrate-reliability --help` | 0 | 0.940 s | Passed; canonical run, named experiment, config, output, and overwrite arguments were shown. |
| `uv run geoembed inspect-evidence` | 0 | 3.491 s | Passed with `ci_status=ok`: 8 indexes, 182 artifacts, 2 locally present/content-verified, 139 locally absent, 41 historically lost, and 0 content mismatches. |
| `uv run python scripts/check_status_consistency.py` | 0 | 0.871 s | Passed: 36 completed tasks across 2 documents. |
| `uv run python scripts/check_evidence_links.py docs/EXTERNAL_VALIDITY.md` | 0 | 0.910 s | Passed. |

The sync failure left the clean environment without the locked project and
scientific dependencies. Pytest consequently reported missing `numpy`,
`pandas`, `torch`, `yaml`, and `geoembeddings` imports across 40 test modules.
These are environment/dependency collection failures, not 40 executed test
failures, and there was no pytest warning summary. The evidence inspector's
locally absent and historically lost artifacts are expected availability
states, not command warnings or scientific failures. No other command emitted
a warning or failure.

Because the mandatory locked sync and test suite did not pass, this revision
does **not** satisfy the locked CPU release gate. The independently runnable
lock, CLI, evidence-inspection, status, and link checks do not repair or replace
that failed gate. This software verification outcome provides no new model
evidence: the T2.7 **do not advance** decision remains binding, and a routine
branch remains unauthorized.

## Historical evidence: initial release smoke

Verification date: 2026-08-11 UTC.
Source commit: `bbc44216c60d8790c2bb93fc5f0b052216f103df` (`init`).

## Environment

- Python 3.12.13
- `uv` 0.11.33
- PyTorch 2.13.0
- Device used for neural smoke run: CPU

## Package and tests

```bash
uv sync --locked --extra dev
uv run pytest
uv run geoembed --version
uv lock --check
```

Result:

```text
13 passed (historical result for bbc44216c60d8790c2bb93fc5f0b052216f103df)
geoembed 0.5.0
lock check passed
```

## End-to-end learned smoke

Command:

```bash
uv run geoembed pipeline \
  --run-dir runs/agent_handoff_smoke50 \
  --experiment-dir experiments/agent_handoff_smoke50 \
  --mode learned \
  --users 50 \
  --days 7 \
  --seed 20260811
```

Result:

- deep simulator validation: passed;
- observed users/events: 50 / 1,023;
- training/validation windows: 613 / 162;
- eight epochs completed;
- best validation loss: 5.6836;
- learned export: 45 users, 135 user/cutoff rows, 128 dimensions;
- test evaluation completed.

## Baseline and comparison smoke

The same run and preparation were used for `baseline`, baseline `evaluate`, and
`compare`.

Result:

- baseline export: 45 users, 135 rows, 685 dimensions;
- shared three-cutoff comparison users: 45;
- JSON and Markdown comparison reports produced successfully.

The sample is intentionally too small for scientific conclusions. For example,
the held-out probe set contains only nine users and fine-geohash future probes
have zero known-label coverage. This run verifies execution and contracts, not
model quality.

## Small-sample validation caution

A 20-user, 4-day run failed deep validation because it did not cover every
episode type and had insufficient cross-region overlap. This is expected from
stochastic small cohorts. Use at least the documented 50-user, 7-day smoke size
for the full `pipeline`; use unit tests for smaller plumbing cases.


## T1.2 episode evaluator

```bash
uv run pytest tests/test_episode_evaluation.py tests/test_dense_export.py tests/test_cli_paths.py
uv run geoembed export-dense --kind baseline --event-stride 1 --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed export-dense --kind learned --event-stride 1 --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --episodes --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --episodes --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Inspect coverage and matched deltas. Tests cover exact boundaries, malformed/overlapping intervals, sparse exports, missing users, duplicate/non-monotonic timestamps, non-finite values, dimensions, and the direct `truth/` boundary. The observed-only dense exporter test runs without any `truth/` directory.

## T1.3 event-removal robustness

```bash
uv run geoembed robustness --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed robustness --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Defaults are versioned in `configs/embedding/single_vector.yaml` (seed
`20260811`; rates `0`, `0.1`, `0.25`, `0.5`). Inspect the two
`robustness/*_event_removal.json` reports for realized thinning, unencodable
keys, matched coverage, cosine drift, and persistent-probe degradation. Full
removal is intentionally reported as unencodable rather than imputed. Event
removal provides partial R7 evidence; it is one of the implemented deterministic
sensitivity views and does not establish real-noise robustness.

Same-run smoke evidence used `smoke/run`, `smoke/experiment`, seed `20260811`,
and 1,176 observed events. At requested rates 0/0.1/0.25/0.5, realized removals
were 0/134/320/582 and matched-row coverage was 0.9787/0.9574/0.9574/0.9504
for both representations. Learned-minus-baseline mean cosine drift was
0.0000/0.0106/0.0312/0.0517; the corresponding frozen-probe-degradation deltas
were approximately 0.0000/-0.0019/0.2637/0.2520. These smoke estimates are not
scientific evidence: only nine held-out probe users were available. Existing
episode comparison still ran; learned-minus-baseline within-episode cosine,
boundary change, and post-episode recovery were -0.0627, +0.0505, and -0.0907.
Persistent/preference probe mean-R2 deltas were -0.4870/-0.1655, so this trained
smoke model does not outperform the baseline on representation-quality axes.

## T1.4 deterministic robustness views

```bash
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed robustness --views gps,timestamp,leave-one-service-out,recent-truncation --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Inspect the matched view specifications and masks, realized perturbations,
coverage, cosine drift, and frozen-probe degradation. These executable GPS,
timestamp, service-removal, and truncation operators are deterministic
sensitivity tests; they are not calibration to real noise or evidence of causal
invariance.

## T1.5 spatial-transfer evaluation

```bash
uv run geoembed evaluate --transfer --kind baseline --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed evaluate --transfer --kind learned --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
uv run geoembed compare --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Inspect train-scaled distance retrieval, boundary-pair cosine, held-out-region
coverage, and seen/unseen geohash slices separately. Empty or unknown-label
slices remain explicit coverage results. These tests do not measure unseen-POI
transfer and do not establish external geographic validity.

## T1.6 temporal/routine evaluator

Run baseline and learned dense exports, both `evaluate --temporal-routine`
commands, and `compare`. Verify identical source hashes, dense users/keys,
temporal definitions, split seed, and row-level coverage. Interpret R3 and R4
axes independently alongside different-user cosine and effective-rank ratio.
The schedule-shift response must come from a passing controlled matched pair;
the weekday/weekend probe alone is not a counterfactual substitute.

## T1.11b stable-identity smoke (2026-08-11)

Affected requirements are R5 and R7; the affected layers are simulator,
run-level manifest contract, and simulator validation. The baseline is the
T1.11a fixed-seed configuration at seed `20260811`; no representation metric or
baseline/learned delta applies because this task establishes matching keys
rather than evaluating embeddings. Reproduce into a new immutable directory:

```bash
uv run geoembed simulate --config configs/simulation/kanto_v1.yaml --run-dir runs/t1.11b-smoke-20260811 --users 50 --days 7 --seed 20260811
uv run geoembed validate --run-dir runs/t1.11b-smoke-20260811
uv run pytest tests/test_simulator_random_streams.py tests/test_layout.py
```

Expected artifacts are `runs/t1.11b-smoke-20260811/manifest.json` and
`runs/t1.11b-smoke-20260811/deep_validation_report.json`. The fixed-seed local
smoke passed validation with these manifest identity SHA-256 values: users
`b323c39534aabc18dd0ddd3ffc0cbe42f370056add1a5e45f11c822603fbbff6`
(50), regions
`6641498b7ff41370905b4c4f3948f0ca4460760ded9ae9b0721a3f498910f21a`
(13), POIs
`599d52497e9967d67e109f16b00bffc524bc9b630f860d7fcff998fd86719901`
(872), episodes
`1fe7adf87a503dba92f65dbda017de67367e167e96c5cdbfc48b8d27a267640c`
(350), choices
`d4c63c48bfc6f592e45b7c826fc2feba0a7419da1585fe5a1fe15ffaadf8cecb`
(350), and trajectories
`9e98c593327c5a451f1b9951a7eb3bf2e6a96433239f948d2b1ea96281b3e859`
(1,800). Observed gzip hashes were events
`f4d03de5aa1a1d5a9ba12fb851108b482cca270bc02e88357935d1a11ac06d94`
and users
`38fecea5177dce38afe3e39f32926fe27c25ebd2b836452208cbb774a140c9c4`.
The resolved stream seeds are world `15122758705215849765`, user latents
`10408480750793963739`, episodes `2984504666824196055`, choices
`61965804204990368`, and observation `11209042838108107995`.

Unit/integration coverage serializes the manifest through JSON, rejects an
unsupported identity schema and inconsistent hash, rejects duplicate identity
inputs, proves identity-set hashes ignore row order, and changes only the
observation stream while asserting all six entity declarations remain equal.
The observed filenames remain exactly the two dataset/1.0 files, and existing
source-boundary tests continue to ensure `prepare`, `baseline`, `train`, and
`export` receive `observed/` only. Exposure/opportunity allowed-change rules and
pair integrity remain limitations delegated to T1.11c--T1.11e; these hashes do
not themselves prove counterfactual validity or causal invariance.

## T1.11c pair-manifest smoke (2026-08-11)

Affected requirements are R5 and R7; the affected layers are the protected
simulator/evaluator contract and CLI. The baseline consists of two T1.11b-style
fixed-root-seed runs that differ only in the observation stream. Reproduce with
two new immutable run directories and then declare the pair:

```bash
uv run geoembed simulate --run-dir runs/t1.11c-reference --users 10 --days 2 --seed 20260811
# Set run.random_streams.observation in a copied YAML before the second command.
uv run geoembed simulate --config /tmp/t1.11c-observation.yaml --run-dir runs/t1.11c-intervention --users 10 --days 2 --seed 20260811
uv run geoembed pair-manifest --reference-run-dir runs/t1.11c-reference --intervention-run-dir runs/t1.11c-intervention --output pairs/t1.11c-observation/pair_manifest.json
uv run pytest tests/test_pair_manifest.py tests/test_cli_paths.py tests/test_layout.py
```

The integration test generates the two fixed-seed runs directly, changes only
the observation stream, requires all six stable identity hashes to match, and
checks immutability plus validated overwrite. Schema tests cover JSON round
trip, missing hashes, unsupported versions, incompatible dataset contracts,
overlapping declarations, and ambiguous matching keys. This task establishes a
declaration only: field-level pair integrity and embedding counterfactual deltas
remain T1.11d and T1.11f, so no causal-invariance claim or R5 metric is made.

## T1.11d pair-integrity smoke (2026-08-11)

Affected requirements are R5 and R7; affected layers are the protected
simulator and evaluator contract. The baseline is the T1.11c declared
observation-only pair. Validate it and run mismatch/gate coverage with:

```bash
uv run geoembed validate-pair --pair-manifest pairs/t1.11c-observation/pair_manifest.json
uv run pytest tests/test_pair_integrity.py tests/test_pair_manifest.py tests/test_cli_paths.py tests/test_layout.py
```

The command produces `pairs/t1.11c-observation/pair_integrity.json` only after
authenticating canonical run paths and current input lineage. Unit coverage
exercises schema, missing-key, duplicate-key, disallowed-field, allowed-field,
missing-report, failing-report, stale-report, and stale-source failures. The
end-to-end observation pair checks persistent user truth, episodes, candidates,
choices, trajectories, observation process, and both observed tables while
requiring every realized difference to match its declaration. This artifact is
a prerequisite rather than an R5/R7 representation result; T1.11e and T1.11f
remain open, and no external causal-validity claim is made.

## T1.11e configured-intervention smoke (2026-08-11)

Affected requirements are R5 and R7; the changed layer is the simulator and
its protected pair-validation surface. The baseline is the passing T1.11d
exposure-only integrity workflow. The fixed seed is `20260811` and the smoke
scale is 10 users over two days:

```bash
for kind in exposure opportunity observation; do
  uv run geoembed simulate-pair --intervention "$kind" \
    --reference-run-dir "/tmp/t1.11e-${kind}-reference" \
    --intervention-run-dir "/tmp/t1.11e-${kind}-intervention" \
    --pair-dir "/tmp/t1.11e-${kind}-pair" \
    --users 10 --days 2 --seed 20260811
done
uv run pytest tests/test_simulate_pair.py tests/test_pair_integrity.py tests/test_pair_manifest.py tests/test_cli_paths.py tests/test_layout.py
```

Each immutable pair produces `pair_manifest.json`, passing
`pair_integrity.json`, and passing `behavioral_diagnostics.json`; each run also
contains its normal validation report and deep structural/behavioral report.
At smoke scale the deep report can retain coverage warnings or failures (for
example sparse travel types), while the command requires every structural
integrity check plus the intervention-specific fixed-seed direction to pass.
This evidence tests the configured synthetic mechanisms and does not calibrate
or validate constants against Tokyo or Kanto.

## T1.7 reliability and offline-efficiency smoke (2026-08-11)

This evidence uses a new immutable replacement identity,
`runs/t0.3-cpu-smoke-20260811` with
`experiments/t0.3-cpu-smoke-20260811`. It is **not** a continuation or recovery
of the lost T0.2 reference. Both exports have 47 users and 141 matched
user/cutoff rows. They share preparation SHA-256
`02c3452c8018ab6d45b591c400743c408f1ed48d7211b13aadded166e168dc10`,
events SHA-256 `cc5c6a8352460ae4907d98b813b28d5a180b31d296895e10524214b0b7886eda`,
and users SHA-256 `d4381e7c160b519bd9422662d89277b2ed9e807d707ea0cc151b3cfa0e36c82d`.
The learned CPU smoke used one epoch, seed `20260806`; reliability used seed
`20260811`, 200 resamples, five bins, and coverage 0.25/0.50/0.75/1.00.

```bash
uv run geoembed evaluate --reliability --kind baseline --run-dir runs/t0.3-cpu-smoke-20260811 --experiment-dir experiments/t0.3-cpu-smoke-20260811 --config /tmp/t0.3-cpu-smoke-20260811.yaml
uv run geoembed evaluate --reliability --kind learned --run-dir runs/t0.3-cpu-smoke-20260811 --experiment-dir experiments/t0.3-cpu-smoke-20260811 --config /tmp/t0.3-cpu-smoke-20260811.yaml
uv run geoembed benchmark --run-dir runs/t0.3-cpu-smoke-20260811 --experiment-dir experiments/t0.3-cpu-smoke-20260811 --config /tmp/t0.3-cpu-smoke-20260811.yaml --warmup 1 --iterations 5
```

Artifacts are `baseline_reliability.json`, `reliability.json`, and
`benchmarks/offline.json` under that experiment. Runtime was Linux x86-64,
Python 3.14.4, PyTorch 2.13.0+cu130, CPU. Reliability command durations were
0.0771 s baseline and 0.0190 s learned; benchmark duration was 0.3447 s.

Axes are deliberately separate:

- **Reliability/coverage:** all 47 users were evaluated and all five bins met
  minimum count. Baseline lowest/full coverage risk was 0.00406/0.00742;
  learned was 0.00465/0.02766. Mean uncertainty/error increased across the five
  ordered bins for both. These smoke values do not establish calibrated
  uncertainty and cutoff bootstrap is not event/window resampling.
- **Frozen-export read/validation:** baseline artifact was 22,016 bytes with
  p50 0.001377 s and 103,785 rows/s; learned was 65,641 bytes with p50
  0.000924 s and 150,876 rows/s.
- **Offline export serialization:** baseline p50 was 0.006155 s and 22,886
  rows/s; learned p50 was 0.013148 s and 10,677 rows/s. This serializes the
  existing frozen arrays to an in-memory NPZ without replacing artifacts.
- **Reliability evaluation:** baseline p50 was 0.018106 s and 2,440 users/s;
  learned p50 was 0.012971 s and 3,607 users/s. Python peak allocation was
  654,524/112,308 bytes and process peak RSS was 543,633,408 bytes for
  baseline/learned respectively. Shared process peak RSS is not an isolated
  per-representation allocation.

No aggregate winner is derived: the learned artifact is larger and its
coverage-risk behavior differs, while timing on this warm filesystem happened
to be lower. These hardware-specific smoke measurements do not measure training
or online incremental updates and are not calibrated real-world uncertainty.

## T4.3a diagnostic-control privacy audit verification (2026-08-13)

Requirement R12 is affected in the evaluator and documentation layers; the
simulator, observed contract, model, and generated artifacts are unchanged. No
authenticated same-lineage member/non-member reference artifact is locally
available, so the executable audit must report membership as `unavailable`
rather than inventing labels. Verify the CLI and local contracts with:

```bash
uv run pytest tests/test_privacy.py tests/test_privacy_input_authentication.py \
  tests/test_privacy_protected_labels.py tests/test_privacy_rendering.py \
  tests/test_privacy_cli.py
uv run geoembed audit-privacy --help
```

For an authenticated lineage, run the canonical command documented in
`docs/COMMAND_REFERENCE.md`; it writes immutable
`AUDIT_OUTPUT_DIR/audits/privacy.{json,md}` under
`geoembeddings-privacy-audit/1.0`. Tests cover strict frozen configuration,
deterministic user splits and matching, attack train/validation/test isolation,
support and unavailable states, seeded bootstrap intervals, input and
protected-label authentication, prohibited conclusions, atomic publication,
and CLI path behavior.

This verifies the T4.3a diagnostic surface, not selection-dependent T4.3. No
representation has the `selected_candidate` role, no diagnostic control is
selected, and the T2.7 **do not advance** decision remains binding. Simulator
attacks do not certify privacy; AUC near 0.5 does not prove safety, and may
instead reflect attack weakness, inadequate support, or threat-model mismatch.

## T4.4 online incremental-update verification

Run `uv run geoembed benchmark --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
--warmup 10 --iterations 100`. In addition to the retained offline report, this
creates immutable `benchmarks/online_workload.json` and canonical
`benchmarks/online.json`. The CPU acceptance path covers both baseline and
learned diagnostic controls, atomic rollback, workload determinism, deterministic
quantiles, immutable vectors, and mandatory full-recomputation checks. CUDA/MPS
are optional and must reuse identical workload metadata; device fallback is not
an accelerator result. Training and hardware-normalized winner claims remain
excluded.

## T1.11f matched-evaluator verification (2026-08-11)

Requirement IDs: R5 and R7. This is an evaluator-only change; the observed
contract and model APIs are unchanged. No complete paired baseline/learned
reference artifact existed before this change, so the baseline is the passing
T1.11e pair-integrity artifact rather than a representation result.

```bash
uv run pytest tests/test_pair_evaluation.py tests/test_pair_integrity.py tests/test_simulate_pair.py
uv run geoembed evaluate-pair --help
```

The evaluator authenticates the manifest/integrity chain and all four frozen
exports before protected labels are opened. Unit coverage exercises matching,
exclusions, drift, retrieval, effective rank, frozen probes, and the modeling
information boundary. A reference-scale paired result is not claimed or
archived; simulator-only validity and uncalibrated observation mechanisms remain
explicit limitations.

## T1.15 change-support verification (2026-08-11)

Requirements R1 and R11 are affected across simulator and protected evaluator;
the public observed contract and all model APIs are unchanged. No complete
reference-scale T0.2 change artifact existed, so acceptance uses matched
fixed-seed statistical/learned dense exports rather than claiming a scientific
model improvement. Unit coverage checks half-open duration and invalid censored
change points. Integration coverage creates both interventions, checks stable
identity hashes and permitted fields, proves change truth is absent from
`observed/`, exercises temporary recovery and sustained right-censoring,
rejects stale pair-integrity inputs, and protects immutable output roots.

```bash
uv run geoembed simulate-pair --intervention temporary-trip --users 10 --days 9 --seed 20260811 \
  --reference-run-dir /tmp/t1.15-control --intervention-run-dir /tmp/t1.15-trip --pair-dir /tmp/t1.15-pair
uv run geoembed validate-pair --pair-manifest /tmp/t1.15-pair/pair_manifest.json
```

Reported curves are matched-control representation drift, not causal evidence
outside the simulator and not proof of factorized disentanglement. Sparse event
histories cause explicit missing-bin exclusions; sustained runs are explicitly
right-censored and therefore cannot report recovery.

## T2.4--T2.7 factorization gate (2026-08-12)

Requirements R1, R4, R5, R6, and R7 affect the observed-only model/export path
and protected evaluators; the observed contract is unchanged. The immutable
replacement identity is `t2.7-factorization-20260812-s20260812-u50-d14`
(50 users, 14 days, simulation seed 20260812). Preparation ran once and its
bytes were copied unchanged into the six immutable experiment roots.

```bash
uv run geoembed compare --run-dir runs/t2.7-factorization-20260812-s20260812-u50-d14 \
  --factorized-experiment factorized_pc=experiments/t2.7-factorization-20260812/factorized_pc \
  --factorized-experiment capacity_matched_single=experiments/t2.7-factorization-20260812/capacity_matched_single \
  --factorized-experiment persistent_only=experiments/t2.7-factorization-20260812/persistent_only \
  --factorized-experiment context_only=experiments/t2.7-factorization-20260812/context_only \
  --factorized-experiment factorized_no_persistent_loss=experiments/t2.7-factorization-20260812/factorized_no_persistent_loss \
  --factorized-experiment factorized_no_context_loss=experiments/t2.7-factorization-20260812/factorized_no_context_loss
uv run python scripts/index_artifacts.py --factorized-comparison \
  experiments/t2.7-factorization-20260812/factorized_pc/comparison/factorized_comparison.json \
  --output docs/artifacts/t2.7-factorization-20260812.json --task-id T2.4-T2.7
```

The comparison rejects source, preparation-definition, cutoff, export-key,
user-mask, and supplemental-definition mismatches. Coverage is 49 export users
and seven held-out probe users. Persistent and combined gates fail; paired
causal claims are deliberately not made after the mandatory rejection gate.

## T3.4 observable naive-ranker verification

Requirement R9 is affected across the observed-only ranking and evaluator
surface; the dataset contract and simulator are unchanged. The implementation
was introduced at source commit
`8215a86ab9cef7b52a2e50a086b65015ffc6aa24` (`Implement observable naive
ranking baselines`). The commands below are a reproducible verification recipe,
not executed evidence in this ledger.

First run the focused ranking tests:

```bash
uv run pytest tests/test_ranking.py
```

Then run all three controls against the same existing immutable dataset and the
same experiment root. Replace `RUN_DIR` with one complete, immutable dataset-2.0
root and `EXPERIMENT_DIR` with its ranking experiment root. Do not use
`--overwrite` when producing evidence for indexing.

```bash
uv run geoembed rank \
  --run-dir RUN_DIR \
  --experiment-dir EXPERIMENT_DIR \
  --model popularity

uv run geoembed rank \
  --run-dir RUN_DIR \
  --experiment-dir EXPERIMENT_DIR \
  --model nearest

uv run geoembed rank \
  --run-dir RUN_DIR \
  --experiment-dir EXPERIMENT_DIR \
  --model category_preference
```

Expected artifacts (not evidence that the commands ran):

```text
EXPERIMENT_DIR/ranking/popularity.npz
EXPERIMENT_DIR/ranking/popularity.json
EXPERIMENT_DIR/ranking/nearest.npz
EXPERIMENT_DIR/ranking/nearest.json
EXPERIMENT_DIR/ranking/category_preference.npz
EXPERIMENT_DIR/ranking/category_preference.json
```

The three reports must carry identical request and available-candidate hashes
before their metrics are compared. Index the immutable dataset identity, source
revision, exact commands, artifact hashes, runtime environment, and results
together; expected paths or metrics must never be presented as executed
evidence.

## T3.5 frozen-embedding ranker verification

T3.5 affects R9 and the observed-only ranking model/evaluator boundary; it does
not change the dataset contract or frozen encoder. With seed `20260812`, first
produce the three T3.4 controls above, then run on the same immutable roots:

```bash
uv run geoembed rank --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR \
  --model frozen_embedding
```

Expected immutable artifacts are
`EXPERIMENT_DIR/ranking/frozen_embedding_checkpoint.npz`,
`frozen_embedding.npz`, and `frozen_embedding.json`. Run
`uv run pytest tests/test_ranking.py` before the complete `uv run pytest`.
Acceptance requires identical request/candidate hashes across all four reports,
finite causal embeddings, reported request/user coverage, and separate deltas
against every control. A non-positive delta is diagnostic evidence and must not
trigger end-to-end encoder tuning. The immutable result below is the first
archived T3.5 lineage; it must not be generalized beyond its stated scope.

### Runtime results (immutable T3.5 lineage, 2026-08-12)

Requirement **R9** was exercised after source revision
`ed3a15e547d6beaba61f7f1e4073dad7f01f1cf9` corrected the observable user-source
authentication and distributed public recommendation requests across the run so
the prepared temporal cutoffs contain disjoint training, validation, and test
requests. This simulator/observed-surface and ranking-evaluator change does not
change the dataset schema. The selected frozen representation is the
`factorized_pc` combined component as a diagnostic export; the prior T2.7
rejection remains binding and this result does **not** authorize end-to-end
encoder tuning.

The new dataset is
`runs/t3.5-evidence-20260812-s20260812` (50 users, 14 days, simulation seed
`20260812`) and the modeling lineage is
`experiments/t3.5-evidence-20260812-factorized-pc` (training seed `20260806`,
frozen-head seed `20260812`). Neither path reused, relabeled, or overwrote an
older artifact. The exact resolved simulator and embedding configurations are
`runs/t3.5-evidence-20260812-s20260812/config.resolved.yaml` and
`experiments/t3.5-evidence-20260812-factorized-pc/prepared/config.resolved.yaml`.
The checkpoint is `model/best_model.pt`, the selected request-time export is
`dense_embeddings.npz`, and the frozen head checkpoint is
`ranking/frozen_embedding_checkpoint.npz` under that experiment root. All exact
commands, seeds, source hashes, artifact byte hashes, predictions, reports, and
coverage are indexed immutably in
`docs/artifacts/t3.5-ranking-20260812.json`.

All four reports authenticate request hash
`ee894bf15de303e0705c4ee6e7cdb4ba133a43a60caaf7c14f79e53d1546fcb7` and
available-candidate hash
`43fc4f02664bfd3ee544c960b079eb8290fe875199f4dfb3ae6b4ef4a9d94a9e`.
Training covered 36/36 requests and users with 891 available candidates;
validation covered 8/8 and 197; test covered 5/6 and 118. The sole exclusion,
`request_043696ab43366cba6030189f`, had no embedding at or before its request
cutoff, so no later vector was substituted.

| Ranker | Recall@1 / @5 / @10 | NDCG@1 / @5 / @10 | MRR |
|---|---:|---:|---:|
| popularity | 0.880 / 0.880 / 0.880 | 0.880 / 0.880 / 0.880 | 0.885895 |
| nearest | 0.000 / 0.140 / 0.360 | 0.000 / 0.070 / 0.134073 | 0.122200 |
| category preference | 0.000 / 0.340 / 0.420 | 0.000 / 0.152486 / 0.180168 | 0.131912 |
| frozen embedding | 0.940 / 0.980 / 0.980 | 0.940 / 0.962619 / 0.962619 | 0.956667 |

Frozen-minus-control deltas remain separate: against popularity, Recall deltas
are `+0.060/+0.100/+0.100`, NDCG deltas are
`+0.060/+0.082619/+0.082619`, and MRR is `+0.070771`; against nearest they are
`+0.940/+0.840/+0.620`, `+0.940/+0.892619/+0.828546`, and `+0.834467`;
against category preference they are `+0.940/+0.640/+0.560`,
`+0.940/+0.810133/+0.782451`, and `+0.824754`. These are observable click
metrics on one small synthetic fixed-seed lineage, not utility, causal, or
generalization evidence. The gate is satisfied for T3.5 because every requested
axis beats every same-contract control with explicit coverage and exclusions;
no encoder tuning was performed or is implied.

## T3.6 exposure-aware ranking verification

T3.6 affects R5/R9 across the observed ranking trainer and protected paired
evaluator. Its prerequisite baseline is the immutable T3.5 lineage indexed at
`docs/artifacts/t3.5-ranking-20260812.json`; the T3.7 transfer evaluator is also
executable. Training receives only observed impressions, interactions, requests,
catalog, events, and frozen embeddings. The only fitted propensity is the public
logging-policy quantity `P(is_shown | candidate_position)` on training requests.
Protected utility and choice probability remain in truth and are opened only
after a current passing pair-integrity report and ranking identities authenticate.

Run the three T3.4 controls and unweighted T3.5 ranker unchanged, then:

```bash
uv run geoembed rank --run-dir RUN --experiment-dir EXP \
  --model exposure_aware --ranking-config configs/ranking/exposure_v1.yaml
uv run geoembed evaluate-pair --pair-manifest PAIR/pair_manifest.json \
  --baseline-experiment-dir REF_BASE INT_BASE \
  --learned-experiment-dir REF_LEARNED INT_LEARNED \
  --ranking-predictions REF_PRED INT_PRED --ranking-reports REF_REPORT INT_REPORT
```

Interpret ESS, weight distribution, clipping, every configured threshold,
observed ranking performance, protected regret/probability recovery, and
coverage separately. These results depend on synthetic exposure and utility
assumptions and are not real-world causal evidence.

## T3.7 frozen ranking-transfer verification

T3.7 affects R8/R9 and only the observed evaluator layer. Its required baseline
is the identical T3.4 controls and corrected T3.5 frozen ranker in one experiment.
After generating those four reports and prediction files, run:

```bash
uv run geoembed evaluate-ranking --run-dir RUN_DIR --experiment-dir EXPERIMENT_DIR
```

Verify `ranking/transfer_slices.json` authenticates the four reports and NPZ
files, records one shared request hash and candidate hash, and reports request,
user, positive-label, and candidate coverage in every slice. The frozen split
must classify cutoff equality as training; post-cutoff POI/catalog identities
must not enter its fitted seen sets. Empty intersections are retained. The
observable early-stage rule uses every request tied at the first post-training
user/region timestamp, with strictly later requests classified late. Utility
regret must remain `unavailable` unless a separate protected evaluator is built
with an explicit truth input.

The integration regression is `tests/test_ranking.py::test_rankers_share_sets_reject_v1_and_never_open_truth`.
It creates all T3.4/T3.5 artifacts on one fixed-seed run and evaluates their
identical prediction/candidate surfaces. Focused unit regressions cover cutoff
equality, unknown POIs, empty slices, duplicate prediction identities, hash
mismatch, and train/test identity leakage. These tests validate the contract,
not external geographic validity or causal recommendation quality.

## Documentation status consistency

`TASKS.md` is the authoritative completion ledger. Run:

```bash
uv run python scripts/check_status_consistency.py
```

The static check compares checked task identifiers with `next task`, `pending`,
and `unimplemented` declarations in `docs/CURRENT_STATUS.md` and
`docs/AGENT_HANDOFF.md`. Its regression fixture specifically rejects the stale
claims that completed T3.5 is next or completed T3.7 is unimplemented. This is
a documentation-integrity check, not experimental evidence for R5, R8, or R9.

## T4.3 participation diagnostic lineage (2026-08-13)

A new 100-user, seven-day immutable diagnostic lineage was generated with
simulator seed `20260813`. The exact commands and SHA-256 identities are indexed
in `docs/artifacts/t4.3-participation-diagnostic-20260813.json`. Deep validation
passed all 22 integrity/mechanism checks and retained one expected small-cohort
travel-volume warning. All six learned controls produced immutable
`training_participation.json` records and frozen exports; the statistical
baseline was generated on the same observed sources and preparation definition.

The frozen privacy strata yield no exported clean nonmembers and no common
support. The attack gate records eligible/matched/unmatched/excluded counts of
100/0/100/100 and realized member/nonmember counts of 0/0; no membership attack
was run. See `docs/decisions/t4.3-participation-diagnostic-20260813.md` for the
bounded interpretation and held-out-user next task.

## T4.3 user-role diagnostic v2 lineage (2026-08-13)

The new 200-user immutable R12 lineage is indexed at
`docs/artifacts/t4.3-user-role-diagnostic-v2-20260813.json` and interpreted at
`docs/decisions/t4.3-user-role-diagnostic-v2-20260813.md`. It uses simulator
seed `20260813` and the unmodified predeclared user-role embedding configuration.
Deep validation passed 22 of 23 checks with one expected small-cohort warning.

The authenticated learned participation record contains 92 users and 1,377
eligible target-training windows plus 42 users and 179 validation/checkpoint-
selection windows. Ninety-four target-train users participated in preprocessing;
the 49 target-test users are recorded separately as clean whole-pipeline
nonparticipants.

Both frozen exports contain only those 49 target-test users. Therefore their
common population contains zero members, fails the frozen total/class/stratum
thresholds, and produces no membership attack. Sensitive probes execute through
the independent gate but the three configured targets are excluded or
unsupported under the code-versioned allowlist. The immutable privacy reports
preserve the unavailable metrics, coverage, split hashes, and exclusions. No
threshold was relaxed, every role remains `diagnostic_control`, and the
selection-dependent conclusion remains unavailable with `no_selected_candidate`.
