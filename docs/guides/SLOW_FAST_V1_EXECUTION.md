# slow_fast_v1 execution guide

This runbook is for the frozen matched slow/fast experiment. It is an
execution recipe, not scientific evidence that the model disentangles
persistent preference, routine, and episode state.

## Purpose and scientific contract

`slow_fast_v1` is an observed-history representation with persistent and
context components. It is compared with the capacity-matched
`slow_fast_capacity_matched_single` control on the same users, cutoffs,
observed source files, seed, objectives, optimizer, batch size, and eight-epoch
budget. The benchmark uses the accepted v4 protected evaluator. Simulator
constants remain hypotheses, and every requirement axis is reported
separately; a single promotion flag is not an aggregate scientific winner.

The frozen benchmark/model source commit is:

```text
1ceeb1ffb18ff6fa35d9ffd804a6ed6c9a160304
```

The execution hardening commit may be a descendant. The runner records both
the frozen source commit and the current execution commit, and refuses to run
if the current checkout is not its descendant or if the frozen model/config
inputs have drifted.

The frozen v4 manifest is:

```text
experiments/multihorizon-profile-s20260817/recoverable_two_state_benchmark_v4/benchmark_freeze_7d5d1d6/benchmark_freeze_manifest.json
sha256: e5f4b29a180b6440fedbd595e6cedba4b935afa1948cc1c04f954b5986540a3c
```

Training, preparation, and export receive only:

```text
RUN_DIR/observed/users_observed.csv.gz
RUN_DIR/observed/observed_events.csv.gz
```

Protected `RUN_DIR/truth/` files are opened only by evaluator code. Do not
copy truth labels, latent preferences, utilities, episodes, choices, or true
coordinates into a model or export directory.

## Prerequisites and capacity checks

Use a clean, committed execution checkout for the model/config inputs. Existing
unrelated dirty files may remain, but the runner rejects drift in the frozen
slow-fast model, training, preparation, export, and evaluator inputs.

```bash
uv sync --locked --extra dev
uv run pytest
uv run geoembed --version
```

The observed held-out run and the completed candidate must already exist. The
runner performs the v4 observed-only preflight, including target chronology,
source hashes, capacity match, seed, and host-memory checks, before control
training. A preflight report with `status: passed` is required before treating
the training stage as eligible to run.

Define the immutable roots once:

```bash
ROOT=/storage/home/pablo.loyola/ideas/geoembeddings
RUN=$ROOT/experiments/multihorizon-profile-s20260817/recoverable_two_state_benchmark_v4/heldout_seed20260823_frozen_7d5d1d6/seed20260823/reference
CANDIDATE=$ROOT/experiments/multihorizon-profile-s20260817/slow_fast_v1_heldout_7d5d1d6/retry1/candidate
RETRY2=$ROOT/experiments/multihorizon-profile-s20260817/slow_fast_v1_heldout_7d5d1d6/retry2
CONTROL=$RETRY2/control
MANIFEST=$ROOT/experiments/multihorizon-profile-s20260817/recoverable_two_state_benchmark_v4/benchmark_freeze_7d5d1d6/benchmark_freeze_manifest.json
CANDIDATE_CONFIG=$ROOT/configs/embedding/slow_fast_v1.yaml
CONTROL_CONFIG=$ROOT/configs/embedding/slow_fast_capacity_matched_single.yaml
```

Every material change gets a new absent experiment root. Never reuse `retry1`,
overwrite its candidate, or place a new control checkpoint beside an old one.

## Prepare and preflight commands

The maintained control runner executes these stages itself for a fresh retry.
The equivalent explicit preparation command is:

```bash
uv run --offline geoembed prepare \
  --run-dir "$RUN" --experiment-dir "$CONTROL" --config "$CONTROL_CONFIG"
```

For a manual preflight after preparation, create a new preflight directory and
run:

```bash
mkdir -p "$RETRY2/preflight"
uv run --offline python -c \
  'import sys,yaml; from geoembeddings.slow_fast_preflight import run_slow_fast_preflight; run_slow_fast_preflight(sys.argv[1],sys.argv[2],yaml.safe_load(open(sys.argv[3])),yaml.safe_load(open(sys.argv[4])),sys.argv[5],sys.argv[6])' \
  "$RUN" "$CANDIDATE/prepared" "$CANDIDATE_CONFIG" "$CONTROL_CONFIG" "$MANIFEST" "$RETRY2/preflight"
```

The preflight must use the candidate preparation and the exact frozen manifest.
It must report identical observed source hashes for candidate and control,
`same_epoch_budget: true`, and `status: passed`.

## Candidate/control workflow

The candidate is trained first and frozen before the control. A valid candidate
has its checkpoint, training report, participation report, and completion
marker authenticated against its prepared metadata and observed source hashes.
The control is then prepared and trained sequentially on the same run. The
control-only runner stops after final-checkpoint validation; it does not run
normal exports, dense exports, or the protected evaluator.

```bash
uv run --offline python scripts/slow_fast_v1_control_runner.py \
  --run-dir "$RUN" \
  --candidate-dir "$CANDIDATE" \
  --control-dir "$CONTROL" \
  --candidate-config "$CANDIDATE_CONFIG" \
  --control-config "$CONTROL_CONFIG" \
  --manifest "$MANIFEST"
```

The runner creates all log, status, prepared, model, and preflight directories
before opening any redirected stream. Each stage writes both a JSON status
record and a one-line `.exit_status` file. `lineage.json` records the frozen
manifest/config/source identities, candidate checkpoint hash, and the fact that
exports and evaluation have not started.

## tmux operation and one-time status checks

Launch detached from the repository root:

```bash
SESSION=geoembed-slow-fast-retry2-control
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec uv run --offline python scripts/slow_fast_v1_control_runner.py --run-dir '$RUN' --candidate-dir '$CANDIDATE' --control-dir '$CONTROL' --candidate-config '$CANDIDATE_CONFIG' --control-config '$CONTROL_CONFIG' --manifest '$MANIFEST'"
```

Record the pane/process identity once immediately after launch:

```bash
tmux list-panes -t "$SESSION" -F '#{pane_pid}' | tee "$RETRY2/status/tmux_pane.pid"
cat "$RETRY2/status/runner.status.json"
cat "$CONTROL/status/control_train.status.json" 2>/dev/null || true
```

Detach with `Ctrl-b d`. Reattach with:

```bash
tmux attach-session -t "$SESSION"
```

One-time checks, without attaching, are:

```bash
tmux has-session -t "$SESSION" 2>/dev/null; echo "tmux_rc=$?"
cat "$RETRY2/status/runner.status.json"
cat "$RETRY2/status/lineage.json"
cat "$RETRY2/preflight/status/preflight.status.json"
cat "$CONTROL/status/control_prepare.status.json"
cat "$CONTROL/status/control_train.status.json"
tail -40 "$CONTROL/logs/control_train.stdout.log"
tail -40 "$CONTROL/logs/control_train.stderr.log"
```

The durable status files, rather than a missing tmux session or a missing
wrapper status file alone, determine whether a stage passed.

## Checkpoint validity rule

Never use a partial checkpoint. A control checkpoint is valid only when all of
these are present and authenticated:

1. `model/best_model.pt` records exactly epoch 8, the configured final epoch;
2. `model/training_report.json` exists and contains eight epoch records;
3. `model/training_participation.json` exists;
4. `training.complete` exists and its checkpoint hash and epoch match the bytes;
5. model variant, seed, config hash, preparation hash, observed source hashes,
   frozen manifest hash, field order, cutoffs, and parameter count match the
   candidate/control contract; and
6. the training stage and final-validation stage have exit status zero.

An epoch-5 or otherwise intermediate `best_model.pt` is not usable, even if it
loads successfully. A wrapper failure after a validated completion marker may
be diagnosed separately, but a wrapper failure before that marker leaves the
checkpoint incomplete.

## Exports after a valid control

Only after the validity rule passes, run the ordinary cutoff export:

```bash
uv run --offline geoembed export \
  --run-dir "$RUN" --experiment-dir "$CONTROL" --config "$CONTROL_CONFIG"
```

For dense, event-aligned embeddings:

```bash
uv run --offline geoembed export-dense \
  --run-dir "$RUN" --experiment-dir "$CONTROL" \
  --config "$CONTROL_CONFIG" --event-stride 1
```

The protected slow-fast evaluator also needs the evaluator-compatible
`model/checkpoint.pt` path. Create it only after validation and only if absent:

```bash
ln -s best_model.pt "$CONTROL/model/checkpoint.pt"
```

The symlink does not create a second checkpoint or change the checkpoint hash.

## Evaluation and promotion

Run the matched evaluator only after both candidate and control have valid
checkpoints and the required normal/dense exports. It authenticates the frozen
v4 manifest and may open truth only within evaluator code:

```bash
uv run --offline python scripts/slow_fast_v1_experiment.py \
  --run-dir "$RUN" \
  --intervention-dir "${RUN%/reference}/intervention" \
  --pair-dir "${RUN%/reference}/pair" \
  --candidate-dir "$CANDIDATE" \
  --control-dir "$CONTROL" \
  --config "$CANDIDATE_CONFIG" \
  --registry "$ROOT/configs/recoverability/recoverable_two_state_benchmark_v4_factor_registry.json" \
  --freeze-manifest "$MANIFEST" \
  --output-dir "$RETRY2/evaluation"
```

The report keeps persistent, context, geometry, next-event, and spatial axes
separate. The current promotion checks require all of the following: candidate
persistent kNN purity improves, candidate persistent separation improves,
candidate context kNN purity improves, candidate context separation improves,
combined next-event accuracy does not decline by more than `0.01` on category,
geohash-5, or geohash-7, and all lineage/protected-truth checks pass. Any false
check means `DO NOT ADVANCE`; do not replace it with a preferred single metric.

## Diagnosing and recovering a stopped launcher

1. Read `runner.status.json`, every stage JSON status, and the corresponding
   stdout/stderr logs.
2. Check whether `control_train.status.json` is `running`, `failed`, or
   `passed`; inspect the recorded exit status and last epoch line.
3. If the stage is `running` but no process exists, treat it as interrupted.
4. Do not export, evaluate, or resume from any checkpoint without
   `training.complete` and the epoch-8/provenance checks above.
5. Do not reuse a stopped directory. Start a new immutable `retry3/control`
   (or another uniquely named root), rerun preparation and preflight, and keep
   the stopped artifacts for audit.

The exact signal may be unavailable when a host, tmux server, or kernel kills a
process. That uncertainty does not turn a partial checkpoint into a final one.

## Cleanup and untracked-artifact rules

Do not delete or overwrite `retry1`, its candidate, old control checkpoints,
dense exports, reports, ZIPs, or generated experiment trees. New `retry2/`
outputs, logs, status files, and checkpoints are generated artifacts and should
remain untracked unless a separate evidence-index task explicitly requests a
small immutable index. Keep source scripts, focused tests, and this guide
tracked. If a run must be abandoned, preserve it and use a new immutable root;
cleanup requires an explicit, validated user request.
