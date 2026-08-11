# Release verification

Verification date: 2026-08-11 UTC.

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
13 passed
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

