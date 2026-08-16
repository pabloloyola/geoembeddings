# Multi-horizon profile supervision integration

This bundle adds an **experimental, observed-only** embedding variant. It is a
complement to the existing pipeline, not a replacement for `single_vector`,
`two_timescale_pc`, or the current default configuration. It must not be
described as selected or production-ready until it passes fresh, matched
evaluation and real-data validation.

## What it adds

`multihorizon_profile_gru` extends the existing MPS-safe two-timescale GRU. The
event encoder and public output contract remain unchanged:

```text
persistent, context, combined
```

During training only, it predicts distributions of later **observed** events:

| Horizon | Route | Labels |
|---|---|---|
| 4 later events | `context` | service, category, region, geohash-5, 4-hour bin, day type |
| 16 later events | `persistent` | same fields |

Inputs for a window are strictly earlier than its target timestamp. Events at
the same timestamp can never appear in each other's history. Future-profile
targets are bounded to the same temporal split, and targets without a later
in-split event are excluded. The model reads only `observed/`; simulator truth
remains evaluator-only.

`multihorizon_profile_detached_control` has the same trunk, heads, labels, and
losses, but detaches the representation supplied to the profile heads. It is
the required control for testing whether profile-loss gradients—not merely
additional capacity—improve the learned user representation.

## Validation and matched development use

The focused regression is:

```bash
uv run pytest tests/test_multihorizon_profile.py
```

Then run the full suite before using a small smoke dataset. For a matched
development comparison, use one immutable dataset root, new experiment roots,
the same model seed, and the two configs:

```bash
for variant in multihorizon_profile_gru multihorizon_profile_detached_control; do
  uv run geoembed prepare --config "configs/embedding/${variant}.yaml" \
    --run-dir RUN_DIR --experiment-dir "experiments/${variant}_seed20260817"
  uv run geoembed train --config "configs/embedding/${variant}.yaml" \
    --run-dir RUN_DIR --experiment-dir "experiments/${variant}_seed20260817"
  uv run geoembed export --config "configs/embedding/${variant}.yaml" \
    --run-dir RUN_DIR --experiment-dir "experiments/${variant}_seed20260817"
done
```

Run existing evaluation/compare surfaces only when immutable source, cutoff,
user, and preparation identities match. Do not use ranking as the primary
selection metric; this work is about reusable user embeddings.

This is development evidence only. Before promotion, repeat with fresh seeds,
matched interventions where applicable, and incoming real-world data under the
same observed-only input contract.
