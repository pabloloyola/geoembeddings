# Migration from the two-project layout

The unified CLI intentionally removes paths that callers previously had to
assemble by hand.

| Earlier argument or command | Unified equivalent |
|---|---|
| `python kanto_simulator.py --output DATASET` | `geoembed simulate --run-dir DATASET` |
| `python validate_kanto.py DATASET` | `geoembed validate --run-dir DATASET` |
| `--observed-dir DATASET/observed` | `--run-dir DATASET` |
| `--truth-dir DATASET/truth` | `--run-dir DATASET` |
| `--prepared-dir EXP/prepared` | `--experiment-dir EXP` |
| `--output-dir EXP/model` | `--experiment-dir EXP` |
| `--checkpoint EXP/model/best_model.pt` | resolved from `--experiment-dir EXP` |
| `--embeddings EXP/embeddings.npz` | resolved from `--experiment-dir EXP` |

Existing simulator runs remain readable when their manifest predates the
explicit contract field, provided the canonical observed files are present.
New runs declare dataset contract `geoembeddings-dataset/1.0` in
`manifest.json`.

