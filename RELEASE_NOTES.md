# v0.5.0 release notes

This is an agent-handoff release built around the tested v0.4.x simulator and
model implementation. There is no intended change to data generation, model
mathematics, training, export, evaluation, or comparison behavior.

Added:

- root `AGENTS.md` with binding invariants and development protocol;
- `START_HERE.md` for remote setup and reproduction;
- complete command/input/output/storage reference;
- architecture and evidence handoff;
- R1--R13 measurable requirement matrix;
- fair experiment and rerun protocol;
- phased simulator/model/recommendation roadmap;
- executable task backlog;
- literature-to-project guide based on the supplied papers;
- release verification evidence;
- reproducible `uv.lock`.

Verified:

- 13 tests passed;
- package reports version 0.5.0;
- locked dependency resolution passes;
- 50-user, 7-day learned pipeline completed on CPU;
- baseline export, protected evaluation, and comparison completed on the same
  smoke dataset.

Known scientific limitations remain explicit in `docs/CURRENT_STATUS.md` and
`docs/REQUIREMENTS_MATRIX.md`.

