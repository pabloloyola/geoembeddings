# Literature guide

The supplied `references/Relevant-papers.txt` is preserved verbatim. This guide
maps those papers to project decisions. The papers provide components and
evaluation ideas; none by itself solves the persistent/routine/context,
exposure, and recommendation problem defined here.

## Priority 1 — Sequence and temporal modeling

### Individual Mobility Prediction via Attentive Marked Temporal Point Processes

Official source: <https://arxiv.org/abs/2109.02715>

AMTPP jointly models trip start time, origin, and destination, with attention,
daily/weekly positional structure, a distribution for peaked inter-event times,
and origin-destination relationships.

Use for:

- R3 time-to-next-event and duration objectives;
- explicit daily/weekly temporal representation;
- a point-process baseline for irregular event time;
- joint rather than independent time/location evaluation.

Do not assume that improved next-trip prediction separates persistent identity
from context. That still requires R1 interventions and branch-specific probes.

### Pretrained Mobility Transformer: A Foundation Model for Human Mobility

Official source: <https://arxiv.org/abs/2406.02578>

PMT uses autoregressive trajectory tokens with spatial and temporal information
and evaluates next-location prediction, trajectory imputation/generation, and
regional characteristic recovery.

Use for:

- a transformer sequence baseline after the evaluator suite exists;
- pretraining objectives and trajectory imputation tests;
- regional and socioeconomic frozen probes;
- multi-city/geographic transfer design.

Do not introduce a large transformer merely to improve capacity. First compare
it with a capacity-matched GRU and verify R1--R8 rather than only next-token
performance.

## Priority 2 — POI representation and recommendation

### Mobility-Embedded POIs: Learning What A Place Is and How It Is Used from Human Movement

Official source: <https://arxiv.org/abs/2601.21149>

ME-POIs aligns temporally contextualized visits with POI representations and
uses multi-scale propagation to help long-tail POIs, emphasizing how places are
used in addition to static identity.

Use for:

- the Phase 4 public POI catalog and candidate encoder;
- POI-function versus POI-identity ablations;
- contrastive visit/POI alignment;
- multi-scale support for sparse and cold POIs;
- unseen-POI and map-enrichment evaluation ideas.

Keep user representations and POI representations conceptually separate. The
project needs both, but persistent user preference cannot be replaced by a
better place embedding.

### ItiNera: Integrating Spatial Optimization with Large Language Models for Open-domain Urban Itinerary Planning

Official source: <https://aclanthology.org/2024.emnlp-industry.104/>

ItiNera decomposes requests, selects POI candidates, spatially orders them, and
generates a personalized itinerary.

Use for:

- candidate generation versus ranking versus spatial ordering boundaries;
- route coherence after candidate relevance has been scored;
- the later itinerary-level Hakone use case.

It is not an immediate modeling priority. First establish request/candidate logs
and single-request ranking. Itinerary optimization is downstream of embedding
quality and candidate scoring.

## Priority 3 — Simulator improvement

### MobEvolve: An Agentic Self-Evolving Heuristic System for Interpretable Human Mobility Generation

Official source: <https://arxiv.org/abs/2606.01640>

MobEvolve treats mobility simulation as an interpretable heuristic system that
is iteratively revised against validation failures and population/individual
distributional targets.

Use for:

- a disciplined calibration loop around the YAML-driven simulator;
- validation diagnostics that propose focused configuration changes;
- preserving interpretability while improving behavioral realism;
- individual and population distribution checks.

Do not allow an agent to modify generator logic solely to make the embedding
model look better. Keep calibration targets separate from downstream model
evaluation, preserve fixed holdout diagnostics, and review every generated code
change.

### Spatio-Temporal Graph Neural Point Process for Traffic Congestion Event Prediction

Official source: <https://ojs.aaai.org/index.php/AAAI/article/view/26669>

STGNPP combines temporal and graph structure with a point process and periodic
gating to predict sparse continuous-time congestion events.

Use for:

- graph-plus-irregular-time baselines if a transport/POI graph is added;
- periodic gating ideas for R3;
- simultaneous event-time and duration prediction;
- reasoning about long-range spatial dependencies.

Its traffic-link graph is not directly equivalent to user mobility histories.
Do not add graph machinery until the simulator exposes a meaningful, stable
graph such as transport connectivity, POI proximity/function, or OD flow.

## Recommended reading and implementation order

1. AMTPP: implement time/duration and periodicity evaluation first.
2. PMT: define a transformer baseline only after the evaluator gate.
3. ME-POIs: design POI metadata/function and cold-start contract.
4. MobEvolve: improve simulator calibration and diagnostic iteration.
5. STGNPP: consider graph/point-process components after graph semantics exist.
6. ItiNera: consider itinerary optimization after candidate ranking works.

## Literature-review rules for the agent

- Read the full paper before implementing a claimed method, not only this guide.
- Record the exact borrowed component and the requirement it addresses.
- Use official papers or repositories as sources for technical details.
- Reproduce a minimal baseline before adapting the component.
- Distinguish faithful reproduction, inspired adaptation, and novel contribution.
- Do not treat results on taxi, metro, traffic, or itinerary datasets as evidence
  for user-preference factorization without a direct experiment here.

