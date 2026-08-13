# Visualization validity diagnostics

The Kanto visualization notebook emits quantitative counterparts to its maps in
`RUN_DIR/notebook_artifacts/mobility_diagnostics.json`, a per-user CSV, a
transition CSV, and matched plot grids for synthetic `age_group` and
`household_type`. Run it only as an evaluator and explicitly authorize protected
inputs:

```bash
GEOEMBED_RUN_DIR=runs/kanto_pilot GEOEMBED_INCLUDE_TRUTH=true \
  uv run --extra viz python scripts/kanto_visualization_validation.py
```

Without `GEOEMBED_INCLUDE_TRUTH=true`, the notebook exits before opening any
truth table. The reusable calculator accepts observed tables alone by default;
home/work distance, realized GPS error, and missing-event rate remain missing
until evaluator code supplies protected tables explicitly. Preparation,
baseline, training, and export do not import this evaluator module.

## Configuration and failure classes

`configs/simulation/kanto_v1.yaml` contains the versioned
`visualization_diagnostics` section: timezone, unique-location rounding, and
all diagnostic thresholds. Thresholds are simulator plausibility hypotheses,
not empirical Kanto bounds. Negative elapsed time after stable timestamp sorting
is an **integrity failure**. Large but structurally valid distance, speed,
activity, GPS-error, or transition values are **behavioral warnings**.

All spatial calculations call the simulator's established haversine helper.
Longitude and latitude are never treated as planar coordinates. Events are
ordered as timezone-aware instants, then assigned to local `Asia/Tokyo` dates.
A zero-duration transition has a reported distance but an undefined speed.

## Metric interpretation and limitations

| Diagnostic | Definition and map correspondence | Limitation |
|---|---|---|
| Consecutive-stop distance | Haversine distance between adjacent observed events for a user; quantifies map path segments. | Straight-line distance is not road, rail, or travel distance; sparse logging can join unrelated stops. |
| Elapsed time | Hours between adjacent timezone-ordered observations. | It measures logging intervals, not trip duration. Equal timestamps legitimately yield zero. |
| Implied straight-line speed | Consecutive distance divided by positive elapsed time. | A high value may reflect sparse or jittered observations rather than physical motion; zero-duration speed is missing. |
| Daily displacement | Sum of consecutive within-local-day straight-line segments. | It excludes boundary-crossing segments and underestimates unobserved travel. |
| Radius of gyration | Root mean squared haversine distance from a user's mean-coordinate center. | The coordinate mean is only descriptive and sparse/outlier events can dominate it. |
| Unique-location count | Count of coordinate pairs rounded at configured precision. | It is resolution-dependent and GPS noise can split one place into several locations. |
| Home/work distance | Haversine distance between protected simulated home and work coordinates. | Available only with truth; it is not an inferred commute route or duration. |
| Events per day | Events divided by represented local dates. | Zero-event days are absent, so this is observed-day density rather than full-run recording rate. |
| GPS error | Haversine displacement from passive observed coordinates to the nearest same-user true stop within 15 minutes. | Available only with truth; ambiguous stops and timestamp jitter affect matching. |
| Missing-event rate | One minus mean configured per-user/service record probability in observation truth. | It is evaluator-only expected missingness, not reconstruction of every unlogged event. |
| Region transitions | Count of adjacent observations whose nonmissing region labels differ. | It depends on event density and boundaries; it is not a trip count. |

## Demographic support and evidence boundary

The JSON reports every distribution overall and separately for `age_group` and
`household_type`, including group sample size, demographic missing count, and
per-metric count/missing rate. Plots use the same per-user values. These labels
and their relationships are synthetic scenario inputs. Small differences can
arise from finite samples, configured mechanisms, opportunity, or observation
bias and **must not be described as evidence about real age or household
populations**. No causal or external-validity claim is supplied by these plots.
