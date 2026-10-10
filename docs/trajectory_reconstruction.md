# Trajectory reconstruction

The default `sg` method ports the full legacy standalone
`object_track_reconstruction.py` Savitzky–Golay workflow. The previous
`main.py -> utils.py` Butterworth batch workflow remains available with
`--filter-method butterworth`. These were different implementations, not merely
different plotting conversions.

## Savitzky–Golay workflow

1. Interpolate positions within the first and last valid observations, using
   cubic, quadratic, or linear interpolation according to available samples.
   Capture these positions and the original headings as **Step 1 (yellow)**.
2. Compute the instantaneous velocity direction from interpolated positions.
   Align observed headings by whole turns and reject circular differences greater
   than 0.1π (18 degrees). Interpolate the rejected or missing headings. Capture
   this result as **Step 2 (green)**.
3. Apply SG filtering to position, using a **31-sample window and polynomial
   order 2**, with SciPy's `mode="interp"`. Recompute the velocity-direction
   reference from filtered positions and repeat heading correction from the
   original observations. SG-filter the corrected heading, derive local motion,
   SG-filter local velocity, then integrate it using right-endpoint samples.
   Distribute the endpoint position error linearly across the track to match
   both filtered-position endpoints. Capture the result as **Step 3 (blue)**.

The saved final velocity is the filtered velocity used in integration. As in the
legacy code, the endpoint position correction is not differentiated back into
this velocity. Acceleration is derived from filtered global velocity and rotated
into the object frame. The gallery and figure use these saved SG Step 3 motion
fields; earlier stages derive motion from their position/heading arrays.

Heading correction now retains the original SG helper exactly: the fixed 0.1π
criterion, frame-index natural cubic interpolation, linear boundary extrapolation,
and zero-as-missing convention. It does not add low-speed fallback, overshoot
clipping, or the Butterworth port's sparse-anchor fallback. This also preserves
legacy edge cases: zero headings are treated as gaps, and some underdetermined
gaps remain zero. Final headings are stored as principal angles. Plots show stored angles directly without additional display unwrapping or
branch alignment. Reconstruction-internal angle processing remains unchanged.

## Why heading can still look abnormal

The original SG workflow has no low-speed heading fallback. At small speeds,
position noise can dominate the velocity direction. Recorded orientation can
also contain approximately π flips, which are not equivalent 2π representation
changes. Reference unwrapping and spline/extrapolation can then create extra
rotation. SG smoothing cannot determine the true orientation of a stationary
object from its position alone.

The original standalone script selected examples using minimum observation
counts, proximity to the ego vehicle, and speed criteria. The moving-example
selector uses these original speed/length thresholds:

| Type | Valid observations | Mean speed | Minimum speed | Maximum speed |
|---|---|---|---|---|
| Vehicle / cyclist | ≥40 | ≥5 m/s | ≥1 m/s | ≤40 m/s |
| Pedestrian | ≥40 | ≥0.5 m/s | ≥0.1 m/s | ≤10 m/s |

Speed is computed from interpolated positions using the same derivative as the
original SG Step 2. These criteria select motion before reconstruction, not
visually pleasing output afterward. The separate 50 m ego-distance plotting
gate is not used for this motion-only request. Retain the highest original
raw-data anomaly ranks, up to 50 per type. If a type has fewer eligible tracks,
keep its actual count. Every inclusion/exclusion is recorded in an audit file.
The 1 m/s reference-speed diagnostic remains visible; it is not an extra filter,
particularly for pedestrians whose original minimum threshold is 0.1 m/s.

## Compatibility adaptations

The port adapts legacy x/y fields to the avOS position array and preserves the
shared clock. The SG numerical path uses the original heading helper, requires
at least 40 valid observations, and retains original SG parameters, velocity
filtering, right-endpoint integration, and linear endpoint correction. No new
heading protection has been added. Array copying, diagnostics, and principal-angle
storage are interface changes. Z is filtered alongside x/y to retain the avOS
three-dimensional position schema. Original valid observations supply the
interpolation anchors; per-object timestamp insertion is not applied on the
existing shared Waymo timeline. All selected data are approximately 10 Hz.

The full SG processing agrees with isolated legacy numerical helpers on a
uniformly sampled moving test trajectory, including final position, heading,
velocity, acceleration, and Step 2. Schema adaptation is documented separately from the unchanged numerical path
for selected, uniformly sampled x/y tracks.

## Butterworth option

The retained Butterworth variant uses filtered-position velocity as the heading
reference and falls back to recorded/filled heading below 1 m/s. It uses
fourth-order forward-backward filtering, with a Waymo default cutoff of 1 Hz.
The previous 150-case gallery explicitly used 0.5 Hz. It also retains the earlier
80 m/s step-speed repair. These cutoff/speed-repair parameters do not apply to SG.
Butterworth does not perform the SG velocity filtering/reintegration stage.

## Commands and diagnostics

```sh
python scripts/reconstruct.py --dataset waymo-open --filter-method sg
python scripts/reconstruct.py --dataset waymo-open --filter-method butterworth --cutoff_freq 0.5
python -m unittest discover -s tests -v
python tools/reconstruction/select_waymo_moving_tracks.py \
  --ranked-index output/previous-selection/all_ranked_candidates.json \
  --output-dir output/moving-selection
python tools/reconstruction/reconstruct_waymo_catalog.py --filter-method sg \
  --selection-index output/moving-selection/trajectory_index.json \
  --output-dir output/waymo-sg-gallery
```

`--sg-window-length` and `--sg-polyorder` expose the SG parameters.
Use a new output directory to preserve existing results. No cutoff frequency is
used by SG. `capture_stages=True` returns independent stage snapshots without
changing the result. The final-stage `heading_outlier_mask` counts rejected
observations separately from missing headings. `diagnostics` reports effective
SG window and reference speed statistics.

Exports contain exact pickle states, snapshots, a rounded offline gallery,
parameters, source hashes, and a trajectory index. Selection exports are
incomplete diagnostic scenes, not full traffic-scene datasets. See
[visualization instructions](reconstruction_visualization.md).
