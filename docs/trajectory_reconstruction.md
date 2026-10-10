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

Heading interpolation uses a natural cubic spline for internal gaps with at
least three anchors and linear extrapolation at the boundaries. Two anchors use
linear interpolation, one uses a constant, and no anchors use the motion
reference. Final headings are stored as principal angles. Plots unwrap display
copies and align whole-turn branches without changing saved values.

## Why heading can still look abnormal

The original SG workflow has no low-speed heading fallback. At small speeds,
position noise can dominate the velocity direction. Recorded orientation can
also contain approximately π flips, which are not equivalent 2π representation
changes. Reference unwrapping and spline/extrapolation can then create extra
rotation. SG smoothing cannot determine the true orientation of a stationary
object from its position alone.

The original standalone script selected examples using minimum observation
counts, proximity to the ego vehicle, and speed criteria. In particular, its
vehicle/cyclist examples required mean speed above 5 m/s, minimum speed at least
1 m/s, and maximum speed at most 40 m/s. Those example-selection rules are not
applied to a request to retain all previously selected diagnostic objects.
The gallery therefore reports the fraction of filtered-position reference
samples below 1 m/s and flags tracks where that fraction exceeds one half.
This diagnostic does not change reconstruction, certify heading accuracy,
or hide a candidate. Additional low-speed or overshoot protection would be an
explicit change to the legacy heading algorithm.

## Compatibility adaptations

The port preserves avOS state arrays and the shared scenario clock. It does not
insert per-object timestamps or filter out examples by ego proximity or speed.
It accepts legal zero headings/coordinates using explicit validity masks, uses
actual timestamps for heading interpolation, handles fewer than three anchors,
and reduces the SG window to the largest available odd length for short tracks.
If that window cannot support the polynomial order, filtering is bypassed.
Fewer than two valid observations or invalid timestamps produce a skip reason.
The legacy SG code processed x/y; the avOS port also filters z to retain its
three-dimensional position schema. SG assumes approximately uniform sampling.

The full SG processing agrees with isolated legacy numerical helpers on a
uniformly sampled moving test trajectory, including final position, heading,
velocity, acceleration, and Step 2. The adaptations above mean this is not a
byte-for-byte replication for every possible input.

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
python scripts/reconstruct_waymo_catalog.py --filter-method sg \
  --selection-index output/previous-selection/trajectory_index.json \
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
