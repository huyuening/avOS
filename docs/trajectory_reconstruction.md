# Trajectory reconstruction

The unified reconstructor restores the heading-correction core of the legacy
batch entry point `main.py -> utils.py::object_track_reconstruction`. That entry
point used Butterworth filtering. The separate Savitzky–Golay implementation is
not the implementation used here. This is a port with compatibility fixes, not a
byte-for-byte copy of the legacy script.

## Processing

1. Use the explicit validity mask and finite position/heading observations.
   Interpolate positions within the first and last valid observations, using
   cubic, quadratic, or linear interpolation according to the available samples.
   Forward-fill missing headings within that interval.
2. Retain the unified implementation's speed-outlier repair, then filter
   positions with a fourth-order forward-backward Butterworth filter. Compute a
   heading reference from the instantaneous velocity direction of those filtered
   positions. Below 1 m/s, use the recorded/filled heading as the reference.
3. Align observed headings to the reference by integer multiples of 2π. At
   moving samples, reject circular differences greater than 0.1π (18 degrees).
   Reconstruct rejected internal samples with a natural cubic spline and boundary
   samples with linear extrapolation. Two anchors use linear interpolation, one
   uses a constant, and zero anchors fall back to the motion reference.
4. Filter the corrected heading and derive local velocity and acceleration from
   the filtered position and heading. Store final headings in [-π, π), while
   plots unwrap display copies to preserve visual continuity.

The reference assumes that heading broadly follows travel direction at speeds
of at least 1 m/s. Reverse motion, sideways movement, and noisy low-speed
orientation need separate interpretation. An outlier flag is a heuristic, not
independent evidence that an observed heading is incorrect.

## Parameters and adaptations

`ReconstructionConfig` sets `heading_reference_min_speed_mps=1.0`,
`heading_tolerance_rad=0.1*pi`, and `filter_order=4`. The Waymo default cutoff is
1 Hz. The diagnostic gallery explicitly uses 0.5 Hz; it does not change the
production default. Sampling frequency is inferred from timestamps unless set.
Butterworth filtering assumes approximately uniform sampling.

The port preserves the current avOS schema, track lengths, and shared scenario
clock. It does not insert per-object timestamps or apply the legacy 5 m step
truncation. It retains the unified speed repair (`max_speed_mps=80`) instead.
It uses actual timestamps for heading interpolation. Zero coordinates, height,
and heading remain valid when the validity mask says so. It uses a 2π unwrap
period instead of the legacy second π-period unwrap, which could reverse a
directed heading. Explicit masks replace zero sentinels, and short-anchor
fallbacks avoid undefined interpolation. Invalid timestamps and tracks with
fewer than two valid observations are skipped with a reason.

The optional legacy velocity-filtering/reintegration block was disabled in the
batch source and has not been enabled in this port.

## Diagnostic stages

Pass `capture_stages=True` to `reconstruct_object_state` to obtain independent
snapshots. This option does not change the final reconstruction.

- **After step 1 (yellow):** interpolated positions and original headings,
  with missing headings left as NaN for inspection.
- **After step 2 (green):** repaired/interpolated positions and corrected
  headings before final heading filtering. Filtered positions are used internally
  to construct its velocity-direction reference.
- **After step 3 (blue):** final filtered positions/headings and derived motion.

`heading_outlier_mask` identifies samples rejected by the direction comparison.
These stage names group processing by purpose. They do not imply that position
filtering happens only after heading correction.

## Running and checking

The ordinary dataset entry point uses this implementation:

```sh
python scripts/reconstruct.py --dataset waymo-open
python -m unittest discover -s tests -v
```

Build an offline diagnostic gallery in a new output directory:

```sh
python scripts/reconstruct_waymo_catalog.py --cutoff-freq 0.5 \
  --output-dir output/waymo-heading-restored
```

To recompute exactly the objects in an existing selection, add
`--selection-index path/to/trajectory_index.json`. Source files are not modified.
Selection exports are diagnostic subsets, not complete traffic scenes. The
output includes exact pickle states, intermediate snapshots, a trajectory index,
source hashes, parameters, and a rounded offline gallery. See
[visualization instructions](reconstruction_visualization.md).
