# Reconstruction visualization

The implementation is described in [trajectory reconstruction](trajectory_reconstruction.md).
The default is the full legacy Savitzky–Golay workflow (31 samples, order 2),
including heading correction, velocity filtering, and position reintegration.
Butterworth remains an explicit alternative. Compatibility adaptations and
low-speed heading limitations are recorded in the method documentation.

## Three-example figure

```sh
python tools/reconstruction/plot_waymo_reconstruction_examples.py \
  --export-current-stages --filter-method sg \
  --output-dir output/waymo-reconstruction-figure
```

The default example uses Waymo scenario `104202f4f2590dff`, with vehicle 1,
cyclist 197, and pedestrian 158. Override these using `--scenario-id` and
`--object-ids` (vehicle, cyclist, pedestrian order). Each example has a 2×3 grid;
three examples are stacked vertically. Labels a, b, and c sit outside the top
left of each group. Their meanings appear in the generated caption.

Outputs include PDF, SVG, PNG, CSV, stage snapshots, and provenance. The style
uses a white background, framed axes, no grid, black original observations,
yellow Step 1, green Step 2, and blue Step 3. Heading panels show Raw data, After step 1, and After step 2 only.
Step 3 remains in position and motion panels. Step 1 fills position gaps but
keeps missing headings as NaN; heading correction/interpolation happens in
Step 2. Thus a missing heading can still leave a gap in Step 1 local motion. SG Step 3 motion panels use the saved filtered velocity and acceleration.
Earlier stages derive motion from their position and heading arrays.

Time axes span 0–20 s. No samples are extrapolated beyond the observation span.
The plot origin and orientation come from the first valid raw observation and
are shared by every stage. Position axes use equal aspect. Y-axis label anchors
are aligned by column.

| Quantity | Default display range |
|---|---|
| Heading | All stored values visible; numeric radian ticks |
| Longitudinal velocity | 0 to 20 m/s |
| Lateral velocity | −5 to 5 m/s |
| Longitudinal acceleration | −8 to 6 m/s² |
| Lateral acceleration | −4 to 4 m/s² |
| Relative longitudinal position | −10 to 150 m |
| Relative lateral position | −40 to 40 m |

Heading values are copied without unwrapping, normalization, or branch alignment.
Numerical jumps (including approximately 2π jumps) remain visible. Missing values
stay missing. Heading limits cover all displayed values with padding, with at
least a 2π-wide window. Other physical quantities retain the standard bounds.
No saved states or reconstructed motion are changed by these display choices.

## Offline selection gallery

```sh
python tools/reconstruction/reconstruct_waymo_catalog.py \
  --filter-method sg --output-dir output/waymo-catalog
```

Open `gallery/index.html` in the output folder. Filter by type, scene, or object,
select one example per type, and export the selection JSON. Full-range mode
reveals curves outside the standard bounds. Heading limits already cover all displayed values by default.

`tools/reconstruction/select_waymo_anomalies.py` selects 50 candidates per type using equal
weights on within-type percentiles of missing fraction, circular angular-speed
P99, acceleration-magnitude P99, and jerk-magnitude P99. Metrics are calculated
from full-precision raw data. A 2π representation jump is not counted as extreme
angular speed. This is a browsing heuristic, not a trajectory-quality benchmark.
The local sample contains only 50 cyclists, so all are retained.

To update the same selected objects after changing the reconstruction:

```sh
python tools/reconstruction/reconstruct_waymo_catalog.py \
  --selection-index output/previous-selection/trajectory_index.json \
  --filter-method sg --output-dir output/refreshed-selection
```

The output directory must be new. Selection ranks are preserved; objects are
not reranked using the new reconstruction. Exact output states remain local and
are not needed in the source-code repository.

## Moving examples with the original SG algorithm

Use `tools/reconstruction/select_waymo_moving_tracks.py` to apply the original SG motion
criteria to the full ranked raw-data index, then pass its selection index to
`reconstruct_waymo_catalog.py --filter-method sg`. The selection audit records
all exclusions. No low-speed or interpolation-overshoot protection is added.
Types with fewer than 50 eligible candidates retain their actual counts.
