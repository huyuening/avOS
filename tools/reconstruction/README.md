# Reconstruction diagnostic tools

These optional tools support inspection and paper figures. The main workflow
entry points remain in `scripts/`: `convert.py`, `reconstruct.py`, `generate.py`,
and `visualize.py`.

- `plot_waymo_reconstruction_examples.py`: plot three examples and export stages.
- `reconstruct_waymo_catalog.py`: reconstruct tracks and build an offline gallery.
- `select_waymo_anomalies.py`: rank raw-data anomalies and select candidates.
- `select_waymo_moving_tracks.py`: apply the original SG motion criteria.
- `templates/waymo_catalog.html`: offline gallery template.

Run these from the repository root, for example:

```sh
python tools/reconstruction/reconstruct_waymo_catalog.py --help
```

See [visualization instructions](../../docs/reconstruction_visualization.md)
and [method details](../../docs/trajectory_reconstruction.md).
