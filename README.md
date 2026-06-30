# avOS

Unified tools for autonomous-driving scenario conversion, reconstruction, generation, and visualization.

## Structure

```text
avos/                    # Python package and internal modules
scripts/                 # Public command-line entrypoints
cfgs/                    # Scenario generator configs
data/                    # Generator inference assets
requirements.txt
```

Public entrypoints:

```bash
python scripts/convert.py --help
python scripts/reconstruct.py --help
python scripts/generate.py --help
python scripts/visualize.py --help
```

## Common Commands

Convert one dataset:

```bash
python scripts/convert.py \
  --dataset waymo-open \
  --raw_dataset_path raw_dataset/Waymo \
  --converted_dataset_path converted-dataset \
  --num_workers 8
```

Reconstruct one converted dataset:

```bash
python scripts/reconstruct.py \
  --dataset waymo-open \
  --converted_dataset_path converted-dataset \
  --reconstructed_dataset_path reconstructed_dataset \
  --num_workers 8
```

Generate scenarios:

```bash
python scripts/generate.py normal -- --help
python scripts/generate.py critical -- --help
```

The generator includes its default config, prompts, and map-code assets. Put the external model weight at `data/checkpoint/generator.pth`; it reads converted scenarios from `converted-dataset/<dataset>` and creates the ignored cache under `datasets/trainval_dataset/<dataset>` on demand.

Visualize a scenario:

```bash
python scripts/visualize.py \
  --converted_dataset_path reconstructed_dataset \
  --dataset waymo-open \
  --scenario_ids SCENARIO_ID \
  --result_path video-dataset
```

Detailed local notes are kept in `README_LOCAL.md` and are not committed.
