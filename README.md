# avOS

avOS, an omni-scenario platform for autonomous driving, and constructs a large-scale, highly diverse and high-quality combined realistic and synthetic scenario dataset, as shown in Fig. 1. The platform comprises three components: (1) a scenario converter, which proposes an avos standard covering object tracks, map features, dynamic map states and extra information to integrate open datasets and increase scenario scale and diversity; (2) a scenario reconstructor, which provides a three-step reconstruction method for vehicle, cyclist and pedestrian state data to improve scenario quality; and (3) a scenario extender, which further generalizes critical and normal scenarios according to user demands.

## Setup

```bash
conda create -n avos python=3.10 -y
conda activate avos
pip install -r requirements.txt
```

The requirement file is intended to be cross-platform for Windows, macOS, and Linux.

If you use the generator, place the external checkpoint at:

```text
data/checkpoint/generator.pth
```

Optional API keys for critical-scenario generation should be provided through
environment variables:

```bash
export MLLM_API_KEY=your_mllm_or_openai_key
```

## Data Preparation

Set common paths in `cfgs/paths.yaml` before running the pipeline:

```yaml
RAW_DATASET_PATH: /path_to/raw_dataset
CONVERTED_DATASET_PATH: /path_to/converted-dataset
RECONSTRUCTED_DATASET_PATH: /path_to/reconstructed_dataset
GENERATOR_INPUT_PATH: /path_to/reconstructed_dataset
GENERATED_DATASET_PATH: /path_to/generated_dataset
VISUALIZATION_INPUT_PATH: /path_to/generated_dataset
VIDEO_DATASET_PATH: /path_to/video-dataset
NUM_WORKERS: 8
```

Relative paths in this file are resolved from the repository root. Use
`--path_config another_paths.yaml` to run with a different path file.

Download the required raw datasets and place each dataset under
`raw_dataset/<dataset>/`. The `<dataset>` folder name should use one of the
supported names below because the same value is passed to `--dataset`. If you
use another raw-data root, update `RAW_DATASET_PATH` in `cfgs/paths.yaml`.

```text
raw_dataset/
  argoverse-1/
  argoverse-2/
  interaction/
  levelxdata/
  lyft-level-5/
  nuplan/
  shifts/
  v2x-seq/
  waymo-open/
```

If a single raw dataset is stored outside `RAW_DATASET_PATH/<dataset>`, set a
dataset-specific override under `RAW_DATASETS` in `cfgs/paths.yaml`.

```yaml
RAW_DATASETS:
  nuplan: /path_to/nuplan
  waymo-open: /path_to/waymo-open
```

Raw dataset links:

| `<dataset>` option and raw main folder name | Link |
| --- | --- |
| argoverse-1 | [Argoverse 1](https://www.argoverse.org/av1.html) |
| argoverse-2 | [Argoverse 2](https://www.argoverse.org/av2.html) |
| interaction | [INTERACTION Dataset](https://interaction-dataset.com/) |
| levelxdata | [leveLXData](https://levelxdata.com/) |
| lyft-level-5 | [Lyft Motion Prediction for Autonomous Vehicles](https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles/data) |
| nuplan | [nuPlan](https://www.nuscenes.org/nuplan) |
| shifts | [Shifts Project](https://github.com/Shifts-Project/shifts) |
| v2x-seq | [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) |
| waymo-open | [Waymo Open Motion Dataset](https://waymo.com/open/download/#motion-dataset) |

`--dataset` also accepts `all` to run every converter, but `all` is not a raw
dataset folder name.

## 1. Convertor

```text
raw_dataset/<dataset>/  ->  converted-dataset/<dataset>/<scenario_id>.pkl
```

Example for Waymo Open Motion Dataset:

```bash
python scripts/convert.py \
  --dataset waymo-open
```

Supported `<dataset>` options:

```text
argoverse-1
argoverse-2
interaction
levelxdata
lyft-level-5
nuplan
shifts
v2x-seq
waymo-open
```

You can convert several local raw dataset folders with one command:

```bash
python scripts/convert.py \
  --dataset waymo-open nuplan interaction
```

The converted output is the canonical input for reconstruction, generation, and
visualization.

## 2. Reconstructor

```text
converted-dataset/<dataset>/<scenario_id>.pkl
  -> reconstructed_dataset/<dataset>/<scenario_id>.pkl
```

Run one reconstructed dataset:

```bash
python scripts/reconstruct.py \
  --dataset waymo-open
```

## 3. Extender

### Normal Generation

Normal generation selects a source scenario by road type and traffic density,
runs the generator, reconstructs the generated trajectories, and saves one new
scenario:

```bash
python scripts/generate.py normal -- \
  --road-type 1 \
  --density-type 2 \
  --device cpu
```

Road types:

```text
1 = intersection
2 = intersection approach
3 = road segment
```

Density types:

```text
1 = dense
2 = medium
3 = sparse
```

Use `--device cuda` or omit `--device` when a CUDA GPU is available.

### Critical Generation

Critical generation selects a source scenario by map-code requirements, asks an
MLLM interpreter to convert a text prompt into agent codes, runs the generator,
reconstructs the generated trajectories, and saves one or more scenarios:

```bash
export MLLM_API_KEY=your_mllm_or_openai_key

python scripts/generate.py critical -- \
  --prompt-name A.6.12 \
  --num-generations 1 \
  --interpreter-models GPT
```

To use the official OpenAI API for the GPT interpreter:

```bash
python scripts/generate.py critical -- \
  --interpreter-models GPT \
  --openai-base-url https://api.openai.com/v1 \
  --gpt-model your_available_vision_model
```

## 4. Visualize Scenarios

Visualization reads scenarios from `VISUALIZATION_INPUT_PATH` and writes videos
to `VIDEO_DATASET_PATH/<dataset>/`.

Visualize a generated scenario:

```bash
python scripts/visualize.py \
  --dataset waymo-open \
  --scenario_ids GENERATED_SCENARIO_ID \
  --ego_view
```

To visualize reconstructed scenarios instead, set
`VISUALIZATION_INPUT_PATH: reconstructed_dataset` in `cfgs/paths.yaml`, or
temporarily override the input path:

```bash
python scripts/visualize.py \
  --converted_dataset_path reconstructed_dataset \
  --dataset waymo-open \
  --scenario_ids SCENARIO_ID \
  --ego_view
```

The argument is named `--converted_dataset_path` for compatibility, but it can
point to `converted-dataset`, `reconstructed_dataset`, or `generated_dataset`.
Use `--ego_view` to rotate each frame so the red self-driving car faces upward.

Useful rendering options:

```bash
python scripts/visualize.py \
  --dataset waymo-open \
  --max_scenarios 10 \
  --num_workers 4 \
  --frame_stride 2 \
  --no_labels
```
