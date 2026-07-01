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

Download the required raw datasets and place each dataset under
`raw_dataset/<dataset>/`. The `<dataset>` folder name should use one of the
supported names below because the same value is passed to `--dataset`.

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

If the datasets follow this layout, pass `--raw_dataset_path raw_dataset`; the
converter will automatically use `raw_dataset/<dataset>`. If a single raw
dataset is stored elsewhere, pass that dataset-specific location instead.

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
  --dataset waymo-open \
  --raw_dataset_path raw_dataset \
  --converted_dataset_path converted-dataset \
  --num_workers 8
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
  --dataset waymo-open nuplan interaction \
  --raw_dataset_path raw_dataset \
  --converted_dataset_path converted-dataset \
  --num_workers 8
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
  --dataset waymo-open \
  --converted_dataset_path converted-dataset \
  --reconstructed_dataset_path reconstructed_dataset \
  --num_workers 8
```

## 3. Extender

### Normal Generation

Normal generation selects a source scenario by road type and traffic density,
runs the generator, reconstructs the generated trajectories, and saves one new
scenario:

```bash
python scripts/generate.py normal -- \
  --subset-name waymo-open \
  --input-path2 reconstructed_dataset \
  --output-path1 generated_dataset \
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
  --subset-name waymo-open \
  --input-path2 reconstructed_dataset \
  --output-path1 generated_dataset \
  --prompt-name A.6.12 \
  --num-generations 1 \
  --interpreter-models GPT
```

To use the official OpenAI API for the GPT interpreter:

```bash
python scripts/generate.py critical -- \
  --subset-name waymo-open \
  --input-path2 reconstructed_dataset \
  --output-path1 generated_dataset \
  --interpreter-models GPT \
  --openai-base-url https://api.openai.com/v1 \
  --gpt-model your_available_vision_model
```

## 4. Visualize Scenarios

Visualization reads any scenario root with the layout
`<root>/<dataset>/<scenario_id>.pkl` and writes videos to
`<result_path>/<dataset>/`.

Visualize a reconstructed scenario:

```bash
python scripts/visualize.py \
  --converted_dataset_path reconstructed_dataset \
  --dataset waymo-open \
  --scenario_ids SCENARIO_ID \
  --result_path video-dataset \
  --ego_view
```

Visualize a generated scenario:

```bash
python scripts/visualize.py \
  --converted_dataset_path generated_dataset \
  --dataset waymo-open \
  --scenario_ids GENERATED_SCENARIO_ID \
  --result_path video-dataset/generated \
  --ego_view
```

The argument is named `--converted_dataset_path` for compatibility, but it can
point to `converted-dataset`, `reconstructed_dataset`, or `generated_dataset`.
Use `--ego_view` to rotate each frame so the red self-driving car faces upward.

Useful rendering options:

```bash
python scripts/visualize.py \
  --converted_dataset_path reconstructed_dataset \
  --dataset waymo-open \
  --max_scenarios 10 \
  --num_workers 4 \
  --frame_stride 2 \
  --no_labels \
  --result_path video-dataset
```
