# avOS: an omni-scenario platform for autonomous driving

avOS constructs a large-scale, highly diverse and high-quality combined real-world and synthetic scenario dataset.

## Setup

```bash
git clone https://github.com/huyuening/avOS.git
cd avOS
conda create -n avos python=3.10 -y
conda activate avos
pip install -r requirements.txt
```

The requirement file is intended to be cross-platform for Windows, macOS, and Linux.

External artifacts for peer review and examples are available on [Figshare](https://doi.org/10.6084/m9.figshare.32845949):

This record contains the scenario extender model weight and a Waymo Open demo
TFRecord. Download the files from the DOI page and place them as follows:

```text
data/checkpoint/generator.pth
/path_to/raw_dataset/waymo-open/training_20s.tfrecord-00000-of-01000
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
CONVERTED_DATASET_PATH: /path_to/dataset/converted-dataset
RECONSTRUCTED_DATASET_PATH: /path_to/dataset/reconstructed_dataset
GENERATOR_INPUT_PATH: /path_to/dataset/reconstructed_dataset
GENERATED_DATASET_PATH: /path_to/dataset/generated_dataset
VISUALIZATION_INPUT_PATH: /path_to/dataset/generated_dataset
VIDEO_DATASET_PATH: /path_to/dataset/video-dataset
NUM_WORKERS: 8
```

Open scenario dataset links: [argoverse-1](https://www.argoverse.org/av1.html),
[argoverse-2](https://www.argoverse.org/av2.html),
[interaction](https://interaction-dataset.com/),
[levelxdata](https://levelxdata.com/),
[lyft-level-5](https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles/data),
[nuplan](https://www.nuscenes.org/nuplan),
[shifts](https://github.com/Shifts-Project/shifts),
[v2x-seq](https://github.com/AIR-THU/DAIR-V2X),
[waymo-open](https://waymo.com/open/download/#motion-dataset).

After downloading a raw dataset, place or symlink its main folder as
`RAW_DATASET_PATH/<dataset>/`. Supported `<dataset>` names are:
`argoverse-1`, `argoverse-2`, `interaction`, `levelxdata`, `lyft-level-5`,
`nuplan`, `shifts`, `v2x-seq`, and `waymo-open`.

`--dataset` also accepts `all` to run every converter, but `all` is not a raw
dataset folder name.

## 1. Convertor

```text
dataset/raw_dataset/<dataset>/  ->  dataset/converted-dataset/<dataset>/<scenario_id>.pkl
```

Converting a full raw dataset can take a long time. Increase `NUM_WORKERS` in
`cfgs/paths.yaml`, or pass a larger `--num_workers` value, according to your CPU
and memory budget.

Example for Waymo Open Motion Dataset:

```bash
python scripts/convert.py \
  --dataset waymo-open
```

Supported `<dataset>` options:

```text
argoverse-1, argoverse-2, interaction, levelxdata, lyft-level-5, nuplan, shifts, v2x-seq, waymo-open
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
dataset/converted-dataset/<dataset>/<scenario_id>.pkl
  -> dataset/reconstructed_dataset/<dataset>/<scenario_id>.pkl
```

Reconstructing a full converted dataset can also be time-consuming. Use as many
`num_workers` as your machine can stably support.

Example for Waymo Open Motion Dataset:

```bash
python scripts/reconstruct.py \
  --dataset waymo-open
```

## 3. Extender

The current Extender is a preview release rather than the full version. It is
trained on the Waymo Open Motion Dataset and currently supports Waymo scenarios
only. Existing limitations include limited source-domain coverage, limited
generation controls, inference-only checkpoint release, and no released
training/fine-tuning pipeline. The full version is coming with Demand2Scenario.

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
1 = intersection, 2 = intersection approach, 3 = road segment
```

Density types:

```text
1 = dense, 2 = medium, 3 = sparse
```

Use `--device cuda` or omit `--device` when a CUDA GPU is available.

### Critical Generation

Critical generation selects a source scenario by map-code requirements, asks an
MLLM interpreter to convert a text prompt into agent codes, runs the generator,
reconstructs the generated trajectories, and saves one or more scenarios:

```bash
export MLLM_API_KEY=your_metachat_or_openai_key

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
  --gpt-model your_available_mllm_model
```

## 4. Visualizer

Example for Waymo Open Motion Dataset:

```bash
python scripts/visualize.py \
  --dataset waymo-open \
  --scenario_ids GENERATED_SCENARIO_ID \
  --ego_view
```

To visualize reconstructed scenarios instead, set
`VISUALIZATION_INPUT_PATH: dataset/reconstructed_dataset` in `cfgs/paths.yaml`, or
temporarily override the input path:

```bash
python scripts/visualize.py \
  --dataset_path dataset/reconstructed_dataset \
  --dataset waymo-open \
  --scenario_ids SCENARIO_ID \
  --ego_view
```

The `--dataset_path` argument can point to `dataset/converted-dataset`,
`dataset/reconstructed_dataset`, or `dataset/generated_dataset`.
Use `--ego_view` to rotate each frame so the red self-driving car faces upward.
