# avOS

avOS provides a unified pipeline for autonomous-driving scenario conversion,
trajectory reconstruction, scenario generation, and visualization. All converted
datasets are saved as Waymo-style pickle files named `[scenario_id].pkl`, so the
output of one step can be used directly by the next step.

For detailed local notes, debugging records, and machine-specific settings, keep
them in `README_LOCAL.md`.

## Setup

Create a Python 3.10 conda environment and install the shared requirements:

```bash
conda create -n avos python=3.10 -y
conda activate avos
pip install -r requirements.txt
```

The requirement file is intended to be cross-platform for Windows, macOS, and
Linux. The converters avoid dataset SDKs with strict platform constraints where
possible: Waymo uses local WOMD protobuf files, nuPlan uses SQLite/GPKG parsing,
Argoverse 2 uses parquet/JSON parsing, Lyft Level 5 uses direct zarr/protobuf
parsing, and Shifts uses bundled protobuf definitions.

If you use the generator, place the external checkpoint at:

```text
data/checkpoint/generator.pth
```

Optional API keys for critical-scenario generation should be provided through
environment variables:

```bash
export MLLM_API_KEY=your_mllm_or_openai_key
```

## Project Layout

```text
avos/                    # Python package and internal modules
scripts/                 # Public command-line entrypoints
cfgs/                    # Generator configs
data/inference_data/     # Prompt and map-code assets used by generation
data/checkpoint/         # External generator checkpoint location
paper/                   # Paper PDF
requirements.txt
```

Use only the public scripts for normal operation:

```bash
python scripts/convert.py --help
python scripts/reconstruct.py --help
python scripts/generate.py --help
python scripts/visualize.py --help
```

## 1. Convert Raw Datasets

Conversion reads a raw dataset and writes unified scenario pickle files:

```text
raw_dataset/<dataset>/  ->  converted-dataset/<dataset>/<scenario_id>.pkl
```

Example for Waymo Open Motion Dataset:

```bash
python scripts/convert.py \
  --dataset waymo-open \
  --raw_dataset_path raw_dataset/Waymo \
  --converted_dataset_path converted-dataset \
  --num_workers 8
```

Supported dataset names:

```text
waymo-open
argoverse-1
argoverse-2
interaction
levelxdata
lyft-level-5
nuplan
shifts
v2x-seq
```

You can convert several datasets with one command:

```bash
python scripts/convert.py \
  --dataset waymo-open nuplan interaction \
  --converted_dataset_path converted-dataset \
  --num_workers 8
```

When `--raw_dataset_path` is provided, select exactly one dataset because each
dataset has a different raw directory structure. The converted output is the
canonical input for reconstruction, generation, and visualization.

## 2. Reconstruct Object Tracks

Reconstruction repairs missing or noisy object-track states in a converted
dataset and writes the same pickle format to a new root directory:

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

For a quick single-scenario check:

```bash
python scripts/reconstruct.py \
  --dataset waymo-open \
  --converted_dataset_path converted-dataset \
  --reconstructed_dataset_path reconstructed_dataset \
  --scenario_ids SCENARIO_ID \
  --num_workers 1
```

The reconstructed files keep the same directory and filename convention, so any
later step that accepts a scenario root can read either `converted-dataset` or
`reconstructed_dataset`.

## 3. Generate Scenarios

Generation uses the Demand2Scenario inference path. It expects Waymo-style
scenario pickles, map-code assets, and a trained checkpoint:

```text
source scenarios:  reconstructed_dataset/waymo-open/<scenario_id>.pkl
map codes:         data/inference_data/map_codes/map_codes.pkl
checkpoint:        data/checkpoint/generator.pth
output:            generated_dataset/waymo-open/<scenario_id>.pkl
```

The default config in `cfgs/infer.yaml` points to `converted-dataset`. To chain
generation after reconstruction, pass `--input-path2 reconstructed_dataset`.

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

For an offline smoke test without calling an MLLM service:

```bash
printf '2\n1\n9\n1\nNone\n1\n1\n' | \
python scripts/generate.py critical -- \
  --subset-name waymo-open \
  --input-path2 reconstructed_dataset \
  --output-path1 generated_dataset \
  --mock-mllm-response='-- agent 1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' \
  --device cpu
```

Generated scenarios are saved with the same pickle structure, so they can be
visualized with the same visualizer used for converted or reconstructed data.

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

## End-to-End Example

The following commands form one connected Waymo Open pipeline:

```bash
conda activate avos

python scripts/convert.py \
  --dataset waymo-open \
  --raw_dataset_path raw_dataset/Waymo \
  --converted_dataset_path converted-dataset \
  --num_workers 8

python scripts/reconstruct.py \
  --dataset waymo-open \
  --converted_dataset_path converted-dataset \
  --reconstructed_dataset_path reconstructed_dataset \
  --num_workers 8

python scripts/generate.py normal -- \
  --subset-name waymo-open \
  --input-path2 reconstructed_dataset \
  --output-path1 generated_dataset \
  --road-type 1 \
  --density-type 2

python scripts/visualize.py \
  --converted_dataset_path generated_dataset \
  --dataset waymo-open \
  --scenario_ids GENERATED_SCENARIO_ID \
  --result_path video-dataset/generated \
  --ego_view
```

Replace `GENERATED_SCENARIO_ID` with the scenario ID printed by
`scripts/generate.py`.
