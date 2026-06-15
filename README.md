# avOS Scenario Convertor

`avOS` 提供统一入口，将多个自动驾驶场景数据集转换为与 `waymo-open-convertor` 一致的 pickle 场景格式，并提供轨迹重构和视频可视化入口。

统一入口是 `src/convertor.py`。`src/scenario_convertor/` 下的各数据集转换器是内部实现，不应直接运行。

视频可视化入口是 `src/visualizer.py`。`src/scenario_visualizer/` 下的可视化模块是内部实现，不应直接运行。

场景重构入口是 `src/reconstructor.py`。`src/scenario_reconstructor/` 下的重构模块是内部实现，不应直接运行。

## 支持的数据集

| `--dataset` | 转换器 | 默认输出子目录 | 默认版本 |
| --- | --- | --- | --- |
| `waymo-open` | Waymo Open Motion Dataset | `waymo-open` | `v1.3.0` |
| `argoverse-1` | Argoverse 1 Motion Forecasting Dataset | `argoverse-1` | `1.1` |
| `argoverse-2` | Argoverse 2 Motion Forecasting Dataset | `argoverse-2` | `2.0` |
| `interaction` | INTERACTION Dataset | `interaction` | `1.1` |
| `levelxdata` | highD/inD/rounD/uniD/exiD | `levelxdata` | `mixed` |
| `lyft-level-5` | Lyft Level 5 Prediction Dataset | `lyft-level-5` | `1.1` |
| `nuplan` | nuPlan Dataset | `nuplan` | `1.1` |
| `shifts` | Shifts Motion Prediction Dataset | `shifts` | `1.0` |
| `v2x-seq` | V2X-Seq Trajectory Forecasting Dataset | `v2x-seq` | `v1.0` |

## 快速开始

### 1. 创建环境

推荐使用 Python 3.10。当前 `requirements.txt` 是通用基础依赖，覆盖 Waymo Open、Argoverse 2 直接 parquet/json 后端、nuPlan 直接解析后端、Lyft Level 5 直接 zarr/protobuf 后端、CSV/XML/地图基础解析依赖以及视频可视化依赖，不包含 `waymo-open-dataset-tf-*`、`av2` 或 `l5kit` 官方包。

```bash
conda create -n avos python=3.10 -y
conda activate avos
pip install -r requirements.txt
```

Windows PowerShell 或 Anaconda Prompt 同样使用：

```powershell
conda create -n avos python=3.10 -y
conda activate avos
pip install -r requirements.txt
```

### 2. 检查入口

```bash
python src/convertor.py --help
python src/reconstructor.py --help
python src/visualizer.py --help
```

### 3. 转换 Waymo Open

```bash
python src/convertor.py \
  --dataset waymo-open \
  --raw_dataset_path /path/to/waymo-open-motion-dataset-v1.3.0 \
  --converted_dataset_path /path/to/converted-dataset \
  --num_workers 8
```

转换后的文件会写入：

```text
/path/to/converted-dataset/waymo-open/*.pkl
```

所有输出文件统一命名为：

```text
[scenario_id].pkl
```

### 4. 重构单个转换后数据集

```bash
python src/reconstructor.py \
  --dataset waymo-open \
  --converted_dataset_path /path/to/converted-dataset \
  --reconstructed_dataset_path /path/to/reconstructed_dataset \
  --num_workers 8
```

重构后的文件会写入：

```text
/path/to/reconstructed_dataset/waymo-open/*.pkl
```

### 5. 可视化场景视频

```bash
python src/visualizer.py \
  --converted_dataset_path converted-dataset \
  --dataset lyft-level-5 \
  --scenario_ids s_15 \
  --result_path video-dataset
```

视频会写入：

```text
video-dataset/lyft-level-5/s_15.mp4
```

## 命令行参数

```bash
python src/convertor.py [options]
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--dataset` | 要转换的数据集。可传多个，例如 `--dataset waymo-open argoverse-1`。默认 `all`。 |
| `--raw_dataset_path` | 原始数据路径。只允许在选择单个数据集时使用。 |
| `--converted_dataset_path` | 转换结果根目录。默认 `/dataset/x-scene/converted-dataset`。 |
| `--subset_name` | 输出子目录名。只允许在选择单个数据集时覆盖。 |
| `--raw_dataset_version` | 原始数据集版本。只允许在选择单个数据集时覆盖。 |
| `--num_workers` | 并行 worker 数。设为 `1` 时使用顺序执行，跨平台调试更稳。 |
| `--continue_on_error` | 转换多个数据集时，一个失败后继续尝试后续数据集。 |
| `--dry_run` | 只打印将要执行的转换命令，不实际转换。 |

示例：

```bash
python src/convertor.py --dataset waymo-open --dry_run
```

```bash
python src/convertor.py \
  --dataset waymo-open \
  --raw_dataset_path /data/waymo \
  --converted_dataset_path /data/converted \
  --subset_name waymo-open-test \
  --raw_dataset_version v1.3.0 \
  --num_workers 1
```

## 转发数据集专属参数

某些转换器有额外参数。额外参数必须在只选择单个数据集时使用，并放在单独的 `--` 后面。

Lyft sample split:

```bash
python src/convertor.py \
  --dataset lyft-level-5 \
  --raw_dataset_path raw_dataset/Lyft \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --split sample \
  --max_scenarios 1
```

nuPlan mini/trainval/test split:

```bash
python src/convertor.py --dataset nuplan -- --split mini
```

Argoverse 1 mini 小规模试跑:

```bash
python src/convertor.py \
  --dataset argoverse-1 \
  --raw_dataset_path "raw_dataset/Argoverse 1" \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --split mini \
  --max_files 1
```

Argoverse 2 mini 小规模试跑:

```bash
python src/convertor.py \
  --dataset argoverse-2 \
  --raw_dataset_path raw_dataset/Argoverse2 \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --split mini \
  --max_files 1
```

INTERACTION 小规模试跑:

```bash
python src/convertor.py \
  --dataset interaction \
  --raw_dataset_path raw_dataset/INTERACTION \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --locations DR_CHN_Merging_ZS \
  --max_files 1 \
  --max_scenarios_per_file 1
```

V2X-Seq mini 小规模试跑:

```bash
python src/convertor.py \
  --dataset v2x-seq \
  --raw_dataset_path raw_dataset/V2X-Seq \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --split mini \
  --max_files 1
```

leveLXData exiD 小规模试跑:

```bash
python src/convertor.py \
  --dataset levelxdata \
  --raw_dataset_path raw_dataset/leveLXData \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --levelx_dataset exiD \
  --max_files 1 \
  --max_scenarios_per_file 1
```

nuPlan 小规模试跑:

```bash
python src/convertor.py \
  --dataset nuplan \
  --raw_dataset_path /path/to/nuplan-motion-dataset-v1.1 \
  --converted_dataset_path /path/to/converted-dataset \
  --num_workers 1 \
  -- \
  --split mini \
  --max_files 1 \
  --max_scenarios_per_file 1
```

如果只想先验证 `.db` 轨迹读取，不读取 `map.gpkg`：

```bash
python src/convertor.py \
  --dataset nuplan \
  --raw_dataset_path /path/to/nuplan-motion-dataset-v1.1 \
  --converted_dataset_path /path/to/converted-dataset \
  --num_workers 1 \
  -- \
  --split mini \
  --max_files 1 \
  --max_scenarios_per_file 1 \
  --skip_maps
```

Lyft train split:

```bash
python src/convertor.py \
  --dataset lyft-level-5 \
  --raw_dataset_path /data/lyft-level-5-motion-dataset \
  -- \
  --split train
```

## 输出格式

每个 `.pkl` 文件是一个 `dict`，顶层字段固定为：

```python
{
    "object_tracks": dict,
    "map_features": dict,
    "dynamic_map_states": dict,
    "extra_information": dict,
}
```

### `object_tracks`

每个 object id 对应一个对象状态：

```python
{
    "type": str,
    "subtype": str,
    "state": {
        "global_center": np.ndarray,       # shape: [T, 3]
        "local_velocity": np.ndarray,      # shape: [T, 2]
        "local_acceleration": np.ndarray,  # shape: [T, 2]
        "heading": np.ndarray,             # shape: [T]
        "size": np.ndarray,                # shape: [T, 3]
        "valid": np.ndarray,               # shape: [T]
    },
    "phase": "CONVERTED",
}
```

### `map_features`

道路元素按 id 存放。常见类型包括：

- `ROAD_LANE`
- `ROAD_LINE`
- `ROAD_EDGE`
- `OTHER`

`ROAD_LANE` 会包含：

```python
{
    "type": "ROAD_LANE",
    "subtype": str,
    "polyline": np.ndarray,
    "speed_limit": float | None,
    "lane_info": {
        "entry_lane_ids": list,
        "exit_lane_ids": list,
        "left_lane_ids": list,
        "right_lane_ids": list,
    },
    "boundary_info": dict,
    "phase": "CONVERTED",
}
```

### `dynamic_map_states`

用于交通灯等动态地图状态。没有动态地图信息的数据集会输出空字典。

### `extra_information`

包含场景元信息：

```python
{
    "id": str,
    "dataset": str,
    "version": str,
    "country": str | None,
    "city": str | None,
    "location": str | None,
    "sdc_id": str | None,
    "scene_length": int,
    "sampling_rate": int,
    "timestamp": np.ndarray,
    "init_time": float | int | None,
}
```

## 场景轨迹重构

统一入口：

```bash
python src/reconstructor.py [options]
```

重构器输入 `converted-dataset/<dataset>/*.pkl`，默认输出 `reconstructed_dataset/<dataset>/*.pkl`。它只处理一个已转换数据集，保持原始 Waymo-style 字典结构和 `[scenario_id].pkl` 文件名，主要重建 object track 的 `global_center`、`local_velocity`、`local_acceleration`、`heading`、`size` 和 `valid`。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--dataset` | 要重构的数据集子目录，例如 `waymo-open`、`lyft-level-5`、`nuplan`。 |
| `--converted_dataset_path` | 转换后的数据根目录。 |
| `--reconstructed_dataset_path` | 重构输出根目录，默认 `reconstructed_dataset`。 |
| `--scenario_ids` | 指定一个或多个场景 id。不传时扫描该数据集目录下的全部 `.pkl`。 |
| `--num_workers` | 并行重构场景数。 |
| `--max_files` | 最多重构多少个场景，适合试跑。 |
| `--cutoff_freq` | 覆盖默认滤波截止频率。默认按数据集选择。 |
| `--min_valid_frames` | 单条 object track 至少需要多少有效帧才重构。 |
| `--max_speed_mps` | 修复插值后超过该速度阈值的跳点，设为 `0` 可关闭。 |
| `--skip_existing` | 输出文件已存在时跳过。 |
| `--dry_run` | 只打印将要处理的文件。 |

快速试跑：

```bash
python src/reconstructor.py \
  --dataset lyft-level-5 \
  --converted_dataset_path converted-dataset \
  --scenario_ids s_15 \
  --num_workers 1
```

重构后可直接可视化输出目录：

```bash
python src/visualizer.py \
  --converted_dataset_path reconstructed_dataset \
  --dataset lyft-level-5 \
  --scenario_ids s_15 \
  --result_path video-dataset
```

## 场景视频可视化

统一入口：

```bash
python src/visualizer.py [options]
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--converted_dataset_path` | 转换后的数据根目录，默认 `converted-dataset`。 |
| `--dataset` | 要可视化的数据集输出子目录，例如 `waymo-open`、`interaction`、`levelxdata`。 |
| `--scenario_ids` | 指定一个或多个场景 id。不传时默认扫描该数据集目录下的全部 `.pkl`。 |
| `--result_path` | 视频输出根目录，默认 `video-dataset`。 |
| `--num_workers` | 并行渲染场景数。 |
| `--max_scenarios` | 最多渲染多少个场景，适合试跑。 |
| `--max_frames` | 每个场景最多渲染多少帧，适合试跑。 |
| `--frame_stride` | 每隔多少帧渲染一帧。大于 1 时会按比例降低默认 FPS，保持视频时长接近真实时间。 |
| `--history_frames` | 显示多少帧历史轨迹，设为 `0` 最快。 |
| `--dpi` | 渲染清晰度，默认 `120`。更高清会更慢。 |
| `--no_labels` | 不绘制车辆 id 和速度文字，速度更快。 |
| `--ego_view` | 以主车为中心旋转视角，让红色主车车头始终朝上。 |

当前可视化流程已经去掉旧版的逐帧 PNG 落盘和二次读图拼视频，改为直接从 Matplotlib canvas 写入 OpenCV `VideoWriter`。静态地图每个场景只绘制一次，轨迹状态也会预先整理，通常明显快于旧脚本。

快速试跑：

```bash
python src/visualizer.py \
  --converted_dataset_path converted-dataset \
  --dataset lyft-level-5 \
  --scenario_ids s_15 \
  --result_path video-dataset \
  --max_frames 10 \
  --history_frames 2 \
  --dpi 80 \
  --no_labels
```

主车车头朝上视角：

```bash
python src/visualizer.py \
  --converted_dataset_path converted-dataset \
  --dataset lyft-level-5 \
  --scenario_ids s_15 \
  --result_path video-dataset \
  --ego_view
```

## 内部模块不能直接运行

请不要直接运行：

```bash
python src/scenario_convertor/waymo-open-convertor.py
python src/scenario_reconstructor/reconstruction.py
python src/scenario_visualizer/scenario_visualization.py
```

内部转换器、重构器和可视化模块会拒绝直接运行，并提示统一入口。

请统一使用：

```bash
python src/convertor.py --dataset waymo-open
python src/reconstructor.py --dataset waymo-open
python src/visualizer.py --dataset waymo-open --scenario_ids SCENARIO_ID
```

## Waymo Open 说明

本项目的 Waymo Open 转换器不依赖 `waymo-open-dataset-tf-*` 官方 Python 包，而是使用仓库内置的 WOMD protobuf：

```python
from scenario_convertor.womd_proto.pb2.scenario_pb2 import Scenario
```

仍然需要 TensorFlow，因为当前代码使用：

```python
tf.data.TFRecordDataset(...)
```

来读取 Waymo TFRecord 文件。

在 macOS Apple Silicon 上，`requirements.txt` 会安装 `tensorflow-macos==2.10.0`。该版本要求 `protobuf < 3.20`，因此本项目固定使用 `protobuf==3.19.6`。

## Argoverse 1 说明

Argoverse 1 转换器直接读取 motion forecasting CSV 和 `map_files/*.xml`，不需要安装官方 `argoverse` SDK。

目录通常应类似：

```text
raw_dataset/Argoverse 1/
├── mini/
│   └── data/
│       ├── 10.csv
│       └── ...
└── map_files/
    ├── pruned_argoverse_MIA_10316_vector_map.xml
    └── pruned_argoverse_PIT_10314_vector_map.xml
```

也支持官方常见的 `train/data`、`val/data` 和 `test_obs/data` 结构。默认 `--split auto` 会按已有目录自动发现；输出文件名会带 split 前缀，例如 mini 输出为 `m_10.pkl`。

## Argoverse 2 说明

Argoverse 2 转换器直接读取 motion forecasting parquet 和 `log_map_archive_*.json`，不需要安装官方 `av2` SDK。parquet 读取依赖 `pyarrow`，已经合并进唯一的 `requirements.txt`。

目录通常应类似：

```text
raw_dataset/Argoverse2/
└── mini/
    └── 00a0a3e0-1508-45f2-9cf5-e427e1446a33/
        ├── scenario_00a0a3e0-1508-45f2-9cf5-e427e1446a33.parquet
        └── log_map_archive_00a0a3e0-1508-45f2-9cf5-e427e1446a33.json
```

也支持官方常见的 `train`、`val`、`validation` 和 `test` 结构。默认 `--split auto` 会按已有目录自动发现；输出文件名使用 AV2 原始 UUID，例如 `00a0a3e0-1508-45f2-9cf5-e427e1446a33.pkl`。如果只想验证轨迹读取，可以加 `--skip_maps` 跳过地图 JSON。

## nuPlan 说明

nuPlan 转换器默认使用直接解析后端：通过 Python 标准库 `sqlite3` 读取 `.db` 轨迹文件，通过 `pyogrio/geopandas` 读取 `map.gpkg` 地图文件。它不依赖 `nuplan-devkit`、Hydra 或 Torch，因此比官方 devkit 路线更容易在 Windows、macOS 和 Linux 上运行。相关依赖已经合并在唯一的 `requirements.txt` 中。

如果环境中曾安装过旧版 Shapely，请运行 `pip install -U -r requirements.txt`。nuPlan 的 `pyogrio` 地图读取需要 Shapely 2.x。

nuPlan 目录通常应类似：

```text
/path/to/nuplan-motion-dataset-v1.1/
└── nuplan/
    └── dataset/
        ├── nuplan-v1.1/
        │   └── splits/
        │       ├── mini/
        │       ├── trainval/
        │       └── test/
        └── maps/
```

常用 nuPlan 专属参数：

| 参数 | 说明 |
| --- | --- |
| `--split` | 选择 `mini`、`trainval` 或 `test`。 |
| `--scenario_duration_s` | 直接解析时的场景窗口长度，默认 `20.0` 秒。 |
| `--scenario_stride_s` | 场景窗口步长，默认 `20.0` 秒。 |
| `--sample_interval` | 每隔多少个 `lidar_pc` 帧采样一次，默认 `2`，通常约等于从 20 Hz 降到 10 Hz。 |
| `--map_radius` | 以 ego 中心查询地图元素的半径，默认 `150` 米。 |
| `--max_files` | 最多转换多少个 `.db` 文件，适合试跑。 |
| `--max_scenarios_per_file` | 每个 `.db` 最多转换多少个窗口，适合试跑。 |
| `--skip_maps` | 跳过 `map.gpkg`，只验证轨迹转换。 |

转换器会自动处理 nuPlan `.db` 中的短 location 名称，例如将 `las_vegas` 匹配到地图目录 `us-nv-las-vegas-strip`。

nuPlan 输出文件的 `[scenario_id]` 使用场景起始 `lidar_pc` token，与 nuPlan devkit 中 `scenario.token` 的含义保持一致。

## Shifts 说明

Shifts 转换器直接读取 `.pb` 场景文件，所需 protobuf 定义已经内置在 `src/scenario_convertor/shifts_proto/`。因此转换 Shifts 不需要完整安装官方 `ysdc_dataset_api` 包。

官方 `ysdc_dataset_api` 的完整安装会引入 Torch、TensorFlow、W&B、Transformers、OpenCV 等训练/可视化依赖，跨平台和版本约束更重；本项目只保留转换所需的 protobuf 解码能力。

Shifts 目录可以是官方的 `train/development/evaluation` 结构，也可以是小样本结构，例如：

```text
raw_dataset/Shifts/
└── mini/
    └── 000/
        ├── 000000.pb
        └── ...
```

小规模试跑：

```bash
python src/convertor.py \
  --dataset shifts \
  --raw_dataset_path raw_dataset/Shifts \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --max_files 1
```

## Lyft Level 5 说明

Lyft 转换器直接读取 L5Kit zarr 数据和 `semantic_map/semantic_map.pb`，不依赖完整 `l5kit` 包。官方 `l5kit 1.5.0` 的 PyPI 元数据只标注 Python 3.7/3.8，并依赖 `numpy~=1.19.0`、`torch<2`、`torchvision<1`、OpenCV、notebook、bokeh、gym 等训练/可视化依赖；这些约束对 macOS Apple Silicon、Windows 和 Python 3.10 都不够友好。

Lyft 目录通常应类似：

```text
raw_dataset/Lyft/
├── sample.zarr/
├── semantic_map/
│   └── semantic_map.pb
└── meta.json
```

如果 `sample.zarr` 中缺少 `.zgroup/.zarray/.zattrs` 这类 zarr 元数据文件，转换器会按 Lyft/L5Kit 固定 dtype 和 chunk 大小直接解码 chunk 文件。缺少 `meta.json` 时仍会输出轨迹，但 `map_features` 会为空；如需地图特征，请补齐包含 `world_to_ecef` 的 `meta.json`，或通过 `--map_metadata_path` 指定。

Lyft 小规模试跑：

```bash
python src/convertor.py \
  --dataset lyft-level-5 \
  --raw_dataset_path raw_dataset/Lyft \
  --converted_dataset_path converted-dataset \
  --num_workers 1 \
  -- \
  --split sample \
  --max_scenarios 1
```

## INTERACTION 说明

INTERACTION 转换器直接读取 `recorded_trackfiles/*_tracks_*.csv` 和 `maps/*.osm_xy`，不需要额外官方 SDK。输出目录统一为 `interaction`，每个 recording 会按 20 秒切片输出。

目录通常应类似：

```text
raw_dataset/INTERACTION/
├── INTERACTION-Dataset-DR-v1_1/
│   ├── maps/
│   │   ├── DR_CHN_Merging_ZS.osm_xy
│   │   └── ...
│   └── recorded_trackfiles/
│       ├── DR_CHN_Merging_ZS/
│       │   ├── vehicle_tracks_000.csv
│       │   └── ...
│       └── ...
└── INTERACTION-Dataset-TC-v1_0/
    ├── maps/
    └── recorded_trackfiles/
```

常用 INTERACTION 专属参数：

| 参数 | 说明 |
| --- | --- |
| `--locations` | 选择 `all` 或一个/多个 location，例如 `DR_CHN_Merging_ZS`。 |
| `--max_files` | 最多转换多少个 recording，适合试跑。 |
| `--max_scenarios_per_file` | 每个 recording 最多转换多少个 20 秒场景。 |
| `--skip_maps` | 跳过 OSM 地图，只验证轨迹转换。 |

输出文件名使用 `[location]_[recording_id]_[scenario_idx].pkl`，例如 `DR_CHN_Merging_ZS_000_0.pkl`，并且文件名与 `extra_information.id` 保持一致。没有任何轨迹的 20 秒空窗口会被跳过。

## leveLXData / exiD 说明

leveLXData 转换器直接读取各子数据集的 `*_recordingMeta.csv`、`*_tracks.csv`、`*_tracksMeta.csv` 三件套，并解析 lanelet OSM 地图，不依赖额外官方 SDK。exiD 已兼容官方长目录名和本项目当前的短目录名 `exiD-v2.1`。

exiD 目录通常应类似：

```text
raw_dataset/leveLXData/
└── exiD-v2.1/
    ├── data/
    │   ├── 00_recordingMeta.csv
    │   ├── 00_tracks.csv
    │   ├── 00_tracksMeta.csv
    │   └── ...
    └── maps/
        └── lanelet2/
            ├── 0_cologne_butzweiler/
            │   └── location0.osm
            └── ...
```

常用 leveLXData 专属参数：

| 参数 | 说明 |
| --- | --- |
| `--levelx_dataset` | 选择 `all`、`highD`、`inD`、`rounD`、`uniD` 或 `exiD`。 |
| `--max_files` | 最多转换多少个 recording，适合试跑。 |
| `--max_scenarios_per_file` | 每个 recording 最多转换多少个 20 秒场景。 |
| `--skip_maps` | 跳过 lanelet OSM 地图，只验证轨迹转换。 |

exiD 每个 recording 会按 20 秒切片输出，文件名使用 `[dataset]_[recording_id]_[scenario_idx].pkl`，例如 `exiD_00_0.pkl`，并且文件名与 `extra_information.id` 保持一致。

## V2X-Seq 说明

V2X-Seq 转换器直接读取轨迹 CSV、交通灯 CSV 和 `maps/yizhuang_hdmap*.json`，不需要额外安装官方 SDK。当前已兼容本项目中的 cooperative mini 结构，也保留对常见 `train/val/test` split 的自动发现。

目录通常应类似：

```text
raw_dataset/V2X-Seq/
├── cooperative-vehicle-infrastructure/
│   ├── cooperative-trajectories/
│   │   └── mini/
│   │       ├── 4.csv
│   │       └── ...
│   └── traffic-light/
│       ├── 4.csv
│       └── ...
└── maps/
    ├── yizhuang_hdmap1.json
    └── ...
```

常用 V2X-Seq 专属参数：

| 参数 | 说明 |
| --- | --- |
| `--split` | 选择 `auto`、`mini`、`train`、`val` 或 `test`。 |
| `--source` | 选择 `auto`、`cooperative`、`single_vehicle` 或 `single_infrastructure`。 |
| `--max_files` | 最多转换多少个轨迹 CSV，适合试跑。 |
| `--skip_maps` | 跳过 HD map JSON，只验证轨迹转换。 |
| `--skip_dynamic_map_states` | 跳过交通灯 CSV。 |

输出文件名会包含来源和 split 前缀，例如 cooperative mini 的 `12.csv` 会输出为 `c_m_12.pkl`，并且文件名与 `extra_information.id` 保持一致。

## 跨平台建议

- Linux: 最推荐，重依赖数据集更容易配置。
- macOS: 轻量数据集通常可用，TensorFlow 依赖可能需要按 CPU 架构处理。
- Windows: Waymo Open 可尝试 TensorFlow 2.10.x 原生环境；nuPlan 默认直接解析后端不依赖 Linux-only 的 devkit 运行链路。
- Argoverse 1 使用 CSV/XML 直接解析，不依赖官方 `argoverse` SDK。
- Argoverse 2 使用 parquet/JSON 直接解析，不依赖官方 `av2` SDK；parquet 读取依赖跨平台 wheel `pyarrow`。
- INTERACTION 使用 CSV 和 OSM XML 直接解析，不依赖额外官方 SDK。
- nuPlan 地图解析依赖 `pyogrio/geopandas`，本质上依赖 GDAL 相关二进制包；优先使用 conda 或官方 wheel，遇到安装问题时可先用 `--skip_maps` 验证轨迹读取。
- Shifts 使用内置 protobuf 解码 `.pb` 文件，不依赖官方完整 `ysdc_dataset_api` 运行环境。
- Lyft Level 5 使用直接 zarr/protobuf 后端，不依赖完整 `l5kit`。如果原始数据缺少 `meta.json`，会跳过地图但保留轨迹转换。
- leveLXData/exiD 使用 CSV 和 lanelet OSM 直接解析，不依赖官方 SDK；地图投影依赖 `pyproj`。
- V2X-Seq 使用 CSV/JSON 直接解析，不依赖额外官方 SDK，主要依赖已有的 `pandas/numpy`。
- 调试时先使用 `--num_workers 1`，确认单进程可运行后再提高并行数。

## 常见问题

### 为什么没有安装 `waymo-open-dataset-tf-*`？

为了降低跨系统兼容难度。当前代码只需要 Waymo 的 `Scenario` protobuf 定义，而这些 pb2 文件已经随仓库提供。

### 为什么 `--raw_dataset_path` 只能在单数据集时使用？

不同数据集的目录结构完全不同。一次转换多个数据集时，应使用各转换器内置默认路径，或逐个数据集分别指定路径。

### 如何只看将要执行的命令？

```bash
python src/convertor.py --dataset all --dry_run
```

### 如何转换多个数据集？

```bash
python src/convertor.py --dataset waymo-open argoverse-1 v2x-seq --converted_dataset_path /data/converted
```

如果希望其中一个失败后继续运行：

```bash
python src/convertor.py --dataset waymo-open argoverse-1 v2x-seq --continue_on_error
```

## 代码结构

```text
.
├── src/
│   ├── convertor.py                 # 统一转换入口
│   ├── reconstructor.py             # 统一重构入口
│   ├── visualizer.py                # 统一可视化入口
│   ├── scenario_convertor/
│   │   ├── internal.py              # 内部运行守卫
│   │   ├── utils.py                 # 公共格式、保存、并行工具
│   │   ├── waymo-open-convertor.py
│   │   ├── argoverse-1-convertor.py
│   │   ├── argoverse-2-convertor.py
│   │   ├── interaction-convertor.py
│   │   ├── levelxdata-convertor.py
│   │   ├── lyft-level-5-convertor.py
│   │   ├── lyft_proto/              # Lyft 地图 protobuf 定义
│   │   ├── nuplan-convertor.py
│   │   ├── shifts-convertor.py
│   │   ├── shifts_proto/            # Shifts 场景 protobuf 定义
│   │   └── v2x-seq-convertor.py
│   ├── scenario_reconstructor/
│   │   ├── reconstruction.py        # 单数据集重构流程
│   │   └── object_track_reconstruction.py
│   └── scenario_visualizer/
│       ├── scenario_visualization.py
│       └── utils.py
├── requirements.txt                 # 通用基础依赖
└── README.md
```
