# avOS Scenario Convertor

`avOS` 提供一个统一入口，将多个自动驾驶场景数据集转换为与 `waymo-open-convertor` 一致的 pickle 场景格式。

统一入口是仓库根目录的 `convertor.py`。`scenario_convertor/` 下的各数据集转换器是内部实现，不应直接运行。

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

Waymo Open 转换器推荐使用 Python 3.8 到 3.10。当前 `requirements.txt` 是 Waymo Open 转换器的最小依赖，不包含 `waymo-open-dataset-tf-*` 官方包。

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
python convertor.py --help
```

### 3. 转换 Waymo Open

```bash
python convertor.py \
  --dataset waymo-open \
  --raw_dataset_path /path/to/waymo-open-motion-dataset-v1.3.0 \
  --converted_dataset_path /path/to/converted-dataset \
  --num_workers 8
```

转换后的文件会写入：

```text
/path/to/converted-dataset/waymo-open/*.pkl
```

## 命令行参数

```bash
python convertor.py [options]
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
python convertor.py --dataset waymo-open --dry_run
```

```bash
python convertor.py \
  --dataset waymo-open \
  --raw_dataset_path /data/waymo \
  --converted_dataset_path /data/converted \
  --subset_name waymo-open-test \
  --raw_dataset_version v1.3.0 \
  --num_workers 1
```

## 转发数据集专属参数

某些转换器有额外参数。额外参数必须在只选择单个数据集时使用，并放在单独的 `--` 后面。

Lyft train split:

```bash
python convertor.py --dataset lyft-level-5 -- --split train
```

nuPlan mini/trainval/test split:

```bash
python convertor.py --dataset nuplan -- --split mini
```

Lyft 自定义 l5kit 配置：

```bash
python convertor.py \
  --dataset lyft-level-5 \
  --raw_dataset_path /data/lyft-level-5-motion-dataset \
  -- \
  --split val \
  --visualisation_config_path /path/to/visualisation_config.yaml
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

## 内部转换器不能直接运行

请不要直接运行：

```bash
python scenario_convertor/waymo-open-convertor.py
```

内部转换器会拒绝直接运行，并提示：

```text
Dataset converters are internal. Use the top-level convertor.py entrypoint.
```

请统一使用：

```bash
python convertor.py --dataset waymo-open
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

## 跨平台建议

- Linux: 最推荐，重依赖数据集更容易配置。
- macOS: 轻量数据集通常可用，TensorFlow 依赖可能需要按 CPU 架构处理。
- Windows: Waymo Open 可尝试 TensorFlow 2.10.x 原生环境；复杂数据集建议 WSL2 或 Docker。
- 调试时先使用 `--num_workers 1`，确认单进程可运行后再提高并行数。

## 常见问题

### 为什么没有安装 `waymo-open-dataset-tf-*`？

为了降低跨系统兼容难度。当前代码只需要 Waymo 的 `Scenario` protobuf 定义，而这些 pb2 文件已经随仓库提供。

### 为什么 `--raw_dataset_path` 只能在单数据集时使用？

不同数据集的目录结构完全不同。一次转换多个数据集时，应使用各转换器内置默认路径，或逐个数据集分别指定路径。

### 如何只看将要执行的命令？

```bash
python convertor.py --dataset all --dry_run
```

### 如何转换多个数据集？

```bash
python convertor.py --dataset waymo-open argoverse-1 v2x-seq --converted_dataset_path /data/converted
```

如果希望其中一个失败后继续运行：

```bash
python convertor.py --dataset waymo-open argoverse-1 v2x-seq --continue_on_error
```

## 代码结构

```text
.
├── convertor.py                     # 唯一公开入口
├── requirements.txt                 # Waymo Open 转换器依赖
└── scenario_convertor/
    ├── internal.py                  # 内部运行守卫
    ├── utils.py                     # 公共格式、保存、并行工具
    ├── waymo-open-convertor.py
    ├── argoverse-1-convertor.py
    ├── argoverse-2-convertor.py
    ├── interaction-convertor.py
    ├── levelxdata-convertor.py
    ├── lyft-level-5-convertor.py
    ├── nuplan-convertor.py
    ├── shifts-convertor.py
    └── v2x-seq-convertor.py
```
