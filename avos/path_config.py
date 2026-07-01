from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH_CONFIG = PROJECT_ROOT / "cfgs" / "paths.yaml"


def load_path_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_project_path(config_path or DEFAULT_PATH_CONFIG)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Path config must be a YAML mapping: {path}")
    return config


def resolve_project_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_PATH_CONFIG

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def config_path_value(config: dict[str, Any], key: str, default: str | Path | None = None) -> str | None:
    value = config_value(config, key, default)
    if value is None:
        return None
    return str(resolve_project_path(value))


def config_int_value(config: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = config_value(config, key, default)
    return None if value is None else int(value)


def raw_dataset_override(config: dict[str, Any], dataset: str) -> str | None:
    raw_datasets = config.get("RAW_DATASETS") or {}
    if not isinstance(raw_datasets, dict):
        raise ValueError("RAW_DATASETS must be a YAML mapping when provided.")

    value = raw_datasets.get(dataset)
    if value in {None, ""}:
        return None
    return str(resolve_project_path(value))
