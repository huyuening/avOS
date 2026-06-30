import copy
import os
from pathlib import Path

import yaml

from config_paths import (
    is_absolute_path,
    optimized_path,
    require_optimized_path,
    require_trainval_path,
    set_optimized_path,
    set_trainval_path,
    trainval_path,
)
from processor.get_train_data import preprocess_file_list


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path):
    if is_absolute_path(path):
        return str(path)
    candidate = Path(path)
    return str(PROJECT_ROOT / candidate)


def load_config(config_or_path):
    if isinstance(config_or_path, (str, os.PathLike)):
        with open(config_or_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    return copy.deepcopy(config_or_path)


def normalize_config_paths(config):
    normalized = copy.deepcopy(config)
    dataset_config = normalized.setdefault("DATASET", {})
    datalist_config = normalized.setdefault("DATALIST", {})

    if optimized_path(normalized):
        set_optimized_path(normalized, resolve_project_path(optimized_path(normalized)))
    if trainval_path(normalized):
        set_trainval_path(normalized, resolve_project_path(trainval_path(normalized)))

    for key in ("PATH", "PROCESSED_PATH"):
        if dataset_config.get(key):
            dataset_config[key] = resolve_project_path(dataset_config[key])

    if datalist_config.get("PATH"):
        datalist_config["PATH"] = resolve_project_path(datalist_config["PATH"])

    return normalized


def normalize_file_name(file_name):
    file_name = file_name.strip()
    if not file_name:
        return None
    if file_name.endswith(".pt"):
        file_name = file_name[:-3] + ".pkl"
    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"
    return file_name


def unique_file_names(file_names):
    seen = set()
    result = []
    for file_name in file_names:
        normalized = normalize_file_name(file_name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def read_file_names_from_modes(config, modes):
    names = []
    datalist_path = config["DATALIST"]["PATH"]
    split_names = config["DATALIST"]["NAME"]
    for mode in modes:
        list_name = split_names.get(mode.upper())
        if not list_name:
            continue
        list_path = os.path.join(datalist_path, list_name)
        if not os.path.exists(list_path):
            continue
        with open(list_path, "r", encoding="utf-8") as file:
            names.extend(line.strip() for line in file)
    return unique_file_names(names)


def existing_source_file_names(config, file_names):
    source_path = require_optimized_path(config)
    existing = []
    missing = []
    for file_name in unique_file_names(file_names):
        if os.path.exists(os.path.join(source_path, file_name)):
            existing.append(file_name)
        else:
            missing.append(file_name)
    return existing, missing


def missing_processed_file_names(config, file_names):
    processed_path = require_trainval_path(config)
    missing = []
    for file_name in unique_file_names(file_names):
        processed_name = file_name.replace(".pkl", ".pt")
        if not os.path.exists(os.path.join(processed_path, processed_name)):
            missing.append(file_name)
    return missing


def ensure_trainval_cache(config_or_path, modes=None, file_names=None, num_workers=None, verbose=True):
    config = normalize_config_paths(load_config(config_or_path))
    modes = modes or ("TRAIN", "VAL", "TEST")

    requested_files = unique_file_names(file_names) if file_names is not None else read_file_names_from_modes(config, modes)
    source_files, missing_sources = existing_source_file_names(config, requested_files)

    if missing_sources and verbose:
        print(f"Skip {len(missing_sources)} requested files not found in {require_optimized_path(config)}.")
    if not source_files:
        raise FileNotFoundError(
            f"No source .pkl files found in {require_optimized_path(config)}. "
            "Run scripts/preprocess.py first."
        )

    missing_cache = missing_processed_file_names(config, source_files)
    if not missing_cache:
        if verbose:
            print(f"Trainval cache ready: {require_trainval_path(config)}")
        return config

    if verbose:
        print(
            f"Trainval cache missing {len(missing_cache)}/{len(source_files)} files. "
            f"Generating from {require_optimized_path(config)}..."
        )
    preprocess_file_list(
        config,
        source_files,
        num_workers=num_workers,
        skip_existing=True,
        desc="TRAINVAL",
        verbose=verbose,
    )
    return config
