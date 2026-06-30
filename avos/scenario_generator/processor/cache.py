import os
import pickle

import torch

from config_paths import require_optimized_path, require_trainval_path
from processor.data_preprocess import get_object_code, process


def normalize_file_name(file_name):
    file_name = file_name.strip()
    if not file_name:
        return None
    if file_name.endswith(".pt"):
        file_name = file_name[:-3] + ".pkl"
    if not file_name.endswith(".pkl"):
        file_name = f"{file_name}.pkl"
    return file_name


def cache_path_for(config, file_name):
    file_name = normalize_file_name(file_name)
    return os.path.join(require_trainval_path(config), file_name.replace(".pkl", ".pt"))


def missing_cache_files(config, file_names):
    missing = []
    for file_name in file_names:
        file_name = normalize_file_name(file_name)
        if file_name and not os.path.exists(cache_path_for(config, file_name)):
            missing.append(file_name)
    return missing


def build_cache_file(config, file_name, index=0):
    file_name = normalize_file_name(file_name)
    source_path = os.path.join(require_optimized_path(config), file_name)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing source scenario: {source_path}")

    output_path = cache_path_for(config, file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(source_path, "rb") as file:
        scenario = pickle.load(file)

    data = process(config, scenario)
    data["file"] = file_name
    data["index"] = index
    get_object_code(config, data)
    torch.save(data, output_path)
    return output_path


def ensure_cache_files(config, file_names, verbose=True):
    built = []
    for index, file_name in enumerate(missing_cache_files(config, file_names)):
        output_path = build_cache_file(config, file_name, index=index)
        built.append(output_path)
        if verbose:
            print(f"Generated trainval cache: {output_path}")
    return built
