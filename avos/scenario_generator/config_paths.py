import re
from pathlib import Path


def is_absolute_path(value):
    if value is None:
        return False
    value = str(value)
    return Path(value).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def dataset_config(config):
    return config.setdefault("DATASET", {})


def optimized_path(config):
    dataset = dataset_config(config)
    return dataset.get("OPTIMIZED_PATH") or dataset.get("PATH")


def trainval_path(config):
    dataset = dataset_config(config)
    return dataset.get("TRAINVAL_PATH") or dataset.get("PROCESSED_PATH")


def set_optimized_path(config, value, keep_legacy=False):
    dataset = dataset_config(config)
    dataset["OPTIMIZED_PATH"] = value
    if keep_legacy:
        dataset["PATH"] = value


def set_trainval_path(config, value, keep_legacy=False):
    dataset = dataset_config(config)
    dataset["TRAINVAL_PATH"] = value
    if keep_legacy:
        dataset["PROCESSED_PATH"] = value


def require_optimized_path(config):
    value = optimized_path(config)
    if not value:
        raise KeyError("DATASET.OPTIMIZED_PATH is not set.")
    return value


def require_trainval_path(config):
    value = trainval_path(config)
    if not value:
        raise KeyError("DATASET.TRAINVAL_PATH is not set.")
    return value
