from pathlib import Path

import yaml

from config_paths import is_absolute_path


SCENARIO_GENERATOR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CFGS_ROOT = PROJECT_ROOT / "cfgs"
DATA_ROOT = PROJECT_ROOT / "data"
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def cfg_path(*parts):
    return str(CFGS_ROOT.joinpath(*parts))


def data_path(*parts):
    return str(DATA_ROOT.joinpath(*parts))


def d2s_path(*parts):
    return str(SCENARIO_GENERATOR_ROOT.joinpath(*parts))


def dataset_path(*parts):
    return str(DATASETS_ROOT.joinpath(*parts))


def first_existing(*paths):
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return str(candidate)
    return str(Path(paths[-1]))


def default_config_path():
    return first_existing(
        PROJECT_ROOT / "config.local.yaml",
        CFGS_ROOT / "train.local.yaml",
        CFGS_ROOT / "train.yaml",
        Path("/private/tmp/d2s_runtime/config.local.yaml"),
    )


def default_infer_config_path():
    return first_existing(
        CFGS_ROOT / "infer.local.yaml",
        CFGS_ROOT / "infer.yaml",
    )


def default_weight_path():
    return first_existing(
        DATA_ROOT / "checkpoint" / "generator.pth",
        DATASETS_ROOT / "result" / "generator.pth",
        DATASETS_ROOT / "result" / "model_weights_epoch_96_v3.pth",
        PROJECT_ROOT / "result" / "generator.pth",
        PROJECT_ROOT / "result" / "model_weights_epoch_96_v3.pth",
    )


def resolve_project_path(path):
    if path is None:
        return None
    if is_absolute_path(path):
        return str(path)
    candidate = Path(path)
    return str(PROJECT_ROOT / candidate)


def load_yaml_config(path):
    if path is None or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_infer_config(path=None):
    return load_yaml_config(path or default_infer_config_path())


def config_value(config, *keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def config_path_value(config, *keys, default=None):
    return resolve_project_path(config_value(config, *keys, default=default))
