import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AVOS_DIR = PROJECT_ROOT / "avos"
GENERATOR_DIR = AVOS_DIR / "scenario_generator"

for path in (AVOS_DIR, GENERATOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from path_config import (
    DEFAULT_PATH_CONFIG,
    config_path_value,
    config_value,
    load_path_config,
)

_CACHE_ROOT = Path(tempfile.gettempdir()) / "avos_generate_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "fontconfig"))


MODES = {
    "normal": "inference_for_normal_scenario",
    "critical": "inference_for_critical_scenario",
}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Unified scenario generation entrypoint."
    )
    parser.add_argument(
        "--path_config",
        "--path-config",
        dest="path_config",
        default=str(DEFAULT_PATH_CONFIG),
        help="YAML file that defines default dataset and output paths.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="normal",
        choices=sorted(MODES),
        help="Generation mode to run.",
    )
    parser.add_argument(
        "generator_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected generator. Prefix them with '--'.",
    )
    return parser


def has_forwarded_arg(args, *names):
    return any(arg == name or arg.startswith(f"{name}=") for arg in args for name in names)


def append_default_arg(args, names, value):
    if value is None or has_forwarded_arg(args, *names):
        return
    args.extend([names[0], str(value)])


def apply_path_config_defaults(generator_args, path_config):
    config = load_path_config(path_config)
    append_default_arg(
        generator_args,
        ("--subset-name", "--subset_name"),
        config_value(config, "DEFAULT_DATASET", "waymo-open"),
    )
    append_default_arg(
        generator_args,
        ("--input-path2", "--input_path2"),
        config_path_value(config, "GENERATOR_INPUT_PATH", "dataset/reconstructed_dataset"),
    )
    append_default_arg(
        generator_args,
        ("--output-path1", "--output_path1"),
        config_path_value(config, "GENERATED_DATASET_PATH", "dataset/generated_dataset"),
    )
    return generator_args


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    generator_args = args.generator_args
    if generator_args and generator_args[0] == "--":
        generator_args = generator_args[1:]
    generator_args = apply_path_config_defaults(generator_args, args.path_config)

    module = __import__(MODES[args.mode], fromlist=["main"])
    return module.main(generator_args)


if __name__ == "__main__":
    raise SystemExit(main())
