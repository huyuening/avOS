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


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    generator_args = args.generator_args
    if generator_args and generator_args[0] == "--":
        generator_args = generator_args[1:]

    module = __import__(MODES[args.mode], fromlist=["main"])
    return module.main(generator_args)


if __name__ == "__main__":
    raise SystemExit(main())
