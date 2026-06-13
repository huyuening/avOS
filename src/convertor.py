import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scenario_convertor.internal import INTERNAL_RUN_ENV


DEFAULT_CONVERTED_DATASET_PATH = "/dataset/x-scene/converted-dataset"


CONVERTERS = {
    "waymo-open": "waymo-open-convertor.py",
    "argoverse-1": "argoverse-1-convertor.py",
    "argoverse-2": "argoverse-2-convertor.py",
    "interaction": "interaction-convertor.py",
    "levelxdata": "levelxdata-convertor.py",
    "lyft-level-5": "lyft-level-5-convertor.py",
    "nuplan": "nuplan-convertor.py",
    "shifts": "shifts-convertor.py",
    "v2x-seq": "v2x-seq-convertor.py",
}


def converter_dir() -> Path:
    return SRC_DIR / "scenario_convertor"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Unified scenario converter. Each converter writes the Waymo-style pickle format."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=["all", *CONVERTERS.keys()],
        help="Dataset converter(s) to run. Use 'all' to run every converter.",
    )
    parser.add_argument(
        "--raw_dataset_path",
        default=None,
        type=str,
        help="Override the raw dataset path. Only valid when one dataset is selected.",
    )
    parser.add_argument(
        "--converted_dataset_path",
        default=DEFAULT_CONVERTED_DATASET_PATH,
        type=str,
        help="The file directory for saving converted datasets.",
    )
    parser.add_argument(
        "--subset_name",
        default=None,
        type=str,
        help="Override the output subset directory. Only valid when one dataset is selected.",
    )
    parser.add_argument(
        "--raw_dataset_version",
        default=None,
        type=str,
        help="Override the raw dataset version. Only valid when one dataset is selected.",
    )
    parser.add_argument(
        "--num_workers",
        default=None,
        type=int,
        help="Number of multiprocessing workers passed to each converter.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        type=str,
        help="Python executable used to launch converter subprocesses.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running later converters if one converter exits with an error.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print converter commands without running them.",
    )
    parser.add_argument(
        "converter_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to a single selected converter. Prefix them with '--'.",
    )
    return parser


def selected_datasets(dataset_args):
    if "all" in dataset_args:
        return list(CONVERTERS.keys())

    selected = []
    for dataset in dataset_args:
        if dataset not in selected:
            selected.append(dataset)
    return selected


def validate_args(args, datasets):
    single_dataset_only = {
        "--raw_dataset_path": args.raw_dataset_path,
        "--subset_name": args.subset_name,
        "--raw_dataset_version": args.raw_dataset_version,
    }
    for flag, value in single_dataset_only.items():
        if value is not None and len(datasets) != 1:
            raise ValueError(f"{flag} can only be used when exactly one dataset is selected.")

    if args.converter_args and len(datasets) != 1:
        raise ValueError("Extra converter args can only be forwarded when exactly one dataset is selected.")


def build_command(args, script_path):
    command = [
        args.python,
        str(script_path),
        "--converted_dataset_path",
        args.converted_dataset_path,
    ]

    if args.raw_dataset_path is not None:
        command.extend(["--raw_dataset_path", args.raw_dataset_path])
    if args.subset_name is not None:
        command.extend(["--subset_name", args.subset_name])
    if args.raw_dataset_version is not None:
        command.extend(["--raw_dataset_version", args.raw_dataset_version])
    if args.num_workers is not None:
        command.extend(["--num_workers", str(args.num_workers)])

    if args.converter_args:
        extra_args = args.converter_args
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        command.extend(extra_args)

    return command


def build_subprocess_env():
    env = os.environ.copy()
    env[INTERNAL_RUN_ENV] = "1"

    pythonpath = env.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(SRC_DIR)

    return env


def run_converter(dataset, args):
    script_path = converter_dir() / CONVERTERS[dataset]
    command = build_command(args, script_path)

    print(f"[scenario_convertor] {dataset}: {' '.join(command)}", flush=True)
    if args.dry_run:
        return 0

    return subprocess.run(command, env=build_subprocess_env()).returncode


def run_converters(args):
    datasets = selected_datasets(args.dataset)
    validate_args(args, datasets)

    failures = []
    for dataset in datasets:
        returncode = run_converter(dataset, args)
        if returncode != 0:
            failures.append((dataset, returncode))
            if not args.continue_on_error:
                return returncode

    if failures:
        print("[scenario_convertor] Failed converters:", flush=True)
        for dataset, returncode in failures:
            print(f"  - {dataset}: exit code {returncode}", flush=True)
        return 1

    return 0


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    return run_converters(args)


if __name__ == "__main__":
    raise SystemExit(main())
