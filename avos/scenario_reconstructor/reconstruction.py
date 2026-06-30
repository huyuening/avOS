from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

if __name__ == "__main__" and __package__ in {None, ""}:
    raise SystemExit("Scenario reconstructor modules are internal. Use scripts/reconstruct.py.")

import numpy as np
from tqdm import tqdm

from scenario_reconstructor.object_track_reconstruction import (
    ReconstructionConfig,
    reconstruct_object_state,
)


RECOMMENDED_RECONSTRUCTED_DATASET_PATH = "reconstructed_dataset"


@dataclass(frozen=True)
class ReconstructionTask:
    dataset: str
    input_path: Path
    output_path: Path
    cutoff_freq: float | None
    min_valid_frames: int
    max_speed_mps: float


@dataclass
class ScenarioStats:
    scenario_id: str
    tracks_total: int = 0
    tracks_reconstructed: int = 0
    tracks_skipped: int = 0
    filled_samples: int = 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct one converted scenario dataset in the unified avOS pickle format.",
    )
    parser.add_argument(
        "--dataset",
        "--subset_name",
        dest="dataset",
        required=True,
        help="Converted dataset subdirectory to reconstruct, for example waymo-open or lyft-level-5.",
    )
    parser.add_argument(
        "--converted_dataset_path",
        required=True,
        help="Root directory containing converted dataset subdirectories.",
    )
    parser.add_argument(
        "--reconstructed_dataset_path",
        required=True,
        help=f"Root directory for reconstructed dataset output. Recommended: {RECOMMENDED_RECONSTRUCTED_DATASET_PATH}.",
    )
    parser.add_argument(
        "--scenario_ids",
        nargs="+",
        default=None,
        help="Optional scenario IDs or pickle filenames to reconstruct.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit for quick reconstruction tests.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Number of worker processes. Recommended: 8. Use 1 for easier debugging.",
    )
    parser.add_argument(
        "--cutoff_freq",
        type=float,
        default=None,
        help="Override the dataset-specific Butterworth cutoff frequency.",
    )
    parser.add_argument(
        "--min_valid_frames",
        type=int,
        default=2,
        help="Minimum valid frames required before reconstructing one object track.",
    )
    parser.add_argument(
        "--max_speed_mps",
        type=float,
        default=80.0,
        help="Repair interpolated points that imply an unrealistically high step speed. Set <=0 to disable.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scenarios whose reconstructed pickle already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned reconstruction files without writing output.",
    )
    return parser


def reconstruct_dataset(args: argparse.Namespace) -> int:
    input_dir = Path(args.converted_dataset_path) / args.dataset
    output_dir = Path(args.reconstructed_dataset_path) / args.dataset

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Converted dataset directory does not exist: {input_dir}")

    tasks = build_tasks(args, input_dir, output_dir)
    if not tasks:
        print(f"[scenario_reconstructor] No scenarios to reconstruct for {args.dataset}.", flush=True)
        return 0

    if args.dry_run:
        for task in tasks:
            print(f"[scenario_reconstructor] {task.input_path} -> {task.output_path}", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = run_tasks(tasks, args.num_workers)
    print_summary(args.dataset, stats, output_dir)
    return 0


def build_tasks(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> list[ReconstructionTask]:
    input_files = resolve_input_files(input_dir, args.scenario_ids)
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    tasks = []
    for input_path in input_files:
        output_path = output_dir / input_path.name
        if args.skip_existing and output_path.exists():
            continue
        tasks.append(
            ReconstructionTask(
                dataset=args.dataset,
                input_path=input_path,
                output_path=output_path,
                cutoff_freq=args.cutoff_freq,
                min_valid_frames=max(1, args.min_valid_frames),
                max_speed_mps=args.max_speed_mps,
            )
        )
    return tasks


def resolve_input_files(input_dir: Path, scenario_ids: Iterable[str] | None) -> list[Path]:
    if scenario_ids:
        input_files = []
        for scenario_id in scenario_ids:
            filename = scenario_id if scenario_id.endswith(".pkl") else f"{scenario_id}.pkl"
            input_path = input_dir / filename
            if not input_path.is_file():
                raise FileNotFoundError(f"Cannot find scenario pickle: {input_path}")
            input_files.append(input_path)
        return input_files

    return sorted(input_dir.glob("*.pkl"))


def run_tasks(tasks: list[ReconstructionTask], num_workers: int) -> list[ScenarioStats]:
    if num_workers <= 1:
        return [process_scenario(task) for task in tqdm(tasks, total=len(tasks))]

    with Pool(processes=num_workers, maxtasksperchild=20) as pool:
        return list(tqdm(pool.imap_unordered(process_scenario, tasks), total=len(tasks)))


def process_scenario(task: ReconstructionTask) -> ScenarioStats:
    scenario = read_pickle(task.input_path)
    timestamp = scenario_timestamp(scenario)
    sampling_rate = scenario_sampling_rate(scenario, timestamp)
    config = ReconstructionConfig(
        dataset=task.dataset,
        cutoff_freq=task.cutoff_freq,
        sampling_rate=sampling_rate,
        min_valid_frames=task.min_valid_frames,
        max_speed_mps=task.max_speed_mps,
    )

    stats = ScenarioStats(scenario_id=task.input_path.stem)
    object_tracks = scenario.get("object_tracks", {})
    stats.tracks_total = len(object_tracks)

    for object_track in object_tracks.values():
        state = object_track.get("state")
        if not isinstance(state, dict):
            stats.tracks_skipped += 1
            continue

        result = reconstruct_object_state(state, timestamp, config)
        if result.reconstructed:
            object_track["state"] = result.state
            stats.tracks_reconstructed += 1
            stats.filled_samples += result.filled_samples
        else:
            stats.tracks_skipped += 1

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pickle(task.output_path, scenario)
    return stats


def scenario_timestamp(scenario: dict) -> np.ndarray:
    extra_information = scenario.get("extra_information", {})
    timestamp = extra_information.get("timestamp")
    if timestamp is None:
        scene_length = int(extra_information.get("scene_length", 0))
        sampling_rate = float(extra_information.get("sampling_rate", 10.0))
        if scene_length <= 0:
            raise ValueError("Scenario is missing both extra_information.timestamp and scene_length.")
        timestamp = np.arange(scene_length, dtype=np.float64) / sampling_rate

    timestamp = np.asarray(timestamp, dtype=np.float64)
    if timestamp.ndim != 1 or len(timestamp) < 2:
        raise ValueError("Scenario timestamp must be a one-dimensional array with at least two values.")
    return timestamp


def scenario_sampling_rate(scenario: dict, timestamp: np.ndarray) -> float:
    sampling_rate = scenario.get("extra_information", {}).get("sampling_rate")
    if sampling_rate is not None and float(sampling_rate) > 0:
        return float(sampling_rate)

    dt = np.diff(timestamp)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 10.0
    return float(1.0 / np.median(dt))


def read_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file)


def write_pickle(path: Path, scenario: dict) -> None:
    with path.open("wb") as file:
        pickle.dump(scenario, file, protocol=pickle.HIGHEST_PROTOCOL)


def print_summary(dataset: str, stats: list[ScenarioStats], output_dir: Path) -> None:
    scenario_count = len(stats)
    tracks_total = sum(item.tracks_total for item in stats)
    tracks_reconstructed = sum(item.tracks_reconstructed for item in stats)
    tracks_skipped = sum(item.tracks_skipped for item in stats)
    filled_samples = sum(item.filled_samples for item in stats)
    print(
        "[scenario_reconstructor] "
        f"{dataset}: {scenario_count} scenarios, "
        f"{tracks_reconstructed}/{tracks_total} tracks reconstructed, "
        f"{tracks_skipped} tracks skipped, "
        f"{filled_samples} samples filled. Output: {output_dir}",
        flush=True,
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    return reconstruct_dataset(args)


if __name__ == "__main__":
    raise SystemExit("Scenario reconstructor modules are internal. Use scripts/reconstruct.py.")
