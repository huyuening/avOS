import argparse
import multiprocessing
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_CONVERTED_DATASET_PATH = "converted-dataset"
DEFAULT_RESULT_PATH = "video-dataset"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="avOS scenario visualizer")
    parser.add_argument(
        "--converted_dataset_path",
        "--old_dataset_path",
        dest="converted_dataset_path",
        default=DEFAULT_CONVERTED_DATASET_PATH,
        help="Converted dataset root directory.",
    )
    parser.add_argument(
        "--dataset",
        "--dataset_name",
        dest="dataset_name",
        default="waymo-open",
        help="Converted dataset subset name, for example waymo-open or interaction.",
    )
    parser.add_argument(
        "--scenario_ids",
        nargs="*",
        default=None,
        help="Scenario ids to visualize. Defaults to all pkl files in the dataset directory.",
    )
    parser.add_argument(
        "--result_path",
        "--output_path",
        dest="result_path",
        default=DEFAULT_RESULT_PATH,
        help="Output video root directory.",
    )
    parser.add_argument("--num_workers", default=1, type=int, help="Number of scenarios to render in parallel.")
    parser.add_argument("--max_scenarios", default=None, type=int, help="Render at most this many scenarios.")
    parser.add_argument("--max_frames", default=None, type=int, help="Render at most this many frames per scenario.")
    parser.add_argument("--frame_stride", default=1, type=int, help="Render every Nth frame.")
    parser.add_argument("--history_frames", default=20, type=int, help="Number of previous frames to show as faded history.")
    parser.add_argument("--dpi", default=120, type=int, help="Video render DPI. Higher values are slower.")
    parser.add_argument("--fps", default=None, type=float, help="Output FPS. Defaults to sampling_rate / frame_stride.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, for example mp4v.")
    parser.add_argument("--no_labels", action="store_true", help="Skip object id and speed labels.")
    parser.add_argument("--ego_view", action="store_true", help="Rotate the scene so the SDC points upward.")
    return parser


def scenario_paths(args):
    dataset_dir = Path(args.converted_dataset_path) / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Cannot find converted dataset directory: {dataset_dir}")

    if args.scenario_ids:
        paths = [dataset_dir / f"{scenario_id}.pkl" for scenario_id in args.scenario_ids]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot find scenario pickle files: {missing}")
    else:
        paths = sorted(dataset_dir.glob("*.pkl"))

    if args.max_scenarios is not None:
        paths = paths[:args.max_scenarios]
    if not paths:
        raise FileNotFoundError(f"Cannot find scenario pickle files under {dataset_dir}")
    return paths


def render_scenario(params):
    args, scenario_path = params
    from scenario_visualizer.utils import scenario_visualization

    with open(scenario_path, "rb") as file:
        data = pickle.load(file)

    scenario_id = data.get("extra_information", {}).get("id", scenario_path.stem)
    traffic_events = pd.DataFrame([])
    return scenario_visualization(
        data,
        traffic_events,
        str(scenario_id),
        args.result_path,
        args.dataset_name,
        dpi=args.dpi,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        history_frames=args.history_frames,
        show_labels=not args.no_labels,
        codec=args.codec,
        fps=args.fps,
        ego_view=args.ego_view,
    )


def visualize(args):
    paths = scenario_paths(args)
    params = [(args, path) for path in paths]
    if args.num_workers == 1:
        return list(tqdm(map(render_scenario, params), total=len(params)))

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        return list(tqdm(pool.imap(render_scenario, params), total=len(params)))


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    visualize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit("Scenario visualizer modules are internal. Use src/visualizer.py.")
