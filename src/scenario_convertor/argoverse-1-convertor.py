try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scenario_convertor.utils import add_common_converter_args, object_state_template, run_tasks, save_scenario_pickle
except ModuleNotFoundError:
    from utils import add_common_converter_args, object_state_template, run_tasks, save_scenario_pickle


MAX_DISTANCE = 150
SPLIT_PREFIX = {"mini": "m", "train": "t", "val": "v", "test": "e"}
SPLIT_DIRS = {
    "mini": ("mini/data", "forecasting_sample/data", "sample/data"),
    "train": ("train/data",),
    "val": ("val/data",),
    "test": ("test_obs/data", "test/data"),
}
CITY_MAP_FILES = {
    "PIT": "pruned_argoverse_PIT_10314_vector_map.xml",
    "MIA": "pruned_argoverse_MIA_10316_vector_map.xml",
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Argoverse 1 Motion Forecasting Dataset Converter")
    add_common_converter_args(
        parser,
        default_raw_dataset_path="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/argoverse-1-motion-dataset-v1.1",
        default_subset_name="argoverse-1",
        default_raw_dataset_version="1.1",
        raw_dataset_help="The file directory for A1MD",
    )
    parser.add_argument("--split", choices=("auto", "mini", "train", "val", "test"), default="auto", help="Split to convert.")
    parser.add_argument("--max_files", default=None, type=int, help="Convert at most this many CSV files.")
    parser.add_argument("--skip_maps", action="store_true", help="Skip Argoverse vector map XML parsing.")
    return parser


def sorted_csv_files(data_dir: Path):
    def sort_key(path: Path):
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

    return sorted(data_dir.glob("*.csv"), key=sort_key)


def discover_split_dirs(raw_dataset_path: Path, split: str):
    split_names = [split] if split != "auto" else ("mini", "train", "val", "test")
    ret = []
    for split_name in split_names:
        for relative_path in SPLIT_DIRS[split_name]:
            data_dir = raw_dataset_path / relative_path
            if data_dir.exists():
                ret.append((split_name, data_dir))
                break
    return ret


def list_scenarios(args):
    raw_dataset_path = Path(args.raw_dataset_path)
    split_dirs = discover_split_dirs(raw_dataset_path, args.split)
    if not split_dirs:
        expected = [str(raw_dataset_path / path) for paths in SPLIT_DIRS.values() for path in paths]
        raise FileNotFoundError(f"Cannot find Argoverse 1 data directory. Expected one of: {expected}")

    scenarios = []
    for split_name, data_dir in split_dirs:
        prefix = SPLIT_PREFIX[split_name]
        for csv_path in sorted_csv_files(data_dir):
            scenario_id = f"{prefix}_{csv_path.stem}"
            scenarios.append((args, scenario_id, str(csv_path)))
            if args.max_files is not None and len(scenarios) >= args.max_files:
                return scenarios
    return scenarios


def get_polyline(road_element_ref_ids, polys):
    points = np.asarray([polys[ref_id] for ref_id in road_element_ref_ids if ref_id in polys], dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float64)))


def extract_map_features(raw_map_feature_data, polys, scenario_center):
    ret = {}
    for way in raw_map_feature_data.findall("way"):
        road_lane_id = way.attrib.get("lane_id") or way.attrib.get("id")
        road_lane_ref_ids = [node.attrib["ref"] for node in way.findall("nd") if "ref" in node.attrib]
        road_lane_polyline = get_polyline(road_lane_ref_ids, polys)
        if road_lane_id is None or len(road_lane_polyline) == 0:
            continue

        tags = {
            tag.attrib["k"]: tag.attrib["v"]
            for tag in way.findall("tag")
            if "k" in tag.attrib and "v" in tag.attrib and tag.attrib["k"] not in {"predecessor", "successor"}
        }
        if np.linalg.norm(road_lane_polyline[len(road_lane_polyline) // 2][0:2] - scenario_center[0:2]) > MAX_DISTANCE:
            continue

        ret[road_lane_id] = {
            "type": "ROAD_LANE",
            "subtype": "UNKNOWN",
            "polyline": road_lane_polyline,
            "speed_limit": None,
            "is_intersection": tags.get("is_intersection"),
            "phase": "CONVERTED",
            "lane_info": {
                "entry_lane_ids": [tag.attrib["v"] for tag in way.findall("tag") if tag.attrib.get("k") == "predecessor"],
                "exit_lane_ids": [tag.attrib["v"] for tag in way.findall("tag") if tag.attrib.get("k") == "successor"],
                "left_lane_ids": [] if tags.get("l_neighbor_id") in (None, "None") else [tags["l_neighbor_id"]],
                "right_lane_ids": [] if tags.get("r_neighbor_id") in (None, "None") else [tags["r_neighbor_id"]],
            },
            "boundary_info": {
                "left_boundary_id": None,
                "right_boundary_id": None,
                "left_boundary_polyline": np.array([]),
                "right_boundary_polyline": np.array([]),
            },
        }
    return ret


def get_object_type(object_type):
    if object_type == "AV":
        return "VEHICLE", "CAR"
    if object_type in {"OTHERS", "AGENT"}:
        return "UNKNOWN", "UNKNOWN"
    raise ValueError(f"Unknown object type: {object_type}")


def extract_object_tracks(raw_object_track_data, timestamps):
    ret = {}
    timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    scenario_length = len(timestamps)

    for object_id, object_track in raw_object_track_data.groupby("TRACK_ID", sort=False):
        object_states = object_state_template(scenario_length)
        object_states["type"], object_states["subtype"] = get_object_type(object_track.iloc[0]["OBJECT_TYPE"])
        object_states["phase"] = "CONVERTED"
        for row in object_track.itertuples(index=False):
            frame = timestamp_index[row.TIMESTAMP]
            object_states["state"]["global_center"][frame] = (row.X, row.Y, 0.0)
            object_states["state"]["valid"][frame] = True
        ret[str(object_id)] = object_states
    return ret


@lru_cache(maxsize=4)
def load_city_map(raw_dataset_path: str, city_name: str):
    if city_name not in CITY_MAP_FILES:
        raise ValueError(f"Unknown city name: {city_name}")
    map_path = Path(raw_dataset_path) / "map_files" / CITY_MAP_FILES[city_name]
    if not map_path.exists():
        raise FileNotFoundError(f"Cannot find Argoverse 1 map file: {map_path}")

    raw_map_feature_data = ET.parse(map_path).getroot()
    polys = {}
    for node in raw_map_feature_data.findall("node"):
        if "id" in node.attrib and "x" in node.attrib and "y" in node.attrib:
            polys[node.attrib["id"]] = (float(node.attrib["x"]), float(node.attrib["y"]))
    return raw_map_feature_data, polys


def city_full_name(city_name: str) -> str:
    if city_name == "PIT":
        return "PITTSBURGH"
    if city_name == "MIA":
        return "MIAMI"
    return city_name


def process_scenario(params):
    args, scenario_id, csv_path = params
    raw_object_track_data = pd.read_csv(csv_path)
    timestamps = np.asarray(raw_object_track_data["TIMESTAMP"].drop_duplicates().to_numpy())
    scenario_length = len(timestamps)
    converted_object_track_data = extract_object_tracks(raw_object_track_data, timestamps)

    av_rows = raw_object_track_data[raw_object_track_data["OBJECT_TYPE"] == "AV"]
    if av_rows.empty:
        raise ValueError(f"Cannot find AV track in {csv_path}.")
    sdc_id = str(av_rows.iloc[0]["TRACK_ID"])
    scenario_center = converted_object_track_data[sdc_id]["state"]["global_center"][scenario_length // 2]
    city_name = raw_object_track_data.iloc[0]["CITY_NAME"]

    if args.skip_maps:
        converted_map_feature_data = {}
    else:
        raw_map_feature_data, polys = load_city_map(args.raw_dataset_path, city_name)
        converted_map_feature_data = extract_map_features(raw_map_feature_data, polys, scenario_center)

    scenario = {
        "object_tracks": converted_object_track_data,
        "map_features": converted_map_feature_data,
        "dynamic_map_states": {},
        "extra_information": {
            "id": scenario_id,
            "dataset": "Argoverse 1",
            "version": args.raw_dataset_version,
            "country": "AMERICA",
            "city": city_full_name(city_name),
            "location": None,
            "sdc_id": sdc_id,
            "scene_length": scenario_length,
            "sampling_rate": 10,
            "timestamp": timestamps,
        },
    }
    save_scenario_pickle(scenario, args, scenario_id)


def convert(args):
    scenarios = list_scenarios(args)
    run_tasks(process_scenario, scenarios, len(scenarios), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
