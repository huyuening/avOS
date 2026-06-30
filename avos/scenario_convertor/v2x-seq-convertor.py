try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scenario_convertor.utils import (
        add_common_converter_args,
        global_to_local_velocity,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
        traffic_light_state_template,
    )
except ModuleNotFoundError:
    from utils import (
        add_common_converter_args,
        global_to_local_velocity,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
        traffic_light_state_template,
    )


MAX_DISTANCE = 150
SCENARIO_SOURCES = (
    ("cooperative", ("cooperative-vehicle-infrastructure/cooperative-trajectories", "cooperative-vehicle-infrastructure/vehicle-trajectories")),
    ("single_vehicle", ("single-vehicle/trajectories",)),
    ("single_infrastructure", ("single-infrastructure/trajectories",)),
)
SOURCE_PREFIX = {"cooperative": "c", "single_vehicle": "sv", "single_infrastructure": "si"}
SPLIT_PREFIX = {"mini": "m", "train": "t", "val": "v", "test": "e"}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="V2X-Seq Motion Prediction Dataset Converter")
    add_common_converter_args(
        parser,
        default_raw_dataset_path="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/v2x-seq-motion-dataset",
        default_subset_name="v2x-seq",
        default_raw_dataset_version="v1.0",
        raw_dataset_help="The file directory for V2X-Seq",
    )
    parser.add_argument("--split", choices=("auto", "mini", "train", "val", "test"), default="auto", help="Split to convert.")
    parser.add_argument(
        "--source",
        choices=("auto", "cooperative", "single_vehicle", "single_infrastructure"),
        default="auto",
        help="Scenario source to convert.",
    )
    parser.add_argument("--max_files", default=None, type=int, help="Convert at most this many trajectory CSV files.")
    parser.add_argument("--skip_maps", action="store_true", help="Skip HD map JSON parsing.")
    parser.add_argument("--skip_dynamic_map_states", action="store_true", help="Skip traffic-light CSV parsing.")
    return parser


def sorted_csv_files(data_dir: Path):
    def sort_key(path: Path):
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

    return sorted(data_dir.glob("*.csv"), key=sort_key)


def discover_source_dirs(raw_dataset_path: Path, split: str, source: str):
    source_defs = [item for item in SCENARIO_SOURCES if source in ("auto", item[0])]
    split_names = ("mini", "train", "val", "test") if split == "auto" else (split,)
    ret = []
    for source_name, relative_roots in source_defs:
        for relative_root in relative_roots:
            root = raw_dataset_path / relative_root
            if not root.exists():
                continue
            for split_name in split_names:
                split_dir = root / split_name
                if split_dir.exists():
                    ret.append((source_name, split_name, split_dir))
    return ret


def list_scenarios(args):
    raw_dataset_path = Path(args.raw_dataset_path)
    source_dirs = discover_source_dirs(raw_dataset_path, args.split, args.source)
    if not source_dirs:
        raise FileNotFoundError(f"Cannot find V2X-Seq trajectory directories under {raw_dataset_path}.")

    scenarios = []
    for source_name, split_name, data_dir in source_dirs:
        scenario_prefix = f"{SOURCE_PREFIX[source_name]}_{SPLIT_PREFIX[split_name]}"
        for csv_path in sorted_csv_files(data_dir):
            scenario_id = f"{scenario_prefix}_{csv_path.stem}"
            traffic_light_path = find_traffic_light_path(raw_dataset_path, split_name, csv_path.stem)
            scenarios.append((args, scenario_id, source_name, str(csv_path), str(traffic_light_path) if traffic_light_path else None))
            if args.max_files is not None and len(scenarios) >= args.max_files:
                return scenarios
    return scenarios


def find_traffic_light_path(raw_dataset_path: Path, split_name: str, stem: str) -> Optional[Path]:
    traffic_root = raw_dataset_path / "cooperative-vehicle-infrastructure" / "traffic-light"
    candidates = [
        traffic_root / split_name / f"{stem}.csv",
        traffic_root / f"{stem}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def point_string_to_xyz(value) -> tuple:
    point = ast.literal_eval(value)
    if len(point) == 2:
        return float(point[0]), float(point[1]), 0.0
    return float(point[0]), float(point[1]), float(point[2])


def point_strings_to_xyz(values) -> np.ndarray:
    if not values:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([point_string_to_xyz(point) for point in values], dtype=np.float64)


def get_road_lane_type_and_subtype(road_lane_type):
    if road_lane_type in {"CITY_DRIVING", "LEFT_TURN_WAITING_ZONE", "ROUNDABOUT", "EMERGENCY_LANE"}:
        return "ROAD_LANE", "URBAN_VEHICLE"
    if road_lane_type == "BIKING":
        return "ROAD_LANE", "URBAN_BIKE"
    return "ROAD_LANE", "URBAN_UNKNOWN"


def extract_road_lines(road_line, is_intersection):
    road_line_info = {
        "type": "ROAD_LINE",
        "subtype": "VIRTUAL" if is_intersection else "UNKNOWN",
        "polyline": point_strings_to_xyz(road_line),
        "phase": "CONVERTED",
    }
    return road_line_info


def is_near(polyline: np.ndarray, scenario_center: np.ndarray) -> bool:
    if len(polyline) == 0:
        return False
    midpoint = polyline[len(polyline) // 2][0:2]
    return np.linalg.norm(midpoint - scenario_center[0:2], axis=-1) <= MAX_DISTANCE


def extract_map_features(map_feature_data, scenario_center):
    ret = {}

    for road_lane_id, road_lane in map_feature_data.get("LANE", {}).items():
        left_id = f"{road_lane_id}_1"
        right_id = f"{road_lane_id}_2"
        left_road_line = extract_road_lines(road_lane.get("left_boundary", []), road_lane.get("is_intersection", False))
        right_road_line = extract_road_lines(road_lane.get("right_boundary", []), road_lane.get("is_intersection", False))

        if not (is_near(left_road_line["polyline"], scenario_center) and is_near(right_road_line["polyline"], scenario_center)):
            continue

        ret[left_id] = left_road_line
        ret[right_id] = right_road_line
        road_lane_polyline = point_strings_to_xyz(road_lane.get("centerline", []))

        road_lane_info = {}
        road_lane_info["type"], road_lane_info["subtype"] = get_road_lane_type_and_subtype(road_lane.get("lane_type"))
        road_lane_info["polyline"] = road_lane_polyline
        road_lane_info["speed_limit"] = None
        road_lane_info["is_intersection"] = road_lane.get("is_intersection")
        road_lane_info["phase"] = "CONVERTED"
        road_lane_info["lane_info"] = {
            "entry_lane_ids": [str(item) for item in road_lane.get("predecessors", [])],
            "exit_lane_ids": [str(item) for item in road_lane.get("successors", [])],
            "left_lane_ids": [] if road_lane.get("l_neighbor_id") in (None, "None") else [str(road_lane["l_neighbor_id"])],
            "right_lane_ids": [] if road_lane.get("r_neighbor_id") in (None, "None") else [str(road_lane["r_neighbor_id"])],
        }
        road_lane_info["boundary_info"] = {
            "left_boundary_id": left_id,
            "right_boundary_id": right_id,
            "left_boundary_polyline": left_road_line["polyline"],
            "right_boundary_polyline": right_road_line["polyline"],
        }
        ret[str(road_lane_id)] = road_lane_info

    for stop_line_id, stop_line in map_feature_data.get("STOPLINE", {}).items():
        polyline = point_strings_to_xyz(stop_line.get("centerline", []))
        if is_near(polyline, scenario_center):
            ret[str(stop_line_id)] = {
                "type": "OTHER",
                "subtype": "STOP_LINE",
                "polyline": polyline,
                "phase": "CONVERTED",
            }

    for crosswalk_id, crosswalk in map_feature_data.get("CROSSWALK", {}).items():
        polygon = point_strings_to_xyz(crosswalk.get("polygon", []))
        if is_near(polygon, scenario_center):
            ret[str(crosswalk_id)] = {
                "type": "OTHER",
                "subtype": "CROSSWALK",
                "polygon": polygon,
                "phase": "CONVERTED",
            }

    return ret


def extract_numbers(text):
    return re.findall(r"\d+", str(text))


def get_object_type_and_subtype(object_type, object_subtype):
    if pd.isna(object_subtype):
        object_subtype = ""
    object_type = str(object_type).upper()
    object_subtype = str(object_subtype).upper()

    if object_type == "VEHICLE" and object_subtype in {"", "CAR"}:
        return "VEHICLE", "CAR"
    if object_type == "VEHICLE" and object_subtype in {"TRUCK", "VAN", "BUS", "UNKNOWN"}:
        return "VEHICLE", object_subtype
    if object_type == "BICYCLE" and object_subtype in {"CYCLIST", "TRICYCLIST", "MOTORCYCLIST", "UNKNOWN"}:
        return "CYCLIST", {"CYCLIST": "BICYCLIST"}.get(object_subtype, object_subtype)
    if object_type == "PEDESTRIAN" and object_subtype in {"PEDESTRIAN", "UNKNOWN"}:
        return "PEDESTRIAN", "PEDESTRIAN"
    return "UNKNOWN", "UNKNOWN"


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(value) else value


def extract_object_tracks(tracks, scenario_length, timestamps):
    ret = {}
    av_tracks = tracks[tracks["tag"] == "AV"]
    sdc_id = str(av_tracks.reset_index(drop=True).at[0, "id"]) if not av_tracks.empty else None
    timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps)}

    for object_id, object_track in tracks.groupby("id", sort=False):
        object_type = object_track["type"].iloc[0]
        object_subtype = object_track["sub_type"].iloc[0]
        object_state = object_state_template(scenario_length)
        object_state["type"], object_state["subtype"] = get_object_type_and_subtype(object_type, object_subtype)
        object_state["phase"] = "CONVERTED"
        for row in object_track.itertuples(index=False):
            frame = timestamp_index[row.timestamp]
            heading = safe_float(row.theta)
            object_state["state"]["global_center"][frame] = (safe_float(row.x), safe_float(row.y), safe_float(getattr(row, "z", 0.0)))
            object_state["state"]["size"][frame] = (
                4.76 if str(object_id) == sdc_id else safe_float(row.length),
                1.87 if str(object_id) == sdc_id else safe_float(row.width),
                1.65 if str(object_id) == sdc_id else safe_float(row.height),
            )
            object_state["state"]["heading"][frame] = heading
            object_state["state"]["local_velocity"][frame] = global_to_local_velocity(safe_float(row.v_x), safe_float(row.v_y), heading)
            object_state["state"]["valid"][frame] = True
        ret[str(object_id)] = object_state
    return ret, sdc_id


def extract_dynamic_map_states(dynamic_map_state_data, scenario_length, timestamps):
    ret = {}
    for lane_id, traffic_light in dynamic_map_state_data.groupby("lane_id", sort=False):
        traffic_light = traffic_light.reset_index(drop=True)
        dynamic_map_state = traffic_light_state_template(lane_id, scenario_length)
        dynamic_map_state["id"] = traffic_light.at[0, "lane_id"]
        dynamic_map_state["stop_point"][0] = traffic_light.at[0, "x"]
        dynamic_map_state["stop_point"][1] = traffic_light.at[0, "y"]
        dynamic_map_state["type"], dynamic_map_state["subtype"] = "TRAFFIC_LIGHT", "UNKNOWN"
        for frame, timestamp in enumerate(timestamps):
            state = traffic_light[traffic_light["timestamp"] == timestamp]
            if not state.empty:
                dynamic_map_state["state"]["color"][frame] = state.reset_index(drop=True)["color_1"].iloc[0]
                dynamic_map_state["state"]["valid"][frame] = True
        dynamic_map_state["controlled_lane"] = None
        dynamic_map_state["phase"] = "CONVERTED"
        ret[str(lane_id)] = dynamic_map_state
    return ret


def scenario_center_from_tracks(object_tracks, sdc_id, scenario_length):
    center_id = sdc_id if sdc_id else next(iter(object_tracks.keys()))
    valid = object_tracks[center_id]["state"]["valid"]
    centers = object_tracks[center_id]["state"]["global_center"]
    if valid.any():
        return centers[np.where(valid)[0][len(np.where(valid)[0]) // 2]]
    return centers[scenario_length // 2]


def map_path_for_tracks(args, tracks: pd.DataFrame) -> Optional[Path]:
    intersection_id = tracks["intersect_id"].iloc[0]
    numbers = extract_numbers(intersection_id)
    if not numbers:
        return None
    map_index = int(numbers[0])
    if map_index < 1 or map_index > 28:
        return None
    map_path = Path(args.raw_dataset_path) / "maps" / f"yizhuang_hdmap{map_index}.json"
    return map_path if map_path.exists() else None


def city_name(value) -> str:
    value = str(value).upper()
    if value == "PEK":
        return "BEIJING"
    return value


def process_scenario(params):
    args, scenario_id, source_name, object_track_data_path, dynamic_map_state_data_path = params
    raw_object_track_data = pd.read_csv(object_track_data_path)
    timestamps = np.asarray(raw_object_track_data["timestamp"].drop_duplicates().to_numpy())
    scenario_length = len(timestamps)
    converted_object_track_data, sdc_id = extract_object_tracks(raw_object_track_data, scenario_length, timestamps)
    scenario_center = scenario_center_from_tracks(converted_object_track_data, sdc_id, scenario_length)

    if args.skip_maps:
        converted_map_feature_data = {}
    else:
        map_path = map_path_for_tracks(args, raw_object_track_data)
        if map_path is None:
            converted_map_feature_data = {}
        else:
            with open(map_path, "r", encoding="utf-8") as file:
                converted_map_feature_data = extract_map_features(json.load(file), scenario_center)

    if args.skip_dynamic_map_states or source_name != "cooperative" or dynamic_map_state_data_path is None:
        converted_dynamic_map_state_data = {}
    else:
        raw_dynamic_map_state_data = pd.read_csv(dynamic_map_state_data_path)
        converted_dynamic_map_state_data = extract_dynamic_map_states(raw_dynamic_map_state_data, scenario_length, timestamps)

    scenario = {
        "object_tracks": converted_object_track_data,
        "map_features": converted_map_feature_data,
        "dynamic_map_states": converted_dynamic_map_state_data,
        "extra_information": {
            "id": scenario_id,
            "dataset": "V2X-Seq",
            "version": args.raw_dataset_version,
            "country": "CHINA",
            "city": city_name(raw_object_track_data["city"].iloc[0]),
            "location": None,
            "sdc_id": sdc_id,
            "scene_length": scenario_length,
            "sampling_rate": 10,
            "timestamp": timestamps - timestamps[0],
            "init_time": timestamps[0],
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
