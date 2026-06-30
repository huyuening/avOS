try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import json
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scenario_convertor.utils import (
        add_common_converter_args,
        global_to_local_velocity,
        get_road_lane_polyline,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )
except ModuleNotFoundError:
    from utils import (
        add_common_converter_args,
        global_to_local_velocity,
        get_road_lane_polyline,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )


SPLIT_DIRS = {
    "mini": ("mini",),
    "train": ("train",),
    "val": ("val", "validation"),
    "test": ("test",),
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Argoverse 2 Motion Forecasting Dataset Converter")
    add_common_converter_args(
        parser,
        default_raw_dataset_path="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/argoverse-2-motion-dataset",
        default_subset_name="argoverse-2",
        default_raw_dataset_version="2.0",
        raw_dataset_help="The file directory for A2MD",
    )
    parser.add_argument("--split", choices=("auto", "mini", "train", "val", "test"), default="auto", help="Split to convert.")
    parser.add_argument("--max_files", default=None, type=int, help="Convert at most this many scenario directories.")
    parser.add_argument("--skip_maps", action="store_true", help="Skip log_map_archive_*.json parsing.")
    return parser


def discover_split_dirs(raw_dataset_path: Path, split: str):
    split_names = [split] if split != "auto" else ("mini", "train", "val", "test")
    ret = []
    for split_name in split_names:
        for relative_path in SPLIT_DIRS[split_name]:
            split_dir = raw_dataset_path / relative_path
            if split_dir.exists():
                ret.append((split_name, split_dir))
                break
    return ret


def scenario_sort_key(path: Path):
    return path.name


def find_scenario_parquet(scenario_dir: Path) -> Path:
    parquet_files = sorted(scenario_dir.glob("scenario_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Cannot find scenario_*.parquet in {scenario_dir}")
    return parquet_files[0]


def find_map_json(scenario_dir: Path, scenario_id: str) -> Optional[Path]:
    preferred = scenario_dir / f"log_map_archive_{scenario_id}.json"
    if preferred.exists():
        return preferred
    map_files = sorted(scenario_dir.glob("log_map_archive_*.json"))
    return map_files[0] if map_files else None


def list_scenarios(args):
    raw_dataset_path = Path(args.raw_dataset_path)
    split_dirs = discover_split_dirs(raw_dataset_path, args.split)
    if not split_dirs:
        expected = [str(raw_dataset_path / path) for paths in SPLIT_DIRS.values() for path in paths]
        raise FileNotFoundError(f"Cannot find Argoverse 2 split directory. Expected one of: {expected}")

    scenarios = []
    for split_name, split_dir in split_dirs:
        for scenario_dir in sorted((path for path in split_dir.iterdir() if path.is_dir()), key=scenario_sort_key):
            parquet_path = find_scenario_parquet(scenario_dir)
            scenario_id = scenario_dir.name
            map_path = find_map_json(scenario_dir, scenario_id)
            scenarios.append((args, scenario_id, str(parquet_path), str(map_path) if map_path else None))
            if args.max_files is not None and len(scenarios) >= args.max_files:
                return scenarios
    return scenarios


def get_object_type(object_type):
    object_type = str(object_type).upper()
    if object_type == "VEHICLE":
        return "VEHICLE", "CAR"
    if object_type == "BUS":
        return "VEHICLE", "BUS"
    if object_type in {"CYCLIST", "RIDERLESS_BICYCLE"}:
        return "CYCLIST", "BICYCLIST"
    if object_type == "MOTORCYCLIST":
        return "CYCLIST", "MOTORCYCLIST"
    if object_type == "PEDESTRIAN":
        return "PEDESTRIAN", "PEDESTRIAN"
    if object_type in {"STATIC", "BACKGROUND"}:
        return "STATIC", "UNKNOWN"
    if object_type == "CONSTRUCTION":
        return "STATIC", "CONSTRUCTION"
    if object_type == "UNKNOWN":
        return "UNKNOWN", "UNKNOWN"
    return "UNKNOWN", object_type


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(value) else value


def extract_object_tracks(raw_object_track_data: pd.DataFrame, scenario_length: int):
    ret = {}
    for object_id, object_track in raw_object_track_data.groupby("track_id", sort=False):
        object_states = object_state_template(scenario_length)
        object_states["type"], object_states["subtype"] = get_object_type(object_track.iloc[0]["object_type"])
        object_states["phase"] = "CONVERTED"

        for row in object_track.itertuples(index=False):
            frame = int(row.timestep)
            if frame < 0 or frame >= scenario_length:
                continue
            heading = safe_float(row.heading)
            velocity_x = safe_float(row.velocity_x)
            velocity_y = safe_float(row.velocity_y)
            object_states["state"]["global_center"][frame] = (safe_float(row.position_x), safe_float(row.position_y), 0.0)
            object_states["state"]["heading"][frame] = heading
            object_states["state"]["local_velocity"][frame] = global_to_local_velocity(velocity_x, velocity_y, heading)
            object_states["state"]["valid"][frame] = True
        ret[str(object_id)] = object_states
    return ret


def point_list_to_xyz(points) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([[safe_float(point.get("x")), safe_float(point.get("y")), safe_float(point.get("z"))] for point in points], dtype=np.float64)


def get_road_line_type(road_line_type):
    road_line_type = str(road_line_type).upper()
    mapping = {
        "DASH_SOLID_YELLOW": ("ROAD_LINE", "DASH_SOLID_YELLOW"),
        "DASH_SOLID_WHITE": ("ROAD_LINE", "DASH_SOLID_WHITE"),
        "DASHED_WHITE": ("ROAD_LINE", "DASHED_WHITE"),
        "DASHED_YELLOW": ("ROAD_LINE", "DASHED_YELLOW"),
        "DOUBLE_SOLID_YELLOW": ("ROAD_LINE", "DOUBLE_SOLID_YELLOW"),
        "DOUBLE_SOLID_WHITE": ("ROAD_LINE", "DOUBLE_SOLID_WHITE"),
        "DOUBLE_DASH_YELLOW": ("ROAD_LINE", "DOUBLE_DASH_YELLOW"),
        "DOUBLE_DASH_WHITE": ("ROAD_LINE", "DOUBLE_DASH_WHITE"),
        "SOLID_YELLOW": ("ROAD_LINE", "SOLID_YELLOW"),
        "SOLID_WHITE": ("ROAD_LINE", "SOLID_WHITE"),
        "SOLID_DASH_WHITE": ("ROAD_LINE", "SOLID_DASH_WHITE"),
        "SOLID_DASH_YELLOW": ("ROAD_LINE", "SOLID_DASH_YELLOW"),
        "SOLID_BLUE": ("ROAD_LINE", "SOLID_BLUE"),
        "NONE": ("ROAD_LINE", "VIRTUAL"),
        "UNKNOWN": ("ROAD_LINE", "UNKNOWN"),
    }
    return mapping.get(road_line_type, ("ROAD_LINE", road_line_type or "UNKNOWN"))


def get_road_lane_type(road_lane_type):
    road_lane_type = str(road_lane_type).upper()
    if road_lane_type == "VEHICLE":
        return "ROAD_LANE", "URBAN_VEHICLE"
    if road_lane_type == "BUS":
        return "ROAD_LANE", "URBAN_BUS"
    if road_lane_type == "BIKE":
        return "ROAD_LANE", "URBAN_BIKE"
    return "ROAD_LANE", "URBAN_UNKNOWN"


def extract_road_line(polyline: np.ndarray, mark_type: str):
    road_line_info = {}
    road_line_info["type"], road_line_info["subtype"] = get_road_line_type(mark_type)
    road_line_info["polyline"] = polyline
    road_line_info["phase"] = "CONVERTED"
    return road_line_info


def extract_lane_segment(lane_segment: dict):
    road_lane_id = str(lane_segment["id"])
    left_road_line_id = f"{road_lane_id}_1"
    right_road_line_id = f"{road_lane_id}_2"
    left_polyline = point_list_to_xyz(lane_segment.get("left_lane_boundary"))
    right_polyline = point_list_to_xyz(lane_segment.get("right_lane_boundary"))
    centerline = point_list_to_xyz(lane_segment.get("centerline"))
    if len(centerline) == 0 and len(left_polyline) and len(right_polyline):
        centerline = get_road_lane_polyline(left_polyline, right_polyline)

    road_lane_info = {}
    road_lane_info["type"], road_lane_info["subtype"] = get_road_lane_type(lane_segment.get("lane_type"))
    road_lane_info["polyline"] = centerline
    road_lane_info["speed_limit"] = None
    road_lane_info["is_intersection"] = lane_segment.get("is_intersection")
    road_lane_info["phase"] = "CONVERTED"
    road_lane_info["lane_info"] = {
        "entry_lane_ids": [str(item) for item in lane_segment.get("predecessors") or []],
        "exit_lane_ids": [str(item) for item in lane_segment.get("successors") or []],
        "left_lane_ids": [str(lane_segment["left_neighbor_id"])] if lane_segment.get("left_neighbor_id") else [],
        "right_lane_ids": [str(lane_segment["right_neighbor_id"])] if lane_segment.get("right_neighbor_id") else [],
    }
    road_lane_info["boundary_info"] = {
        "left_boundary_id": left_road_line_id,
        "right_boundary_id": right_road_line_id,
        "left_boundary_polyline": left_polyline,
        "right_boundary_polyline": right_polyline,
    }

    return {
        left_road_line_id: extract_road_line(left_polyline, lane_segment.get("left_lane_mark_type", "UNKNOWN")),
        right_road_line_id: extract_road_line(right_polyline, lane_segment.get("right_lane_mark_type", "UNKNOWN")),
        road_lane_id: road_lane_info,
    }


def extract_pedestrian_crossing(crossing: dict):
    edge1 = point_list_to_xyz(crossing.get("edge1"))
    edge2 = point_list_to_xyz(crossing.get("edge2"))
    if len(edge1) and len(edge2):
        polygon = np.vstack((edge1, edge2[::-1]))
    else:
        polygon = np.vstack([points for points in (edge1, edge2) if len(points)]) if len(edge1) or len(edge2) else np.zeros((0, 3), dtype=np.float64)
    return {
        "type": "OTHER",
        "subtype": "CROSSWALK",
        "polygon": polygon,
        "phase": "CONVERTED",
    }


def extract_map_features(map_data: dict):
    ret = {}

    for lane_segment in map_data.get("lane_segments", {}).values():
        ret.update(extract_lane_segment(lane_segment))

    for road_edge_id, road_edge in map_data.get("drivable_areas", {}).items():
        ret[str(road_edge.get("id", road_edge_id))] = {
            "type": "ROAD_EDGE",
            "subtype": "UNKNOWN",
            "polyline": point_list_to_xyz(road_edge.get("area_boundary")),
            "phase": "CONVERTED",
        }

    for crosswalk_id, crosswalk in map_data.get("pedestrian_crossings", {}).items():
        ret[str(crosswalk.get("id", crosswalk_id))] = extract_pedestrian_crossing(crosswalk)

    return ret


def get_city_name(city_name):
    city_name = str(city_name).lower()
    mapping = {
        "palo-alto": "PALO_ALTO",
        "austin": "AUSTIN",
        "miami": "MIAMI",
        "dearborn": "DEARBORN",
        "washington-dc": "WASHINGTON_DC",
        "pittsburgh": "PITTSBURGH",
    }
    return mapping.get(city_name, city_name.upper().replace("-", "_"))


def timestamps_seconds(raw_object_track_data: pd.DataFrame, scenario_length: int) -> np.ndarray:
    start = safe_float(raw_object_track_data["start_timestamp"].iloc[0])
    end = safe_float(raw_object_track_data["end_timestamp"].iloc[0])
    if start and end and scenario_length > 1:
        return np.linspace(start, end, scenario_length, dtype=np.float64) / 1_000_000_000
    return np.arange(scenario_length, dtype=np.float64) / 10


def process_scenario(params):
    args, scenario_id, parquet_path, map_path = params
    raw_object_track_data = pd.read_parquet(parquet_path, engine="pyarrow")
    scenario_length = int(raw_object_track_data["num_timestamps"].iloc[0])
    converted_object_track_data = extract_object_tracks(raw_object_track_data, scenario_length)

    if args.skip_maps or map_path is None:
        converted_map_feature_data = {}
    else:
        with open(map_path, "r", encoding="utf-8") as file:
            converted_map_feature_data = extract_map_features(json.load(file))

    focal_track_id = str(raw_object_track_data["focal_track_id"].iloc[0])
    city_name = raw_object_track_data["city"].iloc[0]
    timestamps = timestamps_seconds(raw_object_track_data, scenario_length)

    scenario = {
        "object_tracks": converted_object_track_data,
        "map_features": converted_map_feature_data,
        "dynamic_map_states": {},
        "extra_information": {
            "id": scenario_id,
            "dataset": "Argoverse 2",
            "version": args.raw_dataset_version,
            "country": "AMERICA",
            "city": get_city_name(city_name),
            "location": None,
            "sdc_id": focal_track_id if focal_track_id in converted_object_track_data else None,
            "focal_track_id": focal_track_id,
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
