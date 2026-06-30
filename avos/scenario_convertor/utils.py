import argparse
import multiprocessing
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


DEFAULT_CONVERTED_DATASET_PATH = "/dataset/x-scene/converted-dataset"


def add_common_converter_args(
        parser: argparse.ArgumentParser,
        *,
        default_raw_dataset_path: str,
        default_subset_name: str,
        default_raw_dataset_version: str,
        raw_dataset_help: str,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--raw_dataset_path",
        "--raw_data_path",
        dest="raw_dataset_path",
        default=default_raw_dataset_path,
        type=str,
        help=raw_dataset_help,
    )
    parser.add_argument(
        "--converted_dataset_path",
        default=DEFAULT_CONVERTED_DATASET_PATH,
        type=str,
        help="The file directory for saving the converted dataset",
    )
    parser.add_argument(
        "--subset_name",
        default=default_subset_name,
        type=str,
        help="The output subset directory name",
    )
    parser.add_argument(
        "--raw_dataset_version",
        default=default_raw_dataset_version,
        type=str,
        help="The raw dataset version",
    )
    parser.add_argument(
        "--num_workers",
        default=None,
        type=int,
        help="Number of multiprocessing workers. Defaults to multiprocessing.Pool's default.",
    )
    return parser


def output_dir(args: argparse.Namespace) -> Path:
    return Path(args.converted_dataset_path) / args.subset_name


def canonicalize_scenario(scenario: dict) -> dict:
    scenario.setdefault("object_tracks", {})
    scenario.setdefault("map_features", {})
    scenario.setdefault("dynamic_map_states", {})
    scenario["object_tracks"] = {str(key): value for key, value in scenario["object_tracks"].items()}
    scenario["map_features"] = {str(key): value for key, value in scenario["map_features"].items()}
    scenario["dynamic_map_states"] = {str(key): value for key, value in scenario["dynamic_map_states"].items()}
    extra_information = scenario.setdefault("extra_information", {})
    extra_information.setdefault("init_time", None)

    for map_feature in scenario["map_features"].values():
        if not isinstance(map_feature, dict) or map_feature.get("type") != "ROAD_LANE":
            continue
        lane_info = map_feature.setdefault("lane_info", {})
        for key in ("entry_lane_ids", "exit_lane_ids", "left_lane_ids", "right_lane_ids"):
            value = lane_info.get(key, [])
            if value is None:
                lane_info[key] = []
            elif isinstance(value, dict):
                lane_info[key] = list(value.values()) if value else []
            elif isinstance(value, (list, tuple)):
                lane_info[key] = list(value)
            else:
                lane_info[key] = [value]

    return scenario


def scenario_pickle_name(scenario: dict, scenario_id: Optional[str] = None) -> str:
    extra_information = scenario.setdefault("extra_information", {})
    if "id" not in extra_information and scenario_id is None:
        raise KeyError("scenario['extra_information']['id'] is required for pickle naming.")

    canonical_scenario_id = str(extra_information.get("id", scenario_id))
    extra_information["id"] = canonical_scenario_id

    if scenario_id is not None and str(scenario_id) != canonical_scenario_id:
        raise ValueError(
            f"Pickle name must match scenario id: got {scenario_id!r}, "
            f"expected {canonical_scenario_id!r}."
        )
    if "/" in canonical_scenario_id or "\\" in canonical_scenario_id:
        raise ValueError(f"Scenario id cannot contain path separators: {canonical_scenario_id!r}.")

    return f"{canonical_scenario_id}.pkl"


def save_scenario_pickle(scenario: dict, args: argparse.Namespace, scenario_id: Optional[str] = None) -> Path:
    scenario = canonicalize_scenario(scenario)
    pickle_name = scenario_pickle_name(scenario, scenario_id)

    save_dir = output_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    pickle_file_path = save_dir / pickle_name
    with open(pickle_file_path, "wb") as file:
        pickle.dump(scenario, file)
    return pickle_file_path


def pool_kwargs(args: argparse.Namespace) -> dict:
    return {"processes": args.num_workers} if getattr(args, "num_workers", None) else {}


def run_tasks(worker, params, total: int, args: argparse.Namespace, tqdm_cls, *, initializer=None, initargs=(), pool_options=None):
    if getattr(args, "num_workers", None) == 1:
        if initializer is not None:
            initializer(*initargs)
        list(tqdm_cls(map(worker, params), total=total))
        return

    options = pool_kwargs(args)
    if pool_options:
        options.update(pool_options)
    with multiprocessing.Pool(initializer=initializer, initargs=initargs, **options) as pool:
        list(tqdm_cls(pool.imap(worker, params), total=total))


def interpolate(xyz: np.ndarray, step: float, method: int) -> np.ndarray:
    """
    Interpolate points based on cumulative distances from the first one. Two modes are available:
    INTER_METER: interpolate using step as a meter value over cumulative distances (variable len result)
    INTER_ENSURE_LEN: interpolate using a variable step such that we always get step values
    Args:
        xyz (np.ndarray): XYZ coords
        step (float): param for the interpolation
        method (InterpolationMethod): method to use to interpolate

    Returns:
        np.ndarray: the new interpolated coordinates
    """
    cum_dist = np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=-1))
    cum_dist = np.insert(cum_dist, 0, 0)

    if method == 1:
        step = int(step)
        assert step > 1, "step must be at least 2 with INTER_ENSURE_LEN"
        steps = np.linspace(cum_dist[0], cum_dist[-1], step)

    elif method == 0:
        assert step > 0, "step must be greater than 0 with INTER_FIXED"
        steps = np.arange(cum_dist[0], cum_dist[-1], step)
    else:
        raise NotImplementedError(f"interpolation method should be in 0 or 1")

    xyz_inter = np.empty((len(steps), 3), dtype=xyz.dtype)
    xyz_inter[:, 0] = np.interp(steps, xp=cum_dist, fp=xyz[:, 0])
    xyz_inter[:, 1] = np.interp(steps, xp=cum_dist, fp=xyz[:, 1])
    xyz_inter[:, 2] = np.interp(steps, xp=cum_dist, fp=xyz[:, 2])
    return xyz_inter


def object_state_template(scenario_length_):
    return dict(
        type=None,
        subtype=None,
        state=dict(
            global_center=np.zeros(shape=[scenario_length_, 3], dtype=np.float64),
            local_velocity=np.zeros(shape=[scenario_length_, 2], dtype=np.float64),
            local_acceleration=np.zeros(shape=[scenario_length_, 2], dtype=np.float64),
            heading=np.zeros(shape=scenario_length_, dtype=np.float64),
            size=np.zeros(shape=[scenario_length_, 3], dtype=np.float64),
            valid=np.zeros(shape=scenario_length_, dtype=bool)
        ),
        phase='RAW'
    )


def connect_arrays(array1, array2):
    if len(array1) < 2 or len(array2) < 2:
        return np.concatenate((array1, array2))

    combined_polygon = Polygon(np.concatenate((array1, array2)))

    if combined_polygon.is_valid:
        return np.concatenate((array1, array2))
    else:
        array2_reversed = array2[::-1]
        combined_polygon_reversed = Polygon(np.concatenate((array1, array2_reversed)))
        if combined_polygon_reversed.is_valid:
            return np.concatenate((array1, array2_reversed))
        else:
            raise ValueError("Cannot connect arrays without creating a self-intersecting polygon.")


def traffic_light_state_template(traffic_light_id_, scenario_length_):
    return dict(
        id=traffic_light_id_,
        type=None,
        subtype=None,
        stop_point=np.zeros(shape=[3, ], dtype=np.float64),
        state=dict(
            color=[None]*scenario_length_,
            valid=np.zeros(shape=scenario_length_, dtype=bool)
        ),
        phase='RAW'
    )


def get_multipolygon(ret, relation):
    multipolygon = np.empty((0, 3), float)
    for member in relation.findall('member'):
        ref_id = member.attrib['ref']
        ref_polyline = ret[ref_id]['polyline']
        if len(ref_polyline) != 0:
            multipolygon = np.concatenate((multipolygon, ref_polyline), axis=0)
    multipolygon = multipolygon[ConvexHull(multipolygon[:, :2]).vertices]

    return multipolygon


def get_road_lane_polyline(left_road_line_edge_polyline, right_road_line_edge_polyline):
    if len(left_road_line_edge_polyline) < len(right_road_line_edge_polyline):
        left_road_line_edge_polyline = interpolate(left_road_line_edge_polyline, len(right_road_line_edge_polyline), 1)
    elif len(right_road_line_edge_polyline) < len(left_road_line_edge_polyline):
        right_road_line_edge_polyline = interpolate(right_road_line_edge_polyline, len(left_road_line_edge_polyline), 1)

    road_lane_polyline = (left_road_line_edge_polyline + right_road_line_edge_polyline) / 2

    return road_lane_polyline


def global_to_local_velocity(vx, vy, heading):
    c, s = np.cos(heading), np.sin(heading)
    return vx * c + vy * s, -vx * s + vy * c
