# nuPlan Dataset
#
# This converter uses a direct SQLite/GPKG backend inspired by parse_nuplan.py.
# It avoids nuplan-devkit, Hydra, and Torch at runtime, which makes the default
# path easier to run on Windows, macOS, and Linux.

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import bisect
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm
from utils import (
    add_common_converter_args,
    global_to_local_velocity,
    object_state_template,
    run_tasks,
    save_scenario_pickle,
    traffic_light_state_template,
)


DIRECT_MAP_DEPENDENCY_HINT = (
    "nuPlan map parsing requires optional dependencies. Install them with "
    "`pip install -r requirements.txt`."
)

ROADLINE_MAPPING = {
    0: "DASHED",
    1: "VIRTUAL",
    2: "SOLID",
    3: "VIRTUAL",
}

LANE_TYPE_MAPPING = {
    0: "ROAD",
    1: "BIKE_LANE",
}

STOP_POLYGON_MAPPING = {
    0: "CROSSWALK_STOP_LINE",
    1: "STOP_SIGN",
    2: "TRAFFIC_LIGHT_STOP_LINE",
    3: "TURN_STOP",
    4: "YIELD",
}


@dataclass(frozen=True)
class LidarFrame:
    token: bytes
    token_id: str
    ego_pose_token: Optional[bytes]
    ego_pose_token_id: Optional[str]
    timestamp_us: int


@dataclass
class MapContext:
    location: str
    boundaries: object
    baseline_paths: object
    lanes: object
    lane_connectors: object
    connector_polygons: object
    areas: Dict[str, object]
    stop_polygons: object
    traffic_lights: object
    boundary_geometries: Dict[str, object]
    centerlines_by_lane_id: Dict[str, np.ndarray]
    centerlines_by_connector_id: Dict[str, np.ndarray]
    lane_relationships: Dict[str, dict]
    connector_relationships: Dict[str, dict]
    traffic_light_stop_points: Dict[str, np.ndarray]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="nuPlan Dataset Converter")
    add_common_converter_args(
        parser,
        default_raw_dataset_path="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/nuplan-motion-dataset-v1.1",
        default_subset_name="nuplan",
        default_raw_dataset_version="1.1",
        raw_dataset_help="The file directory for NPMD",
    )
    parser.add_argument(
        "--split",
        choices=("mini", "trainval", "test"),
        default="mini",
        help="The nuPlan split to convert.",
    )
    parser.add_argument(
        "--scenario_duration_s",
        default=20.0,
        type=float,
        help="Direct parser scenario window length in seconds.",
    )
    parser.add_argument(
        "--scenario_stride_s",
        default=20.0,
        type=float,
        help="Direct parser scenario window stride in seconds.",
    )
    parser.add_argument(
        "--sample_interval",
        default=2,
        type=int,
        help="Use every Nth lidar_pc frame. The default converts 20 Hz logs to about 10 Hz.",
    )
    parser.add_argument(
        "--map_radius",
        default=150.0,
        type=float,
        help="Map query radius around the ego center in meters.",
    )
    parser.add_argument(
        "--max_files",
        default=None,
        type=int,
        help="Convert at most this many nuPlan database files.",
    )
    parser.add_argument(
        "--max_scenarios_per_file",
        default=None,
        type=int,
        help="Convert at most this many scenario windows from each database file.",
    )
    parser.add_argument(
        "--skip_maps",
        action="store_true",
        help="Do not read map.gpkg files. Useful for dependency checks.",
    )
    parser.add_argument(
        "--skip_dynamic_map_states",
        action="store_true",
        help="Do not read traffic_light_status rows.",
    )
    parser.add_argument("--ego_length", default=4.9, type=float, help="Fallback ego vehicle length.")
    parser.add_argument("--ego_width", default=2.0, type=float, help="Fallback ego vehicle width.")
    parser.add_argument("--ego_height", default=1.8, type=float, help="Fallback ego vehicle height.")
    return parser


def token_to_str(token) -> str:
    if token is None:
        return ""
    if isinstance(token, memoryview):
        token = token.tobytes()
    if isinstance(token, bytes):
        return token.hex()
    return str(token)


def safe_int(value, default=0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def split_ids(value) -> list:
    if value is None:
        return []
    try:
        if bool(np.isnan(value)):
            return []
    except TypeError:
        pass
    return [item.strip() for item in str(value).split(",") if item.strip()]


def prefixed(prefix: str, value) -> str:
    return f"{prefix}_{safe_int(value, value)}"


def coords_to_xyz(geometry) -> np.ndarray:
    if geometry is None or geometry.is_empty:
        return np.zeros((0, 3), dtype=np.float64)
    if hasattr(geometry, "exterior"):
        coords = np.asarray(geometry.exterior.coords, dtype=np.float64)
    else:
        coords = np.asarray(geometry.coords, dtype=np.float64)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if coords.shape[1] == 2:
        return np.hstack((coords, np.zeros((coords.shape[0], 1), dtype=np.float64)))
    return coords[:, :3]


def row_value(row, key, default=None):
    try:
        value = row[key]
    except (KeyError, TypeError):
        return default
    if value is None:
        return default
    try:
        if bool(np.isnan(value)):
            return default
    except TypeError:
        pass
    return value


def table_columns(connection: sqlite3.Connection, table_name: str) -> set:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def placeholders(values: Sequence[object]) -> str:
    return ",".join("?" for _ in values)


def query_by_tokens(
    connection: sqlite3.Connection,
    query_prefix: str,
    token_column: str,
    tokens: Sequence[bytes],
) -> Iterable[sqlite3.Row]:
    if not tokens:
        return []
    query = f"{query_prefix} WHERE {token_column} IN ({placeholders(tokens)})"
    return connection.execute(query, tokens).fetchall()


def resolve_split_dir(raw_dataset_path: str, split: str) -> Path:
    root = Path(raw_dataset_path)
    candidates = [
        root / "nuplan" / "dataset" / "nuplan-v1.1" / "splits" / split,
        root / "dataset" / "nuplan-v1.1" / "splits" / split,
        root / "nuplan-v1.1" / "splits" / split,
        root / "splits" / split,
        root / split,
        root,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.db")):
            return candidate
    raise FileNotFoundError(
        f"Cannot find nuPlan .db files for split '{split}' under {raw_dataset_path}."
    )


def resolve_maps_root(raw_dataset_path: str) -> Optional[Path]:
    root = Path(raw_dataset_path)
    candidates = [
        root / "nuplan" / "dataset" / "maps",
        root / "dataset" / "maps",
        root / "maps",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.gpkg")):
            return candidate
    return None


def list_db_files(args) -> List[Path]:
    split_dir = resolve_split_dir(args.raw_dataset_path, args.split)
    db_files = sorted(split_dir.rglob("*.db"))
    if args.max_files is not None:
        db_files = db_files[: args.max_files]
    if not db_files:
        raise FileNotFoundError(f"No nuPlan .db files found in {split_dir}.")
    return db_files


def find_map_file(maps_root: Optional[Path], location: str) -> Optional[Path]:
    if maps_root is None:
        return None
    map_files = sorted(maps_root.rglob("*.gpkg"))
    if not map_files:
        return None

    location = str(location)
    preferred = [
        path
        for path in map_files
        if location in {part for part in path.parts} or location in path.stem
    ]
    if preferred:
        map_named = [path for path in preferred if path.name == "map.gpkg"]
        return map_named[0] if map_named else preferred[0]

    return map_files[0] if len(map_files) == 1 else None


def get_location(connection: sqlite3.Connection, db_file: Path) -> str:
    if not table_exists(connection, "log"):
        return db_file.stem
    columns = table_columns(connection, "log")
    if "location" not in columns:
        return db_file.stem
    row = connection.execute("SELECT location FROM log LIMIT 1").fetchone()
    return row["location"] if row and row["location"] else db_file.stem


def get_country_name(map_name: str) -> Optional[str]:
    if map_name in {"us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood", "us-ma-boston"}:
        return "AMERICA"
    if map_name == "sg-one-north":
        return "SINGAPORE"
    return None


def get_city_name(map_name: str) -> Optional[str]:
    city_by_map = {
        "us-nv-las-vegas-strip": "LAS_VEGAS",
        "us-pa-pittsburgh-hazelwood": "PITTSBURGH",
        "us-ma-boston": "BOSTON",
        "sg-one-north": "SINGAPORE",
    }
    return city_by_map.get(map_name)


def get_object_type(category_name: Optional[str]) -> Tuple[str, str]:
    category = (category_name or "").lower()
    if category == "vehicle":
        return "VEHICLE", "UNKNOWN"
    if category == "pedestrian":
        return "PEDESTRIAN", "PEDESTRIAN"
    if category == "bicycle":
        return "CYCLIST", "BICYCLIST"
    if category in {"traffic_cone", "barrier", "czone_sign"}:
        return "STATIC", category.upper()
    if category == "generic_object":
        return "UNKNOWN", "UNKNOWN"
    return "UNKNOWN", category.upper() if category else "UNKNOWN"


def load_lidar_frames(connection: sqlite3.Connection) -> List[LidarFrame]:
    if not table_exists(connection, "lidar_pc"):
        raise ValueError("nuPlan database is missing table lidar_pc.")
    columns = table_columns(connection, "lidar_pc")
    required = {"token", "timestamp"}
    missing = required - columns
    if missing:
        raise ValueError(f"lidar_pc table is missing columns: {sorted(missing)}")

    ego_pose_expr = "ego_pose_token" if "ego_pose_token" in columns else "NULL AS ego_pose_token"
    rows = connection.execute(
        f"SELECT token, {ego_pose_expr}, timestamp FROM lidar_pc ORDER BY timestamp"
    ).fetchall()
    frames = []
    for row in rows:
        ego_pose_token = row["ego_pose_token"]
        frames.append(
            LidarFrame(
                token=row["token"],
                token_id=token_to_str(row["token"]),
                ego_pose_token=ego_pose_token,
                ego_pose_token_id=token_to_str(ego_pose_token) if ego_pose_token else None,
                timestamp_us=int(row["timestamp"]),
            )
        )
    return frames


def build_windows(frames: List[LidarFrame], args) -> List[List[LidarFrame]]:
    if args.sample_interval < 1:
        raise ValueError("--sample_interval must be at least 1.")
    sampled_frames = frames[:: args.sample_interval]
    if not sampled_frames:
        return []

    timestamps = [frame.timestamp_us for frame in sampled_frames]
    if args.scenario_duration_s <= 0:
        windows = [sampled_frames]
    else:
        duration_us = int(args.scenario_duration_s * 1_000_000)
        stride_us = int(args.scenario_stride_s * 1_000_000)
        if duration_us <= 0 or stride_us <= 0:
            raise ValueError("--scenario_duration_s and --scenario_stride_s must be positive.")

        windows = []
        start_index = 0
        while start_index < len(sampled_frames):
            start_ts = timestamps[start_index]
            end_index = bisect.bisect_left(timestamps, start_ts + duration_us, lo=start_index)
            window = sampled_frames[start_index:end_index]
            if window:
                windows.append(window)
            next_start_ts = start_ts + stride_us
            next_index = bisect.bisect_left(timestamps, next_start_ts, lo=start_index + 1)
            if next_index <= start_index:
                next_index = start_index + 1
            start_index = next_index

    if args.max_scenarios_per_file is not None:
        windows = windows[: args.max_scenarios_per_file]
    return windows


def load_track_infos(connection: sqlite3.Connection) -> Dict[str, dict]:
    if not table_exists(connection, "track"):
        return {}
    columns = table_columns(connection, "track")
    select_columns = ["track.token AS track_token"]
    for column in ("width", "length", "height"):
        select_columns.append(f"track.{column}" if column in columns else f"NULL AS {column}")

    query = f"SELECT {', '.join(select_columns)}"
    if table_exists(connection, "category") and "category_token" in columns:
        query += ", category.name AS category_name FROM track LEFT JOIN category ON category.token = track.category_token"
    else:
        query += ", NULL AS category_name FROM track"

    infos = {}
    for row in connection.execute(query).fetchall():
        track_id = token_to_str(row["track_token"])
        infos[track_id] = {
            "category_name": row["category_name"],
            "width": row["width"],
            "length": row["length"],
            "height": row["height"],
        }
    return infos


def yaw_from_quaternion(row: sqlite3.Row) -> float:
    if "yaw" in row.keys() and row["yaw"] is not None:
        return safe_float(row["yaw"])
    qw = safe_float(row["qw"], 1.0)
    qx = safe_float(row["qx"])
    qy = safe_float(row["qy"])
    qz = safe_float(row["qz"])
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def extract_object_tracks(
    connection: sqlite3.Connection,
    frames: List[LidarFrame],
    track_infos: Dict[str, dict],
    args,
) -> dict:
    scenario_length = len(frames)
    frame_index_by_lidar_token = {frame.token_id: index for index, frame in enumerate(frames)}

    ret = {}

    if table_exists(connection, "lidar_box"):
        lidar_tokens = [frame.token for frame in frames]
        rows = query_by_tokens(
            connection,
            (
                "SELECT track_token, lidar_pc_token, x, y, z, width, length, height, "
                "yaw, vx, vy FROM lidar_box"
            ),
            "lidar_pc_token",
            lidar_tokens,
        )

        for row in rows:
            track_id = token_to_str(row["track_token"])
            frame_index = frame_index_by_lidar_token.get(token_to_str(row["lidar_pc_token"]))
            if frame_index is None:
                continue

            if track_id not in ret:
                object_states = object_state_template(scenario_length)
                category = track_infos.get(track_id, {}).get("category_name")
                object_states["type"], object_states["subtype"] = get_object_type(category)
                object_states["phase"] = "CONVERTED"
                ret[track_id] = object_states

            object_states = ret[track_id]
            heading = safe_float(row["yaw"])
            vx = safe_float(row["vx"])
            vy = safe_float(row["vy"])
            local_vx, local_vy = global_to_local_velocity(vx, vy, heading)
            object_states["state"]["global_center"][frame_index] = (
                safe_float(row["x"]),
                safe_float(row["y"]),
                safe_float(row["z"]),
            )
            object_states["state"]["heading"][frame_index] = heading
            object_states["state"]["local_velocity"][frame_index] = (local_vx, local_vy)
            object_states["state"]["size"][frame_index] = (
                safe_float(row["length"], safe_float(track_infos.get(track_id, {}).get("length"))),
                safe_float(row["width"], safe_float(track_infos.get(track_id, {}).get("width"))),
                safe_float(row["height"], safe_float(track_infos.get(track_id, {}).get("height"))),
            )
            object_states["state"]["valid"][frame_index] = True

    ret["ego"] = extract_ego_track(connection, frames, args)
    return ret


def extract_ego_track(connection: sqlite3.Connection, frames: List[LidarFrame], args) -> dict:
    scenario_length = len(frames)
    ego_states = object_state_template(scenario_length)
    ego_states["type"], ego_states["subtype"] = "VEHICLE", "CAR"
    ego_states["phase"] = "CONVERTED"

    if not table_exists(connection, "ego_pose"):
        return ego_states

    ego_tokens = [frame.ego_pose_token for frame in frames if frame.ego_pose_token is not None]
    if not ego_tokens:
        return ego_states

    columns = table_columns(connection, "ego_pose")
    selected_columns = ["token"]
    for column in ("x", "y", "z", "qw", "qx", "qy", "qz", "yaw", "vx", "vy"):
        if column in columns:
            selected_columns.append(column)
        else:
            selected_columns.append(f"NULL AS {column}")

    rows = query_by_tokens(
        connection,
        f"SELECT {', '.join(selected_columns)} FROM ego_pose",
        "token",
        ego_tokens,
    )
    row_by_token = {token_to_str(row["token"]): row for row in rows}

    for index, frame in enumerate(frames):
        if frame.ego_pose_token_id is None:
            continue
        row = row_by_token.get(frame.ego_pose_token_id)
        if row is None:
            continue
        heading = yaw_from_quaternion(row)
        vx = safe_float(row["vx"])
        vy = safe_float(row["vy"])
        local_vx, local_vy = global_to_local_velocity(vx, vy, heading)
        ego_states["state"]["global_center"][index] = (
            safe_float(row["x"]),
            safe_float(row["y"]),
            safe_float(row["z"]),
        )
        ego_states["state"]["heading"][index] = heading
        ego_states["state"]["local_velocity"][index] = (local_vx, local_vy)
        ego_states["state"]["size"][index] = (args.ego_length, args.ego_width, args.ego_height)
        ego_states["state"]["valid"][index] = True

    return ego_states


def import_map_dependencies():
    try:
        import pyogrio
    except ImportError as exc:
        raise ImportError(DIRECT_MAP_DEPENDENCY_HINT) from exc
    return pyogrio


def read_map_layer(pyogrio, file_path: Path, layer_name: str, projection_system=None, fid_as_index=False):
    try:
        frame = pyogrio.read_dataframe(file_path, layer=layer_name, fid_as_index=fid_as_index)
    except Exception:
        return None
    if frame is not None and projection_system and not frame.empty and hasattr(frame, "to_crs"):
        frame = frame.to_crs(projection_system)
    return frame


def load_projection_system(pyogrio, file_path: Path) -> Optional[str]:
    meta = read_map_layer(pyogrio, file_path, "meta")
    if meta is None or meta.empty or "key" not in meta.columns or "value" not in meta.columns:
        return None
    matches = meta[meta["key"] == "projectedCoordSystem"]
    if matches.empty:
        return None
    return matches["value"].iloc[0]


def centerline_lookup(frame, foreign_key: str) -> Dict[str, np.ndarray]:
    lookup = {}
    if frame is None or frame.empty or foreign_key not in frame.columns:
        return lookup
    for _, row in frame.iterrows():
        key = row_value(row, foreign_key)
        geometry = row_value(row, "geometry")
        if key is None or geometry is None or geometry.is_empty:
            continue
        lookup[str(safe_int(key, key))] = coords_to_xyz(geometry)
    return lookup


def load_map_context(map_file: Path, location: str) -> MapContext:
    pyogrio = import_map_dependencies()
    projection_system = load_projection_system(pyogrio, map_file)

    boundaries = read_map_layer(pyogrio, map_file, "boundaries", projection_system, True)
    baseline_paths = read_map_layer(pyogrio, map_file, "baseline_paths", projection_system, True)
    lanes = read_map_layer(pyogrio, map_file, "lanes_polygons", projection_system, True)
    lane_connectors = read_map_layer(pyogrio, map_file, "lane_connectors", projection_system, True)
    connector_polygons = read_map_layer(
        pyogrio,
        map_file,
        "gen_lane_connectors_scaled_width_polygons",
        projection_system,
        True,
    )

    areas = {
        layer_name: read_map_layer(pyogrio, map_file, layer_name, projection_system, True)
        for layer_name in (
            "generic_drivable_areas",
            "carpark_areas",
            "crosswalks",
            "walkways",
            "road_segments",
            "intersections",
        )
    }
    stop_polygons = read_map_layer(pyogrio, map_file, "stop_polygons", projection_system, True)
    traffic_lights = read_map_layer(pyogrio, map_file, "traffic_lights", projection_system, True)

    boundary_geometries = {}
    if boundaries is not None and not boundaries.empty:
        for fid, row in boundaries.iterrows():
            geometry = row_value(row, "geometry")
            if geometry is not None and not geometry.is_empty:
                boundary_geometries[str(safe_int(fid, fid))] = geometry

    centerlines_by_lane_id = centerline_lookup(baseline_paths, "lane_fid")
    centerlines_by_connector_id = centerline_lookup(baseline_paths, "lane_connector_fid")
    lane_relationships, connector_relationships = build_lane_relationships(lanes, lane_connectors)
    traffic_light_stop_points = build_traffic_light_stop_points(
        stop_polygons,
        traffic_lights,
        centerlines_by_connector_id,
    )

    return MapContext(
        location=location,
        boundaries=boundaries,
        baseline_paths=baseline_paths,
        lanes=lanes,
        lane_connectors=lane_connectors,
        connector_polygons=connector_polygons,
        areas=areas,
        stop_polygons=stop_polygons,
        traffic_lights=traffic_lights,
        boundary_geometries=boundary_geometries,
        centerlines_by_lane_id=centerlines_by_lane_id,
        centerlines_by_connector_id=centerlines_by_connector_id,
        lane_relationships=lane_relationships,
        connector_relationships=connector_relationships,
        traffic_light_stop_points=traffic_light_stop_points,
    )


def build_lane_relationships(lanes, lane_connectors) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    lane_relationships = {}
    connector_relationships = {}

    if lanes is not None and not lanes.empty:
        groups = {}
        for fid, row in lanes.iterrows():
            lane_id = str(safe_int(row_value(row, "lane_fid", fid), fid))
            group_id = str(row_value(row, "lane_group_fid", ""))
            groups.setdefault(group_id, []).append((safe_int(row_value(row, "lane_index")), lane_id))
            lane_relationships.setdefault(
                lane_id,
                {"entry": [], "exit": [], "left": [], "right": []},
            )

        for group_lanes in groups.values():
            group_lanes.sort()
            for index, (_, lane_id) in enumerate(group_lanes):
                if index > 0:
                    lane_relationships[lane_id]["left"].append(prefixed("lane", group_lanes[index - 1][1]))
                if index < len(group_lanes) - 1:
                    lane_relationships[lane_id]["right"].append(prefixed("lane", group_lanes[index + 1][1]))

    if lane_connectors is not None and not lane_connectors.empty:
        for fid, row in lane_connectors.iterrows():
            connector_id = str(safe_int(fid, fid))
            entry_lane_id = str(safe_int(row_value(row, "entry_lane_fid", ""), ""))
            exit_lane_id = str(safe_int(row_value(row, "exit_lane_fid", ""), ""))
            connector_key = prefixed("lane_connector", connector_id)

            connector_relationships[connector_id] = {
                "entry": [prefixed("lane", entry_lane_id)] if entry_lane_id else [],
                "exit": [prefixed("lane", exit_lane_id)] if exit_lane_id else [],
                "left": [],
                "right": [],
            }
            if entry_lane_id:
                lane_relationships.setdefault(
                    entry_lane_id,
                    {"entry": [], "exit": [], "left": [], "right": []},
                )["exit"].append(connector_key)
            if exit_lane_id:
                lane_relationships.setdefault(
                    exit_lane_id,
                    {"entry": [], "exit": [], "left": [], "right": []},
                )["entry"].append(connector_key)

    return lane_relationships, connector_relationships


def build_traffic_light_stop_points(stop_polygons, traffic_lights, centerlines_by_connector_id):
    stop_points = {}
    if stop_polygons is not None and not stop_polygons.empty:
        for fid, row in stop_polygons.iterrows():
            geometry = row_value(row, "geometry")
            if geometry is None or geometry.is_empty:
                continue
            centroid = geometry.centroid
            point = np.array([centroid.x, centroid.y, 0.0], dtype=np.float64)
            for traffic_light_id in split_ids(row_value(row, "traffic_light_fids")):
                stop_points[str(safe_int(traffic_light_id, traffic_light_id))] = point
            for connector_id in split_ids(row_value(row, "lane_connector_fids")):
                stop_points.setdefault(str(safe_int(connector_id, connector_id)), point)

    for connector_id, centerline in centerlines_by_connector_id.items():
        if connector_id not in stop_points and len(centerline):
            stop_points[connector_id] = centerline[0]

    if traffic_lights is not None and not traffic_lights.empty:
        for fid, row in traffic_lights.iterrows():
            geometry = row_value(row, "geometry")
            if geometry is None or geometry.is_empty:
                continue
            stop_points.setdefault(
                str(safe_int(fid, fid)),
                np.array([geometry.x, geometry.y, 0.0], dtype=np.float64),
            )
    return stop_points


def within_radius(geometry, center: Point, radius: float) -> bool:
    if geometry is None or geometry.is_empty:
        return False
    if radius <= 0:
        return True
    return geometry.distance(center) <= radius


def extract_map_features(map_context: Optional[MapContext], scenario_center, radius: float) -> dict:
    if map_context is None:
        return {}

    ret = {}
    center = Point(float(scenario_center[0]), float(scenario_center[1]))
    add_boundaries(ret, map_context, center, radius)
    add_lanes(ret, map_context, center, radius)
    add_lane_connectors(ret, map_context, center, radius)
    add_areas(ret, map_context, center, radius)
    add_road_edges(ret, map_context, center, radius)
    return ret


def add_boundaries(ret: dict, map_context: MapContext, center: Point, radius: float) -> None:
    boundaries = map_context.boundaries
    if boundaries is None or boundaries.empty:
        return
    for fid, row in boundaries.iterrows():
        geometry = row_value(row, "geometry")
        if not within_radius(geometry, center, radius):
            continue
        boundary_id = str(safe_int(fid, fid))
        ret[prefixed("boundary", boundary_id)] = {
            "type": "ROAD_LINE",
            "subtype": ROADLINE_MAPPING.get(safe_int(row_value(row, "boundary_type_fid")), "UNKNOWN"),
            "polyline": coords_to_xyz(geometry),
            "phase": "CONVERTED",
        }


def add_lanes(ret: dict, map_context: MapContext, center: Point, radius: float) -> None:
    lanes = map_context.lanes
    if lanes is None or lanes.empty:
        return
    for fid, row in lanes.iterrows():
        geometry = row_value(row, "geometry")
        if not within_radius(geometry, center, radius):
            continue
        lane_id = str(safe_int(row_value(row, "lane_fid", fid), fid))
        left_id = str(safe_int(row_value(row, "left_boundary_fid", ""), ""))
        right_id = str(safe_int(row_value(row, "right_boundary_fid", ""), ""))
        centerline = map_context.centerlines_by_lane_id.get(lane_id, coords_to_xyz(geometry))
        relationships = map_context.lane_relationships.get(
            lane_id,
            {"entry": [], "exit": [], "left": [], "right": []},
        )
        ret[prefixed("lane", lane_id)] = {
            "type": "ROAD_LANE",
            "subtype": LANE_TYPE_MAPPING.get(safe_int(row_value(row, "lane_type_fid")), "UNKNOWN"),
            "polyline": centerline,
            "speed_limit": row_value(row, "speed_limit_mps"),
            "is_intersection": None,
            "lane_info": {
                "entry_lane_ids": relationships["entry"],
                "exit_lane_ids": relationships["exit"],
                "left_lane_ids": relationships["left"],
                "right_lane_ids": relationships["right"],
            },
            "boundary_info": {
                "left_boundary_id": prefixed("boundary", left_id) if left_id else None,
                "right_boundary_id": prefixed("boundary", right_id) if right_id else None,
                "left_boundary_polyline": coords_to_xyz(map_context.boundary_geometries.get(left_id)),
                "right_boundary_polyline": coords_to_xyz(map_context.boundary_geometries.get(right_id)),
            },
            "phase": "CONVERTED",
        }


def add_lane_connectors(ret: dict, map_context: MapContext, center: Point, radius: float) -> None:
    connector_polygons = map_context.connector_polygons
    if connector_polygons is None or connector_polygons.empty:
        return
    for fid, row in connector_polygons.iterrows():
        geometry = row_value(row, "geometry")
        if not within_radius(geometry, center, radius):
            continue
        connector_id = str(safe_int(row_value(row, "lane_connector_fid", fid), fid))
        left_id = str(safe_int(row_value(row, "left_boundary_fid", ""), ""))
        right_id = str(safe_int(row_value(row, "right_boundary_fid", ""), ""))
        centerline = map_context.centerlines_by_connector_id.get(connector_id, coords_to_xyz(geometry))
        relationships = map_context.connector_relationships.get(
            connector_id,
            {"entry": [], "exit": [], "left": [], "right": []},
        )
        ret[prefixed("lane_connector", connector_id)] = {
            "type": "ROAD_LANE",
            "subtype": "LANE_CONNECTOR",
            "polyline": centerline,
            "speed_limit": row_value(row, "speed_limit_mps"),
            "is_intersection": True,
            "lane_info": {
                "entry_lane_ids": relationships["entry"],
                "exit_lane_ids": relationships["exit"],
                "left_lane_ids": relationships["left"],
                "right_lane_ids": relationships["right"],
            },
            "boundary_info": {
                "left_boundary_id": prefixed("boundary", left_id) if left_id else None,
                "right_boundary_id": prefixed("boundary", right_id) if right_id else None,
                "left_boundary_polyline": coords_to_xyz(map_context.boundary_geometries.get(left_id)),
                "right_boundary_polyline": coords_to_xyz(map_context.boundary_geometries.get(right_id)),
            },
            "phase": "CONVERTED",
        }


def add_areas(ret: dict, map_context: MapContext, center: Point, radius: float) -> None:
    layer_specs = {
        "carpark_areas": ("OTHER", "PARKING"),
        "crosswalks": ("OTHER", "CROSSWALK"),
        "walkways": ("OTHER", "SIDEWALK"),
        "intersections": ("OTHER", "INTERSECTION"),
        "generic_drivable_areas": ("OTHER", "DRIVABLE_AREA"),
        "road_segments": ("OTHER", "ROAD_SEGMENT"),
    }
    for layer_name, (feature_type, subtype) in layer_specs.items():
        frame = map_context.areas.get(layer_name)
        if frame is None or frame.empty:
            continue
        for fid, row in frame.iterrows():
            geometry = row_value(row, "geometry")
            if not within_radius(geometry, center, radius):
                continue
            ret[f"{layer_name}_{safe_int(fid, fid)}"] = {
                "type": feature_type,
                "subtype": subtype,
                "polygon": coords_to_xyz(geometry),
                "phase": "CONVERTED",
            }

    stop_polygons = map_context.stop_polygons
    if stop_polygons is not None and not stop_polygons.empty:
        for fid, row in stop_polygons.iterrows():
            geometry = row_value(row, "geometry")
            if not within_radius(geometry, center, radius):
                continue
            ret[f"stop_polygon_{safe_int(fid, fid)}"] = {
                "type": "OTHER",
                "subtype": STOP_POLYGON_MAPPING.get(
                    safe_int(row_value(row, "stop_polygon_type_fid"), -1),
                    "STOP_LINE",
                ),
                "polygon": coords_to_xyz(geometry),
                "phase": "CONVERTED",
            }


def add_road_edges(ret: dict, map_context: MapContext, center: Point, radius: float) -> None:
    geometries = []
    for layer_name in ("road_segments", "intersections", "generic_drivable_areas"):
        frame = map_context.areas.get(layer_name)
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            geometry = row_value(row, "geometry")
            if within_radius(geometry, center, radius):
                geometries.append(geometry)
    if not geometries:
        return
    boundary = unary_union(geometries).boundary
    edge_geometries = getattr(boundary, "geoms", [boundary])
    for index, geometry in enumerate(edge_geometries):
        ret[f"road_edge_{index}"] = {
            "type": "ROAD_EDGE",
            "subtype": "UNKNOWN",
            "polyline": coords_to_xyz(geometry),
            "phase": "CONVERTED",
        }


def get_traffic_light_state(status) -> str:
    if isinstance(status, str):
        upper = status.upper()
        if upper in {"GREEN", "YELLOW", "RED", "UNKNOWN"}:
            return upper
    value = safe_int(status, -1)
    return {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "UNKNOWN"}.get(value, "UNKNOWN")


def extract_dynamic_map_states(
    connection: sqlite3.Connection,
    frames: List[LidarFrame],
    map_context: Optional[MapContext],
    args,
) -> dict:
    if args.skip_dynamic_map_states or not table_exists(connection, "traffic_light_status"):
        return {}

    columns = table_columns(connection, "traffic_light_status")
    if "lidar_pc_token" not in columns or "status" not in columns:
        return {}

    lane_column = None
    for candidate in ("lane_connector_id", "lane_connector_fid", "lane_connector_token"):
        if candidate in columns:
            lane_column = candidate
            break
    if lane_column is None:
        return {}

    frame_index_by_lidar_token = {frame.token_id: index for index, frame in enumerate(frames)}
    rows = query_by_tokens(
        connection,
        f"SELECT lidar_pc_token, {lane_column} AS lane_id, status FROM traffic_light_status",
        "lidar_pc_token",
        [frame.token for frame in frames],
    )

    ret = {}
    scenario_length = len(frames)
    stop_points = map_context.traffic_light_stop_points if map_context is not None else {}
    for row in rows:
        frame_index = frame_index_by_lidar_token.get(token_to_str(row["lidar_pc_token"]))
        if frame_index is None:
            continue
        lane_id = str(safe_int(row["lane_id"], row["lane_id"]))
        state_id = prefixed("traffic_light", lane_id)
        if state_id not in ret:
            ret[state_id] = traffic_light_state_template(state_id, scenario_length)
            stop_point = stop_points.get(lane_id)
            if stop_point is not None:
                ret[state_id]["stop_point"][:] = stop_point[:3]
            ret[state_id]["type"] = "TRAFFIC_LIGHT"
            ret[state_id]["subtype"] = "UNKNOWN"
            ret[state_id]["controlled_lane"] = prefixed("lane_connector", lane_id)
            ret[state_id]["phase"] = "CONVERTED"

        ret[state_id]["state"]["color"][frame_index] = get_traffic_light_state(row["status"])
        ret[state_id]["state"]["valid"][frame_index] = True
    return ret


def scenario_center_from_tracks(object_tracks: dict) -> np.ndarray:
    ego_track = object_tracks.get("ego")
    if ego_track is None:
        return np.array([0.0, 0.0], dtype=np.float64)
    valid = ego_track["state"]["valid"]
    centers = ego_track["state"]["global_center"]
    if np.any(valid):
        valid_centers = centers[valid]
        return valid_centers[len(valid_centers) // 2, :2]
    return centers[len(centers) // 2, :2]


def process_window(
    connection: sqlite3.Connection,
    db_file: Path,
    location: str,
    frames: List[LidarFrame],
    track_infos: Dict[str, dict],
    map_context: Optional[MapContext],
    args,
) -> None:
    scenario_id = f"{db_file.stem}_{frames[0].timestamp_us}_{frames[-1].timestamp_us}"
    object_tracks = extract_object_tracks(connection, frames, track_infos, args)
    scenario_center = scenario_center_from_tracks(object_tracks)
    map_features = {} if args.skip_maps else extract_map_features(map_context, scenario_center, args.map_radius)
    dynamic_map_states = extract_dynamic_map_states(connection, frames, map_context, args)

    timestamps = np.array([frame.timestamp_us for frame in frames], dtype=np.float64) / 1_000_000
    if len(timestamps):
        timestamps = timestamps - timestamps[0]
    sampling_rate = int(round(1.0 / np.median(np.diff(timestamps)))) if len(timestamps) > 1 else 0

    scenario = {
        "object_tracks": object_tracks,
        "map_features": map_features,
        "dynamic_map_states": dynamic_map_states,
        "extra_information": {
            "id": scenario_id,
            "dataset": "nuPlan",
            "version": args.raw_dataset_version,
            "country": get_country_name(location),
            "city": get_city_name(location),
            "location": location,
            "sdc_id": "ego",
            "scene_length": len(frames),
            "sampling_rate": sampling_rate,
            "timestamp": timestamps,
            "init_time": frames[0].timestamp_us,
        },
    }
    save_scenario_pickle(scenario, args, scenario_id)


def process_database(params) -> None:
    args, db_file, maps_root = params
    with sqlite3.connect(db_file) as connection:
        connection.row_factory = sqlite3.Row
        location = get_location(connection, db_file)
        frames = load_lidar_frames(connection)
        windows = build_windows(frames, args)
        if not windows:
            return

        track_infos = load_track_infos(connection)
        map_context = None
        if not args.skip_maps:
            map_file = find_map_file(maps_root, location)
            if map_file is None:
                raise FileNotFoundError(f"Cannot find map.gpkg for nuPlan location '{location}'.")
            map_context = load_map_context(map_file, location)

        for window in windows:
            process_window(connection, db_file, location, window, track_infos, map_context, args)


def convert(args):
    db_files = list_db_files(args)
    maps_root = None if args.skip_maps else resolve_maps_root(args.raw_dataset_path)
    if not args.skip_maps and maps_root is None:
        raise FileNotFoundError(f"Cannot find nuPlan maps root under {args.raw_dataset_path}.")

    params = [(args, db_file, maps_root) for db_file in db_files]
    run_tasks(process_database, params, len(params), args, tqdm, pool_options={"maxtasksperchild": 5})


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
