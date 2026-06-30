try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import json
import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from numcodecs import blosc
from tqdm import tqdm

try:
    from scenario_convertor.utils import (
        add_common_converter_args,
        get_road_lane_polyline,
        global_to_local_velocity,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )
except ModuleNotFoundError:
    from utils import (
        add_common_converter_args,
        get_road_lane_polyline,
        global_to_local_velocity,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )

try:
    from scenario_convertor.lyft_proto.road_network_pb2 import MapFragment
except ModuleNotFoundError:
    try:
        from lyft_proto.road_network_pb2 import MapFragment
    except ModuleNotFoundError:
        MapFragment = None

try:
    import pymap3d as pm
except ModuleNotFoundError:
    pm = None


PERCEPTION_LABEL_COUNT = 17
SCENE_CHUNK_SIZE = 10_000
FRAME_CHUNK_SIZE = 10_000
AGENT_CHUNK_SIZE = 20_000
TL_FACE_CHUNK_SIZE = 10_000
MAX_DISTANCE = 150

SCENE_DTYPE = np.dtype(
    [
        ("frame_index_interval", np.int64, (2,)),
        ("host", "<U16"),
        ("start_time", np.int64),
        ("end_time", np.int64),
    ]
)
FRAME_DTYPE = np.dtype(
    [
        ("timestamp", np.int64),
        ("agent_index_interval", np.int64, (2,)),
        ("traffic_light_faces_index_interval", np.int64, (2,)),
        ("ego_translation", np.float64, (3,)),
        ("ego_rotation", np.float64, (3, 3)),
    ]
)
AGENT_DTYPE = np.dtype(
    [
        ("centroid", np.float64, (2,)),
        ("extent", np.float32, (3,)),
        ("yaw", np.float32),
        ("velocity", np.float32, (2,)),
        ("track_id", np.uint64),
        ("label_probabilities", np.float32, (PERCEPTION_LABEL_COUNT,)),
    ]
)
TL_FACE_DTYPE = np.dtype(
    [
        ("face_id", "<U16"),
        ("traffic_light_id", "<U16"),
        ("traffic_light_face_status", np.float32, (3,)),
    ]
)

dataset = None
map_api = None
split_name = "sample"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Lyft Level 5 Prediction Dataset Converter")
    add_common_converter_args(
        parser,
        default_raw_dataset_path="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/lyft-level-5-motion-dataset",
        default_subset_name="lyft-level-5",
        default_raw_dataset_version="1.1",
        raw_dataset_help="The file directory for L5PD",
    )
    parser.add_argument("--split", choices=("sample", "train", "val"), default="sample", help="The Lyft split to convert.")
    parser.add_argument("--zarr_name", default=None, help="Override the zarr directory name, e.g. sample.zarr.")
    parser.add_argument("--max_scenarios", default=None, type=int, help="Convert at most this many scenes.")
    parser.add_argument("--skip_maps", action="store_true", help="Skip semantic_map.pb parsing.")
    parser.add_argument("--map_metadata_path", default=None, help="Path to meta.json containing world_to_ecef.")
    parser.add_argument("--ego_length", default=4.9, type=float, help="Fallback ego vehicle length.")
    parser.add_argument("--ego_width", default=2.0, type=float, help="Fallback ego vehicle width.")
    parser.add_argument("--ego_height", default=1.8, type=float, help="Fallback ego vehicle height.")
    return parser


def resolve_zarr_path(args) -> Path:
    raw_path = Path(args.raw_dataset_path)
    zarr_name = args.zarr_name
    if zarr_name is None:
        zarr_name = {"sample": "sample.zarr", "train": "train_full.zarr", "val": "validate.zarr"}[args.split]
    if not zarr_name.endswith(".zarr"):
        zarr_name = f"{zarr_name}.zarr"
    zarr_path = raw_path / zarr_name
    if not zarr_path.exists():
        available = sorted(path.name for path in raw_path.glob("*.zarr"))
        raise FileNotFoundError(f"Cannot find {zarr_path}. Available zarr datasets: {available}")
    return zarr_path


class ZarrLyftDataset:
    def __init__(self, zarr_path: Path):
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        self.scenes = root["scenes"]
        self.frames = root["frames"]
        self.agents = root["agents"]
        self.tl_faces = root["traffic_light_faces"] if "traffic_light_faces" in root else np.empty((0,), dtype=TL_FACE_DTYPE)


class RawChunkLyftDataset:
    def __init__(self, zarr_path: Path):
        scenes = self._read_chunks(zarr_path / "scenes", SCENE_DTYPE, SCENE_CHUNK_SIZE)
        valid_scenes = scenes["end_time"] > 0
        self.scenes = scenes[valid_scenes]
        if len(self.scenes) == 0:
            raise ValueError(f"No valid scenes found in {zarr_path}.")

        frame_count = int(self.scenes["frame_index_interval"][-1][1])
        frames = self._read_chunks(zarr_path / "frames", FRAME_DTYPE, FRAME_CHUNK_SIZE)
        self.frames = frames[:frame_count]

        agent_count = int(self.frames["agent_index_interval"][-1][1]) if len(self.frames) else 0
        agents = self._read_chunks(zarr_path / "agents", AGENT_DTYPE, AGENT_CHUNK_SIZE)
        self.agents = agents[:agent_count]

        tl_dir = zarr_path / "traffic_light_faces"
        if tl_dir.exists():
            tl_count = int(self.frames["traffic_light_faces_index_interval"][-1][1]) if len(self.frames) else 0
            tl_faces = self._read_chunks(tl_dir, TL_FACE_DTYPE, TL_FACE_CHUNK_SIZE)
            self.tl_faces = tl_faces[:tl_count]
        else:
            self.tl_faces = np.empty((0,), dtype=TL_FACE_DTYPE)

    @staticmethod
    def _read_chunks(array_path: Path, dtype: np.dtype, chunk_size: int) -> np.ndarray:
        if not array_path.exists():
            raise FileNotFoundError(f"Cannot find zarr array directory: {array_path}")
        chunks = []
        for chunk_path in sorted((path for path in array_path.iterdir() if path.is_file() and path.name.isdigit()), key=lambda path: int(path.name)):
            raw = blosc.decompress(chunk_path.read_bytes())
            chunk = np.frombuffer(raw, dtype=dtype)
            if len(chunk) != chunk_size:
                raise ValueError(f"Unexpected chunk length in {chunk_path}: {len(chunk)} != {chunk_size}")
            chunks.append(chunk)
        if not chunks:
            return np.empty((0,), dtype=dtype)
        return np.concatenate(chunks)


def open_lyft_dataset(zarr_path: Path):
    if (zarr_path / ".zgroup").exists():
        return ZarrLyftDataset(zarr_path)
    warnings.warn(
        f"{zarr_path} is missing zarr metadata files. Falling back to fixed Lyft/L5Kit chunk decoding.",
        RuntimeWarning,
    )
    return RawChunkLyftDataset(zarr_path)


def load_world_to_ecef(args) -> Optional[np.ndarray]:
    metadata_path = Path(args.map_metadata_path) if args.map_metadata_path else Path(args.raw_dataset_path) / "meta.json"
    if not metadata_path.exists():
        warnings.warn(f"Cannot find Lyft meta.json at {metadata_path}; map_features will be empty.", RuntimeWarning)
        return None
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if "world_to_ecef" not in metadata:
        warnings.warn(f"{metadata_path} does not contain world_to_ecef; map_features will be empty.", RuntimeWarning)
        return None
    return np.asarray(metadata["world_to_ecef"], dtype=np.float64)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


class LyftMapAPI:
    def __init__(self, protobuf_map_path: Path, world_to_ecef: np.ndarray):
        if MapFragment is None:
            raise ModuleNotFoundError("Lyft map parsing requires scenario_convertor.lyft_proto.road_network_pb2.")
        if pm is None:
            raise ModuleNotFoundError("Lyft map parsing requires pymap3d.")
        self.ecef_to_world = np.linalg.inv(world_to_ecef)
        with open(protobuf_map_path, "rb") as file:
            map_fragment = MapFragment()
            map_fragment.ParseFromString(file.read())
        self.elements = map_fragment.elements
        self.ids_to_element = {self.id_as_str(element.id): index for index, element in enumerate(self.elements)}

    @staticmethod
    def id_as_str(element_id) -> str:
        return element_id.id.decode("utf-8")

    @staticmethod
    def _undo_e7(value: float) -> float:
        return value / 1e7

    def unpack_deltas_cm(self, dx, dy, dz, frame) -> np.ndarray:
        x = np.cumsum(np.asarray(dx, dtype=np.float64) / 100)
        y = np.cumsum(np.asarray(dy, dtype=np.float64) / 100)
        z = np.cumsum(np.asarray(dz, dtype=np.float64) / 100)
        frame_lat = self._undo_e7(frame.origin.lat_e7)
        frame_lng = self._undo_e7(frame.origin.lng_e7)
        xyz = np.stack(pm.enu2ecef(x, y, z, frame_lat, frame_lng, 0), axis=-1)
        return transform_points(xyz, self.ecef_to_world)

    @staticmethod
    def is_lane(element) -> bool:
        return bool(element.element.HasField("lane"))

    @lru_cache(maxsize=100_000)
    def get_lane_coords(self, element_id: str) -> dict:
        element = self[element_id]
        lane = element.element.lane
        left_boundary = lane.left_boundary
        right_boundary = lane.right_boundary
        xyz_left = self.unpack_deltas_cm(
            left_boundary.vertex_deltas_x_cm,
            left_boundary.vertex_deltas_y_cm,
            left_boundary.vertex_deltas_z_cm,
            lane.geo_frame,
        )
        xyz_right = self.unpack_deltas_cm(
            right_boundary.vertex_deltas_x_cm,
            right_boundary.vertex_deltas_y_cm,
            right_boundary.vertex_deltas_z_cm,
            lane.geo_frame,
        )
        return {"xyz_left": xyz_left, "xyz_right": xyz_right}

    @staticmethod
    def is_crosswalk(element) -> bool:
        if not element.element.HasField("traffic_control_element"):
            return False
        traffic_element = element.element.traffic_control_element
        return bool(traffic_element.HasField("pedestrian_crosswalk") and traffic_element.points_x_deltas_cm)

    @lru_cache(maxsize=100_000)
    def get_crosswalk_coords(self, element_id: str) -> dict:
        element = self[element_id]
        traffic_element = element.element.traffic_control_element
        xyz = self.unpack_deltas_cm(
            traffic_element.points_x_deltas_cm,
            traffic_element.points_y_deltas_cm,
            traffic_element.points_z_deltas_cm,
            traffic_element.geo_frame,
        )
        return {"xyz": xyz}

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.elements[self.ids_to_element[item]]
        if isinstance(item, int):
            return self.elements[item]
        if isinstance(item, bytes):
            return self.elements[self.ids_to_element[item.decode("utf-8")]]
        raise TypeError("LyftMapAPI keys must be str, bytes, or int.")


def initialize_converter(args):
    global dataset, map_api, split_name
    split_name = args.split
    dataset = open_lyft_dataset(resolve_zarr_path(args))
    map_api = None
    if args.skip_maps:
        return
    world_to_ecef = load_world_to_ecef(args)
    semantic_map_path = Path(args.raw_dataset_path) / "semantic_map" / "semantic_map.pb"
    if world_to_ecef is None or not semantic_map_path.exists():
        if not semantic_map_path.exists():
            warnings.warn(f"Cannot find {semantic_map_path}; map_features will be empty.", RuntimeWarning)
        return
    map_api = LyftMapAPI(semantic_map_path, world_to_ecef)


def get_road_lane_type_and_subtype(road_lane_type):
    if road_lane_type in (0, 5):
        return "ROAD_LANE", "URBAN_UNKNOWN"
    if road_lane_type == 1:
        return "ROAD_LANE", "URBAN_ALL"
    if road_lane_type == 2:
        return "ROAD_LANE", "URBAN_VEHICLE"
    if road_lane_type == 3:
        return "ROAD_LANE", "URBAN_BUS"
    if road_lane_type == 4:
        return "ROAD_LANE", "URBAN_BIKE"
    raise ValueError(f"Unknown road lane type: {road_lane_type}")


def get_road_line_edge_type(road_line_edge_type):
    if not road_line_edge_type:
        return "ROAD_LINE", "UNKNOWN"
    edge_type = road_line_edge_type[0]
    if edge_type in (0, 1):
        return "ROAD_LINE", "UNKNOWN"
    if edge_type == 2:
        return "ROAD_LINE", "SOLID_YELLOW"
    if edge_type == 3:
        return "ROAD_LINE", "SOLID_WHITE"
    if edge_type == 4:
        return "ROAD_LINE", "DASHED_YELLOW"
    if edge_type == 5:
        return "ROAD_LINE", "DASHED_WHITE"
    if edge_type == 6:
        return "ROAD_LINE", "DOUBLE_SOLID_YELLOW"
    if edge_type == 7:
        return "ROAD_LINE", "DOUBLE_SOLID_WHITE"
    if edge_type == 8:
        return "ROAD_LINE", "SOLID_DASH_YELLOW"
    if edge_type == 9:
        return "ROAD_LINE", "DASH_SOLID_YELLOW"
    if edge_type in (10, 11, 12):
        return "ROAD_EDGE", "UNKNOWN"
    raise ValueError(f"Unknown road line or road edge type: {road_line_edge_type}")


def extract_road_lines_edges(road_line, road_line_polyline):
    return {
        "type": get_road_line_edge_type(road_line.divider_type)[0],
        "subtype": get_road_line_edge_type(road_line.divider_type)[1],
        "polyline": road_line_polyline,
        "phase": "CONVERTED",
    }


def extract_map_features(scenario_center: np.ndarray):
    if map_api is None:
        return {}
    ret = {}
    for map_feature in map_api.elements:
        if map_api.is_lane(map_feature):
            road_lane_id = map_api.id_as_str(map_feature.id)
            lane_coords = map_api.get_lane_coords(road_lane_id)
            left_polyline = lane_coords["xyz_left"]
            right_polyline = lane_coords["xyz_right"]
            left_mid = left_polyline[len(left_polyline) // 2][0:2]
            right_mid = right_polyline[len(right_polyline) // 2][0:2]
            if np.linalg.norm(left_mid - scenario_center[0:2]) <= MAX_DISTANCE and np.linalg.norm(right_mid - scenario_center[0:2]) <= MAX_DISTANCE:
                left_id = f"{road_lane_id}_1"
                right_id = f"{road_lane_id}_2"
                ret[left_id] = extract_road_lines_edges(map_feature.element.lane.left_boundary, left_polyline)
                ret[right_id] = extract_road_lines_edges(map_feature.element.lane.right_boundary, right_polyline)

                lane_info = {}
                lane_info["type"], lane_info["subtype"] = get_road_lane_type_and_subtype(map_feature.element.lane.access_restriction.type)
                lane_info["polyline"] = get_road_lane_polyline(ret[left_id]["polyline"], ret[right_id]["polyline"])
                lane_info["speed_limit"] = None
                lane_info["is_intersection"] = None
                lane_info["phase"] = "CONVERTED"
                lane_info["lane_info"] = {
                    "entry_lane_ids": {},
                    "exit_lane_ids": {},
                    "left_lane_ids": {},
                    "right_lane_ids": {},
                }
                lane_info["boundary_info"] = {
                    "left_boundary_id": left_id,
                    "right_boundary_id": right_id,
                    "left_boundary_polyline": ret[left_id]["polyline"],
                    "right_boundary_polyline": ret[right_id]["polyline"],
                }
                ret[road_lane_id] = lane_info

        if map_api.is_crosswalk(map_feature):
            crosswalk_id = map_api.id_as_str(map_feature.id)
            crosswalk_polygon = map_api.get_crosswalk_coords(crosswalk_id)["xyz"]
            if len(crosswalk_polygon) and np.linalg.norm(crosswalk_polygon[0][0:2] - scenario_center[0:2]) <= MAX_DISTANCE:
                ret[crosswalk_id] = {
                    "type": "OTHER",
                    "subtype": "CROSSWALK",
                    "polygon": crosswalk_polygon,
                    "phase": "CONVERTED",
                }
    return ret


def get_object_type(label_probabilities):
    label = int(np.argmax(label_probabilities))
    if label in (0, 1, 2, 16):
        return "UNKNOWN", "UNKNOWN"
    if label == 3:
        return "VEHICLE", "CAR"
    if label == 4:
        return "VEHICLE", "VAN"
    if label == 5:
        return "VEHICLE", "TRAM"
    if label == 6:
        return "VEHICLE", "BUS"
    if label == 7:
        return "VEHICLE", "TRUCK"
    if label == 8:
        return "VEHICLE", "EMERGENCY"
    if label == 9:
        return "VEHICLE", "UNKNOWN"
    if label in (10, 12):
        return "CYCLIST", "BICYCLIST"
    if label in (11, 13):
        return "CYCLIST", "MOTORCYCLIST"
    if label == 14:
        return "PEDESTRIAN", "PEDESTRIAN"
    if label == 15:
        return "OTHER", "ANIMAL"
    raise ValueError(f"Unknown object type label: {label}")


def yaw_from_rotation(rotation: np.ndarray) -> float:
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def estimate_global_velocity(centers: np.ndarray, timestamps: np.ndarray, index: int) -> tuple:
    if len(centers) < 2:
        return 0.0, 0.0
    if index == 0:
        prev_index, next_index = 0, 1
    elif index == len(centers) - 1:
        prev_index, next_index = index - 1, index
    else:
        prev_index, next_index = index - 1, index + 1
    dt = float(timestamps[next_index] - timestamps[prev_index])
    if dt <= 0:
        return 0.0, 0.0
    velocity = (centers[next_index] - centers[prev_index]) / dt
    return float(velocity[0]), float(velocity[1])


def extract_ego_track(frames: np.ndarray, timestamps: np.ndarray, args) -> dict:
    ego_state = object_state_template(len(frames))
    ego_state["type"], ego_state["subtype"] = "VEHICLE", "CAR"
    ego_state["phase"] = "CONVERTED"
    centers = np.asarray(frames["ego_translation"], dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        heading = yaw_from_rotation(frame["ego_rotation"])
        vx, vy = estimate_global_velocity(centers, timestamps, frame_index)
        local_velocity = global_to_local_velocity(vx, vy, heading)
        ego_state["state"]["global_center"][frame_index] = frame["ego_translation"]
        ego_state["state"]["heading"][frame_index] = heading
        ego_state["state"]["local_velocity"][frame_index] = local_velocity
        ego_state["state"]["size"][frame_index] = (args.ego_length, args.ego_width, args.ego_height)
        ego_state["state"]["valid"][frame_index] = True
    return ego_state


def extract_object_tracks(frames: np.ndarray, raw_agent_track_data: np.ndarray, agent_range_detail: np.ndarray, timestamps: np.ndarray, args) -> dict:
    scenario_length = len(frames)
    ret = {"ego": extract_ego_track(frames, timestamps, args)}
    if len(raw_agent_track_data) == 0:
        return ret

    agent_offset = int(agent_range_detail[0][0])
    for frame_index, agent_range in enumerate(agent_range_detail):
        start = int(agent_range[0]) - agent_offset
        end = int(agent_range[1]) - agent_offset
        for agent in raw_agent_track_data[start:end]:
            object_id = str(int(agent["track_id"]))
            object_state = ret.get(object_id)
            if object_state is None:
                object_state = object_state_template(scenario_length)
                object_state["type"], object_state["subtype"] = get_object_type(agent["label_probabilities"])
                object_state["phase"] = "CONVERTED"
                ret[object_id] = object_state

            heading = float(agent["yaw"])
            local_velocity = global_to_local_velocity(float(agent["velocity"][0]), float(agent["velocity"][1]), heading)
            object_state["state"]["global_center"][frame_index] = (float(agent["centroid"][0]), float(agent["centroid"][1]), 0.0)
            object_state["state"]["heading"][frame_index] = heading
            object_state["state"]["local_velocity"][frame_index] = local_velocity
            object_state["state"]["size"][frame_index] = agent["extent"]
            object_state["state"]["valid"][frame_index] = True
    return ret


def scenario_prefix() -> str:
    return {"sample": "s", "train": "t", "val": "v"}.get(split_name, split_name[0])


def process_scenario(params):
    args, scenario_index = params
    scene = dataset.scenes[scenario_index]
    frame_start, frame_end = (int(value) for value in scene["frame_index_interval"])
    frames = dataset.frames[frame_start:frame_end]
    if len(frames) == 0:
        return

    timestamps = np.asarray(frames["timestamp"], dtype=np.float64) / 1_000_000_000
    if len(timestamps) > 1 and timestamps[-1] - timestamps[0] == 0:
        timestamps = np.arange(len(timestamps), dtype=np.float64) / 10

    agent_start = int(frames[0]["agent_index_interval"][0])
    agent_end = int(frames[-1]["agent_index_interval"][1])
    agent_range_detail = np.asarray(frames["agent_index_interval"])
    raw_agent_track_data = dataset.agents[agent_start:agent_end]

    object_tracks = extract_object_tracks(frames, raw_agent_track_data, agent_range_detail, timestamps, args)
    scenario_center = object_tracks["ego"]["state"]["global_center"][len(frames) // 2]
    map_features = extract_map_features(scenario_center)

    scenario_id = f"{scenario_prefix()}_{scenario_index}"
    scenario = {
        "object_tracks": object_tracks,
        "map_features": map_features,
        "dynamic_map_states": {},
        "extra_information": {
            "id": scenario_id,
            "dataset": "Lyft Level 5",
            "version": args.raw_dataset_version,
            "country": "AMERICA",
            "city": "PALO_ALTO",
            "location": None,
            "sdc_id": "ego",
            "scene_length": len(frames),
            "sampling_rate": 10,
            "timestamp": timestamps,
        },
    }
    save_scenario_pickle(scenario, args, scenario_id)


def convert(args):
    source_dataset = open_lyft_dataset(resolve_zarr_path(args))
    scenario_ids = list(range(len(source_dataset.scenes)))
    del source_dataset
    if args.max_scenarios is not None:
        scenario_ids = scenario_ids[: args.max_scenarios]
    params = [(args, scenario_id) for scenario_id in scenario_ids]
    run_tasks(
        process_scenario,
        params,
        len(scenario_ids),
        args,
        tqdm,
        initializer=initialize_converter,
        initargs=(args,),
    )


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
