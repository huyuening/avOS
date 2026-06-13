# Shifts Motion Prediction Dataset

# Shifts protobuf definitions are vendored in scenario_convertor.shifts_proto.

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
from pathlib import Path
import numpy as np

from tqdm import tqdm
from utils import add_common_converter_args, object_state_template, global_to_local_velocity, run_tasks, save_scenario_pickle

try:
    from scenario_convertor.shifts_proto import Scene
except ModuleNotFoundError:
    from shifts_proto import Scene


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Shifts Motion Prediction Dataset Converter')
    add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/shifts-motion-dataset',
        default_subset_name='shifts',
        default_raw_dataset_version='1.0',
        raw_dataset_help='The file directory for Shifts',
    )
    parser.add_argument(
        '--max_files',
        default=None,
        type=int,
        help='Convert at most this many Shifts protobuf files.',
    )
    return parser


def get_file_paths(dataset_path: str):
    root = Path(dataset_path)
    if not root.exists():
        return []
    return sorted(str(path) for path in root.rglob('*.pb'))


def read_scene_from_file(filepath: str):
    scene = Scene()
    with open(filepath, 'rb') as file:
        scene.ParseFromString(file.read())
    return scene


def list_filepaths(raw_dataset_path):
    root = Path(raw_dataset_path)
    split_dirs = [
        root / 'train',
        root / 'development',
        root / 'evaluation',
        root / 'mini',
    ]
    existing_split_dirs = [path for path in split_dirs if path.exists()]
    search_dirs = existing_split_dirs or [root]
    filepaths = []
    for search_dir in search_dirs:
        filepaths.extend(get_file_paths(str(search_dir)))
    return sorted(filepaths)


def get_country_name(city_name_):
    if city_name_ == 0:
        return 'UNKNOWN'
    elif city_name_ == 1 or city_name_ == 2 or city_name_ == 3:
        return 'RUSSIA'
    elif city_name_ == 4:
        return 'AMERICA'
    elif city_name_ == 5 or city_name_ == 6:
        return 'ISRAEL'
    else:
        return ValueError("Unknown city name: {}".format(city_name_))


def get_city_name(city_name_):
    if city_name_ == 0:
        return 'UNKNOWN'
    elif city_name_ == 1:
        return 'MOSCOW'
    elif city_name_ == 2:
        return 'SKOLKOVO'
    elif city_name_ == 3:
        return 'INNOPOLIS'
    elif city_name_ == 4:
        return 'ANN_ARBOR'
    elif city_name_ == 5:
        return 'MODIIN'
    elif city_name_ == 6:
        return 'TEL_AVIV'
    else:
        return ValueError("Unknown city name: {}".format(city_name_))


def extract_map_features(raw_map_feature_data_):
    ret = dict()

    crosswalks = raw_map_feature_data_.crosswalks
    road_lines = raw_map_feature_data_.road_polygons
    road_lanes = raw_map_feature_data_.lanes

    for i, road_line in enumerate(road_lines):
        road_line_info = dict()

        road_line_id = '0' + str(i)
        road_line_polyline = np.array([[point.x, point.y, 0] for point in road_line.geometry.points])

        road_line_info['id'] = road_line_id
        road_line_info['type'] = 'ROAD_LINE'
        road_line_info['subtype'] = 'UNKNOWN'
        road_line_info['polyline'] = road_line_polyline
        road_line_info['phase'] = 'CONVERTED'

        ret[road_line_id] = road_line_info

    for i, road_lane in enumerate(road_lanes):
        road_lane_info = dict()

        road_lane_id = '1' + str(i)
        road_lane_polyline = np.array([[point.x, point.y, 0] for point in road_lane.centers])

        road_lane_info['id'] = road_lane_id
        road_lane_info['type'] = 'ROAD_LANE'
        road_lane_info['subtype'] = 'UNKNOWN'
        road_lane_info['polyline'] = road_lane_polyline
        road_lane_info['speed_limit'] = road_lane.max_velocity
        road_lane_info['is_intersection'] = None
        road_lane_info['phase'] = 'CONVERTED'

        road_lane_info['lane_info'] = dict()
        road_lane_info['lane_info']['entry_lane_ids'] = dict()
        road_lane_info['lane_info']['exit_lane_ids'] = dict()
        road_lane_info['lane_info']['left_lane_ids'] = dict()
        road_lane_info['lane_info']['right_lane_ids'] = dict()

        road_lane_info['boundary_info'] = dict()
        road_lane_info['boundary_info']['left_boundary_id'] = None
        road_lane_info['boundary_info']['right_boundary_id'] = None
        road_lane_info['boundary_info']['left_boundary_polyline'] = np.array([])
        road_lane_info['boundary_info']['right_boundary_polyline'] = np.array([])

        ret[road_lane_id] = road_lane_info

    for i, crosswalk in enumerate(crosswalks):
        crosswalk_info = dict()

        crosswalk_id = '2' + str(i)
        crosswalk_polygon = np.array([[point.x, point.y, 0] for point in crosswalk.geometry.points])

        crosswalk_info['id'] = crosswalk_id
        crosswalk_info['type'] = 'OTHER'
        crosswalk_info['subtype'] = 'CROSSWALK'
        crosswalk_info['polygon'] = crosswalk_polygon
        crosswalk_info['phase'] = 'CONVERTED'

        ret[crosswalk_id] = crosswalk_info

    return ret


def extract_object_tracks(raw_object_track_data_, scenario_length_):
    ret = dict()

    av_id = '0'
    av_states = object_state_template(scenario_length_)
    av_states['type'], av_states['subtype'] = 'VEHICLE', 'CAR'
    av_states['phase'] = 'CONVERTED'
    for frame, av_state in enumerate(raw_object_track_data_[0]):
        av_states['state']['global_center'][frame] = (av_state.position.x, av_state.position.y, 0.0)
        av_states['state']['local_velocity'][frame] = global_to_local_velocity(av_state.linear_velocity.x, av_state.linear_velocity.y, av_state.yaw)
        av_states['state']['heading'][frame] = av_state.yaw
        av_states['state']['size'][frame] = (av_state.dimensions.x, av_state.dimensions.y, av_state.dimensions.z)
        av_states['state']['valid'][frame] = True
    ret[av_id] = av_states

    vehicle_ids = []
    for vehicles_track in raw_object_track_data_[1]:
        for vehicle_state in vehicles_track.tracks:
            vehicle_id = vehicle_state.track_id
            vehicle_ids.append(vehicle_id)
    vehicle_ids = list(set(vehicle_ids))

    for vehicle_id in vehicle_ids:
        vehicle_states = object_state_template(scenario_length_)
        vehicle_states['id'] = str(vehicle_id)
        vehicle_states['type'], vehicle_states['subtype'] = 'VEHICLE', 'UNKNOWN'
        vehicle_states['phase'] = 'CONVERTED'
        for frame, vehicles_track in enumerate(raw_object_track_data_[1]):
            for vehicle_state in vehicles_track.tracks:
                if vehicle_state.track_id == vehicle_id:
                    vehicle_states['state']['global_center'][frame] = (vehicle_state.position.x, vehicle_state.position.y, vehicle_state.position.z)
                    vehicle_states['state']['heading'][frame] = vehicle_state.yaw
                    vehicle_states['state']['size'][frame] = (vehicle_state.dimensions.x, vehicle_state.dimensions.y, vehicle_state.dimensions.z)
                    vehicle_states['state']['valid'][frame] = True
        ret[str(vehicle_id)] = vehicle_states

    pedestrian_ids = []
    for pedestrians_track in raw_object_track_data_[2]:
        for pedestrian_state in pedestrians_track.tracks:
            pedestrian_id = pedestrian_state.track_id
            pedestrian_ids.append(pedestrian_id)
    pedestrian_ids = list(set(pedestrian_ids))

    for pedestrian_id in pedestrian_ids:
        pedestrian_states = object_state_template(scenario_length_)
        pedestrian_states['id'] = str(pedestrian_id)
        pedestrian_states['type'], pedestrian_states['subtype'] = 'PEDESTRIAN', 'PEDESTRIAN'
        pedestrian_states['phase'] = 'CONVERTED'
        for frame, pedestrians_track in enumerate(raw_object_track_data_[2]):
            for pedestrian_state in pedestrians_track.tracks:
                if pedestrian_state.track_id == pedestrian_id:
                    pedestrian_states['state']['global_center'][frame] = (pedestrian_state.position.x, pedestrian_state.position.y, pedestrian_state.position.z)
                    pedestrian_states['state']['size'][frame] = (pedestrian_state.dimensions.x, pedestrian_state.dimensions.y, pedestrian_state.dimensions.z)
                    pedestrian_states['state']['valid'][frame] = True
        ret[str(pedestrian_id)] = pedestrian_states

    return ret


def process_scenario(params):
    args, filepath = params
    scenario_ = dict()
    scenario = read_scene_from_file(filepath)

    scenario_id = scenario.id
    scenario_length = len(scenario.past_ego_track)
    city_name = scenario.scene_tags.track

    raw_object_track_data = [scenario.past_ego_track, scenario.past_vehicle_tracks, scenario.past_pedestrian_tracks]
    converted_object_track_data = extract_object_tracks(raw_object_track_data, scenario_length)

    raw_map_feature_data = scenario.path_graph
    converted_map_feature_data = extract_map_features(raw_map_feature_data)

    converted_dynamic_map_state_data = dict()

    extra_information = dict()
    extra_information['id'] = scenario_id
    extra_information['dataset'] = 'Shifts'
    extra_information['version'] = args.raw_dataset_version
    extra_information['country'] = get_country_name(city_name)
    extra_information['city'] = get_city_name(city_name)
    extra_information['location'] = None
    extra_information['sdc_id'] = '0'
    extra_information['scene_length'] = scenario_length
    extra_information['sampling_rate'] = 5
    extra_information['timestamp'] = np.array(range(scenario_length)) / 5

    scenario_['object_tracks'] = converted_object_track_data
    scenario_['map_features'] = converted_map_feature_data
    scenario_['dynamic_map_states'] = converted_dynamic_map_state_data
    scenario_['extra_information'] = extra_information

    save_scenario_pickle(scenario_, args, scenario_id)


def convert(args):
    filepaths = list_filepaths(args.raw_dataset_path)
    if args.max_files is not None:
        filepaths = filepaths[:args.max_files]
    params = [(args, filepath) for filepath in filepaths]
    run_tasks(process_scenario, params, len(filepaths), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
