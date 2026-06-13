# Argoverse 2 Motion Forecasting Dataset

# Information
# object_tracks:        yes (parquet)
#   ObjectType: VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS,
#               STATIC, BACKGROUND, CONSTRUCTION, RIDERLESS_BICYCLE,
#               UNKNOWN

# map_features:         yes (json)
#   LaneType: VEHICLE, BUS, BIKE
#   LaneMarkType: DOUBLE_SOLID_YELLOW, SOLID_YELLOW, DASHED_YELLOW, DASH_SOLID_YELLOW, DOUBLE_DASH_YELLOW, SOLID_DASH_YELLOW
#                 DOUBLE_SOLID_WHITE, SOLID_WHITE, DASHED_WHITE, DASH_SOLID_WHITE, DOUBLE_DASH_WHITE, SOLID_DASH_WHITE
#                 SOLID_BLUE, NONE, UNKNOWN

# dynamic_map_states:   no
#   None

# Installation
# conda create -n a2md python=3.9
# conda activate a2md
# pip install av2 -i https://pypi.tuna.tsinghua.edu.cn/simple

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import numpy as np
import os

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from tqdm import tqdm
from utils import add_common_converter_args, object_state_template, run_tasks, save_scenario_pickle


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Argoverse 2 Motion Forecasting Dataset Converter')
    return add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/argoverse-2-motion-dataset',
        default_subset_name='argoverse-2',
        default_raw_dataset_version='2.0',
        raw_dataset_help='The file directory for A2MD',
    )


def get_object_type(object_type_):
    if object_type_ == ObjectType.VEHICLE:
        return 'VEHICLE', 'CAR'
    elif object_type_ == ObjectType.BUS:
        return 'VEHICLE', 'BUS'
    elif object_type_ == ObjectType.CYCLIST or object_type_ == ObjectType.RIDERLESS_BICYCLE:
        return 'CYCLIST', 'BICYCLIST'
    elif object_type_ == ObjectType.MOTORCYCLIST:
        return 'CYCLIST', 'MOTORCYCLIST'
    elif object_type_ == ObjectType.PEDESTRIAN:
        return 'PEDESTRIAN', 'PEDESTRIAN'
    elif object_type_ == ObjectType.STATIC or object_type_ == ObjectType.BACKGROUND:
        return 'STATIC', 'UNKNOWN'
    elif object_type_ == ObjectType.CONSTRUCTION:
        return 'STATIC', 'CONSTRUCTION'
    elif object_type_ == ObjectType.UNKNOWN:
        return 'UNKNOWN', 'UNKNOWN'
    else:
        raise ValueError("Unknown object type: {}".format(object_type_))


def extract_object_tracks(raw_object_track_data_, scenario_length_):
    ret = dict()
    for object_track in raw_object_track_data_:
        object_id = object_track.track_id
        object_type = object_track.object_type
        object_states = object_state_template(scenario_length_)
        object_states['type'], object_states['subtype'] = get_object_type(object_type)
        object_states['phase'] = 'CONVERTED'
        for object_state in object_track.object_states:
            frame = int(object_state.timestep)
            object_states['state']['global_center'][frame] = (object_state.position[0], object_state.position[1], 0.0)
            object_states['state']['heading'][frame] = object_state.heading
            object_states['state']['valid'][frame] = True
        ret[object_id] = object_states
    return ret


def get_road_line_type(road_line_type):
    if road_line_type == LaneMarkType.DASH_SOLID_YELLOW:
        return 'ROAD_LINE', 'DASH_SOLID_YELLOW'
    elif road_line_type == LaneMarkType.DASH_SOLID_WHITE:
        return 'ROAD_LINE', 'DASH_SOLID_WHITE'
    elif road_line_type == LaneMarkType.DASHED_WHITE:
        return 'ROAD_LINE', 'DASHED_WHITE'
    elif road_line_type == LaneMarkType.DASHED_YELLOW:
        return 'ROAD_LINE', 'DASHED_YELLOW'
    elif road_line_type == LaneMarkType.DOUBLE_SOLID_YELLOW:
        return 'ROAD_LINE', 'DOUBLE_SOLID_YELLOW'
    elif road_line_type == LaneMarkType.DOUBLE_SOLID_WHITE:
        return 'ROAD_LINE', 'DOUBLE_SOLID_WHITE'
    elif road_line_type == LaneMarkType.DOUBLE_DASH_YELLOW:
        return 'ROAD_LINE', 'DOUBLE_DASH_YELLOW'
    elif road_line_type == LaneMarkType.DOUBLE_DASH_WHITE:
        return 'ROAD_LINE', 'DOUBLE_DASH_WHITE'
    elif road_line_type == LaneMarkType.SOLID_YELLOW:
        return 'ROAD_LINE', 'SOLID_YELLOW'
    elif road_line_type == LaneMarkType.SOLID_WHITE:
        return 'ROAD_LINE', 'SOLID_WHITE'
    elif road_line_type == LaneMarkType.SOLID_DASH_WHITE:
        return 'ROAD_LINE', 'SOLID_DASH_WHITE'
    elif road_line_type == LaneMarkType.SOLID_DASH_YELLOW:
        return 'ROAD_LINE', 'SOLID_DASH_YELLOW'
    elif road_line_type == LaneMarkType.SOLID_BLUE:
        return 'ROAD_LINE', 'SOLID_BLUE'
    elif road_line_type == LaneMarkType.NONE:
        return 'ROAD_LINE', 'VIRTUAL'
    elif road_line_type == LaneMarkType.UNKNOWN:
        return 'ROAD_LINE', 'UNKNOWN'
    else:
        raise ValueError("Unknown road line type: {}".format(road_line_type))


def get_road_lane_type(road_lane_type_):
    if road_lane_type_ == LaneType.VEHICLE:
        return 'ROAD_LANE', 'URBAN_VEHICLE'
    elif road_lane_type_ == LaneType.BUS:
        return 'ROAD_LANE', 'URBAN_BUS'
    elif road_lane_type_ == LaneType.BIKE:
        return 'ROAD_LANE', 'URBAN_BIKE'
    else:
        raise ValueError("Unknown road lane type: {}".format(road_lane_type_))


def extract_road_lines(road_line):
    road_line_info = dict()
    road_line_info['type'], road_line_info['subtype'] = get_road_line_type(road_line.mark_type)
    road_line_info['polyline'] = road_line.polyline
    road_line_info['phase'] = 'CONVERTED'
    return road_line_info


def extract_map_features(map_features):
    ret = {}

    drivable_areas = map_features.get_scenario_vector_drivable_areas()
    lane_segments = map_features.get_scenario_lane_segments()
    pedestrian_crossings = map_features.get_scenario_ped_crossings()

    for map_feature in lane_segments:
        left_road_line_id = str(map_feature.id) + '_1'
        right_road_line_id = str(map_feature.id) + '_2'
        left_road_line = extract_road_lines(map_feature.left_lane_marking)
        right_road_line = extract_road_lines(map_feature.right_lane_marking)

        ret[left_road_line_id] = left_road_line
        ret[right_road_line_id] = right_road_line

        road_lane_info = dict()

        road_lane_id = str(map_feature.id)
        road_lane_type = map_feature.lane_type
        road_lane_polyline = map_features.get_lane_segment_centerline(map_feature.id)

        road_lane_info['type'], road_lane_info['subtype'] = get_road_lane_type(road_lane_type)
        road_lane_info['polyline'] = road_lane_polyline
        road_lane_info['speed_limit'] = None
        road_lane_info['is_intersection'] = map_feature.is_intersection
        road_lane_info['phase'] = 'CONVERTED'

        road_lane_info['lane_info'] = dict()
        road_lane_info['lane_info']['entry_lane_ids'] = [str(id) for id in map_feature.predecessors]
        road_lane_info['lane_info']['exit_lane_ids'] = [str(id) for id in map_feature.successors]
        road_lane_info['lane_info']['left_lane_ids'] = [str(map_feature.left_neighbor_id)] if map_feature.left_neighbor_id else []
        road_lane_info['lane_info']['right_lane_ids'] = [str(map_feature.right_neighbor_id)] if map_feature.right_neighbor_id else []

        road_lane_info['boundary_info'] = dict()
        road_lane_info['boundary_info']['left_boundary_id'] = left_road_line_id
        road_lane_info['boundary_info']['right_boundary_id'] = right_road_line_id
        road_lane_info['boundary_info']['left_boundary_polyline'] = left_road_line['polyline']
        road_lane_info['boundary_info']['right_boundary_polyline'] = right_road_line['polyline']

        ret[road_lane_id] = road_lane_info

    for road_edge in drivable_areas:
        road_edge_info = dict()
        road_edge_id = str(road_edge.id)
        points = road_edge.area_boundary
        road_edge_polyline = np.array([[point.x, point.y, point.z] for point in points])

        road_edge_info['type'] = 'ROAD_EDGE'
        road_edge_info['subtype'] = 'UNKNOWN'
        road_edge_info['polyline'] = road_edge_polyline
        road_edge_info['phase'] = 'CONVERTED'

        ret[road_edge_id] = road_edge_info

    for crosswalk in pedestrian_crossings:
        crosswalk_info = dict()
        crosswalk_id = str(crosswalk.id)

        crosswalk_info['type'] = 'OTHER'
        crosswalk_info['subtype'] = 'CROSSWALK'
        crosswalk_info['polygon'] = crosswalk.polygon
        crosswalk_info['phase'] = 'CONVERTED'

        ret[crosswalk_id] = crosswalk_info

    return ret


def get_city_name(city_name):
    if city_name == 'palo-alto':
        return 'PALO_ALTO'
    elif city_name == 'austin':
        return 'AUSTIN'
    elif city_name == 'miami':
        return 'MIAMI'
    elif city_name == 'dearborn':
        return 'DEARBORN'
    elif city_name == 'washington-dc':
        return 'WASHINGTON_DC'
    elif city_name == 'pittsburgh':
        return 'PITTSBURGH'
    else:
        raise ValueError(f'Unknown city name: {city_name}')


def list_scenario_ids(raw_dataset_path):
    return ['tr_' + scenario_id for scenario_id in os.listdir(raw_dataset_path + '/train')] + \
           ['va_' + scenario_id for scenario_id in os.listdir(raw_dataset_path + '/val')] + \
           ['te_' + scenario_id for scenario_id in os.listdir(raw_dataset_path + '/test')]


def process_scenario(params):
    args, scenario_id = params
    a2md_file_dir = args.raw_dataset_path
    scenario = dict()

    if scenario_id[0:2] == 'tr':
        raw_data_path = a2md_file_dir + '/train/' + scenario_id[3:]
    elif scenario_id[0:2] == 'va':
        raw_data_path = a2md_file_dir + '/val/' + scenario_id[3:]
    elif scenario_id[0:2] == 'te':
        raw_data_path = a2md_file_dir + '/test/' + scenario_id[3:]
    else:
        raise ValueError("Unknown data path: {}".format(scenario_id))

    raw_object_track_data_path = Path(raw_data_path + '/scenario_' + scenario_id[3:] + '.parquet')
    raw_object_track_data = scenario_serialization.load_argoverse_scenario_parquet(raw_object_track_data_path)
    scenario_length = len(raw_object_track_data.timestamps_ns)
    converted_object_track_data = extract_object_tracks(raw_object_track_data.tracks, scenario_length)

    raw_map_feature_data_path = Path(raw_data_path + '/log_map_archive_' + scenario_id[3:] + '.json')
    raw_map_feature_data = ArgoverseStaticMap.from_json(raw_map_feature_data_path)
    converted_map_feature_data = extract_map_features(raw_map_feature_data)

    converted_dynamic_map_state_data = dict()

    extra_information = dict()
    extra_information['id'] = scenario_id[3:]
    extra_information['dataset'] = 'Argoverse 2'
    extra_information['version'] = args.raw_dataset_version
    extra_information['country'] = 'AMERICA'
    extra_information['city'] = get_city_name(raw_object_track_data.city_name)
    extra_information['location'] = None
    extra_information['sdc_id'] = 'AV'
    extra_information['scene_length'] = scenario_length
    extra_information['sampling_rate'] = 10
    extra_information['timestamp'] = raw_object_track_data.timestamps_ns / 1000000000

    scenario['object_tracks'] = converted_object_track_data
    scenario['map_features'] = converted_map_feature_data
    scenario['dynamic_map_states'] = converted_dynamic_map_state_data
    scenario['extra_information'] = extra_information

    save_scenario_pickle(scenario, args, scenario_id[3:])


def convert(args):
    scenario_ids = list_scenario_ids(args.raw_dataset_path)
    params = [(args, scenario_id) for scenario_id in scenario_ids]
    run_tasks(process_scenario, params, len(scenario_ids), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
