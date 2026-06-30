# Waymo Open
#   For v1.3.0 and higher

# Installation
#   pip install -r requirements.txt
#   Waymo Scenario protobuf is loaded from scenario_convertor.womd_proto.

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import numpy as np
import os
import tensorflow as tf

from scenario_convertor.utils import (
    add_common_converter_args,
    object_state_template,
    run_tasks,
    save_scenario_pickle,
    traffic_light_state_template,
)
from tqdm import tqdm
from scenario_convertor.womd_proto.pb2.scenario_pb2 import Scenario


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Waymo Open Motion Dataset (WOMD) Converter')
    return add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/waymo-open-motion-dataset-v1.3.0',
        default_subset_name='waymo-open',
        default_raw_dataset_version='v1.3.0',
        raw_dataset_help='The file directory for WOMD',
    )


def get_object_type_and_subtype(object_type_):
    if object_type_ == 0 or object_type_ == 4:
        return 'UNKNOWN', 'UNKNOWN'
    elif object_type_ == 1:
        return 'VEHICLE', 'UNKNOWN'
    elif object_type_ == 2:
        return 'PEDESTRIAN', 'PEDESTRIAN'
    elif object_type_ == 3:
        return 'CYCLIST', 'UNKNOWN'
    else:
        raise ValueError("Unknown object type: {}".format(object_type_))


def extract_object_tracks(raw_object_track_data_, scenario_length_, sdc_track_index_):
    ret = dict()
    for object_track in raw_object_track_data_:
        object_id = str(object_track.id)

        object_states = object_state_template(scenario_length_)

        object_type = object_track.object_type
        object_states['type'], object_states['subtype'] = get_object_type_and_subtype(object_type)
        object_states['phase'] = 'CONVERTED'
        if scenario_length_ != len(object_track.states):
            return ret, str(raw_object_track_data_[sdc_track_index_].id)

        for frame, object_state in enumerate(object_track.states):
            object_states['state']['global_center'][frame] = (object_state.center_x, object_state.center_y, object_state.center_z)
            object_states['state']['heading'][frame] = object_state.heading
            object_states['state']['size'][frame] = (object_state.length, object_state.width, object_state.height)
            object_states['state']['valid'][frame] = object_state.valid
        ret[object_id] = object_states
    return ret, str(raw_object_track_data_[sdc_track_index_].id)


def extract_polyline_or_polygon(poly):
    poly_x = [point.x for point in poly]
    poly_y = [point.y for point in poly]
    poly_z = [point.z for point in poly]
    road_lane_polyline = np.stack((poly_x, poly_y, poly_z), axis=1).astype("float32")
    return road_lane_polyline


def get_road_edge_type_and_subtype(road_edge_type_):
    if road_edge_type_ == 0:
        return 'ROAD_EDGE', 'UNKNOWN'
    elif road_edge_type_ == 1:
        return 'ROAD_EDGE', 'BOUNDARY'
    elif road_edge_type_ == 2:
        return 'ROAD_EDGE', 'MEDIAN'
    else:
        raise ValueError("Unknown road edge type: {}".format(road_edge_type_))


def get_road_line_type_and_subtype(road_line_type_):
    if road_line_type_ == 0:
        return 'ROAD_LINE', 'UNKNOWN'
    elif road_line_type_ == 1:
        return 'ROAD_LINE', 'DASHED_WHITE'
    elif road_line_type_ == 2:
        return 'ROAD_LINE', 'SOLID_WHITE'
    elif road_line_type_ == 3:
        return 'ROAD_LINE', 'DOUBLE_SOLID_WHITE'
    elif road_line_type_ == 4:
        return 'ROAD_LINE', 'DASHED_YELLOW'
    elif road_line_type_ == 5:
        return 'ROAD_LINE', 'DOUBLE_DASH_YELLOW'
    elif road_line_type_ == 6:
        return 'ROAD_LINE', 'SOLID_YELLOW'
    elif road_line_type_ == 7:
        return 'ROAD_LINE', 'DOUBLE_SOLID_YELLOW'
    elif road_line_type_ == 8:
        return 'ROAD_LINE', 'SOLID_DASH_YELLOW'
    else:
        raise ValueError("Unknown road line type: {}".format(road_line_type_))


def get_road_lane_type_and_subtype(road_lane_type_):
    if road_lane_type_ == 0:
        return 'ROAD_LANE', 'URBAN_UNKNOWN'
    elif road_lane_type_ == 1:
        return 'ROAD_LANE', 'FREEWAY'
    elif road_lane_type_ == 2:
        return 'ROAD_LANE', 'URBAN_VEHICLE'
    elif road_lane_type_ == 3:
        return 'ROAD_LANE', 'URBAN_BIKE'
    else:
        raise ValueError("Unknown road lane type: {}".format(road_lane_type_))


def get_left_or_right_lane_id(left_or_right_lane_info):
    for lane_info in left_or_right_lane_info:
        return [lane_info.feature_id]
    return []


def extract_map_features(raw_map_feature_data_):
    ret = dict()

    for map_feature in raw_map_feature_data_:
        road_lane_info = dict()
        road_line_info = dict()
        road_edge_info = dict()
        other_info = dict()
        if map_feature.HasField('lane'):
            road_lane = map_feature.lane
            road_lane_id = str(map_feature.id)
            road_lane_type = road_lane.type
            road_lane_polyline = road_lane.polyline

            left_lane_info = road_lane.left_neighbors
            right_lane_info = road_lane.right_neighbors

            road_lane_info['type'], road_lane_info['subtype'] = get_road_lane_type_and_subtype(road_lane_type)
            road_lane_info['polyline'] = extract_polyline_or_polygon(road_lane_polyline)
            road_lane_info['speed_limit'] = 0.44704 * road_lane.speed_limit_mph

            road_lane_info['lane_info'] = dict()
            road_lane_info['lane_info']['entry_lane_ids'] = [str(lane_id) for lane_id in road_lane.entry_lanes]
            road_lane_info['lane_info']['exit_lane_ids'] = [str(lane_id) for lane_id in road_lane.exit_lanes]
            road_lane_info['lane_info']['left_lane_ids'] = get_left_or_right_lane_id(left_lane_info)
            road_lane_info['lane_info']['right_lane_ids'] = get_left_or_right_lane_id(right_lane_info)

            road_lane_info['boundary_info'] = dict()
            road_lane_info['boundary_info']['left_boundary_id'] = None
            road_lane_info['boundary_info']['right_boundary_id'] = None
            road_lane_info['boundary_info']['left_boundary_polyline'] = np.array([])
            road_lane_info['boundary_info']['right_boundary_polyline'] = np.array([])

            road_lane_info['phase'] = 'CONVERTED'
            ret[road_lane_id] = road_lane_info

        elif map_feature.HasField('road_line'):
            road_line = map_feature.road_line
            road_line_id = str(map_feature.id)
            road_line_type = road_line.type
            road_line_polyline = road_line.polyline

            road_line_info['type'], road_line_info['subtype'] = get_road_line_type_and_subtype(road_line_type)
            road_line_info['polyline'] = extract_polyline_or_polygon(road_line_polyline)
            road_line_info['phase'] = 'CONVERTED'

            ret[road_line_id] = road_line_info

        elif map_feature.HasField('road_edge'):
            road_edge = map_feature.road_edge
            road_edge_id = str(map_feature.id)
            road_edge_type = road_edge.type
            road_edge_polyline = road_edge.polyline

            road_edge_info['type'], road_edge_info['subtype'] = get_road_edge_type_and_subtype(road_edge_type)
            road_edge_info['polyline'] = extract_polyline_or_polygon(road_edge_polyline)
            road_edge_info['phase'] = 'CONVERTED'

            ret[road_edge_id] = road_edge_info

        elif map_feature.HasField('stop_sign'):
            other = map_feature.stop_sign
            other_id = str(map_feature.id)

            other_info = dict()
            other_info['type'] = 'OTHER'
            other_info['subtype'] = 'STOP_SIGN'
            other_info['lane'] = [str(lane_id) for lane_id in other.lane]
            other_info['position'] = np.array([other.position.x, other.position.y, other.position.z])
            other_info['phase'] = 'CONVERTED'

            ret[other_id] = other_info

        elif map_feature.HasField('crosswalk'):
            other = map_feature.crosswalk
            other_id = str(map_feature.id)
            other_polygon = other.polygon

            other_info['type'] = 'OTHER'
            other_info['subtype'] = 'CROSSWALK'
            other_info['polygon'] = extract_polyline_or_polygon(other_polygon)
            other_info['phase'] = 'CONVERTED'

            ret[other_id] = other_info

        elif map_feature.HasField('speed_bump'):
            other = map_feature.speed_bump
            other_id = str(map_feature.id)
            other_polygon = other.polygon

            other_info['type'] = 'OTHER'
            other_info['subtype'] = 'SPEED_BUMP'
            other_info['polygon'] = extract_polyline_or_polygon(other_polygon)
            other_info['phase'] = 'CONVERTED'

            ret[other_id] = other_info

        else:
            continue

    return ret


def get_traffic_light_type_and_subtype(traffic_light_type_):
    if traffic_light_type_ == 0:
        return 'TRAFFIC_LIGHT', 'UNKNOWN'
    elif traffic_light_type_ == 1 or traffic_light_type_ == 2 or traffic_light_type_ == 3:
        return 'TRAFFIC_LIGHT', 'ARROW'
    elif traffic_light_type_ == 4 or traffic_light_type_ == 5 or traffic_light_type_ == 6 or \
            traffic_light_type_ == 7 or traffic_light_type_ == 8:
        return 'TRAFFIC_LIGHT', 'CIRCLE'
    else:
        raise ValueError("Unknown traffic light type: {}".format(traffic_light_type_))


def get_traffic_light_state(traffic_light_type_):
    if traffic_light_type_ == 0:
        return 'UNKNOWN'
    elif traffic_light_type_ == 1 or traffic_light_type_ == 4:
        return 'RED'
    elif traffic_light_type_ == 2 or traffic_light_type_ == 5:
        return 'YELLOW'
    elif traffic_light_type_ == 3 or traffic_light_type_ == 6:
        return 'GREEN'
    elif traffic_light_type_ == 7 or traffic_light_type_ == 8:
        return 'FLASHING'
    else:
        raise ValueError("Unknown traffic light state: {}".format(traffic_light_type_))


def extract_dynamic_map_states(raw_dynamic_map_state_data_, scenario_length_):
    ret = dict()

    for frame, dynamic_map_state in enumerate(raw_dynamic_map_state_data_):
        traffic_lights_state = dynamic_map_state.lane_states

        for traffic_light_state in traffic_lights_state:
            traffic_light_id = str(traffic_light_state.lane)
            traffic_light_type = traffic_light_state.state

            if traffic_light_id not in ret:
                ret[traffic_light_id] = traffic_light_state_template(traffic_light_id, scenario_length_)

            ret[traffic_light_id]['type'], ret[traffic_light_id]['subtype'] = get_traffic_light_type_and_subtype(traffic_light_type)
            ret[traffic_light_id]['state']['color'][frame] = get_traffic_light_state(traffic_light_type)
            ret[traffic_light_id]['state']['valid'][frame] = True
            ret[traffic_light_id]['stop_point'][0] = traffic_light_state.stop_point.x
            ret[traffic_light_id]['stop_point'][1] = traffic_light_state.stop_point.y
            ret[traffic_light_id]['stop_point'][2] = traffic_light_state.stop_point.z
            ret[traffic_light_id]['controlled_lane'] = str(traffic_light_state.lane)
            ret[traffic_light_id]['phase'] = 'CONVERTED'

    return ret


def process_scenario(params):
    args, file_name = params
    segments = tf.data.TFRecordDataset(f'{args.raw_dataset_path}/{file_name}', compression_type='')
    scenario_ = dict()
    for segment in segments:
        scenario = Scenario()
        scenario.ParseFromString(segment.numpy())

        scenario_id = scenario.scenario_id
        scenario_length = len(scenario.timestamps_seconds)

        sdc_track_index = scenario.sdc_track_index
        raw_object_track_data = scenario.tracks
        converted_object_track_data, sdc_id = extract_object_tracks(raw_object_track_data, scenario_length, sdc_track_index)

        if len(converted_object_track_data) == 0:
            continue

        raw_map_feature_data = scenario.map_features
        converted_map_feature_data = extract_map_features(raw_map_feature_data)

        raw_dynamic_map_state_data = scenario.dynamic_map_states
        converted_dynamic_map_state_data = extract_dynamic_map_states(raw_dynamic_map_state_data, scenario_length)

        extra_information = dict()
        extra_information['id'] = scenario_id
        extra_information['dataset'] = 'Waymo Open'
        extra_information['version'] = args.raw_dataset_version
        extra_information['country'] = 'AMERICA'
        extra_information['city'] = None
        extra_information['location'] = None
        extra_information['sdc_id'] = sdc_id
        extra_information['scene_length'] = scenario_length
        extra_information['sampling_rate'] = 10
        extra_information['timestamp'] = np.array(scenario.timestamps_seconds)
        extra_information['init_time'] = None

        scenario_['object_tracks'] = converted_object_track_data
        scenario_['map_features'] = converted_map_feature_data
        scenario_['dynamic_map_states'] = converted_dynamic_map_state_data
        scenario_['extra_information'] = extra_information

        save_scenario_pickle(scenario_, args, scenario_id)


def convert(args):
    file_path = args.raw_dataset_path
    file_name_ls = os.listdir(file_path)
    params = [(args, file_name) for file_name in file_name_ls]
    run_tasks(process_scenario, params, len(file_name_ls), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
