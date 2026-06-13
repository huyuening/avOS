# V2X-Seq Trajectory Forecasting Dataset -> V2X-Seq Scenario Dataset (VSSD)

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import json
import numpy as np
import os
import pandas as pd
import re

from tqdm import tqdm
from utils import add_common_converter_args, object_state_template, run_tasks, save_scenario_pickle, traffic_light_state_template


def build_arg_parser():
    parser = argparse.ArgumentParser(description='V2X-Seq Motion Prediction Dataset (V2X-Seq) Converter')
    return add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/v2x-seq-motion-dataset',
        default_subset_name='v2x-seq',
        default_raw_dataset_version='v1.0',
        raw_dataset_help='The file directory for V2X-Seq',
    )


def list_scenario_ids(raw_dataset_path):
    return ['t_' + i[0:-4] for i in os.listdir(raw_dataset_path+'/cooperative-vehicle-infrastructure/vehicle-trajectories/train')] + \
           ['v_' + i[0:-4] for i in os.listdir(raw_dataset_path+'/cooperative-vehicle-infrastructure/vehicle-trajectories/val')] + \
           ['_t_' + i[0:-4] for i in os.listdir(raw_dataset_path+'/single-vehicle/trajectories/train')] + \
           ['_v_' + i[0:-4] for i in os.listdir(raw_dataset_path+'/single-vehicle/trajectories/val')] + \
           ['__t_' + i[0:-4] for i in os.listdir(raw_dataset_path+'/single-infrastructure/trajectories/train')] + \
           ['__v_' + i[0:-4] for i in os.listdir(raw_dataset_path+'/single-infrastructure/trajectories/val')]


MAX_DISTANCE = 150


def get_road_lane_type_and_subtype(road_lane_type):
    if road_lane_type == 'CITY_DRIVING' or road_lane_type == 'LEFT_TURN_WAITING_ZONE' or \
       road_lane_type == 'ROUNDABOUT' or road_lane_type == 'EMERGENCY_LANE':
        return 'ROAD_LANE', 'URBAN_VEHICLE'
    elif road_lane_type == 'BIKING':
        return 'ROAD_LANE', 'URBAN_BIKE'
    else:
        raise ValueError("Unknown road lane type: {}".format(road_lane_type))


def extract_road_lines(road_line, is_intersection):
    road_line_polyline = np.array([eval(point) for point in road_line])
    road_line_polyline = np.hstack((road_line_polyline, np.zeros((road_line_polyline.shape[0], 1))))

    road_line_info = dict()
    road_line_info['type'] = 'ROAD_LINE'
    road_line_info['subtype'] = 'VIRTUAL' if is_intersection else 'UNKNOWN'
    road_line_info['polyline'] = road_line_polyline
    road_line_info['phase'] = 'CONVERTED'
    return road_line_info


def extract_map_features(map_feature_data, scenario_center_):
    ret = {}

    road_lanes = map_feature_data['LANE']
    stop_lines = map_feature_data['STOPLINE']
    crosswalks = map_feature_data['CROSSWALK']

    for road_lane_id in road_lanes:
        road_lane = road_lanes[road_lane_id]
        left_road_line_id = road_lane_id + '_1'
        right_road_line_id = road_lane_id + '_2'

        left_road_line = extract_road_lines(road_lane['left_boundary'], road_lane['is_intersection'])
        right_road_line = extract_road_lines(road_lane['right_boundary'], road_lane['is_intersection'])

        if np.linalg.norm(left_road_line['polyline'][int(len(left_road_line['polyline']) / 2)][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE and \
           np.linalg.norm(right_road_line['polyline'][int(len(right_road_line['polyline']) / 2)][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE:
            ret[left_road_line_id] = left_road_line
            ret[right_road_line_id] = right_road_line

            road_lane_polyline = np.array([eval(point) for point in road_lane['centerline']])
            road_lane_polyline = np.hstack((road_lane_polyline, np.zeros((road_lane_polyline.shape[0], 1))))

            road_lane_info = dict()
            road_lane_info['type'], road_lane_info['subtype'] = get_road_lane_type_and_subtype(road_lane['lane_type'])
            road_lane_info['polyline'] = road_lane_polyline
            road_lane_info['speed_limit'] = None
            road_lane_info['is_intersection'] = road_lane['is_intersection']
            road_lane_info['phase'] = 'CONVERTED'

            road_lane_info['lane_info'] = dict()
            road_lane_info['lane_info']['entry_lane_ids'] = road_lane['successors']
            road_lane_info['lane_info']['exit_lane_ids'] = road_lane['predecessors']
            road_lane_info['lane_info']['left_lane_ids'] = [road_lane['l_neighbor_id']] if road_lane['l_neighbor_id'] != 'None' else []
            road_lane_info['lane_info']['right_lane_ids'] = [road_lane['r_neighbor_id']] if road_lane['r_neighbor_id'] != 'None' else []

            road_lane_info['boundary_info'] = dict()
            road_lane_info['boundary_info']['left_boundary_id'] = left_road_line_id
            road_lane_info['boundary_info']['right_boundary_id'] = right_road_line_id
            road_lane_info['boundary_info']['left_boundary_polyline'] = left_road_line['polyline']
            road_lane_info['boundary_info']['right_boundary_polyline'] = right_road_line['polyline']

            ret[str(road_lane_id)] = road_lane_info

    for stop_line_id in stop_lines:
        stop_line = stop_lines[stop_line_id]
        stop_line_polyline = np.array([eval(point) for point in stop_line['centerline']])
        stop_line_polyline = np.hstack((stop_line_polyline, np.zeros((stop_line_polyline.shape[0], 1))))

        stop_line_info = dict()
        stop_line_info['type'] = 'OTHER'
        stop_line_info['subtype'] = 'STOP_LINE'
        stop_line_info['polyline'] = stop_line_polyline
        stop_line_info['phase'] = 'CONVERTED'

        if np.linalg.norm(stop_line_info['polyline'][int(len(stop_line_info['polyline']) / 2)][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE:
            ret[str(stop_line_id)] = stop_line_info

    for crosswalk_id in crosswalks:
        crosswalk = crosswalks[crosswalk_id]
        crosswalk_polygon = np.array([eval(point) for point in crosswalk['polygon']])
        crosswalk_polygon = np.hstack((crosswalk_polygon, np.zeros((crosswalk_polygon.shape[0], 1))))

        crosswalk_info = dict()
        crosswalk_info['type'] = 'OTHER'
        crosswalk_info['subtype'] = 'CROSSWALK'
        crosswalk_info['polygon'] = crosswalk_polygon
        crosswalk_info['phase'] = 'CONVERTED'

        if np.linalg.norm(crosswalk_info['polygon'][int(len(crosswalk_info['polygon']) / 2)][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE:
            ret[str(crosswalk_id)] = crosswalk_info

    return ret


def extract_numbers(text):
    numbers = re.findall(r'\d+', text)
    return numbers


def get_object_type_and_subtype(object_type_, object_subtype_):
    if object_type_ == 'VEHICLE' and (object_subtype_ == 'CAR' or pd.isna(object_subtype_)):
        return 'VEHICLE', 'CAR'
    elif object_type_ == 'VEHICLE' and object_subtype_ == 'TRUCK':
        return 'VEHICLE', 'TRUCK'
    elif object_type_ == 'VEHICLE' and object_subtype_ == 'VAN':
        return 'VEHICLE', 'VAN'
    elif object_type_ == 'VEHICLE' and object_subtype_ == 'BUS':
        return 'VEHICLE', 'BUS'
    elif object_type_ == 'VEHICLE' and object_subtype_ == 'UNKNOWN':
        return 'VEHICLE', 'UNKNOWN'
    elif object_type_ == 'BICYCLE' and object_subtype_ == 'CYCLIST':
        return 'CYCLIST', 'BICYCLIST'
    elif object_type_ == 'BICYCLE' and object_subtype_ == 'TRICYCLIST':
        return 'CYCLIST', 'TRICYCLIST'
    elif object_type_ == 'BICYCLE' and object_subtype_ == 'MOTORCYCLIST':
        return 'CYCLIST', 'MOTORCYCLIST'
    elif object_type_ == 'BICYCLE' and object_subtype_ == 'UNKNOWN':
        return 'CYCLIST', 'UNKNOWN'
    elif object_type_ == 'PEDESTRIAN' and (object_subtype_ == 'PEDESTRIAN' or object_subtype_ == 'UNKNOWN'):
        return 'PEDESTRIAN', 'PEDESTRIAN'
    elif (object_type_ == 'BICYCLE' and (object_subtype_ == 'PEDESTRIAN' or object_subtype_ == 'VAN' or object_subtype_ == 'CAR' or object_subtype_ == 'TRAFFICCONE' or object_subtype_ == 'TRUCK' or object_subtype_ == 'UNKNOWN_UNMOVABLE')) or \
         (object_type_ == 'VEHICLE' and (object_subtype_ == 'MOTORCYCLIST' or object_subtype_ == 'TRICYCLIST' or object_subtype_ == 'UNKNOWN_UNMOVABLE')) or \
         (object_type_ == 'PEDESTRIAN' and (object_subtype_ == 'CYCLIST' or object_subtype_ == 'MOTORCYCLIST' or object_subtype_ == 'TRAFFICCONE')):
        return 'UNKNOWN', 'UNKNOWN'
    else:
        raise ValueError("Unknown object type: {}, subtype: {}".format(object_type_, object_subtype_))


def extract_object_tracks(tracks, scenario_length_, timestamps):
    ret = dict()
    if np.sum(tracks['tag'] == 'AV'):
        sdc_id = str(tracks[tracks['tag'] == 'AV'].reset_index(drop=True).at[0, 'id'])
    else:
        sdc_id = None
    for object_id in tracks['id'].unique():
        object_type = tracks[tracks['id'] == object_id]['type'].iloc[0]
        object_subtype = tracks[tracks['id'] == object_id]['sub_type'].iloc[0]
        object_state = object_state_template(scenario_length_)
        object_state['type'], object_state['subtype'] = get_object_type_and_subtype(object_type, object_subtype)
        object_state['phase'] = 'CONVERTED'
        for frame, timestamp in enumerate(timestamps):
            if not tracks[(tracks['timestamp'] == timestamp) & (tracks['id'] == object_id)].empty:
                object_state['state']['global_center'][frame] = np.array([tracks[(tracks['timestamp'] == timestamp) & (tracks['id'] == object_id)]['x'].iloc[0],
                                                                          tracks[(tracks['timestamp'] == timestamp) & (tracks['id'] == object_id)]['y'].iloc[0],
                                                                          0.0])
                if str(object_id) == sdc_id:
                    object_state['state']['size'][frame] = (4.76, 1.87, 1.65)
                else:
                    object_state['state']['size'][frame] = (tracks[(tracks['timestamp'] == timestamp) & (tracks['id'] == object_id)]['length'].iloc[0],
                                                            tracks[(tracks['timestamp'] == timestamp) & (tracks['id'] == object_id)]['width'].iloc[0],
                                                            tracks[(tracks['timestamp'] == timestamp) & (tracks['id'] == object_id)]['height'].iloc[0])
                object_state['state']['heading'][frame] = tracks[(tracks['timestamp'] == timestamp) & (tracks['id'] == object_id)]['theta'].iloc[0]
                object_state['state']['valid'][frame] = True
        ret[str(object_id)] = object_state
    return ret, sdc_id


def extract_dynamic_map_states(dynamic_map_state_data, scenario_length_, timestamps):
    ret = dict()
    for lane_id in dynamic_map_state_data['lane_id'].unique():
        traffic_light = dynamic_map_state_data[dynamic_map_state_data['lane_id'] == lane_id].reset_index(drop=True)

        dynamic_map_state = traffic_light_state_template(lane_id, scenario_length_)
        dynamic_map_state['id'] = traffic_light.at[0, 'lane_id']
        dynamic_map_state['stop_point'][0] = traffic_light.at[0, 'x']
        dynamic_map_state['stop_point'][1] = traffic_light.at[0, 'y']
        dynamic_map_state['type'], dynamic_map_state['subtype'] = 'TRAFFIC_LIGHT', 'UNKNOWN'
        for frame, timestamp in enumerate(timestamps):
            if not traffic_light[traffic_light['timestamp'] == timestamp].empty:
                state = traffic_light[traffic_light['timestamp'] == timestamp].reset_index(drop=True)
                dynamic_map_state['state']['color'][frame] = state['color_1'].iloc[0]
                dynamic_map_state['state']['valid'][frame] = True
        dynamic_map_state['controlled_lane'] = None
        dynamic_map_state['phase'] = 'CONVERTED'

        ret[str(lane_id)] = dynamic_map_state
    return ret


def process_scenario(params):
    args, scenario_id = params
    vsmd_file_dir = args.raw_dataset_path
    scenario = dict()
    if scenario_id[0] == 't':
        object_track_data_path = vsmd_file_dir + '/cooperative-vehicle-infrastructure/vehicle-trajectories/train/' + scenario_id[2:] + '.csv'
    elif scenario_id[0] == 'v':
        object_track_data_path = vsmd_file_dir + '/cooperative-vehicle-infrastructure/vehicle-trajectories/val/' + scenario_id[2:] + '.csv'
    elif scenario_id[0:2] == '_t':
        object_track_data_path = vsmd_file_dir + '/single-vehicle/trajectories/train/' + scenario_id[3:] + '.csv'
    elif scenario_id[0:2] == '_v':
        object_track_data_path = vsmd_file_dir + '/single-vehicle/trajectories/val/' + scenario_id[3:] + '.csv'
    elif scenario_id[0:3] == '__t':
        object_track_data_path = vsmd_file_dir + '/single-infrastructure/trajectories/train/' + scenario_id[4:] + '.csv'
    elif scenario_id[0:3] == '__v':
        object_track_data_path = vsmd_file_dir + '/single-infrastructure/trajectories/val/' + scenario_id[4:] + '.csv'
    else:
        raise ValueError("Unknown scenario id: ".format(scenario_id))

    raw_object_track_data = pd.read_csv(object_track_data_path)
    timestamps = np.array(raw_object_track_data['timestamp'].unique())
    scenario_length = len(timestamps)
    converted_object_track_data, sdc_id = extract_object_tracks(raw_object_track_data, scenario_length, timestamps)

    if sdc_id:
        scenario_center = np.array([converted_object_track_data[sdc_id]['state']['global_center'][int(scenario_length/2), 0],
                                    converted_object_track_data[sdc_id]['state']['global_center'][int(scenario_length/2), 1],
                                    converted_object_track_data[sdc_id]['state']['global_center'][int(scenario_length/2), 2]])
    else:
        fake_sdc_id = list(converted_object_track_data.keys())[0]
        scenario_center = np.array([converted_object_track_data[fake_sdc_id]['state']['global_center'][int(scenario_length/2), 0],
                                    converted_object_track_data[fake_sdc_id]['state']['global_center'][int(scenario_length/2), 1],
                                    converted_object_track_data[fake_sdc_id]['state']['global_center'][int(scenario_length/2), 2]])

    intersection_id = raw_object_track_data['intersect_id'][0]
    numbers = extract_numbers(intersection_id)

    if int(numbers[0]) < 1 or int(numbers[0]) > 28:
        return

    map_feature_data_path = vsmd_file_dir + '/maps/' + 'yizhuang_hdmap' + numbers[0] + '.json'
    with open(map_feature_data_path, 'r', encoding='utf-8') as file:
        raw_map_feature_data = json.load(file)
    converted_map_feature_data = extract_map_features(raw_map_feature_data, scenario_center)

    if scenario_id[0] == 't':
        dynamic_map_state_data_path = vsmd_file_dir + '/cooperative-vehicle-infrastructure/traffic-light/train/' + scenario_id[2:] + '.csv'
    elif scenario_id[0] == 'v':
        dynamic_map_state_data_path = vsmd_file_dir + '/cooperative-vehicle-infrastructure/traffic-light/val/' + scenario_id[2:] + '.csv'
    elif scenario_id[0] == '_':
        dynamic_map_state_data_path = ''
    else:
        raise ValueError("Unknown scenario id: ".format(scenario_id))
    if scenario_id[0] != '_':
        raw_dynamic_map_state_data = pd.read_csv(dynamic_map_state_data_path)
        converted_dynamic_map_state_data = extract_dynamic_map_states(raw_dynamic_map_state_data, scenario_length, timestamps)
    else:
        converted_dynamic_map_state_data = dict()

    extra_information = dict()
    extra_information['id'] = scenario_id
    extra_information['dataset'] = 'V2X-Seq'
    extra_information['version'] = args.raw_dataset_version
    extra_information['country'] = 'CHINA'
    extra_information['city'] = 'BEIJING'
    extra_information['location'] = None
    extra_information['sdc_id'] = sdc_id
    extra_information['scene_length'] = scenario_length
    extra_information['sampling_rate'] = 10
    extra_information['timestamp'] = timestamps - timestamps[0]
    extra_information['init_time'] = timestamps[0]

    scenario['object_tracks'] = converted_object_track_data
    scenario['map_features'] = converted_map_feature_data
    scenario['dynamic_map_states'] = converted_dynamic_map_state_data
    scenario['extra_information'] = extra_information

    save_scenario_pickle(scenario, args, scenario_id)


def convert(args):
    scenario_ids = list_scenario_ids(args.raw_dataset_path)
    params = [(args, scenario_id) for scenario_id in scenario_ids]
    run_tasks(process_scenario, params, len(scenario_ids), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
