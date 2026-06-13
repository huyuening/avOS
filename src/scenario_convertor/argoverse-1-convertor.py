#  Argoverse 1 Motion Forecasting Dataset

# Information
# object_tracks:        yes (csv)
#   object_type: AV, AGENT, OTHERS
#
# map_features:         yes (npy, json, xml)

# Installation
#   No special requirements

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm
from utils import add_common_converter_args, object_state_template, run_tasks, save_scenario_pickle


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Argoverse 1 Motion Forecasting Dataset Converter')
    return add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/argoverse-1-motion-dataset-v1.1',
        default_subset_name='argoverse-1',
        default_raw_dataset_version='1.1',
        raw_dataset_help='The file directory for A1MD',
    )

MAX_DISTANCE = 150


def get_polyline(road_element_ref_ids_, polys_):
    road_element_polyline = np.array([polys_[road_element_ref_id] for road_element_ref_id in road_element_ref_ids_])
    road_element_polyline = np.hstack((road_element_polyline, np.zeros((road_element_polyline.shape[0], 1))))
    return road_element_polyline


def extract_map_features(raw_map_feature_data_, polys_, scenario_center_):
    ret = dict()
    for way in raw_map_feature_data_.findall('way'):
        road_lane_id = way.attrib['lane_id']
        road_lane_info = dict()

        road_lane_ref_ids = [node.attrib['ref'] for node in way.findall('nd')]
        road_lane_polyline = get_polyline(road_lane_ref_ids, polys_)

        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag') if tag.attrib['k'] != 'predecessor' and tag.attrib['k'] != 'successor'}

        road_lane_info['type'] = 'ROAD_LANE'
        road_lane_info['subtype'] = 'UNKNOWN'
        road_lane_info['polyline'] = road_lane_polyline
        road_lane_info['speed_limit'] = None
        road_lane_info['is_intersection'] = tags['is_intersection']
        road_lane_info['phase'] = 'CONVERTED'

        road_lane_info['lane_info'] = dict()
        road_lane_info['lane_info']['entry_lane_ids'] = [tag.attrib['v'] for tag in way.findall('tag') if tag.attrib['k'] == 'predecessor']
        road_lane_info['lane_info']['exit_lane_ids'] = [tag.attrib['v'] for tag in way.findall('tag') if tag.attrib['k'] == 'successor']
        road_lane_info['lane_info']['left_lane_ids'] = [tags['l_neighbor_id']] if tags['l_neighbor_id'] != 'None' else []
        road_lane_info['lane_info']['right_lane_ids'] = [tags['r_neighbor_id']] if tags['r_neighbor_id'] != 'None' else []

        road_lane_info['boundary_info'] = dict()
        road_lane_info['boundary_info']['left_boundary_id'] = None
        road_lane_info['boundary_info']['right_boundary_id'] = None
        road_lane_info['boundary_info']['left_boundary_polyline'] = np.array([])
        road_lane_info['boundary_info']['right_boundary_polyline'] = np.array([])

        if np.linalg.norm(road_lane_polyline[0][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE:
            ret[road_lane_id] = road_lane_info

    return ret


def get_object_type(object_type_):
    if object_type_ == 'AV':
        return 'VEHICLE', 'CAR'
    elif object_type_ == 'OTHERS' or object_type_ == 'AGENT':
        return 'UNKNOWN', 'UNKNOWN'
    else:
        raise ValueError("Unknown object type: {}".format(object_type_))


def extract_object_tracks(raw_object_track_data_, scenario_length_):
    ret = dict()
    for object_id in raw_object_track_data_['TRACK_ID'].unique():
        object_states = object_state_template(scenario_length_)
        object_type = raw_object_track_data_[raw_object_track_data_['TRACK_ID'] == object_id].iloc[0]['OBJECT_TYPE']
        object_states['type'], object_states['subtype'] = get_object_type(object_type)
        object_states['phase'] = 'CONVERTED'
        for frame, timestamp in enumerate(raw_object_track_data_['TIMESTAMP'].unique()):
            if not raw_object_track_data_[(raw_object_track_data_['TIMESTAMP'] == timestamp) & (raw_object_track_data_['TRACK_ID'] == object_id)].empty:
                object_states['state']['global_center'][frame] = (raw_object_track_data_[(raw_object_track_data_['TIMESTAMP'] == timestamp) & (raw_object_track_data_['TRACK_ID'] == object_id)].iloc[0]['X'],
                                                                  raw_object_track_data_[(raw_object_track_data_['TIMESTAMP'] == timestamp) & (raw_object_track_data_['TRACK_ID'] == object_id)].iloc[0]['Y'],
                                                                  0.0)
                object_states['state']['valid'][frame] = True
        ret[object_id] = object_states
    return ret


def list_scenario_ids(raw_dataset_path):
    return ['t_' + os.path.splitext(scenario_id)[0] for scenario_id in os.listdir(raw_dataset_path + '/train/data')] + \
           ['v_' + os.path.splitext(scenario_id)[0] for scenario_id in os.listdir(raw_dataset_path + '/val/data')]


def process_scenario(params):
    args, scenario_id = params
    a1md_file_dir = args.raw_dataset_path
    scenario = dict()

    if scenario_id[0] == 't':
        raw_object_track_data_path = a1md_file_dir + '/train/data/' + scenario_id[2:] + '.csv'
    elif scenario_id[0] == 'v':
        raw_object_track_data_path = a1md_file_dir + '/val/data/' + scenario_id[2:] + '.csv'
    else:
        raise ValueError("Unknown data path: {}".format(scenario_id))
    raw_object_track_data = pd.read_csv(raw_object_track_data_path)
    scenario_length = len(raw_object_track_data['TIMESTAMP'].unique())
    converted_object_track_data = extract_object_tracks(raw_object_track_data, scenario_length)

    scenario_center = converted_object_track_data['00000000-0000-0000-0000-000000000000']['state']['global_center'][int(scenario_length / 2)]

    if raw_object_track_data.at[0, 'CITY_NAME'] == 'PIT':
        raw_map_feature_data_path = a1md_file_dir + '/map_files/pruned_argoverse_PIT_10314_vector_map.xml'
    elif raw_object_track_data.at[0, 'CITY_NAME'] == 'MIA':
        raw_map_feature_data_path = a1md_file_dir + '/map_files/pruned_argoverse_MIA_10316_vector_map.xml'
    else:
        raise ValueError("Unknown city name: {}".format(raw_object_track_data.at[0, 'CITY_NAME']))

    raw_map_feature_data = ET.parse(raw_map_feature_data_path).getroot()

    polys = {}
    for node in raw_map_feature_data.findall('node'):
        node_id = node.attrib['id']
        node_x, node_y = (float(node.attrib['x']), float(node.attrib['y']))
        polys[node_id] = (node_x, node_y)

    converted_map_feature_data = extract_map_features(raw_map_feature_data, polys, scenario_center)

    converted_dynamic_map_state_data = dict()

    extra_information = dict()
    extra_information['id'] = scenario_id
    extra_information['dataset'] = 'Argoverse 1'
    extra_information['version'] = args.raw_dataset_version
    extra_information['country'] = 'AMERICA'
    extra_information['city'] = 'PITTSBURGH' if raw_object_track_data.at[0, 'CITY_NAME'] == 'PIT' else 'MIAMI'
    extra_information['location'] = None
    extra_information['sdc_id'] = raw_object_track_data[raw_object_track_data['OBJECT_TYPE'] == 'AV'].iloc[0]['TRACK_ID']
    extra_information['scene_length'] = scenario_length
    extra_information['sampling_rate'] = 10
    extra_information['timestamp'] = np.array(raw_object_track_data[raw_object_track_data['OBJECT_TYPE'] == 'AV']['TIMESTAMP'])

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
