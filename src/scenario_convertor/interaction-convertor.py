# INTERnational, Adversarial and Cooperative moTION dataset (INTERACTION Dataset)

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import numpy as np
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET

from tqdm import tqdm
from utils import add_common_converter_args, get_multipolygon, get_road_lane_polyline, object_state_template, run_tasks, save_scenario_pickle


def build_arg_parser():
    parser = argparse.ArgumentParser(description='INTERACTION Dataset Converter')
    return add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/human-vehicle-dataset/human-vehicle-trajectory-dataset/interaction-dataset',
        default_subset_name='interaction',
        default_raw_dataset_version='1.1',
        raw_dataset_help='The file directory for INTERACTION',
    )

map_file_names = {
    'DR_CHN_Merging_ZS': 'INTERACTION-Dataset-DR-v1_1',
    'DR_CHN_Roundabout_LN': 'INTERACTION-Dataset-DR-v1_1',
    'DR_DEU_Merging_MT': 'INTERACTION-Dataset-DR-v1_1',
    'DR_DEU_Roundabout_OF': 'INTERACTION-Dataset-DR-v1_1',
    'DR_USA_Intersection_EP0': 'INTERACTION-Dataset-DR-v1_1',
    'DR_USA_Intersection_EP1': 'INTERACTION-Dataset-DR-v1_1',
    'DR_USA_Intersection_GL': 'INTERACTION-Dataset-DR-v1_1',
    'DR_USA_Intersection_MA': 'INTERACTION-Dataset-DR-v1_1',
    'DR_USA_Roundabout_EP': 'INTERACTION-Dataset-DR-v1_1',
    'DR_USA_Roundabout_FT': 'INTERACTION-Dataset-DR-v1_1',
    'DR_USA_Roundabout_SR': 'INTERACTION-Dataset-DR-v1_1',
    'TC_BGR_Intersection_VA': 'INTERACTION-Dataset-TC-v1_0'
}


def get_object_type(object_type_):
    if object_type_ == 'car':
        return 'VEHICLE', 'CAR'
    elif object_type_ == 'truck':
        return 'VEHICLE', 'TRUCK'
    elif object_type_ == 'bus':
        return 'VEHICLE', 'BUS'
    elif object_type_ == 'motorcycle':
        return 'CYCLIST', 'MOTORCYCLIST'
    elif object_type_ == 'bicycle':
        return 'CYCLIST', 'BICYCLIST'
    elif object_type_ == 'pedestrian/bicycle':
        return 'UNKNOWN', 'UNKNOWN'
    else:
        raise ValueError("Unknown object type: {}".format(object_type_))


def get_polyline(road_line_ref_ids_, polys_):
    road_line_polyline = np.array([polys_[road_line_ref_id] for road_line_ref_id in road_line_ref_ids_])
    road_line_polyline = np.hstack((road_line_polyline, np.zeros((road_line_polyline.shape[0], 1))))
    return road_line_polyline


def get_speed_limit(map_file_name_):
    if map_file_name_ == 'DR_CHN_Merging_ZS':
        return 80 / 3.6
    elif map_file_name_ == 'DR_CHN_Roundabout_LN':
        return 30 / 3.6
    elif map_file_name_ == 'DR_DEU_Merging_MT':
        return 50 / 3.6
    elif map_file_name_ == 'DR_DEU_Roundabout_OF':
        return 50 / 3.6
    elif map_file_name_ == 'DR_USA_Intersection_EP0':
        return 15 * 1.61 / 3.6
    elif map_file_name_ == 'DR_USA_Intersection_EP1':
        return 15 * 1.61 / 3.6
    elif map_file_name_ == 'DR_USA_Intersection_GL':
        return 40 * 1.61 / 3.6
    elif map_file_name_ == 'DR_USA_Intersection_MA':
        return 40 * 1.61 / 3.6
    elif map_file_name_ == 'DR_USA_Roundabout_EP':
        return 15 * 1.61 / 3.6
    elif map_file_name_ == 'DR_USA_Roundabout_FT':
        return 25 * 1.61 / 3.6
    elif map_file_name_ == 'DR_USA_Roundabout_SR':
        return 25 * 1.61 / 3.6
    elif map_file_name_ == 'TC_BGR_Intersection_VA':
        return None
    else:
        raise ValueError(f"Unknown map file name: {map_file_name_}")


def get_map_type_and_subtype(map_type, map_subtype):
    if map_type == 'lanelet' and map_subtype == 'highway':
        return 'ROAD_LANE', 'HIGHWAY'
    elif map_type == 'lanelet' and map_subtype == 'road':
        return 'ROAD_LANE', 'URBAN_UNKNOWN'
    elif (map_type == 'line_thin' or map_type == 'line_thick') and map_subtype == 'solid':
        return 'ROAD_LINE', 'SOLID_UNKNOWN'
    elif (map_type == 'line_thick' or map_type == 'line_thin') and map_subtype == 'solid_solid':
        return 'ROAD_LINE', 'DOUBLE_SOLID_UNKNOWN'
    elif map_type == 'line_thin' and map_subtype == 'dashed':
        return 'ROAD_LINE', 'DASHED_UNKNOWN'
    elif map_type == 'virtual' and (map_subtype == '' or map_subtype == 'solid'):
        return 'ROAD_LINE', 'VIRTUAL'
    elif map_type == 'pedestrian_marking' and map_subtype == '':
        return 'ROAD_LINE', 'VIRTUAL'
    elif map_type == 'curbstone' and map_subtype == 'low':
        return 'ROAD_EDGE', 'UNKNOWN'
    elif (map_type == 'guard_rail' or map_type == 'road_border') and map_subtype == '':
        return 'ROAD_EDGE', 'UNKNOWN'
    elif map_type == 'stop_line' and map_subtype == '':
        return 'OTHER', 'STOP_LINE'
    elif map_type == 'traffic_sign' and map_subtype == 'usR1-1':
        return 'OTHER', 'STOP_SIGN'
    elif map_type == 'traffic_sign' and (map_subtype == 'de205' or map_subtype == 'usR1-2'):
        return 'OTHER', 'YIELD_SIGN'
    elif map_type == 'multipolygon' or map_subtype == 'keepout':
        return 'OTHER', 'KEEPOUT'
    elif map_type == 'traffic_sign' and (map_subtype == 'de215' or map_subtype == 'usR3-5R' or map_subtype == 'usR5-1a' or
                                         map_subtype == 'usR6-1L' or map_subtype == 'usR3-2' or map_subtype == 'usR5-1' or
                                         map_subtype == 'usR3-1' or map_subtype == 'usR4-7' or map_subtype == 'usW5-1' or
                                         map_subtype == 'usR4-8a' or map_subtype == ''):
        return None, None
    elif map_type == 'regulatory_element' and (map_subtype == 'speed_limit' or map_subtype == 'right_of_way' or
                                               map_subtype == 'all_way_stop'):
        return None, None
    else:
        raise ValueError(f'Unknown map type: {map_type} and map subtype: {map_subtype}')


def extract_map_features(raw_map_feature_data_, polys_, map_file_name_):
    ret = dict()

    for way in raw_map_feature_data_.findall('way'):
        way_id = way.attrib['id']
        way_info = dict()

        static_map_ref_ids = [node.attrib['ref'] for node in way.findall('nd')]
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}

        if 'subtype' not in tags:
            tags['subtype'] = ''

        way_info['type'], way_info['subtype'] = get_map_type_and_subtype(tags['type'], tags['subtype'])

        if not way_info['type'] or not way_info['subtype']:
            continue

        if way_info['subtype'] == 'CROSSWALK':
            way_info['polygon'] = get_polyline(static_map_ref_ids, polys_)
            way_info['phase'] = 'CONVERTED'
        else:
            way_info['polyline'] = get_polyline(static_map_ref_ids, polys_)
            way_info['phase'] = 'CONVERTED'

        ret[way_id] = way_info

    for relation in raw_map_feature_data_.findall('relation'):
        relation_id = relation.attrib['id']
        if relation_id in ret:
            relation_id = relation_id + '0'

        relation_info = dict()

        tags = {tag.attrib['k']: tag.attrib['v'] for tag in relation.findall('tag')}

        if 'region' not in tags:
            tags['region'] = ''
        if 'one_way' not in tags:
            tags['one_way'] = ''
        if 'location' not in tags:
            tags['location'] = ''
        if 'sign_type' not in tags:
            tags['sign_type'] = ''

        relation_info['type'], relation_info['subtype'] = get_map_type_and_subtype(tags['type'], tags['subtype'])

        if not relation_info['type'] or not relation_info['subtype']:
            continue

        if relation_info['type'] == 'ROAD_LANE':
            members = {member.attrib['ref']: member.attrib['role'] for member in relation.findall('member')}

            left_road_line_id = [id for id in members if members[id] == 'left'][0]
            left_road_line_polyline = ret[left_road_line_id]['polyline']
            right_road_line_id = [id for id in members if members[id] == 'right'][0]
            right_road_line_polyline = ret[right_road_line_id]['polyline']

            relation_info['polyline'] = get_road_lane_polyline(left_road_line_polyline, right_road_line_polyline)
            relation_info['speed_limit'] = get_speed_limit(map_file_name_)
            relation_info['is_intersection'] = None
            relation_info['phase'] = 'CONVERTED'

            relation_info['lane_info'] = dict()
            relation_info['lane_info']['entry_lane_ids'] = dict()
            relation_info['lane_info']['exit_lane_ids'] = dict()
            relation_info['lane_info']['left_lane_ids'] = dict()
            relation_info['lane_info']['right_lane_ids'] = dict()

            relation_info['boundary_info'] = dict()
            relation_info['boundary_info']['left_boundary_id'] = left_road_line_id
            relation_info['boundary_info']['right_boundary_id'] = right_road_line_id
            relation_info['boundary_info']['left_boundary_polyline'] = left_road_line_polyline
            relation_info['boundary_info']['right_boundary_polyline'] = right_road_line_polyline

        elif relation_info['type'] == 'OTHER' and relation_info['subtype'] == 'KEEPOUT':
            relation_info['polygon'] = get_multipolygon(ret, relation)
            relation_info['phase'] = 'CONVERTED'

        else:
            raise ValueError(f"Unknown relation type: {relation_info['type']} and relation subtype: {relation_info['subtype']}")

        ret[relation_id] = relation_info

    return ret


def extract_object_tracks(raw_object_track_data_part_, scenario_idx_, scenario_length_):
    ret = dict()
    for object_id in raw_object_track_data_part_['track_id'].unique():
        object_track = raw_object_track_data_part_[raw_object_track_data_part_['track_id'] == object_id]

        object_id = str(object_id)
        object_type = object_track['agent_type'].iloc[0]
        object_state = object_state_template(scenario_length_)

        object_state['type'], object_state['subtype'] = get_object_type(object_type)
        object_state['phase'] = 'CONVERTED'
        for idx, frame in enumerate(range(scenario_idx_*scenario_length_+1, (scenario_idx_+1)*scenario_length_+1)):
            if not object_track[object_track['frame_id'] == frame].empty:
                object_state['state']['global_center'][idx] = (object_track[object_track['frame_id'] == frame]['x'].item(), object_track[object_track['frame_id'] == frame]['y'].item(), 0.0)
                if 'psi_rad' in object_track.columns:
                    object_state['state']['heading'][idx] = object_track[object_track['frame_id'] == frame]['psi_rad'].item()
                if 'length' in object_track.columns and 'width' in object_track.columns:
                    object_state['state']['size'][idx] = (object_track[object_track['frame_id'] == frame]['length'].item(), object_track[object_track['frame_id'] == frame]['width'].item(), 0.0)
                object_state['state']['valid'][idx] = True
        ret[object_id] = object_state
    return ret


def process_scenario(params):
    args, map_file_name = params
    base_dir = args.raw_dataset_path.rstrip('/') + '/'
    scenario_duration = 20
    scenario_length = 20 * 10

    object_track_file_dir = base_dir + map_file_names[map_file_name] + '/recorded_trackfiles/' + map_file_name

    raw_object_track_data_paths = [object_track_file_dir + '/' + i for i in os.listdir(object_track_file_dir)]
    raw_map_feature_data_path = base_dir + map_file_names[map_file_name] + '/maps/' + map_file_name + '.osm_xy'

    max_number = max([int(f[-6: -4]) for f in raw_object_track_data_paths])
    for idx in range(max_number + 1):
        if idx < 10:
            pattern = f'00{idx}'
        else:
            pattern = f'0{idx}'
        filtered_raw_object_track_data_paths = [f for f in raw_object_track_data_paths if re.search(pattern, f)]

        raw_object_track_data = pd.concat([pd.read_csv(f) for f in filtered_raw_object_track_data_paths])
        scenario_num = raw_object_track_data['frame_id'].max() // scenario_length

        scenario = dict()
        for scenario_idx in range(scenario_num):

            raw_object_track_data_part = raw_object_track_data[
                (raw_object_track_data['frame_id'] < scenario_idx * scenario_length + scenario_duration * 10 + 1) &
                (raw_object_track_data['frame_id'] >= scenario_idx * scenario_length + 1)]

            converted_object_track_data = extract_object_tracks(raw_object_track_data_part, scenario_idx, scenario_length)

            raw_map_feature_data = ET.parse(raw_map_feature_data_path).getroot()

            polys = {}
            for node in raw_map_feature_data.findall('node'):
                node_id = node.attrib['id']
                node_x, node_y = (node.attrib['x'], node.attrib['y'])
                polys[node_id] = (node_x, node_y)

            converted_map_feature_data = extract_map_features(raw_map_feature_data, polys, map_file_name)

            converted_dynamic_map_states = dict()

            extra_information = dict()
            extra_information['id'] = str(idx) + '_' + str(scenario_idx) + '_' + map_file_name
            extra_information['dataset'] = 'INTERACTION'

            if map_file_name == 'DR_CHN_Merging_ZS' or map_file_name == 'DR_CHN_Roundabout_LN':
                extra_information['version'] = '1.1'
                extra_information['country'] = 'CHINA'
            elif map_file_name == 'DR_DEU_Merging_MT' or map_file_name == 'DR_DEU_Roundabout_OF':
                extra_information['version'] = '1.1'
                extra_information['country'] = 'GERMANY'
            elif map_file_name == 'DR_USA_Intersection_EP0' or map_file_name == 'DR_USA_Intersection_EP1' or \
                 map_file_name == 'DR_USA_Intersection_GL' or map_file_name == 'DR_USA_Intersection_MA' or \
                 map_file_name == 'DR_USA_Roundabout_EP' or map_file_name == 'DR_USA_Roundabout_FT' or \
                 map_file_name == 'DR_USA_Roundabout_SR':
                extra_information['version'] = '1.1'
                extra_information['country'] = 'AMERICA'
            elif map_file_name == 'TC_BGR_Intersection_VA':
                extra_information['version'] = '1.0'
                extra_information['country'] = 'BULGARIA'
            else:
                raise ValueError("Unknown country name: {}".format(map_file_name[3:6]))

            extra_information['city'] = None
            extra_information['location'] = None
            extra_information['sdc_id'] = None
            extra_information['scene_length'] = scenario_length
            extra_information['sampling_rate'] = 10
            extra_information['timestamp'] = np.array(range(scenario_length)) / 10

            scenario['object_tracks'] = converted_object_track_data
            scenario['map_features'] = converted_map_feature_data
            scenario['dynamic_map_states'] = converted_dynamic_map_states
            scenario['extra_information'] = extra_information

            save_scenario_pickle(scenario, args, extra_information['id'])


def convert(args):
    map_names = list(map_file_names.keys())
    params = [(args, map_file_name) for map_file_name in map_names]
    run_tasks(process_scenario, params, len(map_names), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
