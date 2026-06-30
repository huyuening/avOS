# INTERnational, Adversarial and Cooperative moTION dataset (INTERACTION Dataset)

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as ET

from pathlib import Path
from tqdm import tqdm
try:
    from scenario_convertor.utils import (
        add_common_converter_args,
        get_multipolygon,
        get_road_lane_polyline,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )
except ModuleNotFoundError:
    from utils import (
        add_common_converter_args,
        get_multipolygon,
        get_road_lane_polyline,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description='INTERACTION Dataset Converter')
    add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/human-vehicle-dataset/human-vehicle-trajectory-dataset/interaction-dataset',
        default_subset_name='interaction',
        default_raw_dataset_version='1.1',
        raw_dataset_help='The file directory for INTERACTION',
    )
    parser.add_argument(
        '--locations',
        nargs='+',
        default=['all'],
        help='Locations to convert. Use "all" or names such as DR_CHN_Merging_ZS.',
    )
    parser.add_argument('--max_files', default=None, type=int, help='Convert at most this many recording ids.')
    parser.add_argument(
        '--max_scenarios_per_file',
        default=None,
        type=int,
        help='Convert at most this many 20-second scenarios per recording.',
    )
    parser.add_argument('--skip_maps', action='store_true', help='Skip OSM map parsing.')
    return parser


MAP_DATASET_DIRS = {
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


def selected_locations(args):
    if args.locations == ['all'] or 'all' in args.locations:
        return list(MAP_DATASET_DIRS.keys())
    unknown = sorted(set(args.locations) - set(MAP_DATASET_DIRS))
    if unknown:
        raise ValueError(f"Unknown INTERACTION locations: {unknown}")
    return args.locations


def file_sort_key(path):
    match = re.search(r'_(\d+)\.csv$', Path(path).name)
    return int(match.group(1)) if match else Path(path).name


def discover_recordings(args):
    raw_dataset_path = Path(args.raw_dataset_path)
    recordings = []
    for map_file_name in selected_locations(args):
        dataset_dir = raw_dataset_path / MAP_DATASET_DIRS[map_file_name]
        track_dir = dataset_dir / 'recorded_trackfiles' / map_file_name
        map_path = dataset_dir / 'maps' / f'{map_file_name}.osm_xy'
        if not track_dir.exists():
            raise FileNotFoundError(f'Cannot find INTERACTION track directory: {track_dir}')
        if not args.skip_maps and not map_path.exists():
            raise FileNotFoundError(f'Cannot find INTERACTION map file: {map_path}')

        groups = {}
        for csv_path in sorted(track_dir.glob('*_tracks_*.csv'), key=file_sort_key):
            match = re.search(r'_(\d+)\.csv$', csv_path.name)
            if not match:
                continue
            groups.setdefault(match.group(1), []).append(csv_path)
        for recording_id in sorted(groups, key=int):
            recordings.append((args, map_file_name, recording_id, [str(path) for path in sorted(groups[recording_id])], str(map_path)))
            if args.max_files is not None and len(recordings) >= args.max_files:
                return recordings
    return recordings


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
    if not road_line_ref_ids_:
        return np.zeros((0, 3), dtype=np.float64)
    road_line_polyline = np.asarray([polys_[road_line_ref_id] for road_line_ref_id in road_line_ref_ids_], dtype=np.float64)
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

        if 'type' not in tags:
            tags['type'] = ''
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

        if 'type' not in tags:
            tags['type'] = ''
        if 'subtype' not in tags:
            tags['subtype'] = ''
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


def extract_object_tracks(raw_object_track_data_part_, scenario_start_frame_, scenario_length_):
    ret = dict()
    for object_id, object_track in raw_object_track_data_part_.groupby('track_id', sort=False):
        object_type = object_track['agent_type'].iloc[0]
        object_state = object_state_template(scenario_length_)

        object_state['type'], object_state['subtype'] = get_object_type(object_type)
        object_state['phase'] = 'CONVERTED'
        frame_indices = object_track['frame_id'].to_numpy(dtype=np.int64) - scenario_start_frame_
        valid_rows = (frame_indices >= 0) & (frame_indices < scenario_length_)
        frame_indices = frame_indices[valid_rows]
        object_track = object_track.loc[valid_rows]

        object_state['state']['global_center'][frame_indices, 0] = object_track['x'].to_numpy(dtype=np.float64)
        object_state['state']['global_center'][frame_indices, 1] = object_track['y'].to_numpy(dtype=np.float64)

        if 'psi_rad' in object_track.columns:
            headings = object_track['psi_rad'].fillna(0.0).to_numpy(dtype=np.float64)
        elif 'vx' in object_track.columns and 'vy' in object_track.columns:
            headings = np.arctan2(object_track['vy'].fillna(0.0).to_numpy(dtype=np.float64), object_track['vx'].fillna(0.0).to_numpy(dtype=np.float64))
        else:
            headings = np.zeros(len(frame_indices), dtype=np.float64)
        object_state['state']['heading'][frame_indices] = headings

        if 'length' in object_track.columns and 'width' in object_track.columns:
            object_state['state']['size'][frame_indices, 0] = object_track['length'].fillna(0.0).to_numpy(dtype=np.float64)
            object_state['state']['size'][frame_indices, 1] = object_track['width'].fillna(0.0).to_numpy(dtype=np.float64)
        if 'vx' in object_track.columns and 'vy' in object_track.columns:
            vx = object_track['vx'].fillna(0.0).to_numpy(dtype=np.float64)
            vy = object_track['vy'].fillna(0.0).to_numpy(dtype=np.float64)
            object_state['state']['local_velocity'][frame_indices, 0] = vx * np.cos(headings) + vy * np.sin(headings)
            object_state['state']['local_velocity'][frame_indices, 1] = -vx * np.sin(headings) + vy * np.cos(headings)

        object_state['state']['valid'][frame_indices] = True
        ret[str(object_id)] = object_state
    return ret


def load_map_features(map_feature_data_path, map_file_name, skip_maps):
    if skip_maps:
        return {}

    raw_map_feature_data = ET.parse(map_feature_data_path).getroot()
    polys = {}
    for node in raw_map_feature_data.findall('node'):
        node_id = node.attrib['id']
        node_x, node_y = (float(node.attrib['x']), float(node.attrib['y']))
        polys[node_id] = (node_x, node_y)
    return extract_map_features(raw_map_feature_data, polys, map_file_name)


def interaction_metadata(map_file_name):
    metadata = {
        'dataset': 'INTERACTION',
        'version': '1.0' if map_file_name == 'TC_BGR_Intersection_VA' else '1.1',
        'country': None,
        'city': None,
        'location': map_file_name,
    }
    if map_file_name.startswith('DR_CHN'):
        metadata['country'] = 'CHINA'
    elif map_file_name.startswith('DR_DEU'):
        metadata['country'] = 'GERMANY'
    elif map_file_name.startswith('DR_USA'):
        metadata['country'] = 'AMERICA'
    elif map_file_name.startswith('TC_BGR'):
        metadata['country'] = 'BULGARIA'
    else:
        raise ValueError(f'Unknown country name: {map_file_name[3:6]}')
    return metadata


def process_recording(params):
    args, map_file_name, recording_id, track_paths, map_feature_data_path = params
    scenario_duration = 20
    scenario_length = 20 * 10

    raw_object_track_data = pd.concat([pd.read_csv(path) for path in track_paths], ignore_index=True)
    scenario_num = int(raw_object_track_data['frame_id'].max() // scenario_length)
    if args.max_scenarios_per_file is not None:
        scenario_num = min(scenario_num, args.max_scenarios_per_file)

    converted_map_feature_data = load_map_features(map_feature_data_path, map_file_name, args.skip_maps)
    converted_dynamic_map_states = dict()
    base_metadata = interaction_metadata(map_file_name)

    for scenario_idx in range(scenario_num):
        scenario_start_frame = scenario_idx * scenario_length + 1
        raw_object_track_data_part = raw_object_track_data[
            (raw_object_track_data['frame_id'] >= scenario_start_frame) &
            (raw_object_track_data['frame_id'] < scenario_start_frame + scenario_length)]
        if raw_object_track_data_part.empty:
            continue

        converted_object_track_data = extract_object_tracks(raw_object_track_data_part, scenario_start_frame, scenario_length)

        scenario_id = f'{map_file_name}_{recording_id}_{scenario_idx}'
        extra_information = dict(base_metadata)
        extra_information['id'] = scenario_id
        extra_information['sdc_id'] = None
        extra_information['scene_length'] = scenario_length
        extra_information['sampling_rate'] = 10
        extra_information['timestamp'] = np.array(range(scenario_length)) / 10
        extra_information['init_time'] = (scenario_start_frame - 1) / 10

        scenario = {
            'object_tracks': converted_object_track_data,
            'map_features': converted_map_feature_data,
            'dynamic_map_states': converted_dynamic_map_states,
            'extra_information': extra_information,
        }
        save_scenario_pickle(scenario, args, scenario_id)


def convert(args):
    recordings = discover_recordings(args)
    if not recordings:
        raise FileNotFoundError(f'Cannot find INTERACTION track files under {args.raw_dataset_path}.')
    run_tasks(process_recording, recordings, len(recordings), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
