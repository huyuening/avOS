# levelXData

# Highway Drone Dataset, highD
# Intersections Drone Dataset, inD
# Roundabouts Drone Dataset, rounD
# Entries and Exits Drone Dataset, exiD
# Universities Drone Dataset, uniD

# Object tracks             yes (csv)
#   Object type: car, bicycle, pedestrian, truck_bus

# Map features             yes (osm)

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

from pathlib import Path
from pyproj import Proj
from tqdm import tqdm
try:
    from scenario_convertor.utils import (
        add_common_converter_args,
        connect_arrays,
        get_multipolygon,
        get_road_lane_polyline,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )
except ModuleNotFoundError:
    from utils import (
        add_common_converter_args,
        connect_arrays,
        get_multipolygon,
        get_road_lane_polyline,
        object_state_template,
        run_tasks,
        save_scenario_pickle,
    )


LEVELX_DATASETS = {
    'highD': {
        'data_dirs': ('highway-drone-dataset-v1.0/data', 'highD-v1.0/data'),
    },
    'inD': {
        'data_dirs': ('intersections-drone-dataset-v1.1/data', 'inD-v1.1/data'),
        'map_dirs': ('intersections-drone-dataset-v1.1/maps/lanelets', 'inD-v1.1/maps/lanelets'),
    },
    'rounD': {
        'data_dirs': ('roundabouts-drone-dataset-v1.1/data', 'rounD-v1.1/data'),
        'map_dirs': ('roundabouts-drone-dataset-v1.1/maps/lanelets', 'rounD-v1.1/maps/lanelets'),
    },
    'uniD': {
        'data_dirs': ('universities-drone-dataset-v1.1/data', 'uniD-v1.1/data'),
        'map_dirs': ('universities-drone-dataset-v1.1/maps/lanelet', 'uniD-v1.1/maps/lanelet'),
    },
    'exiD': {
        'data_dirs': ('entries-and-exits-drone-dataset-v2.1/data', 'exiD-v2.1/data'),
        'map_dirs': ('entries-and-exits-drone-dataset-v2.1/maps/lanelet2', 'exiD-v2.1/maps/lanelet2'),
    },
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description='levelXData Converter')
    add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/human-vehicle-dataset/human-vehicle-trajectory-dataset/levelXData',
        default_subset_name='levelxdata',
        default_raw_dataset_version='mixed',
        raw_dataset_help='The file directory for levelXData',
    )
    parser.add_argument(
        '--levelx_dataset',
        '--levelx-dataset',
        choices=('all', *LEVELX_DATASETS.keys()),
        default='all',
        help='levelXData subset to convert.',
    )
    parser.add_argument('--max_files', default=None, type=int, help='Convert at most this many recordings.')
    parser.add_argument(
        '--max_scenarios_per_file',
        default=None,
        type=int,
        help='Convert at most this many 20-second scenarios per recording.',
    )
    parser.add_argument('--skip_maps', action='store_true', help='Skip lanelet map parsing.')
    return parser


def natural_file_key(path):
    stem = Path(path).stem.split('_')[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def infer_levelx_dataset(path):
    normalized_path = str(path).replace(os.sep, '/')
    if 'highway-drone-dataset' in normalized_path or '/highD-' in normalized_path:
        return 'highD'
    if 'intersections-drone-dataset' in normalized_path or '/inD-' in normalized_path:
        return 'inD'
    if 'roundabouts-drone-dataset' in normalized_path or '/rounD-' in normalized_path:
        return 'rounD'
    if 'universities-drone-dataset' in normalized_path or '/uniD-' in normalized_path:
        return 'uniD'
    if 'entries-and-exits-drone-dataset' in normalized_path or '/exiD-' in normalized_path:
        return 'exiD'
    return None


def selected_levelx_datasets(args):
    if args.levelx_dataset != 'all':
        return [args.levelx_dataset]
    inferred_dataset = infer_levelx_dataset(args.raw_dataset_path)
    if inferred_dataset:
        return [inferred_dataset]
    return list(LEVELX_DATASETS.keys())


def candidate_dirs(raw_dataset_path, dataset_name, key):
    raw_dataset_path = Path(raw_dataset_path)
    candidates = []
    if key == 'data_dirs':
        if raw_dataset_path.name == 'data':
            candidates.append(raw_dataset_path)
        candidates.append(raw_dataset_path / 'data')
    elif key == 'map_dirs':
        candidates.extend(
            (
                raw_dataset_path / 'maps' / 'lanelets',
                raw_dataset_path / 'maps' / 'lanelet',
                raw_dataset_path / 'maps' / 'lanelet2',
            )
        )
    candidates.extend(raw_dataset_path / relative_dir for relative_dir in LEVELX_DATASETS[dataset_name].get(key, ()))
    seen = set()
    ret = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            ret.append(candidate)
    return ret


def find_dataset_map_dir(raw_dataset_path, dataset_name):
    map_dirs = candidate_dirs(raw_dataset_path, dataset_name, 'map_dirs')
    return map_dirs[0] if map_dirs else None


def discover_recording_files(data_dir, dataset_name):
    ret = []
    for recording_meta in sorted(Path(data_dir).glob('*_recordingMeta.csv'), key=natural_file_key):
        recording_id = recording_meta.name.split('_')[0]
        tracks = recording_meta.with_name(f'{recording_id}_tracks.csv')
        tracks_meta = recording_meta.with_name(f'{recording_id}_tracksMeta.csv')
        if not tracks.exists() or not tracks_meta.exists():
            raise FileNotFoundError(f'Incomplete levelXData recording trio for {recording_meta}.')
        ret.append((str(recording_meta), str(tracks), str(tracks_meta), dataset_name, recording_id))
    return ret


def list_matching_files(args):
    all_matching_files = []
    for dataset_name in selected_levelx_datasets(args):
        for data_dir in candidate_dirs(args.raw_dataset_path, dataset_name, 'data_dirs'):
            all_matching_files.extend(discover_recording_files(data_dir, dataset_name))

    if args.max_files is not None:
        all_matching_files = all_matching_files[:args.max_files]
    return all_matching_files


def get_object_type(object_type):
    if object_type == 'car' or object_type == 'Car':
        return 'VEHICLE', 'CAR'
    elif object_type == 'truck' or object_type == 'Truck':
        return 'VEHICLE', 'TRUCK'
    elif object_type == 'van':
        return 'VEHICLE', 'VAN'
    elif object_type == 'trailer':
        return 'VEHICLE', 'TRAILER'
    elif object_type == 'bus':
        return 'VEHICLE', 'BUS'
    elif object_type == 'truck_bus':
        return 'VEHICLE', 'UNKNOWN'
    elif object_type == 'bicycle':
        return 'CYCLIST', 'BICYCLIST'
    elif object_type == 'motorcycle':
        return 'CYCLIST', 'MOTORCYCLIST'
    elif object_type == 'pedestrian':
        return 'PEDESTRIAN', 'UNKNOWN'
    elif object_type == 'animal':
        return 'OTHER', 'ANIMAL'
    else:
        raise ValueError("Unknown object type: {}".format(object_type))


def extract_object_tracks(raw_object_track_data_, scenario_start_frame_, scenario_length_):
    ret = dict()
    tracks, tracks_meta = raw_object_track_data_
    tracks_meta = tracks_meta.set_index('trackId', drop=False)

    for object_id, object_track in tracks.groupby('trackId', sort=False):
        if object_id not in tracks_meta.index:
            continue
        object_track_meta = tracks_meta.loc[object_id]
        if isinstance(object_track_meta, pd.DataFrame):
            object_track_meta = object_track_meta.iloc[0]

        object_type = object_track_meta['class']
        object_state = object_state_template(scenario_length_)

        object_state['type'], object_state['subtype'] = get_object_type(object_type)
        object_state['phase'] = 'CONVERTED'
        frame_indices = (object_track['frame'].to_numpy(dtype=np.int64) - scenario_start_frame_).astype(np.int64)
        valid_rows = (frame_indices >= 0) & (frame_indices < scenario_length_)
        frame_indices = frame_indices[valid_rows]
        object_track = object_track.loc[valid_rows]

        object_state['state']['global_center'][frame_indices, 0] = object_track['xCenter'].to_numpy(dtype=np.float64)
        object_state['state']['global_center'][frame_indices, 1] = object_track['yCenter'].to_numpy(dtype=np.float64)
        object_state['state']['size'][frame_indices, 0] = float(object_track_meta['length'])
        object_state['state']['size'][frame_indices, 1] = float(object_track_meta['width'])
        if 'heading' in object_track.columns:
            headings = object_track['heading'].to_numpy(dtype=np.float64) * np.pi / 180
            object_state['state']['heading'][frame_indices] = headings
        else:
            headings = object_state['state']['heading'][frame_indices]
        if 'xVelocity' in object_track.columns and 'yVelocity' in object_track.columns:
            vx = object_track['xVelocity'].to_numpy(dtype=np.float64)
            vy = object_track['yVelocity'].to_numpy(dtype=np.float64)
            object_state['state']['local_velocity'][frame_indices, 0] = vx * np.cos(headings) + vy * np.sin(headings)
            object_state['state']['local_velocity'][frame_indices, 1] = -vx * np.sin(headings) + vy * np.cos(headings)
        if 'xAcceleration' in object_track.columns and 'yAcceleration' in object_track.columns:
            ax = object_track['xAcceleration'].to_numpy(dtype=np.float64)
            ay = object_track['yAcceleration'].to_numpy(dtype=np.float64)
            object_state['state']['local_acceleration'][frame_indices, 0] = ax * np.cos(headings) + ay * np.sin(headings)
            object_state['state']['local_acceleration'][frame_indices, 1] = -ax * np.sin(headings) + ay * np.cos(headings)
        object_state['state']['valid'][frame_indices] = True
        ret[str(object_id)] = object_state
    return ret


def get_speed_limit(speed_limit):
    if pd.isna(speed_limit) or str(speed_limit).lower() in {'', 'none', 'nan'}:
        return None
    match = re.search(r'(\d+)\s*km/h', speed_limit)
    return int(match.group(1)) if match else None


def get_crosswalk_polygon(ret, relation):
    for member in relation.findall('member'):
        if member.attrib['role'] == 'left':
            left_road_line_edge_id = member.attrib['ref']
        if member.attrib['role'] == 'right':
            right_road_line_edge_id = member.attrib['ref']

    left_road_line_edge_polyline = ret[left_road_line_edge_id]['polyline']
    right_road_line_edge_polyline = ret[right_road_line_edge_id]['polyline']
    crosswalk_polygon = connect_arrays(left_road_line_edge_polyline, right_road_line_edge_polyline)

    return crosswalk_polygon


def get_polyline(road_line_ref_ids_, polys_):
    road_line_polyline = np.array(
        [np.pad(polys_[road_line_ref_id], (0, 1), 'constant', constant_values=(0,)) for road_line_ref_id in
         road_line_ref_ids_])
    return road_line_polyline


def get_map_type_and_subtype(map_type, map_subtype):
    if map_type == 'virtual' and (map_subtype == '' or map_subtype == 'low' or map_subtype == 'solid' or
                                  map_subtype == 'high'):
        return 'ROAD_LINE', 'VIRTUAL'
    elif map_type == 'zebra_marking' and map_subtype == '':
        return 'ROAD_LINE', 'VIRTUAL'
    elif (map_type == 'line_thin' or map_type == 'line_thick') and (map_subtype == 'solid' or map_subtype == ''):
        return 'ROAD_LINE', 'SOLID_UNKNOWN'
    elif (map_type == 'line_thin' or map_type == 'line_thick') and map_subtype == 'dashed':
        return 'ROAD_LINE', 'DASHED_UNKNOWN'
    elif (map_type == 'line_thin' or map_type == 'line_thick') and map_subtype == 'solid_dashed':
        return 'ROAD_LINE', 'SOLID_DASHED_UNKNOWN'
    elif map_type == 'lanelet' and map_subtype == 'highway':
        return 'ROAD_LANE', 'HIGHWAY'
    elif map_type == 'lanelet' and (map_subtype == 'road' or map_subtype == '' or map_subtype == 'emergency_lane'):
        return 'ROAD_LANE', 'URBAN_VEHICLE'
    elif map_type == 'lanelet' and map_subtype == 'play_street':
        return 'ROAD_LANE', 'URBAN_UNKNOWN'
    elif map_type == 'lanelet' and map_subtype == 'bus_lane':
        return 'ROAD_LANE', 'URBAN_BUS'
    elif map_type == 'lanelet' and map_subtype == 'bicycle_lane':
        return 'ROAD_LANE', 'URBAN_BIKE'
    elif map_type == 'lanelet' and map_subtype == 'walkway':
        return 'ROAD_LANE', 'URBAN_PEDESTRIAN'
    elif map_type == 'curbstone' and (map_subtype == 'high' or map_subtype == 'low' or map_subtype == ''):
        return 'ROAD_EDGE', 'UNKNOWN'
    elif map_type == 'road_border' and (map_subtype == '' or map_subtype == 'solid' or map_subtype == 'low' or
                                        map_subtype == 'high'):
        return 'ROAD_EDGE', 'UNKNOWN'
    elif map_type == 'traffic_sign' and map_subtype == 'de205':
        return 'OTHER', 'YIELD_SIGN'
    elif map_type == 'traffic_sign' and map_subtype == 'de206':
        return 'OTHER', 'STOP_SIGN'
    elif (map_type == 'wall' or map_type == 'multipolygon') and map_subtype == 'building':
        return 'OTHER', 'BUILDING'
    elif map_type == 'wall' and (map_subtype == '' or map_subtype == 'low'):
        return 'OTHER', 'WALL'
    elif map_type == 'fence' and map_subtype == '':
        return 'OTHER', 'FENCE'
    elif map_type == 'lanelet' and map_subtype == 'crosswalk':
        return 'OTHER', 'CROSSWALK'
    elif map_type == 'multipolygon' and map_subtype == 'emergency_lane':
        return 'OTHER', 'EMERGENCY_LANE'
    elif map_type == 'multipolygon' and map_subtype == 'traffic_island':
        return 'OTHER', 'TRAFFIC_ISLAND'
    elif map_type == 'multipolygon' and map_subtype == 'vegetation':
        return 'OTHER', 'VEGETATION'
    elif map_type == 'multipolygon' and map_subtype == 'bicycle_lane':
        return 'OTHER', 'BICYCLE_LANE'
    elif map_type == 'multipolygon' and map_subtype == 'walkway':
        return 'OTHER', 'WALKWAY'
    elif map_type == 'multipolygon' and map_subtype == 'intersection':
        return 'OTHER', 'INTERSECTION'
    elif map_type == 'multipolygon' and (map_subtype == 'parking' or map_subtype == 'bicycle_parking'):
        return 'OTHER', 'PARKING'
    elif map_type == 'multipolygon' and map_subtype == 'park':
        return 'OTHER', 'PARK'
    elif map_type == 'multipolygon' and map_subtype == 'keepout':
        return 'OTHER', 'KEEPOUT'
    elif map_type == 'arrow' and (map_subtype == 'straight_left' or map_subtype == 'straight' or map_subtype == 'left' or
                                  map_subtype == 'right', map_subtype == 'straight_right'):
        return None, None
    elif map_type == 'traffic_sign' and (map_subtype == 'de301' or map_subtype == 'de357' or map_subtype == 'de262' or
                                         map_subtype == 'de283' or map_subtype == 'de314' or map_subtype == 'de102' or
                                         map_subtype == 'de306'):
        return None, None
    elif map_type == 'de274-30' and map_subtype == '':
        return None, None
    elif map_type == 'traffic_light' and map_subtype == 'red_yellow_green':
        return None, None
    elif map_type == 'regulatory_element' and (map_subtype == 'right_of_way' or map_subtype == 'traffic_light'):
        return None, None
    elif map_type == 'multipolygon' and (map_subtype == 'pedestrian_seat' or map_subtype == 'freespace'):
        return None, None
    else:
        raise ValueError(f'Unknown map type: {map_type} and map subtype: {map_subtype}')


def extract_map_features(raw_map_feature_data_, polys_):
    ret = {}

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

        way_info['polyline'] = get_polyline(static_map_ref_ids, polys_)
        way_info['phase'] = 'CONVERTED'

        ret[way_id] = way_info

    for relation in raw_map_feature_data_.findall('relation'):
        relation_id = relation.attrib['id']
        if relation_id in ret:
            relation_id = relation_id + '0'

        relation_info = dict()

        tags = {tag.attrib['k']: tag.attrib['v'] for tag in relation.findall('tag')}

        if 'speed_limit' not in tags:
            tags['speed_limit'] = ''
        if 'region' not in tags:
            tags['region'] = ''
        if 'one_way' not in tags:
            tags['one_way'] = ''
        if 'location' not in tags:
            tags['location'] = ''
        if 'subtype' not in tags:
            tags['subtype'] = ''

        relation_info['type'], relation_info['subtype'] = get_map_type_and_subtype(tags['type'], tags['subtype'])

        if not relation_info['type'] or not relation_info['subtype']:
            continue

        if relation_info['type'] == 'ROAD_LANE':
            members = {member.attrib['ref']: member.attrib['role'] for member in relation.findall('member')}

            left_road_line_id = [id for id in members if members[id] == 'left'][0]
            left_road_line_polyline = ret[left_road_line_id]['polyline']
            right_road_line_id = [id for id in members if members[id] == 'right'][0]
            right_road_line_polyline = ret[right_road_line_id]['polyline']
            speed_limit = get_speed_limit(tags['speed_limit'])

            relation_info['polyline'] = get_road_lane_polyline(left_road_line_polyline, right_road_line_polyline)
            relation_info['speed_limit'] = speed_limit / 3.6 if speed_limit is not None else None
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

        elif relation_info['type'] == 'OTHER' and (relation_info['subtype'] == 'EMERGENCY_LANE' or
                                                   relation_info['subtype'] == 'TRAFFIC_ISLAND' or
                                                   relation_info['subtype'] == 'VEGETATION' or
                                                   relation_info['subtype'] == 'BICYCLE_LANE' or
                                                   relation_info['subtype'] == 'WALKWAY' or
                                                   relation_info['subtype'] == 'INTERSECTION' or
                                                   relation_info['subtype'] == 'PARKING' or
                                                   relation_info['subtype'] == 'PARK' or
                                                   relation_info['subtype'] == 'KEEPOUT' or
                                                   relation_info['subtype'] == 'BUILDING'):
            relation_info['polygon'] = get_multipolygon(ret, relation)
            relation_info['phase'] = 'CONVERTED'

        elif relation_info['type'] == 'OTHER' and relation_info['subtype'] == 'CROSSWALK':
            relation_info['polygon'] = get_crosswalk_polygon(ret, relation)
            relation_info['phase'] = 'CONVERTED'

        else:
            raise ValueError(f"Unknown relation type: {relation_info['type']} and relation subtype: {relation_info['subtype']}")

        ret[relation_id] = relation_info

    return ret


LEVELX_LOCATION_INFO = {
    'inD': {
        1: ('01_bendplatz', '1.1', 'GERMANY', 'AACHEN', 'BENDPLATZ'),
        2: ('02_frankenburg', '1.1', 'GERMANY', 'AACHEN', 'FRANKENBURG'),
        3: ('03_heckstrasse', '1.1', 'GERMANY', 'AACHEN', 'HECKSTRASSE'),
        4: ('04_aseag', '1.1', 'GERMANY', 'AACHEN', 'ASEAG'),
    },
    'rounD': {
        0: ('0_neuweiler', '1.1', 'GERMANY', 'AACHEN', 'NEUWEILER'),
        1: ('1_kackertstrasse', '1.1', 'GERMANY', 'AACHEN', 'KACKERTSTRASSE'),
        2: ('2_thiergarten', '1.1', 'GERMANY', 'ALSDORF', 'THIERGARTEN'),
    },
    'uniD': {
        0: ('0_superc', '1.1', 'GERMANY', 'AACHEN', 'SUPERC'),
    },
    'exiD': {
        0: ('0_cologne_butzweiler', '2.1', 'GERMANY', 'COLOGNE', 'BUTZWEILER'),
        1: ('1_cologne_fortiib', '2.1', 'GERMANY', 'COLOGNE', 'FORTIIB'),
        2: ('2_aachen_brand', '2.1', 'GERMANY', 'AACHEN', 'BRAND'),
        3: ('3_bergheim_roemer', '2.1', 'GERMANY', 'BERGHEIM', 'ROEMER'),
        4: ('4_cologne_klettenberg', '2.1', 'GERMANY', 'COLOGNE', 'KLETTENBERG'),
        5: ('5_aachen_laurensberg', '2.1', 'GERMANY', 'AACHEN', 'LAURENSBERG'),
        6: ('6_merzenich_rather', '2.1', 'GERMANY', 'MERZENICH', 'RATHER'),
    },
}


def get_levelx_location_info(dataset_name, location_id):
    if dataset_name == 'highD':
        return {
            'map_file_name': '',
            'version': '1.0',
            'country': 'GERMANY',
            'city': 'COLOGNE',
            'location': None,
        }

    location_info = LEVELX_LOCATION_INFO.get(dataset_name, {}).get(int(location_id))
    if location_info is None:
        raise ValueError(f'Wrong location ID for {dataset_name}: {location_id}')

    map_file_name, version, country, city, location = location_info
    return {
        'map_file_name': map_file_name,
        'version': version,
        'country': country,
        'city': city,
        'location': location,
    }


def load_map_features(args, object_track_meta, dataset_name, location_id, map_file_name):
    if not map_file_name or args.skip_maps:
        return {}

    map_directory_path = find_dataset_map_dir(args.raw_dataset_path, dataset_name)
    if map_directory_path is None:
        raise FileNotFoundError(f'Cannot find lanelet map directory for {dataset_name}.')

    project_rule = {'proj': 'utm', 'ellps': 'WGS84', 'zone': 32, 'datum': 'WGS84'}
    projector = Proj(**project_rule)
    origin = (object_track_meta['xUtmOrigin'].item(), object_track_meta['yUtmOrigin'].item())

    raw_map_feature_data_path = Path(map_directory_path) / map_file_name / f'location{location_id}.osm'
    if not raw_map_feature_data_path.exists():
        raise FileNotFoundError(f'Cannot find lanelet map file: {raw_map_feature_data_path}')
    raw_map_feature_data = ET.parse(raw_map_feature_data_path).getroot()

    polys = {}
    for node in raw_map_feature_data.findall('node'):
        new = projector(node.attrib['lon'], node.attrib['lat'])
        node_id = node.attrib['id']
        node_x, node_y = (new[0] - origin[0], new[1] - origin[1])
        polys[node_id] = (node_x, node_y)

    return extract_map_features(raw_map_feature_data, polys)


def process_scenario(params):
    args, file = params
    recordingMeta = file[0]
    tracks = file[1]
    tracksMeta = file[2]
    dataset_name = file[3]
    recording_id = file[4]

    object_track1 = pd.read_csv(recordingMeta)
    object_track2 = pd.read_csv(tracks, low_memory=False)
    object_track3 = pd.read_csv(tracksMeta)

    if 'id' in object_track2.columns:
        object_track2.rename(columns={'id': 'trackId'}, inplace=True)
        object_track2.rename(columns={'x': 'xCenter'}, inplace=True)
        object_track2.rename(columns={'y': 'yCenter'}, inplace=True)
    if 'id' in object_track3.columns:
        object_track3.rename(columns={'id': 'trackId'}, inplace=True)
        object_track3.rename(columns={'width': 'length'}, inplace=True)
        object_track3.rename(columns={'height': 'width'}, inplace=True)

    frame_rate = int(object_track1['frameRate'].item()) if 'frameRate' in object_track1.columns else 25
    scenario_duration = 20
    scenario_num = int(object_track1['duration'].item() // scenario_duration)
    if args.max_scenarios_per_file is not None:
        scenario_num = min(scenario_num, args.max_scenarios_per_file)
    scenario_length = scenario_duration * frame_rate
    locationId = int(object_track1['locationId'].item()) if 'locationId' in object_track1.columns else 0
    location_info = get_levelx_location_info(dataset_name, locationId)
    converted_map_feature_data = load_map_features(
        args,
        object_track1,
        dataset_name,
        locationId,
        location_info['map_file_name'],
    )
    converted_dynamic_map_state_data = dict()

    for scenario_idx in range(scenario_num):
        scenario = dict()

        scenario_start_frame = scenario_idx * scenario_length
        object_track2_part = object_track2[
            (object_track2['frame'] < scenario_start_frame + scenario_length) &
            (object_track2['frame'] >= scenario_start_frame)]
        raw_object_track_data = [object_track2_part, object_track3]
        converted_object_track_data = extract_object_tracks(raw_object_track_data, scenario_start_frame, scenario_length)

        extra_information = dict()
        extra_information['id'] = f'{dataset_name}_{recording_id}_{scenario_idx}'
        extra_information['dataset'] = dataset_name
        extra_information['version'] = location_info['version']
        extra_information['country'] = location_info['country']
        extra_information['city'] = location_info['city']
        extra_information['location'] = location_info['location']
        extra_information['sdc_id'] = None
        extra_information['scene_length'] = scenario_length
        extra_information['sampling_rate'] = frame_rate
        extra_information['timestamp'] = np.array(range(scenario_length)) / frame_rate
        extra_information['init_time'] = scenario_start_frame / frame_rate

        scenario['object_tracks'] = converted_object_track_data
        scenario['map_features'] = converted_map_feature_data
        scenario['dynamic_map_states'] = converted_dynamic_map_state_data
        scenario['extra_information'] = extra_information

        save_scenario_pickle(scenario, args, extra_information['id'])


def convert(args):
    all_matching_files = list_matching_files(args)
    if not all_matching_files:
        raise FileNotFoundError(f"Cannot find levelXData recording CSV trios under {args.raw_dataset_path}.")
    params = [(args, file) for file in all_matching_files]
    run_tasks(process_scenario, params, len(all_matching_files), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
