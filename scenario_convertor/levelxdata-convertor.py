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

from pyproj import Proj
from tqdm import tqdm
from utils import add_common_converter_args, connect_arrays, get_multipolygon, get_road_lane_polyline, object_state_template, run_tasks, save_scenario_pickle


LEVELX_DATASETS = {
    'highD': {
        'data_dir': 'highway-drone-dataset-v1.0/data',
        'num_iter': 61,
    },
    'inD': {
        'data_dir': 'intersections-drone-dataset-v1.1/data',
        'map_dir': 'intersections-drone-dataset-v1.1/maps/lanelets',
        'num_iter': 33,
    },
    'rounD': {
        'data_dir': 'roundabouts-drone-dataset-v1.1/data',
        'map_dir': 'roundabouts-drone-dataset-v1.1/maps/lanelets',
        'num_iter': 24,
    },
    'uniD': {
        'data_dir': 'universities-drone-dataset-v1.1/data',
        'map_dir': 'universities-drone-dataset-v1.1/maps/lanelet',
        'num_iter': 13,
    },
    'exiD': {
        'data_dir': 'entries-and-exits-drone-dataset-v2.1/data',
        'map_dir': 'entries-and-exits-drone-dataset-v2.1/maps/lanelet2',
        'num_iter': 33,
    },
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description='levelXData Converter')
    return add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/human-vehicle-dataset/human-vehicle-trajectory-dataset/levelXData',
        default_subset_name='levelxdata',
        default_raw_dataset_version='mixed',
        raw_dataset_help='The file directory for levelXData',
    )


def find_files_with_criteria(directory, criteria):
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if all(criterion in file for criterion in criteria):
                matching_files.append(os.path.join(root, file))
    return matching_files


def list_matching_files(raw_dataset_path):
    all_matching_files = []
    for dataset, config in LEVELX_DATASETS.items():
        directory_path = os.path.join(raw_dataset_path, config['data_dir'])
        for i in range(config['num_iter']):
            search_criteria = [f"{i:02d}", "csv"]
            matching_files = find_files_with_criteria(directory_path, search_criteria)
            if matching_files:
                all_matching_files.append(sorted(matching_files))
    return all_matching_files


def infer_levelx_dataset(path):
    normalized_path = path.replace(os.sep, '/')
    if 'highway-drone-dataset' in normalized_path:
        return 'highD'
    if 'intersections-drone-dataset' in normalized_path:
        return 'inD'
    if 'roundabouts-drone-dataset' in normalized_path:
        return 'rounD'
    if 'universities-drone-dataset' in normalized_path:
        return 'uniD'
    if 'entries-and-exits-drone-dataset' in normalized_path:
        return 'exiD'
    raise ValueError("Unknown levelXData path: {}".format(path))


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


def extract_object_tracks(raw_object_track_data_, scenario_idx_, scenario_length_):
    ret = dict()
    for object_id in raw_object_track_data_[0]['trackId'].unique():
        object_track_2 = raw_object_track_data_[0][raw_object_track_data_[0]['trackId'] == object_id]
        object_track_3 = raw_object_track_data_[1][raw_object_track_data_[1]['trackId'] == object_id]

        object_id = str(object_id)
        object_type = object_track_3['class'].item()
        object_state = object_state_template(scenario_length_)

        object_state['type'], object_state['subtype'] = get_object_type(object_type)
        object_state['phase'] = 'CONVERTED'
        for idx, frame in enumerate(range(scenario_idx_ * scenario_length_, (scenario_idx_ + 1) * scenario_length_)):
            if not object_track_2[object_track_2['frame'] == frame].empty:
                object_state['state']['global_center'][idx] = (object_track_2[object_track_2['frame'] == frame]['xCenter'].item(), object_track_2[object_track_2['frame'] == frame]['yCenter'].item(), 0.0)
                object_state['state']['size'][idx] = (object_track_3['length'].item(), object_track_3['width'].item(), 0.0)
                if 'heading' in object_track_2.columns:
                    object_state['state']['heading'][idx] = (object_track_2[object_track_2['frame'] == frame]['heading'].item()) * np.pi / 180
                object_state['state']['valid'][idx] = True
        ret[object_id] = object_state
    return ret


def get_speed_limit(speed_limit):
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

            relation_info['polyline'] = get_road_lane_polyline(left_road_line_polyline, right_road_line_polyline)
            relation_info['speed_limit'] = get_speed_limit(tags['speed_limit']) / 3.6
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


def process_scenario(params):
    args, file = params
    recordingMeta = file[0]
    tracks = file[1]
    tracksMeta = file[2]
    dataset_name = infer_levelx_dataset(recordingMeta)

    object_track1 = pd.read_csv(recordingMeta)
    object_track2 = pd.read_csv(tracks, low_memory=False)
    object_track3 = pd.read_csv(tracksMeta)

    random_name = str(len(object_track2))

    if 'id' in object_track2.columns:
        object_track2.rename(columns={'id': 'trackId'}, inplace=True)
        object_track2.rename(columns={'x': 'xCenter'}, inplace=True)
        object_track2.rename(columns={'y': 'yCenter'}, inplace=True)
    if 'id' in object_track3.columns:
        object_track3.rename(columns={'id': 'trackId'}, inplace=True)
        object_track3.rename(columns={'width': 'length'}, inplace=True)
        object_track3.rename(columns={'height': 'width'}, inplace=True)

    scenario_duration = 20
    scenario_num = int(object_track1['duration'].item() // scenario_duration)
    scenario_length = scenario_duration * 25

    for scenario_idx in range(scenario_num):
        scenario = dict()

        object_track2_part = object_track2[
            (object_track2['frame'] < scenario_idx * scenario_length + scenario_duration * 25) &
            (object_track2['frame'] >= scenario_idx * scenario_length)]
        raw_object_track_data = [object_track2_part, object_track3]
        converted_object_track_data = extract_object_tracks(raw_object_track_data, scenario_idx, scenario_length)

        locationId = object_track1['locationId'].item()

        if dataset_name == 'highD':
            map_file_name = ''
        elif dataset_name == 'inD':
            map_directory_path = os.path.join(args.raw_dataset_path, LEVELX_DATASETS[dataset_name]['map_dir'])
            if locationId == 1:
                map_file_name = '01_bendplatz'
            elif locationId == 2:
                map_file_name = '02_frankenburg'
            elif locationId == 3:
                map_file_name = '03_heckstrasse'
            elif locationId == 4:
                map_file_name = '04_aseag'
            else:
                raise ValueError("Wrong location ID: {}".format(locationId))
        elif dataset_name == 'rounD':
            map_directory_path = os.path.join(args.raw_dataset_path, LEVELX_DATASETS[dataset_name]['map_dir'])
            if locationId == 0:
                map_file_name = '0_neuweiler'
            elif locationId == 1:
                map_file_name = '1_kackertstrasse'
            elif locationId == 2:
                map_file_name = '2_thiergarten'
            else:
                raise ValueError("Wrong location ID: {}".format(locationId))
        elif dataset_name == 'uniD':
            map_directory_path = os.path.join(args.raw_dataset_path, LEVELX_DATASETS[dataset_name]['map_dir'])
            if locationId == 0:
                map_file_name = '0_superc'
            else:
                raise ValueError("Wrong location ID: {}".format(locationId))
        elif dataset_name == 'exiD':
            map_directory_path = os.path.join(args.raw_dataset_path, LEVELX_DATASETS[dataset_name]['map_dir'])
            if locationId == 0:
                map_file_name = '0_cologne_butzweiler'
            elif locationId == 1:
                map_file_name = '1_cologne_fortiib'
            elif locationId == 2:
                map_file_name = '2_aachen_brand'
            elif locationId == 3:
                map_file_name = '3_bergheim_roemer'
            elif locationId == 4:
                map_file_name = '4_cologne_klettenberg'
            elif locationId == 5:
                map_file_name = '5_aachen_laurensberg'
            elif locationId == 6:
                map_file_name = '6_merzenich_rather'
            else:
                raise ValueError("Wrong location ID: {}".format(locationId))
        else:
            raise ValueError("Unknown levelXData dataset: {}".format(dataset_name))

        if map_file_name:
            project_rule = {'proj': 'utm', 'ellps': 'WGS84', 'zone': 32, 'datum': 'WGS84'}
            projector = Proj(**project_rule)
            origin = (object_track1['xUtmOrigin'].item(), object_track1['yUtmOrigin'].item())

            raw_map_feature_data_path = map_directory_path + '/' + map_file_name + '/' + 'location' + str(
                locationId) + '.osm'
            raw_map_feature_data = ET.parse(raw_map_feature_data_path).getroot()

            polys = {}
            for node in raw_map_feature_data.findall('node'):
                new = projector(node.attrib['lon'], node.attrib['lat'])
                node_id = node.attrib['id']
                node_x, node_y = (new[0] - origin[0], new[1] - origin[1])
                polys[node_id] = (node_x, node_y)

            converted_map_feature_data = extract_map_features(raw_map_feature_data, polys)
        else:
            converted_map_feature_data = dict()

        converted_dynamic_map_state_data = dict()

        extra_information = dict()
        extra_information['id'] = random_name + '_' + str(scenario_idx)

        if dataset_name == 'highD':
            extra_information['dataset'] = 'highD'
            extra_information['version'] = '1.0'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'COLOGNE'
            extra_information['location'] = None
        elif dataset_name == 'inD' and locationId == 1:
            extra_information['dataset'] = 'inD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'BENDPLATZ'
        elif dataset_name == 'inD' and locationId == 2:
            extra_information['dataset'] = 'inD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'FRANKENBURG'
        elif dataset_name == 'inD' and locationId == 3:
            extra_information['dataset'] = 'inD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'HECKSTRASSE'
        elif dataset_name == 'inD' and locationId == 4:
            extra_information['dataset'] = 'inD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'ASEAG'
        elif dataset_name == 'rounD' and locationId == 0:
            extra_information['dataset'] = 'rounD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'NEUWEILER'
        elif dataset_name == 'rounD' and locationId == 1:
            extra_information['dataset'] = 'rounD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'KACKERTSTRASSE'
        elif dataset_name == 'rounD' and locationId == 2:
            extra_information['dataset'] = 'rounD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'ALSDORF'
            extra_information['location'] = 'THIERGARTEN'
        elif dataset_name == 'uniD':
            extra_information['dataset'] = 'uniD'
            extra_information['version'] = '1.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'SUPERC'
        elif dataset_name == 'exiD' and locationId == 0:
            extra_information['dataset'] = 'exiD'
            extra_information['version'] = '2.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'COLOGNE'
            extra_information['location'] = 'BUTZWEILER'
        elif dataset_name == 'exiD' and locationId == 1:
            extra_information['dataset'] = 'exiD'
            extra_information['version'] = '2.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'COLOGNE'
            extra_information['location'] = 'FORTIIB'
        elif dataset_name == 'exiD' and locationId == 2:
            extra_information['dataset'] = 'exiD'
            extra_information['version'] = '2.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'BRAND'
        elif dataset_name == 'exiD' and locationId == 3:
            extra_information['dataset'] = 'exiD'
            extra_information['version'] = '2.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'BERGHEIM'
            extra_information['location'] = 'ROEMER'
        elif dataset_name == 'exiD' and locationId == 4:
            extra_information['dataset'] = 'exiD'
            extra_information['version'] = '2.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'COLOGNE'
            extra_information['location'] = 'KLETTENBERG'
        elif dataset_name == 'exiD' and locationId == 5:
            extra_information['dataset'] = 'exiD'
            extra_information['version'] = '2.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'AACHEN'
            extra_information['location'] = 'LAURENSBERG'
        elif dataset_name == 'exiD' and locationId == 6:
            extra_information['dataset'] = 'exiD'
            extra_information['version'] = '2.1'
            extra_information['country'] = 'GERMANY'
            extra_information['city'] = 'MERZENICH'
            extra_information['location'] = 'RATHER'
        else:
            raise ValueError(f"Unknown recordingMeta: {recordingMeta} and location id: {locationId}")

        extra_information['sdc_id'] = None
        extra_information['scene_length'] = scenario_length
        extra_information['sampling_rate'] = 25
        extra_information['timestamp'] = np.array(range(scenario_length)) / 25

        scenario['object_tracks'] = converted_object_track_data
        scenario['map_features'] = converted_map_feature_data
        scenario['dynamic_map_states'] = converted_dynamic_map_state_data
        scenario['extra_information'] = extra_information

        save_scenario_pickle(scenario, args, extra_information['id'])


def convert(args):
    all_matching_files = list_matching_files(args.raw_dataset_path)
    params = [(args, file) for file in all_matching_files]
    run_tasks(process_scenario, params, len(all_matching_files), args, tqdm)


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
