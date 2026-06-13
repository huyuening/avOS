# Lyft Level 5 Prediction Dataset -> Lyft Level 5 Scenario Dataset (L5SD)

# Information
# object_tracks         yes (zarr)
#   ObjectType: CAR, VAN, TRAM, BUS, TRUCK, EMERGENCY_VEHICLE, OTHER_VEHICLE, CYCLIST, MOTORCYCLIST, PEDESTRIAN
#               BICYCLE, MOTORCYCLE
#               ANIMAL
#               NOT_SET, UNKNOWN, DONTCARE1, DONTCARE2

# map_features          yes (pb)
#

# dynamic_map_states    no
#

# Installation
# 1) conda create -n Xdata python=3.8
# 2) conda activate Xdata
# 3) pip install l5kit -i https://pypi.tuna.tsinghua.edu.cn/simple
# 4) pip install protobuf==3.19.6 -i https://pypi.tuna.tsinghua.edu.cn/simple

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import numpy as np
import warnings

from l5kit.configs import load_config_data
from l5kit.configs.config import load_metadata
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.map_api import MapAPI
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from tqdm import tqdm
from utils import add_common_converter_args, get_road_lane_polyline, object_state_template, run_tasks, save_scenario_pickle


warnings.filterwarnings("ignore")

data_path = None
train = False
scenarios = None
av_dataset = None


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Lyft Level 5 Prediction Dataset Converter')
    add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/lyft-level-5-motion-dataset',
        default_subset_name='lyft-level-5',
        default_raw_dataset_version='1.1',
        raw_dataset_help='The file directory for L5PD',
    )
    parser.add_argument(
        '--split',
        choices=('train', 'val'),
        default='val',
        help='The Lyft split to convert.',
    )
    parser.add_argument(
        '--visualisation_config_path',
        default='../visualisation_config.yaml',
        type=str,
        help='Path to the l5kit visualisation config.',
    )
    return parser


def initialize_converter(args):
    global data_path, train, scenarios, av_dataset
    data_path = args.raw_dataset_path.rstrip('/') + '/'
    train = args.split == 'train'
    if train:
        scenarios = ChunkedDataset(data_path + "train_full.zarr").open()
    else:
        scenarios = ChunkedDataset(data_path + "validate.zarr").open()

    cfg = load_config_data(args.visualisation_config_path)
    rast = build_rasterizer(cfg, LocalDataManager(data_path))
    av_dataset = EgoDataset(cfg, scenarios, rast)

MAX_DISTANCE = 150


def get_road_lane_type_and_subtype(road_lane_type):
    if road_lane_type == 0 or road_lane_type == 5:
        return 'ROAD_LANE', 'URBAN_UNKNOWN'
    elif road_lane_type == 1:
        return 'ROAD_LANE', 'URBAN_ALL'
    elif road_lane_type == 2:
        return 'ROAD_LANE', 'URBAN_VEHICLE'
    elif road_lane_type == 3:
        return 'ROAD_LANE', 'URBAN_BUS'
    elif road_lane_type == 4:
        return 'ROAD_LANE', 'URBAN_BIKE'
    else:
        return ValueError("Unknown road lane type: {}".format(road_lane_type))


def get_road_line_edge_type(road_line_edge_type):
    if road_line_edge_type:
        if road_line_edge_type[0] == 0 or road_line_edge_type[0] == 1:
            return 'ROAD_LINE', 'UNKNOWN'
        elif road_line_edge_type[0] == 2:
            return 'ROAD_LINE', 'SOLID_YELLOW'
        elif road_line_edge_type[0] == 3:
            return 'ROAD_LINE', 'SOLID_WHITE'
        elif road_line_edge_type[0] == 4:
            return 'ROAD_LINE', 'DASHED_YELLOW'
        elif road_line_edge_type[0] == 5:
            return 'ROAD_LINE', 'DASHED_WHITE'
        elif road_line_edge_type[0] == 6:
            return 'ROAD_LINE', 'DOUBLE_SOLID_YELLOW'
        elif road_line_edge_type[0] == 7:
            return 'ROAD_LINE', 'DOUBLE_SOLID_WHITE'
        elif road_line_edge_type[0] == 8:
            return 'ROAD_LINE', 'SOLID_DASH_YELLOW'
        elif road_line_edge_type[0] == 9:
            return 'ROAD_LINE', 'DASH_SOLID_YELLOW'
        elif road_line_edge_type[0] == 10 or road_line_edge_type[0] == 11 or road_line_edge_type[0] == 12:
            return 'ROAD_EDGE', 'UNKNOWN'
        else:
            raise ValueError("Unknown road line or road edge type: {}".format(road_line_edge_type))
    else:
        return 'ROAD_LINE', 'UNKNOWN'


def extract_road_lines_edges(road_line, road_line_polyline):
    road_line_edge_info = dict()
    road_line_edge_info['type'], road_line_edge_info['subtype'] = get_road_line_edge_type(road_line.divider_type)
    road_line_edge_info['polyline'] = road_line_polyline
    road_line_edge_info['phase'] = 'CONVERTED'
    return road_line_edge_info


def filter_map_features_by_distance(map_features: np.ndarray, centroid: np.ndarray, max_distance: float) -> np.ndarray:
    """Filter agents by distance, cut to `max_distance` and sort the result
    Args:
        map_features (np.ndarray): array of map_features
        centroid (np.ndarray): centroid towards which compute distance
        max_distance (float): max distance to cut off
    Returns:
        np.ndarray: agents sorted and cut to max_distance
    """
    agents_dist = np.linalg.norm(map_features["centroid"] - centroid, axis=-1)
    agents = map_features[agents_dist < max_distance]
    agents_dist = agents_dist[agents_dist < max_distance]
    agents = agents[np.argsort(agents_dist)]
    return agents


def extract_map_features(raw_map_feature_data_, scenario_center_, mapAPI):
    ret = dict()
    for map_feature in raw_map_feature_data_[0]:
        if mapAPI.is_lane(map_feature):
            road_lane_id = mapAPI.id_as_str(map_feature.id)

            left_road_line_edge_id = mapAPI.id_as_str(map_feature.id) + '_1'
            right_road_line_edge_id = mapAPI.id_as_str(map_feature.id) + '_2'
            left_road_line_edge_polyline = mapAPI.get_lane_coords(road_lane_id)['xyz_left']
            right_road_line_edge_polyline = mapAPI.get_lane_coords(road_lane_id)['xyz_right']

            if np.linalg.norm(left_road_line_edge_polyline[int(len(left_road_line_edge_polyline) / 2)][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE and \
               np.linalg.norm(right_road_line_edge_polyline[int(len(right_road_line_edge_polyline) / 2)][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE:
                left_road_line_edge = extract_road_lines_edges(map_feature.element.lane.left_boundary,
                                                               left_road_line_edge_polyline)
                right_road_line_edge = extract_road_lines_edges(map_feature.element.lane.right_boundary,
                                                                right_road_line_edge_polyline)

                ret[left_road_line_edge_id] = left_road_line_edge
                ret[right_road_line_edge_id] = right_road_line_edge

                road_lane_info = dict()

                road_lane_info['type'], road_lane_info['subtype'] = get_road_lane_type_and_subtype(map_feature.element.lane.access_restriction.type)
                road_lane_info['polyline'] = get_road_lane_polyline(ret[left_road_line_edge_id]['polyline'], ret[right_road_line_edge_id]['polyline'])
                road_lane_info['speed_limit'] = None
                road_lane_info['is_intersection'] = None
                road_lane_info['phase'] = 'CONVERTED'

                road_lane_info['lane_info'] = dict()
                road_lane_info['lane_info']['entry_lane_ids'] = dict()
                road_lane_info['lane_info']['exit_lane_ids'] = dict()
                road_lane_info['lane_info']['left_lane_ids'] = dict()
                road_lane_info['lane_info']['right_lane_ids'] = dict()

                road_lane_info['boundary_info'] = dict()
                road_lane_info['boundary_info']['left_boundary_id'] = left_road_line_edge_id
                road_lane_info['boundary_info']['right_boundary_id'] = right_road_line_edge_id
                road_lane_info['boundary_info']['left_boundary_polyline'] = ret[left_road_line_edge_id]['polyline']
                road_lane_info['boundary_info']['right_boundary_polyline'] = ret[right_road_line_edge_id]['polyline']

                ret[road_lane_id] = road_lane_info

        if mapAPI.is_crosswalk(map_feature):
            crosswalk_info = dict()

            crosswalk_id = mapAPI.id_as_str(map_feature.id)
            crosswalk_polygon = mapAPI.get_crosswalk_coords(crosswalk_id)['xyz']
            crosswalk_info['type'] = 'OTHER'
            crosswalk_info['subtype'] = 'CROSSWALK'
            crosswalk_info['polygon'] = crosswalk_polygon
            crosswalk_info['phase'] = 'CONVERTED'

            if np.linalg.norm(crosswalk_polygon[0][0:2] - scenario_center_[0:2], axis=-1) <= MAX_DISTANCE:
                ret[crosswalk_id] = crosswalk_info

    return ret


def get_object_type(object_type_):
    if np.argmax(object_type_) == 0 or np.argmax(object_type_) == 1 or np.argmax(object_type_) == 2 or \
       np.argmax(object_type_) == 16:
        return 'UNKNOWN', 'UNKNOWN'
    elif np.argmax(object_type_) == 3:
        return 'VEHICLE', 'CAR'
    elif np.argmax(object_type_) == 4:
        return 'VEHICLE', 'VAN'
    elif np.argmax(object_type_) == 5:
        return 'VEHICLE', 'TRAM'
    elif np.argmax(object_type_) == 6:
        return 'VEHICLE', 'BUS'
    elif np.argmax(object_type_) == 7:
        return 'VEHICLE', 'TRUCK'
    elif np.argmax(object_type_) == 8:
        return 'VEHICLE', 'EMERGENCY'
    elif np.argmax(object_type_) == 9:
        return 'VEHICLE', 'UNKNOWN'
    elif np.argmax(object_type_) == 10 or np.argmax(object_type_) == 12:
        return 'CYCLIST', 'BICYCLIST'
    elif np.argmax(object_type_) == 11 or np.argmax(object_type_) == 13:
        return 'CYCLIST', 'MOTORCYCLIST'
    elif np.argmax(object_type_) == 14:
        return 'PEDESTRIAN', 'PEDESTRIAN'
    elif np.argmax(object_type_) == 15:
        return 'OTHER', 'ANIMAL'
    else:
        raise ValueError("Unknown object type: {}".format(object_type_))


def extract_object_tracks(raw_agent_track_data_, agent_range_detail_, raw_av_track_data_, scenario_length_):
    ret = dict()
    av_id = str(raw_av_track_data_[0]['track_id'])
    av_state = object_state_template(scenario_length_)
    av_state['type'], av_state['subtype'] = 'VEHICLE', 'CAR'
    av_state['phase'] = 'CONVERTED'
    for frame in range(0, scenario_length_):
        av_state['state']['global_center'][frame] = (raw_av_track_data_[frame]['centroid'][0], raw_av_track_data_[frame]['centroid'][1], 0.0)
        av_state['state']['heading'][frame] = raw_av_track_data_[frame]['yaw']
        av_state['state']['size'][frame] = raw_av_track_data_[frame]['extent']
        av_state['state']['valid'][frame] = True
    ret[av_id] = av_state

    for object_id in np.unique(raw_agent_track_data_['track_id']):
        object_type = raw_agent_track_data_[raw_agent_track_data_['track_id'] == object_id]['label_probabilities'][0, :]
        object_state = object_state_template(scenario_length_)
        object_state['type'], object_state['subtype'] = get_object_type(object_type)
        object_state['phase'] = 'CONVERTED'
        for frame, agent_range in enumerate(agent_range_detail_):
            agent_range_0 = agent_range_detail_[0, 0]
            object_ids_per_timestep = raw_agent_track_data_['track_id'][agent_range[0]-agent_range_0: agent_range[1]-agent_range_0]
            agent_tracks_per_timestep = raw_agent_track_data_[agent_range[0]-agent_range_0: agent_range[1]-agent_range_0]
            if np.isin(object_id, object_ids_per_timestep):
                object_id_idx = np.where(object_ids_per_timestep == object_id)[0][0]
                object_state['state']['global_center'][frame] = (agent_tracks_per_timestep['centroid'][object_id_idx][0], agent_tracks_per_timestep['centroid'][object_id_idx][1], 0.0)
                object_state['state']['heading'][frame] = agent_tracks_per_timestep['yaw'][object_id_idx]
                object_state['state']['size'][frame] = (agent_tracks_per_timestep['extent'][object_id_idx][0], agent_tracks_per_timestep['extent'][object_id_idx][1], agent_tracks_per_timestep['extent'][object_id_idx][2])
                object_state['state']['valid'][frame] = True
        ret[str(object_id)] = object_state
    return ret


def process_scenario(params):
    args, scenario_id = params
    scenario_ = dict()

    frame_range = scenarios.scenes[scenario_id][0]
    scenario_length = frame_range[1] - frame_range[0]
    timestamp = scenarios.frames['timestamp'][frame_range[0]:frame_range[1]] / 1000000000
    if timestamp[-1] - timestamp[0] == 0:
        timestamp = np.array(range(len(timestamp))) / 10

    agent_range = [scenarios.frames[frame_range[0]][1][0], scenarios.frames[frame_range[1]][1][0]]
    agent_range_detail = scenarios.frames['agent_index_interval'][frame_range[0]:frame_range[1]]

    raw_agent_track_data = scenarios.agents[agent_range[0]: agent_range[1]]
    raw_av_track_data = av_dataset.get_scene_dataset(scenario_id)

    converted_object_track_data = extract_object_tracks(raw_agent_track_data, agent_range_detail, raw_av_track_data, scenario_length)

    sdc_id = str(raw_av_track_data[0]['track_id'])
    scenario_center = converted_object_track_data[sdc_id]['state']['global_center'][int(scenario_length / 2)]

    dm = LocalDataManager(data_path)
    dataset_meta = load_metadata(dm.require("meta.json"))
    semantic_map_filepath = dm.require("semantic_map/semantic_map.pb")
    world_to_ecef = np.array(dataset_meta["world_to_ecef"])
    mapAPI = MapAPI(semantic_map_filepath, world_to_ecef)
    raw_map_feature_data = [mapAPI.elements, mapAPI.bounds_info]
    converted_map_feature_data = extract_map_features(raw_map_feature_data, scenario_center, mapAPI)

    converted_dynamic_map_state_data = dict()

    extra_information = dict()
    if train:
        extra_information['id'] = 't_' + str(scenario_id)
    else:
        extra_information['id'] = 'v_' + str(scenario_id)
    extra_information['dataset'] = 'Lyft Level 5'
    extra_information['version'] = args.raw_dataset_version
    extra_information['country'] = 'AMERICA'
    extra_information['city'] = 'PALO_ALTO'
    extra_information['location'] = None
    extra_information['sdc_id'] = sdc_id
    extra_information['scene_length'] = scenario_length
    extra_information['sampling_rate'] = 10
    extra_information['timestamp'] = timestamp

    scenario_['object_tracks'] = converted_object_track_data
    scenario_['map_features'] = converted_map_feature_data
    scenario_['dynamic_map_states'] = converted_dynamic_map_state_data
    scenario_['extra_information'] = extra_information

    save_scenario_pickle(scenario_, args, extra_information['id'])


def convert(args):
    scenario_ids = list(range(134622)) if args.split == 'train' else list(range(16220))
    params = [(args, scenario_id) for scenario_id in scenario_ids]
    run_tasks(
        process_scenario,
        params,
        len(scenario_ids),
        args,
        tqdm,
        initializer=initialize_converter,
        initargs=(args,),
        pool_options={'maxtasksperchild': 10},
    )


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
