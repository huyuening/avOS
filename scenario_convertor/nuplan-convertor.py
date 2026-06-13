# nuPlan Dataset

# Object tracks             yes (db)
#   Object track type:      VEHICLE, PEDESTRIAN, BICYCLE,
#                           TRAFFIC_CONE, BARRIER, CZONE_SIGN,
#                           GENERIC_OBJECT,
#                           EGO

# Map features              yes (gpkg)
#   Road line type:
#   Road lane type:
#   Road edge type:
#   Other type:             LANE, INTERSECTION, STOP_LINE, TURN_STOP, CROSSWALK, ...

# Dynamic map states        yes (gpkg)
#   Traffic light type:
#   Traffic light state:    GREEN, YELLOW, RED, UNKNOWN

# Installation (python_version >= 3.9)
# cd && git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
# pip install -e .
# pip install -r requirements_torch.txt
# pip install -r requirements.txt
# -i https://pypi.tuna.tsinghua.edu.cn/simple

# export NUPLAN_DATA_ROOT="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/nuplan-motion-dataset-v1.1/nuplan/dataset"
# export NUPLAN_MAPS_ROOT="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/nuplan-motion-dataset-v1.1/nuplan/dataset/maps"
# export NUPLAN_EXP_ROOT="/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/nuplan-motion-dataset-v1.1/nuplan/exp"

try:
    from scenario_convertor.internal import ensure_internal_converter_run, run_converter_main
except ModuleNotFoundError:
    from internal import ensure_internal_converter_run, run_converter_main

ensure_internal_converter_run()

import argparse
import geopandas as gpd
import hydra
import numpy as np
import nuplan
import os
import tempfile

from dataclasses import dataclass
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.utils import set_up_common_builder
from pathlib import Path
from shapely.ops import unary_union
from tqdm import tqdm
from utils import add_common_converter_args, object_state_template, run_tasks, save_scenario_pickle, traffic_light_state_template


def build_arg_parser():
    parser = argparse.ArgumentParser(description='nuPlan Dataset Converter')
    add_common_converter_args(
        parser,
        default_raw_dataset_path='/dataset/x-scene/autonomous-vehicle-dataset/autonomous-vehicle-motion-dataset/nuplan-motion-dataset-v1.1',
        default_subset_name='nuplan',
        default_raw_dataset_version='1.1',
        raw_dataset_help='The file directory for NPMD',
    )
    parser.add_argument(
        '--split',
        choices=('mini', 'trainval', 'test'),
        default='mini',
        help='The nuPlan split to convert.',
    )
    parser.add_argument(
        '--builder',
        default='nuplan',
        type=str,
        help='nuPlan scenario builder name.',
    )
    return parser

@dataclass
class HydraConfigPaths:
    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def construct_simulation_hydra_paths(base_config_path):
    common_dir = Path(base_config_path, 'config', 'common').resolve().as_uri()
    config_name = 'default_simulation'
    config_path = os.path.join(base_config_path, 'config', 'simulation')
    experiment_dir = Path(base_config_path, 'experiments').resolve().as_uri()
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def get_nuplan_scenarios(data_root_, map_root_, builder_):
    nuplan_package_path = os.path.dirname(nuplan.__file__)

    dataset_parameters = [
        "scenario_builder={}".format(builder_),
        "scenario_builder.scenario_mapping.subsample_ratio_override=0.5",
        "scenario_builder.data_root={}".format(data_root_),
        "scenario_builder.map_root={}".format(map_root_),

        "scenario_filter=all_scenarios",
        "scenario_filter.remove_invalid_goals=true",
        "scenario_filter.shuffle=false",
        "scenario_filter.timestamp_threshold_s=20"
    ]

    base_config_path = os.path.join(nuplan_package_path, "planning", "script")
    simulation_hydra_paths = construct_simulation_hydra_paths(base_config_path)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path)

    save_dir = tempfile.mkdtemp()
    ego_controller = 'perfect_tracking_controller'
    observation = 'box_observation'

    overrides = [
        f'group={save_dir}',
        'worker=sequential',
        f'ego_controller={ego_controller}',
        f'observation={observation}',
        f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
        'output_dir=${group}/${experiment}',
        'metric_dir=${group}/${experiment}',
        *dataset_parameters,
    ]
    overrides.extend(
        [
            f'job_name=planner_tutorial', 'experiment=${experiment_name}/${job_name}',
            f'experiment_name=planner_tutorial'
        ]
    )

    cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=overrides)

    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    scenario_builder = build_scenario_builder(cfg=cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)

    return scenario_builder.get_scenarios(scenario_filter, common_builder.worker)


def get_country_name(map_name):
    if map_name == 'us-nv-las-vegas-strip':
        return 'AMERICA'
    elif map_name == 'us-pa-pittsburgh-hazelwood':
        return 'AMERICA'
    elif map_name == 'us-ma-boston':
        return 'AMERICA'
    elif map_name == 'sg-one-north':
        return 'SINGAPORE'
    else:
        raise ValueError("Unknown map name: {}".format(map_name))


def get_city_name(map_name):
    if map_name == 'us-nv-las-vegas-strip':
        return 'LAS_VEGAS'
    elif map_name == 'us-pa-pittsburgh-hazelwood':
        return 'PITTSBURGH'
    elif map_name == 'us-ma-boston':
        return 'BOSTON'
    elif map_name == 'sg-one-north':
        return 'SINGAPORE'
    else:
        raise ValueError("Unknown map name: {}".format(map_name))


def get_object_type(object_type_):
    if object_type_ == 0:
        return 'VEHICLE', 'UNKNOWN'
    elif object_type_ == 1:
        return 'PEDESTRIAN', 'PEDESTRIAN'
    elif object_type_ == 2:
        return 'CYCLIST', 'BICYCLIST'
    elif object_type_ == 3:
        return 'STATIC', 'TRAFFIC_CONE'
    elif object_type_ == 4:
        return 'STATIC', 'BARRIER'
    elif object_type_ == 5 or object_type_ == 6:
        return 'UNKNOWN', 'UNKNOWN'
    elif object_type_ == 7:
        return 'VEHICLE', 'CAR'
    else:
        raise ValueError("Unknown object type: {}".format(object_type_))


def extract_object_tracks(raw_av_track_data_, raw_object_track_data_, scenario_length_):
    ret = dict()

    object_ids = []
    for objects_track in raw_object_track_data_:
        for object_track in objects_track:
            object_id = object_track.track_token
            object_ids.append(object_id)
    object_ids = list(set(object_ids))

    for object_id in object_ids:
        object_states = object_state_template(scenario_length_)
        for frame, objects_track in enumerate(raw_object_track_data_):
            object_states['phase'] = 'CONVERTED'
            for object_track in objects_track:
                object_type = object_track.tracked_object_type.value
                if object_track.track_token == object_id:
                    object_states['type'], object_states['subtype'] = get_object_type(object_type)
                    object_states['state']['global_center'][frame] = (object_track.center.x, object_track.center.y, 0.0)
                    object_states['state']['heading'][frame] = object_track.center.heading
                    object_states['state']['size'][frame] = (object_track.box.length, object_track.box.width, object_track.box.height)
                    object_states['state']['valid'][frame] = True
        ret[object_id] = object_states

    av_id = 'ego'
    av_states = object_state_template(scenario_length_)
    av_states['type'], av_states['subtype'] = 'VEHICLE', 'CAR'
    av_states['phase'] = 'CONVERTED'
    for frame, av_track in enumerate(raw_av_track_data_):
        av_states['state']['global_center'][frame] = (av_track.center.x, av_track.center.y, 0.0)
        av_states['state']['heading'][frame] = av_track.center.heading
        av_states['state']['size'][frame] = (av_track.car_footprint.length, av_track.car_footprint.width, av_track.car_footprint.height)
        av_states['state']['valid'][frame] = True
    ret[av_id] = av_states

    return ret


def extract_map_features(raw_map_feature_data_, scenario_center_):
    ret = dict()
    raw_map_feature_data_.initialize_all_layers()

    raw_map_features = [
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.LANE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ]

    radius = 150
    center_for_query = Point2D(*scenario_center_)
    nearby_map_features = raw_map_feature_data_.get_proximal_map_objects(center_for_query, radius, raw_map_features)

    road_lanes = nearby_map_features[SemanticMapLayer.LANE] + nearby_map_features[SemanticMapLayer.LANE_CONNECTOR]
    for road_lane in road_lanes:
        road_lane_info = dict()

        road_lane_id = road_lane.id
        road_lane_type = 'ROAD_LANE'
        road_lane_subtype = 'UNKNOWN'
        road_lane_polyline = np.column_stack(road_lane.baseline_path.linestring.xy)
        left_line_polyline = np.column_stack(road_lane.left_boundary.linestring.xy)
        right_line_polyline = np.column_stack(road_lane.right_boundary.linestring.xy)

        road_lane_polyline = np.hstack((road_lane_polyline, np.zeros((road_lane_polyline.shape[0], 1))))
        left_line_polyline = np.hstack((left_line_polyline, np.zeros((left_line_polyline.shape[0], 1))))
        right_line_polyline = np.hstack((right_line_polyline, np.zeros((right_line_polyline.shape[0], 1))))
        road_lane_speed_limit = road_lane.speed_limit_mps

        road_lane_info['type'] = road_lane_type
        road_lane_info['subtype'] = road_lane_subtype
        road_lane_info['polyline'] = road_lane_polyline
        road_lane_info['speed_limit'] = road_lane_speed_limit
        road_lane_info['is_intersection'] = None
        road_lane_info['phase'] = 'CONVERTED'

        road_lane_info['lane_info'] = dict()
        road_lane_info['lane_info']['entry_lane_ids'] = [entry_lane.id for entry_lane in road_lane.incoming_edges]
        road_lane_info['lane_info']['exit_lane_ids'] = [exit_lane.id for exit_lane in road_lane.outgoing_edges]
        road_lane_info['lane_info']['left_lane_ids'] = [road_lane.adjacent_edges[0].id] if road_lane.adjacent_edges[0] else []
        road_lane_info['lane_info']['right_lane_ids'] = [road_lane.adjacent_edges[1].id] if road_lane.adjacent_edges[1] else []

        road_lane_info['boundary_info'] = dict()
        road_lane_info['boundary_info']['left_boundary_id'] = road_lane.left_boundary.id
        road_lane_info['boundary_info']['right_boundary_id'] = road_lane.right_boundary.id
        road_lane_info['boundary_info']['left_boundary_polyline'] = left_line_polyline
        road_lane_info['boundary_info']['right_boundary_polyline'] = right_line_polyline

        ret[road_lane_id] = road_lane_info

    road_lines = nearby_map_features[SemanticMapLayer.LANE] + nearby_map_features[SemanticMapLayer.LANE_CONNECTOR]
    num_lanes = len(nearby_map_features[SemanticMapLayer.LANE])
    for i, road_line in enumerate(road_lines):
        left_road_line_info = dict()
        right_road_line_info = dict()
        left_road_line_id = road_line.left_boundary.id
        right_road_line_id = road_line.right_boundary.id
        left_road_line_polyline = np.column_stack(road_line.left_boundary.linestring.xy)
        right_road_line_polyline = np.column_stack(road_line.right_boundary.linestring.xy)
        left_road_line_polyline = np.hstack((left_road_line_polyline, np.zeros((left_road_line_polyline.shape[0], 1))))
        right_road_line_polyline = np.hstack((right_road_line_polyline, np.zeros((right_road_line_polyline.shape[0], 1))))

        left_road_line_info['type'] = 'ROAD_LINE'
        right_road_line_info['type'] = 'ROAD_LINE'
        if i < num_lanes:
            left_road_line_info['subtype'] = 'UNKNOWN'
            right_road_line_info['subtype'] = 'UNKNOWN'
        else:
            left_road_line_info['subtype'] = 'VIRTUAL'
            right_road_line_info['subtype'] = 'VIRTUAL'

        left_road_line_info['polyline'] = left_road_line_polyline
        right_road_line_info['polyline'] = right_road_line_polyline

        left_road_line_info['phase'] = 'CONVERTED'
        right_road_line_info['phase'] = 'CONVERTED'

        ret[left_road_line_id] = left_road_line_info
        ret[right_road_line_id] = right_road_line_info

    for stop_line in nearby_map_features[SemanticMapLayer.STOP_LINE]:
        stop_line_info = dict()

        stop_line_id = stop_line.id
        stop_line_polygon = np.column_stack(stop_line.polygon.boundary.xy)
        stop_line_polygon = np.hstack((stop_line_polygon, np.zeros((stop_line_polygon.shape[0], 1))))

        stop_line_info['type'] = 'OTHER'
        stop_line_info['subtype'] = 'STOP_LINE'
        stop_line_info['polygon'] = stop_line_polygon
        stop_line_info['phase'] = 'CONVERTED'

        ret[stop_line_id] = stop_line_info

    for crosswalk in nearby_map_features[SemanticMapLayer.CROSSWALK]:
        crosswalk_info = dict()
        crosswalk_id = crosswalk.id
        crosswalk_polygon = np.column_stack(crosswalk.polygon.exterior.xy)
        crosswalk_polygon = np.hstack((crosswalk_polygon, np.zeros((crosswalk_polygon.shape[0], 1))))

        crosswalk_info['type'] = 'OTHER'
        crosswalk_info['subtype'] = 'CROSSWALK'
        crosswalk_info['polygon'] = crosswalk_polygon
        crosswalk_info['phase'] = 'CONVERTED'

        ret[crosswalk_id] = crosswalk_info

    for car_park in nearby_map_features[SemanticMapLayer.CARPARK_AREA]:
        car_park_info = dict()
        car_park_id = car_park.id
        car_park_polygon = np.column_stack(car_park.polygon.exterior.xy)
        car_park_polygon = np.hstack((car_park_polygon, np.zeros((car_park_polygon.shape[0], 1))))

        car_park_info['type'] = 'OTHER'
        car_park_info['subtype'] = 'PARKING'
        car_park_info['polygon'] = car_park_polygon
        car_park_info['phase'] = 'CONVERTED'

        ret[car_park_id] = car_park_info

    for walkway in nearby_map_features[SemanticMapLayer.WALKWAYS]:
        walkway_info = dict()
        walkway_id = walkway.id
        walkway_polygon = np.column_stack(walkway.polygon.exterior.xy)
        walkway_polygon = np.hstack((walkway_polygon, np.zeros((walkway_polygon.shape[0], 1))))

        walkway_info['type'] = 'OTHER'
        walkway_info['subtype'] = 'SIDEWALK'
        walkway_info['polygon'] = walkway_polygon
        walkway_info['phase'] = 'CONVERTED'

        ret[walkway_id] = walkway_info

    road_edge_1 = [intersection.polygon for intersection in nearby_map_features[SemanticMapLayer.INTERSECTION]]
    road_edge_2 = [road_block.polygon for road_block in nearby_map_features[SemanticMapLayer.ROADBLOCK]]
    road_edges = gpd.GeoSeries(unary_union(road_edge_1+road_edge_2)).boundary.explode(index_parts=True)
    for road_edge_id, road_edge in enumerate(road_edges[0]):
        block_points = np.array(list(i for i in zip(road_edge.coords.xy[0], road_edge.coords.xy[1])))
        block_points = np.hstack((block_points, np.zeros((block_points.shape[0], 1))))
        ret[str(road_edge_id)] = {'type': 'ROAD_EDGE', 'subtype': 'UNKNOWN', 'polyline': block_points, 'phase': 'CONVERTED'}

    return ret


def extract_stop_point(traffic_light_id_, raw_map_feature_data_):
    traffic_light_control_lane = raw_map_feature_data_.get_map_object(traffic_light_id_, SemanticMapLayer.LANE_CONNECTOR)
    stop_point = np.column_stack(traffic_light_control_lane.baseline_path.linestring.xy)[0, :]

    return stop_point


def get_traffic_light_state(traffic_light_state_):
    if traffic_light_state_.value == 0:
        return 'GREEN'
    elif traffic_light_state_.value == 1:
        return 'YELLOW'
    elif traffic_light_state_.value == 2:
        return 'RED'
    elif traffic_light_state_.value == 3:
        return 'UNKNOWN'
    else:
        raise ValueError("Unknown traffic light state: {}".format(traffic_light_state_))


def extract_dynamic_map_states(raw_dynamic_map_state_data_, raw_map_feature_data_, scenario_length_):
    ret = dict()

    for frame, dynamic_map_state in enumerate(raw_dynamic_map_state_data_):
        for traffic_light_id, traffic_light_state in dynamic_map_state.items():
            traffic_light_type = 'TRAFFIC_LIGHT'
            traffic_light_subtype = 'UNKNOWN'
            stop_point = extract_stop_point(traffic_light_id, raw_map_feature_data_)

            if traffic_light_id not in ret:
                ret[traffic_light_id] = traffic_light_state_template(traffic_light_id, scenario_length_)
            ret[traffic_light_id]['type'] = traffic_light_type
            ret[traffic_light_id]['subtype'] = traffic_light_subtype
            ret[traffic_light_id]['state']['color'][frame] = get_traffic_light_state(traffic_light_state)
            ret[traffic_light_id]['state']['valid'][frame] = True
            ret[traffic_light_id]['stop_point'][0] = stop_point[0]
            ret[traffic_light_id]['stop_point'][1] = stop_point[1]
            ret[traffic_light_id]['controlled_lane'] = None
            ret[traffic_light_id]['phase'] = 'CONVERTED'

    return ret


def process_scenario(params):
    args, scenario = params
    scenario_ = dict()

    scenario_id = scenario.token

    scenario_length = scenario.get_number_of_iterations()

    init_time = round(scenario.start_time.time_us / 100000) * 100000

    raw_av_track_data = [scenario.get_ego_state_at_iteration(i) for i in range(scenario_length)]
    raw_object_track_data = [scenario.get_tracked_objects_at_iteration(i).tracked_objects for i in range(scenario_length)]
    converted_object_track_data = extract_object_tracks(raw_av_track_data, raw_object_track_data, scenario_length)

    scenario_center = (converted_object_track_data['ego']['state']['global_center'][int(scenario_length / 2), 0],
                       converted_object_track_data['ego']['state']['global_center'][int(scenario_length / 2), 1])

    raw_map_feature_data = scenario.map_api
    converted_map_feature_data = extract_map_features(raw_map_feature_data, scenario_center)

    raw_dynamic_map_state_data = [
        {str(t.lane_connector_id): t.status
         for t in scenario.get_traffic_light_status_at_iteration(i)} for i in range(scenario_length)
    ]

    converted_dynamic_map_state_data = extract_dynamic_map_states(raw_dynamic_map_state_data, raw_map_feature_data, scenario_length)

    extra_information = dict()
    extra_information['id'] = scenario_id
    extra_information['dataset'] = 'nuPlan'
    extra_information['version'] = args.raw_dataset_version
    extra_information['country'] = get_country_name(scenario.map_api.map_name)
    extra_information['city'] = get_city_name(scenario.map_api.map_name)
    extra_information['location'] = None
    extra_information['sdc_id'] = 'ego'
    extra_information['scene_length'] = scenario_length
    extra_information['sampling_rate'] = 10
    extra_information['timestamp'] = np.array(range(scenario_length)) / 10
    extra_information['init_time'] = init_time

    scenario_['object_tracks'] = converted_object_track_data
    scenario_['map_features'] = converted_map_feature_data
    scenario_['dynamic_map_states'] = converted_dynamic_map_state_data
    scenario_['extra_information'] = extra_information

    save_scenario_pickle(scenario_, args, scenario_id)


def convert(args):
    data_root = os.path.join(args.raw_dataset_path, 'nuplan/dataset/nuplan-v1.1/splits', args.split)
    map_root = os.path.join(args.raw_dataset_path, 'nuplan/dataset/maps')
    scenarios = get_nuplan_scenarios(data_root, map_root, args.builder)

    params = [(args, scenario) for scenario in scenarios]
    run_tasks(
        process_scenario,
        params,
        len(scenarios),
        args,
        tqdm,
        pool_options={'maxtasksperchild': 10},
    )


def main(argv=None):
    return run_converter_main(build_arg_parser, convert, argv)


if __name__ == "__main__":
    main()
