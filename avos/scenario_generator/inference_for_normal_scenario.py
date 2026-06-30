import argparse
import os
import pickle
import random

import numpy as np
import torch

from config_paths import require_optimized_path, require_trainval_path, set_optimized_path
from inference.common import (
    build_generated_scenario,
    infer_scenes,
    load_config,
    load_model,
    make_dataloader,
    reconstruct_generated_scenario,
    require_cached_samples,
    resolve_device,
    save_scenario,
)
from inference.defaults import (
    config_path_value,
    config_value,
    default_config_path,
    default_infer_config_path,
    default_weight_path,
    load_infer_config,
)


DENSITY_LABELS = {
    1: "dense (>= 32 agents)",
    2: "medium (8-31 agents)",
    3: "sparse (1-7 agents)",
}

ROAD_TYPE_LABELS = {
    1: "intersection",
    2: "intersection approach",
    3: "road segment",
}


def build_parser():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--infer-config", "--infer_config", dest="infer_config", default=default_infer_config_path())
    known_args, _ = base_parser.parse_known_args()
    infer_config = load_infer_config(known_args.infer_config)

    parser = argparse.ArgumentParser(
        description="Generate one normal scenario by road type and traffic density.",
        parents=[base_parser],
    )
    parser.add_argument("--config-path", default=config_path_value(infer_config, "CONFIG_PATH", default=default_config_path()))
    parser.add_argument("--weight-path", default=config_path_value(infer_config, "WEIGHT_PATH", default=default_weight_path()))
    parser.add_argument("--input-path1", "--input_path1", dest="map_code_path",
                        default=config_path_value(infer_config, "DATA", "MAP_CODE_PATH", default="data/inference_data/map_codes"))
    parser.add_argument("--input-path2", "--input_path2", dest="source_path",
                        default=config_path_value(
                            infer_config,
                            "DATA",
                            "OPTIMIZED_PATH",
                            default=config_value(infer_config, "DATA", "SOURCE_PATH", default="datasets/optimized_dataset"),
                        ),
                        help="Root directory that contains <subset_name>/<scenario_id>.pkl.")
    parser.add_argument("--subset-name", "--subset_name", dest="subset_name",
                        default=config_value(infer_config, "SUBSET_NAME", default="waymo-open"))
    parser.add_argument("--output-path1", "--output_path1", dest="output_path",
                        default=config_path_value(infer_config, "OUTPUT", "SCENARIO_PATH", default="datasets/generated_dataset"))
    parser.add_argument("--road-type", type=int, default=config_value(infer_config, "NORMAL", "ROAD_TYPE", default=None),
                        help="1: intersection, 2: intersection approach, 3: road segment.")
    parser.add_argument("--density-type", type=int, default=config_value(infer_config, "NORMAL", "DENSITY_TYPE", default=None),
                        help="1: dense, 2: medium, 3: sparse.")
    parser.add_argument("--max-density-candidates", type=int,
                        default=config_value(infer_config, "NORMAL", "MAX_DENSITY_CANDIDATES", default=20),
                        help="Stop density scanning after this many matched candidates. Use 0 to scan all.")
    parser.add_argument("--reconstructor",
                        default=config_value(
                            infer_config,
                            "NORMAL",
                            "RECONSTRUCTOR",
                            default=config_value(infer_config, "RECONSTRUCTOR", default="motion_reconstructor_v2"),
                        ),
                        help="Trajectory reconstructor: motion_reconstructor_v2.")
    parser.add_argument("--device", default=config_value(infer_config, "DEVICE", default=None))
    parser.add_argument("--seed", type=int, default=config_value(infer_config, "SEED", default=None))
    parser.add_argument("--verbose", action="store_true",
                        default=config_value(infer_config, "NORMAL", "VERBOSE", default=False),
                        help="Print detailed normal inference progress logs.")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                        help="Suppress detailed normal inference progress logs.")

    return parser


def log(args, message):
    if getattr(args, "verbose", False):
        print(message, flush=True)


def prompt_for_int(value, prompt, valid_values):
    if value is None:
        value = int(input(prompt))
    if value not in valid_values:
        raise ValueError(f"Expected one of {sorted(valid_values)}, got {value}.")
    return value


def map_matching(map_codes, road_type):
    matched_scenario_ids = []
    for scenario_id, map_code in map_codes.items():
        map_code_1 = map_code[0]
        if map_code_1[-1] == road_type:
            matched_scenario_ids.append(scenario_id)
    return matched_scenario_ids


def density_matches(num_agents, density_type):
    if density_type == 1:
        return num_agents >= 32
    if density_type == 2:
        return 8 <= num_agents < 32
    if density_type == 3:
        return 1 <= num_agents < 8
    raise ValueError(f"Unsupported density_type: {density_type}")


def source_root_from_config(config, subset_name):
    dataset_path = require_optimized_path(config)
    if os.path.basename(dataset_path) == subset_name:
        return os.path.dirname(dataset_path)
    return dataset_path


def filter_available_ids(config, scenario_ids):
    processed_path = require_trainval_path(config)
    available_ids = []
    for scenario_id in scenario_ids:
        pt_path = os.path.join(processed_path, scenario_id + ".pt")
        if os.path.exists(pt_path):
            available_ids.append(scenario_id)
    return available_ids


def filter_source_ids(source_root, subset_name, scenario_ids):
    available_ids = []
    for scenario_id in scenario_ids:
        pkl_path = os.path.join(source_root, subset_name, scenario_id + ".pkl")
        if os.path.exists(pkl_path):
            available_ids.append(scenario_id)
    return available_ids


def load_map_codes(map_code_path, subset_name):
    candidates = [
        os.path.join(map_code_path, "map_codes.pkl"),
        os.path.join(map_code_path, subset_name, "map_codes.pkl"),
    ]
    map_code_file = next((path for path in candidates if os.path.exists(path)), candidates[0])
    with open(map_code_file, "rb") as file:
        return pickle.load(file)


def resolve_map_code_file(map_code_path, subset_name):
    candidates = [
        os.path.join(map_code_path, "map_codes.pkl"),
        os.path.join(map_code_path, subset_name, "map_codes.pkl"),
    ]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


def load_source_scenario(source_root, subset_name, scenario_id):
    with open(os.path.join(source_root, subset_name, scenario_id) + ".pkl", "rb") as file:
        return pickle.load(file)


def effective_agent_count(source_scenario, config):
    object_tracks = source_scenario["object_tracks"]
    sdc_id = source_scenario["extra_information"]["sdc_id"]
    object_ids = [sdc_id] + [object_id for object_id in object_tracks if object_id != sdc_id]
    max_timestep = config["MAX_TIMESTEP"]
    object_range = config["OBJECT_RANGE"]
    max_object_num = config["MAX_OBJECT_NUM"]
    scene_length = min(max_timestep, len(object_tracks[sdc_id]["state"]["valid"]))

    centers = np.zeros((scene_length, len(object_ids), 2))
    headings = np.zeros((scene_length, len(object_ids)))
    valid = np.zeros((scene_length, len(object_ids)), dtype=bool)
    for object_idx, object_id in enumerate(object_ids):
        state = object_tracks[object_id]["state"]
        centers[:, object_idx] = state["global_center"][:scene_length, :2]
        headings[:, object_idx] = state["heading"][:scene_length]
        valid[:, object_idx] = state["valid"][:scene_length].astype(bool)

    ego_pos = centers[0, 0].reshape(1, 1, 2)
    ego_heading = headings[0, 0]
    rel_xy = centers - ego_pos
    rel_x = np.cos(-ego_heading) * rel_xy[..., 0] - np.sin(-ego_heading) * rel_xy[..., 1]
    rel_y = np.sin(-ego_heading) * rel_xy[..., 0] + np.cos(-ego_heading) * rel_xy[..., 1]
    first_frame_mask = (
        valid[0]
        & (np.abs(rel_x[0]) < object_range)
        & (np.abs(rel_y[0]) < object_range)
    )
    return min(int(np.sum(first_frame_mask)), max_object_num)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        log(args, f"Setting random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    road_type = prompt_for_int(
        args.road_type,
        "Road type (intersection (1), intersection approach (2), road segment (3)): ",
        {1, 2, 3},
    )
    density_type = prompt_for_int(
        args.density_type,
        "Density type (dense (1), medium (2), sparse (3)): ",
        {1, 2, 3},
    )

    log(args, "=== Normal scenario inference ===")
    log(args, f"Road type: {road_type} ({ROAD_TYPE_LABELS[road_type]})")
    log(args, f"Density type: {density_type} ({DENSITY_LABELS[density_type]})")
    log(args, f"Config path: {args.config_path}")
    log(args, f"Weight path: {args.weight_path}")
    log(args, f"Map-code root: {args.map_code_path}")
    log(args, f"Optimized source root: {args.source_path}")
    log(args, f"Output root: {args.output_path}")

    log(args, "Loading train config...")
    config = load_config(args.config_path)
    source_root = args.source_path or source_root_from_config(config, args.subset_name)
    config.setdefault("DATASET", {})
    set_optimized_path(config, os.path.join(source_root, args.subset_name))
    log(args, f"Resolved optimized source root: {source_root}")
    log(args, f"Resolved optimized scenario dir: {config['DATASET']['OPTIMIZED_PATH']}")
    log(args, f"Resolved trainval cache dir: {require_trainval_path(config)}")

    map_code_file = resolve_map_code_file(args.map_code_path, args.subset_name)
    log(args, f"Loading map codes: {map_code_file}")
    map_codes = load_map_codes(args.map_code_path, args.subset_name)
    log(args, f"Loaded map-code entries: {len(map_codes)}")

    log(args, "Matching road type from map codes...")
    matched_ids = map_matching(map_codes, road_type)
    log(args, f"Road-type matched IDs before source-file filtering: {len(matched_ids)}")

    log(args, "Checking optimized .pkl availability...")
    source_ids = filter_source_ids(source_root, args.subset_name, matched_ids)
    log(args, f"Available optimized .pkl source IDs: {len(source_ids)}")
    if not source_ids:
        raise RuntimeError(
            f"No optimized .pkl files matched road_type={road_type}. "
            f"Matched {len(matched_ids)} map-code IDs before filtering by DATA.OPTIMIZED_PATH."
        )

    random.shuffle(source_ids)
    candidates = []
    max_density_candidates = args.max_density_candidates
    log(args, "Scanning source scenarios for density match...")
    if max_density_candidates and max_density_candidates > 0:
        log(args, f"Density scan target: stop after {max_density_candidates} matched candidates.")
    else:
        log(args, "Density scan target: scan all source scenarios.")
    for index, scenario_id in enumerate(source_ids, start=1):
        source_scenario = load_source_scenario(source_root, args.subset_name, scenario_id)
        estimated_agents = effective_agent_count(source_scenario, config)
        if density_matches(estimated_agents, density_type):
            candidates.append((scenario_id, source_scenario, estimated_agents))
            if max_density_candidates and len(candidates) >= max_density_candidates:
                log(
                    args,
                    f"  scanned {index}/{len(source_ids)} source scenarios, "
                    f"density candidates={len(candidates)}; stopping early"
                )
                break
        if index == 1 or index == len(source_ids) or index % 100 == 0:
            log(
                args,
                f"  scanned {index}/{len(source_ids)} source scenarios, "
                f"density candidates={len(candidates)}"
            )

    if not candidates:
        raise RuntimeError(
            f"No source scenario satisfied road_type={road_type} ({ROAD_TYPE_LABELS[road_type]}) "
            f"and density_type={density_type} ({DENSITY_LABELS[density_type]}). "
            f"Available road-type source candidates: {len(source_ids)}."
        )

    log(args, f"Density-matched candidate count: {len(candidates)}")
    device = resolve_device(args.device)
    log(args, f"Using device: {device}")
    log(args, "Loading generator model...")
    model = load_model(config, args.weight_path, device)
    log(args, "Generator model loaded.")

    for candidate_index, (scenario_id, source_scenario, estimated_agents) in enumerate(candidates, start=1):
        log(
            args,
            f"Trying candidate {candidate_index}/{len(candidates)}: "
            f"{scenario_id} (estimated agents={estimated_agents})"
        )
        require_cached_samples(config, [scenario_id + ".pkl"])
        log(args, "Building dataloader...")
        loader = make_dataloader(
            config,
            mode="val",
            batch_size=1,
            shuffle=False,
            data_list=[scenario_id + ".pkl"],
        )
        log(args, "Loading first batch...")
        batch = next(iter(loader))
        scenario_id = os.path.splitext(batch["file"][0])[0]
        num_agents = int(torch.sum(batch["object_mask"][0]).item())
        log(args, f"Loaded batch scenario={scenario_id}, model agent count={num_agents}")
        if not density_matches(num_agents, density_type):
            log(args, "Candidate skipped after exact batch density check.")
            continue

        log(args, "Running model inference...")
        with torch.no_grad():
            scene = infer_scenes(config, model, batch, device)[0]
        log(args, "Building generated scenario...")
        scenario = build_generated_scenario(scene, source_scenario, recompute_kinematics=False)
        log(args, "Reconstructing generated scenario...")
        scenario = reconstruct_generated_scenario(
            scenario,
            reconstructor_method=args.reconstructor,
        )
        log(args, "Saving generated scenario...")
        output_file = save_scenario(scenario, args.output_path, args.subset_name, scenario_id)

        print(
            f"Generated normal scenario: {output_file}\n"
            f"  road_type={road_type} ({ROAD_TYPE_LABELS[road_type]}), "
            f"density={DENSITY_LABELS[density_type]}, agents={num_agents}"
        )

        return

    raise RuntimeError(
        f"No matched scenario satisfied density_type={density_type} ({DENSITY_LABELS[density_type]}). "
        f"Available road-type candidates: {len(candidates)}."
    )
