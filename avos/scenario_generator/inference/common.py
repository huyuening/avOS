import os
import pickle
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_paths import require_optimized_path
from processor.dataset import CustomDataset
from inference.defaults import dataset_path
from generator.model import ScenarioGenerator
from reconstructor.utils import update_vel_acc_jerk
from generator.utils import fc_collate_fn, format_labels, process_scene, rotate


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict(weight_path, device):
    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if checkpoint and all(key.startswith("module.") for key in checkpoint.keys()):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}

    return checkpoint


def load_model(config, weight_path, device):
    model = ScenarioGenerator(config).to(device)
    model.load_state_dict(load_state_dict(weight_path, device))
    model.eval()
    return model


def make_dataloader(config, mode="val", batch_size=1, shuffle=True, data_list=None):
    dataset = CustomDataset(config, mode=mode, data_list=data_list)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=fc_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def batch_to_device(batch, device):
    return {
        key: (
            value.to(device, dtype=torch.float32)
            if torch.is_tensor(value) and value.dtype == torch.float64
            else value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def infer_scenes(config, model, batch, device, add_labels=True):
    batch = batch_to_device(batch, device)
    if add_labels:
        batch = format_labels(batch)
    outputs = model(batch)
    return process_scene(config, outputs, batch)


def infer_scenes_from_batch(config, model, batch, device):
    with torch.no_grad():
        return infer_scenes(config, model, batch, device)


def source_root_from_config(config, subset_name):
    dataset_path = Path(require_optimized_path(config))
    if dataset_path.name == subset_name:
        return str(dataset_path.parent)
    return str(dataset_path)


def load_source_scenario(input_root, subset_name, scenario_id):
    with open(os.path.join(input_root, subset_name, scenario_id) + ".pkl", "rb") as file:
        return pickle.load(file)


def as_global_center_3d(center_xy):
    center = np.asarray(center_xy)
    if center.ndim != 2:
        raise ValueError(f"Expected global_center with 2 dimensions, got shape {center.shape}.")
    if center.shape[1] == 3:
        return center
    if center.shape[1] != 2:
        raise ValueError(f"Expected global_center with 2 or 3 columns, got shape {center.shape}.")
    center_z = np.zeros((center.shape[0], 1), dtype=center.dtype)
    return np.concatenate([center, center_z], axis=1)


def build_generated_scenario(scene, source_scenario, recompute_kinematics=True):
    num_timesteps = len(scene["complete_pos"])
    timestamp = np.arange(num_timesteps) / 10
    ego_id = source_scenario["extra_information"]["sdc_id"]
    ego_pos = source_scenario["object_tracks"][ego_id]["state"]["global_center"][0, :2]
    ego_hdn = source_scenario["object_tracks"][ego_id]["state"]["heading"][0]

    complete_pos = scene["complete_pos"]
    complete_hdn = np.concatenate(
        [scene["init_hdn"][np.newaxis, :, np.newaxis], scene["future_hdn"]],
        axis=0,
    )

    complete_pos = rotate(complete_pos[..., 0], complete_pos[..., 1], ego_hdn)
    complete_pos += ego_pos
    complete_hdn += ego_hdn

    object_tracks = {}
    for idx in range(len(scene["init_pos"])):
        object_track = {
            "type": scene["type"][idx],
            "subtype": scene["subtype"][idx],
            "phase": "generated",
        }
        state = {}

        if recompute_kinematics:
            center_x = complete_pos[:, idx, 0]
            center_y = complete_pos[:, idx, 1]
            center_z = np.zeros(len(timestamp))
            heading = complete_hdn[:, idx].squeeze()
            vel_x, vel_y, acc_x, acc_y, _, _ = update_vel_acc_jerk(
                timestamp, center_x, center_y, heading
            )
            state["global_center"] = np.column_stack([center_x, center_y, center_z])
            state["local_velocity"] = np.column_stack([vel_x, vel_y])
            state["local_acceleration"] = np.column_stack([acc_x, acc_y])
        else:
            state["global_center"] = as_global_center_3d(complete_pos[:, idx])
            state["local_velocity"] = np.concatenate(
                [scene["init_vel"][idx][np.newaxis, :], scene["future_vel"][:, idx]],
                axis=0,
            )
            state["local_acceleration"] = np.zeros((num_timesteps, 2))

        state["heading"] = complete_hdn[:, idx].squeeze()
        state["size"] = np.repeat(scene["init_size"][idx][np.newaxis, :], num_timesteps, axis=0)
        state["valid"] = np.array([True] * num_timesteps, dtype=bool)
        object_track["state"] = state
        object_tracks[str(idx)] = object_track

    scenario = {
        "map_features": source_scenario["map_features"],
        "dynamic_map_states": source_scenario["dynamic_map_states"],
        "extra_information": source_scenario["extra_information"],
        "object_tracks": object_tracks,
    }
    scenario["extra_information"]["scene_length"] = len(timestamp)
    scenario["extra_information"]["timestamp"] = timestamp
    scenario["extra_information"]["sdc_id"] = list(object_tracks.keys())[0]
    return scenario


def reconstruct_generated_scenario(
    scenario,
    show_solver_warnings=False,
    visualize=False,
    reconstructor_method=None,
):
    from reconstructor.reconstructor_selector import get_object_track_reconstruction

    object_track_reconstruction = get_object_track_reconstruction(reconstructor_method)

    args = SimpleNamespace(
        show_solver_warnings=show_solver_warnings,
        visualize=visualize,
    )
    timestamp = scenario["extra_information"]["timestamp"]

    for object_id in list(scenario["object_tracks"].keys()):
        object_track = scenario["object_tracks"][object_id]
        object_track["state"]["global_center"] = as_global_center_3d(
            object_track["state"]["global_center"]
        )
        _, reconstructed_object_state, _, _ = object_track_reconstruction(
            args,
            timestamp,
            object_track["state"],
            object_track["type"],
            object_id,
        )
        if reconstructed_object_state is not None:
            object_track["state"] = reconstructed_object_state

    return scenario


def save_scenario(scenario, output_root, subset_name, scenario_id):
    output_dir = os.path.join(output_root, subset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, scenario_id) + ".pkl"
    with open(output_file, "wb") as file:
        pickle.dump(scenario, file)
    return output_file


def generate_dataset_samples(
    config_path,
    weight_path,
    input_root=None,
    output_root=dataset_path("generated_dataset"),
    subset_name="waymo-open",
    mode="val",
    batch_size=1,
    max_samples=None,
    shuffle=True,
    device=None,
    recompute_kinematics=True,
    reconstructor_method=None,
):
    config = load_config(config_path)
    source_root = input_root or source_root_from_config(config, subset_name)
    device = resolve_device(device)
    model = load_model(config, weight_path, device)
    loader = make_dataloader(config, mode=mode, batch_size=batch_size, shuffle=shuffle)

    saved_files = []
    seen_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            scenes = infer_scenes(config, model, batch, device)
            for scene_idx, scene in enumerate(scenes):
                scenario_id = os.path.splitext(batch["file"][scene_idx])[0]
                source_scenario = load_source_scenario(source_root, subset_name, scenario_id)
                scenario = build_generated_scenario(
                    scene,
                    source_scenario,
                    recompute_kinematics=recompute_kinematics,
                )
                scenario = reconstruct_generated_scenario(
                    scenario,
                    reconstructor_method=reconstructor_method,
                )
                saved_files.append(save_scenario(scenario, output_root, subset_name, scenario_id))
                seen_samples += 1
                if max_samples is not None and seen_samples >= max_samples:
                    return saved_files

    return saved_files
