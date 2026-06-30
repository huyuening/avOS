import os
import pickle
import torch
import multiprocessing
import concurrent.futures
import yaml
import argparse

from tqdm import tqdm

from processor.data_preprocess import process, get_object_code
from inference.defaults import default_config_path
from config_paths import require_optimized_path, require_trainval_path


# ==========================================
# 1. Define the per-file processing function (must be standalone for multiprocessing)
# ==========================================
def process_single_file(task_info):
    """
    Worker function for processing one file
    Returns: True on success or an error message on failure
    """
    config, file_name, index, raw_data_path, processed_data_path, skip_existing = task_info

    # source file and output file paths
    raw_file_path = os.path.join(raw_data_path, file_name)
    save_name = file_name.replace('.pkl', '.pt')  # adjust according to the actual suffix
    processed_file_path = os.path.join(processed_data_path, save_name)

    if skip_existing and os.path.exists(processed_file_path):
        return True

    try:
        # --- original core processing logic ---
        with open(raw_file_path, 'rb') as file:
            scenario = pickle.load(file)

        data = process(config, scenario)
        data['file'] = file_name
        data['index'] = index
        get_object_code(config, data)
        # -------------------------

        # save as a .pt file
        torch.save(data, processed_file_path)
        return True

    except Exception as e:
        # return an error message because exceptions in multiprocessing can deadlock or be hard to debug
        return f"Error processing {file_name}: {str(e)}"


def preprocess_file_list(config, file_names, num_workers=None, skip_existing=False, desc="CACHE", verbose=True):
    raw_data_path = require_optimized_path(config)
    processed_data_path = require_trainval_path(config)
    os.makedirs(processed_data_path, exist_ok=True)

    seen = set()
    data_list = []
    for file_name in file_names:
        file_name = file_name.strip()
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        data_list.append(file_name)

    tasks = []
    missing_sources = []
    for index, file_name in enumerate(data_list):
        raw_file_path = os.path.join(raw_data_path, file_name)
        if not os.path.exists(raw_file_path):
            missing_sources.append(file_name)
            continue
        tasks.append((config, file_name, index, raw_data_path, processed_data_path, skip_existing))

    if missing_sources and verbose:
        print(f"Skip {len(missing_sources)} files not found in {raw_data_path}.")
    if not tasks:
        if verbose:
            print(f"No source .pkl files found for {desc}.")
        return

    max_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
    if verbose:
        print(f"Generate .pt cache for {len(tasks)} files using {max_workers} processes.")

    failed_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc=desc,
            disable=not verbose,
        ):
            result = future.result()
            if result is not True:
                failed_files.append(result)

    if failed_files:
        print(f"{desc} finished with {len(failed_files)} failed files:")
        for err in failed_files[:5]:
            print(f"   - {err}")
        if len(failed_files) > 5:
            print("   - ...")
    elif verbose:
        print(f"{desc} .pt cache is ready.")


# ==========================================
# 2. main multiprocessing scheduler
# ==========================================
def preprocess_and_save_parallel(config, modes=None, num_workers=None, skip_existing=False):
    modes = modes or ['TRAIN', 'VAL', 'TEST']

    # get CPU core count and leave one core for the OS
    max_workers = num_workers or max(1, multiprocessing.cpu_count()-1)
    print(f"Start multiprocessing with worker count: {max_workers}")

    for mode in modes:
        if mode not in config['DATALIST']['NAME']:
            continue

        list_path = os.path.join(config['DATALIST']['PATH'], config['DATALIST']['NAME'][mode])
        with open(list_path, 'r') as file:
            data_list = [line.strip() for line in file.readlines()]

        print(f"\nProcessing {mode} split with {len(data_list)} samples...")
        preprocess_file_list(
            config,
            data_list,
            num_workers=max_workers,
            skip_existing=skip_existing,
            desc=mode,
        )


# ==========================================
# 3. execution entry point
# ==========================================
def main(argv=None):
    parser = argparse.ArgumentParser(description='Preprocess .pkl scenarios into .pt training samples.')
    parser.add_argument('--config-path', type=str, default=default_config_path())
    parser.add_argument('--modes', nargs='+', default=['TRAIN', 'VAL', 'TEST'],
                        help='Dataset splits to preprocess, e.g. TRAIN VAL TEST.')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args(argv)

    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    preprocess_and_save_parallel(
        config,
        modes=args.modes,
        num_workers=args.num_workers,
        skip_existing=args.skip_existing,
    )
