"""Velocity-heading trajectory reconstructor.

This is the default reconstructor used by current preprocessing and inference.
"""

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

try:
    from .velocity_heading_reconstructor import (
        object_track_reconstruction_velocity_heading,
        process_single_file_velocity_heading,
    )
except ImportError:
    from reconstructor.velocity_heading_reconstructor import (
        object_track_reconstruction_velocity_heading,
        process_single_file_velocity_heading,
    )


object_track_reconstruction = object_track_reconstruction_velocity_heading
process_single_file = process_single_file_velocity_heading


def main(argv=None):
    project_root = Path(__file__).resolve().parents[3]
    datasets_root = project_root / "datasets"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=str(datasets_root / "converted_dataset"))
    parser.add_argument("--output_path", type=str, default=str(datasets_root / "optimized_dataset"))
    parser.add_argument("--subset_name", type=str, default="waymo-open")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=8)
    parser.add_argument(
        "--show_solver_warnings",
        "--show-solver-warnings",
        dest="show_solver_warnings",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable this flag to compare trajectories before and after smoothing.",
    )
    args = parser.parse_args(argv)

    input_dir = os.path.join(args.input_path, args.subset_name)
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    file_list = [file_name for file_name in os.listdir(input_dir) if file_name.endswith(".pkl")]
    num_workers = max(1, args.num_workers or multiprocessing.cpu_count() - 1)

    print(
        f"Found {len(file_list)} scenario files. "
        f"Start processing with motion_reconstructor_v2 using {num_workers} processes..."
    )
    if not file_list:
        print("No scenario files found. Skip reconstruction.")
        return

    failed_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_file, args, args.input_path, args.output_path, file_name): file_name
            for file_name in file_list
        }

        for future in tqdm(as_completed(futures), total=len(file_list), desc="Processing"):
            file_name = futures[future]
            try:
                processed_file, success = future.result()
                if not success:
                    tqdm.write(f"Failed to process file: {processed_file}")
                    failed_files.append(processed_file)
            except Exception as exc:
                tqdm.write(f"Exception occurred while processing {file_name}: {str(exc)}")
                failed_files.append(file_name)

    if failed_files:
        print(f"Processing finished with {len(failed_files)} failed files.")
        for failed_file in failed_files[:10]:
            print(f"  - {failed_file}")
        if len(failed_files) > 10:
            print("  - ...")
    else:
        print("All files processed successfully!")


if __name__ == "__main__":
    main()
