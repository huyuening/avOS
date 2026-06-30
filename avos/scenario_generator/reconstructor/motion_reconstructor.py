import argparse
import casadi as ca
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm


class AgentConfig:
    """
    Final balanced version: high position weight, high heading weight, and flexible lateral-slip absorption
    """
    CONFIGS = {
        'vehicle': {
            'v_max': 40.0, 'v_min': -5.0,
            'a_max': 8.0,  'a_min': -10.0,
            'j_max': 30.0,
            'w_max': 2.0,
            'alat_max': 5.8,
            'alpha_max': 5.0,
            'weights': {
                'pos': 8000.0,       # keep 800 to keep RMSE strongly prioritized
                'heading': 20.0,     # major boost from 0.5 to 20.0 to strongly improve heading fit
                'j_lon': 1.0,
                'yaw_accel': 50.0,   # maintain steering smoothness
                'slack_lat': 50.0    # relaxed from 150 to 50 to allow flexible body slip and preserve position RMSE
            }
        },
        'cyclist': {
            'v_max': 20.0, 'v_min': -2.0,
            'a_max': 5.0,  'a_min': -6.0,
            'j_max': 25.0,
            'w_max': 3.0,  'alat_max': 4.0, 'alpha_max': 8.0,
            'weights': {
                'pos': 4000.0,
                'heading': 10.0,     # increased from 1.0 to 10.0
                'j_lon': 1.0,
                'yaw_accel': 80.0,
                'slack_lat': 30.0    # relaxed to allow slight lateral deviation for cyclists
            }
        },
        'pedestrian': {
            'v_max': 10.0, 'v_min': -3.0,
            'a_max': 8.0,  'a_min': -8.0,
            'j_max': 50.0,
            'w_max': 10.0, 'alpha_max': 30.0, 'alat_max': 10.0,
            'weights': {
                'pos': 8000.0,
                'heading': 10.0,     # increased so pedestrian heading follows appropriately
                'j_lon': 0.5,
                'yaw_accel': 5.0,
                'slack_lat': 10.0    # highly relaxed because pedestrians can move omnidirectionally, so heading and displacement may decouple
            }
        }
    }


class TrajectoryOptimizer:
    def __init__(self, agent_type='vehicle', dt=0.1, show_solver_warnings=False):
        self.dt = dt
        self.show_solver_warnings = show_solver_warnings
        type_str = str(agent_type).lower()
        if 'ped' in type_str:
            self.cfg = AgentConfig.CONFIGS['pedestrian']
        elif 'cyc' in type_str or 'bike' in type_str:
            self.cfg = AgentConfig.CONFIGS['cyclist']
        else:
            self.cfg = AgentConfig.CONFIGS['vehicle']

    def preprocess_data(self, x, y, h):
        """Input preprocessing: handle missing frames (NaNs) and provide an initial rollout for the solver"""
        N = len(x)
        t = np.arange(N) * self.dt
        valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(h)
        if np.sum(valid_mask) < 3: return None, None, None, None

        # fill missing values by linear interpolation
        x_init = interp1d(t[valid_mask], x[valid_mask], kind='linear', fill_value="extrapolate")(t)
        y_init = interp1d(t[valid_mask], y[valid_mask], kind='linear', fill_value="extrapolate")(t)

        # special angle unwrapping
        h_valid_unwrapped = np.unwrap(h[valid_mask])
        h_init = interp1d(t[valid_mask], h_valid_unwrapped, kind='linear', fill_value="extrapolate")(t)
        h_init = (h_init + np.pi) % (2 * np.pi) - np.pi
        return x_init, y_init, h_init, valid_mask.astype(float)

    def solve(self, x_obs, y_obs, h_obs):
        """
        Optimization solver based on the paper "A behaviourally underpinned approach..."
        Inputs only include: x, y, heading
        """
        weights = self.cfg['weights']
        N = len(x_obs)
        x_init, y_init, h_init, mask = self.preprocess_data(x_obs, y_obs, h_obs)
        if mask is None: return None, None, None

        opti = ca.Opti()

        # --- 1. State variables (State Variables) ---
        # Corresponds to Eq. (3), including x, y, heading, velocity, acceleration, and yaw_rate
        X = opti.variable(6, N)
        x, y, h, v, a, omega = X[0, :], X[1, :], X[2, :], X[3, :], X[4, :], X[5, :]

        # --- 2. Control variables (Control Variables) ---
        # Corresponds to Eq. (4). To avoid division by zero in curvature kappa=omega/v when v=0,
        # use jerk_lon and alpha (angular acceleration) as core controls
        U = opti.variable(3, N - 1)
        jerk_lon, alpha, v_lat_slack = U[0, :], U[1, :], U[2, :]

        J = 0  # objective function

        # 1. trajectory fidelity penalty (avoids vanishing gradients)
        delta = 2.0  # breakpoint: L2 squared error within 2 m (strong RMSE reduction), L1 linear error beyond 2 m (robust to outliers with nonzero gradient)

        for i in range(N):
            if mask[i] > 0:
                err_x2 = (x[i] - x_obs[i]) ** 2
                err_y2 = (y[i] - y_obs[i]) ** 2
                dist_sq = err_x2 + err_y2

                # core replacement: true Pseudo-Huber loss
                # Formula: delta^2 * (sqrt(1 + (error/delta)^2) - 1)
                J += weights['pos'] * mask[i] * (delta ** 2 * (ca.sqrt(1.0 + dist_sq / (delta ** 2)) - 1.0))

                # heading fit
                J += weights['heading'] * mask[i] * (1 - ca.cos(h[i] - h_obs[i]))

        # Control variable smoothness penalty
        J += weights['j_lon'] * ca.sumsqr(jerk_lon)
        J += weights['yaw_accel'] * ca.sumsqr(alpha)
        # penalize nonholonomic constraint violations (toward a pure kinematic model)
        J += weights['slack_lat'] * ca.sumsqr(v_lat_slack)

        opti.minimize(J)

        # --- 4. Discrete-time vehicle kinematic constraints corresponding to Eq. (5) ---
        for i in range(N - 1):
            # position integration (with slack variables to absorb abnormal noise)
            dx = (v[i] * ca.cos(h[i]) - v_lat_slack[i] * ca.sin(h[i])) * self.dt
            dy = (v[i] * ca.sin(h[i]) + v_lat_slack[i] * ca.cos(h[i])) * self.dt
            opti.subject_to(x[i + 1] == x[i] + dx)
            opti.subject_to(y[i + 1] == y[i] + dy)

            # higher-order heading and velocity integration
            opti.subject_to(h[i + 1] == h[i] + omega[i] * self.dt + 0.5 * alpha[i] * (self.dt ** 2))
            opti.subject_to(omega[i + 1] == omega[i] + alpha[i] * self.dt)
            opti.subject_to(v[i + 1] == v[i] + a[i] * self.dt)
            opti.subject_to(a[i + 1] == a[i] + jerk_lon[i] * self.dt)

        # --- 5. physical bounds corresponding to Eq. (6) - Eq. (11) ---
        opti.subject_to(opti.bounded(self.cfg['v_min'], v, self.cfg['v_max']))  # Eq.(6) velocity bound
        opti.subject_to(opti.bounded(self.cfg['a_min'], a, self.cfg['a_max']))  # Eq.(7) acceleration bound
        opti.subject_to(opti.bounded(-self.cfg['j_max'], jerk_lon, self.cfg['j_max']))  # Eq.(8) jerk bound

        # core lateral acceleration limit corresponding to Eq. (10)
        # physical formula: a_lat = v * omega <= g*(e+fs)
        alat = v * omega
        opti.subject_to(opti.bounded(-self.cfg['alat_max'], alat, self.cfg['alat_max']))

        opti.subject_to(opti.bounded(-self.cfg['w_max'], omega, self.cfg['w_max']))  # physical steering limit
        opti.subject_to(opti.bounded(-self.cfg['alpha_max'], alpha, self.cfg['alpha_max']))  # corresponds to Eq. (11)

        # --- 6. Initial guess (accelerates solving) ---
        opti.set_initial(x, x_init);
        opti.set_initial(y, y_init);
        opti.set_initial(h, h_init)
        v_guess = np.append(np.hypot(np.diff(x_init), np.diff(y_init)) / self.dt, 0)
        opti.set_initial(v, v_guess)

        h_diff = np.diff(h_init)
        h_diff = (h_diff + np.pi) % (2 * np.pi) - np.pi
        omega_guess = np.append(h_diff / self.dt, 0)
        opti.set_initial(omega, omega_guess)
        opti.set_initial(alpha, np.zeros(N - 1))

        # opti.solver("ipopt", {"expand": True, "print_time": False},
        #             {"max_iter": 300, "print_level": 0, "sb": "yes", "acceptable_tol": 1e-3})

        # increase max_iter from 300 to 1000 and slightly relax acceptable_tol
        opti.solver("ipopt", {"expand": True, "print_time": False},
                    {"max_iter": 1000,
                     "print_level": 0,
                     "sb": "yes",
                     "acceptable_tol": 1e-2,
                     "acceptable_iter": 15}) # allow early exit near an acceptable suboptimal solution

        try:
            sol = opti.solve()
            # change: directly extract the internally constrained kinematic variables
            v_opt = sol.value(v)
            a_opt = sol.value(a)
            omega_opt = sol.value(omega)
            j_lon_opt = np.append(sol.value(jerk_lon), 0.0)  # Pad control variable length

            # internally constrained lateral acceleration
            alat_opt = v_opt * omega_opt
            # internally computed lateral jerk
            j_lat_opt = np.append(np.diff(alat_opt) / self.dt, 0.0)

            return sol.value(x), sol.value(y), sol.value(h), v_opt, a_opt, omega_opt, alat_opt, j_lon_opt, j_lat_opt

        except Exception:
            # return debug values when not converged
            if self.show_solver_warnings:
                print("Warning: Solver not fully converged, returning relaxed values.")
            v_opt = opti.debug.value(v)
            omega_opt = opti.debug.value(omega)
            alat_opt = v_opt * omega_opt
            return (opti.debug.value(x), opti.debug.value(y), opti.debug.value(h),
                    v_opt, opti.debug.value(a), omega_opt, alat_opt,
                    np.append(opti.debug.value(jerk_lon), 0), np.append(np.diff(alat_opt) / self.dt, 0))


# ==========================================
# 2. helpers and visualization module
# ==========================================
def smooth_z_axis(z_raw, valid_mask):
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2: return np.nan_to_num(z_raw)
    z_interp = interp1d(valid_indices, z_raw[valid_mask], kind='linear', fill_value="extrapolate")(
        np.arange(len(z_raw)))
    return uniform_filter1d(z_interp, size=5)


def calc_kinematics_local(x, y, h, dt):
    vx = np.gradient(x, dt);
    vy = np.gradient(y, dt)
    ax = np.gradient(vx, dt);
    ay = np.gradient(vy, dt)
    v_lon = vx * np.cos(h) + vy * np.sin(h);
    v_lat = -vx * np.sin(h) + vy * np.cos(h)
    a_lon = ax * np.cos(h) + ay * np.sin(h);
    a_lat = -ax * np.sin(h) + ay * np.cos(h)
    j_lon = np.gradient(a_lon, dt);
    j_lat = np.gradient(a_lat, dt)
    return v_lon, v_lat, a_lon, a_lat, j_lon, j_lat


def fill_nan_for_calc(arr):
    valid = ~np.isnan(arr)
    if np.sum(valid) < 2: return arr
    return interp1d(np.where(valid)[0], arr[valid], kind='linear', fill_value="extrapolate")(np.arange(len(arr)))


def plot_six_graphs(t_seg, raw_data, opt_data, object_id, t_range, agent_type):
    """Standalone 6-panel comparison visualization function"""
    x_raw, y_raw, h_raw, v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw = raw_data
    x_opt, y_opt, h_opt, v_lon_opt, v_lat_opt, a_lon_opt, a_lat_opt = opt_data
    t_min, t_max = t_range

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"Trajectory Optimization | ID: {object_id} | Type: {agent_type}", fontsize=16)

    mean_x, mean_y, mean_h = np.nanmean(x_opt), np.nanmean(y_opt), np.nanmean(h_opt)
    mean_v_lon, mean_v_lat = np.nanmean(v_lon_opt), np.nanmean(v_lat_opt)
    mean_a_lon, mean_a_lat = np.nanmean(a_lon_opt), np.nanmean(a_lat_opt)

    N_pos = 40.0;
    N_h = np.pi;
    N_v_lon = 15.0;
    N_v_lat = 5.0;
    N_a_lon = 8.0;
    N_a_lat = 5.0

    axs[0, 0].plot(x_raw, y_raw, 'rx-', label='Raw (Noisy/NaNs)', alpha=0.5)
    axs[0, 0].plot(x_opt, y_opt, 'b-', label='Optimized', linewidth=2)
    axs[0, 0].set_title("1. Trajectory (X-Y)");
    axs[0, 0].legend();
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim(mean_x - N_pos, mean_x + N_pos);
    axs[0, 0].set_ylim(mean_y - N_pos, mean_y + N_pos);
    axs[0, 0].axis('equal')

    plots = [
        (0, 1, h_raw, h_opt, "2. Heading", mean_h, N_h),
        (1, 0, v_lon_raw, v_lon_opt, "3. Lon Velocity", mean_v_lon, N_v_lon),
        (1, 1, v_lat_raw, v_lat_opt, "4. Lat Velocity", mean_v_lat, N_v_lat),
        (2, 0, a_lon_raw, a_lon_opt, "5. Lon Acceleration", mean_a_lon, N_a_lon),
        (2, 1, a_lat_raw, a_lat_opt, "6. Lat Acceleration", mean_a_lat, N_a_lat)
    ]
    for r, c, raw, opt, title, mean_val, N_val in plots:
        axs[r, c].plot(t_seg, raw, 'rx-', alpha=0.5, label='Raw')
        axs[r, c].plot(t_seg, opt, 'b-', linewidth=2, label='Optimized')
        axs[r, c].set_title(title);
        axs[r, c].legend();
        axs[r, c].grid(True)
        axs[r, c].set_xlim(t_min, t_max);
        axs[r, c].set_ylim(mean_val - N_val, mean_val + N_val)

    plt.tight_layout()
    plt.show(block=True)


# ==========================================
# 3. Core logic: data reconstruction and in-place backfilling
# ==========================================
def object_track_reconstruction(args, timestamp, object_state, object_type, object_id=None):
    dt = float(np.median(np.diff(timestamp)))
    if dt > 10: dt /= 1e6
    if dt == 0 or np.isnan(dt): dt = 0.1

    valid_flag = np.array(object_state['valid']).flatten().astype(bool)
    valid_indices = np.where(valid_flag)[0]
    if len(valid_indices) < 3:
        return None, None, None, None

    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1
    interp_range = slice(start_idx, end_idx)

    x_raw = object_state['global_center'][interp_range, 0].copy()
    y_raw = object_state['global_center'][interp_range, 1].copy()
    z_raw = object_state['global_center'][interp_range, 2].copy()
    h_raw = object_state['heading'][interp_range].copy()
    size_raw = object_state['size'][interp_range, :].copy()
    valid_seg = valid_flag[interp_range]

    x_raw[~valid_seg] = np.nan
    y_raw[~valid_seg] = np.nan
    h_raw[~valid_seg] = np.nan

    # 1. call the optimizer and receive all internal variables directly
    optimizer = TrajectoryOptimizer(
        agent_type=object_type,
        dt=dt,
        show_solver_warnings=getattr(args, "show_solver_warnings", False),
    )
    res = optimizer.solve(x_raw, y_raw, h_raw)

    if res[0] is None:
        return None, None, None, None

    x_opt, y_opt, h_opt, v_lon, a_lon, omega_opt, a_lat, j_lon, j_lat = res

    # because the center-axis nonholonomic model is used, lateral velocity is theoretically zero (or tiny and absorbed by lateral slip)
    v_lat = np.zeros_like(v_lon)

    # 2. auxiliary smoothing and computation
    z_opt = smooth_z_axis(z_raw, valid_seg)
    mean_size = np.nanmean(size_raw[valid_seg], axis=0)
    reconstructed_size_value = np.tile(mean_size, (len(x_opt), 1))

    # for raw data, use post hoc differentiation for plotting comparison
    x_fill, y_fill, h_fill = fill_nan_for_calc(x_raw), fill_nan_for_calc(y_raw), fill_nan_for_calc(h_raw)
    v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw, j_lon_raw, j_lat_raw = calc_kinematics_local(x_fill, y_fill, h_fill, dt)

    # # 1. Call the optimizer
    # optimizer = TrajectoryOptimizer(agent_type=object_type, dt=dt)
    # x_opt, y_opt, h_opt = optimizer.solve(x_raw, y_raw, h_raw)
    # if x_opt is None:
    #     return None, None, None, None
    #
    # # 2. auxiliary smoothing and computation
    # z_opt = smooth_z_axis(z_raw, valid_seg)
    # mean_size = np.nanmean(size_raw[valid_seg], axis=0)
    # reconstructed_size_value = np.tile(mean_size, (len(x_opt), 1))
    # v_lon, v_lat, a_lon, a_lat, j_lon, j_lat = calc_kinematics_local(x_opt, y_opt, h_opt, dt)
    #
    # x_fill, y_fill, h_fill = fill_nan_for_calc(x_raw), fill_nan_for_calc(y_raw), fill_nan_for_calc(h_raw)
    # v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw, j_lon_raw, j_lat_raw = calc_kinematics_local(x_fill, y_fill, h_fill, dt)

    # 3. visualization module call (when --visualize is enabled)
    if getattr(args, 'visualize', False):
        full_time_array = np.arange(len(timestamp)) * dt
        t_seg = full_time_array[interp_range]
        t_range = (full_time_array[0], full_time_array[-1])

        raw_data = (x_raw, y_raw, h_raw, v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw)
        opt_data = (x_opt, y_opt, h_opt, v_lon, v_lat, a_lon, a_lat)
        plot_six_graphs(t_seg, raw_data, opt_data, object_id, t_range, object_type)

    # 4. data assembly
    reconstructed_timestamp = timestamp[interp_range]

    filled_part_data = np.column_stack((
        x_fill, y_fill, v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw, j_lon_raw, j_lat_raw, h_fill
    ))

    noise_removal_part_data = np.column_stack((
        x_opt, y_opt, v_lon, v_lat, a_lon, a_lat, j_lon, j_lat, h_opt
    ))

    # =======================================================
    # 5. write back to object_state directly inside the function (in-place)
    # =======================================================
    object_state['global_center'][interp_range, :] = np.column_stack((x_opt, y_opt, z_opt))
    object_state['local_velocity'][interp_range, :] = np.column_stack((v_lon, v_lat))
    object_state['local_acceleration'][interp_range, :] = np.column_stack((a_lon, a_lat))

    # initialize local_jerk if it does not exist in the source data
    if 'local_jerk' not in object_state:
        object_state['local_jerk'] = np.zeros((len(object_state['valid']), 2))
    object_state['local_jerk'][interp_range, :] = np.column_stack((j_lon, j_lat))

    object_state['heading'][interp_range] = h_opt
    object_state['size'][interp_range, :] = reconstructed_size_value
    object_state['valid'][interp_range] = True

    return reconstructed_timestamp, object_state, filled_part_data, noise_removal_part_data


# ==========================================
# 4. file I/O and main control flow
# ==========================================
def read_file(path, file, subset_name):
    # compatible with the original structure design
    full_path = os.path.join(path, subset_name, file)
    with open(full_path, 'rb') as f:
        return pickle.load(f)


def save_file(path, scenario, file, subset_name):
    full_dir = os.path.join(path, subset_name)
    os.makedirs(full_dir, exist_ok=True)
    with open(os.path.join(full_dir, file), 'wb') as f:
        pickle.dump(scenario, f)


def process_single_file(args, converted_subset_path, reconstructed_subset_path, file):
    """single-file entry point for external or multiprocessing calls"""
    try:
        scenario = read_file(converted_subset_path, file, args.subset_name)

        object_tracks = scenario['object_tracks']
        extra_information = scenario['extra_information']

        # iterate over all agents in the file
        for object_id in list(object_tracks.keys()):
            object_track = object_tracks[object_id]

            timestamp = extra_information['timestamp']
            object_state = object_track['state']
            object_type = object_track['type']

            # call the reconstruction function and receive the backfilled state plus extra statistics
            _, reconstructed_object_state, _, _ = object_track_reconstruction(
                args, timestamp, object_state, object_type, object_id
            )

            # per the design: if the optimizer treats this trajectory as invalid or too short and returns None
            if reconstructed_object_state is None:
                # print(f"Warning: The timestamp is invalid or track too short in {file} (ID: {object_id}).")
                # the original code returned file, False here, which would drop the entire scenario.
                # use continue to skip this problematic agent while keeping the other valid agents in the scenario:
                continue

                # update the dictionary (the function modifies object_state in place; this assignment is kept for clarity)
            object_track['state'] = reconstructed_object_state

        save_file(reconstructed_subset_path, scenario, file, args.subset_name)
        return file, True

    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return file, False


def main(argv=None):
    project_root = Path(__file__).resolve().parents[3]
    datasets_root = project_root / "datasets"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=str(datasets_root / "converted_dataset"))
    parser.add_argument('--output_path', type=str, default=str(datasets_root / "optimized_dataset"))
    parser.add_argument('--subset_name', type=str, default='waymo-open')
    parser.add_argument('--num_workers', '--num-workers', dest='num_workers', type=int, default=8)
    parser.add_argument('--show_solver_warnings', '--show-solver-warnings',
                        dest='show_solver_warnings',
                        action='store_true',
                        default=False)
    parser.add_argument('--visualize', action='store_true', default=False, help='enable this flag to compare trajectories before and after smoothing')
    args = parser.parse_args(argv)

    # # simple usage example (this loop can be changed to concurrent.futures multiprocessing for speed)
    # input_dir = os.path.join(args.input_path, args.subset_name)
    # if not os.path.exists(input_dir):
    #     print(f"Directory not found: {input_dir}")
    #     exit()
    #
    # file_list = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    #
    # print(f"Found {len(file_list)} scenario files. Start processing...")
    # for file in tqdm(file_list):
    #     # arguments: args, source root, target root, file name
    #     file_name, success = process_single_file(args, args.input_path, args.output_path, file)
    #     if not success:
    #         print(f"Failed to process file: {file_name}")
    #
    # print("All processing finished.")

    input_dir = os.path.join(args.input_path, args.subset_name)
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    file_list = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    # file_list = ['102837b835b8ac89.pkl']

    num_workers = max(1, args.num_workers or multiprocessing.cpu_count() - 1)

    print(f"Found {len(file_list)} scenario files. Start processing using {num_workers} processes...")
    if not file_list:
        print("No scenario files found. Skip reconstruction.")
        return

    failed_files = []  # record failed files

    # create multiprocessing pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 1. submit all tasks to the process pool
        # map futures to file names so exceptions can be traced back to the file
        futures = {
            executor.submit(process_single_file, args, args.input_path, args.output_path, file): file
            for file in file_list
        }

        # 2. use as_completed with tqdm to display progress for completed tasks
        for future in tqdm(as_completed(futures), total=len(file_list), desc="Processing"):
            file = futures[future]  # get the original file name for the completed task
            try:
                # get function result
                file_name, success = future.result()
                if not success:
                    # use tqdm.write instead of print while tqdm is active to avoid corrupting the progress bar
                    tqdm.write(f"Failed to process file: {file_name}")
                    failed_files.append(file_name)

            except Exception as e:
                # catch unhandled exceptions from process_single_file, such as out-of-memory errors or solver crashes
                tqdm.write(f"Exception occurred while processing {file}: {str(e)}")
                failed_files.append(file)
