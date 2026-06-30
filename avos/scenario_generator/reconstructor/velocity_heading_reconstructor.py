import copy
import os
import pickle
from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.interpolate import interp1d

try:
    from .motion_reconstructor import (
        AgentConfig,
        calc_kinematics_local,
        fill_nan_for_calc,
        smooth_z_axis,
    )
except ImportError:
    from reconstructor.motion_reconstructor import (
        AgentConfig,
        calc_kinematics_local,
        fill_nan_for_calc,
        smooth_z_axis,
    )


@dataclass
class VelocityHeadingConfig:
    min_speed_mps: float = 1.0
    heading_outlier_deg: float = 75.0
    reverse_speed_tolerance_mps: float = 0.0
    raw_heading_spike_deg: float = 120.0
    raw_heading_spike_support_deg: float = 45.0
    raw_heading_spike_max_run: int = 3
    reverse_motion_speed_mps: float = 0.2
    reverse_motion_min_run: int = 3
    reverse_motion_min_ratio: float = 0.05
    high_speed_heading_weight: float = 1.0
    low_speed_heading_weight: float = 0.5
    velocity_smoothing_window: int = 3
    heading_smoothing_window: int = 5
    position_weight: float = 3000.0
    heading_weight: float = 9000.0
    force_nonnegative_vehicle_speed: bool = True


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_diff(angle, reference):
    return wrap_angle(angle - reference)


def _interp_nan(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.zeros_like(values)
    if valid.sum() == 1:
        return np.full_like(values, values[valid][0])
    indices = np.arange(len(values))
    return interp1d(indices[valid], values[valid], kind="linear", fill_value="extrapolate")(indices)


def _moving_average(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_heading(heading, window):
    heading = np.asarray(heading, dtype=float)
    if window <= 1 or len(heading) < 3:
        return heading
    return _moving_average(np.unwrap(heading), min(window, len(heading)))


def _interpolate_angles(angle, anchor_mask, fallback):
    angle = np.asarray(angle, dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    indices = np.arange(len(angle))
    anchor_mask = anchor_mask & np.isfinite(angle)

    if anchor_mask.sum() == 0:
        return np.unwrap(_interp_nan(fallback))

    anchor_idx = indices[anchor_mask]
    anchor_values = np.unwrap(angle[anchor_mask])
    if anchor_mask.sum() == 1:
        return np.full_like(angle, anchor_values[0], dtype=float)

    return np.interp(indices, anchor_idx, anchor_values)


def _remove_short_heading_flips(
    heading,
    valid_mask,
    spike_deg=120.0,
    support_deg=45.0,
    max_run=3,
):
    heading = np.asarray(heading, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    cleaned = heading.copy()
    if max_run <= 0:
        return cleaned, np.zeros_like(valid_mask, dtype=bool)

    finite_idx = np.where(valid_mask & np.isfinite(heading))[0]
    if len(finite_idx) < 3:
        return cleaned, np.zeros_like(valid_mask, dtype=bool)

    finite_heading = heading[finite_idx]
    jump_deg = np.rad2deg(np.abs(angle_diff(finite_heading[1:], finite_heading[:-1])))
    large_jumps = np.where(jump_deg > spike_deg)[0]
    rejected = np.zeros_like(valid_mask, dtype=bool)

    for left_jump, right_jump in zip(large_jumps[:-1], large_jumps[1:]):
        run_len = right_jump - left_jump
        if run_len < 1 or run_len > max_run:
            continue

        before_heading = finite_heading[left_jump]
        after_heading = finite_heading[right_jump + 1]
        support_diff = np.rad2deg(np.abs(angle_diff(before_heading, after_heading)))
        if support_diff > support_deg:
            continue

        run_idx = finite_idx[left_jump + 1:right_jump + 1]
        cleaned[run_idx] = np.nan
        rejected[run_idx] = True

    return cleaned, rejected


def longest_true_run(mask):
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _state_velocity_direction(x, y, valid_mask, dt, global_velocity=None, smoothing_window=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    vx = np.full_like(x, np.nan, dtype=float)
    vy = np.full_like(y, np.nan, dtype=float)

    if global_velocity is not None:
        global_velocity = np.asarray(global_velocity, dtype=float)
        if global_velocity.ndim == 2 and global_velocity.shape[1] >= 2:
            vx = global_velocity[:, 0].copy()
            vy = global_velocity[:, 1].copy()

    velocity_valid = valid_mask & np.isfinite(vx) & np.isfinite(vy)
    if velocity_valid.sum() < 3 or np.nanmax(np.hypot(vx[velocity_valid], vy[velocity_valid])) < 1e-3:
        x_fill = _interp_nan(np.where(valid_mask, x, np.nan))
        y_fill = _interp_nan(np.where(valid_mask, y, np.nan))
        vx = np.gradient(x_fill, dt)
        vy = np.gradient(y_fill, dt)
        velocity_valid = valid_mask & np.isfinite(vx) & np.isfinite(vy)

    vx = _moving_average(_interp_nan(np.where(velocity_valid, vx, np.nan)), smoothing_window)
    vy = _moving_average(_interp_nan(np.where(velocity_valid, vy, np.nan)), smoothing_window)
    speed = np.hypot(vx, vy)
    heading = np.arctan2(vy, vx)
    return speed, heading


def has_sustained_reverse_motion(
    x,
    y,
    raw_heading,
    valid_mask,
    dt,
    global_velocity=None,
    config=None,
):
    config = config or VelocityHeadingConfig()
    raw_heading_clean, _ = _remove_short_heading_flips(
        raw_heading,
        valid_mask,
        spike_deg=config.raw_heading_spike_deg,
        support_deg=config.raw_heading_spike_support_deg,
        max_run=config.raw_heading_spike_max_run,
    )
    speed, velocity_heading = _state_velocity_direction(
        x,
        y,
        valid_mask,
        dt,
        global_velocity=global_velocity,
        smoothing_window=config.velocity_smoothing_window,
    )
    forward_speed = speed * np.cos(angle_diff(velocity_heading, raw_heading_clean))
    reverse_mask = (
        valid_mask
        & np.isfinite(forward_speed)
        & (forward_speed < -config.reverse_motion_speed_mps)
    )
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return False

    reverse_ratio = float(np.sum(reverse_mask)) / float(valid_count)
    return (
        longest_true_run(reverse_mask) >= config.reverse_motion_min_run
        and reverse_ratio >= config.reverse_motion_min_ratio
    )


def build_velocity_heading_observation(
    x,
    y,
    raw_heading,
    valid_mask,
    dt,
    object_type,
    global_velocity=None,
    config=None,
):
    config = config or VelocityHeadingConfig()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    raw_heading = np.asarray(raw_heading, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    raw_heading_clean, raw_heading_spike = _remove_short_heading_flips(
        raw_heading,
        valid_mask,
        spike_deg=config.raw_heading_spike_deg,
        support_deg=config.raw_heading_spike_support_deg,
        max_run=config.raw_heading_spike_max_run,
    )

    speed, velocity_heading = _state_velocity_direction(
        x,
        y,
        valid_mask,
        dt,
        global_velocity=global_velocity,
        smoothing_window=config.velocity_smoothing_window,
    )

    type_str = str(object_type).lower()
    use_velocity_heading = "ped" not in type_str
    diff_to_velocity = np.abs(angle_diff(raw_heading_clean, velocity_heading))
    forward_speed = speed * np.cos(angle_diff(velocity_heading, raw_heading_clean))
    base_reliable_velocity = (
        use_velocity_heading
        & valid_mask
        & np.isfinite(raw_heading_clean)
        & np.isfinite(velocity_heading)
        & (speed >= config.min_speed_mps)
    )
    reverse_velocity = (
        base_reliable_velocity
        & np.isfinite(forward_speed)
        & (forward_speed < -config.reverse_speed_tolerance_mps)
    )
    reliable_velocity = base_reliable_velocity & ~reverse_velocity

    raw_clean_mask = valid_mask & np.isfinite(raw_heading_clean)
    raw_filled = _interpolate_angles(raw_heading_clean, raw_clean_mask, raw_heading_clean)
    corrected_anchor = np.full_like(raw_heading, np.nan, dtype=float)
    heading_weight = np.zeros_like(raw_heading, dtype=float)

    outlier = reliable_velocity & (np.rad2deg(diff_to_velocity) > config.heading_outlier_deg)

    corrected_anchor[reliable_velocity & ~outlier] = raw_heading_clean[reliable_velocity & ~outlier]
    corrected_anchor[outlier] = velocity_heading[outlier]
    heading_weight[reliable_velocity] = config.high_speed_heading_weight

    if not use_velocity_heading:
        raw_valid = valid_mask & np.isfinite(raw_heading_clean)
        corrected_anchor[raw_valid] = raw_heading_clean[raw_valid]
        heading_weight[raw_valid] = config.high_speed_heading_weight

    anchor_mask = np.isfinite(corrected_anchor)
    heading_observation = _interpolate_angles(corrected_anchor, anchor_mask, raw_filled)
    heading_observation = _smooth_heading(heading_observation, config.heading_smoothing_window)

    low_speed_mask = valid_mask & ~reliable_velocity
    heading_weight[low_speed_mask] = np.maximum(
        heading_weight[low_speed_mask],
        config.low_speed_heading_weight,
    )

    diagnostics = {
        "velocity_heading_enabled": bool(use_velocity_heading),
        "valid_count": int(valid_mask.sum()),
        "reliable_velocity_count": int(reliable_velocity.sum()),
        "reverse_velocity_rejected_count": int(reverse_velocity.sum()),
        "raw_heading_spike_rejected_count": int(raw_heading_spike.sum()),
        "low_speed_count": int(low_speed_mask.sum()),
        "velocity_corrected_count": int(outlier.sum()),
        "mean_speed_mps": float(np.nanmean(speed[valid_mask])) if valid_mask.any() else None,
        "max_speed_mps": float(np.nanmax(speed[valid_mask])) if valid_mask.any() else None,
        "mean_abs_raw_velocity_heading_diff_deg": (
            float(np.nanmean(np.rad2deg(diff_to_velocity[reliable_velocity])))
            if reliable_velocity.any()
            else None
        ),
        "max_abs_raw_velocity_heading_diff_deg": (
            float(np.nanmax(np.rad2deg(diff_to_velocity[reliable_velocity])))
            if reliable_velocity.any()
            else None
        ),
    }
    return heading_observation, heading_weight, speed, velocity_heading, diagnostics


def _angle_series_for_plot(angle):
    angle = np.asarray(angle, dtype=float)
    if np.isfinite(angle).sum() < 2:
        return np.rad2deg(angle)
    filled = _interp_nan(angle)
    return np.rad2deg(np.unwrap(filled))


def _normalize_figsize(value):
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.lower().replace("x", ",").split(",")
        if len(parts) != 2:
            return None
        try:
            return (float(parts[0].strip()), float(parts[1].strip()))
        except ValueError:
            return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return None


def plot_velocity_heading_graphs(
    t_seg,
    raw_data,
    clean_data,
    opt_data,
    object_id,
    t_range,
    agent_type,
    output_dir=None,
    figsize=None,
):
    import matplotlib.pyplot as plt

    x_raw, y_raw, h_raw, v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw = raw_data
    heading_clean, heading_weight, velocity_speed, velocity_heading = clean_data
    x_opt, y_opt, h_opt, v_lon_opt, v_lat_opt, a_lon_opt, a_lat_opt = opt_data
    t_min, t_max = t_range
    figsize = figsize or (12, 9)

    fig, axs = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(
        f"Velocity-Heading Reconstruction | ID: {object_id} | Type: {agent_type}",
        fontsize=16,
    )

    axs[0, 0].plot(x_raw, y_raw, "rx-", label="Raw", alpha=0.45)
    axs[0, 0].plot(x_opt, y_opt, "b-", label="Optimized", linewidth=2)
    axs[0, 0].set_title("1. Trajectory (X-Y)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis("equal")

    axs[0, 1].plot(t_seg, _angle_series_for_plot(h_raw), "r-", alpha=0.45, label="Raw heading")
    axs[0, 1].plot(t_seg, _angle_series_for_plot(heading_clean), "g-", linewidth=2, label="Clean heading target")
    axs[0, 1].plot(t_seg, _angle_series_for_plot(velocity_heading), "k--", alpha=0.6, label="Velocity direction")
    axs[0, 1].plot(t_seg, _angle_series_for_plot(h_opt), "b-", linewidth=2, label="Optimized heading")
    axs[0, 1].set_title("2. Heading")
    axs[0, 1].set_ylabel("deg")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    raw_heading_err = np.rad2deg(np.abs(angle_diff(h_raw, heading_clean)))
    opt_heading_err = np.rad2deg(np.abs(angle_diff(h_opt, heading_clean)))
    velocity_heading_err = np.rad2deg(np.abs(angle_diff(velocity_heading, heading_clean)))
    axs[1, 0].plot(t_seg, raw_heading_err, "r-", alpha=0.45, label="Raw vs clean")
    axs[1, 0].plot(t_seg, velocity_heading_err, "k--", alpha=0.6, label="Velocity vs clean")
    axs[1, 0].plot(t_seg, opt_heading_err, "b-", linewidth=2, label="Optimized vs clean")
    axs[1, 0].set_title("3. Heading Error to Clean Target")
    axs[1, 0].set_ylabel("deg")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(t_seg, heading_weight, "m-", linewidth=2, label="Heading weight")
    axs[1, 1].plot(t_seg, velocity_speed, "c-", alpha=0.75, label="Velocity speed")
    axs[1, 1].set_title("4. Heading Weight and Velocity Speed")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[2, 0].plot(t_seg, v_lon_raw, "r-", alpha=0.45, label="Raw")
    axs[2, 0].plot(t_seg, v_lon_opt, "b-", linewidth=2, label="Optimized")
    axs[2, 0].plot(t_seg, velocity_speed, "k--", alpha=0.5, label="Global speed")
    axs[2, 0].set_title("5. Longitudinal Velocity")
    axs[2, 0].set_ylabel("m/s")
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    axs[2, 1].plot(t_seg, v_lat_raw, "r-", alpha=0.45, label="Raw")
    axs[2, 1].plot(t_seg, v_lat_opt, "b-", linewidth=2, label="Optimized")
    axs[2, 1].set_title("6. Lateral Velocity")
    axs[2, 1].set_ylabel("m/s")
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    axs[3, 0].plot(t_seg, a_lon_raw, "r-", alpha=0.45, label="Raw")
    axs[3, 0].plot(t_seg, a_lon_opt, "b-", linewidth=2, label="Optimized")
    axs[3, 0].set_title("7. Longitudinal Acceleration")
    axs[3, 0].set_xlabel("time (s)")
    axs[3, 0].set_ylabel("m/s^2")
    axs[3, 0].legend()
    axs[3, 0].grid(True)

    axs[3, 1].plot(t_seg, a_lat_raw, "r-", alpha=0.45, label="Raw")
    axs[3, 1].plot(t_seg, a_lat_opt, "b-", linewidth=2, label="Optimized")
    axs[3, 1].set_title("8. Lateral Acceleration")
    axs[3, 1].set_xlabel("time (s)")
    axs[3, 1].set_ylabel("m/s^2")
    axs[3, 1].legend()
    axs[3, 1].grid(True)

    for ax in axs.flat:
        if ax is not axs[0, 0]:
            ax.set_xlim(t_min, t_max)

    plt.tight_layout()
    if output_dir:
        output_path = os.path.join(output_dir, f"velocity_heading_reconstruction_{object_id}.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
    else:
        plt.show(block=True)


class VelocityHeadingTrajectoryOptimizer:
    def __init__(
        self,
        agent_type="vehicle",
        dt=0.1,
        show_solver_warnings=False,
        config=None,
        allow_reverse_speed=False,
    ):
        self.dt = dt
        self.show_solver_warnings = show_solver_warnings
        self.velocity_heading_config = config or VelocityHeadingConfig()
        self.allow_reverse_speed = allow_reverse_speed

        type_str = str(agent_type).lower()
        if "ped" in type_str:
            cfg_key = "pedestrian"
        elif "cyc" in type_str or "bike" in type_str:
            cfg_key = "cyclist"
        else:
            cfg_key = "vehicle"
        self.cfg = copy.deepcopy(AgentConfig.CONFIGS[cfg_key])
        self.cfg["weights"]["pos"] = self.velocity_heading_config.position_weight
        self.cfg["weights"]["heading"] = self.velocity_heading_config.heading_weight
        if (
            self.velocity_heading_config.force_nonnegative_vehicle_speed
            and cfg_key in {"vehicle", "cyclist"}
            and not self.allow_reverse_speed
        ):
            self.cfg["v_min"] = max(0.0, self.cfg["v_min"])

    def preprocess_data(self, x, y, heading_observation, heading_weight):
        n_frames = len(x)
        t = np.arange(n_frames) * self.dt
        position_mask = np.isfinite(x) & np.isfinite(y)
        if position_mask.sum() < 3:
            return None

        x_init = interp1d(
            t[position_mask],
            x[position_mask],
            kind="linear",
            fill_value="extrapolate",
        )(t)
        y_init = interp1d(
            t[position_mask],
            y[position_mask],
            kind="linear",
            fill_value="extrapolate",
        )(t)

        heading_mask = np.isfinite(heading_observation) & (heading_weight > 0)
        if heading_mask.sum() == 0:
            dx = np.gradient(x_init, self.dt)
            dy = np.gradient(y_init, self.dt)
            h_init = np.arctan2(dy, dx)
        else:
            h_init = _interpolate_angles(heading_observation, heading_mask, heading_observation)

        safe_heading_weight = np.where(heading_mask, heading_weight, 0.0)
        return x_init, y_init, h_init, position_mask.astype(float), safe_heading_weight

    def solve(self, x_obs, y_obs, heading_observation, heading_weight):
        weights = self.cfg["weights"]
        n_frames = len(x_obs)
        preprocessed = self.preprocess_data(x_obs, y_obs, heading_observation, heading_weight)
        if preprocessed is None:
            return (None,) * 9

        x_init, y_init, h_init, position_mask, safe_heading_weight = preprocessed

        opti = ca.Opti()
        states = opti.variable(6, n_frames)
        x, y, h, v, a, omega = states[0, :], states[1, :], states[2, :], states[3, :], states[4, :], states[5, :]
        controls = opti.variable(3, n_frames - 1)
        jerk_lon, alpha, v_lat_slack = controls[0, :], controls[1, :], controls[2, :]

        objective = 0
        delta = 2.0

        for i in range(n_frames):
            if position_mask[i] > 0:
                err_x2 = (x[i] - x_obs[i]) ** 2
                err_y2 = (y[i] - y_obs[i]) ** 2
                dist_sq = err_x2 + err_y2
                objective += weights["pos"] * position_mask[i] * (
                    delta ** 2 * (ca.sqrt(1.0 + dist_sq / (delta ** 2)) - 1.0)
                )

            if safe_heading_weight[i] > 0:
                objective += (
                    weights["heading"]
                    * safe_heading_weight[i]
                    * (1 - ca.cos(h[i] - heading_observation[i]))
                )

        objective += weights["j_lon"] * ca.sumsqr(jerk_lon)
        objective += weights["yaw_accel"] * ca.sumsqr(alpha)
        objective += weights["slack_lat"] * ca.sumsqr(v_lat_slack)
        opti.minimize(objective)

        for i in range(n_frames - 1):
            dx = (v[i] * ca.cos(h[i]) - v_lat_slack[i] * ca.sin(h[i])) * self.dt
            dy = (v[i] * ca.sin(h[i]) + v_lat_slack[i] * ca.cos(h[i])) * self.dt
            opti.subject_to(x[i + 1] == x[i] + dx)
            opti.subject_to(y[i + 1] == y[i] + dy)
            opti.subject_to(h[i + 1] == h[i] + omega[i] * self.dt + 0.5 * alpha[i] * (self.dt ** 2))
            opti.subject_to(omega[i + 1] == omega[i] + alpha[i] * self.dt)
            opti.subject_to(v[i + 1] == v[i] + a[i] * self.dt)
            opti.subject_to(a[i + 1] == a[i] + jerk_lon[i] * self.dt)

        opti.subject_to(opti.bounded(self.cfg["v_min"], v, self.cfg["v_max"]))
        opti.subject_to(opti.bounded(self.cfg["a_min"], a, self.cfg["a_max"]))
        opti.subject_to(opti.bounded(-self.cfg["j_max"], jerk_lon, self.cfg["j_max"]))
        alat = v * omega
        opti.subject_to(opti.bounded(-self.cfg["alat_max"], alat, self.cfg["alat_max"]))
        opti.subject_to(opti.bounded(-self.cfg["w_max"], omega, self.cfg["w_max"]))
        opti.subject_to(opti.bounded(-self.cfg["alpha_max"], alpha, self.cfg["alpha_max"]))

        opti.set_initial(x, x_init)
        opti.set_initial(y, y_init)
        opti.set_initial(h, h_init)
        v_guess = np.append(np.hypot(np.diff(x_init), np.diff(y_init)) / self.dt, 0.0)
        if self.cfg["v_min"] >= 0:
            v_guess = np.maximum(v_guess, self.cfg["v_min"])
        opti.set_initial(v, v_guess)
        opti.set_initial(a, np.zeros(n_frames))
        h_diff = angle_diff(np.diff(h_init), 0.0)
        opti.set_initial(omega, np.append(h_diff / self.dt, 0.0))
        opti.set_initial(alpha, np.zeros(n_frames - 1))

        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "max_iter": 1000,
                "print_level": 0,
                "sb": "yes",
                "acceptable_tol": 1e-2,
                "acceptable_iter": 15,
            },
        )

        try:
            sol = opti.solve()
            v_opt = sol.value(v)
            omega_opt = sol.value(omega)
            alat_opt = v_opt * omega_opt
            return (
                sol.value(x),
                sol.value(y),
                sol.value(h),
                v_opt,
                sol.value(a),
                omega_opt,
                alat_opt,
                np.append(sol.value(jerk_lon), 0.0),
                np.append(np.diff(alat_opt) / self.dt, 0.0),
            )
        except Exception:
            if self.show_solver_warnings:
                print("Warning: Solver not fully converged, returning relaxed values.")
            v_opt = opti.debug.value(v)
            omega_opt = opti.debug.value(omega)
            alat_opt = v_opt * omega_opt
            return (
                opti.debug.value(x),
                opti.debug.value(y),
                opti.debug.value(h),
                v_opt,
                opti.debug.value(a),
                omega_opt,
                alat_opt,
                np.append(opti.debug.value(jerk_lon), 0.0),
                np.append(np.diff(alat_opt) / self.dt, 0.0),
            )


def object_track_reconstruction_velocity_heading(
    args,
    timestamp,
    object_state,
    object_type,
    object_id=None,
    config=None,
):
    config = config or VelocityHeadingConfig()
    dt = float(np.median(np.diff(timestamp)))
    if dt > 10:
        dt /= 1e6
    if dt == 0 or np.isnan(dt):
        dt = 0.1

    valid_flag = np.array(object_state["valid"]).flatten().astype(bool)
    valid_indices = np.where(valid_flag)[0]
    if len(valid_indices) < 3:
        return None, None, None, None

    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1
    interp_range = slice(start_idx, end_idx)

    global_center = np.asarray(object_state["global_center"])
    if global_center.shape[1] == 2:
        z_full = np.zeros(len(global_center), dtype=float)
    else:
        z_full = global_center[:, 2]

    x_raw = global_center[interp_range, 0].copy()
    y_raw = global_center[interp_range, 1].copy()
    z_raw = z_full[interp_range].copy()
    h_raw = object_state["heading"][interp_range].copy()
    size_raw = object_state["size"][interp_range, :].copy()
    valid_seg = valid_flag[interp_range]
    global_velocity = object_state.get("global_velocity")
    if global_velocity is not None:
        global_velocity = np.asarray(global_velocity)[interp_range, :]

    x_raw[~valid_seg] = np.nan
    y_raw[~valid_seg] = np.nan
    h_raw[~valid_seg] = np.nan

    allow_reverse_speed = has_sustained_reverse_motion(
        x_raw,
        y_raw,
        h_raw,
        valid_seg,
        dt,
        global_velocity=global_velocity,
        config=config,
    )
    heading_observation, heading_weight, velocity_speed, velocity_heading, _ = build_velocity_heading_observation(
        x_raw,
        y_raw,
        h_raw,
        valid_seg,
        dt,
        object_type,
        global_velocity=global_velocity,
        config=config,
    )

    optimizer = VelocityHeadingTrajectoryOptimizer(
        agent_type=object_type,
        dt=dt,
        show_solver_warnings=getattr(args, "show_solver_warnings", False),
        config=config,
        allow_reverse_speed=allow_reverse_speed,
    )
    res = optimizer.solve(x_raw, y_raw, heading_observation, heading_weight)
    if res[0] is None:
        return None, None, None, None

    x_opt, y_opt, h_opt, v_lon, a_lon, omega_opt, a_lat, j_lon, j_lat = res
    v_lat = np.zeros_like(v_lon)

    z_opt = smooth_z_axis(z_raw, valid_seg)
    mean_size = np.nanmean(size_raw[valid_seg], axis=0)
    reconstructed_size_value = np.tile(mean_size, (len(x_opt), 1))

    x_fill = fill_nan_for_calc(x_raw)
    y_fill = fill_nan_for_calc(y_raw)
    h_fill = fill_nan_for_calc(h_raw)
    v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw, j_lon_raw, j_lat_raw = calc_kinematics_local(
        x_fill,
        y_fill,
        h_fill,
        dt,
    )

    if getattr(args, "visualize", False):
        full_time_array = np.arange(len(timestamp)) * dt
        t_seg = full_time_array[interp_range]
        t_range = (full_time_array[0], full_time_array[-1])

        raw_data = (x_raw, y_raw, h_raw, v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw)
        clean_data = (heading_observation, heading_weight, velocity_speed, velocity_heading)
        opt_data = (x_opt, y_opt, h_opt, v_lon, v_lat, a_lon, a_lat)
        output_dir = (
            getattr(args, "visualize_output_dir", None)
            or getattr(args, "visualization_output_dir", None)
            or getattr(args, "visualize_save_dir", None)
        )
        figsize = _normalize_figsize(getattr(args, "visualize_figsize", None))
        plot_velocity_heading_graphs(
            t_seg,
            raw_data,
            clean_data,
            opt_data,
            object_id,
            t_range,
            object_type,
            output_dir=output_dir,
            figsize=figsize,
        )

    reconstructed_timestamp = timestamp[interp_range]
    filled_part_data = np.column_stack((
        x_fill, y_fill, v_lon_raw, v_lat_raw, a_lon_raw, a_lat_raw, j_lon_raw, j_lat_raw, h_fill
    ))
    noise_removal_part_data = np.column_stack((
        x_opt, y_opt, v_lon, v_lat, a_lon, a_lat, j_lon, j_lat, h_opt
    ))

    if global_center.shape[1] == 2:
        object_state["global_center"][interp_range, :] = np.column_stack((x_opt, y_opt))
    else:
        object_state["global_center"][interp_range, :] = np.column_stack((x_opt, y_opt, z_opt))
    object_state["local_velocity"][interp_range, :] = np.column_stack((v_lon, v_lat))
    object_state["local_acceleration"][interp_range, :] = np.column_stack((a_lon, a_lat))
    if "local_jerk" not in object_state:
        object_state["local_jerk"] = np.zeros((len(object_state["valid"]), 2))
    object_state["local_jerk"][interp_range, :] = np.column_stack((j_lon, j_lat))
    object_state["heading"][interp_range] = h_opt
    object_state["size"][interp_range, :] = reconstructed_size_value
    object_state["valid"][interp_range] = True

    return reconstructed_timestamp, object_state, filled_part_data, noise_removal_part_data


def read_file(path, file, subset_name):
    full_path = os.path.join(path, subset_name, file)
    with open(full_path, "rb") as f:
        return pickle.load(f)


def save_file(path, scenario, file, subset_name):
    full_dir = os.path.join(path, subset_name)
    os.makedirs(full_dir, exist_ok=True)
    with open(os.path.join(full_dir, file), "wb") as f:
        pickle.dump(scenario, f)


def process_single_file_velocity_heading(args, converted_subset_path, reconstructed_subset_path, file):
    try:
        scenario = read_file(converted_subset_path, file, args.subset_name)
        object_tracks = scenario["object_tracks"]
        extra_information = scenario["extra_information"]

        for object_id in list(object_tracks.keys()):
            object_track = object_tracks[object_id]
            timestamp = extra_information["timestamp"]
            object_state = object_track["state"]
            object_type = object_track["type"]
            _, reconstructed_object_state, _, _ = object_track_reconstruction_velocity_heading(
                args,
                timestamp,
                object_state,
                object_type,
                object_id,
            )
            if reconstructed_object_state is None:
                continue
            object_track["state"] = reconstructed_object_state

        save_file(reconstructed_subset_path, scenario, file, args.subset_name)
        return file, True
    except Exception as exc:
        print(f"Error processing {file}: {str(exc)}")
        return file, False


object_track_reconstruction = object_track_reconstruction_velocity_heading
