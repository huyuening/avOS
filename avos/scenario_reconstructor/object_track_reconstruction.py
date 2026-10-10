from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline, interp1d


DEFAULT_CUTOFF_FREQ = 1.0

DATASET_CUTOFF_FREQ = {
    "argoverse-1": 2.0,
    "argoverse-2": 2.0,
    "interaction": 1.0,
    "levelxdata": 1.0,
    "lyft-level-5": 0.5,
    "nuplan": 0.5,
    "shifts": 1.0,
    "v2x-seq": 0.5,
    "waymo-open": 1.0,
}


@dataclass(frozen=True)
class ReconstructionConfig:
    dataset: str
    cutoff_freq: float | None = None
    sampling_rate: float | None = None
    min_valid_frames: int = 2
    max_speed_mps: float = 80.0
    filter_order: int = 4
    heading_reference_min_speed_mps: float = 1.0
    heading_tolerance_rad: float = 0.1 * np.pi

    @property
    def resolved_cutoff_freq(self) -> float:
        if self.cutoff_freq is not None:
            return self.cutoff_freq
        return DATASET_CUTOFF_FREQ.get(self.dataset, DEFAULT_CUTOFF_FREQ)


@dataclass
class TrackReconstructionResult:
    state: dict
    reconstructed: bool
    skipped_reason: str | None = None
    filled_samples: int = 0
    stage_states: dict[int, dict] | None = None
    heading_outlier_mask: np.ndarray | None = None


def reconstruct_object_state(
    object_state: dict,
    timestamp: np.ndarray,
    config: ReconstructionConfig,
    *,
    capture_stages: bool = False,
) -> TrackReconstructionResult:
    timestamp = np.asarray(timestamp, dtype=np.float64)
    state = _copy_state(object_state)

    if (timestamp.ndim != 1 or len(timestamp) < 2
            or not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0)):
        return TrackReconstructionResult(state=state, reconstructed=False, skipped_reason="invalid_timestamp")

    center = _get_global_center(state)
    heading = _as_1d_array(state.get("heading"), len(timestamp), dtype=np.float64)
    valid = _as_1d_array(state.get("valid"), len(timestamp), dtype=bool)
    size = _as_2d_array(state.get("size"), len(timestamp), 3)

    if center is None or heading is None or valid is None:
        return TrackReconstructionResult(state=state, reconstructed=False, skipped_reason="missing_required_state")

    if len(center) != len(timestamp):
        return TrackReconstructionResult(state=state, reconstructed=False, skipped_reason="state_length_mismatch")

    finite_mask = np.isfinite(center[:, 0]) & np.isfinite(center[:, 1]) & np.isfinite(heading)
    valid = valid & finite_mask
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < max(2, config.min_valid_frames):
        return TrackReconstructionResult(state=state, reconstructed=False, skipped_reason="not_enough_valid_frames")

    start_index = int(valid_indices[0])
    end_index = int(valid_indices[-1])
    segment_indices = np.arange(start_index, end_index + 1)
    segment_valid = valid[segment_indices]
    valid_segment_indices = segment_indices[segment_valid]

    if valid_segment_indices.size < max(2, config.min_valid_frames):
        return TrackReconstructionResult(state=state, reconstructed=False, skipped_reason="not_enough_valid_frames")

    filled_center = _interpolate_center(center, valid_segment_indices, segment_indices)
    stage_states = {} if capture_stages else None
    if capture_stages:
        raw_segment_heading = heading[segment_indices].copy()
        raw_segment_heading[~segment_valid] = np.nan
        stage_states[1] = _diagnostic_snapshot(
            state, segment_indices, filled_center, raw_segment_heading, timestamp,
        )
    # The legacy batch workflow forward-fills missing headings before deriving
    # a reference from the filtered positions. Zero is a valid angle here.
    filled_heading = _forward_fill_heading(heading, valid_segment_indices, segment_indices)
    filled_center, filled_heading = _repair_speed_outliers(
        filled_center, filled_heading, timestamp[segment_indices], config,
    )

    sampling_rate = _sampling_rate(timestamp[segment_indices], config.sampling_rate)
    cutoff_freq = config.resolved_cutoff_freq
    smoothed_center = np.column_stack(
        (
            _butterworth_filter(filled_center[:, 0], cutoff_freq, sampling_rate, config.filter_order),
            _butterworth_filter(filled_center[:, 1], cutoff_freq, sampling_rate, config.filter_order),
            _butterworth_filter(filled_center[:, 2], cutoff_freq, sampling_rate, config.filter_order),
        )
    )
    reference_heading, moving = calculate_reference_heading(
        timestamp[segment_indices], smoothed_center[:, :2], filled_heading,
        config.heading_reference_min_speed_mps,
    )
    corrected_heading, rejected = correct_heading(
        filled_heading, reference_heading, timestamp[segment_indices],
        config.heading_tolerance_rad, moving,
    )
    if capture_stages:
        stage_states[2] = _diagnostic_snapshot(
            state, segment_indices, filled_center, corrected_heading, timestamp,
        )
    smoothed_heading = _normalize_angle(
        _butterworth_filter(corrected_heading, cutoff_freq, sampling_rate, config.filter_order)
    )
    heading_outlier_mask = np.zeros(len(timestamp), dtype=bool)
    heading_outlier_mask[segment_indices] = rejected

    local_velocity, local_acceleration = _motion_from_position(
        timestamp[segment_indices],
        smoothed_center[:, 0],
        smoothed_center[:, 1],
        smoothed_heading,
    )

    state["global_center"] = _as_2d_array(state.get("global_center"), len(timestamp), 3, fill=0.0)
    state["local_velocity"] = _as_2d_array(state.get("local_velocity"), len(timestamp), 2, fill=0.0)
    state["local_acceleration"] = _as_2d_array(state.get("local_acceleration"), len(timestamp), 2, fill=0.0)
    state["heading"] = _as_1d_array(state.get("heading"), len(timestamp), dtype=np.float64, fill=0.0)
    state["size"] = _as_2d_array(size, len(timestamp), 3, fill=0.0)
    state["valid"] = _as_1d_array(state.get("valid"), len(timestamp), dtype=bool, fill=False)

    state["global_center"][segment_indices] = smoothed_center
    state["local_velocity"][segment_indices] = local_velocity
    state["local_acceleration"][segment_indices] = local_acceleration
    state["heading"][segment_indices] = smoothed_heading
    state["size"][segment_indices] = _representative_size(size, valid)
    state["valid"][segment_indices] = True

    filled_samples = int((~segment_valid).sum())
    if capture_stages:
        stage_states[3] = _copy_state(state)
    return TrackReconstructionResult(state=state, reconstructed=True,
                                     filled_samples=filled_samples, stage_states=stage_states,
                                     heading_outlier_mask=heading_outlier_mask)


def _diagnostic_snapshot(state, indices, center, heading, timestamp):
    """Copy actual intermediate position/heading arrays without changing the run.

    Motion fields are derived for inspection only. Step 1 is position
    interpolation, with original headings retained and missing headings NaN.
    Step 2 uses the velocity direction of filtered positions to identify and
    interpolate heading outliers. Step 3 stores filtered positions/headings
    and their derived motion. Position filtering is also used internally to
    construct the Step 2 reference, as in the legacy batch implementation.
    The underlying filter differs from the manuscript's SG workflow.
    """
    snapshot = _copy_state(state)
    n = len(timestamp)
    snapshot["global_center"] = _as_2d_array(snapshot.get("global_center"), n, 3, fill=0.)
    snapshot["heading"] = _as_1d_array(snapshot.get("heading"), n, float, fill=0.)
    snapshot["valid"] = _as_1d_array(snapshot.get("valid"), n, bool, fill=False)
    snapshot["local_velocity"] = _as_2d_array(snapshot.get("local_velocity"), n, 2, fill=0.)
    snapshot["local_acceleration"] = _as_2d_array(snapshot.get("local_acceleration"), n, 2, fill=0.)
    snapshot["global_center"][indices] = center
    snapshot["heading"][indices] = heading
    snapshot["valid"][indices] = True
    velocity, acceleration = _motion_from_position(timestamp[indices], center[:, 0], center[:, 1], heading)
    snapshot["local_velocity"][indices] = velocity
    snapshot["local_acceleration"][indices] = acceleration
    return snapshot


def _copy_state(state: dict) -> dict:
    copied = {}
    for key, value in state.items():
        copied[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return copied


def _get_global_center(state: dict) -> np.ndarray | None:
    if "global_center" in state:
        center = np.asarray(state["global_center"], dtype=np.float64)
        if center.ndim == 2 and center.shape[1] >= 2:
            if center.shape[1] == 2:
                center = np.column_stack((center, np.zeros(len(center), dtype=np.float64)))
            return center[:, :3]

    if "center_x" in state and "center_y" in state:
        center_x = np.asarray(state["center_x"], dtype=np.float64)
        center_y = np.asarray(state["center_y"], dtype=np.float64)
        center_z = np.asarray(state.get("center_z", np.zeros_like(center_x)), dtype=np.float64)
        if center_x.shape == center_y.shape == center_z.shape:
            return np.column_stack((center_x, center_y, center_z))

    return None


def _as_1d_array(value, length: int, dtype, fill=None) -> np.ndarray | None:
    if value is None:
        if fill is None:
            return None
        return np.full(length, fill, dtype=dtype)

    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 or len(array) != length:
        return None
    return np.array(array, copy=True)


def _as_2d_array(value, length: int, width: int, fill=None) -> np.ndarray | None:
    if value is None:
        if fill is None:
            return None
        return np.full((length, width), fill, dtype=np.float64)

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or len(array) != length:
        return None
    if array.shape[1] < width:
        pad = np.zeros((length, width - array.shape[1]), dtype=np.float64)
        array = np.column_stack((array, pad))
    return np.array(array[:, :width], copy=True)


def _interpolate_center(center: np.ndarray, valid_indices: np.ndarray, segment_indices: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            _interpolate_series(valid_indices, center[valid_indices, dim], segment_indices)
            for dim in range(3)
        ]
    )


def _interpolate_series(valid_indices: np.ndarray, values: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    kind = _interpolation_kind(valid_indices.size)
    interpolator = interp1d(valid_indices, values, kind=kind, bounds_error=False, fill_value="extrapolate")
    return np.asarray(interpolator(target_indices), dtype=np.float64)


def _interpolation_kind(num_points: int) -> str:
    if num_points >= 4:
        return "cubic"
    if num_points == 3:
        return "quadratic"
    return "linear"


def _forward_fill_heading(heading, valid_indices, segment_indices):
    source = np.searchsorted(valid_indices, segment_indices, side="right") - 1
    return np.unwrap(np.asarray(heading, float)[valid_indices[source]])


def calculate_reference_heading(timestamp, center_xy, heading, min_speed_mps=1.0):
    """Legacy velocity-direction reference with a low-speed fallback.

    Headings are directed angles, so unwrap with a 2*pi period. The legacy
    helper's second, pi-period unwrap can introduce a 180-degree flip.
    """
    velocity = np.gradient(center_xy, timestamp, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    moving = speed >= min_speed_mps
    reference = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    reference[~moving] = np.asarray(heading)[~moving]
    return np.unwrap(reference), moving


def correct_heading(heading, reference, timestamp, tolerance_rad=0.1 * np.pi,
                    moving=None):
    """Restore legacy threshold rejection and interpolation without zero sentinels.

    Accepted angles align to the reference by whole turns. Rejected interior
    samples use a natural cubic spline (three or more anchors); boundaries
    use linear extrapolation. With two anchors use linear interpolation,
    with one use its value, and with none use the motion reference.
    """
    heading = np.asarray(heading, dtype=float)
    reference = np.asarray(reference, dtype=float)
    timestamp = np.asarray(timestamp, dtype=float)
    moving = np.ones(len(heading), bool) if moving is None else np.asarray(moving, bool)
    aligned = heading - np.round((heading - reference) / (2 * np.pi)) * (2 * np.pi)
    rejected = moving & (np.abs(_normalize_angle(heading - reference)) > tolerance_rad)
    anchors = np.flatnonzero(~rejected)
    result = aligned.copy()
    if not rejected.any():
        return result, rejected
    if len(anchors) == 0:
        return reference.copy(), rejected
    if len(anchors) == 1:
        result[rejected] = aligned[anchors[0]]
        return result, rejected
    linear = interp1d(timestamp[anchors], aligned[anchors], kind="linear",
                      bounds_error=False, fill_value="extrapolate")
    result[rejected] = linear(timestamp[rejected])
    if len(anchors) >= 3:
        inside = rejected & (timestamp > timestamp[anchors[0]]) & (timestamp < timestamp[anchors[-1]])
        cubic = CubicSpline(timestamp[anchors], aligned[anchors], bc_type="natural")
        result[inside] = cubic(timestamp[inside])
    return result, rejected


def _repair_speed_outliers(
    center: np.ndarray,
    heading: np.ndarray,
    timestamp: np.ndarray,
    config: ReconstructionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.max_speed_mps <= 0 or len(center) < 4:
        return center, heading

    dt = np.diff(timestamp)
    if np.any(dt <= 0):
        return center, heading

    step_distance = np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1)
    speed = step_distance / dt
    bad_steps = np.flatnonzero(speed > config.max_speed_mps)
    if bad_steps.size == 0:
        return center, heading

    keep = np.ones(len(center), dtype=bool)
    keep[np.clip(bad_steps + 1, 0, len(center) - 1)] = False
    keep[0] = True
    keep[-1] = True
    if keep.sum() < max(2, config.min_valid_frames):
        return center, heading

    indices = np.arange(len(center))
    repaired_center = np.column_stack(
        [_interpolate_series(indices[keep], center[keep, dim], indices) for dim in range(3)]
    )
    repaired_heading = _normalize_angle(np.interp(indices, indices[keep], np.unwrap(heading[keep])))
    return repaired_center, repaired_heading


def _sampling_rate(timestamp: np.ndarray, configured_sampling_rate: float | None) -> float:
    if configured_sampling_rate is not None and configured_sampling_rate > 0:
        return float(configured_sampling_rate)

    dt = np.diff(timestamp)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 10.0
    return float(1.0 / np.median(dt))


def _butterworth_filter(data: np.ndarray, cutoff_freq: float, sampling_rate: float, order: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if len(data) < max(order + 2, 5) or cutoff_freq <= 0 or sampling_rate <= 0:
        return data.copy()

    nyquist = sampling_rate * 0.5
    normalized_cutoff = cutoff_freq / nyquist
    if normalized_cutoff <= 0 or normalized_cutoff >= 1:
        return data.copy()

    b, a = signal.butter(order, normalized_cutoff, btype="low")
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    if padlen < 1:
        return data.copy()
    return signal.filtfilt(b, a, data, padlen=padlen)


def _motion_from_position(
    timestamp: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    heading: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_order = 2 if len(timestamp) >= 3 else 1
    velocity_x = np.gradient(center_x, timestamp, edge_order=edge_order)
    velocity_y = np.gradient(center_y, timestamp, edge_order=edge_order)
    acceleration_x = np.gradient(velocity_x, timestamp, edge_order=edge_order)
    acceleration_y = np.gradient(velocity_y, timestamp, edge_order=edge_order)

    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)

    local_velocity = np.column_stack(
        (
            velocity_x * cos_heading + velocity_y * sin_heading,
            -velocity_x * sin_heading + velocity_y * cos_heading,
        )
    )
    local_acceleration = np.column_stack(
        (
            acceleration_x * cos_heading + acceleration_y * sin_heading,
            -acceleration_x * sin_heading + acceleration_y * cos_heading,
        )
    )
    return local_velocity, local_acceleration


def _representative_size(size: np.ndarray | None, valid: np.ndarray) -> np.ndarray:
    if size is None:
        return np.zeros(3, dtype=np.float64)

    valid_size = size[valid]
    valid_size = valid_size[np.all(np.isfinite(valid_size), axis=1)]
    valid_size = valid_size[np.any(valid_size > 0, axis=1)]
    if len(valid_size) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.mean(valid_size, axis=0)


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


if __name__ == "__main__":
    raise SystemExit("Scenario reconstructor modules are internal. Use scripts/reconstruct.py.")
