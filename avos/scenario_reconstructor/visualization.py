"""Paper-style reconstruction plots, independent of the reconstruction algorithm.

Inputs are examples loaded by scripts/plot_waymo_reconstruction_examples.py.
Optional real intermediate snapshots enable After step 1/2/3 legends. Without
these snapshots the plot explicitly compares recorded and reconstructed data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


AXIS_LIMITS = {
    "position_x_m": (-10., 150.), "position_y_m": (-40., 40.),
    "heading_rad": (-np.pi, np.pi), "longitudinal_velocity_mps": (0., 20.),
    "lateral_velocity_mps": (-5., 5.), "longitudinal_acceleration_mps2": (-8., 6.),
    "lateral_acceleration_mps2": (-4., 4.), "time_s": (0., 20.),
}


def _position_in_initial_frame(state, indices, origin, heading):
    xy = np.asarray(state["global_center"], float)[indices, :2] - origin
    c, s = np.cos(heading), np.sin(heading)
    return xy @ np.array([[c, -s], [s, c]])


def continuous_heading_display(series):
    """Unwrap display copies, align branches, and choose a movable 2*pi window.

    Missing samples remain missing. Only integer multiples of 2*pi are added;
    neither stored states nor motion quantities are changed.
    """
    values = []
    for source in series:
        a = np.array(source, dtype=float, copy=True)
        valid = np.isfinite(a)
        if valid.any():
            a[valid] = np.unwrap(a[valid])
        values.append(a)
    reference = values[0]
    for a in values[1:]:
        shared = np.isfinite(reference) & np.isfinite(a)
        if shared.any():
            offset = 2 * np.pi * np.round(np.median(a[shared] - reference[shared]) / (2 * np.pi))
            a -= offset
    finite = np.concatenate([a[np.isfinite(a)] for a in values])
    center = (finite.min() + finite.max()) / 2 if finite.size else 0.
    limits = (center - np.pi, center + np.pi)
    ticks = np.arange(np.ceil(limits[0] / 1.5), np.floor(limits[1] / 1.5) + 1) * 1.5
    return values, limits, ticks


def plot_reconstruction_examples(examples, output_stem, scenario_id, reference_bounds=None,
                                 current_pipeline_note=False):
    """Save a paper-style PNG/PDF/SVG with a 2x3 panel for each example.

    reference_bounds, when explicitly supplied, maps participant names to
    {"longitudinal": [low, high], "lateral": [low, high]} in m/s^2.
    Bounds are displayed as reference lines, not enforced physical constraints.
    No limits from the vehicle example are assumed for cyclists or pedestrians.
    Returns the figure for inspection; the caller is responsible for closing it.
    """
    if not examples:
        raise ValueError("At least one trajectory example is required")
    modes = [bool(e.get("stages")) for e in examples]
    if any(modes) and not all(modes):
        raise ValueError("Stage snapshots must be supplied for every example")
    for e in examples:
        if e.get("stages") and set(e["stages"]) != {1, 2, 3}:
            raise ValueError("All three stage snapshots are required")
    reference_bounds = reference_bounds or {}
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
             "axes.edgecolor": "#555555", "axes.linewidth": .8,
             "axes.grid": False, "legend.fontsize": 8.5,
             "legend.frameon": True, "legend.framealpha": .85,
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12, 3.95 * len(examples)))
        outer = fig.add_gridspec(len(examples), 1, left=.084, right=.978,
                                bottom=.055, top=.962, hspace=.20)
        for row, e in enumerate(examples):
            panel = outer[row].subgridspec(2, 3, hspace=.40, wspace=.38)
            axes = [fig.add_subplot(panel[r, c]) for r in range(2) for c in range(3)]
            group_box = outer[row].get_position(fig)
            fig.text(group_box.x0 - .065, group_box.y1 + .007, chr(97+row),
                     ha="left", va="bottom", fontsize=12, weight="bold")
            idx, times, valid = e["indices"], e["times"], e["valid"]
            raw, rec = e["raw"], e["rec"]
            origin = np.asarray(raw["global_center"], float)[idx[0], :2]
            initial_heading = float(raw["heading"][idx[0]])
            raw_xy = _position_in_initial_frame(raw, idx, origin, initial_heading)
            raw_h = np.asarray(raw["heading"], float)[idx]
            raw_h = raw_h.copy()
            raw_h[~valid] = np.nan
            heading_sources = ([raw_h] + [np.asarray(e['stages'][n]['state']['heading'])[idx] for n in (1, 2, 3)]
                               if e.get('stages') else [raw_h, np.asarray(rec['heading'])[idx]])
            heading_values, heading_limits, heading_ticks = continuous_heading_display(heading_sources)
            axes[0].plot(raw_xy[valid, 0], raw_xy[valid, 1], "k.", ms=2, label="Raw data")
            axes[1].plot(times[valid], heading_values[0][valid], "k.", ms=2, label="Raw data")
            if e.get("stages"):
                stage1, stage2, stage3 = [e["stages"][i] for i in (1, 2, 3)]
                for stage, color, label in [(stage1, "#E5B700", "After step 1"), (stage3, "blue", "After step 3")]:
                    xy = _position_in_initial_frame(stage["state"], idx, origin, initial_heading)
                    axes[0].plot(xy[:, 0], xy[:, 1], color=color, lw=.85, label=label)
                for angles, color, label in [(heading_values[1], "#E5B700", "After step 1"), (heading_values[2], "green", "After step 2"), (heading_values[3], "blue", "After step 3")]:
                    axes[1].plot(times, angles, color=color, lw=.85, label=label)
                motion_series = [(stage1["motion"], "#E5B700", "After step 1"),
                                 (stage3["motion"], "blue", "After step 3")]
            else:
                rec_xy = _position_in_initial_frame(rec, idx, origin, initial_heading)
                axes[0].plot(rec_xy[:, 0], rec_xy[:, 1], color="blue", lw=.85, label="Reconstructed")
                axes[1].plot(times, heading_values[1], color="green", lw=.85, label="Reconstructed")
                motion_series = [(e["raw_motion"], "#E5B700", "From raw data"),
                                 (e["rec_motion"], "blue", "Reconstructed")]
            labels = ["Longitudinal velocity (m/s)", "Lateral velocity (m/s)",
                      r"Longitudinal acceleration (m/s$^2$)", r"Lateral acceleration (m/s$^2$)"]
            for col, ax in enumerate(axes[2:]):
                for motion, color, label in motion_series:
                    ax.plot(times, motion[:, col], color=color, lw=.85, label=label)
                ax.set_ylabel(labels[col])
            axes[0].set(xlabel="Longitudinal position (m)", ylabel="Lateral position (m)")
            axes[0].set_aspect("equal", adjustable="box")
            axes[1].set_ylabel("Heading (rad)")
            for ax in axes[1:]:
                ax.set_xlabel("Time (s)")
                ax.set_xlim(0, 20)
            for direction, ax in [("longitudinal", axes[4]), ("lateral", axes[5])]:
                bounds = reference_bounds.get(e["label"].lower(), {}).get(direction)
                if bounds is not None:
                    if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                        raise ValueError(f"Invalid {direction} reference bounds")
                    for value in bounds:
                        ax.axhline(value, color="red", lw=.7)
                        ax.text(.012, value, f"{value:g}", color="red", fontsize=8,
                                va="center", transform=ax.get_yaxis_transform())
            for ax in axes:
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                ax.tick_params(direction="out", length=3, width=.7)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.margins(y=.09)
                # All panels have equal widths. A fixed axes-relative label
                # anchor aligns each of the three columns across all six rows,
                # independently of tick widths or single/multiline labels.
                # Use the original grid slot in figure coordinates so an
                # equal-aspect trajectory box cannot shift column labels.
                slot = ax.get_position(original=True)
                ax.yaxis.set_label_coords(slot.x0 - .24 * slot.width,
                                         slot.y0 + .5 * slot.height,
                                         transform=fig.transFigure)
                ax.yaxis.label.set_horizontalalignment("center")
                ax.yaxis.label.set_verticalalignment("center")
                ax.yaxis.label.set_multialignment("center")
                ax.legend(loc="best", handlelength=2, borderpad=.4, labelspacing=.25)
            for ax in axes[1:]:
                ax.set_xticks([0, 5, 10, 15, 20])
            axes[0].set_xlim(*AXIS_LIMITS["position_x_m"])
            axes[0].set_ylim(*AXIS_LIMITS["position_y_m"])
            axes[0].set_xticks([0, 50, 100, 150])
            axes[0].set_yticks([-40, -20, 0, 20, 40])
            limits = ["heading_rad", "longitudinal_velocity_mps", "lateral_velocity_mps",
                      "longitudinal_acceleration_mps2", "lateral_acceleration_mps2"]
            for ax, key in zip(axes[1:], limits):
                ax.set_ylim(*AXIS_LIMITS[key])
            axes[1].set_ylim(*heading_limits)
            axes[1].set_yticks(heading_ticks, labels=[f'{0.0 if v == 0 else v:.1f}' for v in heading_ticks])
            for ax, ticks in zip(axes[2:], [[0, 5, 10, 15, 20], [-5, -2.5, 0, 2.5, 5],
                                           [-8, -4, 0, 3, 6], [-4, -2, 0, 2, 4]]):
                ax.set_yticks(ticks)
        stem = Path(output_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".svg", ".png"):
            fig.savefig(stem.with_suffix(suffix), dpi=300)
        stages_text = ("Yellow, green, and blue curves denote the results after interpolation (step 1), "
                       "heading processing (step 2), and trajectory processing (step 3), respectively. "
                       if all(modes) else "The panels compare recorded trajectories and reconstructed outputs. ")
        caption = ("Examples of object trajectory reconstruction from Waymo Open: "
                   "(a) vehicle, (b) cyclist, and (c) pedestrian. Black dots show raw observations. "
                   + stages_text +
                   "Corresponding motion panels use common axis limits across the three examples. "
                   "All time axes span 0-20 s, with no extrapolation outside each observed interval. "
                   "Heading display copies are unwrapped and aligned to a common angular branch for each object, retaining missing samples. "
                   "Each heading panel uses a movable window of width 2*pi rad, with numerical ticks spaced 1.5 rad apart. "
                   "For a trajectory spanning more than 2*pi, portions outside this window remain clipped; the gallery full-range mode shows the complete span. "
                   "Position is expressed in the initial object frame. Fixed engineering reference ranges are used for motion quantities, "
                   "and portions exceeding these ranges are clipped only in the display; the underlying data are retained.")
        if current_pipeline_note:
            caption += " The plotted results use the current repository's Butterworth implementation, which differs from the Savitzky-Golay filter described in the manuscript."
        caption += "\n\nSource: scenario " + scenario_id + "; " + ", ".join(
            f"({chr(97+i)}) object {e['object_id']}" for i, e in enumerate(examples)) + ".\n"
        stem.with_name(stem.name + "_caption.txt").write_text(caption)
        return fig
