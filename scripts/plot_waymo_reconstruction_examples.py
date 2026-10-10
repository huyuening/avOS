"""Plot three recorded Waymo tracks and their existing avOS reconstruction outputs.

This never modifies source datasets. --export-current-stages reruns reconstruction
with the restored velocity-reference heading correction and captures each stage.
SG Step 3 uses the saved filtered velocity and acceleration; other stages derive motion from positions/headings.
Paper styling is the default; --style modern retains the earlier design.
Real stage snapshots may be supplied with --stage1-root/--stage2-root/--stage3-root.
Run in the avOS environment. All paths may be overridden on the CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
import sys
from pathlib import Path
import pickle
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "104202f4f2590dff"
EXAMPLES = [("Vehicle", "1"), ("Cyclist", "197"), ("Pedestrian", "158")]
RAW = "#CD6337"
REC = "#007B9B"
GAP = "#E8DCBC"


def contiguous_runs(mask):
    indices = np.flatnonzero(mask)
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def derived_motion(state, times, valid):
    """Differentiate within valid runs, never across unobserved intervals.

    Both versions use actual timestamps, numpy.gradient(edge_order=2), and
    rotation by their own heading. Each derivative is masked at run endpoints,
    where a centered stencil is unavailable. Acceleration also masks the next
    sample at each end, whose stencil would depend on an endpoint derivative.
    """
    result = np.full((len(times), 4), np.nan)
    xy = np.asarray(state["global_center"], float)[:, :2]
    heading = np.asarray(state["heading"], float)
    mask = valid & np.isfinite(xy).all(axis=1) & np.isfinite(heading)
    for run in contiguous_runs(mask):
        if len(run) < 3:
            continue
        velocity = np.gradient(xy[run], times[run], axis=0, edge_order=2)
        acceleration = np.gradient(velocity, times[run], axis=0, edge_order=2)
        c, s = np.cos(heading[run]), np.sin(heading[run])
        result[run, 0] = velocity[:, 0] * c + velocity[:, 1] * s
        result[run, 1] = -velocity[:, 0] * s + velocity[:, 1] * c
        result[run, 2] = acceleration[:, 0] * c + acceleration[:, 1] * s
        result[run, 3] = -acceleration[:, 0] * s + acceleration[:, 1] * c
        result[run[[0, -1]], :2] = np.nan
        result[run[:2], 2:] = np.nan
        result[run[-2:], 2:] = np.nan
    return result


def load_pair(raw_root, reconstructed_root, scenario_id=SCENARIO, object_ids=None):
    paths = [root / f"{scenario_id}.pkl" for root in (raw_root, reconstructed_root)]
    # The user-owned local avOS files are trusted pickle inputs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        records = [pickle.loads(path.read_bytes()) for path in paths]
    a, b = records
    for data in records:
        assert data["extra_information"]["id"] == scenario_id
        assert data["extra_information"]["dataset"] == "Waymo Open"
    times = np.asarray(a["extra_information"]["timestamp"], float)
    np.testing.assert_array_equal(times, b["extra_information"]["timestamp"])
    assert np.all(np.diff(times) > 0)
    examples = []
    selected_examples = EXAMPLES if object_ids is None else list(zip([e[0] for e in EXAMPLES], object_ids))
    for label, object_id in selected_examples:
        ao, bo = [data["object_tracks"][object_id] for data in records]
        assert ao["type"].upper() == bo["type"].upper() == label.upper()
        raw, rec = ao["state"], bo["state"]
        valid = np.asarray(raw["valid"], bool)
        idx = np.flatnonzero(valid)
        selected = np.arange(idx[0], idx[-1] + 1)
        rv = np.asarray(rec["valid"], bool)
        assert rv[selected].all(), "Reconstruction must cover the displayed interval"
        rmask = rv.copy()
        rmask[:idx[0]] = False
        rmask[idx[-1]+1:] = False
        raw_motion = derived_motion(raw, times, valid)
        rec_motion = derived_motion(rec, times, rmask)
        examples.append(dict(label=label, object_id=object_id, raw=raw, rec=rec,
                             indices=selected, valid=valid[selected],
                             times=times[selected], raw_motion=raw_motion[selected],
                             rec_motion=rec_motion[selected]))
    return paths, examples


def draw(examples, out, scenario_id=SCENARIO):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.3, "axes.labelsize": 8,
                         "xtick.labelsize": 7.3, "ytick.labelsize": 7.3,
                         "axes.linewidth": .6, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(10.8, 13.5), facecolor="white")
    grid = fig.add_gridspec(3, 1, left=.077, right=.978, bottom=.044, top=.885, hspace=.30)
    fig.text(.077, .98, "Waymo trajectory reconstruction examples", fontsize=14, weight="bold", va="top")
    fig.text(.077, .954, "Vehicle, cyclist and pedestrian | paired recorded and reconstructed trajectories", fontsize=9, color="#47505A")
    handles = [Line2D([], [], color=RAW, lw=.8, marker="o", markersize=3, markerfacecolor="white", label="Recorded"),
               Line2D([], [], color=REC, lw=1.5, label="Reconstructed"),
               Patch(facecolor=GAP, alpha=.6, label="Missing recorded frames")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.069, .94), frameon=False, ncol=3, fontsize=8.5,
               handlelength=2.4, columnspacing=2)
    all_axes = []
    for number, e in enumerate(examples):
        inner = grid[number].subgridspec(2, 3, hspace=.49, wspace=.34)
        axs = np.array([[fig.add_subplot(inner[row, col]) for col in range(3)] for row in range(2)]).flatten()
        all_axes.extend(axs)
        top = grid[number].get_position(fig).y1
        t, idx, valid = e["times"], e["indices"], e["valid"]
        fig.text(.077, top + .025, f"({chr(97+number)}) {e['label']}", fontsize=10.5, weight="bold", color="#163B49")
        fig.text(.29, top + .026, f"Object {e['object_id']}  |  {t[0]:.1f}-{t[-1]:.1f} s  |  {int((~valid).sum())} missing frames", fontsize=8, color="#47505A")
        raw_xy = np.asarray(e["raw"]["global_center"], float)[idx, :2].copy()
        rec_xy = np.asarray(e["rec"]["global_center"], float)[idx, :2].copy()
        origin = raw_xy[0].copy()
        raw_xy -= origin
        rec_xy -= origin
        raw_xy[~valid] = np.nan
        ax = axs[0]
        ax.plot(raw_xy[:, 0], raw_xy[:, 1], color=RAW, lw=.7, marker="o", markersize=2,
                markerfacecolor="white", markeredgewidth=.5, alpha=.85, zorder=3)
        ax.plot(rec_xy[:, 0], rec_xy[:, 1], color=REC, lw=1.4, zorder=2)
        if (~valid).any():
            ax.scatter(rec_xy[~valid, 0], rec_xy[~valid, 1], s=17, facecolors=GAP,
                       edgecolors=REC, linewidths=.5, zorder=4)
        ax.set(title="Trajectory", xlabel="Relative X (m)", ylabel="Relative Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(.08)
        raw_heading = np.asarray(e["raw"]["heading"], float)[idx].copy()
        rec_heading = np.asarray(e["rec"]["heading"], float)[idx].copy()
        raw_heading[~valid] = np.nan
        # Do not unwrap the plotted heading: retain the recorded discontinuities.
        panels = [(raw_heading, rec_heading, "Heading", "Heading (rad)"),
                  (e["raw_motion"][:, 0], e["rec_motion"][:, 0], "Longitudinal velocity", "Velocity (m/s)"),
                  (e["raw_motion"][:, 1], e["rec_motion"][:, 1], "Lateral velocity", "Velocity (m/s)"),
                  (e["raw_motion"][:, 2], e["rec_motion"][:, 2], "Longitudinal acceleration", r"Acceleration (m/s$^2$)"),
                  (e["raw_motion"][:, 3], e["rec_motion"][:, 3], "Lateral acceleration", r"Acceleration (m/s$^2$)")]
        edges = np.r_[t[0] - (t[1]-t[0])/2, (t[:-1]+t[1:])/2, t[-1]+(t[-1]-t[-2])/2]
        for ax, (raw_y, rec_y, title, ylabel) in zip(axs[1:], panels):
            for missing in contiguous_runs(~valid):
                if len(missing):
                    ax.axvspan(edges[missing[0]], edges[missing[-1]+1], color=GAP, alpha=.6, lw=0, zorder=0)
            ax.plot(t, raw_y, color=RAW, lw=.7, marker="o", markersize=1.7,
                    markerfacecolor="white", markeredgewidth=.4, alpha=.85, zorder=2)
            ax.plot(t, rec_y, color=REC, lw=1.25, zorder=3)
            ax.set(title=title, xlabel="Scenario time (s)", ylabel=ylabel, xlim=(t[0], t[-1]))
            ax.margins(y=.10)
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#727B83")
            ax.grid(True, color="#DFE4E7", lw=.4, zorder=-1)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(length=2.5, width=.55)
    fig.text(.077, .011, f"Waymo Open | Scenario {scenario_id} | Existing avOS reconstruction outputs", fontsize=7, color="#54606A")
    assert len(all_axes) == 18
    stem = out / "waymo_reconstruction_three_examples"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def export_data(paths, examples, out, scenario_id=SCENARIO):
    info = {"dataset": "Waymo Open", "scenario_id": scenario_id,
            "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            "reconstruction": "Existing paired outputs, not recomputed by this script. Original algorithm parameters are not recorded in the input pickle.",
            "selection": "Explicitly selected object IDs from one paired scenario containing all three participant types. Illustrative, not a random or aggregate performance sample.",
            "motion_derivation": "Identical numpy.gradient differentiation of position using actual timestamps; rotation using each series' heading. Raw zero-filled local_velocity/local_acceleration fields are not used. No differentiation across missing frames; velocity needs a 3-sample centered stencil and acceleration a 5-sample centered stencil.",
            "heading": "Stored angles are displayed without unwrapping. Differences by integer multiples of 2*pi represent the same orientation.",
            "examples": []}
    for e in examples:
        idx, t, valid = e["indices"], e["times"], e["valid"]
        meta = {"type": e["label"], "object_id": e["object_id"], "start_s": float(t[0]), "end_s": float(t[-1]), "frames": len(idx), "recorded_valid_frames": int(valid.sum()), "missing_frames": int((~valid).sum()), "missing_frame_indices": idx[~valid].tolist()}
        info["examples"].append(meta)
        with (out / f"{e['label'].lower()}_object_{e['object_id']}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "time_s", "recorded_valid", "recorded_x_m", "recorded_y_m", "recorded_heading_rad", "reconstructed_x_m", "reconstructed_y_m", "reconstructed_heading_rad", "recorded_v_lon_mps", "recorded_v_lat_mps", "recorded_a_lon_mps2", "recorded_a_lat_mps2", "reconstructed_v_lon_mps", "reconstructed_v_lat_mps", "reconstructed_a_lon_mps2", "reconstructed_a_lat_mps2"])
            for j, i in enumerate(idx):
                a, b = e["raw"], e["rec"]
                values = [int(i), float(t[j]), int(valid[j]), *(a["global_center"][i, :2] if valid[j] else [np.nan, np.nan]), a["heading"][i] if valid[j] else np.nan, *b["global_center"][i, :2], b["heading"][i], *e["raw_motion"][j], *e["rec_motion"][j]]
                writer.writerow(["" if isinstance(x, (float, np.floating)) and not np.isfinite(x) else x for x in values])
    (out / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")


def load_stages(stage_roots, examples, scenario_id, raw_path, out):
    """Read genuine intermediate avOS snapshots; never approximate missing stages."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        raw_times = pickle.loads(raw_path.read_bytes())["extra_information"]["timestamp"]
        sources = []
        for step, root in enumerate(stage_roots, 1):
            path = root / f"{scenario_id}.pkl"
            data = pickle.loads(path.read_bytes())
            if data["extra_information"]["id"] != scenario_id or data["extra_information"]["dataset"] != "Waymo Open":
                raise ValueError(f"Stage {step} scenario/dataset mismatch")
            times = np.asarray(data["extra_information"]["timestamp"], float)
            np.testing.assert_array_equal(times, raw_times)
            sources.append({"step": step, "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for e in examples:
                obj = data["object_tracks"][e["object_id"]]
                if obj["type"].upper() != e["label"].upper():
                    raise ValueError(f"Stage {step} object type mismatch")
                state, idx = obj["state"], e["indices"]
                valid = np.asarray(state["valid"], bool)
                heading_finite = np.isfinite(state["heading"][idx])
                # Step 1 fills positions; heading remains unprocessed. Missing
                # recorded headings can legitimately remain NaN until step 2.
                heading_ok = heading_finite.all() if step != 1 else heading_finite[e["valid"]].all()
                if not valid[idx].all() or not np.isfinite(state["global_center"][idx, :2]).all() or not heading_ok:
                    raise ValueError(f"Stage {step} has missing/nonfinite states in the plotted interval")
                mask = np.zeros(len(times), bool)
                mask[idx] = True
                use_saved = step == 3 and data["extra_information"].get("reconstruction_config", {}).get("filter_method") == "sg"
                motion = (np.column_stack((state["local_velocity"],state["local_acceleration"])) if use_saved else derived_motion(state, times, mask))[idx]
                e.setdefault("stages", {})[step] = {"state": state, "motion": motion}
                values = np.column_stack((idx, times[idx], state["global_center"][idx, :2], state["heading"][idx], motion))
                np.savetxt(out / f"{e['label'].lower()}_object_{e['object_id']}_step{step}.csv", values, delimiter=",",
                           header="frame,time_s,x_m,y_m,heading_rad,v_lon_mps,v_lat_mps,a_lon_mps2,a_lat_mps2", comments="")
    return sources


def export_current_stages(examples, raw_path, out, cutoff_freq=None, filter_method="sg", sg_window_length=31, sg_polyorder=2):
    """Run the existing reconstructor with diagnostic capture, never overwrite datasets."""
    sys.path.insert(0, str(ROOT / "avos"))
    from scenario_reconstructor.object_track_reconstruction import ReconstructionConfig, reconstruct_object_state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        source = pickle.loads(raw_path.read_bytes())
    meta = source["extra_information"]
    times = np.asarray(meta["timestamp"], float)
    config = ReconstructionConfig(dataset="waymo-open", cutoff_freq=cutoff_freq, filter_method=filter_method,
                                  sg_window_length=sg_window_length, sg_polyorder=sg_polyorder)
    if filter_method == "sg" and cutoff_freq is not None:
        raise ValueError("--cutoff-freq applies to Butterworth only")
    nyquist = .5 / np.median(np.diff(times))
    if filter_method == "butterworth" and not 0 < config.resolved_cutoff_freq < nyquist:
        raise ValueError(f"Cutoff frequency must be between 0 and the Nyquist frequency ({nyquist:g} Hz)")
    snapshots = {step: {"extra_information": dict(meta), "object_tracks": {},
                        "map_features": {}, "dynamic_map_states": {}} for step in (1, 2, 3)}
    for e in examples:
        result = reconstruct_object_state(e["raw"], times, config, capture_stages=True)
        if not result.reconstructed:
            raise ValueError(f"Cannot reconstruct object {e['object_id']}: {result.skipped_reason}")
        for step in (1, 2, 3):
            snapshots[step]["object_tracks"][e["object_id"]] = {
                "type": e["label"].upper(), "state": result.stage_states[step]}
        e["rec"] = result.state
        valid = np.zeros(len(times), bool)
        valid[e["indices"]] = True
        e["rec_motion"] = (np.column_stack((result.state["local_velocity"],result.state["local_acceleration"])) if filter_method == "sg"
                           else derived_motion(result.state, times, valid))[e["indices"]]
    roots = []
    definitions = config.stage_definitions
    for step, data in snapshots.items():
        data["extra_information"].update(diagnostic_subset=True, stage=step,
            stage_definition=definitions[step], reconstruction_config=asdict(config))
        root = out / "stages" / f"step{step}"
        root.mkdir(parents=True, exist_ok=True)
        (root / raw_path.name).write_bytes(pickle.dumps(data, protocol=4))
        roots.append(root)
    (out / "stage_definitions.json").write_text(json.dumps({
        "implementation": config.description,
        "config": asdict(config), "resolved_cutoff_hz": config.resolved_cutoff_freq if filter_method == "butterworth" else None,
        "sampling": "Inferred from actual timestamps", "stages": definitions}, indent=2) + "\n")
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "dataset/converted-dataset/waymo-open")
    parser.add_argument("--reconstructed-root", type=Path, default=ROOT / "dataset/reconstructed_dataset/waymo-open")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/pdf/waymo_reconstruction_paper_style")
    parser.add_argument("--style", choices=["paper", "modern"], default="paper")
    parser.add_argument("--scenario-id", default=SCENARIO)
    parser.add_argument("--object-ids", nargs=3, metavar=("VEHICLE", "CYCLIST", "PEDESTRIAN"))
    parser.add_argument("--stage1-root", type=Path, help="Directory of real after-step-1 scenario pickle files")
    parser.add_argument("--stage2-root", type=Path)
    parser.add_argument("--stage3-root", type=Path)
    parser.add_argument("--export-current-stages", action="store_true",
                        help="Rerun the selected reconstruction pipeline and export true stage snapshots")
    parser.add_argument("--filter-method", choices=["sg", "butterworth"], default="sg")
    parser.add_argument("--sg-window-length", type=int, default=31)
    parser.add_argument("--sg-polyorder", type=int, default=2)
    parser.add_argument("--cutoff-freq", type=float,
                        help="Butterworth cutoff in Hz for --export-current-stages; lower values smooth more strongly")
    parser.add_argument("--reference-bounds-json", type=Path, help="Optional explicitly specified acceleration reference lines")
    args = parser.parse_args()
    stage_roots = [args.stage1_root, args.stage2_root, args.stage3_root]
    if args.cutoff_freq is not None and not args.export_current_stages:
        parser.error("--cutoff-freq requires --export-current-stages")
    if args.export_current_stages and any(stage_roots):
        parser.error("Do not combine --export-current-stages with external stage roots")
    if any(stage_roots) and not all(stage_roots):
        parser.error("Supply all of --stage1-root, --stage2-root and --stage3-root, or none")
    if args.style == "modern" and (any(stage_roots) or args.reference_bounds_json or args.export_current_stages):
        parser.error("Stage snapshots and reference lines require --style paper")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, examples = load_pair(args.raw_root, args.reconstructed_root, args.scenario_id, args.object_ids)
    if args.export_current_stages:
        stage_roots = export_current_stages(examples, paths[0], args.output_dir, args.cutoff_freq, args.filter_method, args.sg_window_length, args.sg_polyorder)
        paths[1] = stage_roots[2] / paths[0].name
    export_data(paths, examples, args.output_dir, args.scenario_id)
    stage_sources = load_stages(stage_roots, examples, args.scenario_id, paths[0], args.output_dir) if all(stage_roots) else []
    bounds = json.loads(args.reference_bounds_json.read_text()) if args.reference_bounds_json else {}
    if args.style == "paper":
        sys.path.insert(0, str(ROOT / "avos"))
        from scenario_reconstructor.visualization import plot_reconstruction_examples
        fig = plot_reconstruction_examples(examples, args.output_dir / "waymo_reconstruction_paper_style", args.scenario_id, bounds,
                                           current_pipeline_note=(json.loads((args.output_dir / "stage_definitions.json").read_text())["implementation"] if args.export_current_stages else False))
        plt.close(fig)
    else:
        # The legacy plot uses the same numerical curves as before.
        draw(examples, args.output_dir, args.scenario_id)
    provenance_path = args.output_dir / "provenance.json"
    info = json.loads(provenance_path.read_text())
    info.update(style=args.style, stage_sources=stage_sources, reference_bounds=bounds,
                stage_labels_enabled=bool(stage_sources),
                position_frame="Origin at first valid recorded position, rotated by the negative initial recorded heading" if args.style == "paper" else "Translation only")
    if args.style == "paper":
        from scenario_reconstructor.visualization import AXIS_LIMITS
        info["plot_axis_limits"] = dict(AXIS_LIMITS, heading_rad="Per-object movable 2*pi-wide window after display unwrapping")
        info["heading_display"] = "Display copies are unwrapped and aligned across stages. Each object has a movable 2*pi-wide heading window with 1.5-rad tick spacing. Stored angles and motion quantities are unchanged."
        info["panel_labels"] = {chr(97+i): e["label"] for i, e in enumerate(examples)}
    if stage_sources:
        info["reconstruction"] = "Paper curves use the three explicitly supplied stage snapshot files. The script validates their scenario, timestamps and object types but does not infer or rerun their generating algorithms. Paired final-output CSV files are also retained separately."
    if args.export_current_stages:
        info["reconstruction"] = "Rerun of current avOS reconstructor with diagnostic capture. See stage_definitions.json. The selected filter and parameters are recorded below."
        info["reconstructor_source_sha256"] = hashlib.sha256((ROOT / "avos/scenario_reconstructor/object_track_reconstruction.py").read_bytes()).hexdigest()
        settings = json.loads((args.output_dir / "stage_definitions.json").read_text())
        info["filter_settings"] = settings
        caption_path = args.output_dir / "waymo_reconstruction_paper_style_caption.txt"
        with caption_path.open("a") as f:
            f.write("\nFilter settings: " + settings["implementation"] + "\n")
    provenance_path.write_text(json.dumps(info, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
