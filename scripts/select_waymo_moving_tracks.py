"""Select moving examples using the legacy SG script's speed/frame criteria.

Ranks are inherited from a full-precision raw-data anomaly index. Selection is
independent of reconstruction results and never pads a type with ineligible data.
The legacy ego-distance plotting gate is not used for this motion-only selection.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'avos'))
from scenario_reconstructor.object_track_reconstruction import _interpolate_center

SPEED_LIMITS = {'VEHICLE': (5., 1., 40.), 'CYCLIST': (5., 1., 40.),
                'PEDESTRIAN': (.5, .1, 10.)}


def motion_eligibility(state, timestamp, kind):
    """Speed magnitude equals legacy Step 2 local-speed magnitude by rotation."""
    center = np.asarray(state['global_center'], float)
    valid = (np.asarray(state['valid'], bool) & np.isfinite(center[:, :2]).all(axis=1)
             & np.isfinite(state['heading']))
    observed = np.flatnonzero(valid)
    result = {'sg_observed_frames': len(observed), 'sg_eligible': False,
              'sg_selection_reason': ''}
    if len(observed) < 40:
        result['sg_selection_reason'] = 'fewer_than_40_observations'
        return result
    idx = np.arange(observed[0], observed[-1]+1)
    filled = _interpolate_center(center, observed, idx)
    speed = np.linalg.norm(np.gradient(filled[:, :2], timestamp[idx], axis=0, edge_order=2), axis=1)
    mean, minimum, maximum = float(speed.mean()), float(speed.min()), float(speed.max())
    result.update(sg_mean_speed_mps=mean, sg_min_speed_mps=minimum, sg_max_speed_mps=maximum)
    mean_limit, min_limit, max_limit = SPEED_LIMITS[kind]
    failures = []
    if mean < mean_limit: failures.append('mean_speed_below_threshold')
    if minimum < min_limit: failures.append('minimum_speed_below_threshold')
    if maximum > max_limit: failures.append('maximum_speed_above_threshold')
    result['sg_eligible'] = not failures
    result['sg_selection_reason'] = ','.join(failures) or 'eligible'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=ROOT/'dataset/converted-dataset/waymo-open')
    parser.add_argument('--ranked-index', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--per-type', type=int, default=50)
    args = parser.parse_args()
    if args.per_type <= 0: parser.error('--per-type must be positive')
    if args.output_dir.exists(): raise ValueError('Use a new output directory')
    ranked = json.loads(args.ranked_index.read_text())
    assessed, sources = [], []
    for sid in sorted({e['scenario'] for e in ranked}):
        path = args.raw_root/(sid+'.pkl')
        source = path.read_bytes()
        data = pickle.loads(source)
        times = np.asarray(data['extra_information']['timestamp'], float)
        assert np.isfinite(times).all() and np.all(np.diff(times)>0)
        for entry in (e for e in ranked if e['scenario']==sid):
            state = data['object_tracks'][entry['object']]['state']
            assessed.append({**entry, **motion_eligibility(state, times, entry['type'])})
        sources.append({'scenario':sid, 'sha256':hashlib.sha256(source).hexdigest()})
    selected, eligible_counts = [], {}
    for kind in SPEED_LIMITS:
        eligible = [e for e in assessed if e['type']==kind and e['sg_eligible']]
        eligible.sort(key=lambda e:e['anomaly_rank'])
        eligible_counts[kind] = len(eligible)
        for rank, entry in enumerate(eligible[:args.per_type], 1):
            selected.append({**entry, 'moving_selection_rank':rank})
    args.output_dir.mkdir(parents=True)
    for name, entries in [('trajectory_index',selected), ('eligibility_audit',assessed)]:
        (args.output_dir/(name+'.json')).write_text(json.dumps(entries,indent=2))
        with (args.output_dir/(name+'.csv')).open('w',newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for e in entries for k in e)))
            writer.writeheader();writer.writerows(entries)
    summary = dict(assessed=len(assessed), eligible_counts=eligible_counts,
                   selected_counts=dict(collections.Counter(e['type'] for e in selected)),
                   requested_per_type=args.per_type, minimum_observations=40,
                   speed_limits=dict(zip(SPEED_LIMITS,[dict(mean_at_least=v[0],min_at_least=v[1],max_at_most=v[2]) for v in SPEED_LIMITS.values()])),
                   source_files=sources, ranked_index_sha256=hashlib.sha256(args.ranked_index.read_bytes()).hexdigest(),
                   policy='Original SG speed and 40-observation criteria; no ego-distance gate; retain highest original anomaly ranks, up to requested count per type. No reconstruction-output filtering.')
    (args.output_dir/'selection_method.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='source_files'},indent=2))

if __name__=='__main__': main()
