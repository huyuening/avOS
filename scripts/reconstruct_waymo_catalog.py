"""Reconstruct every local Waymo object and build an offline selection gallery.

Preserves source data. Exact states are saved as pickle, while the gallery uses
six-decimal display values. Short/invalid tracks remain indexed with a reason.
"""
from __future__ import annotations
import argparse, collections, copy, csv, hashlib, json, pickle, sys, warnings
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'avos'))
from scenario_reconstructor.object_track_reconstruction import ReconstructionConfig, reconstruct_object_state
from plot_waymo_reconstruction_examples import derived_motion


def numbers(a):
    a=np.asarray(a,dtype=float)
    return [[None if not np.isfinite(x) else round(float(x),6) for x in row] for row in a]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=ROOT/'dataset/converted-dataset/waymo-open')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/Waymo-All-Trajectories-20261010')
    p.add_argument('--cutoff-freq',type=float,default=None,help='Butterworth only')
    p.add_argument('--filter-method',choices=['sg','butterworth'],default='sg')
    p.add_argument('--sg-window-length',type=int,default=31)
    p.add_argument('--sg-polyorder',type=int,default=2)
    p.add_argument('--selection-index', type=Path, help='Existing trajectory_index.json to refresh exactly the same objects')
    args=p.parse_args(); out=args.output_dir
    selection = json.loads(args.selection_index.read_text()) if args.selection_index else None
    selected = {(e['scenario'],str(e['object'])): e for e in selection} if selection is not None else None
    if out.exists(): raise ValueError('Use a new output directory to preserve existing results')
    for sub in ['gallery/data','reconstructed','stages/step1','stages/step2','stages/step3']: (out/sub).mkdir(parents=True)
    config=ReconstructionConfig(dataset='waymo-open',cutoff_freq=args.cutoff_freq,filter_method=args.filter_method,sg_window_length=args.sg_window_length,sg_polyorder=args.sg_polyorder)
    if args.cutoff_freq is not None and (args.filter_method=='sg' or args.cutoff_freq<=0):raise ValueError('Positive cutoff applies to Butterworth only')
    index=[]; sources=[]; totals=collections.Counter(); errors=[]
    paths=sorted(args.raw_root.glob('*.pkl'))
    if selected is not None:
        paths=[f for f in paths if any(sid==f.stem for sid,oid in selected)]
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    for scene_number,path in enumerate(paths,1):
        source_bytes=path.read_bytes(); source_hash=hashlib.sha256(source_bytes).hexdigest()
        data=pickle.loads(source_bytes);meta=data['extra_information'];sid=str(meta['id'])
        assert meta['dataset']=='Waymo Open' and path.stem==sid
        times=np.asarray(meta['timestamp'],float)
        assert np.isfinite(times).all() and np.all(np.diff(times)>0)
        if args.filter_method=='butterworth':assert config.resolved_cutoff_freq < .5/np.median(np.diff(times))
        if selected is not None:
            data['object_tracks']={oid:o for oid,o in data['object_tracks'].items() if (sid,str(oid)) in selected}
            data['extra_information']=dict(meta,diagnostic_subset=True)
        rec=copy.deepcopy(data)
        stages={n:dict(extra_information={**meta,'diagnostic_subset':True,'stage':n,'reconstruction_config':asdict(config)},object_tracks={},map_features={},dynamic_map_states={}) for n in (1,2,3)}
        gallery={}
        for oid,obj in data['object_tracks'].items():
            oid=str(oid); kind=obj['type'].upper();raw=obj['state']
            valid=np.asarray(raw['valid'],bool)&np.isfinite(raw['global_center'][:,:2]).all(axis=1)&np.isfinite(raw['heading'])
            recorded=np.flatnonzero(valid)
            status='reconstructed';reason=''
            try:
                result=reconstruct_object_state(raw,times,config,capture_stages=True)
                if not result.reconstructed:status='skipped';reason=result.skipped_reason
            except Exception as exc:
                status='error';reason=f'{type(exc).__name__}: {exc}';result=None;errors.append(dict(scenario=sid,object=oid,reason=reason))
            totals[kind]+=1;totals[status]+=1
            entry=dict(scenario=sid,object=oid,type=kind,status=status,reason=reason,valid_frames=len(recorded),start_s=float(times[recorded[0]]) if len(recorded) else None,end_s=float(times[recorded[-1]]) if len(recorded) else None,missing_frames=0,travel_m=0.,heading_jumps=0)
            if selected is not None:
                entry={**selected[(sid,oid)],**entry}
            if result is not None and result.diagnostics:entry.update(result.diagnostics)
            entry['heading_outlier_samples']=int(result.heading_outlier_mask.sum()) if status=='reconstructed' else 0
            if len(recorded):
                idx=np.arange(recorded[0],recorded[-1]+1);mask=np.zeros(len(times),bool);mask[idx]=True
                entry['missing_frames']=int((~valid[idx]).sum())
                xy=np.asarray(raw['global_center'])[recorded,:2]
                entry['travel_m']=round(float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),3)
                entry['heading_jumps']=int((np.abs(np.diff(np.asarray(raw['heading'])[recorded]))>np.pi).sum())
                origin=xy[0];h=float(raw['heading'][recorded[0]]);rot=np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
                def rows(state,motion_valid,stored_motion=False):
                    local=(np.asarray(state['global_center'])[idx,:2]-origin)@rot
                    motion=(np.column_stack((state["local_velocity"],state["local_acceleration"])) if stored_motion else derived_motion(state,times,motion_valid))[idx]
                    return numbers(np.column_stack([times[idx],local,np.asarray(state['heading'])[idx],motion]))
                rawrows=rows(raw,valid)
                for j,i in enumerate(idx):
                    if not valid[i]: rawrows[j][1:]=[None]*7
                display=dict(raw=rawrows,stages={},meta=entry)
                if status=='reconstructed':
                    rec['object_tracks'][oid]['state']=result.state
                    for n in (1,2,3):
                        stage=result.stage_states[n]
                        stages[n]['object_tracks'][oid]={**obj,'state':stage}
                        display['stages'][str(n)]=rows(stage,mask,stored_motion=(n==3 and config.filter_method=='sg'))
                    for n in (1,2,3):
                        assert np.isfinite(result.stage_states[n]['global_center'][idx]).all()
                    assert np.isfinite(result.state['heading'][idx]).all()
                    assert np.asarray(result.state['valid'])[idx].all()
                gallery[oid]=display
            else:gallery[oid]=dict(raw=[],stages={},meta=entry)
            if status!='reconstructed':
                for n in (1,2,3):stages[n]['object_tracks'][oid]={**obj,'reconstruction_status':status,'reconstruction_reason':reason}
            index.append(entry)
        rec['extra_information']['reconstruction_provenance']={'config':asdict(config),'source_sha256':source_hash,'implementation':config.description}
        (out/'reconstructed'/path.name).write_bytes(pickle.dumps(rec,protocol=4))
        for n in (1,2,3):(out/f'stages/step{n}'/path.name).write_bytes(pickle.dumps(stages[n],protocol=4))
        (out/'gallery/data'/f'{sid}.js').write_text('window.registerScene('+json.dumps(sid)+','+json.dumps(gallery,separators=(',',':'),allow_nan=False)+');\n')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash
        sources.append(dict(scenario=sid,path=str(path.resolve()),sha256=source_hash,objects=len(data['object_tracks'])))
        print(f'{scene_number}/{len(paths)} scenes; {len(index)} tracks; {dict(totals)}',flush=True)
    summary=dict(scenarios=len(paths),tracks=len(index),totals=dict(totals),config=asdict(config),source_files=sources,errors=errors,source_data_unchanged=True,reconstructor_sha256=hashlib.sha256((ROOT/'avos/scenario_reconstructor/object_track_reconstruction.py').read_bytes()).hexdigest(),stages=config.stage_definitions,note=config.description, selection_index=str(args.selection_index) if args.selection_index else None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'trajectory_index.json').write_text(json.dumps(index,indent=2))
    with (out/'trajectory_index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for e in index for k in e)));w.writeheader();w.writerows(index)
    (out/'gallery/catalog.js').write_text('window.CATALOG='+json.dumps(index,separators=(',',':'),allow_nan=False)+';')
    template=(ROOT/'scripts/templates/waymo_catalog.html').read_text().replace('__METHOD_DESCRIPTION__',config.description)
    if selected is not None:
        assert {(e['scenario'],e['object']) for e in index}==set(selected), 'Selection objects missing from inputs'
        index.sort(key=lambda e: (['VEHICLE','CYCLIST','PEDESTRIAN'].index(e['type']),e.get('anomaly_rank',0)))
        (out/'gallery/catalog.js').write_text('window.CATALOG='+json.dumps(index,separators=(',',':'),allow_nan=False)+';')
        template=template.replace('浏览全部本地 Waymo 轨迹。',f'保持原选中的 {len(index)} 条轨迹，已使用恢复朝向校正的新方法重构。').replace('全部轨迹</option>','全部入选轨迹</option>')
    (out/'gallery/index.html').write_text(template)
    (out/'README.md').write_text(f'''# Waymo 全部轨迹重构与选图

共 {len(paths)} 个场景、{len(index)} 条轨迹，全部已尝试重构，状态详见 summary.json。

双击 gallery/index.html 可离线浏览，每条轨迹显示 2×3 六个图。按类型、场景或对象筛选，使用“选为该类型示例”，挑选车辆、骑行者、行人各一个，最后导出选择 JSON。可以直接回复场景编号与对象编号。默认展示全部对象，包括短轨迹与未成功重构对象，避免静默筛选。

- reconstructed/：重构结果，保留地图等源字段。指定 selection-index 时仅含入选对象，并标记 diagnostic_subset，不是完整交通场景。
- stages/step1..3/：所有对象的中间阶段快照（不重复地图）；未重构对象保留原状态及原因。
- trajectory_index.csv/json：完整轨迹索引。
- gallery/data/：六位小数的浏览数据。pickle 保留计算精度。

当前方法：{config.description}。阶段定义见 summary.json。SG 版 Step 2 基于插值位置的速度方向修正朝向，Step 3 基于 SG 位置重新修正朝向、平滑朝向和速度，并再积分位置。SG 最终运动图使用保存的滤波速度/加速度。源数据及旧结果未覆盖。低速对象的速度方向可能不可靠，原 SG 示例筛选条件没有用于删除本次候选。

显示时间轴 0–20 s，短轨迹仅显示已有区间。朝向仅在显示时连续展开，每条轨迹使用可移动的 2π 宽窗口，数字刻度间隔 1.5 rad。固定工程显示范围会裁切超范围曲线，但不更改数据。“查看完整范围”仅供选图检查。
''')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_files','errors')},indent=2))

if __name__=='__main__':main()
