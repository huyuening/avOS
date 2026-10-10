"""Select 50 tracks per participant type using transparent within-type ranks.

Does not change reconstructed states. Ignores 2*pi heading representation jumps
in scoring. Outputs a compact offline gallery plus exact stage-state subsets.
"""
from pathlib import Path
import argparse, collections, csv, hashlib, json, pickle, shutil, warnings
from plot_waymo_reconstruction_examples import derived_motion
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
METRICS=['missing_fraction','angular_speed_p99_rad_s','acceleration_p99_m_s2','jerk_p99_m_s3']

def scores(rows,entry):
 a=np.array([[np.nan if v is None else v for v in row] for row in rows],float)
 denom=entry['valid_frames']+entry['missing_frames'];result={'missing_fraction':entry['missing_frames']/denom if denom else 0}
 vals={k:[] for k in METRICS[1:]}
 if len(a)>1:
  t=a[:,0];h=a[:,3];dt=np.diff(t);dh=np.diff(h)
  good=np.isfinite(dh)&(dt>0)
  circ=np.arctan2(np.sin(dh[good]),np.cos(dh[good]));vals[METRICS[1]]=np.abs(circ/dt[good])
  # Rotate body-frame acceleration back to the scene frame before jerk derivation.
  al,at=a[:,6],a[:,7];c,s=np.cos(h),np.sin(h)
  acc=np.column_stack([al*c-at*s,al*s+at*c]);mag=np.linalg.norm(acc,axis=1)
  vals[METRICS[2]]=mag[np.isfinite(mag)]
  da=np.diff(acc,axis=0);good=np.isfinite(da).all(axis=1)&(dt>0)
  vals[METRICS[3]]=np.linalg.norm(da[good],axis=1)/dt[good]
 for k in METRICS[1:]:
  values=np.asarray(vals[k]);result[k]=float(np.percentile(values,99)) if len(values) else None
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'output/Waymo-All-Trajectories-20261010');p.add_argument('--output-dir',type=Path,default=ROOT/'output/Waymo-Selected-150-Continuous-20261010');args=p.parse_args()
 src,out=args.source,args.output_dir
 if out.exists():raise ValueError('Use a new output directory')
 (out/'gallery/data').mkdir(parents=True)
 index=json.loads((src/'trajectory_index.json').read_text());by_scene={};ranked=[]
 source_paths={e['scenario']:Path(e['path']) for e in json.loads((src/'summary.json').read_text())['source_files']}
 warnings.filterwarnings('ignore',category=DeprecationWarning)
 for f in sorted((src/'gallery/data').glob('*.js')):
  text=f.read_text();sid,data=json.loads('['+text[len('window.registerScene('):-3]+']')
  by_scene[sid]=data
  source=pickle.loads(source_paths[sid].read_bytes());times=np.asarray(source['extra_information']['timestamp'],float)
  for e in [e for e in index if e['scenario']==sid and e['status']=='reconstructed']:
   state=source['object_tracks'][e['object']]['state'];valid=np.asarray(state['valid'],bool)&np.isfinite(state['heading'])&np.isfinite(state['global_center'][:,:2]).all(axis=1)
   heading=np.asarray(state['heading'],float).copy();heading[~valid]=np.nan
   rows=np.column_stack([times,np.asarray(state['global_center'])[:,:2],heading,derived_motion(state,times,valid)])
   row=dict(e);row.update(scores(rows,e));ranked.append(row)
 selected=[]
 for kind in ['VEHICLE','CYCLIST','PEDESTRIAN']:
  group=[e for e in ranked if e['type']==kind]
  for e in group:e['anomaly_score']=0.;e['available_metrics']=sum(e[k] is not None for k in METRICS)
  for k in METRICS:
   valid=[e for e in group if e[k] is not None]
   values=np.array([e[k] for e in valid]);ranks=(rankdata(values,method='average')-.5)/len(values)
   for e,v,rank in zip(valid,values,ranks):e['anomaly_score']+=float(rank)*25 if v>1e-5 else 0.
  group.sort(key=lambda e:(-e['anomaly_score'],-(e['end_s']-e['start_s']),e['scenario'],e['object']))
  for rank,e in enumerate(group,1):e['anomaly_rank']=rank;e['anomaly_score']=round(e['anomaly_score'],3)
  assert len(group)>=50,(kind,len(group));selected.extend(group[:50])
 for sid in sorted(set(e['scenario'] for e in selected)):
  entries=[e for e in selected if e['scenario']==sid];oids={e['object'] for e in entries};data={e['object']:{**by_scene[sid][e['object']],'meta':e} for e in entries}
  (out/'gallery/data'/f'{sid}.js').write_text('window.registerScene('+json.dumps(sid)+','+json.dumps(data,separators=(',',':'),allow_nan=False)+');\n')
  for sub in ['reconstructed','stages/step1','stages/step2','stages/step3']:
   full=pickle.loads((src/sub/f'{sid}.pkl').read_bytes());full['object_tracks']={oid:o for oid,o in full['object_tracks'].items() if str(oid) in oids};full['extra_information']['diagnostic_subset']=True
   full['extra_information']['selection_note']='50 tracks per type, anomaly-ranking gallery subset; not a complete scenario'
   (out/sub).mkdir(parents=True,exist_ok=True);(out/sub/f'{sid}.pkl').write_bytes(pickle.dumps(full,protocol=4))
 for name,data in [('trajectory_index',selected),('all_ranked_candidates',ranked)]:
  (out/f'{name}.json').write_text(json.dumps(data,indent=2))
  with (out/f'{name}.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (out/'gallery/catalog.js').write_text('window.CATALOG='+json.dumps(selected,separators=(',',':'))+';')
 template=(src/'gallery/index.html').read_text().replace('浏览全部本地 Waymo 轨迹。','按类型保留异常评分最高的 50 条，共 150 条。').replace('全部轨迹</option>','全部入选轨迹</option>').replace('全部 ${CATALOG.length} 条','入选 ${CATALOG.length} 条').replace('所有对象均列入索引，不能重构的对象保留原因。','按缺失比例、周期角速度、加速度及 jerk 的 P99 在各类型内部等权排序。骑行者仅有 50 条，因此全部保留。排名是选图辅助，不是物理有效性判定。').replace('· 朝向大跳变 ${e.heading_jumps} 次','· 周期表示跳变 ${e.heading_jumps} 次${e.anomaly_rank?` · 类型内排名 ${e.anomaly_rank} · 异常评分 ${e.anomaly_score.toFixed(1)}/100`:""}')
 (out/'gallery/index.html').write_text(template)
 summary={'metric_source':'Full-precision original pickle states; no gallery rounding', 'selected':dict(collections.Counter(e['type'] for e in selected)),'candidate_counts':dict(collections.Counter(e['type'] for e in ranked)),'metrics':METRICS,'ranking':'Equal-weight within-type percentile ranks, 25 points each; absent or effectively zero metrics score zero. Only successfully reconstructed tracks are candidates. Ties use longer duration then stable identifiers.','angle_rule':'Angular speed uses the circular difference atan2(sin(delta),cos(delta)); 2*pi representation jumps do not raise the score.','display':'Display stored headings without unwrapping or branch alignment, with full heading range. Motion and heading states unchanged.','note':'Cyclists: all 50 available tracks retained. Anomaly scores are exploratory selection aids, not validated quality labels.'}
 (out/'selection_method.json').write_text(json.dumps(summary,indent=2))
 (out/'README.md').write_text('''# 连续朝向与 150 条候选轨迹

打开 gallery/index.html。车辆、骑行者、行人各 50 条，默认按类型内异常排名排列。筛选按原始观测的缺失比例、角速度 P99、加速度幅值 P99、jerk 幅值 P99 的类型内百分位等权组合。角速度使用周期差分，因此 ±π / 2π 表示跳变不会单独提高异常分数。可用骑行者恰有 50 条，全部保留。评分仅供选图，不是重构方法排名或质量真值。

朝向直接显示各阶段保存值，不做显示展开或分支对齐；默认纵轴完整覆盖这些值，保留数值跳变。没有修改任何重构数值或运动量。

trajectory_index.csv 是 150 条选中轨迹及其异常指标。all_ranked_candidates.csv 保留全部 7,016 条可重构候选的指标以便核查。stages 与 reconstructed 仅保留所选对象，因此是诊断子集，不是完整场景。全部原始及重构轨迹仍保存在原完整目录。
''')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
