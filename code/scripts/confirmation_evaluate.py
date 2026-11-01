"""Thun-only calibration, then immutable new-recording candidate-admission audit."""
import argparse,json,sys
from collections import defaultdict
import numpy as np
from confirmation_common import ROOT,OUT,CONFIG,SCALES,read,save,sha,freeze,variants,sequences
sys.path.insert(0,str(ROOT/'vendor/spikefpn_snapshot'))
from utils.tools import nms
from riskaware.criterion_audit import filter_anchors,native_detections,ground_truth
from riskaware.metrics import match_frame,Det
from riskaware.calibration_audit import threshold_at_fppi
from riskaware.evidence_runs import evidence_runs
def meta(model,seq):return read(OUT/(model+'__'+seq+'.json'))
def load(record):
    p=ROOT/record['path']
    if sha(p)!=record['sha256']:raise ValueError('Inference cache changed')
    with np.load(p) as f:return {k:f[k] for k in ['boxes','scores','rank_scores']}
def masks(data,m,cfg):
    n=[g['height']*g['width'] for g in m['geometry']];bounds=np.cumsum([0]+n);result={'dense':np.ones(sum(n),bool)}
    for i,name in enumerate(m['variants']):
        for cap in cfg['capacities']:
            selected=np.zeros(sum(n),bool)
            for j,size in enumerate(n):
                score=data['rank_scores'][i,bounds[j]:bounds[j+1]];order=np.lexsort((np.arange(size),-score));selected[bounds[j]+order[:max(1,int(np.floor(size*cap)))]]=True
            result[name+'@'+str(cap)]=selected
    return result
def detect(data,mask,theta,m,cfg,gt):
    f=filter_anchors(data['boxes'],data['scores'],mask,m['model_provenance']['anchors'],theta,cfg['nms'],nms)
    if m['model_provenance']['coordinates']=='spike256_normalized':dets=native_detections(f)
    else:dets=[Det(tuple(map(float,b)),float(s),int(c)) for b,s,c in zip(f[0],f[1],f[2])]
    return dets,match_frame(dets,gt,cfg['iou'],theta)
def truth(seq,cfg):
    p=ROOT/'datasets/dsec_pilot/train'/seq;grid=np.loadtxt(p/'image_timestamps.txt',dtype=np.int64);labels=np.load(p/'object_detections/left/tracks.npy')
    return grid,{int(t):ground_truth(labels[labels['t']==t],cfg['class_map'],cfg['small_area_fraction'])[0] for t in np.unique(labels['t'])}
def evaluation_sources():
    return ['code/scripts/confirmation_evaluate.py','code/scripts/confirmation_common.py','code/scripts/cache_confirmation_spike.py','code/scripts/cache_confirmation_rvt.py',
            'code/configs/confirmation_v01.json','code/configs/rvt_reference_v01.json','code/configs/diverse_adaptation_v01.json',
            'code/scripts/run_rvt_reference.py','code/scripts/run_diverse_adaptation.py','code/riskaware/criterion_audit.py','code/riskaware/metrics.py','code/riskaware/calibration_audit.py','code/riskaware/evidence_runs.py']
def calibrate(cfg):
    if (OUT/'frozen_thresholds.json').exists():raise FileExistsError('Calibration already frozen')
    # No new-recording predictions are consumed in this phase.
    results={};seq=cfg['calibration_sequence'];_,gts=truth(seq,cfg)
    old=read(ROOT/'experiments/adapted_criterion_v01/frozen_thresholds.json')
    for model in cfg['models']:
        m=meta(model,seq)
        if not all(r['all_open_exact'] and r.get('explicit_state_replay_exact',True) for r in m['alignment']):raise AssertionError('Preflight not passed')
        if model.startswith('spike_') and m['legacy_exact_queries']!=len(m['frames']):raise AssertionError('Thun protected regression incomplete')
        scores=defaultdict(list);loaded=[]
        for i,r in enumerate(m['frames']):
            d=load(r);selected=masks(d,m,cfg);gt=gts.get(r['time_us'],[]);loaded.append((d,selected,gt))
            for tag,mask in selected.items():
                det,match=detect(d,mask,cfg['confidence_floor'],m,cfg,gt);used=set(match.values());scores[tag].extend(x.score for j,x in enumerate(det) if j not in used)
            if (i+1)%50==0:print('Calibration candidates',model,i+1,flush=True)
        th={tag:threshold_at_fppi(v,len(loaded),cfg['target_calibration_fppi'],cfg['confidence_floor']) for tag,v in scores.items()};actual={tag:dict(tp=0,fp=0,gt=0) for tag in th}
        for d,selected,gt in loaded:
            for tag,mask in selected.items():
                det,match=detect(d,mask,th[tag],m,cfg,gt);v=actual[tag];v['tp']+=len(match);v['fp']+=len(det)-len(match);v['gt']+=len(gt)
        if any(v['fp']>len(loaded)*cfg['target_calibration_fppi'] for v in actual.values()):raise AssertionError('Actual Thun FP budget guard failed')
        if model.startswith('spike_'):
            key='original' if model=='spike_original' else 'full_conv'
            if th['dense']!=old[key]['dense_threshold']:raise AssertionError('Protected dense threshold changed')
        results[model]=dict(thresholds=th,dense_threshold=th['dense'],queries=len(loaded),actual=actual,cache_receipt_sha256=sha(OUT/(model+'__'+seq+'.json')))
        print('Thun calibration passed',model,'dense',th['dense'],actual['dense'],flush=True)
    save(OUT/'frozen_thresholds.json',results)
    save(OUT/'evaluation_freeze.json',dict(config_sha256=sha(CONFIG),sources={p:sha(ROOT/p) for p in evaluation_sources()},thresholds_sha256=sha(OUT/'frozen_thresholds.json'),cohort_thun_sha256=sha(OUT/'cohorts'/f'{seq}.json'),frozen_before_new_recording_inference=True))
def evaluate(cfg):
    if (OUT/'summary.json').exists():raise FileExistsError('Completed confirmation results cannot be overwritten')
    frozen=read(OUT/'evaluation_freeze.json')
    if sha(CONFIG)!=frozen['config_sha256'] or sha(OUT/'frozen_thresholds.json')!=frozen['thresholds_sha256']:raise ValueError('Frozen protocol changed')
    for p,h in frozen['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen evaluation source changed '+p)
    th=read(OUT/'frozen_thresholds.json');all_result={};all_runs={}
    with (OUT/'frames.jsonl').open('w') as handle,(OUT/'loss_runs.jsonl').open('w') as loss:
        for seq in cfg['confirmation_sequences']:
            grid,gts=truth(seq,cfg);idx={int(t):i for i,t in enumerate(grid)};median=float(np.median(np.diff(grid)));mm={model:meta(model,seq) for model in cfg['models']}
            times=[r['time_us'] for r in mm[cfg['models'][0]]['frames']]
            if any(times!=[r['time_us'] for r in m['frames']] for m in mm.values()):raise ValueError('Cross-model query population differs')
            totals=defaultdict(lambda:defaultdict(int));histories=defaultdict(list);tracks=set()
            for i,t in enumerate(times):
                gt=gts.get(t,[]);tracks.update(g.track_id for g in gt);small={g.track_id for g in gt if g.key};data={model:load(m['frames'][i]) for model,m in mm.items()};selected={model:masks(data[model],m,cfg) for model,m in mm.items()};dense={};memo={}
                def outcome(model,mask,theta):
                    key=(model,mask.tobytes(),theta)
                    if key not in memo:
                        det,matched=detect(data[model],mask,theta,mm[model],cfg,gt);memo[key]=({gt[j].track_id for j in matched},len(det)-len(matched))
                    return memo[key]
                for model in cfg['models']:dense[model]=outcome(model,selected[model]['dense'],th[model]['dense_threshold'])[0]
                shared=set.intersection(*dense.values())
                for model in cfg['models']:
                    ref=dense[model]
                    for regime in cfg['regimes']:
                        for tag,mask in selected[model].items():
                            theta=th[model]['dense_threshold'] if regime=='dense_thun_threshold' else th[model]['thresholds'][tag];ids,fp=outcome(model,mask,theta)
                            r=dict(sequence=seq,time_us=t,grid_index=idx[t],model=model,regime=regime,condition=tag,threshold=theta,gt_ids=sorted(g.track_id for g in gt),dense_ids=sorted(ref),matched_ids=sorted(ids),shared_dense_ids=sorted(shared),
                                   gt=len(gt),tp=len(ids),fp=fp,dense_tp=len(ref),kept=len(ref&ids),lost=len(ref-ids),gained=len(ids-ref),small_gt=len(small),small_tp=len(small&ids),small_dense_tp=len(small&ref),small_kept=len(small&ref&ids),shared_dense_tp=len(shared),shared_kept=len(shared&ids))
                            handle.write(json.dumps(r,separators=(',',':'))+'\n');key=(model,regime,tag);histories[key].append(r);totals[key]['queries']+=1
                            for k in ['gt','tp','fp','dense_tp','kept','lost','gained','small_gt','small_tp','small_dense_tp','small_kept','shared_dense_tp','shared_kept']:totals[key][k]+=r[k]
                if (i+1)%100==0:print('Confirmation evaluate',seq,i+1,'/',len(times),flush=True)
            all_result[seq]={};all_runs[seq]={}
            for (model,regime,tag),v in totals.items():
                v.update(preservation=v['kept']/v['dense_tp'] if v['dense_tp'] else None,shared_preservation=v['shared_kept']/v['shared_dense_tp'] if v['shared_dense_tp'] else None,
                         recall=v['tp']/v['gt'] if v['gt'] else None,fppi=v['fp']/v['queries'],precision=v['tp']/(v['tp']+v['fp']) if v['tp']+v['fp'] else None,
                         small_preservation=v['small_kept']/v['small_dense_tp'] if v['small_dense_tp'] else None,small_recall=v['small_tp']/v['small_gt'] if v['small_gt'] else None,gt_tracks=len(tracks))
                tr=defaultdict(lambda:[0,0])
                for r in histories[(model,regime,tag)]:
                    for identity in r['dense_ids']:tr[identity][0]+=1;tr[identity][1]+=identity in r['matched_ids']
                v.update(dense_tracks=len(tr),track_equal_preservation=float(np.mean([b/a for a,b in tr.values()])) if tr else None)
                all_result[seq].setdefault(model,{}).setdefault(regime,{})[tag]=dict(v)
                runs,stats=evidence_runs(histories[(model,regime,tag)],median,cfg['gap_factor']);all_runs[seq].setdefault(model,{}).setdefault(regime,{})[tag]=stats
                for r in runs:loss.write(json.dumps(dict(sequence=seq,model=model,regime=regime,condition=tag,**r))+'\n')
            save(OUT/'evaluation_status.json',dict(state='running',completed_sequence=seq));print('Completed confirmation sequence',seq,flush=True)
    save(OUT/'summary.json',dict(config=cfg,conditions=all_result,population_role='four named previously unused recording groups; same dataset/city, not official test or broad generalization'))
    save(OUT/'continuity_summary.json',all_runs);save(OUT/'evaluation_status.json',dict(state='complete',verification_pending=True))
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibrate','evaluate'],required=True);args=p.parse_args();cfg=freeze()
    if args.phase=='calibrate':calibrate(cfg)
    else:evaluate(cfg)
if __name__=='__main__':main()
