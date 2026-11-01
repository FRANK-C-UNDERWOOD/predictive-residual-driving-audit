"""Post-exposure fixed-carrier mechanism and actually replayed FP-budget analysis."""
from pathlib import Path
from collections import defaultdict
import argparse,json,sys,time,os
import numpy as np
from scipy.stats import rankdata
from confirmation_common import ROOT,read,save,sha
from confirmation_evaluate import truth
from riskaware.criterion_audit import filter_anchors,native_detections
from riskaware.metrics import Det,match_frame
from riskaware.calibration_audit import threshold_at_fppi
from utils.tools import nms

OUT=ROOT/'experiments/fp_mechanism_v01';PARENT=ROOT/'experiments/confirmation_v01';CONFIG=ROOT/'code/configs/fp_mechanism_v01.json'
DT=np.dtype([('q','<i4'),('anchor','<i4'),('score','<f4'),('identity','<i8')])
def metadata(model,seq):return read(PARENT/(model+'__'+seq+'.json'))
def sources():return ['code/scripts/run_fp_mechanism.py','code/scripts/run_final_analyses.py','code/configs/fp_mechanism_v01.json','docs/31_同误报与反转机制分析协议.md','code/scripts/confirmation_common.py','code/scripts/confirmation_evaluate.py','code/riskaware/criterion_audit.py','code/riskaware/metrics.py','code/riskaware/calibration_audit.py','vendor/spikefpn_snapshot/utils/tools.py']
def frozen():
    cfg=read(CONFIG);OUT.mkdir(exist_ok=True)
    manifest=dict(config=cfg,sources={p:sha(ROOT/p) for p in sources()},parents={p:sha(PARENT/p) for p in ['artifact_manifest.json','summary.json','continuity_summary.json','run_manifest.json','evaluation_freeze.json','frozen_thresholds.json']},paper_before_sha256=sha(ROOT/'paper/main.tex'))
    p=OUT/'freeze.json'
    if p.exists() and read(p)!=manifest:raise ValueError('Frozen secondary-analysis inputs changed')
    save(p,manifest);return cfg
def canonical(data,mask,theta,m,gt):
    f=filter_anchors(data['boxes'],data['scores'],mask,m['model_provenance']['anchors'],theta,.5,nms)
    dets=native_detections(f) if m['model_provenance']['coordinates']=='spike256_normalized' else [Det(tuple(map(float,b)),float(s),int(c)) for b,s,c in zip(f[0],f[1],f[2])]
    matches=match_frame(dets,gt,.5,theta);ids=np.full(len(dets),-1,np.int64)
    for gi,di in matches.items():ids[di]=gt[gi].track_id
    return f,matches,ids
def ranked(values,sizes):
    mask=np.zeros(sum(sizes),bool);start=0
    for size in sizes:
        order=np.lexsort((np.arange(size),-values[start:start+size]));mask[start+order[:max(1,int(np.floor(.2*size)))]]=True;start+=size
    return mask
def corr(x,y):
    a=rankdata(x);b=rankdata(y)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def coordinates(m):
    result=[];strides=[]
    for g in m['geometry']:
        y,x=np.indices((g['height'],g['width']));xy=np.column_stack(((x.ravel()+.5)*g['stride'],(y.ravel()+.5)*g['stride']))
        if m['model_provenance']['coordinates']=='spike256_normalized':xy=(xy-np.array([0,32]))/.4
        result.append(xy);strides.extend([g['stride']]*len(xy))
    return np.concatenate(result),np.array(strides)
def load_raw(r,features=False):
    p=ROOT/r['path']
    if features and sha(p)!=r['sha256']:raise ValueError('Protected raw cache changed')
    with np.load(p) as z:return {k:z[k] for k in (['boxes','scores','rank_scores','h8','h16','h32'] if features else ['boxes','scores'])}
def extract(cfg):
    parent_cfg=read(ROOT/'code/configs/confirmation_v01.json');old=read(PARENT/'summary.json')['conditions'];theta=read(PARENT/'frozen_thresholds.json');all_stats={}
    for model in cfg['models']:
        for seq in [cfg['calibration_sequence']]+cfg['sequences']:
            dest=OUT/(model+'__'+seq+'.json')
            if dest.exists():continue
            m=metadata(model,seq);_,gts=truth(seq,parent_cfg);sizes=[g['height']*g['width'] for g in m['geometry']];xy,strides=coordinates(m);pad=(xy[:,0]<0)|(xy[:,0]>=640)|(xy[:,1]<0)|(xy[:,1]>=480);n=sum(sizes)
            prior=None;chunks={tag:[] for tag in cfg['methods']};packed=[];refs=[];totals={tag:defaultdict(int) for tag in cfg['methods']};stats=defaultdict(float);count=0;start=time.monotonic();folder=OUT/'floor'/model/seq;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'common_frames.jsonl').open('w') as frames,(folder/'dense_support.jsonl').open('w') as support,(folder/'query_mechanism.jsonl').open('w') as query:
                for ai,r in enumerate(m['all_frames']):
                    d=load_raw(r,True);h={s:d['h'+str(s)] for s in [8,16,32]};p=h if prior is None else prior
                    if not r['warmup']:
                        qi=count;t=r['time_us'];gt=gts.get(t,[]);av=[];pv=[];nv=[];ov=[];uv=[];opposition=[];negative=elements=0
                        for s,x in h.items():
                            pred=p[s];a=np.abs(x);b=np.abs(pred);delta=np.abs(x-pred);u=np.abs(a-b);same=(np.signbit(x)==np.signbit(pred));over=np.where(same,np.minimum(a,b),0)
                            av.append(a.mean(1));pv.append(b.mean(1));nv.append(delta.mean(1));ov.append(over.mean(1));uv.append(u.mean(1));opposition.append(np.where(~same,np.minimum(a,b),0).mean(1));negative+=int((x<0).sum());elements+=x.size
                            # Float64 identity on original float32 observed inputs; never recover ranks from a rounded mean.
                            xx=x.astype(np.float64);pp=pred.astype(np.float64);nn=np.abs(xx-pp).sum(1);aa=np.abs(xx).sum(1);bb=np.abs(pp).sum(1);oo=np.where(same,np.minimum(np.abs(xx),np.abs(pp)),0).sum(1);err=np.abs(nn-(aa+bb-2*oo));bound=32*np.finfo(np.float64).eps*x.shape[1]*np.maximum(aa+bb,1)
                            if np.any(err>bound):raise AssertionError('Signed-overlap identity failed')
                            stats['max_identity_absolute_error']=max(stats['max_identity_absolute_error'],float(err.max()))
                        A,B,N,O,U,S=[np.concatenate(v) for v in [av,pv,nv,ov,uv,opposition]]
                        if not np.array_equal(N,d['rank_scores'][m['variants'].index('ema/absolute')]) or not np.array_equal(A,d['rank_scores'][m['variants'].index('current_activity')]):raise AssertionError('Canonical observed-predictor reconstruction failed')
                        masks={tag:np.ones(n,bool) if tag=='dense' else ranked(U if tag=='unsigned_residual' else d['rank_scores'][m['variants'].index(tag)],sizes) for tag in cfg['methods']};packed.append(np.packbits(np.stack(list(masks.values())),axis=1))
                        common={tag:canonical(d,mask,theta[model]['dense_threshold'],m,gt) for tag,mask in masks.items()};dense_f,dense_match,dense_labels=common['dense'];reference=set(map(int,dense_labels[dense_labels>=0]));refs.extend((qi,identity) for identity in sorted(reference));dense_anchors=dense_f[3][dense_labels>=0];dense_sites=dense_anchors//m['model_provenance']['anchors']
                        for tag,mask in masks.items():
                            f,match,labels=canonical(d,mask,cfg['confidence_floor'],m,gt);z=np.empty(len(labels),DT);z['q']=qi;z['anchor']=f[3];z['score']=f[1];z['identity']=labels;chunks[tag].append(z)
                            ff,mm,ll=common[tag];ids=set(map(int,ll[ll>=0]));v=totals[tag];v['queries']+=1;v['gt']+=len(gt);v['tp']+=len(mm);v['fp']+=len(ll)-len(mm);v['dense_tp']+=len(reference);v['kept']+=len(ids&reference);v['gained']+=len(ids-reference);v['source_covered']+=int(mask[dense_sites].sum());v['selected_slots']+=int(mask.sum());v['selected_padding']+=int((mask&pad).sum())
                            frames.write(json.dumps(dict(q=qi,time_us=t,method=tag,matched_ids=sorted(ids),dense_ids=sorted(reference),tp=len(mm),fp=len(ll)-len(mm),gt=len(gt),source_covered=int(mask[dense_sites].sum())),separators=(',',':'))+'\n')
                        act=masks['current_activity'];res=masks['ema/absolute'];head=d['scores'].reshape(n,m['model_provenance']['anchors'],-1).max((1,2));inside=np.zeros(n,bool)
                        for g in gt:inside|=(xy[:,0]>=g.box[0])&(xy[:,0]<=g.box[2])&(xy[:,1]>=g.box[1])&(xy[:,1]<=g.box[3])&~pad
                        qstats=dict(q=qi,time_us=t,negative_fraction=negative/elements,activity_head_spearman=corr(A,head),residual_head_spearman=corr(N,head),activity_residual_spearman=corr(A,N),activity_gt_center_slots=int((act&inside).sum()),residual_gt_center_slots=int((res&inside).sum()),activity_padding_slots=int((act&pad).sum()),residual_padding_slots=int((res&pad).sum()),selected_slots=int(act.sum()),dense_support=len(reference),same_mask=bool(np.array_equal(act,res)),unsigned_mask_same=bool(np.array_equal(masks['unsigned_residual'],res)),opposition_mean=float(S.mean()),residual_mean=float(N.mean()))
                        query.write(json.dumps(qstats,separators=(',',':'))+'\n')
                        for gi,di in dense_match.items():
                            g=gt[gi];anchor=int(dense_f[3][di]);site=anchor//m['model_provenance']['anchors'];stats['dense_support']+=1;stats['support_both' if act[site] and res[site] else 'support_residual_only' if res[site] else 'support_activity_only' if act[site] else 'support_neither']+=1
                            support.write(json.dumps(dict(q=qi,time_us=t,identity=g.track_id,anchor=anchor,site=site,stride=int(strides[site]),small=g.key,confidence=float(dense_f[1][di]),activity=float(A[site]),past_activity=float(B[site]),residual=float(N[site]),overlap=float(O[site]),unsigned_residual=float(U[site]),opposition=float(S[site]),temporal_adjustment=float(N[site]-A[site]),activity_selected=bool(act[site]),residual_selected=bool(res[site]),activity_detected=g.track_id in set(map(int,common['current_activity'][2])),residual_detected=g.track_id in set(map(int,common['ema/absolute'][2]))),separators=(',',':'))+'\n')
                        count+=1
                        if count%200==0:save(OUT/'status.json',dict(state='running',phase='extract',model=model,sequence=seq,queries=count));print('FP/mechanism extract',model,seq,count,round(time.monotonic()-start,1),'s',flush=True)
                    prior={s:cfg['alpha']*p[s]+(1-cfg['alpha'])*x for s,x in h.items()}
            if count!=len(m['frames']):raise AssertionError('Full cohort not retained')
            for tag,items in chunks.items():
                z=np.concatenate(items) if items else np.empty(0,DT);np.savez_compressed(folder/(tag.replace('/','_')+'.npz'),detections=z)
            np.savez_compressed(folder/'masks.npz',masks=np.stack(packed),reference=np.array(refs,dtype=np.int64).reshape(-1,2))
            regressions=0
            if seq in old:
                for tag,v in totals.items():
                    if tag=='unsigned_residual':continue
                    expected=old[seq][model]['dense_thun_threshold']['dense' if tag=='dense' else tag+'@0.2']
                    if any(v[k]!=expected[k] for k in ['queries','gt','tp','fp','dense_tp','kept','gained']):raise AssertionError('Protected common-threshold regression '+model+seq+tag)
                    regressions+=1
            save(dest,dict(model=model,sequence=seq,queries=count,common_totals={k:dict(v) for k,v in totals.items()},mechanism_totals=dict(stats),protected_common_regressions=regressions,parent_metadata_sha256=sha(PARENT/(model+'__'+seq+'.json'))));print('Completed extract',model,seq,flush=True)
    save(OUT/'extraction_complete.json',dict(passed=True))
def floor_path(model,seq,tag):return OUT/'floor'/model/seq/(tag.replace('/','_')+'.npz')
def floor_fps(model,seq,tag):
    with np.load(floor_path(model,seq,tag)) as z:d=z['detections'];return d['score'][d['identity']<0].astype(np.float64)
def replay_points(cfg,model,seq,plans):
    m=metadata(model,seq);pc=read(ROOT/'code/configs/confirmation_v01.json');_,gts=truth(seq,pc);folder=OUT/'floor'/model/seq
    with np.load(folder/'masks.npz') as f:packed=f['masks'];rr=f['reference']
    refs=defaultdict(set)
    for q,i in rr:refs[int(q)].add(int(i))
    n=sum(g['height']*g['width'] for g in m['geometry']);result={};pending=plans;attempt=0
    trace=OUT/'points'/model/(seq+'.jsonl');trace.parent.mkdir(parents=True,exist_ok=True)
    with trace.open('w') as handle:
        while pending:
            totals={p['key']:defaultdict(int) for p in pending};fp_scores=defaultdict(list)
            for qi,r in enumerate(m['frames']):
                data=load_raw(r);gt=gts.get(r['time_us'],[]);ms=np.unpackbits(packed[qi],axis=1)[:,:n].astype(bool);memo={}
                for p in pending:
                    mi=cfg['methods'].index(p['method']);memo_key=(mi,p['threshold'])
                    if memo_key not in memo:memo[memo_key]=canonical(data,ms[mi],p['threshold'],m,gt)
                    f,match,labels=memo[memo_key];ids=set(map(int,labels[labels>=0]));v=totals[p['key']];v['queries']+=1;v['gt']+=len(gt);v['tp']+=len(match);v['fp']+=len(labels)-len(match);v['dense_tp']+=len(refs[qi]);v['kept']+=len(ids&refs[qi]);v['gained']+=len(ids-refs[qi]);fp_scores[p['key']].extend(float(x) for x in f[1][labels<0])
                    handle.write(json.dumps(dict(key=p['key'],attempt=attempt,q=qi,time_us=r['time_us'],threshold=p['threshold'],tp=len(match),fp=len(labels)-len(match),gt=len(gt),dense_ids=sorted(refs[qi]),matched_ids=sorted(ids)),separators=(',',':'))+'\n')
                if (qi+1)%200==0:save(OUT/'status.json',dict(state='running',phase='points',model=model,sequence=seq,queries=qi+1,attempt=attempt));print('Actual threshold replay',model,seq,qi+1,'conditions',len(pending),'attempt',attempt,flush=True)
            retry=[]
            for p in pending:
                v=dict(totals[p['key']]);guard=p['regime']!='thun_calibrated'
                if guard and v['fp']>p['allowed_fp']:
                    theta=threshold_at_fppi(fp_scores[p['key']],v['queries'],p['target'],cfg['confidence_floor']);p=dict(p,threshold=max(theta,float(np.nextafter(np.float32(p['threshold']),np.float32(np.inf)))));retry.append(p);continue
                v.update(recall=v['tp']/v['gt'] if v['gt'] else None,preservation=v['kept']/v['dense_tp'] if v['dense_tp'] else None,fppi=v['fp']/v['queries'],precision=v['tp']/(v['tp']+v['fp']) if v['tp']+v['fp'] else None)
                result[p['key']]=dict(p,**v,attempt=attempt,under_budget=v['fp']<=p['allowed_fp'],target_shortfall=p['allowed_fp']-v['fp'],floor_limited=p['threshold']==cfg['confidence_floor'] and v['fp']<p['allowed_fp'])
            pending=retry;attempt+=1
            if attempt>32:raise RuntimeError('Actual NMS FP guard did not converge')
    return result
def points(cfg):
    thun=cfg['calibration_sequence'];calfile=OUT/'calibration_points.json'
    if calfile.exists():cal=read(calfile)
    else:
        cal={}
        for model in cfg['models']:
            count=len(metadata(model,thun)['frames']);plans=[]
            for method in cfg['methods']:
                fps=floor_fps(model,thun,method)
                for target in cfg['fp_targets']:
                    key=method+'@'+str(target);plans.append(dict(key=key,method=method,target=target,regime='calibration',allowed_fp=int(np.floor(target*count)),threshold=threshold_at_fppi(fps,count,target,cfg['confidence_floor'])))
            cal[model]=replay_points(cfg,model,thun,plans)
        save(calfile,cal);save(OUT/'calibration_freeze.json',dict(calibration_points_sha256=sha(calfile),sources={p:sha(ROOT/p) for p in sources()},role='Thun-only threshold selection, secondary exposed-data analysis; not a new blind confirmation'))
    for model in cfg['models']:
        for seq in cfg['sequences']:
            dest=OUT/'points'/(model+'__'+seq+'.json')
            if dest.exists():continue
            count=len(metadata(model,seq)['frames']);plans=[]
            for method in cfg['methods']:
                fps=floor_fps(model,seq,method)
                for target in cfg['fp_targets']:
                    for regime in cfg['regimes']:
                        key=regime+'|'+method+'@'+str(target);theta=cal[model][method+'@'+str(target)]['threshold'] if regime=='thun_calibrated' else threshold_at_fppi(fps,count,target,cfg['confidence_floor']);plans.append(dict(key=key,method=method,target=target,regime=regime,allowed_fp=int(np.floor(target*count)),threshold=theta))
            save(dest,dict(model=model,sequence=seq,conditions=replay_points(cfg,model,seq,plans)));print('Completed actual points',model,seq,flush=True)
    save(OUT/'points_complete.json',dict(passed=True))
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['extract','points'],required=True);a=p.parse_args();cfg=frozen()
    if a.phase=='extract':extract(cfg)
    else:points(cfg)
if __name__=='__main__':main()
