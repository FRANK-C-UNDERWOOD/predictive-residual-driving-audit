"""Independent masks, greedy identity matching and all reported FP-point endpoints."""
from collections import defaultdict
import json
import numpy as np
from confirmation_common import ROOT,read,save,sha
from confirmation_evaluate import truth
from riskaware.criterion_audit import filter_anchors
from utils.tools import nms
OUT=ROOT/'experiments/fp_mechanism_v01';PARENT=ROOT/'experiments/confirmation_v01'
def independent_match(boxes,scores,classes,gt):
    result=np.full(len(scores),-1,np.int64);used=set()
    for i in sorted(range(len(scores)),key=lambda j:-float(scores[j])):
        best=None;best_iou=.5;a=boxes[i]
        for j,g in enumerate(gt):
            if j in used or g.cls!=int(classes[i]):continue
            b=g.box;inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]));union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter;v=inter/union if union>0 else 0.
            if v>=best_iou:best=j;best_iou=v
        if best is not None:used.add(best);result[i]=gt[best].track_id
    return result
def native(boxes,m):return (boxes*256-np.array([0.,32.,0.,32.]))/.4 if m['model_provenance']['coordinates']=='spike256_normalized' else boxes.astype(np.float64)
def replay(d,mask,theta,m,gt):
    f=filter_anchors(d['boxes'],d['scores'],mask,m['model_provenance']['anchors'],theta,.5,nms);labels=independent_match(native(f[0],m),f[1],f[2],gt);return f,labels
def main():
    cfg=read(ROOT/'code/configs/fp_mechanism_v01.json');frozen=read(OUT/'freeze.json');checks=score_queries=point_rows=floor_outputs=exceptions=sampled_nms=0
    for p,h in frozen['sources'].items():
        if sha(ROOT/p)!=h:raise AssertionError('Secondary-analysis source drift '+p)
    for p,h in frozen['parents'].items():
        if sha(PARENT/p)!=h:raise AssertionError('Protected parent changed '+p)
    parentcfg=read(ROOT/'code/configs/confirmation_v01.json');thresholds=read(PARENT/'frozen_thresholds.json');cal=read(OUT/'calibration_points.json')
    for model in cfg['models']:
        for seq in [cfg['calibration_sequence']]+cfg['sequences']:
            m=read(PARENT/(model+'__'+seq+'.json'));receipt=read(OUT/(model+'__'+seq+'.json'));folder=OUT/'floor'/model/seq;_,gts=truth(seq,parentcfg);sizes=[g['height']*g['width'] for g in m['geometry']];bounds=np.cumsum([0]+sizes);n=sum(sizes);aa=m['model_provenance']['anchors']
            with np.load(folder/'masks.npz') as z:packed=z['masks'];rr=z['reference']
            refs=defaultdict(set)
            for q,i in rr:refs[int(q)].add(int(i))
            floor={};slices={}
            for method in cfg['methods']:
                with np.load(folder/(method.replace('/','_')+'.npz')) as z:floor[method]=z['detections']
                a=floor[method]['q'];slices[method]=np.searchsorted(a,np.arange(len(m['frames'])+1))
            common=defaultdict(list);support=defaultdict(list);mechanism={}
            for name,dest in [('common_frames.jsonl',common),('dense_support.jsonl',support)]:
                with (folder/name).open() as f:
                    for line in f:r=json.loads(line);dest[r['q']].append(r)
            with (folder/'query_mechanism.jsonl').open() as f:
                for line in f:r=json.loads(line);mechanism[r['q']]=r
            points=cal[model] if seq==cfg['calibration_sequence'] else read(OUT/'points'/(model+'__'+seq+'.json'))['conditions'];by_q=defaultdict(list)
            with (OUT/'points'/model/(seq+'.jsonl')).open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['attempt']==points[r['key']]['attempt']:by_q[r['q']].append(r)
            totals=defaultdict(lambda:defaultdict(int));ctotals=defaultdict(lambda:defaultdict(int));prior=None;qi=0
            for record in m['all_frames']:
                path=ROOT/record['path']
                if sha(path)!=record['sha256']:raise AssertionError('Raw input cache changed')
                with np.load(path) as z:d={k:z[k] for k in ['boxes','scores','rank_scores','h8','h16','h32']}
                h={s:d['h'+str(s)] for s in [8,16,32]};p=h if prior is None else prior
                if not record['warmup']:
                    t=record['time_us'];gt=gts.get(t,[]);uu=np.concatenate([(np.maximum(np.abs(x),np.abs(p[s]))-np.minimum(np.abs(x),np.abs(p[s]))).mean(1) for s,x in h.items()]);rebuilt=[]
                    for method in cfg['methods']:
                        mask=np.ones(n,bool) if method=='dense' else np.zeros(n,bool)
                        if method!='dense':
                            values=uu if method=='unsigned_residual' else d['rank_scores'][m['variants'].index(method)]
                            for lo,hi in zip(bounds[:-1],bounds[1:]):order=np.argsort(-values[lo:hi],kind='stable');mask[lo+order[:max(1,int(np.floor(.2*(hi-lo))))]]=True
                        rebuilt.append(mask)
                    masks=np.stack(rebuilt)
                    if not np.array_equal(masks,np.unpackbits(packed[qi],axis=1)[:,:n].astype(bool)):raise AssertionError('Independent hard mask mismatch')
                    score_queries+=1;local={};rawscore=d['scores'];rawclass=rawscore.argmax(1);rawconfidence=rawscore.max(1)
                    for mi,method in enumerate(cfg['methods']):
                        lo,hi=slices[method][qi:qi+2];rows=floor[method][lo:hi];anchors=rows['anchor'];boxes=native(d['boxes'][anchors],m)
                        if not np.array_equal(rows['score'],rawconfidence[anchors]) or not np.all(masks[mi][anchors//aa]):raise AssertionError('Floor anchor admission/probability mismatch')
                        labels=independent_match(boxes,rows['score'],rawclass[anchors],gt)
                        if not np.array_equal(labels,rows['identity']):raise AssertionError('Independent floor matching failed')
                        floor_outputs+=len(rows);local[method]=rows
                        if qi in [0,len(m['frames'])//2,len(m['frames'])-1]:
                            ff,ll=replay(d,masks[mi],cfg['confidence_floor'],m,gt)
                            if not np.array_equal(ff[3],anchors) or not np.array_equal(ll,labels):raise AssertionError('Sampled raw floor NMS failed')
                            sampled_nms+=1
                    memo={}
                    def actual(method,theta,force=False):
                        nonlocal exceptions,sampled_nms
                        rows=local[method];cut=rows[rows['score']>=theta];ids=set(map(int,cut['identity'][cut['identity']>=0]));fp=int((cut['identity']<0).sum())
                        if force:
                            ff,ll=replay(d,masks[cfg['methods'].index(method)],theta,m,gt);sampled_nms+=1;return set(map(int,ll[ll>=0])),int((ll<0).sum()),ff,ll
                        return ids,fp,None,None
                    for r in common[qi]+by_q[qi]:
                        ispoint='key' in r;plan=points[r['key']] if ispoint else dict(method=r['method'],threshold=thresholds[model]['dense_threshold']);method=plan['method'];theta=plan['threshold'];key=(method,theta)
                        if key not in memo:memo[key]=actual(method,theta)
                        ids,fp,_,_=memo[key]
                        if ids!=set(r['matched_ids']) or fp!=r['fp']:
                            # Upstream unstable equal-score NMS can invalidate a confidence prefix.
                            ids,fp,_,_=actual(method,theta,True);memo[key]=(ids,fp,None,None);exceptions+=1
                        if ids!=set(r['matched_ids']) or fp!=r['fp'] or r['tp']!=len(ids) or r['gt']!=len(gt) or set(r['dense_ids'])!=refs[qi]:raise AssertionError('Actual threshold identity/FP mismatch')
                        if ispoint:
                            if r['threshold']!=theta:raise AssertionError('Wrong reported threshold')
                            v=totals[r['key']];point_rows+=1
                        else:v=ctotals[method]
                        for k,value in dict(queries=1,gt=len(gt),tp=len(ids),fp=fp,dense_tp=len(refs[qi]),kept=len(ids&refs[qi]),gained=len(ids-refs[qi])).items():v[k]+=value
                        if ispoint and qi in [0,len(m['frames'])//2,len(m['frames'])-1]:
                            ids2,fp2,_,_=actual(method,theta,True)
                            if ids2!=ids or fp2!=fp:raise AssertionError('Sampled full point replay failed')
                    if len(common[qi])!=len(cfg['methods']) or len(by_q[qi])!=len(points):raise AssertionError('Missing condition-query rows')
                    ff,ll=replay(d,masks[0],thresholds[model]['dense_threshold'],m,gt);actual_support={(int(i),int(a)) for i,a in zip(ll,ff[3]) if i>=0}
                    if refs[qi]!={i for i,a in actual_support} or {(r['identity'],r['anchor']) for r in support[qi]}!=actual_support:raise AssertionError('Dense source identity/anchor mismatch')
                    A=np.concatenate([np.abs(x).mean(1) for x in h.values()]);N=np.concatenate([np.abs(x-p[s]).mean(1) for s,x in h.items()]);B=np.concatenate([np.abs(p[s]).mean(1) for s in h]);O=np.concatenate([np.where(np.sign(x)==np.sign(p[s]),np.minimum(np.abs(x),np.abs(p[s])),0).mean(1) for s,x in h.items()]);S=np.concatenate([np.where(np.sign(x)!=np.sign(p[s]),np.minimum(np.abs(x),np.abs(p[s])),0).mean(1) for s,x in h.items()])
                    am=masks[cfg['methods'].index('current_activity')];nm=masks[cfg['methods'].index('ema/absolute')]
                    for r in support[qi]:
                        j=r['site'];expected=dict(activity=A[j],past_activity=B[j],residual=N[j],overlap=O[j],unsigned_residual=uu[j],opposition=S[j],temporal_adjustment=N[j]-A[j])
                        if any(float(v)!=r[k] for k,v in expected.items()) or r['activity_selected']!=bool(am[j]) or r['residual_selected']!=bool(nm[j]) or j!=r['anchor']//aa:raise AssertionError('Support feature mechanism mismatch')
                    q=mechanism[qi];negative=sum(int((x<0).sum()) for x in h.values())/sum(x.size for x in h.values())
                    if q['negative_fraction']!=negative or q['unsigned_mask_same']!=bool(np.array_equal(masks[-1],nm)):raise AssertionError('Representation/sign counter mismatch')
                    qi+=1
                    if qi%300==0:save(OUT/'verification_status.json',dict(state='running',model=model,sequence=seq,queries=qi,point_rows=point_rows));print('Independent FP/mechanism',model,seq,qi,flush=True)
                prior={s:cfg['alpha']*p[s]+(1-cfg['alpha'])*x for s,x in h.items()}
            for key,v in totals.items():
                point=points[key]
                if any(point[k]!=value for k,value in v.items()):raise AssertionError('Point aggregate mismatch')
                if point['regime']!='thun_calibrated' and point['fp']>point['allowed_fp']:raise AssertionError('Actual FP budget not met')
                if point['recall']!=(point['tp']/point['gt'] if point['gt'] else None) or point['preservation']!=(point['kept']/point['dense_tp'] if point['dense_tp'] else None) or point['fppi']!=point['fp']/point['queries']:raise AssertionError('Endpoint ratio mismatch')
                checks+=1
            for method,v in ctotals.items():
                if any(receipt['common_totals'][method][k]!=value for k,value in v.items()):raise AssertionError('Common point aggregate mismatch')
                checks+=1
            if qi!=len(m['frames']):raise AssertionError('Query cohort incomplete')
            print('Verified entire carrier/clip',model,seq,flush=True)
    save(OUT/'verification.json',dict(passed=True,checks=checks,causal_mask_queries=score_queries,point_query_rows=point_rows,floor_detections_independently_matched=floor_outputs,prefix_exceptions_replayed=exceptions,sampled_raw_nms_checks=sampled_nms,scope='All masks, floor identity matching, reported endpoint identities and aggregate FP budgets; raw NMS shared with parent and sampled/exception replays. Software counts are not independent driving samples.'));save(OUT/'verification_status.json',dict(state='complete'));print('Final secondary analyses independently verified',flush=True)
if __name__=='__main__':main()
