"""Explain the already-declared source/NMS gap; no new masks or FP points."""
from collections import defaultdict
import json
import numpy as np
from confirmation_common import ROOT,read,save,sha
from confirmation_evaluate import truth
from run_fp_mechanism import canonical,metadata,load_raw
OUT=ROOT/'experiments/fp_mechanism_v01';DEST=OUT/'source_survival';METHODS=['ema/absolute','current_activity']
def main():
    if not read(OUT/'verification.json')['passed']:raise ValueError('Base verification required')
    DEST.mkdir(exist_ok=True);cfg=read(ROOT/'code/configs/fp_mechanism_v01.json');pc=read(ROOT/'code/configs/confirmation_v01.json');th=read(ROOT/'experiments/confirmation_v01/frozen_thresholds.json');summary=[];rows=0
    for model in cfg['models']:
        for seq in cfg['sequences']:
            dest=DEST/(model+'__'+seq+'.json')
            if dest.exists():summary.extend(read(dest)['conditions']);continue
            m=metadata(model,seq);_,gts=truth(seq,pc);folder=OUT/'floor'/model/seq;points=read(OUT/'points'/(model+'__'+seq+'.json'))['conditions'];support=defaultdict(list);common=defaultdict(dict);reported=defaultdict(dict)
            with (folder/'dense_support.jsonl').open() as f:
                for line in f:r=json.loads(line);support[r['q']].append(r)
            with (folder/'common_frames.jsonl').open() as f:
                for line in f:r=json.loads(line);common[r['q']][r['method']]=r
            with (OUT/'points'/model/(seq+'.jsonl')).open() as f:
                for line in f:
                    r=json.loads(line)
                    if r['key'] not in points:continue
                    p=points[r['key']]
                    if p['regime']=='retrospective_sequence_budget' and p['target']==1.0 and p['method'] in METHODS and p['attempt']==r['attempt']:reported[r['q']][p['method']]=r
            with np.load(folder/'masks.npz') as f:packed=f['masks']
            n=sum(g['height']*g['width'] for g in m['geometry']);totals=defaultdict(lambda:defaultdict(int));trace=DEST/(model+'__'+seq+'.jsonl')
            with trace.open('w') as handle:
                for q,r in enumerate(m['frames']):
                    d=load_raw(r);gt=gts.get(r['time_us'],[]);masks=np.unpackbits(packed[q],axis=1)[:,:n].astype(bool)
                    for regime in ['common_dense','retrospective_1fp']:
                        for method in METHODS:
                            threshold=th[model]['dense_threshold'] if regime=='common_dense' else points['retrospective_sequence_budget|'+method+'@1.0']['threshold'];mask=masks[cfg['methods'].index(method)];f,matched,labels=canonical(d,mask,threshold,m,gt);ids=set(map(int,labels[labels>=0]));expected=common[q][method] if regime=='common_dense' else reported[q][method]
                            if ids!=set(expected['matched_ids']) or len(labels)-len(matched)!=expected['fp']:raise AssertionError('Source replay differs from protected endpoint')
                            surviving=set(map(int,f[3]));matched_anchor={gt[gi].track_id:int(f[3][di]) for gi,di in matched.items()};above=d['scores'].max(1)>=threshold;v=totals[(regime,method)];v['queries']+=1;v['tp']+=len(matched);v['fp']+=len(labels)-len(matched)
                            for source in support[q]:
                                anchor=source['anchor'];identity=source['identity'];selected=bool(mask[source['site']]);eligible=selected and bool(above[anchor]);detected=identity in ids
                                if detected:
                                    category='original_match' if matched_anchor[identity]==anchor else 'admitted_alternative' if eligible else 'alternative_after_mask' if not selected else 'alternative_after_confidence'
                                elif eligible:category='nms_loss' if anchor not in surviving else 'assignment_loss'
                                else:category='mask_miss' if not selected else 'confidence_miss'
                                v[category]+=1;v['dense_instances']+=1;v['effective_source_coverage']+=eligible;v['kept']+=detected
                                handle.write(json.dumps(dict(q=q,time_us=r['time_us'],regime=regime,method=method,identity=identity,original_anchor=anchor,matched_anchor=matched_anchor.get(identity),mask_selected=selected,above_threshold=bool(above[anchor]),original_survives_nms=anchor in surviving,identity_detected=detected,category=category),separators=(',',':'))+'\n');rows+=1
                    if (q+1)%500==0:save(DEST/'status.json',dict(state='running',model=model,sequence=seq,queries=q+1));print('Source survival',model,seq,q+1,flush=True)
            conditions=[];cats=['original_match','admitted_alternative','alternative_after_mask','alternative_after_confidence','nms_loss','assignment_loss','mask_miss','confidence_miss']
            for (regime,method),v in totals.items():
                v=dict(v)
                for c in cats:v.setdefault(c,0)
                if sum(v[c] for c in cats)!=v['dense_instances'] or v['kept']!=sum(v[c] for c in cats[:4]) or v['effective_source_coverage']!=v['original_match']+v['admitted_alternative']+v['nms_loss']+v['assignment_loss']:raise AssertionError('Source partition identity failed')
                expected=read(OUT/(model+'__'+seq+'.json'))['common_totals'][method] if regime=='common_dense' else points['retrospective_sequence_budget|'+method+'@1.0']
                if any(v[k]!=expected[k] for k in ['queries','tp','fp','kept']):raise AssertionError('Source partition aggregate changed endpoints')
                conditions.append(dict(model=model,sequence=seq,regime=regime,method=method,**v))
            save(dest,dict(conditions=conditions,trace_sha256=sha(trace)));summary.extend(conditions)
    # Independently count the saved categories and reconstruct every algebraic term.
    checked=0;verified_rows=0
    for p in DEST.glob('*__*.jsonl'):
        counts=defaultdict(lambda:defaultdict(int))
        with p.open() as f:
            for line in f:
                r=json.loads(line);key=(r['regime'],r['method']);counts[key][r['category']]+=1;counts[key]['dense_instances']+=1;counts[key]['kept']+=r['identity_detected'];counts[key]['effective_source_coverage']+=r['mask_selected'] and r['above_threshold'];verified_rows+=1
                if r['category']=='nms_loss' and (r['original_survives_nms'] or r['identity_detected']):raise AssertionError('False NMS attribution')
                if r['category']=='assignment_loss' and (not r['original_survives_nms'] or r['identity_detected']):raise AssertionError('False match attribution')
        model,seq=p.stem.split('__')
        for (regime,method),v in counts.items():
            expected=next(r for r in summary if r['model']==model and r['sequence']==seq and r['regime']==regime and r['method']==method)
            if any(expected[k]!=value for k,value in v.items()):raise AssertionError('Independent source category tally failed')
            checked+=1
    save(DEST/'summary.json',summary);save(DEST/'verification.json',dict(passed=True,conditions=checked,identity_query_rows=verified_rows,source_sha256=sha(ROOT/'code/scripts/trace_source_survival.py'),scope='Actual raw NMS/matching replay at existing common and retrospective1FP thresholds; source/alternative category tally. No new model, criterion or threshold.'))
    save(DEST/'status.json',dict(state='complete'));print('Source survival diagnosis complete',flush=True)
if __name__=='__main__':main()
