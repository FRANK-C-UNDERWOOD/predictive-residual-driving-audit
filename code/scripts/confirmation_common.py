"""Architecture-neutral frozen-cohort bookkeeping and rectangular causal scores."""
from pathlib import Path
from collections import deque
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
OUT=ROOT/'experiments/confirmation_v01';CONFIG=ROOT/'code/configs/confirmation_v01.json';SCALES=[8,16,32]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.part');temp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def variants(cfg):return [p+'/'+n for p in cfg['predictors'] for n in cfg['normalizers']]+cfg['references']
def sequences(cfg):return [cfg['calibration_sequence']]+cfg['confirmation_sequences']
def freeze():
    cfg=read(CONFIG);OUT.mkdir(exist_ok=True)
    sources=['code/configs/confirmation_v01.json','code/scripts/confirmation_common.py','code/scripts/cache_confirmation_spike.py','code/scripts/cache_confirmation_rvt.py',
             'code/scripts/run_rvt_reference.py','code/scripts/run_diverse_adaptation.py','code/riskaware/dsec_pilot.py','code/riskaware/torch_spikefpn.py',
             'code/riskaware/criterion_audit.py','code/riskaware/metrics.py','code/riskaware/evidence_runs.py','code/riskaware/calibration_audit.py']
    m=dict(config=cfg,sources={p:sha(ROOT/p) for p in sources},selection_sha256=sha(OUT/'data_selection.json'),
           adapted_checkpoint=read(ROOT/'experiments/diverse_adaptation_v01/full_conv_training.json'),
           spike_source_receipt_sha256=sha(ROOT/'vendor/spikefpn_snapshot/_UPSTREAM.json'),rvt_source_receipt_sha256=sha(ROOT/'vendor/rvt_snapshot/_UPSTREAM.json'),
           rvt_runtime_receipt_sha256=sha(ROOT/'vendor/rvt_runtime_manifest.json'),paper_main_sha256=sha(ROOT/'paper/main.tex'))
    p=OUT/'run_manifest.json'
    if p.exists() and read(p)!=m:raise ValueError('Confirmation source/config drift')
    save(p,m);return cfg
def cohort(reader,cfg):
    grid=np.loadtxt(reader.path/'image_timestamps.txt',dtype=np.int64)
    times=[int(t) for t in grid if int(t)-180000>=reader.offset_us+reader.first_local_us and int(t)<=reader.offset_us+reader.last_local_us]
    if len(times)<=cfg['warmup']:raise ValueError('No complete evaluation cohort')
    record=dict(sequence=reader.sequence,grid_sha256=sha(reader.path/'image_timestamps.txt'),label_sha256=sha(reader.label_path),all_times=times,warmup=cfg['warmup'],queries=len(times)-cfg['warmup'])
    p=OUT/'cohorts'/(reader.sequence+'.json')
    if p.exists() and read(p)!=record:raise ValueError('Cohort drift')
    save(p,record);return grid,times
class Scores:
    def __init__(self,cfg,shapes):self.cfg=cfg;self.shapes=shapes;self.history=deque(maxlen=cfg['lag_queries']+1);self.previous=None
    def observe(self,h,counts,seq_index,grid_index):
        h={s:np.ascontiguousarray(v,dtype=np.float32) for s,v in h.items()};prior=self.history[-1] if self.history else h;lag=self.history[0] if self.history else prior;previous=self.previous if self.previous is not None else h
        scores={};error={p:dict(absolute=0.,elements=0) for p in self.cfg['predictors']};num=[];mag=[];zero=[]
        for s,x in h.items():
            height,width=self.shapes[s];c=x.shape[1]
            if len(x)!=height*width:raise ValueError('Descriptor/grid mismatch')
            moved=np.roll(prior[s].reshape(height,width,c),(height//3,width//3),axis=(0,1)).reshape(x.shape)
            preds=dict(zero=np.zeros_like(x),persistence=previous[s],ema=prior[s],ema_lag3=lag[s],ema_roll=moved,ema_global=np.broadcast_to(prior[s].mean(0,keepdims=True),x.shape))
            magnitude=np.abs(x).sum(1);mag.append(magnitude);zero.append(magnitude==0)
            for name,p in preds.items():
                delta=np.abs(x-p);n=delta.sum(1)
                if name=='ema':num.append(n)
                values=dict(absolute=delta.mean(1),current=n/(magnitude+self.cfg['epsilon']),symmetric=n/(magnitude+np.abs(p).sum(1)+self.cfg['epsilon']))
                for norm,v in values.items():scores.setdefault(name+'/'+norm,[]).append(v)
                error[name]['absolute']+=float(delta.sum(dtype=np.float64));error[name]['elements']+=x.size
            refs=dict(current_activity=np.abs(x).mean(1),ema_activity=np.abs(prior[s]).mean(1),event_rate=counts[s].reshape(-1),random=np.random.default_rng(self.cfg['ranking_seed']+seq_index*100000+grid_index*100+s).random(len(x)))
            for name,v in refs.items():scores.setdefault(name,[]).append(v)
        self.history.append({s:self.cfg['alpha']*prior[s]+(1-self.cfg['alpha'])*x for s,x in h.items()});self.previous={s:x.copy() for s,x in h.items()}
        return dict(rank_scores=np.stack([np.concatenate(scores[n]) for n in variants(self.cfg)]),n_ema=np.concatenate(num),magnitude=np.concatenate(mag),zero_descriptor=np.concatenate(zero)),error
def record_cache(model,seq,rows,shapes,errors,alignment,provenance,legacy_checks=0):
    cfg=read(CONFIG);save(OUT/(model+'__'+seq+'.json'),dict(model=model,sequence=seq,complete=True,geometry=[dict(stride=s,height=shapes[s][0],width=shapes[s][1]) for s in SCALES],
             variants=variants(cfg),all_frames=rows,frames=rows[cfg['warmup']:],prediction_error=errors,alignment=alignment,model_provenance=provenance,legacy_exact_queries=legacy_checks,
             role='calibration' if seq==cfg['calibration_sequence'] else 'new_frozen_confirmation'))
