"""Explicit-state RVT stream, own FPN seam and one-to-one site candidates."""
import argparse,time
import numpy as np,torch
import torch.nn.functional as F
from confirmation_common import ROOT,OUT,read,sha,freeze,cohort,Scores,SCALES,sequences,record_cache
import run_rvt_reference as original
def clone_state(x):
    if x is None:return None
    if torch.is_tensor(x):return x.clone()
    if isinstance(x,dict):return {k:clone_state(v) for k,v in x.items()}
    return type(x)(clone_state(v) for v in x)
def equal_state(a,b):
    if a is None:return b is None
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal_state(a[k],b[k]) for k in a)
    return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);args=p.parse_args();cfg=freeze();model='rvt_stream'
    if args.sequence not in sequences(cfg):raise ValueError('Unregistered sequence')
    dest=OUT/(model+'__'+args.sequence+'.json')
    if dest.exists():
        if not read(dest)['complete']:raise ValueError('Incomplete receipt')
        print('Verified existing RVT cache',args.sequence,flush=True);return
    net,provenance=original.load_model()
    if provenance['checkpoint_sha256']!=cfg['rvt_checkpoint_sha256']:raise ValueError('RVT weights differ')
    if list(net.yolox_head.strides)!=SCALES:raise ValueError('Unexpected RVT output strides')
    reader=original.DSECPilotSequence(ROOT/'datasets/dsec_pilot',args.sequence);grid,times=cohort(reader,cfg);shapes={s:(512//s,640//s) for s in SCALES};predictor=Scores(cfg,shapes);rep=original.StackedHistogram(10,480,640,count_cutoff=255,fastmode=False);captured=[]
    hook=net.yolox_head.register_forward_pre_hook(lambda mod,a:captured.append([x.detach().clone() for x in a[0]]));states=None;previous_t=None;rows=[];alignment=[];errors={p:dict(absolute=0.,elements=0) for p in cfg['predictors']};folder=OUT/'cache'/model/args.sequence;folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with torch.inference_mode():
        for i,t in enumerate(times):
            duration=50000 if previous_t is None else t-previous_t
            if duration<=0:raise ValueError('Non-increasing stream')
            events,_=reader.read_window(t,duration);input_tensors=[torch.from_numpy(events[k].astype(np.int64)).cuda() for k in ['x','y','p','t']];x=F.pad(rep.construct(*input_tensors).float().unsqueeze(0),(0,0,0,32));captured.clear()
            diagnostic=i in [0,len(times)//2,len(times)-1];previous_states=clone_state(states) if diagnostic else None
            output,_,next_states=net(x,previous_states=states);feature=list(captured[0])
            if len(captured)!=1 or any(tuple(v.shape[-2:])!=shapes[s] for s,v in zip(SCALES,feature)):raise ValueError('RVT FPN/head seam geometry mismatch')
            if diagnostic:
                head,_=net.yolox_head(feature);open_exact=torch.equal(output,head)
                net(torch.zeros_like(x),previous_states=None)
                replay,_,replay_states=net(x,previous_states=previous_states);state_exact=torch.equal(output,replay) and equal_state(next_states,replay_states)
                alignment.append(dict(time_us=t,all_open_exact=open_exact,explicit_state_replay_exact=state_exact))
                if not open_exact or not state_exact:raise AssertionError('RVT head or explicit-state continuation failed')
            states=next_states;previous_t=t;h={s:np.ascontiguousarray(v[0].permute(1,2,0).cpu().numpy().reshape(-1,v.shape[1])) for s,v in zip(SCALES,feature)};counts={}
            for s in SCALES:
                height,width=shapes[s];sites=(events['y'].astype(np.int64)//s)*width+events['x'].astype(np.int64)//s;counts[s]=np.bincount(sites,minlength=height*width)
            arrays,error=predictor.observe(h,counts,sequences(cfg).index(args.sequence),int(np.searchsorted(grid,t)))
            out=output[0].cpu().numpy();boxes=np.concatenate((out[:,:2]-out[:,2:4]/2,out[:,:2]+out[:,2:4]/2),1);raw=out[:,4:5]*out[:,5:8];classes=raw.argmax(1);scores=np.zeros((len(out),2),np.float32);valid=classes!=1;mapped=np.where(classes[valid]==2,0,1);scores[np.flatnonzero(valid),mapped]=raw[valid,classes[valid]]
            if len(boxes)!=sum(a*b for a,b in shapes.values()):raise ValueError('RVT anchor-free site mapping differs')
            if i>=cfg['warmup']:
                for name,r in error.items():
                    for k,v in r.items():errors[name][k]+=v
            arrays.update(boxes=boxes,scores=scores,time_us=t,packet_duration_us=duration)
            for s in SCALES:arrays['h'+str(s)]=h[s];arrays['count'+str(s)]=counts[s]
            path=folder/(str(t)+'.npz');np.savez_compressed(path,**arrays);rows.append(dict(time_us=t,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),warmup=i<cfg['warmup']))
            if (i+1)%50==0:print('rvt_stream',args.sequence,i+1,'/',len(times),round(time.monotonic()-start,1),'s',flush=True)
    hook.remove();reader.close();provenance.update(anchors=1,coordinates='native',state='explicit carried recurrence; reset at clip start only',seam='actual FPN maps before YOLOX head stems; no feature/window equivalence to SpikeFPN claimed')
    record_cache(model,args.sequence,rows,shapes,errors,alignment,provenance);print('Completed RVT stream confirmation cache',args.sequence,flush=True)
if __name__=='__main__':main()
