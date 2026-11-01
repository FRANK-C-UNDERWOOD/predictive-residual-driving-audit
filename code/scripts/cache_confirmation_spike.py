"""Unchanged SpikeFPN carriers, actual pre-smooth features and frozen query grids."""
import argparse,time
from collections import defaultdict
import numpy as np,torch
from confirmation_common import ROOT,OUT,read,save,sha,freeze,cohort,Scores,SCALES,sequences,record_cache
from riskaware.dsec_pilot import DSECPilotSequence
from riskaware.torch_spikefpn import NAMES,replay_dense
from run_diverse_adaptation import selected_model
from run_adapted_criterion import cache as old_cache,frame as old_frame
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['spike_original','spike_adapted'],required=True);p.add_argument('--sequence',required=True);args=p.parse_args();cfg=freeze()
    if args.sequence not in sequences(cfg):raise ValueError('Unregistered sequence')
    dest=OUT/(args.model+'__'+args.sequence+'.json')
    if dest.exists():
        if not read(dest)['complete']:raise ValueError('Incomplete receipt')
        print('Verified existing complete spike cache',args.model,args.sequence,flush=True);return
    variant='original' if args.model=='spike_original' else 'full_conv';net=selected_model(variant,read(ROOT/'code/configs/diverse_adaptation_v01.json'));reader=DSECPilotSequence(ROOT/'datasets/dsec_pilot',args.sequence);grid,times=cohort(reader,cfg);shapes={s:(256//s,256//s) for s in SCALES};predictor=Scores(cfg,shapes);captured={s:[] for s in SCALES};hooks=[]
    for s in SCALES:
        def capture(mod,a,s=s):captured[s].append(a[0].detach().clone())
        hooks.append(getattr(net.feature,NAMES[s][0]).register_forward_pre_hook(capture))
    reference=old_cache(variant,args.sequence) if args.sequence==cfg['calibration_sequence'] else None
    errors={p:dict(absolute=0.,elements=0) for p in cfg['predictors']};rows=[];alignment=[];legacy=0;folder=OUT/'cache'/args.model/args.sequence;folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with torch.inference_mode():
        for i,t in enumerate(times):
            for v in captured.values():v.clear()
            x,_,counts=reader.signed_stack(t,return_counts=True);output=net(torch.from_numpy(x).cuda());shared={s:list(captured[s]) for s in SCALES}
            if any(len(v)!=3 for v in shared.values()):raise ValueError('Three actual pre-smooth inputs required')
            h={}
            for s,v in shared.items():
                z=torch.stack(v)[:,0].cpu().numpy();h[s]=np.ascontiguousarray(z.mean(0).transpose(1,2,0).reshape(-1,z.shape[1]))
            arrays,error=predictor.observe(h,counts,sequences(cfg).index(args.sequence),int(np.searchsorted(grid,t)));boxes=output[3].clamp(0,1)[0].cpu().numpy();scores=(output[1].softmax(-1)*output[0].sigmoid())[0].cpu().numpy()
            if i in [0,len(times)//2,len(times)-1]:
                replay,_,_=replay_dense(net,shared);exact=all(torch.equal(a,b) for a,b in zip(output,replay));alignment.append(dict(time_us=t,all_open_exact=exact))
                if not exact:raise AssertionError('Spike full-open graph failed')
            if i>=cfg['warmup']:
                for name,r in error.items():
                    for k,v in r.items():errors[name][k]+=v
                if reference:
                    old=old_frame(reference['frames'][i-cfg['warmup']])
                    if not np.array_equal(boxes,old['boxes']) or not np.array_equal(scores,old['scores']) or not np.array_equal(arrays['rank_scores'],old['rank_scores']):raise AssertionError('Protected Thun raw or22score regression failed')
                    legacy+=1
            arrays.update(boxes=boxes,scores=scores,time_us=t)
            for s in SCALES:arrays['h'+str(s)]=h[s];arrays['count'+str(s)]=counts[s].reshape(-1)
            path=folder/(str(t)+'.npz');np.savez_compressed(path,**arrays);rows.append(dict(time_us=t,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),warmup=i<cfg['warmup']))
            if (i+1)%50==0:print(args.model,args.sequence,i+1,'/',len(times),round(time.monotonic()-start,1),'s',flush=True)
    for hook in hooks:hook.remove()
    reader.close();record_cache(args.model,args.sequence,rows,shapes,errors,alignment,dict(variant=variant,checkpoint='pinned original plus selected116 delta where applicable',anchors=3,coordinates='spike256_normalized',state='reset per180ms query; EMA full-grid across queries'),legacy)
    print('Completed spike confirmation cache',args.model,args.sequence,flush=True)
if __name__=='__main__':main()
