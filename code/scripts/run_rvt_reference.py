"""Official RVT architecture and strict weights, with bounded native DSEC history."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from riskaware.dsec_pilot import DSECPilotSequence
SOURCE=ROOT/'vendor/rvt_snapshot'
# A separate process prevents upstream models/utils module namespace collisions.
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(ROOT/'vendor/rvt_runtime_deps'))
from omegaconf import OmegaConf
from models.detection.yolox_extension.models.detector import YoloXDetector
from data.utils.representations import StackedHistogram
CONFIG=ROOT/'code/configs/rvt_reference_v01.json';OUT=ROOT/'experiments/rvt_reference_v01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_model():
    cfg=read(CONFIG);p=ROOT/cfg['checkpoint']
    if sha(p)!=cfg['checkpoint_sha256']:raise ValueError('RVT checkpoint hash changed')
    torch.manual_seed(0);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    mdl=OmegaConf.load(SOURCE/'config/model/maxvit_yolox/default.yaml').model
    mdl.backbone.embed_dim=32;mdl.fpn.depth=.33
    mdl.backbone.in_res_hw=[512,640];mdl.backbone.stage.attention.partition_size=[8,10];mdl.head.num_classes=3
    model=YoloXDetector(mdl)
    # This trusted, SHA-verified upstream Lightning checkpoint has optimizer metadata.
    ckpt=torch.load(p,map_location='cpu',weights_only=False)
    if not all(k.startswith('mdl.') for k in ckpt['state_dict']):raise ValueError('Unexpected checkpoint prefixes')
    state={k[4:]:v for k,v in ckpt['state_dict'].items()}
    model.load_state_dict(state,strict=True)
    model=model.cuda().eval()
    return model,dict(checkpoint_sha256=sha(p),strict_load=True,state_keys=len(state),
        parameters=sum(p.numel() for p in model.parameters()),model_config=OmegaConf.to_container(mdl,resolve=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True);parser.add_argument('--limit',type=int);args=parser.parse_args()
    cfg=read(CONFIG)
    if args.sequence not in cfg['sequences']:raise ValueError('Unregistered sequence')
    folder=OUT/args.sequence/cfg['method'];folder.mkdir(parents=True,exist_ok=True)
    manifest=dict(config=cfg,sources={p:sha(ROOT/p) for p in ['code/scripts/run_rvt_reference.py','code/configs/rvt_reference_v01.json','code/riskaware/dsec_pilot.py','vendor/rvt_snapshot/_UPSTREAM.json','vendor/rvt_dependency_downloads/receipt.json']})
    mp=folder/'run_manifest.json'
    if mp.exists() and read(mp)!=manifest:raise ValueError('Source/config drift')
    save(mp,manifest);model,provenance=load_model();save(folder/'model_load.json',provenance)
    records=read(ROOT/'experiments/readout_adaptation_v01'/(args.sequence+'_cache.json'))['frames'][10:]
    if args.limit:records=records[:args.limit]
    reader=DSECPilotSequence(ROOT/'datasets/dsec_pilot',args.sequence)
    rep=StackedHistogram(10,480,640,count_cutoff=255,fastmode=False);started=time.monotonic()
    with torch.inference_mode():
        for i,r in enumerate(records):
            t=r['time_us'];p=folder/(str(t)+'.npz')
            if p.exists():continue
            tick=time.perf_counter();events,start=reader.read_window(t,180000)
            native_seconds=time.perf_counter()-tick;tick=time.perf_counter();inputs=[]
            for step in range(3):
                lo=start+step*60000;hi=lo+60000;mask=(events['t']>lo)&(events['t']<=hi)
                tensors=[torch.from_numpy(events[k][mask].astype(np.int64)).cuda() for k in ['x','y','p','t']]
                encoded=rep.construct(*tensors).float().unsqueeze(0)
                inputs.append(F.pad(encoded,(0,0,0,32)))
            torch.cuda.synchronize();prep=time.perf_counter()-tick;tick=time.perf_counter();states=None
            for step,x in enumerate(inputs):
                output,_,states=model(x,previous_states=states,retrieve_detections=step==2)
            torch.cuda.synchronize();gpu=time.perf_counter()-tick
            out=output[0].cpu().numpy();b=np.empty((len(out),4),dtype=np.float32)
            b[:,:2]=out[:,:2]-out[:,2:4]/2;b[:,2:]=out[:,:2]+out[:,2:4]/2
            # Keep the three-way argmax; removing class1 before argmax would change the task.
            raw=out[:,4:5]*out[:,5:8];classes=raw.argmax(1);keep=classes!=1
            b=b[keep];scores=np.zeros((int(keep.sum()),2),dtype=np.float32)
            mapped=np.where(classes[keep]==2,0,1)
            scores[np.arange(len(scores)),mapped]=raw[keep,classes[keep]]
            np.savez_compressed(p,time_us=t,boxes=b,scores=scores,timing=np.array([native_seconds,prep,gpu]),raw_events=len(events['x']))
            if (i+1)%25==0:print(f'RVT {args.sequence} {i+1}/{len(records)} {time.monotonic()-started:.1f}s',flush=True)
    reader.close();save(folder/'complete.json',dict(sequence=args.sequence,method=cfg['method'],frames=len(records),
        prediction_sha256={str(r['time_us']):sha(folder/(str(r['time_us'])+'.npz')) for r in records},model=provenance,limited=args.limit is not None))
    print('Finished RVT',args.sequence,len(records),flush=True)
if __name__=='__main__':main()
