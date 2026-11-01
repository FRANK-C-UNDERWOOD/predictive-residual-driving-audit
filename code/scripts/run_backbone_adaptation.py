"""Six-epoch controlled convolution adaptation; checkpoint fixed before evaluation."""
from pathlib import Path
import argparse,hashlib,json,sys,time,gc
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from riskaware.torch_spikefpn import load_official_model
from riskaware.dsec_pilot import DSECPilotSequence
from riskaware.readout_adaptation import training_labels
from riskaware.trainable_backbone import configure,differentiable_fixed_bn,tensor_hash,accumulation_weight
from riskaware.criterion_audit import ground_truth,filter_anchors,native_detections
from riskaware.metrics import match_frame
from riskaware.calibration_audit import threshold_at_fppi
OUT=ROOT/'experiments/backbone_adaptation_v01';CONFIG=ROOT/'code/configs/backbone_adaptation_v01.json'
PARENT=ROOT/'experiments/predictor_mechanism_v01'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    part=p.with_suffix(p.suffix+'.part');part.write_text(json.dumps(x,indent=2,allow_nan=False));part.replace(p)

def freeze(cfg):
    OUT.mkdir(exist_ok=True)
    sources=['code/scripts/run_backbone_adaptation.py','code/riskaware/trainable_backbone.py',
        'code/scripts/probe_backbone_adaptation.py','code/tests/test_trainable_backbone.py',
        'code/riskaware/dsec_pilot.py','code/riskaware/readout_adaptation.py','code/riskaware/torch_spikefpn.py',
        'code/riskaware/criterion_audit.py','code/riskaware/calibration_audit.py','code/riskaware/metrics.py',
        'vendor/spikefpn_snapshot/utils/criterion.py','vendor/spikefpn_snapshot/utils/create_labels.py','vendor/spikefpn_snapshot/utils/tools.py']
    m=dict(config=cfg,config_sha256=sha(CONFIG),sources={p:sha(ROOT/p) for p in sources},
           preflight_sha256=sha(OUT/'preflight.json'),parent_manifest_sha256=sha(PARENT/'artifact_manifest.json'))
    path=OUT/'run_manifest.json'
    if path.exists() and read(path)!=m:raise ValueError('Source/config drift')
    save(path,m)
    if not read(OUT/'preflight.json')['passed']:raise ValueError('Preflight failed')

def cache_training(cfg):
    if (OUT/'training_inputs.json').exists():
        old=read(OUT/'training_inputs.json')
        for item in old['files']:
            if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Training inputs changed')
        return
    model,_=load_official_model()
    from utils.create_labels import gt_creator
    seq=cfg['train_sequences'][0]
    rows=read(ROOT/'experiments/readout_adaptation_v01'/f'{seq}_cache.json')['frames'][cfg['warmup_complete_windows']:]
    reader=DSECPilotSequence(ROOT/'datasets/dsec_pilot',seq)
    x=np.lib.format.open_memmap(OUT/'training_inputs.npy',mode='w+',dtype=np.int8,shape=(len(rows),3,3,256,256))
    targets=[];counts=dict(supported=0,ignored=0,resized_tiny=0)
    for i,row in enumerate(rows):
        t=row['time_us'];array,_=reader.signed_stack(t,**cfg['input'])
        if not np.isin(array,[-1,0,1]).all():raise ValueError('Invalid signed input')
        x[i]=array[0]
        labels,c=training_labels(reader.labels[reader.labels['t']==t],cfg['class_map_dsec_to_detector'])
        for k in counts:counts[k]+=c[k]
        targets.append(gt_creator(256,model.stride,[labels],model.anchor_list,multi_anchor=True)[0].numpy())
        if (i+1)%100==0:print('train input cache',i+1,'/',len(rows),flush=True)
    reader.close();x.flush();del x,model;gc.collect();torch.cuda.empty_cache()
    np.save(OUT/'training_targets.npy',np.stack(targets));np.save(OUT/'training_times.npy',[r['time_us'] for r in rows])
    files=[OUT/n for n in ['training_inputs.npy','training_targets.npy','training_times.npy']]
    raw=ROOT/'datasets/dsec_pilot/train'/seq
    save(OUT/'training_inputs.json',dict(sequence=seq,windows=len(rows),label_counts=counts,
         files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files],
         raw_inputs=[dict(path=(raw/p).relative_to(ROOT).as_posix(),sha256=sha(raw/p)) for p in ['events/left/events.h5','image_timestamps.txt','object_detections/left/tracks.npy']]))

def train(cfg):
    if (OUT/'training_summary.json').exists():return
    data=np.load(OUT/'training_inputs.npy',mmap_mode='r');target=np.load(OUT/'training_targets.npy',mmap_mode='r')
    n=len(data);micro=read(OUT/'preflight.json')['accepted_microbatch'];records=[]
    for variant in ['readout','full_conv']:
        completed=OUT/(variant+'_training.json')
        if completed.exists():records.append(read(completed));continue
        model,provenance=load_official_model();params=configure(model,variant)
        from utils.criterion import build_criterion
        from utils.spikefpn_config import spikefpn_cfg
        from utils.tools import calculate_loss_new
        criterion=build_criterion(SimpleNamespace(scale_loss='positive'),spikefpn_cfg,2)
        frozen=tensor_hash({k:v for k,v in model.state_dict().items() if k not in params})
        groups=[dict(params=[p for k,p in params.items() if k.startswith('head_det_')],lr=cfg['head_lr'])]
        if variant=='full_conv':groups.append(dict(params=[p for k,p in params.items() if k.startswith('feature.')],lr=cfg['feature_conv_lr']))
        optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
        gen=torch.Generator().manual_seed(cfg['seed']);logs=[];updates=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,cfg['epochs']+1):
            order=torch.randperm(n,generator=gen).tolist();sums=np.zeros(4);seen=0;maxgrad=0
            for begin in range(0,n,cfg['effective_batch_size']):
                ids=order[begin:begin+cfg['effective_batch_size']]
                effective=torch.from_numpy(np.array(target[ids],copy=True)).cuda();optimizer.zero_grad(set_to_none=True);batchloss=np.zeros(4)
                for mi in range(0,len(ids),micro):
                    sub=ids[mi:mi+micro];x=torch.from_numpy(np.array(data[sub],dtype=np.float32,copy=True)).cuda()
                    y=effective[mi:mi+len(sub)]
                    with differentiable_fixed_bn():
                        output=model(x);losses=calculate_loss_new(output[0],output[1],output[3],y,criterion)
                        weight=accumulation_weight(y,effective)
                        if not all(bool(torch.isfinite(z)) for z in losses):raise ValueError('Nonfinite training loss')
                        (losses[-1]*weight).backward()
                    batchloss+=np.array([float(v.detach()*weight) for v in losses])
                    del output,losses,x,y
                if updates==0:
                    absent=[k for k,p in params.items() if p.grad is None]
                    if absent:raise ValueError('Trainable parameters missing gradients: '+str(absent))
                norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['gradient_clip_norm'])
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                maxgrad=max(maxgrad,float(norm));optimizer.step();updates+=1;sums+=batchloss*len(ids);seen+=len(ids)
                if updates%20==0:print(variant,'epoch',epoch,'update',updates,'elapsed',round(time.monotonic()-started,1),flush=True)
            after=tensor_hash({k:v for k,v in model.state_dict().items() if k not in params})
            if after!=frozen:raise ValueError('Frozen BN/neuron state changed')
            row=dict(epoch=epoch,losses=(sums/seen).tolist(),updates=updates,windows=seen,max_gradient=maxgrad,
                     order_sha256=hashlib.sha256(np.array(order,dtype='<i8').tobytes()).hexdigest(),elapsed_seconds=time.monotonic()-started)
            logs.append(row);save(OUT/(variant+'_progress.json'),dict(variant=variant,logs=logs))
            if epoch in (1,3,cfg['epochs']):
                torch.save(dict(parameters={k:v.detach().cpu() for k,v in model.state_dict().items() if k in params},epoch=epoch,variant=variant),OUT/f'{variant}_epoch{epoch}.pth')
            print(variant,'finished epoch',epoch,'loss',row['losses'][-1],flush=True)
        checkpoint=OUT/f'{variant}_epoch{cfg["epochs"]}.pth'
        record=dict(variant=variant,checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint),
            windows=n,updates=updates,logs=logs,trainable_parameters=sum(p.numel() for p in params.values()),
            frozen_hash_before=frozen,frozen_hash_after=after,seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),accepted_microbatch=micro,model=provenance)
        save(completed,record);records.append(record);del model,optimizer,params;gc.collect();torch.cuda.empty_cache()
    if records[0]['updates']!=records[1]['updates'] or [r['order_sha256'] for r in records[0]['logs']]!=[r['order_sha256'] for r in records[1]['logs']]:raise ValueError('Control training exposure differs')
    save(OUT/'training_summary.json',dict(config=cfg,variants=records))

def load_variant(variant,cfg):
    model,prov=load_official_model()
    if variant!='original':
        record=read(OUT/(variant+'_training.json'));p=ROOT/record['checkpoint']
        if sha(p)!=record['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        state=torch.load(p,map_location='cpu',weights_only=True)['parameters']
        expected=set(configure(model,variant))
        if set(state)!=expected:raise ValueError('Wrong trained parameter keys')
        with torch.no_grad():
            for k,v in state.items():model.state_dict()[k].copy_(v)
    for p in model.parameters():p.requires_grad_(False)
    return model

def evaluate(cfg):
    if (OUT/'evaluation_summary.json').exists():return
    from utils.tools import nms
    from check_fullmodel_alignment import Recorder,flatten,compare_arrays,compare_traces
    model={v:load_variant(v,cfg) for v in cfg['variants']};thresholds={};summary={};alignment=[];metadata=[]
    mask=np.ones(1344,dtype=bool);outcomes=OUT/'frames.jsonl'
    with torch.inference_mode(),outcomes.open('w') as log:
        for seq in cfg['calibration_sequences']+cfg['evaluation_sequences']:
            cache=read(PARENT/(seq+'_score_cache.json'));rows=cache['frames'];reader=DSECPilotSequence(ROOT/'datasets/dsec_pilot',seq)
            metadata.append(dict(sequence=seq,inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in [reader.event_path,reader.label_path,reader.path/'image_timestamps.txt']]))
            truth={int(t):ground_truth(reader.labels[reader.labels['t']==t],cfg['class_map_dsec_to_detector'],cfg['small_area_fraction'])[0] for t in np.unique(reader.labels['t'])}
            preds={v:[] for v in cfg['variants']}
            for i,row in enumerate(rows):
                t=row['time_us'];array,_=reader.signed_stack(t,**cfg['input']);x=torch.from_numpy(array).cuda()
                for variant,net in model.items():
                    output=net(x);boxes=output[3].clamp(0,1)[0].cpu().numpy();scores=(output[1].softmax(-1)*output[0].sigmoid())[0].cpu().numpy()
                    preds[variant].append((boxes,scores))
                    if i in [0,len(rows)//2,len(rows)-1]:
                        recorder=Recorder()
                        with recorder.install(net):a=net(x)
                        second=Recorder()
                        with second.install(net),differentiable_fixed_bn():b=net(x)
                        raw=compare_arrays(flatten(a),flatten(b),0,0);layers=compare_traces(recorder.values,second.values,0,0)
                        alignment.append(dict(sequence=seq,variant=variant,time_us=t,raw=raw,layers=layers))
                        if not raw['passed'] or not layers['passed']:raise ValueError('Trained original vs differentiable full-graph mismatch')
                        if variant=='original':
                            p=ROOT/row['path']
                            if sha(p)!=row['sha256']:raise ValueError('Old raw cache changed')
                            with np.load(p) as old:
                                if not np.array_equal(old['boxes'],boxes) or not np.array_equal(old['scores'],scores):raise ValueError('Original checkpoint raw predictions changed')
                if (i+1)%100==0:print('infer',seq,i+1,'/',len(rows),flush=True)
            reader.close();summary[seq]={}
            for variant in cfg['variants']:
                folder=OUT/'predictions';folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{seq}_{variant}.npz',times=np.array([r['time_us'] for r in rows]),boxes=np.stack([p[0] for p in preds[variant]]),scores=np.stack([p[1] for p in preds[variant]]))
                def metrics(theta,record=False):
                    totals=dict(frames=len(rows),gt=0,tp=0,fp=0,small_gt=0,small_tp=0);fps=[]
                    for row,(boxes,scores) in zip(rows,preds[variant]):
                        gt=truth.get(row['time_us'],[]);d=native_detections(filter_anchors(boxes,scores,mask,3,theta,cfg['nms_threshold'],nms))
                        matched=match_frame(d,gt,cfg['iou_threshold'],theta);used=set(matched.values())
                        fps.extend(v.score for j,v in enumerate(d) if j not in used)
                        r=dict(sequence=seq,variant=variant,time_us=row['time_us'],threshold=theta,gt=len(gt),tp=len(matched),fp=len(d)-len(matched),small_gt=sum(g.key for g in gt),small_tp=sum(gt[k].key for k in matched),matched_ids=sorted(gt[k].track_id for k in matched))
                        for k in totals:
                            if k!='frames':totals[k]+=r[k]
                        if record:log.write(json.dumps(r)+'\n')
                    totals.update(recall=totals['tp']/totals['gt'] if totals['gt'] else None,small_recall=totals['small_tp']/totals['small_gt'] if totals['small_gt'] else None,fppi=totals['fp']/totals['frames'])
                    return totals,fps
                if seq in cfg['calibration_sequences']:
                    _,fps=metrics(cfg['calibration_floor']);thresholds[variant]=threshold_at_fppi(fps,len(rows),cfg['calibration_target_fppi'],cfg['calibration_floor'])
                fixed,_=metrics(cfg['confidence_fixed']);cal,_=metrics(thresholds[variant],True)
                if seq in cfg['calibration_sequences'] and cal['fppi']>cfg['calibration_target_fppi']+1e-12:raise ValueError('Calibration guard failed')
                summary[seq][variant]=dict(fixed=fixed,calibrated=cal,threshold=thresholds[variant])
                print('measured',seq,variant,'recall',cal['recall'],'small',cal['small_recall'],'FPPI',cal['fppi'],flush=True)
            if seq in cfg['calibration_sequences']:save(OUT/'frozen_thresholds.json',dict(thresholds=thresholds,actual=summary[seq],frozen_before_other_sequences=True))
            save(OUT/'evaluation_progress.json',summary);save(OUT/'alignment.json',alignment)
    save(OUT/'evaluation_summary.json',dict(config=cfg,conditions=summary,thresholds=thresholds,input_manifests=metadata))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','cache','train','evaluate'],default='all');args=parser.parse_args()
    cfg=read(CONFIG);freeze(cfg)
    if args.phase in ('all','cache'):cache_training(cfg)
    if args.phase in ('all','train'):train(cfg)
    if args.phase in ('all','evaluate'):evaluate(cfg)

if __name__=='__main__':main()
