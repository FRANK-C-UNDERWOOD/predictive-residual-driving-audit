"""Fixed multi-group training schedule; validation selection precedes calibration."""
from pathlib import Path
import sys,json,time,gc,hashlib
import numpy as np
import torch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import run_backbone_adaptation as base
from riskaware.trainable_backbone import configure,differentiable_fixed_bn,tensor_hash,accumulation_weight
from riskaware.dsec_pilot import DSECPilotSequence
from riskaware.readout_adaptation import training_labels
from riskaware.criterion_audit import ground_truth,filter_anchors,native_detections
from riskaware.metrics import match_frame
from riskaware.calibration_audit import threshold_at_fppi
OUT=ROOT/'experiments/diverse_adaptation_v01';CONFIG=ROOT/'code/configs/diverse_adaptation_v01.json'
read,save,sha=base.read,base.save,base.sha

def prepare(cfg):
    if (OUT/'input_manifest.json').exists():
        for item in read(OUT/'input_manifest.json')['files']:
            if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Cached training/validation input changed')
        return
    model,_=base.load_official_model()
    from utils.create_labels import gt_creator
    allrows=[];counts={};validation=[];alignment=[]
    for seq in cfg['train_sequences']+cfg['validation_sequences']:
        reader=DSECPilotSequence(ROOT/'datasets/dsec_pilot',seq)
        grid=np.loadtxt(reader.path/'image_timestamps.txt',dtype=np.int64)
        times=[int(t) for t in grid if int(t)-180000>=reader.offset_us+reader.first_local_us and int(t)<=reader.offset_us+reader.last_local_us][10:]
        counts[seq]=dict(queries=len(times),supported=0,ignored=0,resized_tiny=0)
        p=OUT/(seq+'_inputs.npy');arr=np.lib.format.open_memmap(p,mode='w+',dtype=np.int8,shape=(len(times),3,3,256,256));targets=[]
        for i,t in enumerate(times):
            x,_=reader.signed_stack(t,**cfg['input']);arr[i]=x[0]
            labs,c=training_labels(reader.labels[reader.labels['t']==t],cfg['class_map_dsec_to_detector'])
            for k in c:counts[seq][k]+=c[k]
            if seq in cfg['train_sequences']:targets.append(gt_creator(256,model.stride,[labs],model.anchor_list,multi_anchor=True)[0].numpy())
            if i in [0,len(times)//2,len(times)-1]:
                with torch.no_grad():
                    xx=torch.from_numpy(x).cuda();a=model(xx)
                    with differentiable_fixed_bn():b=model(xx)
                exact=all(torch.equal(u,v) for u,v in zip(a,b));alignment.append(dict(sequence=seq,time_us=t,exact=exact))
                if not exact:raise ValueError('New-group input alignment failed')
            if (i+1)%100==0:print('prepare',seq,i+1,'/',len(times),flush=True)
        arr.flush();del arr;np.save(OUT/(seq+'_times.npy'),times)
        if targets:
            np.save(OUT/(seq+'_targets.npy'),np.stack(targets));allrows.extend([(seq,i) for i in range(len(times))])
        else:validation.extend([(seq,i) for i in range(len(times))])
        reader.close()
    generator=torch.Generator().manual_seed(cfg['seed']);schedule=[];pool=[]
    while len(schedule)<cfg['updates']:
        while len(pool)<cfg['effective_batch_size']:pool.extend(torch.randperm(len(allrows),generator=generator).tolist())
        schedule.append(pool[:cfg['effective_batch_size']]);pool=pool[cfg['effective_batch_size']:]
    np.save(OUT/'schedule.npy',schedule)
    files=list(OUT.glob('*_inputs.npy'))+list(OUT.glob('*_targets.npy'))+list(OUT.glob('*_times.npy'))+[OUT/'schedule.npy']
    save(OUT/'input_manifest.json',dict(training_rows=allrows,validation_rows=validation,counts=counts,alignment=alignment,
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files]))
    del model;gc.collect();torch.cuda.empty_cache()

def validate(net,cfg,variant,update):
    from utils.tools import nms
    allfp=[];alltp=[];smalltp=[];ngt=ns=nframes=0
    for seq in cfg['validation_sequences']:
        times=np.load(OUT/(seq+'_times.npy'));x=np.load(OUT/(seq+'_inputs.npy'),mmap_mode='r')
        labels=np.load(ROOT/'datasets/dsec_pilot/train'/seq/'object_detections/left/tracks.npy')
        preds=[]
        with torch.no_grad():
            for i,t in enumerate(times):
                output=net(torch.from_numpy(np.array(x[i:i+1],dtype=np.float32)).cuda())
                boxes=output[3].clamp(0,1)[0].cpu().numpy();scores=(output[1].softmax(-1)*output[0].sigmoid())[0].cpu().numpy();preds.append((boxes,scores))
                gt,_=ground_truth(labels[labels['t']==t],cfg['class_map_dsec_to_detector'],cfg['small_area_fraction'])
                det=native_detections(filter_anchors(boxes,scores,np.ones(1344,bool),3,.3,.5,nms));matched=match_frame(det,gt,.5,.3);rev={di:gi for gi,di in matched.items()}
                ngt+=len(gt);ns+=sum(g.key for g in gt);nframes+=1
                for j,d in enumerate(det):
                    if j in rev:
                        alltp.append(d.score)
                        if gt[rev[j]].key:smalltp.append(d.score)
                    else:allfp.append(d.score)
        folder=OUT/'validation_predictions';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{seq}_{variant}_{update}.npz',times=times,boxes=np.stack([v[0] for v in preds]),scores=np.stack([v[1] for v in preds]))
    theta=threshold_at_fppi(allfp,nframes,1.,.3);tp=sum(s>=theta for s in alltp);fp=sum(s>=theta for s in allfp)
    if fp>nframes:raise ValueError('Validation budget guard')
    actual_tp=actual_fp=0
    for seq in cfg['validation_sequences']:
        labels=np.load(ROOT/'datasets/dsec_pilot/train'/seq/'object_detections/left/tracks.npy')
        with np.load(OUT/'validation_predictions'/f'{seq}_{variant}_{update}.npz') as data:
            for t,b,s in zip(data['times'],data['boxes'],data['scores']):
                gt,_=ground_truth(labels[labels['t']==t],cfg['class_map_dsec_to_detector'],cfg['small_area_fraction'])
                det=native_detections(filter_anchors(b,s,np.ones(1344,bool),3,theta,.5,nms));matched=match_frame(det,gt,.5,theta)
                actual_tp+=len(matched);actual_fp+=len(det)-len(matched)
    if (actual_tp,actual_fp)!=(tp,fp):raise ValueError('Validation final NMS/prefix mismatch')
    result=dict(variant=variant,update=update,tp=tp,fp=fp,gt=ngt,queries=nframes,threshold=theta,recall=tp/ngt if ngt else 0,
                small_tp=sum(s>=theta for s in smalltp),small_gt=ns,scope='Internal validation selection, not held-out results')
    save(OUT/f'{variant}_validation_{update}.json',result);print('validation',result,flush=True);return result

def train(cfg):
    manifest=read(OUT/'input_manifest.json');rows=manifest['training_rows'];schedule=np.load(OUT/'schedule.npy')
    arrays={s:np.load(OUT/(s+'_inputs.npy'),mmap_mode='r') for s in cfg['train_sequences']}
    target={s:np.load(OUT/(s+'_targets.npy'),mmap_mode='r') for s in cfg['train_sequences']}
    records=[]
    for variant in ['readout','full_conv']:
        if (OUT/(variant+'_training.json')).exists():records.append(read(OUT/(variant+'_training.json')));continue
        model,_=base.load_official_model();params=configure(model,variant)
        from utils.criterion import build_criterion
        from utils.spikefpn_config import spikefpn_cfg
        from utils.tools import calculate_loss_new
        criterion=build_criterion(SimpleNamespace(scale_loss='positive'),spikefpn_cfg,2)
        frozen=tensor_hash({k:v for k,v in model.state_dict().items() if k not in params})
        groups=[dict(params=[p for k,p in params.items() if k.startswith('head_det_')],lr=cfg['head_lr'])]
        if variant=='full_conv':groups.append(dict(params=[p for k,p in params.items() if k.startswith('feature.')],lr=cfg['feature_conv_lr']))
        opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']);logs=[];valid=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for update,ids in enumerate(schedule,1):
            selected=[rows[int(i)] for i in ids];y=torch.from_numpy(np.stack([target[s][i] for s,i in selected])).cuda()
            opt.zero_grad(set_to_none=True);values=np.zeros(4)
            for begin in range(0,len(ids),cfg['micro_batch_size']):
                sub=selected[begin:begin+cfg['micro_batch_size']];x=torch.from_numpy(np.stack([arrays[s][i] for s,i in sub]).astype(np.float32)).cuda();yt=y[begin:begin+len(sub)]
                with differentiable_fixed_bn():
                    o=model(x);losses=calculate_loss_new(o[0],o[1],o[3],yt,criterion);w=accumulation_weight(yt,y)
                    if not all(torch.isfinite(z) for z in losses):raise ValueError('Nonfinite loss')
                    (losses[-1]*w).backward()
                values+=np.array([float(z.detach()*w) for z in losses]);del o,losses,x
            if update==1 and any(p.grad is None for p in params.values()):raise ValueError('Missing gradient')
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),cfg['gradient_clip_norm'])
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
            opt.step();logs.append(dict(update=update,losses=values.tolist(),sample_ids=ids.tolist()))
            if update%20==0:print(variant,update,'/',cfg['updates'],'seconds',round(time.monotonic()-start,1),flush=True)
            if update in cfg['checkpoint_updates']:
                if tensor_hash({k:v for k,v in model.state_dict().items() if k not in params})!=frozen:raise ValueError('Frozen state drift')
                torch.save(dict(parameters={k:v.detach().cpu() for k,v in model.state_dict().items() if k in params},variant=variant,update=update),OUT/f'{variant}_update{update}.pth')
                valid.append(validate(model,cfg,variant,update));save(OUT/(variant+'_progress.json'),dict(logs=logs,validation=valid))
        chosen=sorted(valid,key=lambda r:(-r['recall'],r['update']))[0];path=OUT/f'{variant}_update{chosen["update"]}.pth'
        record=dict(variant=variant,updates=cfg['updates'],selected_update=chosen['update'],selection=chosen,validation=valid,
            checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),schedule_sha256=sha(OUT/'schedule.npy'),
            trainable_parameters=sum(p.numel() for p in params.values()),frozen_hash_before=frozen,frozen_hash_after=tensor_hash({k:v for k,v in model.state_dict().items() if k not in params}),
            seconds=time.monotonic()-start,logs=logs,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(OUT/(variant+'_training.json'),record);records.append(record);del model,opt,params;gc.collect();torch.cuda.empty_cache()
    save(OUT/'training_summary.json',dict(config=cfg,variants=records))

def selected_model(variant,cfg):
    model,_=base.load_official_model()
    if variant!='original':
        r=read(OUT/(variant+'_training.json'));p=ROOT/r['checkpoint']
        if sha(p)!=r['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        state=torch.load(p,map_location='cpu',weights_only=True)['parameters'];expected=set(configure(model,variant))
        if set(state)!=expected:raise ValueError('Selected keys mismatch')
        with torch.no_grad():
            for k,v in state.items():model.state_dict()[k].copy_(v)
    for p in model.parameters():p.requires_grad_(False)
    return model

def main():
    cfg=read(CONFIG)
    for role in ['train_sequences','validation_sequences','calibration_sequences','evaluation_sequences']:
        for other in ['train_sequences','validation_sequences','calibration_sequences','evaluation_sequences']:
            if role!=other and set(cfg[role])&set(cfg[other]):raise ValueError('Sequence role overlap')
    names=['code/scripts/run_diverse_adaptation.py','code/configs/diverse_adaptation_v01.json','code/riskaware/trainable_backbone.py','code/scripts/run_backbone_adaptation.py']
    receipt=dict(config=cfg,sources={p:sha(ROOT/p) for p in names},data_receipt_sha256=sha(OUT/'data_receipt.json'),parent_preflight_sha256=sha(ROOT/'experiments/backbone_adaptation_v01/preflight.json'))
    p=OUT/'run_manifest.json'
    if p.exists() and read(p)!=receipt:raise ValueError('Frozen run changed')
    save(p,receipt);save(OUT/'STATUS.json',dict(stage='preparing_inputs',training_started=False,completed_data_receipt=True));prepare(cfg)
    if not (OUT/'original_validation_0.json').exists():
        original,_=base.load_official_model();validate(original,cfg,'original',0);del original;gc.collect();torch.cuda.empty_cache()
    save(OUT/'STATUS.json',dict(stage='training_and_internal_validation',training_started=True,completed_data_receipt=True));train(cfg)
    save(OUT/'STATUS.json',dict(stage='selected_model_calibration_and_evaluation',training_started=True,completed_data_receipt=True))
    base.OUT=OUT;base.CONFIG=CONFIG;base.load_variant=selected_model;base.evaluate(cfg)
    save(OUT/'STATUS.json',dict(stage='awaiting_independent_reconstruction',training_started=True,completed_data_receipt=True))

if __name__=='__main__':main()
