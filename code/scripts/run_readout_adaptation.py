"""Readout-only adaptation with frozen SNN features and sequence-separated roles."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from riskaware.torch_spikefpn import load_official_model, unpack_heads, SCALES, NAMES
from riskaware.dsec_pilot import DSECPilotSequence
from riskaware.criterion_audit import ground_truth, filter_anchors, native_detections
from riskaware.metrics import match_frame, RunLengthRule
from riskaware.protocol_eval import evaluate_events
from riskaware.calibration_audit import threshold_at_fppi
from riskaware.readout_adaptation import training_labels, eligible_events

OUT=ROOT/'experiments/readout_adaptation_v01'
CONFIG=ROOT/'code/configs/readout_adaptation_v01.json'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,value): p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_manifest(cfg):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'run_manifest.json'
    sources=['code/scripts/run_readout_adaptation.py','code/riskaware/readout_adaptation.py',
             'code/riskaware/dsec_pilot.py','code/riskaware/torch_spikefpn.py',
             'code/riskaware/criterion_audit.py','code/riskaware/calibration_audit.py',
             'code/riskaware/protocol_eval.py','code/riskaware/metrics.py']
    current=dict(config=cfg,config_sha256=sha(CONFIG),sources={p:sha(ROOT/p) for p in sources})
    if path.exists() and read(path)!=current:
        raise ValueError('Refusing to reuse adaptation artifacts after source/config changes')
    save(path,current)


def cached_forward(model,batch):
    heads={}
    for s in SCALES:
        raw=getattr(model,NAMES[s][1])(batch[s])
        heads[s]=raw.permute(0,2,3,1).reshape(raw.shape[0],-1,raw.shape[1])
    return unpack_heads(model,heads)


def cache_sequence(cfg,seq):
    model,provenance=load_official_model()
    if provenance['checkpoint_sha256']!=cfg['checkpoint_sha256']: raise ValueError('Unexpected checkpoint')
    reader=DSECPilotSequence(ROOT/'datasets/dsec_pilot',seq)
    grid=np.loadtxt(reader.path/'image_timestamps.txt',dtype=np.int64)
    folder=OUT/'cache'/seq;folder.mkdir(parents=True,exist_ok=True)
    record_path=OUT/(seq+'_cache.json')
    previous={r['time_us']:r for r in read(record_path)['frames']} if record_path.exists() else {}
    observed={};handles=[]
    for s in SCALES:
        def hook(module,args,s=s): observed[s]=args[0].detach()
        handles.append(getattr(model,NAMES[s][1]).register_forward_pre_hook(hook))
    frames=[];excluded=[];started=time.monotonic()
    with torch.inference_mode():
        for t in grid:
            t=int(t)
            if t-180000<reader.offset_us+reader.first_local_us or t>reader.offset_us+reader.last_local_us:
                excluded.append(t);continue
            p=folder/(str(t)+'.npz')
            if t in previous and p.is_file() and sha(p)==previous[t]['sha256']:
                frames.append(previous[t]);continue
            array,input_record=reader.signed_stack(t,**cfg['input'])
            x=torch.from_numpy(array).cuda()
            original=model(x)
            # Hooks store the input of each pointwise readout, i.e. final-step spikes.
            spikes={s:observed[s].cpu().numpy() for s in SCALES}
            if any(not np.isin(a,[0,1]).all() for a in spikes.values()):
                raise ValueError('Cache requires exact binary final-step features')
            replay=cached_forward(model,{s:torch.from_numpy(spikes[s].astype(np.uint8)).cuda().float() for s in SCALES})
            errors=[float((a-b).abs().max()) for a,b in zip(original,replay)]
            if any(not torch.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(original,replay)):
                raise ValueError('Feature cache changes original outputs')
            np.savez_compressed(p,**{'z'+str(s):a[0].astype(np.uint8) for s,a in spikes.items()})
            frames.append(dict(time_us=t,path=p.relative_to(ROOT).as_posix(),sha256=sha(p),max_output_error=max(errors),
                               raw_events=input_record['raw_events']))
            if len(frames)%25==0:
                save(record_path,dict(sequence=seq,complete_windows=len(frames),frames=frames,
                    skipped_incomplete_windows=excluded,model=provenance,cache_seconds=time.monotonic()-started,
                    complete=False))
                print(f'cache {seq}: {len(frames)} windows, elapsed {time.monotonic()-started:.1f}s',flush=True)
    for handle in handles: handle.remove()
    reader.close()
    save(record_path,dict(sequence=seq,complete_windows=len(frames),frames=frames,
        skipped_incomplete_windows=excluded,model=provenance,cache_seconds=time.monotonic()-started,complete=True))
    print(f'cache {seq}: finished {len(frames)} windows',flush=True)


def load_frames(cfg,seq):
    record=read(OUT/(seq+'_cache.json'))
    if not record['complete']: raise ValueError('Incomplete sequence cache')
    return record['frames'][cfg['warmup_complete_windows']:]


def labels_by_time(seq):
    folder=ROOT/'datasets/dsec_pilot/train'/seq
    labels=np.load(folder/'object_detections/left/tracks.npy',allow_pickle=False)
    return {int(t):labels[labels['t']==t] for t in np.unique(labels['t'])}


def train(cfg):
    model,provenance=load_official_model()
    from utils.create_labels import gt_creator
    from utils.criterion import build_criterion
    from utils.spikefpn_config import spikefpn_cfg
    from utils.tools import calculate_loss_new
    arrays={s:[] for s in SCALES};targets=[];label_counts=dict(supported=0,ignored=0,resized_tiny=0)
    n=0
    for seq in cfg['train_sequences']:
        labels=labels_by_time(seq)
        for frame in load_frames(cfg,seq):
            p=ROOT/frame['path']
            if sha(p)!=frame['sha256']: raise ValueError('Training feature cache modified')
            with np.load(p,allow_pickle=False) as data:
                for s in SCALES: arrays[s].append(torch.from_numpy(data['z'+str(s)].copy()))
            rows=labels.get(frame['time_us'],[])
            converted,counts=training_labels(rows,cfg['class_map_dsec_to_detector'],cfg['small_area_fraction'])
            for k in label_counts: label_counts[k]+=counts[k]
            targets.append(gt_creator(256,model.stride,[converted],model.anchor_list,multi_anchor=True)[0])
            n+=1
    arrays={s:torch.stack(v).cuda() for s,v in arrays.items()}
    target=torch.stack(targets).cuda()
    criterion=build_criterion(SimpleNamespace(scale_loss='positive'),spikefpn_cfg,num_classes=2)
    checkpoints=[]
    for seed in cfg['seeds']:
        model,_=load_official_model()
        torch.manual_seed(seed)
        for name,p in model.named_parameters():
            p.requires_grad_(name.split('.')[0] in cfg['trainable_modules'])
        frozen_before=state_hash({k:v for k,v in model.state_dict().items() if k.split('.')[0] not in cfg['trainable_modules']})
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        generator=torch.Generator().manual_seed(seed)
        log=[];gradient_checked=False;start=time.monotonic()
        for epoch in range(1,cfg['epochs']+1):
            order=torch.randperm(n,generator=generator).tolist()
            sums=np.zeros(4);seen=0;max_grad=0.
            for begin in range(0,n,cfg['batch_size']):
                ids=order[begin:begin+cfg['batch_size']]
                batch={s:arrays[s][ids].float() for s in SCALES}
                optimizer.zero_grad(set_to_none=True)
                outputs=cached_forward(model,batch)
                losses=calculate_loss_new(outputs[0],outputs[1],outputs[3],target[ids],criterion)
                if not all(torch.isfinite(v) for v in losses): raise ValueError('Nonfinite training loss; no sample skipping')
                losses[-1].backward()
                if not gradient_checked:
                    if any(p.grad is not None for name,p in model.named_parameters() if name.split('.')[0] not in cfg['trainable_modules']):
                        raise ValueError('Gradient leaked into frozen model')
                    if not all(any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters())
                               for name in cfg['trainable_modules']): raise ValueError('Missing readout gradients')
                    gradient_checked=True
                grad=torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clip_norm'])
                if not torch.isfinite(grad): raise ValueError('Nonfinite gradient')
                max_grad=max(max_grad,float(grad));optimizer.step()
                sums+=np.array([float(v.detach()) for v in losses])*len(ids);seen+=len(ids)
            log.append(dict(epoch=epoch,n=seen,objectness=float(sums[0]/seen),classification=float(sums[1]/seen),
                box=float(sums[2]/seen),total=float(sums[3]/seen),max_unclipped_gradient_norm=max_grad))
            frozen_after=state_hash({k:v for k,v in model.state_dict().items() if k.split('.')[0] not in cfg['trainable_modules']})
            if frozen_after!=frozen_before: raise ValueError('Frozen parameters or BN buffers changed')
            if epoch in (1,4,cfg['epochs']):
                path=OUT/f'heads_seed{seed}_epoch{epoch}.pth'
                torch.save(dict(seed=seed,epoch=epoch,heads={k:v.detach().cpu() for k,v in model.state_dict().items()
                    if k.split('.')[0] in cfg['trainable_modules']}),path)
            print(f'train seed {seed}, epoch {epoch}: loss {log[-1]["total"]:.4f}',flush=True)
        final=OUT/f'heads_seed{seed}_epoch{cfg["epochs"]}.pth'
        receipt=dict(seed=seed,epochs=cfg['epochs'],n_windows=n,training_label_counts=label_counts,
            trainable_parameters=sum(p.numel() for p in trainable),frozen_hash_before=frozen_before,
            frozen_hash_after=frozen_after,gradients_checked=gradient_checked,checkpoint=final.relative_to(ROOT).as_posix(),
            checkpoint_sha256=sha(final),training_seconds=time.monotonic()-start,loss_trace=log,model=provenance)
        save(OUT/f'train_seed{seed}.json',receipt);checkpoints.append(receipt)
    save(OUT/'training_summary.json',dict(config_sha256=sha(CONFIG),checkpoints=checkpoints,
        scope=cfg['seed_scope'],evaluation_labels_used_for_checkpoint_selection=False))


def load_variant(model,variant,cfg):
    official=torch.load(ROOT/'vendor/spikefpn_snapshot/weights/gad_weight.pth',map_location='cpu',weights_only=True)
    if variant=='original': state={k:v for k,v in official.items() if k.split('.')[0] in cfg['trainable_modules']}
    else:
        seed=int(variant.removeprefix('adapted_seed'))
        state=torch.load(OUT/f'heads_seed{seed}_epoch{cfg["epochs"]}.pth',map_location='cpu',weights_only=True)['heads']
    with torch.no_grad():
        for name,value in state.items(): model.state_dict()[name].copy_(value)


def evaluate(cfg):
    model,provenance=load_official_model()
    from utils.tools import nms
    variants=['original']+['adapted_seed'+str(s) for s in cfg['seeds']]
    scores_dir=OUT/'predictions';scores_dir.mkdir(exist_ok=True)
    summaries={};thresholds={};temporal={};frames_log=[];temporal_details=[]
    # Calibration is completed for every variant before any evaluation metrics.
    for seq in cfg['calibration_sequences']+cfg['evaluation_sequences']:
        calibration=seq in cfg['calibration_sequences']
        records=load_frames(cfg,seq);labels=labels_by_time(seq)
        grid=np.loadtxt(ROOT/'datasets/dsec_pilot/train'/seq/'image_timestamps.txt',dtype=np.int64)
        origin=int(grid[0]);grids={seq:[(int(t)-origin)/1e6 for t in grid]}
        truths={};tracks={};ignored=0
        for t in grid:
            gts,count=ground_truth(labels.get(int(t),[]),cfg['class_map_dsec_to_detector'],cfg['small_area_fraction'])
            truths[int(t)]=gts;ignored+=count
            for g in gts: tracks.setdefault((seq,g.track_id),{})[(int(t)-origin)/1e6]=g
        events,exclusions=eligible_events(tracks,{seq:(records[0]['time_us']-origin)/1e6},
            {seq:(records[-1]['time_us']-origin)/1e6},cfg['horizon_seconds'],cfg['max_annotation_gap_seconds'])
        summaries[seq]={};temporal[seq]={}
        for variant in variants:
            load_variant(model,variant,cfg)
            predictions=[]
            with torch.inference_mode():
                for record in records:
                    p=ROOT/record['path']
                    if sha(p)!=record['sha256']: raise ValueError('Evaluation feature cache modified')
                    with np.load(p,allow_pickle=False) as data:
                        batch={s:torch.from_numpy(data['z'+str(s)].copy()).unsqueeze(0).cuda().float() for s in SCALES}
                    outputs=cached_forward(model,batch)
                    predictions.append((outputs[3].clamp(0,1)[0].cpu().numpy(),
                        (outputs[1].softmax(-1)*outputs[0].sigmoid())[0].cpu().numpy()))
            # These scores are stored before metrics, allowing independent replay.
            np.savez_compressed(scores_dir/(seq+'_'+variant+'.npz'),
                times=np.array([r['time_us'] for r in records],dtype=np.int64),
                boxes=np.stack([p[0] for p in predictions]),scores=np.stack([p[1] for p in predictions]))
            mask=np.ones(1344,dtype=bool)
            def run_at(threshold,collect=False):
                total=dict(frames=len(records),gt=0,tp=0,fp=0,small_gt=0,small_tp=0)
                detections={seq:{}};fp_scores=[];perframe=[]
                for record,(boxes,scores) in zip(records,predictions):
                    f=filter_anchors(boxes,scores,mask,3,threshold,cfg['nms_threshold'],nms)
                    dets=native_detections(f);gts=truths[record['time_us']]
                    match=match_frame(dets,gts,cfg['iou_threshold'],threshold);used=set(match.values())
                    fp_scores.extend(d.score for i,d in enumerate(dets) if i not in used)
                    row=dict(sequence=seq,variant=variant,time_us=record['time_us'],threshold=threshold,
                        gt=len(gts),tp=len(match),fp=len(dets)-len(match),small_gt=sum(g.key for g in gts),
                        small_tp=sum(gts[i].key for i in match))
                    for k in ('gt','tp','fp','small_gt','small_tp'): total[k]+=row[k]
                    detections[seq][(record['time_us']-origin)/1e6]=dets
                    if collect: perframe.append(row)
                total.update(recall=total['tp']/total['gt'] if total['gt'] else None,
                    small_recall=total['small_tp']/total['small_gt'] if total['small_gt'] else None,
                    fppi=total['fp']/total['frames'])
                return total,detections,fp_scores,perframe
            if calibration:
                _,_,fp_scores,_=run_at(cfg['calibration_floor'])
                thresholds[variant]=threshold_at_fppi(fp_scores,len(records),cfg['calibration_target_fppi'],cfg['calibration_floor'])
            summaries[seq][variant]={};temporal[seq][variant]={}
            for endpoint,threshold in (('fixed',cfg['confidence_fixed']),('calibrated',thresholds[variant])):
                total,detections,_,perframe=run_at(threshold,True)
                if calibration and endpoint=='calibrated' and total['fppi']>cfg['calibration_target_fppi']+1e-12:
                    raise ValueError('Exact calibration FPPI guard failed before evaluation')
                result=evaluate_events(events,tracks,detections,grids,horizon=cfg['horizon_seconds'],
                    delta=cfg['deadline_seconds'],rule=RunLengthRule(cfg['reliable_run_length'],cfg['iou_threshold'],threshold),
                    max_annotation_gap=cfg['max_annotation_gap_seconds'])
                summaries[seq][variant][endpoint]=dict(total,threshold=threshold,ignored_source_truth=ignored)
                temporal[seq][variant][endpoint]=dict(n_events=len(result['outcomes']),n_timely=sum(o.timely for o in result['outcomes']),
                    timely_rate=float(result['rate']) if result['outcomes'] else None,
                    finite_delays=[o.latency for o in result['outcomes'] if np.isfinite(o.latency)],exclusions=exclusions)
                for row in perframe: frames_log.append(dict(row,endpoint=endpoint))
                for o in result['outcomes']:
                    temporal_details.append(dict(sequence=seq,variant=variant,endpoint=endpoint,event_id=o.event_id,
                        track_id=o.obj_id,onset=o.onset,timely=o.timely,latency=o.latency if np.isfinite(o.latency) else None))
            print(f'eval {seq}, {variant}: calibrated recall {summaries[seq][variant]["calibrated"]["recall"]:.4f}',flush=True)
        if calibration: save(OUT/'frozen_thresholds.json',dict(thresholds=thresholds,config_sha256=sha(CONFIG),
            note='Every variant calibrated and checked before evaluation sequences; no checkpoint selection'))
    save(OUT/'evaluation_summary.json',dict(config_sha256=sha(CONFIG),scope=cfg['scope'],thresholds=thresholds,
        conditions=summaries,temporal=temporal,limitations=cfg['limitations'],model=provenance))
    for name,rows in (('frames.jsonl',frames_log),('temporal_outcomes.jsonl',temporal_details)):
        with (OUT/name).open('w') as f:
            for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    print('Finished bounded readout adaptation and fixed-protocol evaluation.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['cache','train','evaluate','all'],default='all')
    parser.add_argument('--sequences',nargs='*');args=parser.parse_args()
    cfg=read(CONFIG);freeze_manifest(cfg)
    if args.phase in ('cache','all'):
        for seq in args.sequences or cfg['train_sequences']+cfg['calibration_sequences']+cfg['evaluation_sequences']:
            cache_sequence(cfg,seq)
    if args.phase in ('train','all'): train(cfg)
    if args.phase in ('evaluate','all'): evaluate(cfg)


if __name__=='__main__': main()
