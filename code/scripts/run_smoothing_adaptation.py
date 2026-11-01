"""Joint smoothing/readout adaptation; reuse the frozen measurement protocol."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import run_readout_adaptation as base
from riskaware.trainable_smoothing import forward_from_presmooth,is_trainable

CONFIG=ROOT/'code/configs/smoothing_adaptation_v01.json'
OUT=ROOT/'experiments/smoothing_adaptation_v01'
PRIOR=ROOT/'experiments/readout_adaptation_v01'
sha,read,save=base.sha,base.read,base.save


def stage_receipt(stage,cfg):
    OUT.mkdir(parents=True,exist_ok=True)
    cache=read(OUT/'cache_run_manifest.json')
    if cache['config_sha256']!=sha(CONFIG): raise ValueError('Configuration changed after cache extraction')
    if any(sha(ROOT/p)!=h for p,h in cache['sources'].items()): raise ValueError('Cache implementation changed')
    source_names=['code/scripts/run_smoothing_adaptation.py','code/riskaware/trainable_smoothing.py',
        'code/scripts/run_readout_adaptation.py','code/riskaware/readout_adaptation.py',
        'code/riskaware/protocol_eval.py','code/riskaware/criterion_audit.py','code/riskaware/calibration_audit.py']
    receipt=dict(config=cfg,config_sha256=sha(CONFIG),sources={p:sha(ROOT/p) for p in source_names},
        prior_artifact_sha256=sha(PRIOR/'artifact_manifest.json'))
    p=OUT/(stage+'_run_manifest.json')
    if p.exists() and read(p)!=receipt: raise ValueError('Stage source/config changed: '+stage)
    save(p,receipt)


def load_frames(cfg,seq):
    data=read(OUT/(seq+'_cache.json'))
    if not data['complete']: raise ValueError('Incomplete cache: '+seq)
    return data['frames'][cfg['warmup_complete_windows']:]


def wait_for(seq):
    path=OUT/(seq+'_cache.json');start=time.monotonic();last=-100
    while True:
        try:
            if path.exists() and read(path)['complete']: return
        except json.JSONDecodeError: pass
        if time.monotonic()-start>7200: raise TimeoutError(seq)
        if time.monotonic()-start-last>=30:
            print('Waiting for pre-smooth cache: '+seq,flush=True);last=time.monotonic()-start
        time.sleep(5)


def load_training(cfg,model):
    from utils.create_labels import gt_creator
    arrays={s:[] for s in base.SCALES};targets=[];counts=dict(supported=0,ignored=0,resized_tiny=0)
    for seq in cfg['train_sequences']:
        labels=base.labels_by_time(seq)
        for frame in load_frames(cfg,seq):
            p=ROOT/frame['path']
            if sha(p)!=frame['sha256']: raise ValueError('Cache modified')
            with np.load(p,allow_pickle=False) as data:
                for s in base.SCALES: arrays[s].append(torch.from_numpy(data['z'+str(s)].copy()))
            converted,c=base.training_labels(labels.get(frame['time_us'],[]),cfg['class_map_dsec_to_detector'],cfg['small_area_fraction'])
            for k in counts: counts[k]+=c[k]
            targets.append(gt_creator(256,model.stride,[converted],model.anchor_list,multi_anchor=True)[0])
    return {s:torch.stack(v).cuda() for s,v in arrays.items()},torch.stack(targets).cuda(),counts


def configure_gradients(model,cfg):
    for name,p in model.named_parameters(): p.requires_grad_(is_trainable(name,cfg))
    model.eval()
    return {k:v for k,v in model.state_dict().items() if not is_trainable(k,cfg)}


def gradient_groups(model,cfg):
    names=cfg['additional_trainable_prefixes']+[n+'.' for n in cfg['trainable_modules']]
    groups={prefix:sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
        if name.startswith(prefix) and p.grad is not None) for prefix in names}
    frozen=sum(p.grad is not None for name,p in model.named_parameters() if not is_trainable(name,cfg))
    if frozen or any(not np.isfinite(v) or v<=0 for v in groups.values()): raise ValueError('Incorrect gradient routing')
    return dict(gradient_absolute_sums=groups,frozen_gradient_count=frozen)


def train(cfg):
    for seq in cfg['train_sequences']: wait_for(seq)
    stage_receipt('train',cfg)
    model,provenance=base.load_official_model()
    arrays,target,counts=load_training(cfg,model);n=len(target)
    from utils.criterion import build_criterion
    from utils.spikefpn_config import spikefpn_cfg
    from utils.tools import calculate_loss_new
    criterion=build_criterion(SimpleNamespace(scale_loss='positive'),spikefpn_cfg,num_classes=2)
    checkpoints=[]
    for seed in cfg['seeds']:
        model,_=base.load_official_model();torch.manual_seed(seed)
        frozen=configure_gradients(model,cfg);before=base.state_hash(frozen)
        head_params=[p for name,p in model.named_parameters() if name.split('.')[0] in cfg['trainable_modules']]
        smooth_params=[p for name,p in model.named_parameters() if any(name.startswith(prefix) for prefix in cfg['additional_trainable_prefixes'])]
        params=head_params+smooth_params
        optimizer=torch.optim.AdamW([dict(params=head_params,lr=cfg['learning_rate']),
                                    dict(params=smooth_params,lr=cfg['smoothing_learning_rate'])],weight_decay=cfg['weight_decay'])
        generator=torch.Generator().manual_seed(seed);log=[];start=time.monotonic();updates=0
        probe=None;torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,cfg['epochs']+1):
            order=torch.randperm(n,generator=generator).tolist();sums=np.zeros(4);seen=0;maximum_grad=0
            for begin in range(0,n,cfg['batch_size']):
                ids=order[begin:begin+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                outputs=forward_from_presmooth(model,{s:arrays[s][ids].float() for s in base.SCALES})
                losses=calculate_loss_new(outputs[0],outputs[1],outputs[3],target[ids],criterion)
                if not all(torch.isfinite(v) for v in losses): raise ValueError('Nonfinite loss; no silent skipping')
                losses[-1].backward()
                if probe is None: probe=gradient_groups(model,cfg)
                grad=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip_norm'])
                if not torch.isfinite(grad): raise ValueError('Nonfinite gradient')
                maximum_grad=max(maximum_grad,float(grad));optimizer.step();updates+=1
                sums+=np.array([float(v.detach()) for v in losses])*len(ids);seen+=len(ids)
            after=base.state_hash({k:v for k,v in model.state_dict().items() if not is_trainable(k,cfg)})
            if before!=after: raise ValueError('Frozen parameters/BN state changed')
            log.append(dict(epoch=epoch,n=seen,total=float(sums[3]/seen),objectness=float(sums[0]/seen),
                classification=float(sums[1]/seen),box=float(sums[2]/seen),updates=updates,
                order_sha256=hashlib.sha256(np.array(order,dtype='<i8').tobytes()).hexdigest(),max_unclipped_gradient=maximum_grad))
            if epoch in (1,4,cfg['epochs']):
                torch.save(dict(seed=seed,epoch=epoch,parameters={k:v.detach().cpu() for k,v in model.state_dict().items()
                    if is_trainable(k,cfg)}),OUT/f'joint_seed{seed}_epoch{epoch}.pth')
            save(OUT/f'train_seed{seed}_progress.json',dict(seed=seed,loss_trace=log,gradient_probe=probe))
            print(f'joint train seed {seed}, epoch {epoch}: loss {log[-1]["total"]:.4f}',flush=True)
        path=OUT/f'joint_seed{seed}_epoch{cfg["epochs"]}.pth'
        record=dict(seed=seed,epochs=cfg['epochs'],n_windows=n,trainable_parameters=sum(p.numel() for p in params),
            trainable_smoothing_parameters=sum(p.numel() for p in smooth_params),gradient_probe=probe,
            frozen_hash_before=before,frozen_hash_after=after,training_label_counts=counts,updates=updates,
            checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),loss_trace=log,
            training_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),model=provenance)
        save(OUT/f'train_seed{seed}.json',record);checkpoints.append(record)
    save(OUT/'training_summary.json',dict(config_sha256=sha(CONFIG),checkpoints=checkpoints,
        scope=cfg['seed_scope'],evaluation_labels_used_for_checkpoint_selection=False))


def load_variant(model,variant,cfg):
    if variant=='original':
        state=torch.load(ROOT/'vendor/spikefpn_snapshot/weights/gad_weight.pth',map_location='cpu',weights_only=True)
        state={k:v for k,v in state.items() if is_trainable(k,cfg)}
    else:
        seed=int(variant.removeprefix('adapted_seed'))
        state=torch.load(OUT/f'joint_seed{seed}_epoch{cfg["epochs"]}.pth',map_location='cpu',weights_only=True)['parameters']
    expected={k for k in model.state_dict() if is_trainable(k,cfg)}
    if set(state)!=expected: raise ValueError('Unexpected trained-parameter keys')
    with torch.no_grad():
        for name,value in state.items(): model.state_dict()[name].copy_(value)


def evaluate(cfg):
    stage_receipt('evaluation',cfg)
    # Reuse the exact readout-experiment evaluator, replacing only model loading
    # and forward/cache interfaces. Its adapted_seed keys mean JOINT here.
    base.OUT,base.CONFIG=OUT,CONFIG
    def waiting_frames(config,seq):
        wait_for(seq)
        return load_frames(config,seq)
    base.load_frames=waiting_frames
    base.cached_forward=forward_from_presmooth
    base.load_variant=load_variant
    base.evaluate(cfg)
    result=read(OUT/'evaluation_summary.json')
    result['variant_key_semantics']='adapted_seedN denotes jointly trained smoothing+readout, not the old readout-only model'
    save(OUT/'evaluation_summary.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['train','evaluate','all'],default='all');args=parser.parse_args()
    cfg=read(CONFIG)
    if args.phase in ('train','all'): train(cfg)
    if args.phase in ('evaluate','all'): evaluate(cfg)


if __name__=='__main__': main()
