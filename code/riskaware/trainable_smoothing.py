"""Frozen-BN folding that keeps gradients to the original smoothing weights."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .torch_spikefpn import NAMES, SCALES, unpack_heads


def folded_tensors(module):
    conv,bn=module.conv1,module.bn
    if bn.training: raise ValueError('This adapter requires fixed BN statistics')
    if bn.running_mean is None or bn.running_var is None: raise ValueError('BN running buffers required')
    inv=torch.rsqrt(bn.running_var+bn.eps)
    bw=bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
    bb=bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
    cb=conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    weight=(conv.weight*(bw*inv).reshape(-1,1,1,1)).to(conv.weight.dtype)
    bias=((cb-bn.running_mean)*inv*bw+bb).to(cb.dtype)
    return weight,bias


def smooth_sequence(module,values,return_history=False):
    """values: B,Q,C,H,W. State resets per window, persists over internal Q."""
    if values.ndim!=5 or values.shape[1]<1: raise ValueError('Expected nonempty B,Q,C,H,W')
    weight,bias=folded_tensors(module)
    conv=module.conv1
    mem=None;history=[]
    for q in range(values.shape[1]):
        current=F.conv2d(values[:,q],weight,bias,conv.stride,conv.padding,conv.dilation,conv.groups)
        mem=current if mem is None else mem+current
        spike=module.act_fun(mem,module.b)
        mem=mem*.2*(1.-spike)
        if return_history: history.append(spike)
    return (spike,history,mem) if return_history else spike


def forward_from_presmooth(model,batch):
    heads={}
    for s in SCALES:
        smooth,head=NAMES[s]
        spikes=smooth_sequence(getattr(model.feature,smooth),batch[s])
        raw=getattr(model,head)(spikes)
        heads[s]=raw.permute(0,2,3,1).reshape(raw.shape[0],-1,raw.shape[1])
    return unpack_heads(model,heads)


def is_trainable(name,cfg):
    return name.split('.')[0] in cfg['trainable_modules'] or any(name.startswith(p) for p in cfg['additional_trainable_prefixes'])
