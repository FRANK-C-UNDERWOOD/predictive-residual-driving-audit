"""Autograd-preserving fixed-BN fusion for the pinned SpikeFPN full graph.

Only replace the inference fusion helper inside a scoped context. Original
neuron recurrences, mixed-current branches, decoder and surrogate functions
remain unchanged. Inference/evaluation uses the unmodified upstream forward.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


class FoldedConv:
    def __init__(self,conv,bn):
        if bn.training or conv.training:
            raise ValueError('Only fixed-BN eval-mode folding is supported')
        self.conv,self.bn=conv,bn

    def __call__(self,x):
        c,b=self.conv,self.bn
        scale=b.weight*torch.rsqrt(b.running_var+b.eps)
        weight=(c.weight*scale.reshape(-1,1,1,1)).to(c.weight.dtype)
        bias0=c.bias if c.bias is not None else torch.zeros_like(b.running_mean)
        bias=((bias0-b.running_mean)*torch.rsqrt(b.running_var+b.eps)*b.weight+b.bias).to(bias0.dtype)
        return F.conv2d(x,weight,bias,c.stride,c.padding,c.dilation,c.groups)


@contextmanager
def differentiable_fixed_bn():
    from models import spike_neurons
    previous=spike_neurons.fuse_conv_bn_eval
    spike_neurons.fuse_conv_bn_eval=FoldedConv
    try:yield
    finally:spike_neurons.fuse_conv_bn_eval=previous


def configure(model,variant):
    if variant not in ('readout','full_conv'):raise ValueError(variant)
    model.eval();allowed=set()
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d) and (name.startswith('head_det_') or
            (variant=='full_conv' and name.startswith('feature.'))):
            for p in module.parameters(recurse=False):allowed.add(id(p))
    for p in model.parameters():p.requires_grad_(id(p) in allowed)
    return {name:p for name,p in model.named_parameters() if p.requires_grad}


def tensor_hash(values):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(values.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def accumulation_weight(micro_target,effective_target):
    """Undo each microbatch's clamped denominator, apply the batch denominator.

    Even a zero-positive microbatch contributes its negative-anchor loss.
    Target column zero is positive assignment before calculate_loss_new prepends IoU.
    """
    return micro_target[...,0].sum().clamp(min=1)/effective_target[...,0].sum().clamp(min=1)
