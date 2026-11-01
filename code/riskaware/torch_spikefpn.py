"""Pinned upstream SpikeFPN and an inference-only pre-smooth execution adapter.

The official implementation remains unmodified in vendor/spikefpn_snapshot.
This adapter currently supports the audited three-scale, batch-one GAD graph.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor/spikefpn_snapshot"
SCALES = (8, 16, 32)
NAMES = {8: ("smooth3", "head_det_3"), 16: ("smooth2", "head_det_2"),
         32: ("smooth1", "head_det_1")}


def load_official_model(device="cuda"):
    receipt = json.loads((SOURCE / "_UPSTREAM.json").read_text())
    checkpoint = SOURCE / "weights/gad_weight.pth"
    expected = next(item["sha256"] for item in receipt["files"] if item["path"] == "weights/gad_weight.pth")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint differs from the verified official snapshot")
    sys.path.insert(0, str(SOURCE))
    from models.architecture import SpikeFPN_GAD
    from utils.spikefpn_config import spikefpn_cfg

    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = SpikeFPN_GAD(device=torch.device(device), input_size=256, num_classes=2,
                        cfg=spikefpn_cfg, center_sample=False, init_channels=3, time_steps=3)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # The published checkpoint omits encoder/lsnn members that GAD.forward
    # does not use. Refuse any missing live key, rather than copying the
    # original test script's unrestricted strict=False load.
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    if incompatible.unexpected_keys or any(
        not key.startswith(("encoder.", "lsnn.")) for key in missing
    ):
        raise RuntimeError("Checkpoint has missing live keys or unexpected keys: " + str(incompatible))

    def reject_inactive_call(module, args):
        raise RuntimeError("A checkpoint-omitted, inactive module was unexpectedly executed")

    model.encoder.register_forward_pre_hook(reject_inactive_call)
    model.lsnn.register_forward_pre_hook(reject_inactive_call)
    model.set_mem_keys(["feature.cells[%d]" % i for i in range(len(model.feature.cells))])
    return model.to(device).eval(), dict(
        upstream_commit=receipt["commit"], checkpoint_sha256=expected,
        checkpoint_load="weights_only=True; all live keys required; omitted inactive members guarded",
        missing_inactive_keys=missing, unexpected_keys=list(incompatible.unexpected_keys),
        original_checkpoint_caveat="Unqualified strict=True fails because unused encoder/lsnn members are absent",
        input_size=256,
        time_steps=3, channels=3, precision="float32; TF32 disabled",
        source_debug_copy="Original stem0.detach().cpu().numpy() retained in all shared passes",
    )


@contextmanager
def bypass_smoothing(model):
    """Expose pre-smooth tensors without executing convolution or neuron updates."""
    saved = {}
    try:
        for smooth, _ in NAMES.values():
            module = getattr(model.feature, smooth)
            saved[smooth] = module.forward
            module.forward = lambda x, param: x
        yield
    finally:
        for name, forward in saved.items():
            getattr(model.feature, name).forward = forward


def cache_shared(model, inputs):
    if inputs.shape[0] != 1 or inputs.shape[1:3] != (3, 3):
        raise ValueError("Audited adapter requires batch one and 3 x 3 temporal channels")
    model.clear_mem()
    cached = {s: [] for s in SCALES}
    with bypass_smoothing(model):
        for q in range(model.time_steps):
            params = dict(mixed_at_mem=True, left_or_right="left", is_first=q == 0)
            coarse, middle, fine = model.feature(inputs[:, q], params)
            for stride, value in ((8, fine), (16, middle), (32, coarse)):
                cached[stride].append(value)
    return cached


def unpack_heads(model, heads, indices=None):
    objs, classes, regs, boxes = [], [], [], []
    anchors, count = model.num_anchors, model.num_classes
    for level, stride in enumerate(SCALES):
        logits = heads[stride]  # B,K,A(C+5)
        batch, sites, _ = logits.shape
        obj = logits[..., :anchors].reshape(batch, sites * anchors, 1)
        cls = logits[..., anchors:anchors * (1 + count)].reshape(batch, sites * anchors, count)
        reg = logits[..., anchors * (1 + count):].reshape(batch, sites, anchors, 4)
        if indices is None:
            idx = torch.arange(sites, device=logits.device)
        else:
            idx = indices[stride]
        grid = model.grid_xy[level][:, idx]
        anchor_wh = model.anchor_wh[level][:, idx]
        if model.center_sample:
            xy = (reg[..., :2].sigmoid() * 2 - 1 + grid) * stride
        else:
            xy = (reg[..., :2].sigmoid() + grid) * stride
        wh = reg[..., 2:].exp() * anchor_wh
        box = torch.cat((xy - wh / 2, xy + wh / 2), -1).reshape(batch, sites * anchors, 4)
        objs.append(obj)
        classes.append(cls)
        regs.append(reg)
        boxes.append(box / model.input_size)
    return tuple(torch.cat(values, dim=1) for values in (objs, classes, regs, boxes))


def replay_dense(model, cached, capture=False):
    heads, spikes = {}, {}
    for stride in SCALES:
        smooth_name, head_name = NAMES[stride]
        smooth = getattr(model.feature, smooth_name)
        history = []
        for q, value in enumerate(cached[stride]):
            out = smooth(value, dict(mixed_at_mem=False, is_first=q == 0))
            if capture:
                history.append(out.detach().clone())
        raw = getattr(model, head_name)(out)
        heads[stride] = raw.permute(0, 2, 3, 1).reshape(raw.shape[0], -1, raw.shape[1])
        spikes[stride] = history
    return unpack_heads(model, heads), heads, spikes


def replay_selected(model, cached, indices, capture=False):
    """Compute selected output dot products; never compute a full smoothing map."""
    heads, histories, counters = {}, {}, {}
    for stride in SCALES:
        smooth_name, head_name = NAMES[stride]
        module, head = getattr(model.feature, smooth_name), getattr(model, head_name)
        conv = fuse_conv_bn_eval(module.conv1, module.bn)
        if conv.kernel_size != (3, 3) or conv.stride != (1, 1) or conv.groups != 1:
            raise ValueError("Unaudited convolution geometry")
        values = cached[stride]
        batch, in_channels, height, width = values[0].shape
        idx = indices[stride]
        if idx.ndim != 1 or len(torch.unique(idx)) != len(idx) or (
            len(idx) and (int(idx.min()) < 0 or int(idx.max()) >= height * width)
        ):
            raise ValueError("Invalid selected coordinates")
        sites, out_channels = len(idx), conv.out_channels
        rows, cols = idx // width, idx % width
        offsets_y = torch.arange(3, device=idx.device).repeat_interleave(3)
        offsets_x = torch.arange(3, device=idx.device).repeat(3)
        mem = torch.zeros((batch, sites, out_channels), device=values[0].device, dtype=values[0].dtype)
        history = []
        for value in values:
            padded = F.pad(value, (1, 1, 1, 1))
            patches = padded[:, :, rows[:, None] + offsets_y, cols[:, None] + offsets_x]
            patches = patches.permute(0, 2, 1, 3).reshape(batch, sites, in_channels * 9)
            current = F.linear(patches, conv.weight.reshape(out_channels, -1), conv.bias)
            charge = mem + current
            spike = (charge > 0.3).to(charge.dtype)
            mem = charge * 0.2 * (1 - spike)
            if capture:
                history.append(spike.detach().clone())
        heads[stride] = F.linear(spike, head.weight[:, :, 0, 0], head.bias)
        histories[stride] = history
        counters[stride] = dict(
            selected_sites=sites, total_sites=height * width,
            nominal_conv_macs=len(values) * sites * out_channels * in_channels * 9,
            nominal_readout_macs=sites * out_channels * head.out_channels,
            neuron_updates=len(values) * sites * out_channels,
            scope="Branch dot-product slots; complete pipeline and memory are separate",
        )
    return unpack_heads(model, heads, indices), heads, histories, counters


def postprocess(model, outputs, threshold=.3):
    from utils.tools import get_box_score
    return get_box_score(outputs[0], outputs[1], outputs[3], num_classes=model.num_classes,
                         conf_thresh=threshold, nms_thresh=.5)
