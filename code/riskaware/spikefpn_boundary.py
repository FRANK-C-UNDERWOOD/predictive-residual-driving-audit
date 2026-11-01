"""Executable reference for the SpikeFPN pre-smooth insertion boundary.

Consumes CACHED pre-smooth pyramids; does not implement the upstream backbone.
No torch, trained checkpoint, GPU kernel, detection dataset or energy measurement.
Public source mapping is documented in docs/09_底层算法与门控接入.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

STRIDE_MODULES = {
    8: ("smooth3", "head_det_3", 96, 192),
    16: ("smooth2", "head_det_2", 192, 384),
    32: ("smooth1", "head_det_1", 384, 768),
}


@dataclass(frozen=True)
class BranchWeights:
    conv: np.ndarray       # Cout,Cin,3,3; inference BN already folded
    conv_bias: np.ndarray  # Cout
    readout: np.ndarray    # A*(C+5),Cout
    readout_bias: np.ndarray
    decay: float = 0.2
    threshold: float = 0.3


def fold_batch_norm(weight, bias, mean, variance, gamma, beta, epsilon):
    w = np.asarray(weight, dtype=float)
    scale = np.asarray(gamma) / np.sqrt(np.asarray(variance) + epsilon)
    return w * scale[:, None, None, None], (np.asarray(bias)-mean)*scale+beta


def export_torch_branch(model, stride: int) -> BranchWeights:
    """Weight bridge only. Requires an already loaded, eval-mode upstream model.

    Not exercised against a real torch checkpoint in this workspace.
    """
    smooth_name, head_name, _, _ = STRIDE_MODULES[stride]
    smooth, head = getattr(model.feature, smooth_name), getattr(model, head_name)
    if smooth.training or smooth.bn.training or head.training:
        raise ValueError("The export contract requires evaluation mode.")
    array = lambda value: value.detach().cpu().numpy()
    conv, bn = smooth.conv1, smooth.bn
    if (conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups) != ((3, 3), (1, 1), (1, 1), (1, 1), 1):
        raise ValueError("Upstream smoothing geometry differs from the audited contract.")
    if (head.kernel_size, head.stride, head.padding, head.groups) != ((1, 1), (1, 1), (0, 0), 1):
        raise ValueError("Upstream readout geometry differs from the audited contract.")
    w = array(conv.weight)
    b = np.zeros(w.shape[0]) if conv.bias is None else array(conv.bias)
    gamma = np.ones_like(b) if bn.weight is None else array(bn.weight)
    beta = np.zeros_like(b) if bn.bias is None else array(bn.bias)
    folded, folded_bias = fold_batch_norm(
        w, b, array(bn.running_mean), array(bn.running_var), gamma, beta, bn.eps
    )
    rw = array(head.weight)
    if rw.shape[-2:] != (1, 1):
        raise ValueError("Expected a pointwise upstream detection readout.")
    rb = np.zeros(rw.shape[0]) if head.bias is None else array(head.bias)
    return BranchWeights(folded, folded_bias, rw[:, :, 0, 0], rb)


def selected_smooth_readout(pyramid, mask, parameters: BranchWeights) -> dict:
    """Run 3x3 convolution + LIF only at admitted OUTPUT sites for every microstep.

    Input halos still read unselected neighboring sites. There is no zero-masked
    dense convolution. Neuron state starts at zero for every macro-window.
    """
    x = np.asarray(pyramid, dtype=float)  # Q,Cin,Hs,Ws
    chosen = np.asarray(mask)
    if x.ndim != 4 or x.shape[0] < 1 or chosen.dtype != bool or chosen.shape != x.shape[2:]:
        raise ValueError("Expected Q,Cin,H,W and a boolean H,W mask.")
    q_steps, c_in, height, width = x.shape
    w, b = np.asarray(parameters.conv), np.asarray(parameters.conv_bias)
    rw, rb = np.asarray(parameters.readout), np.asarray(parameters.readout_bias)
    if w.ndim != 4 or w.shape[1:] != (c_in, 3, 3) or b.shape != (w.shape[0],):
        raise ValueError("Expected folded 3x3 stride-one weights.")
    if rw.ndim != 2 or rw.shape[1] != w.shape[0] or rb.shape != (rw.shape[0],):
        raise ValueError("Readout shape does not match the smooth layer.")
    if not all(np.isfinite(v).all() for v in (x, w, b, rw, rb)):
        raise ValueError("Finite inputs and parameters required.")
    if not 0 <= parameters.decay <= 1 or not np.isfinite(parameters.threshold):
        raise ValueError("Invalid neuron settings.")
    coords = np.argwhere(chosen)
    c_out, k = w.shape[0], len(coords)
    membrane = np.zeros((k, c_out))
    spikes = np.zeros_like(membrane)
    for q in range(q_steps):
        padded = np.pad(x[q], ((0, 0), (1, 1), (1, 1)))
        for index, (row, col) in enumerate(coords):
            patch = padded[:, row:row+3, col:col+3]
            current = np.einsum("ocij,cij->o", w, patch) + b
            before_reset = membrane[index] + current
            spikes[index] = before_reset > parameters.threshold
            membrane[index] = before_reset * parameters.decay * (1-spikes[index])
    logits = spikes @ rw.T + rb if k else np.empty((0, rw.shape[0]))
    return dict(
        coordinates=coords, logits=logits, final_membrane=membrane,
        counters=dict(
            smooth_output_sites=q_steps*k,
            smooth_dot_products=q_steps*k*c_out,
            smooth_mac_slots=q_steps*k*c_out*c_in*9,
            readout_mac_slots=k*rw.shape[0]*c_out,
            membrane_updates=q_steps*k*c_out,
            selected_sites=k, total_sites=height*width,
            counter_scope="Executed reference loops/dense dot-product slots, not hardware AC or energy.",
        ),
    )


def _normalise(values):
    scale = float(np.quantile(values, 0.99)) if values.size else 0
    return np.clip(values/(scale+1e-12), 0, 1) if scale > 1e-12 else np.zeros_like(values)


def choose_sites(scores, age, capacity: int, age_threshold: int):
    score = np.asarray(scores, dtype=float)
    ages = np.asarray(age, dtype=float)
    if score.shape != ages.shape or score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("Finite aligned score/age maps required.")
    if not np.isfinite(ages).all() or np.any(ages < 0):
        raise ValueError("Invalid age map.")
    if not 0 <= capacity <= score.size or int(capacity) != capacity:
        raise ValueError("Invalid per-scale capacity.")
    capacity = int(capacity)
    flat, a = score.ravel(), ages.ravel()
    idx = np.arange(flat.size)
    stale = a >= age_threshold if age_threshold > 0 else np.zeros(flat.size, dtype=bool)
    old, other = idx[stale], idx[~stale]
    old = old[np.lexsort((old, -a[old]))]
    other = other[np.lexsort((other, -flat[other]))]
    selected = np.concatenate((old, other))[:capacity]
    mask = np.zeros(flat.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(score.shape), int(np.sum(stale & ~mask))


def run_cached_window(pyramids: dict, weights: dict, previous_prediction: dict | None,
                      importance: dict | None, capacities: dict, *,
                      policy="hybrid", alpha=0.7, residual_weight=0.6,
                      importance_weight=0.4, ages=None, age_threshold=4) -> dict:
    """Concrete seam: cached Z_s -> prediction/risk ranking -> sparse SNN -> logits.

    Importance maps are causal inputs supplied by the current-proposal/track route.
    Capacities are explicit per-scale inputs; they do NOT assert total-cost matching.
    This function is not a complete SpikeFPN forward or a trained hazard estimator.
    """
    if policy not in ("dense", "residual", "importance", "hybrid"):
        raise ValueError("Unknown policy.")
    if not 0 <= alpha <= 1 or min(residual_weight, importance_weight) < 0:
        raise ValueError("Invalid policy settings.")
    if set(pyramids) != set(weights) or set(pyramids) != set(capacities):
        raise ValueError("Each scale needs cached features, parameters and a capacity.")
    use_prediction = policy in ("residual", "hybrid")
    use_importance = policy in ("importance", "hybrid")
    current, residuals, masks, results, next_ages, overflow = {}, {}, {}, {}, {}, {}
    for stride in sorted(pyramids):
        z = np.asarray(pyramids[stride], dtype=float)
        if z.ndim != 4 or not z.shape[0]:
            raise ValueError("Each pyramid must be Q,C,H,W.")
        h = z.mean(axis=0)
        current[stride] = h
        age = np.zeros(h.shape[1:]) if ages is None else np.asarray(ages[stride])
        if use_prediction:
            past = np.zeros_like(h) if previous_prediction is None else np.asarray(previous_prediction[stride])
            if past.shape != h.shape or not np.isfinite(past).all():
                raise ValueError("Previous coarse prediction shape mismatch.")
            error = np.abs(h-past).sum(axis=0)/(np.abs(h).sum(axis=0)+1e-12)
            residuals[stride] = error
        if use_importance:
            if importance is None or stride not in importance:
                raise ValueError("Causal importance maps are required by this policy.")
            risk = np.asarray(importance[stride], dtype=float)
            if risk.shape != h.shape[1:] or not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
                raise ValueError("Importance must be a finite spatial map in [0,1].")
        if policy == "dense":
            score, capacity = np.ones(h.shape[1:]), h.shape[1]*h.shape[2]
        elif policy == "residual":
            score, capacity = _normalise(error), capacities[stride]
        elif policy == "importance":
            score, capacity = _normalise(risk), capacities[stride]
        else:
            score = residual_weight*_normalise(error)+importance_weight*_normalise(risk)
            capacity = capacities[stride]
        mask, excess = choose_sites(score, age, capacity, age_threshold)
        masks[stride], overflow[stride] = mask, excess
        results[stride] = selected_smooth_readout(z, mask, weights[stride])
        next_ages[stride] = np.where(mask, 0, age+1)
    # Form next-window predictions only AFTER every conditional output was computed.
    next_prediction = None
    if use_prediction:
        next_prediction = {
            stride: alpha*(np.zeros_like(h) if previous_prediction is None else previous_prediction[stride])
                    +(1-alpha)*h for stride, h in current.items()
        }
    return dict(branches=results, masks=masks, residuals=residuals,
                next_prediction=next_prediction, next_ages=next_ages, stale_overflow=overflow,
                shared_cost_included=False,
                limitation="Upstream shared forward, proposal/track route and full-cost profiling are external.")


def decode_selected(logits, coordinates, anchors, stride: int, image_hw: tuple[int, int],
                    *, classes: int, center_sample=False):
    """Match the upstream channel layout and GLOBAL grid coordinates.

    Returns raw object/class logits and decoded boxes. Apply the original score
    transformation and NMS separately; they are not redefined here.
    """
    values = np.asarray(logits, dtype=float)
    coords = np.asarray(coordinates)
    anchor = np.asarray(anchors, dtype=float)
    if anchor.ndim != 2 or anchor.shape[1] != 2 or np.any(anchor <= 0):
        raise ValueError("Positive A,2 anchor sizes required.")
    a, k = len(anchor), len(coords)
    if values.shape != (k, a*(classes+5)) or coords.shape != (k, 2):
        raise ValueError("Logit/coordinate/anchor shape mismatch.")
    height, width = image_hw
    if min(height, width, stride) <= 0 or classes < 1:
        raise ValueError("Invalid image geometry or class count.")
    obj = values[:, :a]
    cls = values[:, a:a*(classes+1)].reshape(k, a, classes)
    reg = values[:, a*(classes+1):].reshape(k, a, 4)
    sig = 1/(1+np.exp(-np.clip(reg[..., :2], -700, 700)))
    offset = 2*sig-1 if center_sample else sig
    grid_xy = coords[:, [1, 0]][:, None, :]
    xy = (offset+grid_xy)*stride
    wh = np.exp(reg[..., 2:])*anchor[None]
    boxes = np.concatenate((xy-wh/2, xy+wh/2), axis=-1)
    if not np.isfinite(boxes).all():
        raise ValueError("Nonfinite decoded boxes.")
    norm = np.array([width, height, width, height])
    return dict(objectness_logits=obj, class_logits=cls, boxes=boxes,
                normalized_boxes=boxes/norm, global_grid_xy=grid_xy)
