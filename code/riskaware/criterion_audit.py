"""Offline criterion interventions. These functions do not skip detector work."""
from __future__ import annotations
import numpy as np
from .metrics import Det, GT

SCALES = (8, 16, 32)


def select_capacity(scores, ratio):
    if not 0 < ratio <= 1:
        raise ValueError("Capacity ratio must be in (0,1]")
    chosen, capacities = [], {}
    for scale in SCALES:
        values = np.asarray(scores[scale]).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Nonempty finite scores required")
        k = max(1, int(np.floor(ratio * len(values))))
        order = np.lexsort((np.arange(len(values)), -values))
        mask = np.zeros(len(values), dtype=bool)
        mask[order[:k]] = True
        chosen.append(mask)
        capacities[scale] = k
    return np.concatenate(chosen), capacities


def filter_anchors(boxes, scores, site_mask, anchors, threshold, nms_threshold, nms_fn):
    """Reuse the upstream per-class NMS rule, retaining original anchor IDs."""
    if boxes.shape != (len(scores), 4) or len(site_mask)*anchors != len(scores):
        raise ValueError("Head/site geometry mismatch")
    candidate = np.flatnonzero(np.repeat(site_mask, anchors))
    classes = np.argmax(scores[candidate], axis=1)
    confidences = scores[candidate, classes]
    keep = confidences >= threshold
    candidate, classes, confidences = candidate[keep], classes[keep], confidences[keep]
    flags = np.zeros(len(candidate), dtype=bool)
    for cls in range(scores.shape[1]):
        ids = np.flatnonzero(classes == cls)
        if len(ids):
            ids_kept = nms_fn(boxes[candidate[ids]], confidences[ids], nms_threshold)
            flags[ids[np.asarray(ids_kept, dtype=int)]] = True
    return boxes[candidate[flags]], confidences[flags], classes[flags], candidate[flags]


def native_detections(filtered, ratio=.4, pad_top=32., pad_left=0.):
    boxes, scores, classes, _ = filtered
    boxes = (boxes * 256 - np.array([pad_left, pad_top, pad_left, pad_top])) / ratio
    return [Det(tuple(map(float, b)), float(s), int(c)) for b, s, c in zip(boxes, scores, classes)]


def ground_truth(rows, class_map, small_fraction):
    objects, ignored = [], 0
    for row in rows:
        source_class = int(row["class_id"])
        if str(source_class) not in class_map:
            ignored += 1
            continue
        x1, y1 = max(0., float(row["x"])), max(0., float(row["y"]))
        x2 = min(640., float(row["x"] + row["w"]))
        y2 = min(480., float(row["y"] + row["h"]))
        if x2 <= x1 or y2 <= y1:
            ignored += 1
            continue
        objects.append(GT((x1,y1,x2,y2), class_map[str(source_class)], int(row["track_id"]),
                          (x2-x1)*(y2-y1)/(640*480) <= small_fraction, ("small-area",)))
    return objects, ignored
