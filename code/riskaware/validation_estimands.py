"""Predeclared recording-level and common-support validation estimands."""
import math
import numpy as np

def recall_difference(a, b, denominator):
    return 100. * (a - b) / denominator if denominator else None

def motion_interaction(low_a, low_b, low_gt, high_a, high_b, high_gt):
    low = recall_difference(low_a, low_b, low_gt)
    high = recall_difference(high_a, high_b, high_gt)
    return low - high if low is not None and high is not None else None

def weighted_cell_interaction(cells, weights):
    """Missing positive-weight support invalidates the estimand; never renormalize."""
    total = 0.
    for key, weight in weights.items():
        if weight <= 0:
            continue
        if key not in cells:
            return None
        value = motion_interaction(**cells[key])
        if value is None:
            return None
        total += weight * value
    return total if abs(sum(weights.values()) - 1.) < 1e-10 else None

def recording_summary(values):
    values = list(values)
    defined = [v for v in values if v is not None]
    return dict(planned=len(values), defined=len(defined),
                mean=float(np.mean(defined)) if defined else None,
                median=float(np.median(defined)) if defined else None,
                positive=sum(v > 0 for v in defined), negative=sum(v < 0 for v in defined),
                tied=sum(v == 0 for v in defined),
                all_planned_defined=len(values) == len(defined))
