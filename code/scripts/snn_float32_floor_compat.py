"""Preserve the selector's float32 floor boundary in FP calibration validation.

Raw detector scores, filtering and returned threshold semantics are unchanged.
Only an exact float32 encoding of the declared floor is represented by that
declared floor while passing through the original float64 input validator.
"""
import numpy as np
from snn_secondary_common import ROOT
from riskaware.calibration_audit import threshold_at_fppi as original_threshold


def threshold_at_fppi(false_positive_scores, frame_count, target_fppi, floor=.3):
    values = np.asarray(false_positive_scores, dtype=np.float64)
    if values.ndim == 1 and np.isfinite(floor) and 0 <= floor <= 1:
        encoded = float(np.float32(floor))
        if encoded < floor:
            boundary = values == encoded
            if boundary.any():
                values = values.copy()
                values[boundary] = floor
    return original_threshold(values, frame_count, target_fppi, floor)


def install():
    import riskaware.calibration_audit as module
    module.threshold_at_fppi = threshold_at_fppi
