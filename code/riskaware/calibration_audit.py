"""Development score calibration on calibration false positives only."""
from __future__ import annotations
import math
import numpy as np


def threshold_at_fppi(false_positive_scores, frame_count, target_fppi, floor=.3):
    scores = np.asarray(false_positive_scores, dtype=np.float64)
    if (scores.ndim != 1 or not np.isfinite(scores).all() or frame_count < 1
        or not isinstance(frame_count, int) or not math.isfinite(target_fppi) or target_fppi < 0
        or not math.isfinite(floor) or not 0 <= floor <= 1
        or np.any(scores < floor) or np.any(scores > 1)):
        raise ValueError("Finite score probabilities, positive frame count and nonnegative FPPI required")
    allowed = math.floor(target_fppi * frame_count)
    if len(scores) <= allowed:
        return float(floor)
    excluded_score = np.sort(scores)[::-1][allowed]
    # Detector confidences are float32. A float64 successor can be rounded
    # back to the excluded value by NumPy's scalar comparison promotion.
    return float(np.nextafter(np.float32(excluded_score), np.float32(np.inf)))
