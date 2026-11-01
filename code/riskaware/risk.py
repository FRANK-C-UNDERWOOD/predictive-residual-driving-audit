"""Causal risk proxy computed from the current coarse observation.

Rules that the paper must defend, and that this module enforces:

1. The risk proxy may use ONLY information available at time t: the current
   coarse feature/detection map, its own track history, and ego speed if
   available.  No future frames, no ground truth, no simulator state.
2. It is cheap by construction: a handful of arithmetic ops per track, not a
   second network.  Its cost still enters the ledger (cost.py) because
   reviewers will ask.
3. It returns interpretable components, so Fig. 3 (one object's residual /
   risk / gate / detection timeline) can explain individual failures.

The proxy is deliberately geometric, not "true collision probability".  In
simulation the evaluation label comes from an independent path-conflict rule
(labels.py); on real data we only claim a "task-critical subset" (docs/03).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class RiskParams:
    horizon_s: float = 1.5             # conflicts beyond this horizon are not "urgent"
    lane_half_width_norm: float = 0.12  # |dx|/W inside this counts as "in path"
    ego_center_x_norm: float = 0.5
    w_ttc: float = 0.45
    w_lateral: float = 0.25
    w_loom: float = 0.20
    w_reappear: float = 0.10
    min_area_frac: float = 0.002


@dataclass
class TrackState:
    box: Box
    t: float
    area: float
    width: float


@dataclass
class RiskEstimator:
    """Per-track risk with named components; projects onto the region grid."""

    params: RiskParams = field(default_factory=RiskParams)
    history: Dict[int, TrackState] = field(default_factory=dict)
    last_risk: Dict[int, float] = field(default_factory=dict)
    last_seen: Dict[int, float] = field(default_factory=dict)

    def forget(self, max_tracks: int = 256) -> None:
        if len(self.history) > max_tracks:
            for k in list(self.history)[: len(self.history) - max_tracks]:
                self.history.pop(k, None)

    def update(
        self,
        track_id: int,
        box: Box,
        t: float,
        image_w: float,
        image_h: float,
        reappear: bool = False,
    ) -> Dict[str, float]:
        x1, y1, x2, y2 = box
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        area = w * h
        prev = self.history.get(track_id)
        self.history[track_id] = TrackState(box, t, area, w)

        cx = 0.5 * (x1 + x2) / max(1e-6, image_w)
        dx = abs(cx - self.params.ego_center_x_norm)
        lateral = float(np.exp(-((dx / self.params.lane_half_width_norm) ** 2)))

        ttc = 2.0 * self.params.horizon_s
        looming = 0.0
        if prev is not None and t > prev.t:
            dt = t - prev.t
            darea = (area - prev.area) / max(1e-9, prev.area)
            looming = _clip01(darea / max(1e-6, 0.5 * dt))  # 50%/s growth == saturated
            dw = (w - prev.width) / dt
            if dw > 1e-6:
                ttc = float(w / dw)
        ttc_risk = _clip01(1.0 - ttc / self.params.horizon_s)

        # Reappearance bonus decays over ~0.5 s after the track re-enters.
        if reappear:
            self.last_seen[track_id] = t
        since = t - self.last_seen.get(track_id, -1e9)
        reappear_risk = float(np.exp(-max(0.0, since) / 0.5)) if reappear else 0.0

        small = _clip01(
            1.0 - (area / max(1e-9, image_w * image_h)) / max(1e-9, self.params.min_area_frac)
        )
        score = (
            self.params.w_ttc * ttc_risk
            + self.params.w_lateral * lateral
            + self.params.w_loom * looming
            + self.params.w_reappear * reappear_risk
        ) * (0.5 + 0.5 * small)
        score = _clip01(score)
        self.last_risk[track_id] = score
        return {
            "risk": score,
            "ttc_s": ttc,
            "ttc_risk": ttc_risk,
            "lateral": lateral,
            "looming": looming,
            "reappear": reappear_risk,
            "small": small,
            "dx_norm": dx,
        }

    def region_risk_map(
        self,
        grid_h: int,
        grid_w: int,
        image_w: float,
        image_h: float,
    ) -> np.ndarray:
        """Project the latest per-track risk onto the coarse region grid.

        Uses only the current step (last_risk is refreshed by update()).  If one
        region holds several tracks, the maximum risk wins -- conservative.
        """
        m = np.zeros((grid_h, grid_w), dtype=np.float64)
        for tid, st in self.history.items():
            r = self.last_risk.get(tid, 0.0)
            if r <= 0.0:
                continue
            x1, y1, x2, y2 = st.box
            r0 = int(np.clip(y1 / image_h * grid_h, 0, grid_h - 1))
            r1 = int(np.clip(np.ceil(y2 / image_h * grid_h), r0 + 1, grid_h))
            c0 = int(np.clip(x1 / image_w * grid_w, 0, grid_w - 1))
            c1 = int(np.clip(np.ceil(x2 / image_w * grid_w), c0 + 1, grid_w))
            m[r0:r1, c0:c1] = np.maximum(m[r0:r1, c0:c1], r)
        return m.reshape(-1)
