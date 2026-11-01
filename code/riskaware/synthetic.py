"""Toy sequences for pipeline sanity checks ONLY.

This module lets you exercise the whole measurement pipeline (gating -> masking
-> per-region detector proxy -> matching -> timely detection -> cost ledger)
before any real data is wired in.  It is a *mechanism simulator*, not a dataset
and not a result: nothing produced from it may appear in the paper as evidence.

What it does emulate, because these are the mechanisms the paper's claims rest
on:
  * a causal predictor (exponential moving average) that is good on smooth
    background and poor on object appearance -> residual spikes at novelty;
  * risk computed from track geometry only (risk.py), so "risk" and "residual"
    are genuinely different signals;
  * a detector proxy that can only emit a detection for an object whose region
    was actually processed, so suppression has a real, traceable cost;
  * occlusion gaps and cut-ins, so reappearance and rapid risk escalation exist
    as test cases.

Everything is precomputed once so that the evaluation pass and the method pass
see bit-identical data and identical object trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import metrics


@dataclass
class ToyConfig:
    n_seq: int = 8
    steps: int = 200
    dt: float = 0.05
    grid_h: int = 8
    grid_w: int = 14
    feat_dim: int = 8
    image_w: float = 448.0
    image_h: float = 256.0
    objects_per_seq: int = 10
    seed: int = 0
    noise: float = 0.25
    background_flicker: int = 40
    object_events: int = 400


@dataclass
class ToyObject:
    obj_id: int
    cls: int
    kind: str
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    t_in: int
    t_out: int
    occluded: List[Tuple[int, int]] = field(default_factory=list)
    path_conflict: bool = False
    key: bool = False


@dataclass
class StepData:
    t: float
    events: np.ndarray          # (GH, GW)
    features: np.ndarray        # (GH, GW, D)
    gts: List[Tuple[int, metrics.GT]]
    visible: Dict[int, bool]


class ToySequence:
    def __init__(self, cfg: ToyConfig, seq_id: int, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.seq_id = seq_id
        self.name = "toy_%02d" % seq_id
        self.rng = rng
        self.objects: List[ToyObject] = []
        self.steps_data: List[StepData] = []
        self.base_events = rng.integers(5, 30, size=(cfg.grid_h, cfg.grid_w)).astype(float)
        self.base_feat = rng.normal(0, 0.05, size=(cfg.grid_h, cfg.grid_w, cfg.feat_dim))

    # ---------------------------------------------------------------- gen ---
    def build(self) -> None:
        cfg, rng = self.cfg, self.rng
        for k in range(cfg.objects_per_seq):
            kind = "pedestrian" if rng.random() < 0.55 else "vehicle"
            cls = 1 if kind == "pedestrian" else 2
            lateral = bool(rng.random() < 0.6)
            x0 = 0.02 if rng.random() < 0.5 else 0.98
            x = x0 if lateral else float(rng.uniform(0.25, 0.75))
            y = float(rng.uniform(0.55, 0.85))
            w = float(rng.uniform(0.10, 0.26))
            h = float(rng.uniform(0.20, 0.55))
            vx = (0.35 if x0 < 0.5 else -0.35) * float(rng.uniform(0.6, 1.2)) if lateral else 0.0
            vy = float(rng.uniform(-0.05, 0.05))
            t_in = int(rng.integers(5, cfg.steps // 2))
            t_out = int(rng.integers(t_in + 20, cfg.steps))
            occ: List[Tuple[int, int]] = []
            if rng.random() < 0.6:
                s = int(rng.integers(t_in + 5, max(t_in + 6, t_out - 6)))
                occ.append((s, s + int(rng.integers(3, 7))))
            conflict = lateral and rng.random() < 0.7
            key = kind == "pedestrian" or (w * h) < 0.03
            self.objects.append(
                ToyObject(k, cls, kind, x, y, w, h, vx, vy, t_in, t_out, occ, conflict, key)
            )

    def precompute(self) -> None:
        """Materialise every step once.  Run and evaluation share this array."""
        cfg = self.cfg
        for t in range(cfg.steps):
            events = self.base_events + self.rng.integers(
                0, cfg.background_flicker, size=self.base_events.shape
            ).astype(float)
            feats = self.base_feat + self.rng.normal(0, cfg.noise * 0.1, size=self.base_feat.shape)
            gts: List[Tuple[int, metrics.GT]] = []
            visible_map: Dict[int, bool] = {}
            for o in self.objects:
                if not (o.t_in <= t < o.t_out):
                    continue
                visible = not any(a <= t < b for a, b in o.occluded)
                visible_map[o.obj_id] = visible
                o.x = float(np.clip(o.x + o.vx * cfg.dt, 0.0, 1.0))
                o.y = float(np.clip(o.y + o.vy * cfg.dt, 0.0, 1.0))
                if o.path_conflict and o.vx != 0.0 and abs(o.x - 0.5) < 0.12:
                    o.vx = 0.0
                box = (
                    (o.x - o.w / 2) * cfg.image_w,
                    (o.y - o.h / 2) * cfg.image_h,
                    (o.x + o.w / 2) * cfg.image_w,
                    (o.y + o.h / 2) * cfg.image_h,
                )
                if not visible:
                    continue
                c0 = int(np.clip((o.x - o.w / 2) * cfg.grid_w, 0, cfg.grid_w - 1))
                c1 = int(np.clip(np.ceil((o.x + o.w / 2) * cfg.grid_w), c0 + 1, cfg.grid_w))
                r0 = int(np.clip((o.y - o.h / 2) * cfg.grid_h, 0, cfg.grid_h - 1))
                r1 = int(np.clip(np.ceil((o.y + o.h / 2) * cfg.grid_h), r0 + 1, cfg.grid_h))
                events[r0:r1, c0:c1] += cfg.object_events
                feats[r0:r1, c0:c1] += self._obj_pattern(o)
                gts.append(
                    (
                        o.obj_id,
                        metrics.GT(
                            box=box,
                            cls=o.cls,
                            track_id=o.obj_id,
                            key=o.key,
                            key_reasons=("pedestrian",) if o.kind == "pedestrian" else (("small",) if o.key else ()),
                        ),
                    )
                )
            self.steps_data.append(
                StepData(t * cfg.dt, events, feats, gts, visible_map)
            )

    def _obj_pattern(self, o: ToyObject) -> np.ndarray:
        rng = np.random.default_rng((self.cfg.seed * 1000 + self.seq_id * 37 + o.obj_id) % (2**32))
        base = rng.normal(0, 0.6, size=self.cfg.feat_dim)
        base[0] = 1.0 if o.kind == "pedestrian" else -1.0
        return base


def make_dataset(cfg: ToyConfig) -> List[ToySequence]:
    rng = np.random.default_rng(cfg.seed)
    seqs = []
    for i in range(cfg.n_seq):
        s = ToySequence(cfg, i, rng)
        s.build()
        s.precompute()
        seqs.append(s)
    return seqs


def gt_tracks(seqs: List[ToySequence]) -> Dict[Tuple[str, int], Dict[float, metrics.GT]]:
    out: Dict[Tuple[str, int], Dict[float, metrics.GT]] = {}
    for s in seqs:
        for st in s.steps_data:
            for oid, g in st.gts:
                out.setdefault((s.name, oid), {})[st.t] = g
    return out


def key_events(seqs: List[ToySequence], delta_frames: int = 1) -> List[metrics.KeyObjectEvent]:
    """Onset = first frame the object is visible (after 'delta_frames' guard)."""
    evs: List[metrics.KeyObjectEvent] = []
    for s in seqs:
        first: Dict[int, float] = {}
        last: Dict[int, float] = {}
        for st in s.steps_data:
            for oid, g in st.gts:
                if not g.key:
                    continue
                first.setdefault(oid, st.t)
                last[oid] = st.t
        for oid, t_on in first.items():
            evs.append(
                metrics.KeyObjectEvent(
                    seq=s.name, obj_id=oid, t_on=t_on, t_end=last[oid], kind="object"
                )
            )
    return evs
