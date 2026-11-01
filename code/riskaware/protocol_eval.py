"""v0.3 fresh/system and new-run/maintained-run temporal endpoints.

Annotation times are supplied explicitly, including times with no GT objects.
Exclusions use the declared full horizon and are reported independently of outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .metrics import Det, KeyObjectEvent, RunLengthRule, frame_level_gt, match_frame


@dataclass(frozen=True)
class EventOutcome:
    seq: str
    event_id: str
    obj_id: int
    endpoint: str
    timely: bool
    detected_within_horizon: bool
    latency: float
    onset: float
    completion_time: float | None


def evaluate_events(
    events: Sequence[KeyObjectEvent], gt_tracks: dict, detections: dict,
    annotation_times: dict, *, horizon: float, delta: float,
    rule: RunLengthRule, max_annotation_gap: float,
    source: str = "fresh", run_mode: str = "new",
) -> dict:
    if source not in ("fresh", "system") or run_mode not in ("new", "maintained"):
        raise ValueError("Unknown output source or run mode.")
    if not all(math.isfinite(x) for x in (delta, horizon, max_annotation_gap)) or not 0 <= delta <= horizon or horizon <= 0 or max_annotation_gap <= 0:
        raise ValueError("Invalid horizon, deadline or annotation-gap limit.")
    if rule.run_length < 1:
        raise ValueError("Positive reliable run length required.")
    grids = {}
    for seq, times in annotation_times.items():
        ts = np.asarray(times, dtype=float)
        if ts.ndim != 1 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError("Annotation grids must be finite and strictly increasing.")
        grids[str(seq)] = ts.tolist()
    gt_frames = frame_level_gt(gt_tracks)
    matches = {}
    for seq, grid in grids.items():
        for t in grid:
            objs = sorted(gt_frames.get(seq, {}).get(t, []), key=lambda pair: pair[0])
            raw = detections.get(seq, {}).get(t, [])
            if any(getattr(det, "source", None) not in ("fresh", "propagated") for det in raw):
                raise ValueError("Each prediction must declare fresh or propagated source.")
            kept = [det for det in raw if source == "system" or det.source == "fresh"]
            assigned = match_frame(kept, [gt for _, gt in objs], rule.iou_thr, rule.score_thr)
            matches[seq, t] = {objs[index][0] for index in assigned}
    outcomes, excluded, identities = [], [], set()
    for ev in events:
        event_id = str(ev.meta.get("event_id", "%s:%s:%s" % (ev.seq, ev.obj_id, ev.t_on)))
        identity = (ev.seq, event_id)
        if identity in identities:
            raise ValueError("Duplicate evaluation event.")
        identities.add(identity)
        if run_mode == "maintained" and ev.kind != "risk_escalation":
            raise ValueError("Maintained-run endpoint is restricted to risk escalation.")
        grid = grids.get(ev.seq, [])
        end = ev.t_on + horizon
        if not grid or grid[0] > ev.t_on or grid[-1] < end - 1e-12 or ev.t_end < end - 1e-12:
            excluded.append(dict(seq=ev.seq, event_id=event_id, reason="incomplete-full-horizon"))
            continue
        run, previous, completion = 0, None, None
        for t in grid:
            if t > end + 1e-12:
                break
            if run_mode == "new" and t < ev.t_on:
                continue
            if previous is not None and t - previous > max_annotation_gap + 1e-12:
                run = 0
            run = run + 1 if ev.obj_id in matches[ev.seq, t] else 0
            previous = t
            if t >= ev.t_on and run >= rule.run_length:
                completion = t
                break
        latency = math.inf if completion is None else float(completion - ev.t_on)
        outcomes.append(EventOutcome(ev.seq, event_id, ev.obj_id,
                                     source + "-" + run_mode, latency <= delta,
                                     completion is not None, latency, ev.t_on, completion))
    return dict(rate=(sum(o.timely for o in outcomes) / len(outcomes) if outcomes else math.nan),
                outcomes=outcomes, exclusions=excluded)


def latency_summary(outcomes: Sequence[EventOutcome], horizon: float) -> dict:
    """Unconditional ECDF inverse retains infinities; capped summaries are named."""
    if horizon <= 0:
        raise ValueError("Positive horizon required.")
    if not outcomes:
        return dict(n=0, p50_unconditional=math.nan, p95_unconditional=math.nan)
    lat = np.array([o.latency for o in outcomes], dtype=float)
    if np.any(np.isnan(lat)) or np.any(lat < 0):
        raise ValueError("Nonnegative or infinite delays required.")
    finite = lat[np.isfinite(lat)]
    ordered = np.sort(lat)

    def ecdf_quantile(q):
        return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])

    capped = np.minimum(lat, horizon)
    return dict(n=len(lat), n_detected=len(finite),
                p50_unconditional=ecdf_quantile(0.5), p95_unconditional=ecdf_quantile(0.95),
                p50_conditional=float(np.quantile(finite, 0.5)) if finite.size else math.nan,
                p95_conditional=float(np.quantile(finite, 0.95)) if finite.size else math.nan,
                mean_capped=float(capped.mean()), p95_capped=float(np.quantile(capped, 0.95)))
