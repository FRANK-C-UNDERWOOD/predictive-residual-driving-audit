"""Evaluation-side label construction.  Never used inside the method.

Two independent things are built here:

  key objects   -- which objects count as task-critical, by a rule fixed BEFORE
                   looking at any model output (docs/03).
  onset times   -- when the clock starts, both for new objects and for objects
                   whose risk escalates.

The provenance guard is deliberate: every record carries source="simulator" or
"annotation", and records whose boxes/onsets came from the evaluated model are
rejected.  A key-object label derived from the model would make H1 circular.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


class LabelProvenanceError(ValueError):
    pass


@dataclass
class KeyRule:
    """Quantitative key-object rule.  All thresholds are fixed before E2 runs."""

    min_visible_frames: int = 3
    small_area_frac: float = 0.01          # relative to image area
    occlusion_gap_frames: int = 3
    risk_onset_run: int = 2                # consecutive frames that start the clock
    vulnerable_classes: Tuple[int, ...] = (1,)  # e.g. pedestrian
    distractable: bool = False


@dataclass
class ObjectTrack:
    seq: str
    obj_id: int
    cls: int
    t: List[float] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)
    visible: List[bool] = field(default_factory=list)
    path_conflict: List[bool] = field(default_factory=list)
    source: str = "simulator"

    def add(self, t: float, box: Box, visible: bool, path_conflict: bool) -> None:
        self.t.append(t)
        self.boxes.append(box)
        self.visible.append(visible)
        self.path_conflict.append(path_conflict)

    # ------------------------------------------------------------- rules ---
    def visibility_start(self, rule: KeyRule) -> Optional[float]:
        run = 0
        for i, v in enumerate(self.visible):
            run = run + 1 if v else 0
            if run >= rule.min_visible_frames:
                return self.t[i]
        return None

    def risk_onset(self, rule: KeyRule) -> Optional[float]:
        run = 0
        for i, c in enumerate(self.path_conflict):
            run = run + 1 if c else 0
            if run >= rule.risk_onset_run:
                return self.t[i]
        return None

    def reasons(
        self, rule: KeyRule, image_w: float, image_h: float
    ) -> List[str]:
        out: List[str] = []
        if self.cls in rule.vulnerable_classes:
            out.append("vulnerable")
        areas = [
            (b[2] - b[0]) * (b[3] - b[1]) / max(1.0, image_w * image_h)
            for b in self.boxes
        ]
        if areas and float(np.median(areas)) <= rule.small_area_frac:
            out.append("small")
        gap = 0
        for v in self.visible:
            gap = 0 if v else gap + 1
            if gap == rule.occlusion_gap_frames:
                out.append("reappeared")
                break
        if any(self.path_conflict):
            out.append("path-conflict")
        return out

    def check_provenance(self) -> None:
        if self.source not in ("simulator", "annotation", "dataset"):
            raise LabelProvenanceError(
                "object labels must come from simulator state or human annotation, "
                f"got source={self.source!r}"
            )


@dataclass
class LabelBundle:
    tracks: Dict[Tuple[str, int], ObjectTrack] = field(default_factory=dict)

    def add_track(self, tr: ObjectTrack) -> None:
        tr.check_provenance()
        self.tracks[(tr.seq, tr.obj_id)] = tr

    def key_objects(
        self, rule: KeyRule, image_w: float, image_h: float
    ) -> Dict[Tuple[str, int], List[str]]:
        out: Dict[Tuple[str, int], List[str]] = {}
        for k, tr in self.tracks.items():
            rs = tr.reasons(rule, image_w, image_h)
            if rs:
                out[k] = rs
        return out

    def onsets(
        self, rule: KeyRule
    ) -> List[Tuple[str, int, float, str]]:
        """(seq, obj_id, t_on, kind) for every evaluated event."""
        out = []
        for (seq, oid), tr in self.tracks.items():
            t_vis = tr.visibility_start(rule)
            if t_vis is not None:
                out.append((seq, oid, t_vis, "object"))
            t_risk = tr.risk_onset(rule)
            if t_risk is not None:
                out.append((seq, oid, t_risk, "risk_escalation"))
        return out


def path_conflict_rule(
    center_x_frac: float,
    half_width_frac: float,
    ego_half_width_frac: float = 0.05,
    horizon: float = 1.5,
    ttc: Optional[float] = None,
) -> bool:
    """Independent geometric conflict test used for SIMULATION labels only.

    Deliberately simple and inspectable: an object conflicts if its lateral
    extent overlaps the ego corridor and (if a TTC is available) the TTC is
    inside the horizon.  It uses privileged simulator state, never a detector.
    """
    if ttc is not None and ttc > horizon:
        return False
    return abs(center_x_frac - 0.5) < (ego_half_width_frac + half_width_frac)
