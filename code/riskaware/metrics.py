"""Shared detection helpers plus LEGACY temporal/statistical functions.

Use protocol_eval.py and protocol_stats.py for manuscript v0.3 endpoints and
event-weighted inference. The legacy temporal functions below remain available
for historical synthetic checks and do not implement the new source tags.

Design rules enforced here (see docs/02):
  * A detection is only "reliable" after R consecutive matched frames.
  * An object that is never reliably detected is a FAILURE, not a censored point.
  * Latency statistics are reported twice: conditional on success, and with
    failures imputed at a documented value (default: the evaluation horizon).
  * Confidence intervals are clustered by sequence, never by frame.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


# ---------------------------------------------------------------- geometry ---
def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class Det:
    box: Box
    score: float
    cls: int = 0
    track_id: int = -1
    source: str = "fresh"
    last_fresh_time: float | None = None


@dataclass(frozen=True)
class GT:
    box: Box
    cls: int = 0
    track_id: int = -1
    key: bool = False
    key_reasons: Tuple[str, ...] = ()


def match_frame(
    dets: Sequence[Det],
    gts: Sequence[GT],
    iou_thr: float = 0.5,
    score_thr: float = 0.0,
) -> Dict[int, int]:
    """Greedy one-to-one matching, highest score first.

    Returns {gt_index: det_index}.  Unmatched GTs are false negatives, unmatched
    Dets are false positives, exactly as in the AP definition below.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    used_gt: set = set()
    out: Dict[int, int] = {}
    for di in order:
        d = dets[di]
        if d.score < score_thr:
            continue
        best, best_iou = -1, iou_thr
        for gi, g in enumerate(gts):
            if gi in used_gt or g.cls != d.cls:
                continue
            v = iou(d.box, g.box)
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            used_gt.add(best)
            out[best] = di
    return out


# ------------------------------------------------------- average precision ---
def average_precision(
    frames: Iterable[Tuple[Sequence[Det], Sequence[GT]]],
    iou_thr: float = 0.5,
    n_classes: int = 1,
    strict: bool = True,
) -> Dict[str, float]:
    """All-point interpolated AP@iou_thr, per class, plus mAP.

    frames: iterable of (dets, gts) pairs.  Classes with no GT are reported as
    NaN and excluded from mAP.
    """
    per_class: Dict[int, List[Tuple[float, int]]] = {c: [] for c in range(n_classes)}
    n_gt: Dict[int, int] = {c: 0 for c in range(n_classes)}
    for dets, gts in frames:
        unknown_gt = [g.cls for g in gts if g.cls not in n_gt]
        if unknown_gt:
            if strict:
                raise ValueError("ground-truth class outside range(n_classes): %r" % unknown_gt)
            warnings.warn("dropping out-of-range ground-truth classes", RuntimeWarning, stacklevel=2)
        matched = match_frame(dets, gts, iou_thr=iou_thr)
        matched_ids = set(matched.values())
        for gi, g in enumerate(gts):
            if g.cls in n_gt:
                n_gt[g.cls] += 1
        for di, d in enumerate(dets):
            if d.cls not in per_class:
                # A detection of a class that has no labels cannot be a TP.  It is
                # still a false positive at the operating point, so silently
                # dropping it would flatter precision; make the caller decide.
                if strict:
                    raise ValueError(
                        "detection class %d outside range(n_classes=%d); pass strict=False to "
                        "drop them (and accept the precision inflation)" % (d.cls, n_classes)
                    )
                warnings.warn(
                    "dropping detection of class %d outside range(n_classes=%d): precision is "
                    "now measured differently from the specification" % (d.cls, n_classes),
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            per_class[d.cls].append((d.score, 1 if di in matched_ids else 0))

    aps: Dict[int, float] = {}
    for c, rows in per_class.items():
        if n_gt[c] == 0:
            aps[c] = float("nan")
            continue
        rows.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in rows]) if rows else np.array([0.0])
        fp = np.cumsum([1 - r[1] for r in rows]) if rows else np.array([0.0])
        if len(rows) == 0:
            aps[c] = 0.0
            continue
        recall = tp / max(1, n_gt[c])
        precision = tp / np.maximum(tp + fp, 1e-12)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[1.0], precision, [0.0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        aps[c] = ap

    valid = [v for v in aps.values() if not math.isnan(v)]
    out = {f"AP@{c}": v for c, v in aps.items()}
    out[f"mAP@{iou_thr}"] = float(np.mean(valid)) if valid else float("nan")
    return out


def false_positives_per_frame(
    frames: Iterable[Tuple[Sequence[Det], Sequence[GT]]],
    iou_thr: float = 0.5,
    score_thr: float = 0.0,
) -> float:
    """FPPI at the FROZEN operating score threshold (not at all detections).

    Thresholds are calibrated on validation to a common FPPI target and frozen;
    this function then reports the achieved value at that threshold.  Counting
    every detection regardless of score would mix operating points and make the
    "matched false-positive rate" comparison meaningless.
    """
    n_fp, n = 0, 0
    for dets, gts in frames:
        kept = [d for d in dets if d.score >= score_thr]
        matched = match_frame(kept, gts, iou_thr=iou_thr, score_thr=score_thr)
        n_fp += len(kept) - len(matched)
        n += 1
    return n_fp / max(1, n)


# ------------------------------------------------------------ timeliness ----
@dataclass(frozen=True)
class KeyObjectEvent:
    """One evaluation unit: a key object instance that becomes detectable at t_on."""

    seq: str
    obj_id: int
    t_on: float           # onset of the object (or of its risk escalation)
    t_end: float          # last frame of the track (end of evaluation window)
    kind: str = "object"  # "object" | "risk_escalation"
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunLengthRule:
    """Reliability rule: R consecutive frames matched with IoU>=iou and score>=s."""

    run_length: int = 3
    iou_thr: float = 0.5
    score_thr: float = 0.3


@dataclass(frozen=True)
class TimelyOutcome:
    seq: str
    obj_id: int
    success: bool
    latency: float          # seconds from t_on to first reliable detection
    frames_to_reliable: int
    deadline: float


def frame_level_gt(
    gt_tracks: Dict[Tuple[str, int], Dict[float, GT]]
) -> Dict[str, Dict[float, List[Tuple[int, GT]]]]:
    """Invert per-object tracks into per-frame GT lists, for one-to-one matching.

    The evaluation unit is (sequence, object), but the reliability rule must be
    applied frame by frame with one-to-one assignment, otherwise a single
    detection could "reliably detect" two nearby objects and inflate R_timely.
    """
    out: Dict[str, Dict[float, List[Tuple[int, GT]]]] = {}
    for (seq, obj_id), track in gt_tracks.items():
        for t, gt in track.items():
            out.setdefault(seq, {}).setdefault(t, []).append((obj_id, gt))
    return out


def _reliable_latency(
    seq: str,
    obj_id: int,
    frame_gts: Dict[float, List[Tuple[int, GT]]],
    dets_by_t: Dict[float, Sequence[Det]],
    t_on: float,
    t_end: float,
    rule: RunLengthRule,
) -> Tuple[bool, float, int]:
    times = sorted(t for t in frame_gts if t_on <= t <= t_end)
    run = 0
    for i, t in enumerate(times):
        objs = frame_gts[t]
        ids = [oid for oid, _ in objs]
        if obj_id not in ids:
            run = 0
            continue
        gts = [g for _, g in objs]
        matched = match_frame(dets_by_t.get(t, []), gts, iou_thr=rule.iou_thr,
                              score_thr=rule.score_thr)
        ok = ids.index(obj_id) in matched
        if ok:
            run += 1
            if run >= rule.run_length:
                # Conservative definition: the clock stops at the frame where the
                # run-length criterion is SATISFIED, not at the first hit.  With
                # run_length=3 at 20 Hz this adds <= 0.10 s and never flatters us.
                return True, float(t - t_on), i + 1
        else:
            run = 0
    return False, float(max(0.0, t_end - t_on)), 0


def timely_detection_rate(
    events: Sequence[KeyObjectEvent],
    gt_tracks: Dict[Tuple[str, int], Dict[float, GT]],
    dets_by_seq: Dict[str, Dict[float, Sequence[Det]]],
    delta: float,
    rule: RunLengthRule = RunLengthRule(),
    step_s: float = None,
) -> Tuple[float, List[TimelyOutcome]]:
    """R_timely(delta) = reliably detected within delta / all eligible events.

    Objects whose track ends before t_on + delta are excluded from the
    denominator and reported separately (see split_censored).  If step_s is
    given, a delta shorter than the confirmation window warns: the metric would
    be zero for every method by construction.
    """
    if step_s is not None and delta < rule.run_length * step_s:
        warnings.warn(
            "delta=%.3fs is shorter than the confirmation window %.3fs (run_length x step); "
            "R_timely will be 0 for every method" % (delta, rule.run_length * step_s),
            RuntimeWarning,
            stacklevel=2,
        )
    outcomes: List[TimelyOutcome] = []
    frame_gts = frame_level_gt(gt_tracks)
    for ev in events:
        ok, lat, nfr = _reliable_latency(
            ev.seq,
            ev.obj_id,
            frame_gts.get(ev.seq, {}),
            dets_by_seq.get(ev.seq, {}),
            ev.t_on,
            ev.t_end,
            rule,
        )
        success = ok and lat <= delta
        outcomes.append(
            TimelyOutcome(ev.seq, ev.obj_id, success, lat, nfr, ev.t_on + delta)
        )
    if not outcomes:
        return float("nan"), outcomes
    return sum(o.success for o in outcomes) / len(outcomes), outcomes


def latency_summary(
    outcomes: Sequence[TimelyOutcome], horizon: float
) -> Dict[str, float]:
    """Latency distribution with failures handled explicitly.

    conditional_* keeps only successes (optimistic; always report alongside).
    *_with_failures imputes the evaluation horizon for failures (conservative).
    """
    lat = np.array([o.latency for o in outcomes], dtype=float)
    ok = np.array([o.success for o in outcomes], dtype=bool)
    if lat.size == 0:
        return {}
    imp = np.where(ok, lat, horizon)
    out = {
        "n": float(lat.size),
        "n_success": float(ok.sum()),
        "fail_rate": float(1.0 - ok.mean()),
        "p50_with_failures": float(np.percentile(imp, 50)),
        "p95_with_failures": float(np.percentile(imp, 95)),
        "mean_with_failures": float(imp.mean()),
    }
    if ok.any():
        out["p50_conditional"] = float(np.percentile(lat[ok], 50))
        out["p95_conditional"] = float(np.percentile(lat[ok], 95))
        out["mean_conditional"] = float(lat[ok].mean())
    else:
        out["p50_conditional"] = float("nan")
        out["p95_conditional"] = float("nan")
        out["mean_conditional"] = float("nan")
    return out


# ------------------------------------------------------------ statistics ----
def cluster_bootstrap_ci(
    values_by_seq: Dict[str, np.ndarray],
    stat: str = "mean",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile CI resampling whole sequences (not frames).

    Frames inside a sequence are strongly correlated; treating them as
    independent would make every CI far too narrow.
    """
    rng = np.random.default_rng(seed)
    seqs = list(values_by_seq)
    if not seqs:
        return float("nan"), float("nan"), float("nan")
    per_seq = np.array([float(np.mean(values_by_seq[s])) for s in seqs], dtype=float)
    point = float(per_seq.mean())
    if stat == "mean":
        # Vectorised: resampling whole sequences means resampling these per-sequence
        # means, so draw the index matrix once.  Same estimator, ~100x faster, which
        # matters because the analysis applies it to every comparison and every seed.
        idx = rng.integers(0, len(seqs), size=(n_boot, len(seqs)))
        boots = per_seq[idx].mean(axis=1)
    elif stat == "median":
        boots = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            pick = rng.choice(len(seqs), size=len(seqs), replace=True)
            vals = [np.atleast_1d(values_by_seq[seqs[i]]) for i in pick]
            boots[b] = float(np.median(np.concatenate(vals)))
    else:  # pragma: no cover - explicit failure beats a silent wrong number
        raise ValueError(f"unsupported stat: {stat}")
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_cluster_delta(
    a_by_seq: Dict[str, np.ndarray],
    b_by_seq: Dict[str, np.ndarray],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired difference a-b with sequence-clustered CI (used for E3/E4).

    The paired design is the point: both methods are evaluated on the same
    objects, so the difference is measured per object and only the *sequence*
    structure is resampled.
    """
    seqs = [s for s in a_by_seq if s in b_by_seq]
    if not seqs:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    diff = {s: np.atleast_1d(a_by_seq[s]) - np.atleast_1d(b_by_seq[s]) for s in seqs}
    point, lo, hi = cluster_bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)
    return {"delta": point, "lo": lo, "hi": hi, "n_seq": float(len(seqs))}


def tost_equivalence(
    a_by_seq: Dict[str, np.ndarray],
    b_by_seq: Dict[str, np.ndarray],
    margin: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Two one-sided test for equivalence within +-margin, clustered bootstrap.

    Used to *reject* H3 ("risk-only is as good as hybrid") rather than to
    confirm superiority: equivalence at a pre-registered margin means the
    predictive term did not earn its cost.
    """
    d = paired_cluster_delta(a_by_seq, b_by_seq, n_boot=n_boot, alpha=alpha, seed=seed)
    lo, hi = d["lo"], d["hi"]
    equivalent = (lo > -margin) and (hi < margin)
    return {
        "delta": d["delta"],
        "lo": lo,
        "hi": hi,
        "margin": margin,
        "equivalent": 1.0 if equivalent else 0.0,
    }
