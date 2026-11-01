"""v0.3 analysis primitives. Synthetic checks are not detector experiments.

The estimand is the event-weighted mean, conditional on the trained seeds.
Whole sequences are bootstrapped; seeds are averaged within the same event.
Superiority uses a two-sided 95% interval by default (directional tail 2.5%).
Equivalence uses a separate 90% bootstrap interval at TOST-style alpha=0.05.
Percentile containment is approximate, not a finite-sample exact TOST.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

VERSION = "protocol-v0.3"


@dataclass(frozen=True)
class AnalysisConfig:
    practical_margin: float
    ci_alpha: float = 0.05
    equivalence_alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    min_sequences: int = 3

    def __post_init__(self):
        if not np.isfinite(self.practical_margin) or self.practical_margin <= 0:
            raise ValueError("A positive, justified practical margin is required.")
        if not 0 < self.ci_alpha < 1 or not 0 < self.equivalence_alpha < 0.5:
            raise ValueError("Invalid interval/test alpha.")
        if self.n_boot < 100 or self.min_sequences < 2:
            raise ValueError("Insufficient resampling settings.")


@dataclass(frozen=True)
class CostEvidence:
    """Aggregate full cost in common declared units; never infer it from K."""
    mode: str
    target: float
    cost_a: float
    cost_b: float
    tolerance_fraction: float
    validation_calibration_id: str

    def eligible(self) -> bool:
        vals = (self.target, self.cost_a, self.cost_b, self.tolerance_fraction)
        if not all(np.isfinite(x) for x in vals):
            raise ValueError("Finite full-cost records are required.")
        if self.target <= 0 or min(self.cost_a, self.cost_b, self.tolerance_fraction) < 0:
            raise ValueError("Invalid target, cost or tolerance.")
        if not self.validation_calibration_id:
            raise ValueError("Record the frozen validation calibration identifier.")
        tol = self.target * self.tolerance_fraction + 1e-12
        return (
            self.mode == "matched-total-cost"
            and abs(self.cost_a - self.target) <= tol
            and abs(self.cost_b - self.target) <= tol
            and abs(self.cost_a - self.cost_b) <= tol
        )


def cluster_event_ci(
    values_by_seq: Mapping[str, Sequence[float]], *,
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Ratio of resampled event totals to counts, not mean of sequence means."""
    if not 0 < alpha < 1 or n_boot < 1:
        raise ValueError("Invalid resampling settings.")
    arrays = {str(k): np.asarray(v, dtype=float) for k, v in values_by_seq.items()}
    if not arrays or any(v.ndim != 1 or not v.size or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonempty finite one-dimensional event vectors required.")
    seqs = sorted(arrays)
    sums = np.array([arrays[k].sum() for k in seqs])
    counts = np.array([arrays[k].size for k in seqs], dtype=float)
    pick = np.random.default_rng(seed).integers(0, len(seqs), size=(n_boot, len(seqs)))
    estimates = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return dict(estimate=float(sums.sum() / counts.sum()), lo=float(lo), hi=float(hi),
                confidence=1-alpha, n_sequences=len(seqs), n_events=int(counts.sum()))


def _binary(value) -> float:
    result = float(value)
    if result not in (0.0, 1.0):
        raise ValueError("Timely outcomes must be 0 or 1.")
    return result


def _index(rows: Sequence[dict]) -> dict:
    out = {}
    for row in rows:
        key = (str(row["seq"]), str(row["event_id"]), int(row["seed"]))
        if key in out:
            raise ValueError("Duplicate (sequence,event,seed) record: %r" % (key,))
        out[key] = _binary(row["timely"])
    return out


def _collapse_seeds(values: dict) -> dict:
    by_event = {}
    for (seq, event, seed), value in sorted(values.items()):
        by_event.setdefault((seq, event), {})[seed] = float(value)
    seed_sets = {tuple(sorted(seed_values)) for seed_values in by_event.values()}
    if len(seed_sets) > 1:
        raise ValueError("Every event must have the same complete seed set.")
    by_seq = {}
    for (seq, _), seed_values in by_event.items():
        by_seq.setdefault(seq, []).append(float(np.mean(list(seed_values.values()))))
    return by_seq


def paired_event_difference(a_rows: Sequence[dict], b_rows: Sequence[dict]) -> dict:
    a, b = _index(a_rows), _index(b_rows)
    if set(a) != set(b):
        raise ValueError("Methods must have identical event/seed sets; do not silently drop failures.")
    return _collapse_seeds({key: a[key] - b[key] for key in a})


def _fingerprint(rows: Sequence[dict]) -> str:
    items = [(list(key), value) for key, value in sorted(_index(rows).items())]
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()


def did_values(pairs: Sequence[dict]) -> dict:
    """Matched key/ordinary event pairs with both dense and residual outcomes.

    Matching is fixed from allowed covariates upstream and recorded separately.
    Dense detectability is not imposed as the matching criterion here.
    """
    values, key_used, ordinary_used = {}, set(), set()
    for row in pairs:
        seq, seed = str(row["seq"]), int(row["seed"])
        key = (seq, str(row["pair_id"]), seed)
        key_event = (seq, str(row["key_event_id"]), seed)
        ordinary_event = (seq, str(row["ordinary_event_id"]), seed)
        if key in values or key_event in key_used or ordinary_event in ordinary_used:
            raise ValueError("Duplicate pair or reused event in 1:1 matching.")
        if key_event == ordinary_event:
            raise ValueError("Key and ordinary events must be distinct.")
        key_used.add(key_event)
        ordinary_used.add(ordinary_event)
        values[key] = (
            _binary(row["key_dense"]) - _binary(row["key_residual"])
            - _binary(row["ordinary_dense"]) + _binary(row["ordinary_residual"])
        )
    return _collapse_seeds(values)


def _interval(values: dict, cfg: AnalysisConfig, alpha: float) -> dict:
    return cluster_event_ci(values, n_boot=cfg.n_boot, alpha=alpha, seed=cfg.seed)


def _verdict(name: str, status: str, supported, **details) -> dict:
    return dict(protocol_version=VERSION, hypothesis=name, status=status,
                supported=supported, **details)


def h1_did(pairs: Sequence[dict], cfg: AnalysisConfig) -> dict:
    values = did_values(pairs)
    if len(values) < cfg.min_sequences:
        return _verdict("H1", "insufficient-data", None, n_sequences=len(values))
    ci = _interval(values, cfg, cfg.ci_alpha)
    status = "supported" if ci["lo"] > 0 else "inconclusive"
    if ci["hi"] < 0:
        status = "opposite-direction"
    return _verdict("H1", status, True if status == "supported" else None, contrast="D1", interval=ci)


def h2_importance(a_rows: Sequence[dict], b_rows: Sequence[dict],
                  cfg: AnalysisConfig, evidence: CostEvidence) -> dict:
    if not evidence.eligible():
        return _verdict("H2", "unmatched-cost", None, evidence=asdict(evidence))
    values = paired_event_difference(a_rows, b_rows)
    if len(values) < cfg.min_sequences:
        return _verdict("H2", "insufficient-data", None, n_sequences=len(values))
    ci = _interval(values, cfg, cfg.ci_alpha)
    status = "supported" if ci["lo"] > 0 else "inconclusive"
    if ci["hi"] < 0:
        status = "opposite-direction"
    return _verdict("H2", status, True if status == "supported" else None,
                    contrast="D2", interval=ci, evidence=asdict(evidence),
                    method_fingerprint=_fingerprint(a_rows), configuration=asdict(cfg))


def h3_prediction(a_rows: Sequence[dict], b_rows: Sequence[dict],
                  cfg: AnalysisConfig, evidence: CostEvidence, *, h2_result: dict) -> dict:
    if not evidence.eligible():
        return _verdict("H3", "unmatched-cost", None, evidence=asdict(evidence))
    values = paired_event_difference(a_rows, b_rows)
    if len(values) < cfg.min_sequences:
        return _verdict("H3", "insufficient-data", None, n_sequences=len(values))
    ci = _interval(values, cfg, cfg.ci_alpha)
    eq_ci = _interval(values, cfg, 2 * cfg.equivalence_alpha)
    equivalent = eq_ci["lo"] > -cfg.practical_margin and eq_ci["hi"] < cfg.practical_margin
    h2_supported = (
        h2_result.get("protocol_version") == VERSION
        and h2_result.get("hypothesis") == "H2"
        and h2_result.get("status") == "supported"
        and h2_result.get("method_fingerprint") == _fingerprint(a_rows)
        and h2_result.get("configuration") == asdict(cfg)
        and h2_result.get("evidence", {}).get("validation_calibration_id") == evidence.validation_calibration_id
        and h2_result.get("evidence", {}).get("target") == evidence.target
        and h2_result.get("evidence", {}).get("cost_a") == evidence.cost_a
    )
    if equivalent:
        status, supported = "equivalent", False
    elif ci["hi"] < -cfg.practical_margin:
        status, supported = "inferior", False
    elif ci["lo"] > cfg.practical_margin and h2_supported:
        status, supported = "supported", True
    elif ci["lo"] > cfg.practical_margin:
        status, supported = "d3-positive-h2-unresolved", None
    else:
        status, supported = "inconclusive", None
    return _verdict("H3", status, supported, contrast="D3", interval=ci,
                    equivalence_interval=eq_ci, practical_margin=cfg.practical_margin,
                    equivalent=equivalent, h2_supported=h2_supported,
                    evidence=asdict(evidence))
