"""Compute-allocation gates under one common budget interface.

The single most important design choice in this file: **every gate returns a
mask with exactly the same number of selected regions** in the equal-capacity
diagnostic. This is not equal total cost. Full-cost matching is checked separately
by protocol_stats.CostEvidence.

Interface
---------
Each gate receives the information available at macro-step t and returns a
boolean mask over regions plus the scores it used.  Gates differ ONLY in the
score they rank by; budgeting, tie-breaking and the self-masking guard are
shared code in BudgetAllocator.

Gates
-----
FullDense       keep everything (cost/performance reference, not a baseline)
RandomGate      random top-k                  (is the gain just dropping redundancy?)
EventRateGate   number of events per region   (is a trivial dynamic policy enough?)
ResidualGate    predictive residual |h - h'|  (the original direction-two idea)
RiskGate        risk proxy only              (does prediction earn its cost?)
HybridGate      w_e*residual + w_r*risk      (the proposed method)

Guards
------
max_suppressed_steps is a legacy name for a SOFT age-priority threshold.
When stale demand exceeds capacity, the oldest regions are selected and overflow
is recorded. There is no hard maximum-suppression guarantee. Coarse residuals
remain available everywhere when the frontend is always on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------------------------------------- normalisation --
def robust_norm(x: np.ndarray, q: float = 0.99) -> np.ndarray:
    """Scale by a high quantile so a single outlier region cannot dominate."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    s = float(np.quantile(x, q))
    if not np.isfinite(s) or s <= 1e-12:
        return np.zeros_like(x)
    return np.clip(x / s, 0.0, 1.0)


# ---------------------------------------------------------------- budget ----
@dataclass
class BudgetAllocator:
    """Shared selected-region capacity; auxiliary and executed costs can differ."""

    keep_ratio: float = 0.5
    max_suppressed_steps: int = 4

    def n_keep(self, n_regions: int) -> int:
        return int(max(1, min(n_regions, round(self.keep_ratio * n_regions))))

    def select(
        self,
        scores: np.ndarray,
        n_regions: int,
        age: Optional[np.ndarray] = None,
        forced: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a bool mask with exactly n_keep(True).

        Ranking is by descending score; ties break on index so the result is
        deterministic. Stale regions are considered first by descending age.
        When stale demand exceeds capacity, only the oldest fit.
        """
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != n_regions:
            raise ValueError(f"expected {n_regions} scores, got {scores.size}")
        k = self.n_keep(n_regions)
        if not np.isfinite(scores).all():
            raise ValueError("Scores must be finite.")
        ages = np.zeros(n_regions) if age is None else np.asarray(age).reshape(-1)
        if ages.size != n_regions or not np.isfinite(ages).all() or np.any(ages < 0):
            raise ValueError("Invalid region ages.")
        stale = (ages >= self.max_suppressed_steps) if self.max_suppressed_steps > 0 else np.zeros(n_regions, dtype=bool)
        if forced is not None:
            explicit = np.asarray(forced).reshape(-1)
            if explicit.size != n_regions:
                raise ValueError("Invalid explicit priority mask.")
            stale = stale | (explicit > 0)
        indices = np.arange(n_regions)
        old = indices[stale]
        old = old[np.lexsort((old, -ages[old]))]
        other = indices[~stale]
        other = other[np.lexsort((other, -scores[other]))]
        order = np.concatenate((old, other))
        mask = np.zeros(n_regions, dtype=bool)
        mask[order[:k]] = True
        return mask


# ------------------------------------------------------------ score gates ---
class Gate:
    """Base gate: maps observations to a per-region score."""

    name = "gate"

    def score(self, obs: "Observation") -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return "<" + self.name + ">"


@dataclass
class Observation:
    """Everything a gate is allowed to see at step t.

    Deliberately explicit: a gate that needs more than this (future frames,
    ground truth, detector logits) cannot be implemented without editing this
    dataclass, which would show up in review.
    """

    step: int
    features: np.ndarray            # (N, D) current coarse features
    prediction: np.ndarray          # (N, D) predictor output, same space
    event_counts: np.ndarray        # (N,) events per region this step
    risk: np.ndarray                # (N,) causal risk proxy in [0, 1]
    age: np.ndarray                 # (N,) steps since last processed
    residual: Optional[np.ndarray] = None  # (N,) filled by pipeline before gating


class FullDense(Gate):
    name = "full-dense"

    def score(self, obs: Observation) -> np.ndarray:
        return np.ones(obs.features.shape[0])


class RandomGate(Gate):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def score(self, obs: Observation) -> np.ndarray:
        return self.rng.random(obs.features.shape[0])


class EventRateGate(Gate):
    name = "event-rate"

    def score(self, obs: Observation) -> np.ndarray:
        return robust_norm(obs.event_counts)


class ResidualGate(Gate):
    name = "residual-only"

    def score(self, obs: Observation) -> np.ndarray:
        r = obs.residual
        if r is None:
            r = np.linalg.norm(obs.features - obs.prediction, axis=1)
        return robust_norm(r)


class RiskGate(Gate):
    name = "risk-only"

    def score(self, obs: Observation) -> np.ndarray:
        return robust_norm(obs.risk)


@dataclass
class HybridGate(Gate):
    """Proposed gate: residual + risk, each robustly normalised.

    w_e = 0 degenerates to RiskGate and w_r = 0 to ResidualGate, which is how the
    ablation (E4) is implemented -- one code path, three behaviours.
    """

    w_e: float = 0.6
    w_r: float = 0.4
    name: str = "hybrid"

    def score(self, obs: Observation) -> np.ndarray:
        r = obs.residual
        if r is None:
            r = np.linalg.norm(obs.features - obs.prediction, axis=1)
        return self.w_e * robust_norm(r) + self.w_r * robust_norm(obs.risk)


# -------------------------------------------------------------- pipeline ----
@dataclass
class GateDecision:
    mask: np.ndarray
    scores: np.ndarray
    n_selected: int
    n_regions: int
    forced: int = 0
    stale_eligible: int = 0
    stale_overflow: int = 0

    @property
    def retained_fraction(self) -> float:
        return self.n_selected / max(1, self.n_regions)


@dataclass
class GateController:
    """Gate + budget + staleness bookkeeping."""

    gate: Gate
    allocator: BudgetAllocator = field(default_factory=BudgetAllocator)

    def decide(self, obs: Observation) -> GateDecision:
        n = obs.features.shape[0]
        scores = np.asarray(self.gate.score(obs), dtype=np.float64).reshape(-1)
        stale = (obs.age >= self.allocator.max_suppressed_steps) if self.allocator.max_suppressed_steps > 0 else np.zeros(n, dtype=bool)
        k = self.allocator.n_keep(n)
        mask = self.allocator.select(scores, n, age=obs.age)
        return GateDecision(
            mask=mask,
            scores=scores,
            n_selected=int(mask.sum()),
            n_regions=n,
            forced=int(np.sum(mask & stale)),
            stale_eligible=int(stale.sum()),
            stale_overflow=int(np.sum(stale & ~mask)),
        )


# ------------------------------------------------------------ state keep ----
@dataclass
class StateTracker:
    """What happens to a region that is *not* selected.

    Two mechanisms, both ablatable (E4 "state retention" arm):
      decay         -- feature state decays geometrically instead of being zeroed
      staleness     -- age increments, and feeds the self-masking guard
      coast_decay   -- confidence of a coasting (unupdated) track decays with age

    The paper must show that "not updated" is not silently read as "no object".
    """

    decay: float = 0.8
    age: np.ndarray = field(default_factory=lambda: np.zeros(0))
    state: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def reset(self, n_regions: int, dim: int) -> None:
        self.state = np.zeros((n_regions, dim), dtype=np.float64)
        self.age = np.zeros(n_regions, dtype=np.float64)

    def update(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.state.shape != features.shape:
            self.reset(features.shape[0], features.shape[1])
        self.state = np.where(mask[:, None], features, self.decay * self.state)
        self.age = np.where(mask, 0.0, self.age + 1.0)
        return self.state

    def coast_confidence(self, base: float = 1.0, floor: float = 0.05) -> np.ndarray:
        return np.maximum(floor, base * np.exp(-self.age / 3.0))


def apply_mask(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-out suppressed regions before they reach the detector.

    This helper only returns a zero-masked dense array. It does NOT establish
    conditional operator skipping, a runtime speedup, or saved executed work.
    """
    return np.where(mask[:, None], features, 0.0)
