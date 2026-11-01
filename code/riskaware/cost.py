"""Operation accounting.  The paper may only claim what this ledger can prove.

Naming discipline (docs/04): unless a hardware cost model is attached, we say
"activity reduction" for spike counts, "compute reduction" for operation counts,
"estimated energy" only with an explicit model, and "measured energy" only with
a device and stated measurement boundary.

The ledger includes the predictor, the risk module, the gate, the data movement
implied by the mask, and any always-on dense path.  A method that saves 40% of
detector ops but pays 35% for the predictor is not a 40% saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ModuleCost:
    """Per-macro-step operation count for one module."""

    name: str
    ops: float
    note: str = ""


@dataclass
class CostLedger:
    """Accumulates operations across steps, kept separately per module.

    ops() is the forecast for one step: dense detector cost is proportional to
    the number of *processed* regions, always-on modules are not.
    """

    modules: Dict[str, float] = field(default_factory=dict)
    steps: int = 0
    n_regions: int = 0

    def add(self, name: str, ops: float) -> None:
        self.modules[name] = self.modules.get(name, 0.0) + float(ops)

    def add_step(
        self,
        processed_regions: int,
        n_regions: int,
        per_region_ops: Dict[str, float],
        dense_ops: Dict[str, float],
    ) -> None:
        """One macro-step.

        per_region_ops: modules whose cost scales with the number of processed
                        regions (conditional detector stages only). An always-on
                        coarse frontend belongs in dense_ops at its full cost.
        dense_ops:      modules that run regardless of the mask (predictor, risk
                        estimator, gate).  These are the ones that can eat the
                        entire saving.
        """
        self.steps += 1
        self.n_regions = n_regions
        share = processed_regions / max(1, n_regions)
        for name, ops in per_region_ops.items():
            self.add(name, ops * share * n_regions)
        for name, ops in dense_ops.items():
            self.add(name, ops)

    # ------------------------------------------------------------ summary ---
    @property
    def total_ops(self) -> float:
        return float(sum(self.modules.values()))

    def breakdown(self) -> List[ModuleCost]:
        return [ModuleCost(k, v) for k, v in sorted(self.modules.items())]

    def fraction(self, name: str) -> float:
        t = self.total_ops
        return self.modules.get(name, 0.0) / t if t > 0 else float("nan")

    def relative_to(self, reference_total_ops: float) -> Dict[str, float]:
        """Net compute reduction versus a dense reference run.

        Net reduction is the headline number: it is the only one that can
        support a "cheaper" claim.
        """
        ref = float(reference_total_ops)
        return {
            "total_ops": self.total_ops,
            "reference_ops": ref,
            "net_compute_reduction": 1.0 - self.total_ops / ref if ref > 0 else float("nan"),
        }


@dataclass
class SparsityStats:
    """Activity statistics, reported separately from compute."""

    spikes_by_layer: Dict[str, float] = field(default_factory=dict)
    regions_processed: float = 0.0
    region_steps: float = 0.0

    def add(self, layer: str, spikes: float) -> None:
        self.spikes_by_layer[layer] = self.spikes_by_layer.get(layer, 0.0) + spikes

    def add_step(self, processed: int, total: int) -> None:
        self.regions_processed += processed
        self.region_steps += total

    @property
    def activity_ratio(self) -> float:
        return self.regions_processed / self.region_steps if self.region_steps else float("nan")

    def total_spikes(self) -> float:
        return float(sum(self.spikes_by_layer.values()))


def break_even(
    dense_detector_ops_per_step: float,
    skipped_fraction: float,
    always_on_ops_per_step: float,
) -> Dict[str, float]:
    """When is the predictor worth running?

    saved = dense * skipped_fraction; the predictor/risk/gate cost must be below
    that.  The returned break_even_skipped_fraction is the minimum fraction of
    skipped detector work at which the whole scheme is neutral, i.e. the number
    that should appear in the paper's "operating range" figure.
    """
    saved = dense_detector_ops_per_step * skipped_fraction
    net = saved - always_on_ops_per_step
    be = always_on_ops_per_step / dense_detector_ops_per_step if dense_detector_ops_per_step > 0 else float("nan")
    return {
        "dense_ops": dense_detector_ops_per_step,
        "skipped_fraction": skipped_fraction,
        "saved_ops": saved,
        "always_on_ops": always_on_ops_per_step,
        "net_saved_ops": net,
        "break_even_skipped_fraction": be,
        "worth_it": 1.0 if net > 0 else 0.0,
    }
