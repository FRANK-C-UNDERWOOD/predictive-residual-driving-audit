"""Typed full-cost receipts. This validates accounting, not operator execution."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

MODULES = ("bin", "front", "pred", "risk", "alloc", "cond", "maint", "io", "post")


@dataclass
class FullCostAudit:
    records: dict = field(default_factory=dict)

    def record(self, module: str, counts: dict[str, float], *, basis: str) -> None:
        if module not in MODULES or module in self.records:
            raise ValueError("Unknown or duplicate module.")
        if basis not in ("measured", "modeled", "not-required", "unmeasured"):
            raise ValueError("Declare evidence basis for every module.")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in counts.values()):
            raise ValueError("Operation counts must be finite and nonnegative.")
        if basis in ("not-required", "unmeasured") and counts:
            raise ValueError("Absent or unavailable work must not carry fabricated counts.")
        self.records[module] = dict(counts=dict(counts), basis=basis)

    def scalarize(self, weights: dict[str, float]) -> dict:
        missing = [m for m in MODULES if m not in self.records or self.records[m]["basis"] == "unmeasured"]
        if missing:
            return dict(total=None, complete=False, missing=missing)
        operation_types = {k for receipt in self.records.values() for k in receipt["counts"]}
        if not operation_types <= set(weights):
            raise ValueError("Every counted operation class needs a declared weight.")
        if any(not math.isfinite(float(weights[k])) or weights[k] < 0 for k in operation_types):
            raise ValueError("Invalid scalarization weight.")
        modules = {
            m: sum(float(v) * weights[k] for k, v in record["counts"].items())
            for m, record in self.records.items()
        }
        return dict(total=sum(modules.values()), modules=modules, complete=True, missing=[],
                    has_modeled_terms=any(x["basis"] == "modeled" for x in self.records.values()),
                    statement="Modeled scalar cost; not device energy or proof of skipped kernels.")


def net_saving(method: dict, dense: dict) -> float:
    if not method.get("complete") or not dense.get("complete") or dense["total"] <= 0:
        raise ValueError("Complete positive dense and complete method receipts required.")
    return 1 - method["total"] / dense["total"]
