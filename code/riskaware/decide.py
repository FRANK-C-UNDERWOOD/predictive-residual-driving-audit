"""LEGACY v0.1 decision rules; retained for historical synthetic runs.

Use protocol_stats.py for the revised manuscript. The legacy H1 estimand,
sequence weighting and H3 interval convention differ from protocol v0.3.

The paper pre-registers three decision rules.  If those rules live only in prose,
they can be read differently after the results arrive.  This module makes them
code: each function takes the per-object outcome table, applies exactly the rule
stated in docs/01, and returns a Verdict with the statistic, its sequence-clustered
interval, and the rule text that produced the verdict.

Input convention
----------------
Every comparison is a mapping  seq -> array of per-object outcomes (0/1 for timely
success, or any real number such as latency).  Pairing is at the object level:
both members of a pair must come from the same sequence and refer to the same
evaluation unit.  Sequences are the resampling unit, never frames.

Nothing here decides what the paper claims; it decides what the pre-registered
rule says.  A rule that fires on noise is a bug, which is why
scripts/validate_instrument.py measures the false-positive rate of these
functions on known-null data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics


@dataclass(frozen=True)
class DecisionConfig:
    """Frozen before the test split is opened (code/configs/default.json)."""

    alpha: float = 0.05
    n_boot: int = 1000
    seed: int = 0
    h3_margin: float = 0.02
    min_pairs: int = 30
    min_seq: int = 3   # below this the clustered bootstrap is not a distribution


@dataclass
class Verdict:
    hypothesis: str
    supported: Optional[bool]          # None -> inconclusive / insufficient data
    status: str                        # supported | not-supported | equivalent | insufficient-data
    statistic: Dict[str, float] = field(default_factory=dict)
    rule: str = ""
    note: str = ""

    def line(self) -> str:
        return "%s: %s (%s)" % (self.hypothesis, self.status, self.rule)

    def to_dict(self) -> Dict[str, object]:
        return {
            "hypothesis": self.hypothesis,
            "status": self.status,
            "supported": self.supported,
            "rule": self.rule,
            "note": self.note,
            "statistic": self.statistic,
        }


# ---------------------------------------------------------------- matching --
def match_pairs(
    rows: Sequence[Dict[str, object]],
    treated_key: str = "is_key",
    covariate: str = "dense_detected",
    caliper: float = 0.0,
) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    """1:1 greedy matching of treated rows to control rows inside a sequence.

    Used for H1: match each key object to a non-key object with the same
    dense-model detectability (the confound is "key objects are just harder"),
    within the same sequence.  Deterministic: rows are sorted by object id.

    caliper is expressed in units of the covariate's standard deviation; 0 means
    exact matching on the covariate value.
    """
    by_seq: Dict[str, Dict[bool, List[Dict[str, object]]]] = {}
    for r in rows:
        seq = str(r["seq"])
        is_key = bool(int(float(r[treated_key])))  # type: ignore[arg-type]
        by_seq.setdefault(seq, {True: [], False: []})
        by_seq[seq][is_key].append(r)

    pairs: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    for seq, groups in by_seq.items():
        treated = sorted(groups[True], key=lambda r: float(r["obj_id"]))
        controls = sorted(groups[False], key=lambda r: float(r["obj_id"]))
        values = np.array([float(r[covariate]) for r in rows if covariate in r], dtype=float)
        sd = float(values.std()) if values.size > 0 else 0.0
        tol = caliper * sd
        used: set = set()
        for t in treated:
            tv = float(t[covariate])
            best_i, best_d = -1, None
            for i, c in enumerate(controls):
                if i in used:
                    continue
                d = abs(float(c[covariate]) - tv)
                if d > tol:
                    continue
                if best_d is None or d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0:
                used.add(best_i)
                pairs.append((t, controls[best_i]))
    return pairs


def _by_seq(pairs: Sequence[Tuple[Dict[str, object], Dict[str, object]]], field_a: str, field_b: str):
    a: Dict[str, List[float]] = {}
    b: Dict[str, List[float]] = {}
    for t, c in pairs:
        seq = str(t["seq"])
        a.setdefault(seq, []).append(float(t[field_a]))  # type: ignore[arg-type]
        b.setdefault(seq, []).append(float(c[field_b]))  # type: ignore[arg-type]
    return {k: np.array(v) for k, v in a.items()}, {k: np.array(v) for k, v in b.items()}


# ------------------------------------------------------------------- H1 -----
def h1_selective_loss(
    pairs: Sequence[Tuple[Dict[str, object], Dict[str, object]]],
    outcome_field: str = "timely",
    cfg: DecisionConfig = DecisionConfig(),
) -> Verdict:
    """H1: key objects lose MORE timely detections than matched controls.

    Rule (docs/01): supported iff (i) the paired gap key-minus-control is
    negative with a sequence-clustered 95% interval excluding zero, and (ii) at
    least min_pairs pairs are available.  A uniform loss across object types
    fails the rule, as it must: sparse computation costs something everywhere.
    """
    if len(pairs) < cfg.min_pairs:
        return Verdict(
            hypothesis="H1",
            supported=None,
            status="insufficient-data",
            rule="paired key-minus-control gap < 0 with clustered 95% CI excluding 0",
            statistic={"n_pairs": float(len(pairs)), "min_pairs": float(cfg.min_pairs)},
        )
    a, b = _by_seq(pairs, outcome_field, outcome_field)
    d = metrics.paired_cluster_delta(a, b, n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed)
    supported = d["delta"] < 0.0 and d["hi"] < 0.0
    return Verdict(
        hypothesis="H1",
        supported=supported,
        status="supported" if supported else "not-supported",
        statistic={
            "delta_key_minus_control": d["delta"],
            "lo": d["lo"],
            "hi": d["hi"],
            "n_pairs": float(len(pairs)),
            "n_seq": d.get("n_seq", float("nan")),
        },
        rule="paired key-minus-control gap < 0 with clustered 95% CI excluding 0",
        note="" if supported else "a non-negative or non-significant gap does not support H1",
    )


# ------------------------------------------------------------------- H2 -----
def h2_delta(
    method: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    cfg: DecisionConfig = DecisionConfig(),
) -> Verdict:
    """H2: the hybrid beats residual-only at matched budget.

    Rule: supported iff the lower bound of the paired, sequence-clustered 95%
    interval for hybrid-minus-residual-only is above zero.
    """
    n_seq = min(len(method), len(baseline))
    if n_seq < cfg.min_seq:
        return Verdict(
            hypothesis="H2",
            supported=None,
            status="insufficient-data",
            statistic={"n_seq": float(n_seq), "min_seq": float(cfg.min_seq)},
            rule="hybrid-minus-residual-only paired clustered 95% CI lower bound > 0",
            note="fewer independent sequences than the clustered bootstrap requires",
        )
    d = metrics.paired_cluster_delta(method, baseline, n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed)
    finite = bool(np.isfinite(d["lo"]) and np.isfinite(d["hi"]))
    supported = finite and d["lo"] > 0.0
    return Verdict(
        hypothesis="H2",
        supported=supported if finite else None,
        status=("supported" if supported else "not-supported") if finite else "insufficient-data",
        statistic={"delta_hybrid_minus_residual": d["delta"], "lo": d["lo"], "hi": d["hi"],
                   "n_seq": d.get("n_seq", float("nan"))},
        rule="hybrid-minus-residual-only paired clustered 95% CI lower bound > 0",
    )


# ------------------------------------------------------------------- H3 -----
def h3_complementarity(
    method: Dict[str, np.ndarray],
    risk_only: Dict[str, np.ndarray],
    cfg: DecisionConfig = DecisionConfig(),
) -> Verdict:
    """H3: prediction and risk are complementary, so the hybrid beats risk-only.

    Rule: TOST against the pre-registered margin m.  If the two-sided interval
    lies inside [-m, m] the two are equivalent and H3 is REJECTED -- this is the
    outcome that says the predictor did not earn its cost.
    """
    n_seq = min(len(method), len(risk_only))
    if n_seq < cfg.min_seq:
        return Verdict(
            hypothesis="H3",
            supported=None,
            status="insufficient-data",
            statistic={"n_seq": float(n_seq), "min_seq": float(cfg.min_seq)},
            rule="hybrid-minus-risk-only outside the +/- %.2f equivalence band on the favourable side" % cfg.h3_margin,
            note="fewer independent sequences than the clustered bootstrap requires",
        )
    d = metrics.tost_equivalence(
        method, risk_only, margin=cfg.h3_margin, n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed
    )
    if not (np.isfinite(d["lo"]) and np.isfinite(d["hi"])):
        return Verdict(
            hypothesis="H3",
            supported=None,
            status="insufficient-data",
            statistic={"lo": d["lo"], "hi": d["hi"], "margin": cfg.h3_margin},
            rule="hybrid-minus-risk-only outside the +/- %.2f equivalence band on the favourable side" % cfg.h3_margin,
            note="non-finite interval",
        )
    equivalent = bool(d["equivalent"])
    width = float(d["hi"] - d["lo"])
    if equivalent:
        status, supported = "equivalent", False
    elif d["lo"] > cfg.h3_margin:
        status, supported = "supported", True
    elif d["hi"] < -cfg.h3_margin:
        # Significantly worse than risk-only: a clear negative, not "unresolved".
        status, supported = "not-supported", False
    elif width > 2.0 * cfg.h3_margin:
        # The interval is wider than the margin, so neither equivalence nor a
        # clear advantage is established.  Saying "not supported" here would be
        # wrong: the honest verdict is that the study cannot answer H3 yet.
        status, supported = "inconclusive", None
    else:
        status, supported = "not-supported", False
    return Verdict(
        hypothesis="H3",
        supported=supported,
        status=status,
        statistic={"delta_hybrid_minus_risk": d["delta"], "lo": d["lo"], "hi": d["hi"],
                   "margin": cfg.h3_margin, "equivalent": d["equivalent"]},
        rule="hybrid-minus-risk-only outside the +/- %.2f equivalence band on the favourable side" % cfg.h3_margin,
        note=("risk-only is equivalent to the hybrid: the predictor does not earn its cost"
              if equivalent else
              ("interval wider than the equivalence band: H3 is unresolved by this study"
               if status == "inconclusive" else "")),
    )


# ------------------------------------------------------------- reporting ----
def format_verdicts_md(verdicts: Sequence[Verdict], title: str = "Decision verdicts") -> str:
    lines = ["# " + title, "",
             "Generated by code/riskaware/decide.py from the per-object result table.",
             "Every verdict below is produced by the pre-registered rule stated in the",
             "'rule' column; changing a rule means changing this file, which is reviewed.", "",
             "| hypothesis | status | delta [95% CI] | rule |", "|---|---|---|---|"]
    for v in verdicts:
        s = v.statistic
        if "lo" in s and np.isfinite(s.get("lo", float("nan"))):
            delta = "%.3f [%.3f, %.3f]" % (s.get("delta_hybrid_minus_residual",
                                                 s.get("delta_hybrid_minus_risk",
                                                       s.get("delta_key_minus_control", float("nan")))),
                                           s["lo"], s["hi"])
        else:
            delta = "-"
        lines.append("| %s | **%s** | %s | %s |" % (v.hypothesis, v.status, delta, v.rule))
        if v.note:
            lines.append("| | | %s | |" % v.note)
    return chr(10).join(lines) + chr(10)
