"""riskaware: minimal reference implementation for the paper

"Risk-Aware Predictive Suppression for Compute-Efficient Spiking Driving Perception".

This package is the *harness* that the experiments in docs/01_实验方案与实验协议.md
are meant to run against.  It contains no trained network and no dataset loader:
it implements the parts that must be identical across all compared methods, so
that the comparison is fair:

  metrics.py   -- timely-detection, latency, AP/mAP, clustered bootstrap CIs
  gates.py     -- predictive / risk / random / event-rate / hybrid gating under
                  one common compute-budget interface
  risk.py      -- causal, geometry-based risk proxy (never uses the detector)
  labels.py    -- key-object assignment and onset-time rules (label side only)
  cost.py      -- per-module operation ledger and break-even analysis
  synthetic.py -- toy sequences used only for pipeline sanity checks
  plotting.py  -- dependency-light plotting (Pillow) for sanity figures

Everything here is deterministic given a seed.
"""

__version__ = "0.1.0"
