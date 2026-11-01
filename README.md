# Predictive residual driving audit

Reproducibility package for **When Do Predictive Residuals Preserve Detection Evidence? An Allocation Audit in Event-Based Driving Perception**. Version v0.19; manuscript preparation for IEEE T-IV, **not an accepted publication**. Repository owner: FRANK-C-UNDERWOOD. Paper authorship is not inferred from this GitHub handle.

## Run the numerical reproduction (CPU, no driving data or weights)

Use Python 3.12. From the repository root:

```sh
python -m venv .venv-audit
# Windows: .venv-audit/Scripts/activate
# Linux/macOS: source .venv-audit/bin/activate
python -m pip install -r requirements-audit.txt
python reproduce.py
```

The entry verifies the distributed file hashes, independently rebuilds the 21 recording interactions from integer TP/GT counts, reproduces their nominal recording-level intervals and all leave-one-recording-out summaries, and checks **864 discovery + 504 RVT primary + 1008 SNN secondary = 2376 actual operating points**. Outputs are `results/reproduced.json` and `results/interactions.csv`. The results directory is intentionally untracked.

Expected principal contrasts (percentage points):

- RVT image-motion primary: mean **-10.400851**, nominal interval **[-18.920328, -1.881375]**.
- Original SpikeFPN S0 size secondary: **-12.478871**, **[-22.445309, -2.512433]**.
- Adapted SpikeFPN S1 size secondary: **-7.840683**, **[-19.867238, +4.185873]**; uncertain, not confirmation.

Seven recordings underlie each estimate. Two SNN checkpoints do not create fourteen independent groups. Negative size interaction means a smaller residual-over-activity benefit on small targets, not systematic small-target harm. Reverse outcomes, actual false-positive counts and unmatched points remain in the complete tables. No multiplicity-controlled or safety claim is made.

## Three levels of reproduction

1. **Published aggregate arithmetic:** the default CPU command above; self-contained and tested. It does not recreate detections or independently validate the raw annotations.
2. **Candidate/NMS/identity replay:** the original evaluators and independent verifiers are included under `code/scripts`; raw cached candidate arrays and original annotations are external inputs, not distributed here. Their original names and checksums remain in the archived manifests.
3. **Full event-to-detector execution/training:** original source interfaces, configurations, model source snapshots and provenance are included. This requires provider data, official weights, the selected adaptation or its training reconstruction, and the native runtime. See `FULL_REPLAY.md`. No fresh GPU run of this exported checkout is claimed; CPU success must not be relabeled full-model reproduction.

## What is distributed

- `code/riskaware`, `code/scripts`, `code/configs`: inference interfaces, ranking, evaluation, original verification and training/configuration source.
- `vendor/spikefpn_snapshot` / `vendor/rvt_snapshot`: pinned source snapshots with upstream licenses; no weights.
- `experiments/*`: complete aggregate tables, actual operating points, original execution freezes, source/member receipts, verification receipts and figure outputs.
- `RELEASE_MANIFEST.json`: hashes of distributed files; original experiment manifests additionally reference deliberately absent raw assets. Missing raw assets are not declared corrupt or independently validated by this release.

The original worker files and their scientific freeze records are preserved, including the post-failure float32-floor validation amendment. Some historical audit entry points intentionally refuse overwriting sealed results and require their original assets. Do not delete freezes to force a successful verification. Full reconstruction belongs in a separate working copy/output directory.

## Availability, licenses and citations

Data are obtained from the original providers: https://dsec.ifi.uzh.ch/dsec-detection/ and https://github.com/uzh-rpg/dsec-det . No events, annotation arrays, checkpoints, virtual environments or credentials are included. Source snapshots retain upstream GPLv3 (SpikeFPN) and MIT (RVT) notices; project additions use GPLv3. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

Related original systems include SpikeFPN, RVT, Predictive Coding Light and DiffPC. **This audit is not a reproduction of a complete PCL or DiffPC training circuit.** DiffPC trains predictive-coding networks with sparse ternary communication; this work audits task evidence under fixed detector outputs. See the linked primary paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/aa31dc84098add7dd2ffdd20646f2043-Abstract-Conference.html .

Use release v0.19 and its immutable commit/checksum when referring to this package. No paper DOI or acceptance status is invented. Issues may report reproduction failures, including missing assets or unsupported environments. OpenAI Codex assisted with code, analysis and documentation; the archived measurements and explicit checks delimit the reported evidence.
