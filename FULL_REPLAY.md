# Full-model replay: assets and limits

The fast CPU entry is a count-level reproduction. The archived detector workers require the following additional assets and runtime. These requirements are explicit rather than hidden behind a claim that every archived command is a one-click fresh run.

## Sources and runtime

- SpikeFPN source revision: `90fc523215f6121d9dd9c6fc63ae9225361af8cb` (bundled source, GPLv3).
- RVT source revision: `b80f5683a6e2d5de65d4bde8105d796ccb50dbb1` (bundled source, MIT).
- Native execution environment: Windows, Python 3.12, CUDA PyTorch 2.3.1+cu121 / torchvision 0.18.1+cu121; TF32 disabled, deterministic setup in the workers. `requirements-full-observed.txt` lists observed project packages. `requirements-rvt.txt` lists the separately loaded pure-Python RVT dependencies. Install the latter into the environment or the original `vendor/rvt_runtime_deps` target directory. No cross-platform or different-driver bit-exact guarantee is made.

## Weights

The original SpikeFPN checkpoint belongs at `vendor/spikefpn_snapshot/weights/gad_weight.pth`. Get it from the pinned upstream repository (`_UPSTREAM.json` has the SHA256; the confirmation config also records it). The official RVT Tiny 1Mpx checkpoint belongs at `vendor/rvt_snapshot/weights/rvt-t-1mpx.ckpt`, from https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-t.ckpt ; required SHA256 is `4298fb5290e3bf6bb0914f37dacb402ed8b759236ad5d211c25056fae7b45ab7`.

S1 additionally requires the selected116 parameter delta named in `experiments/diverse_adaptation_v01/full_conv_training.json`. It is not a second upstream pretrained checkpoint. Rebuilding it uses the fixed 348-update adaptation recipe on Zurich00/03/05, internal selection on Zurich06, and Thun00 calibration. Weights are intentionally not redistributed by this package. Without that asset or training reconstruction, do not label an S0 run S1. `code/configs/diverse_adaptation_v01.json` is authoritative.

## Event data and annotations

Provider entry points: https://dsec.ifi.uzh.ch/dsec-detection/ and https://github.com/uzh-rpg/dsec-det . Use the detection extension's distorted native coordinates; do not silently substitute rectified events/boxes. Relevant official download script: https://github.com/uzh-rpg/dsec-det/blob/master/scripts/download_dsec_extra.sh .

Expected paths:

```
datasets/dsec_pilot/train/<sequence>/events/left/events.h5
datasets/dsec_pilot/train/<sequence>/object_detections/left/tracks.npy
datasets/dsec_pilot/train/<sequence>/image_timestamps.txt
datasets/dsec_confirmation/test/<sequence>/events/left/events.h5
datasets/dsec_confirmation/test/<sequence>/object_detections/left/tracks.npy
datasets/dsec_confirmation/test/<sequence>/image_timestamps.txt
```

Use train paths for Thun00 calibration and the discovery recordings Zurich07_a/08_a/09_a/10_a, plus the named adaptation inputs. The seven new detection-test groups are thun_01_a, thun_02_a, interlaken_01_a and zurich_city_12_a/13_a/14_a/15_a. Provider/member URLs, native timestamp/label hashes, event CRC/SHA and source receipts are archived in the corresponding experiment directories. Segments from one recording are not independent groups. The original workstation released the Thun01 raw event member under its storage policy; this public release contains no raw member for any sequence.

## Actual entry points

- `cache_confirmation_spike.py` / `cache_confirmation_rvt.py`: original discovery full forwards and causal masks, including calibration/preflight roles.
- `run_fp_mechanism.py --phase extract` then `--phase points`: discovery raw candidate and actual-FP replay; run only after parent assets exist.
- `cache_driving_confirmation.py --model rvt_stream --sequence <name>`: new-recording RVT full forward; then `evaluate_driving_confirmation.py`, `summarize_driving_confirmation.py` and `verify_driving_confirmation_v02.py`.
- `cache_snn_secondary.py --sequence <name>`: paired S0/S1 full forwards. Follow with `evaluate_snn_secondary_v07.py --model <model> --sequence <name>` and `join_snn_secondary.py`, then the SNN summarizer/verifier.

The SNN v07 entry installs the documented exact-float32-floor validation adapter. Do not revert to a failing validation path or alter raw scores to bypass it. Full-grid predictors update after scoring everywhere. Every detector and state update still runs; masks only admit candidates before NMS. SpikeFPN uses nine 20ms signed bins and per-query neuron reset; the RVT audit uses a continuing explicit-state stream, first packet 50ms and later native query intervals, ten histogram bins per polarity (20 channels), not the older three-packet/reset reference mode.

Original freeze assertions require named parent manifests, calibration outputs and input assets. Public aggregate files alone do not satisfy these dependencies. The release retains guards rather than disabling integrity checks; consult the configuration/source records when reconstructing a clean run. Do not run the old serial acquisition orchestration against the sealed archive to overwrite results or delete downloaded members. Full replay can take hours and has not been rerun merely to package this release.
