"""Trained full-model alignment on actual event windows, with explicit scope."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from riskaware.dsec_pilot import DSECPilotSequence
from riskaware.torch_spikefpn import (
    load_official_model, cache_shared, replay_dense, replay_selected, postprocess, SCALES, NAMES,
)


def flatten(value):
    if torch.is_tensor(value):
        return [value.detach().cpu().numpy().copy()]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in flatten(item)]
    return []


class Recorder:
    def __init__(self):
        self.phase = "original"
        self.values = {}

    @contextmanager
    def install(self, model):
        self.values = {}
        handles = []
        watched = {"feature.stem0", "feature.stem1"}
        watched |= {"feature.cells.%d" % i for i in range(10)}
        watched |= {"feature." + name for name in
                    ("connect_1", "connect_2", "connect_3", "up_conv1", "up_conv2",
                     "smooth1", "smooth2", "smooth3")}
        watched |= {head for _, head in NAMES.values()}
        for name, module in model.named_modules():
            if name not in watched:
                continue

            def before(mod, args, name=name):
                if ".smooth" in name and self.phase != "replay":
                    self.values.setdefault(name + ":input", []).append(flatten(args[0]))

            def after(mod, args, output, name=name):
                if ".smooth" in name and self.phase == "shared":
                    return
                arrays = flatten(output)
                arrays += flatten(getattr(mod, "mem", None))
                arrays += flatten(getattr(mod, "a", None))
                self.values.setdefault(name, []).append(arrays)

            handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def compare_arrays(left, right, atol, rtol):
    if len(left) != len(right):
        return dict(passed=False, reason="Array count mismatch")
    peak = 0.
    for a, b in zip(left, right):
        if a.shape != b.shape:
            return dict(passed=False, reason="Shape mismatch", shapes=[list(a.shape), list(b.shape)])
        if a.size:
            peak = max(peak, float(np.max(np.abs(a - b))))
        if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False):
            return dict(passed=False, max_absolute_error=peak)
    return dict(passed=True, max_absolute_error=peak)


def compare_traces(a, b, atol, rtol):
    if set(a) != set(b):
        return dict(passed=False, reason="Module trace key mismatch", keys=[sorted(a), sorted(b)])
    details = {}
    for key in a:
        left = [x for call in a[key] for x in call]
        right = [x for call in b[key] for x in call]
        details[key] = compare_arrays(left, right, atol, rtol)
    return dict(passed=all(item["passed"] for item in details.values()), modules=details)


def main():
    config = json.loads((ROOT / "code/configs/model_alignment_v01.json").read_text())
    atol, rtol = config["numeric"]["atol"], config["numeric"]["rtol"]
    model, provenance = load_official_model()
    if provenance["upstream_commit"] != config["source_commit"]:
        raise ValueError("Source differs from the declared acceptance configuration")
    reader = DSECPilotSequence(ROOT / "datasets/dsec_pilot", "thun_00_a")
    times = reader.annotation_times[[20, len(reader.annotation_times)//2, len(reader.annotation_times)-5]]
    cases = [reader.signed_stack(int(t)) for t in times]
    reader.close()
    cases.append((np.zeros((1, 3, 3, 256, 256), dtype=np.float32),
                  dict(case="synthetic zero-input reset", synthetic=True)))
    report = dict(
        scope="Full-model engineering alignment with official GEN1-trained weights; real DSEC inputs are transfer inputs",
        excluded_claims=config["claims_excluded"], settings=config, model=provenance, cases=[],
    )
    with torch.inference_mode():
        for array, metadata in cases:
            inputs = torch.from_numpy(array).cuda()
            original = Recorder()
            with original.install(model):
                dense = model(inputs)
            refactored = Recorder()
            with refactored.install(model):
                refactored.phase = "shared"
                cached = cache_shared(model, inputs)
                refactored.phase = "replay"
                replay, heads, dense_spikes = replay_dense(model, cached, capture=True)
            raw_check = compare_arrays(flatten(dense), flatten(replay), atol, rtol)
            layer_check = compare_traces(original.values, refactored.values, atol, rtol)
            post_check = compare_arrays(flatten_numpy(postprocess(model, dense)),
                                        flatten_numpy(postprocess(model, replay)), atol, rtol)
            generator = torch.Generator(device="cpu").manual_seed(config["masks"]["seed"])
            masks = {}
            for stride in SCALES:
                height, width = cached[stride][0].shape[-2:]
                k = max(1, int(height * width * config["masks"]["retention"]))
                masks[stride] = torch.randperm(height * width, generator=generator)[:k].sort().values.cuda()
            selected, sparse_heads, sparse_spikes, counts = replay_selected(
                model, cached, masks, capture=True)
            selected_checks = {}
            spike_mismatches = 0
            for stride in SCALES:
                idx = masks[stride]
                selected_checks[stride] = compare_arrays(
                    flatten(heads[stride][:, idx]), flatten(sparse_heads[stride]), atol, rtol)
                for expected, actual in zip(dense_spikes[stride], sparse_spikes[stride]):
                    expected = expected.permute(0, 2, 3, 1).reshape(1, -1, expected.shape[1])[:, idx]
                    spike_mismatches += int(torch.count_nonzero(expected != actual).item())
            site_offsets, anchor_offsets = 0, 0
            expected_selected = [[], [], [], []]
            for stride in SCALES:
                idx = masks[stride]
                n = heads[stride].shape[1]
                anchor_idx = (idx[:, None] * model.num_anchors
                              + torch.arange(model.num_anchors, device=idx.device)).flatten() + anchor_offsets
                for slot in (0, 1, 3):
                    expected_selected[slot].append(replay[slot][:, anchor_idx])
                expected_selected[2].append(replay[2][:, site_offsets + idx])
                site_offsets += n
                anchor_offsets += n * model.num_anchors
            selected_decode_check = compare_arrays(
                flatten(tuple(torch.cat(items, 1) for items in expected_selected)),
                flatten(selected), atol, rtol)
            all_indices = {s: torch.arange(heads[s].shape[1], device=inputs.device) for s in SCALES}
            sparse_all, _, all_spikes, _ = replay_selected(model, cached, all_indices, capture=True)
            sparse_all_check = compare_arrays(flatten(replay), flatten(sparse_all), atol, rtol)
            all_mismatches = 0
            for stride in SCALES:
                for expected, actual in zip(dense_spikes[stride], all_spikes[stride]):
                    expected = expected.permute(0, 2, 3, 1).reshape(1, -1, expected.shape[1])
                    all_mismatches += int(torch.count_nonzero(expected != actual).item())
            entry = dict(
                input=metadata, input_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                all_open_raw=raw_check, all_open_layers_and_states=layer_check,
                all_open_postprocess=post_check, selected_head_checks=selected_checks,
                selected_spike_mismatches=spike_mismatches, branch_counters=counts,
                selected_global_decode_check=selected_decode_check,
                all_sites_selected_backend=sparse_all_check,
                all_sites_selected_spike_mismatches=all_mismatches,
            )
            entry["passed"] = all((raw_check["passed"], layer_check["passed"], post_check["passed"],
                                   all(x["passed"] for x in selected_checks.values()),
                                   spike_mismatches == config["spike_mismatch_allowed"],
                                   selected_decode_check["passed"], sparse_all_check["passed"],
                                   all_mismatches == config["spike_mismatch_allowed"]))
            report["cases"].append(entry)
            print("Case %d: %s; partial spike mismatches %d" % (
                len(report["cases"]), entry["passed"], spike_mismatches), flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"])
    target = ROOT / "experiments/pilot_v01/fullmodel_alignment.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


def flatten_numpy(value):
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in flatten_numpy(item)]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
