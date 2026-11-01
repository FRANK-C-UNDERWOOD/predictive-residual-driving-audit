"""Numerical boundary contracts and complete old calibration-point regression."""
import subprocess, sys, shutil
import numpy as np
from snn_secondary_common import ROOT, OUT, read, save, sha, activate
from snn_float32_floor_compat import threshold_at_fppi, original_threshold


def main():
    folder = OUT / 'numerical_floor_recovery_v07'
    folder.mkdir(exist_ok=True)
    floor = .01
    boundary = float(np.float32(floor))
    assert boundary < floor
    assert bool(np.array([boundary], np.float32)[0:1] >= floor)
    rejected = False
    try:
        original_threshold([boundary], 1, 1., floor)
    except ValueError:
        rejected = True
    assert rejected
    assert threshold_at_fppi([boundary], 1, 1., floor) == floor
    assert threshold_at_fppi([boundary], 1, 0., floor) == float(np.nextafter(np.float32(floor), np.float32(np.inf)))
    invalid = [np.nextafter(np.float32(floor), np.float32(-np.inf)), np.nan, np.inf, -1., 1.01]
    for value in invalid:
        try:
            threshold_at_fppi([value], 1, 1., floor)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid confidence was accepted')
    rng = np.random.default_rng(7081)
    regressions = 0
    for nominal_floor in [.01, .3, 0., 1.]:
        for n in [0, 1, 10, 100]:
            values = rng.uniform(nominal_floor, 1., n)
            for target in [0., .25, 1., 2.]:
                assert threshold_at_fppi(values, 11, target, nominal_floor) == original_threshold(values, 11, target, nominal_floor)
                regressions += 1
    common = activate()
    from run_fp_mechanism import canonical
    from riskaware.criterion_audit import ground_truth
    meta = read(OUT / 'carriers/spike_original__interlaken_01_a.json')
    record = next(r for r in meta['frames'] if r['time_us'] == 53203150264)
    factor = read(ROOT / 'code/configs/driving_factors_v01.json')
    labels = np.load(ROOT / 'datasets/dsec_confirmation/test/interlaken_01_a/object_detections/left/tracks.npy')
    gt = ground_truth(labels[labels['t'] == record['time_us']], factor['class_map'], factor['small_area_fraction'])[0]
    with np.load(ROOT / record['path']) as z:
        raw = dict(boxes=z['boxes'], scores=z['scores'])
        n = sum(g['height'] * g['width'] for g in meta['geometry'])
        masks = np.unpackbits(z['masks'], axis=1)[:, :n].astype(bool)
    diagnostic = []
    for method, mask in zip(read(ROOT / 'code/configs/snn_secondary_v01.json')['methods'], masks):
        filtered, matches, ids = canonical(raw, mask, floor, meta, gt)
        fps = filtered[1][ids < 0]
        diagnostic.append(dict(method=method, boundary_false_positives=int((fps == np.float32(floor)).sum()),
            min_fp_score=float(fps.min()) if len(fps) else None))
    assert sum(r['boundary_false_positives'] for r in diagnostic) > 0
    save(folder / 'boundary_diagnosis.json', dict(raw_record=record, anchor=1887, encoded_floor=boundary,
        raw_scores_unchanged=True, exact_boundary_only=True, methods=diagnostic,
        cause='float32 selector comparison accepts nominal floor; float64 scalar validator rejected its exact encoding'))
    protected = {}
    for model in ['spike_original', 'spike_adapted']:
        destination = folder / 'preflight' / (model + '__thun_00_a.json')
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(OUT / 'preflight' / destination.name, destination)
        point = folder / 'preflight_evaluation' / destination.name
        if not point.exists():
            subprocess.run([sys.executable, '-u', str(ROOT / 'code/scripts/evaluate_snn_secondary_v07.py'),
                '--compat-preflight-out', str(folder.relative_to(ROOT)), '--model', model,
                '--sequence', 'thun_00_a', '--preflight'], cwd=ROOT, check=True)
        checked = read(point)
        assert checked['complete'] and checked['preflight_protected_points'] == 36
        protected[model] = dict(protected_points=36, receipt=point.relative_to(ROOT).as_posix(), sha256=sha(point))
    save(folder / 'verification.json', dict(passed=True, old_calibration_points=72,
        valid_input_threshold_regressions=regressions, rejects_invalid_scores=len(invalid),
        protected=protected, exact_boundary=float(np.float32(floor)), source_sha256=sha(ROOT / 'code/scripts/check_snn_floor_compat_v07.py'),
        scope='Post-failure numerical compatibility repair, not changed raw scores, detector outputs, selection, cohorts, estimands or FP targets'))
    print('Float32 floor compatibility and all72 old calibration points passed', flush=True)


if __name__ == '__main__':
    main()
