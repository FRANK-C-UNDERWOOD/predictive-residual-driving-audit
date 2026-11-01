"""Preserve the failed attempt and pin a post-failure numerical-only amendment."""
import datetime, zipfile
from snn_secondary_common import ROOT, OUT, read, save, sha, assert_frozen


def main():
    assert_frozen()
    assert not (OUT / 'pipeline.lock').exists()
    proof = OUT / 'numerical_floor_recovery_v07/verification.json'
    assert read(proof)['passed'] and read(proof)['old_calibration_points'] == 72
    assert read(OUT / 'pipeline_status.json')['state'] == 'failed'
    archive = ROOT / 'provenance/snn_float32_floor_failure_v07.zip'
    files = [OUT / 'pipeline_status.json', OUT / 'review_status.json',
        OUT / 'numerical_floor_recovery_v07/boundary_diagnosis.json',
        ROOT / 'code/riskaware/calibration_audit.py', ROOT / 'code/scripts/evaluate_driving_confirmation.py',
        *sorted((OUT / 'points').glob('*__*.json')), *sorted((OUT / 'checks').glob('*__*.json'))]
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(ROOT).as_posix())
    sources = ['code/scripts/' + name + '.py' for name in ['snn_float32_floor_compat',
        'evaluate_snn_secondary_v07', 'run_snn_continuation_v07', 'check_snn_floor_compat_v07',
        'record_snn_floor_amendment_v07']]
    save(OUT / 'numerical_floor_compat_v07.json', dict(fixed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={path: sha(ROOT / path) for path in sources},
        failed_attempt_archive=archive.relative_to(ROOT).as_posix(), archive_sha256=sha(archive),
        verification_path=proof.relative_to(ROOT).as_posix(), verification_sha256=sha(proof),
        initial_freeze_sha256=sha(OUT / 'freeze.json'),
        observed_before_amendment='Four Thun01/02 checkpoint-recording pairs complete; Interlaken raw outputs cached, original replay failed before point production.',
        prior_complete_point_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((OUT / 'points').glob('*__*.json'))},
        amendment='Accept only the exact float32 encoding of the declared floor during float64 input validation. Raw probabilities, selector comparisons, returned threshold semantics, native model execution, cohorts, estimands and FP targets unchanged.',
        status='Post-failure numerical compatibility repair, not part of original pre-output freeze; all prior completed results retained'))
    print('Numerical-only amendment pinned; failed attempt and all four completed pairs preserved', flush=True)


if __name__ == '__main__':
    main()
