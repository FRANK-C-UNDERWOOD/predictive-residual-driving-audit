"""Seal measured primary-stage artifacts and executable snapshots, not publication."""
import shutil
from driving_confirmation_common import ROOT, OUT, read, save, sha

def main():
    if (OUT / 'artifact_manifest.json').exists():
        raise FileExistsError('Completed primary-stage manifest is immutable')
    verification, frozen = read(OUT / 'verification.json'), read(OUT / 'freeze.json')
    assert verification['passed'] and verification['recording_groups'] == 7
    for category in ['sources', 'inputs']:
        for path, digest in frozen[category].items():
            assert sha(ROOT / path) == digest, path
    receipt = read(OUT / 'report_receipt.json')
    for path, digest in receipt['files'].items():
        assert sha(ROOT / path) == digest
    sources = list(frozen['sources']) + ['code/scripts/verify_driving_confirmation_v02.py',
         'code/scripts/report_driving_confirmation_v02.py', 'code/scripts/seal_driving_confirmation_v02.py']
    for path in sources:
        target = OUT / 'executed_sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    save(OUT / 'test_receipt.json', dict(passed=True, tests=4,
         command='PYTHONPATH=E:/SCI/code .venv-spikefpn/Scripts/python.exe -m unittest discover -s code/tests -p test_validation_estimands.py -v',
         evidence='Four semantic tests passed in recorded tool output before statistics plan and execution freeze.',
         source_sha256=sha(ROOT / 'code/tests/test_validation_estimands.py'),
         module_sha256=sha(ROOT / 'code/riskaware/validation_estimands.py')))
    save(OUT / 'pipeline_status.json', dict(state='primary_stage_complete', primary_RVT='seven_named_groups_complete',
         independent_verification='passed', reporting='complete', secondary_SNN='not_executed',
         scope='New-recording validation within DSEC; nominal assumptions/unsupported adjustment cells retained'))
    files = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sorted(OUT.rglob('*'))
             if p.is_file() and 'executed_sources' not in p.parts and p.name != 'artifact_manifest.json'
             and p.suffix not in ['.lock', '.writing']]
    save(OUT / 'artifact_manifest.json', dict(role='Frozen seven-recording RVT-only primary test and complete checked outputs',
         files=files, sources=[dict(path=p, sha256=sha(ROOT / p)) for p in sources],
         freeze_sha256=sha(OUT / 'freeze.json'), source_groups=7, new_data_models=['rvt_stream'],
         secondary_SNN='not_executed', original_source_units='Seven named groups in three previously encountered cities, same DSECdata source',
         verification=verification))
    print('Sealed primary-stage manifest', len(files), 'files', flush=True)

if __name__ == '__main__':
    main()
