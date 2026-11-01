"""Seal completed subgroup measurements without rewriting prior experiments."""
import shutil
from confirmation_common import ROOT, read, save, sha

OUT = ROOT / 'experiments/driving_factors_v01'

def main():
    if (OUT / 'artifact_manifest.json').exists():
        raise FileExistsError('Completed measurement manifest is immutable')
    verification = read(OUT / 'verification.json')
    assert verification['passed']
    receipt = read(OUT / 'report_receipt.json')
    for path, digest in receipt['files'].items():
        assert sha(ROOT / path) == digest
    sources = list(read(OUT / 'freeze.json')['sources']) + [
        'code/scripts/verify_driving_factors_v01.py', 'code/scripts/report_driving_factors_v01.py',
        'code/scripts/finalize_driving_factors_v01.py']
    for path in sources:
        target = OUT / 'executed_sources' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    complete = read(OUT / 'complete.json')
    complete['independent_verification_pending'] = False
    complete['independent_verification_passed'] = True
    save(OUT / 'complete.json', complete)
    save(OUT / 'test_receipt.json', dict(passed=True, tests=5,
         command='PYTHONPATH=E:/SCI/code .venv-spikefpn/Scripts/python.exe -m unittest discover -s code/tests -p test_driving_factors.py -v',
         source_sha256=sha(ROOT / 'code/tests/test_driving_factors.py'),
         module_sha256=sha(ROOT / 'code/riskaware/driving_factors.py'),
         evidence='Five named tests passed in the recorded tool output before annotation-rule freeze; not rerun as extra scientific samples.'))
    files = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sorted(OUT.rglob('*'))
             if p.is_file() and 'executed_sources' not in p.parts and p.name != 'artifact_manifest.json']
    save(OUT / 'artifact_manifest.json', dict(role='Completed retrospective annotation factors, continuity and source-stratum joins; not independent external validation',
         files=files, sources=[dict(path=p, sha256=sha(ROOT / p)) for p in sources],
         parent_manifest_sha256=sha(ROOT / 'experiments/fp_mechanism_v01/artifact_manifest.json'),
         independent_verification=verification, protected_prior_numeric_receipts=32,
         prospective_design_path='code/configs/driving_confirmation_v02.json',
         prospective_design_sha256=sha(ROOT / 'code/configs/driving_confirmation_v02.json')))
    print('Sealed subgroup measurement artifacts', len(files), 'files and', len(sources), 'sources', flush=True)

if __name__ == '__main__':
    main()
