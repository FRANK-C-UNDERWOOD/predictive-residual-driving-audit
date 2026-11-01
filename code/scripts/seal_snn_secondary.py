"""Seal secondary measurements and actual source snapshots, without publication."""
import shutil
from snn_secondary_common import ROOT, OUT, CONFIG, read, save, sha, assert_frozen

def main():
    assert not (OUT / 'artifact_manifest.json').exists()
    frozen = assert_frozen()
    verified = read(OUT / 'verification.json')
    assert verified['passed'] and verified['pair_receipts'] == 14
    reporter = read(OUT / 'report_receipt.json')
    for path, digest in reporter['files'].items():
        assert sha(ROOT / path) == digest, path
    numerical = read(OUT / 'numerical_floor_compat_v07.json')
    assert sha(ROOT / numerical['failed_attempt_archive']) == numerical['archive_sha256']
    assert sha(ROOT / numerical['verification_path']) == numerical['verification_sha256']
    for path, digest in numerical['prior_complete_point_hashes'].items():
        assert sha(ROOT / path) == digest, path
    sources = set(frozen['sources'])
    for name in ['transport_freeze.json','verification_source_freeze.json',
                 'transport_recovery_v02.json','transport_direct_route_v03.json',
                 'transport_tail_recovery_v04.json','transport_covering_cache_v05.json','continuation_v06.json',
                 'numerical_floor_compat_v07.json']:
        addition = read(OUT / name)
        for path, digest in addition['sources'].items():
            assert sha(ROOT / path) == digest, path
            sources.add(path)
    sources.update(['code/scripts/report_snn_secondary.py','code/scripts/seal_snn_secondary.py',
                    'code/scripts/finish_snn_secondary_review.py'])
    for path in sorted(sources):
        dest = OUT / 'executed_sources' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
    save(OUT / 'pipeline_status.json', dict(state='complete', measurements='all_seven_pairs_complete',
         independent_verification='passed', reporting='complete', checkpoints=read(CONFIG)['models'], stateful_reuse='not_executed'))
    files = [dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(OUT.rglob('*'))
             if p.is_file() and 'executed_sources' not in p.parts and p.name!='artifact_manifest.json' and p.suffix not in ['.lock','.writing']]
    save(OUT / 'artifact_manifest.json', dict(role='Previously specified SNN secondary completion on same seven records, after RVT primary',
         files=files,sources=[dict(path=p,sha256=sha(ROOT/p)) for p in sorted(sources)],
         recording_groups=7,checkpoints=read(CONFIG)['models'],architecture_count=1,
         freeze_sha256=sha(OUT/'freeze.json'),verification=verified,
         statistics='Nominal secondary descriptive intervals; no multiplicity-controlled or causal size claim',stateful_reuse='not_executed'))
    print('SNN secondary artifact manifest sealed',len(files),'files',flush=True)

if __name__=='__main__':main()
