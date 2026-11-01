"""Serial exact-member acquisition, paired SNN inference and all actual FP points."""
import argparse, os, subprocess, sys, traceback
from snn_secondary_common import ROOT, OUT, CONFIG, read, save, sha, assert_frozen

def main():
    cfg = read(CONFIG)
    assert_frozen()
    if (OUT / 'secondary_results.json').exists():
        print('Secondary measurements already complete; no rerun', flush=True)
        return
    lock = OUT / 'pipeline.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    phase = 'initializing'
    def run(script, *args):
        nonlocal phase
        phase = script + ' ' + ' '.join(args)
        save(OUT / 'pipeline_status.json', dict(state='running', phase=phase, pid=os.getpid()))
        subprocess.run([sys.executable, '-u', str(ROOT / 'code/scripts' / script), *args], cwd=ROOT, check=True)
    try:
        for seq in cfg['sequences']:
            completed = [OUT / 'points' / (model + '__' + seq + '.json') for model in cfg['models']]
            if all(p.exists() and read(p)['complete'] for p in completed):
                continue
            run('fetch_snn_secondary.py', '--sequence', seq)
            run('cache_snn_secondary.py', '--sequence', seq)
            for model in cfg['models']:
                run('evaluate_snn_secondary.py', '--model', model, '--sequence', seq)
                run('join_snn_secondary.py', '--model', model, '--sequence', seq)
            receipt = read(OUT / 'data_receipts' / (seq + '.json'))
            target = (ROOT / receipt['event']['path']).resolve()
            allowed = (ROOT / 'datasets/dsec_confirmation/test').resolve()
            expected = (allowed / seq / 'events/left/events.h5').resolve()
            target.relative_to(allowed)
            assert target == expected and seq in cfg['sequences']
            assert all(read(p)['complete'] for p in completed)
            if target.exists():
                assert sha(target) == receipt['event']['sha256']
                save(OUT / 'released_public_inputs' / (seq + '.json'), dict(path=target.relative_to(ROOT).as_posix(), sha256=receipt['event']['sha256'],
                     reason='Exact reacquired new public member only, after both SNN complete raw caches/72 actual points and annotation joins. Parent primary outputs/old data unchanged.'))
                target.unlink()
        run('summarize_snn_secondary.py')
        save(OUT / 'pipeline_status.json', dict(state='measurements_complete', independent_verification='pending', stateful_reuse='not_executed', pid=os.getpid()))
    except Exception as error:
        save(OUT / 'pipeline_status.json', dict(state='failed', phase=phase, error=repr(error), traceback=traceback.format_exc(), pid=os.getpid()))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
