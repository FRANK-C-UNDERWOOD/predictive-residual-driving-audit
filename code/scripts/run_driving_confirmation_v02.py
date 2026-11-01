"""Complete the frozen primary stage serially, retaining failures and all points."""
import os, subprocess, sys, time, traceback
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha, assert_frozen

def main():
    cfg = read(CONFIG)
    assert_frozen()
    if (OUT / 'primary_results.json').exists():
        print('Primary results already complete; no experiment rerun', flush=True)
        return
    lock = OUT / 'primary_pipeline.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    phase = 'initializing'
    def run(script, *args):
        nonlocal phase
        phase = script + ' ' + ' '.join(args)
        save(OUT / 'pipeline_status.json', dict(state='running', stage='primary_RVT', phase=phase, pid=os.getpid()))
        subprocess.run([sys.executable, '-u', str(ROOT / 'code/scripts' / script), *args], cwd=ROOT, check=True)
    try:
        # One large input acquisition was explicitly started before this orchestrator.
        # Never duplicate its partial extraction while it is still active.
        for seq in cfg['sequences']:
            point = OUT / 'points' / ('rvt_stream__' + seq + '.json')
            if point.exists() and read(point)['complete']:
                continue
            data = OUT / 'data_receipts' / (seq + '.json')
            partial = ROOT / 'datasets/dsec_confirmation/test' / seq / 'events/left/events.h5.part'
            if not data.exists() and partial.exists():
                phase = 'waiting_for_started_named_public_input ' + seq
                started = time.monotonic()
                while not data.exists():
                    save(OUT / 'pipeline_status.json', dict(state='running', stage='primary_RVT', phase=phase,
                        downloaded_member_bytes=partial.stat().st_size if partial.exists() else None, pid=os.getpid()))
                    if time.monotonic() - started > 7200:
                        raise TimeoutError('Previously started public input did not finish; partial retained')
                    time.sleep(10)
            if not data.exists():
                run('fetch_driving_confirmation_events.py', '--phase', 'fetch', '--sequence', seq)
            receipt = read(data)
            expected = (read(OUT / 'extra_events_members.json') if seq == 'thun_02_a' else
                        next(r for r in read(OUT / 'availability' / (seq + '.json'))['requests'] if r['kind'] == 'events'))
            assert receipt['etag'] == expected['etag'] and receipt['archive_bytes'] == expected['bytes']
            run('cache_driving_confirmation.py', '--model', 'rvt_stream', '--sequence', seq)
            run('evaluate_driving_confirmation.py', '--model', 'rvt_stream', '--sequence', seq)
            # Release only this task's newly downloaded public member after complete
            # checked raw caches/actual-point receipts exist. Never delete old inputs.
            path = (ROOT / receipt['event']['path']).resolve()
            expected_path = (ROOT / 'datasets/dsec_confirmation/test' / seq / 'events/left/events.h5').resolve()
            path.relative_to((ROOT / 'datasets/dsec_confirmation/test').resolve())
            if path != expected_path or seq not in cfg['sequences']:
                raise ValueError('Refusing storage cleanup outside exact new input path')
            if path.exists():
                if sha(path) != receipt['event']['sha256'] or not read(point)['complete']:
                    raise ValueError('Refusing cleanup without verified source and complete output')
                save(OUT / 'released_public_inputs' / (seq + '.json'), dict(path=path.relative_to(ROOT).as_posix(),
                     sha256=receipt['event']['sha256'], source_url=receipt['requested_url'],
                     reason='Project disk budget; new public input only, after complete primary caches and actual points. Reacquire exact member for SNN secondary or raw-forward reproduction.',
                     raw_cache_receipt='carriers/rvt_stream__' + seq + '.json', point_receipt='points/rvt_stream__' + seq + '.json'))
                path.unlink()
            save(OUT / 'pipeline_status.json', dict(state='running', stage='primary_RVT', completed_sequence=seq, pid=os.getpid()))
        run('summarize_driving_confirmation.py')
        save(OUT / 'pipeline_status.json', dict(state='primary_measurements_complete', secondary_SNN='not_executed', verification_pending=True, pid=os.getpid()))
    except Exception as error:
        save(OUT / 'pipeline_status.json', dict(state='failed', stage='primary_RVT', phase=phase,
             error=repr(error), traceback=traceback.format_exc(), pid=os.getpid()))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
