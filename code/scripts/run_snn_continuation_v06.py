"""Resume the frozen scientific workers, retaining raw inputs now space permits."""
import os,subprocess,sys,time,traceback
from snn_secondary_common import ROOT,OUT,PARENT,CONFIG,read,save,sha,assert_frozen

def main():
    assert_frozen();cfg=read(CONFIG)
    for path,digest in read(OUT/'continuation_v06.json')['sources'].items():assert sha(ROOT/path)==digest,path
    if (OUT/'secondary_results.json').exists():print('All secondary measurements already complete',flush=True);return
    lock=OUT/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    phase='initializing'
    def status(**extra):save(OUT/'pipeline_status.json',dict(state='running',phase=phase,pid=os.getpid(),**extra))
    def run(script,*args):
        nonlocal phase
        phase=script+' '+' '.join(args);status()
        subprocess.run([sys.executable,'-u',str(ROOT/'code/scripts'/script),*args],cwd=ROOT,check=True)
    try:
        for seq in cfg['sequences']:
            points=[OUT/'points'/(model+'__'+seq+'.json') for model in cfg['models']]
            if all(p.exists() and read(p)['complete'] for p in points):continue
            expected=read(PARENT/'data_receipts'/(seq+'.json'));target=ROOT/expected['event']['path']
            phase='waiting_for_exact_input '+seq;status()
            # Network prefetch and inference share no output writers.
            while not target.exists():
                acquisition=OUT/'acquisition_locks'/(seq+'.lock')
                prefetch=read(OUT/'prefetch_status.json') if (OUT/'prefetch_status.json').exists() else {}
                if acquisition.exists() or prefetch.get('state') in ['running','retrying']:
                    status(waiting_sequence=seq);time.sleep(5);continue
                run('acquire_snn_input_v06.py','--sequence',seq)
            while (OUT/'acquisition_locks'/(seq+'.lock')).exists():time.sleep(2)
            assert (OUT/'data_receipts'/(seq+'.json')).exists()
            run('cache_snn_secondary.py','--sequence',seq)
            for model in cfg['models']:
                run('evaluate_snn_secondary.py','--model',model,'--sequence',seq)
                run('join_snn_secondary.py','--model',model,'--sequence',seq)
            save(OUT/'retained_public_inputs'/(seq+'.json'),dict(path=target.relative_to(ROOT).as_posix(),sha256=expected['event']['sha256'],
                 reason='Disk capacity now sufficient; retain exact raw public member for reproducibility and later stateful-reuse experiments. No scientific worker/endpoint changed.'))
        run('summarize_snn_secondary.py')
        save(OUT/'pipeline_status.json',dict(state='measurements_complete',independent_verification='pending',stateful_reuse='old_calibration_probe_only',pid=os.getpid()))
    except Exception as error:
        save(OUT/'pipeline_status.json',dict(state='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
