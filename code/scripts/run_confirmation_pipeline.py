"""Serial GPU workers; Thun preflight/calibration precede untouched clip inference."""
import subprocess,sys,time,os,traceback
from confirmation_common import ROOT,OUT,CONFIG,read,save,sha
def main():
    cfg=read(CONFIG);lock=OUT/'pipeline.lock';phase='waiting_for_initial_Thun'
    with lock.open('x') as f:f.write(str(os.getpid()))
    def run(script,*args):
        nonlocal phase
        phase=script+' '+' '.join(args);save(OUT/'pipeline_status.json',dict(state='running',phase=phase,pid=os.getpid()));subprocess.run([sys.executable,'-u',str(ROOT/'code/scripts'/script),*args],cwd=ROOT,check=True)
    try:
        # One original Thun worker was started before this orchestrator. Never duplicate it.
        started=time.monotonic()
        while not (OUT/'spike_original__thun_00_a.json').exists():
            if time.monotonic()-started>3600:raise TimeoutError('Initial preflight did not finish')
            time.sleep(10)
        for model in ['spike_adapted']:
            if not (OUT/(model+'__thun_00_a.json')).exists():run('cache_confirmation_spike.py','--model',model,'--sequence','thun_00_a')
        if not (OUT/'rvt_stream__thun_00_a.json').exists():run('cache_confirmation_rvt.py','--sequence','thun_00_a')
        if not (OUT/'frozen_thresholds.json').exists():run('confirmation_evaluate.py','--phase','calibrate')
        # Dependency expansion is a receipt, not a change to inference/protocol.
        save(OUT/'dependency_boundary.json',dict(rvt_config_sha256=sha(ROOT/'code/configs/rvt_reference_v01.json'),diverse_config_sha256=sha(ROOT/'code/configs/diverse_adaptation_v01.json'),frozen_before_heldout_inference=True))
        pending=list(cfg['confirmation_sequences']);wait_start=time.monotonic()
        while pending:
            ready=[s for s in pending if (OUT/(s+'_data_receipt.json')).exists()]
            if not ready:
                phase='waiting_for_named_data';save(OUT/'pipeline_status.json',dict(state='running',phase=phase,pending=pending,pid=os.getpid()))
                if time.monotonic()-wait_start>21600:raise TimeoutError('Named public data downloads not complete')
                time.sleep(15);continue
            for seq in ready:
                receipt=read(OUT/(seq+'_data_receipt.json'))
                for r in receipt['files']:
                    if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Downloaded input changed')
                for model in cfg['models']:
                    if (OUT/(model+'__'+seq+'.json')).exists():continue
                    if model=='rvt_stream':run('cache_confirmation_rvt.py','--sequence',seq)
                    else:run('cache_confirmation_spike.py','--model',model,'--sequence',seq)
                pending.remove(seq)
        if not (OUT/'summary.json').exists():run('confirmation_evaluate.py','--phase','evaluate')
        if not (OUT/'verification.json').exists():run('verify_confirmation.py')
        run('summarize_confirmation.py');save(OUT/'pipeline_status.json',dict(state='complete',pid=os.getpid()))
    except Exception as error:
        save(OUT/'pipeline_status.json',dict(state='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()));raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
