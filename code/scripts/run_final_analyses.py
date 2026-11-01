"""Serial final secondary analyses; preserve the completed parent experiment."""
import os,subprocess,sys,traceback
from confirmation_common import ROOT,save
OUT=ROOT/'experiments/fp_mechanism_v01'
def main():
    OUT.mkdir(exist_ok=True);lock=OUT/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    phase='start'
    try:
        for script,args in [('run_fp_mechanism.py',['--phase','extract']),('run_fp_mechanism.py',['--phase','points']),('verify_fp_mechanism.py',[]),('summarize_fp_mechanism.py',[])]:
            phase=script+' '+' '.join(args);save(OUT/'pipeline_status.json',dict(state='running',phase=phase,pid=os.getpid()));subprocess.run([sys.executable,'-u',str(ROOT/'code/scripts'/script),*args],cwd=ROOT,check=True)
        save(OUT/'pipeline_status.json',dict(state='complete',pid=os.getpid()))
    except Exception as error:
        save(OUT/'pipeline_status.json',dict(state='failed',phase=phase,error=repr(error),traceback=traceback.format_exc()));raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
