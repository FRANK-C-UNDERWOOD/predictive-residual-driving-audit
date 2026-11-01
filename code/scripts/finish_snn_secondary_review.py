"""Finish independent review/report after all frozen measurements and pair checks."""
import subprocess,sys,time,traceback
from snn_secondary_common import ROOT,OUT,read,save,sha

def main():
    if (OUT/'artifact_manifest.json').exists():print('SNN artifacts already sealed; no report overwrite',flush=True);return
    started=time.monotonic()
    while True:
        status=read(OUT/'pipeline_status.json')
        if status['state']=='failed':raise RuntimeError('Measurement pipeline failed; cannot review incomplete results')
        checks=list((OUT/'checks').glob('*__*.json'))
        if status['state'] in ['measurements_complete','complete'] and (OUT/'secondary_results.json').exists() and len(checks)==14 and not (OUT/'pipeline.lock').exists():break
        save(OUT/'review_status.json',dict(state='waiting_for_complete_measurements_and_checks',checked_pairs=len(checks)))
        time.sleep(5)
    phase='final_independent_verification'
    log=OUT/'final_review.log'
    try:
        for script in ['verify_snn_secondary.py','report_snn_secondary.py']:
            phase=script
            if script.startswith('report') and (OUT/'report_receipt.json').exists():continue
            save(OUT/'review_status.json',dict(state='running',phase=phase))
            with log.open('a',encoding='utf-8') as f:
                f.write('\n'+script+'\n');f.flush()
                subprocess.run([sys.executable,'-u',str(ROOT/'code/scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        save(OUT/'review_status.json',dict(state='ready_for_visual_review',independent_verification='passed',reporting='complete',
            report_receipt_sha256=sha(OUT/'report_receipt.json'),elapsed_s=time.monotonic()-started))
        print('All SNN measurements independently checked; report and scientific figure ready for visual review',flush=True)
    except Exception as error:
        save(OUT/'review_status.json',dict(state='failed',phase=phase,error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
