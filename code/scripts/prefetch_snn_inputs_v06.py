"""Serial network prefetch overlaps the single GPU pipeline, never inference."""
import subprocess,sys,time
from snn_secondary_common import ROOT,OUT,CONFIG,read,save
cfg=read(CONFIG)
for seq in cfg['sequences'][2:]:
    for attempt in range(3):
        result=subprocess.run([sys.executable,'-u',str(ROOT/'code/scripts/acquire_snn_input_v06.py'),'--sequence',seq],cwd=ROOT)
        if result.returncode==0:break
        save(OUT/'prefetch_status.json',dict(state='retrying',sequence=seq,attempt=attempt+1))
        time.sleep(5)
    else:
        save(OUT/'prefetch_status.json',dict(state='failed',sequence=seq));raise SystemExit(1)
    save(OUT/'prefetch_status.json',dict(state='running',last_completed=seq))
save(OUT/'prefetch_status.json',dict(state='complete',sequences=cfg['sequences'][2:]))
print('All remaining exact public inputs ready',flush=True)
