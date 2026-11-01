"""Download only the four named new public recording groups after config freeze."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,urllib.request
import fetch_pilot_assets as fetcher
ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'code/configs/confirmation_v01.json';OUT=ROOT/'experiments/confirmation_v01'
def main():
    cfg=json.loads(CONFIG.read_text());OUT.mkdir(exist_ok=True)
    plan=dict(config=cfg,config_sha256=fetcher.sha256(CONFIG),roles_fixed_before_event_downloads=True)
    p=OUT/'data_selection.json'
    if p.exists() and json.loads(p.read_text())!=plan:raise ValueError('Selection/config drift')
    p.write_text(json.dumps(plan,indent=2))
    def one(seq):
        a=dict(id=seq+'_events',url=f'https://download.ifi.uzh.ch/rpg/DSEC/train/{seq}/{seq}_events_left.zip',destination=f'datasets/downloads/{seq}_events_left.zip',extract_subdir=f'train/{seq}/events/left')
        record=fetcher.fetch(a);fetcher.extract(record)
        folder=ROOT/'datasets/dsec_pilot/train'/seq;grid=folder/'image_timestamps.txt'
        if not grid.exists():
            with urllib.request.urlopen(f'https://download.ifi.uzh.ch/rpg/DSEC/train/{seq}/{seq}_image_timestamps.txt',timeout=90) as response:grid.write_bytes(response.read())
        paths=[folder/'events/left/events.h5',grid,folder/'object_detections/left/tracks.npy']
        r=dict(sequence=seq,download=record,files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=fetcher.sha256(p)) for p in paths])
        (OUT/(seq+'_data_receipt.json')).write_text(json.dumps(r,indent=2));return r
    with ThreadPoolExecutor(max_workers=2) as executor:receipts=list(executor.map(one,cfg['confirmation_sequences']))
    (OUT/'data_receipt.json').write_text(json.dumps(dict(plan=plan,assets=receipts),indent=2));print('All four frozen confirmation assets verified',flush=True)
if __name__=='__main__':main()
