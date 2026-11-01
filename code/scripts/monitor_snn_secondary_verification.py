"""Read-only verification as completed sequence pairs become available."""
import time
from snn_secondary_common import OUT, CONFIG, read
from verify_snn_secondary import verify_one

cfg = read(CONFIG)
for seq in cfg['sequences']:
    for model in cfg['models']:
        aggregate = OUT / 'aggregates' / (model + '__' + seq + '.json')
        while not aggregate.exists():
            status = read(OUT / 'pipeline_status.json')
            if status['state'] == 'failed':
                raise RuntimeError('Measurement pipeline failed; verification monitor stopping')
            time.sleep(5)
        dest = OUT / 'checks' / (model + '__' + seq + '.json')
        if not dest.exists():
            verify_one(model, seq)
print('All fourteen SNN pair checks finished', flush=True)
