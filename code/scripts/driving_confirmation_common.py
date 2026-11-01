"""New-recording bookkeeping; historical experiments are never redirected."""
from pathlib import Path
import json, hashlib, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
OUT = ROOT / 'experiments/driving_confirmation_v02'
CONFIG = ROOT / 'code/configs/driving_confirmation_v02_run.json'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)

def sha(p):
    digest = hashlib.sha256()
    with p.open('rb') as f:
        while chunk := f.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()

def assert_frozen():
    frozen = read(OUT / 'freeze.json')
    for path, digest in frozen['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Execution source drift: ' + path)
    return frozen

def geometry(model):
    return [dict(stride=s, height=(512 if model == 'rvt_stream' else 256) // s,
                 width=(640 if model == 'rvt_stream' else 256) // s) for s in [8, 16, 32]]

def masks(values, shapes, capacity=.2):
    sizes = [g['height'] * g['width'] for g in shapes]
    selected = np.zeros(sum(sizes), bool)
    start = 0
    for size in sizes:
        rank = np.lexsort((np.arange(size), -values[start:start + size]))
        selected[start + rank[:max(1, int(np.floor(capacity * size)))]] = True
        start += size
    return selected

def compact_masks(h, predictor, arrays, variants, cfg, shapes, prior):
    n = sum(g['height'] * g['width'] for g in shapes)
    unsigned = np.concatenate([np.abs(np.abs(h[s]) - np.abs(prior[s])).mean(1) for s in [8, 16, 32]])
    result = []
    for name in cfg['methods']:
        if name == 'dense':
            result.append(np.ones(n, bool))
        else:
            value = unsigned if name == 'unsigned_residual' else arrays['rank_scores'][variants.index(name)]
            result.append(masks(value, shapes, cfg['capacity']))
    return np.packbits(np.stack(result), axis=1)

from riskaware.dsec_pilot import DSECPilotSequence

class ValidationSequence(DSECPilotSequence):
    """Same reader methods and raw distorted-coordinate semantics, test asset paths."""
    def __init__(self, sequence):
        if sequence == 'thun_00_a':
            super().__init__(ROOT / 'datasets/dsec_pilot', sequence)
            return
        import h5py, hdf5plugin
        self.sequence = sequence
        self.path = ROOT / 'datasets/dsec_confirmation/test' / sequence
        self.event_path = self.path / 'events/left/events.h5'
        self.label_path = self.path / 'object_detections/left/tracks.npy'
        self.file = h5py.File(self.event_path, 'r')
        self.events = self.file['events']
        self.offset_us = int(np.asarray(self.file['t_offset']).item())
        self.labels = np.load(self.label_path, allow_pickle=False)
        self.annotation_times = np.unique(self.labels['t']).astype(np.int64)
        self.ms_to_idx = np.asarray(self.file['ms_to_idx'], dtype=np.int64)
        self.height, self.width = 480, 640
        self.event_count = len(self.events['t'])
        self.first_local_us, self.last_local_us = int(self.events['t'][0]), int(self.events['t'][-1])

def query_times(reader, cfg):
    grid = np.loadtxt(reader.path / 'image_timestamps.txt', dtype=np.int64)
    times = [int(t) for t in grid if int(t) - 180000 >= reader.offset_us + reader.first_local_us
             and int(t) <= reader.offset_us + reader.last_local_us]
    if len(times) <= cfg['warmup']:
        raise ValueError('No complete evaluation cohort')
    return grid, times
