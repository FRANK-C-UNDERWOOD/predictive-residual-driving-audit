"""Separate secondary output context; never mutate the completed RVT trial."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from driving_confirmation_common import read, save, sha
PARENT = ROOT / 'experiments/driving_confirmation_v02'
OUT = ROOT / 'experiments/snn_secondary_v01'
CONFIG = ROOT / 'code/configs/snn_secondary_v01.json'

def assert_frozen():
    frozen = read(OUT / 'freeze.json')
    for category in ['sources', 'inputs']:
        for path, digest in frozen[category].items():
            if sha(ROOT / path) != digest:
                raise ValueError('Secondary execution drift: ' + path)
    return frozen

def activate():
    """Rebind output/config only for unchanged shared read/evaluate helpers."""
    import driving_confirmation_common as common
    common.OUT, common.CONFIG, common.assert_frozen = OUT, CONFIG, assert_frozen
    return common
