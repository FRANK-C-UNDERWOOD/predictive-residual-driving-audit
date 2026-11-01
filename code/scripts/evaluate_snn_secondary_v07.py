"""Unchanged evaluator with the documented float32-boundary validation adapter."""
import argparse, runpy, sys
from pathlib import Path
from snn_secondary_common import ROOT, activate
from snn_float32_floor_compat import install


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--compat-preflight-out')
    known, remaining = parser.parse_known_args()
    common = activate()
    if known.compat_preflight_out:
        assert '--preflight' in remaining
        common.OUT = ROOT / Path(known.compat_preflight_out)
    install()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_path(str(ROOT / 'code/scripts/evaluate_driving_confirmation.py'), run_name='__main__')


if __name__ == '__main__':
    main()
