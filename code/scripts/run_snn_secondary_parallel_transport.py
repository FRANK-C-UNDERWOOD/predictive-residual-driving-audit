"""The frozen SNN pipeline with a pinned, download-only transport replacement."""
import subprocess
from snn_secondary_common import ROOT, OUT, read, sha

def main():
    receipt = read(OUT / 'transport_freeze.json')
    for path, digest in receipt['sources'].items():
        assert sha(ROOT / path) == digest, path
    original = subprocess.run
    def dispatch(command, *args, **kwargs):
        old = str(ROOT / 'code/scripts/fetch_snn_secondary.py')
        if isinstance(command, list) and old in command:
            command = [str(ROOT / 'code/scripts/fetch_snn_secondary_parallel.py') if value == old else value for value in command]
        return original(command, *args, **kwargs)
    subprocess.run = dispatch
    from run_snn_secondary import main as run
    run()

if __name__ == '__main__':
    main()
