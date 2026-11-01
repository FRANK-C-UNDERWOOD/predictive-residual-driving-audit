"""Offline materialization of hash-verified pure-Python runtime dependencies.

Public downloads and expected digests are recorded in the download receipt.
No package setup script is executed and the SpikeFPN environment is unchanged.
"""
from pathlib import Path
import hashlib,json,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[2];DOWNLOAD=ROOT/'vendor/rvt_dependency_downloads';OUT=ROOT/'vendor/rvt_runtime_deps'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt=json.loads((DOWNLOAD/'receipt.json').read_text());OUT.mkdir(exist_ok=True)
    for r in receipt:
        p=DOWNLOAD/r['url'].rsplit('/',1)[-1]
        if sha(p)!=r['sha256']:raise ValueError('Downloaded dependency hash mismatch')
        if p.suffix=='.whl':
            with zipfile.ZipFile(p) as z:
                for item in z.infolist():
                    dest=(OUT/item.filename).resolve();dest.relative_to(OUT.resolve())
                    if not item.is_dir():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(item))
        elif r['name']=='antlr4-python3-runtime':
            with tarfile.open(p) as t:
                for item in t.getmembers():
                    if '/src/antlr4/' in item.name and item.isfile():
                        dest=(OUT/'antlr4'/item.name.split('/src/antlr4/')[1]).resolve();dest.relative_to(OUT.resolve())
                        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(t.extractfile(item).read())
        else:raise ValueError('Unsupported dependency distribution')
    record=dict(scope='Standalone Python source directories, not a changed venv',
        download_receipt_sha256=sha(DOWNLOAD/'receipt.json'),
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts])
    (ROOT/'vendor/rvt_runtime_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Verified standalone RVT runtime sources:',len(record['files']))
if __name__=='__main__':main()
