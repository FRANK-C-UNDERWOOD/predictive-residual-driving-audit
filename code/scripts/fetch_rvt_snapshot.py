"""Fetch the official RVT source at a pinned Git commit and public Tiny weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'vendor/rvt_snapshot'
API='https://api.github.com/repos/uzh-rpg/RVT'
HEAD={'User-Agent':'SCI-research-reproduction/1.0'}
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=HEAD),timeout=120) as r:return r.read()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    commit=json.loads(fetch(API+'/commits/master'))['sha']
    tree=json.loads(fetch(API+'/git/trees/'+commit+'?recursive=1'))
    if tree.get('truncated'):raise ValueError('Truncated tree')
    members=[x for x in tree['tree'] if x['type']=='blob' and (x['path'].endswith(('.py','.yaml','.txt','.md')) or 'LICENSE' in x['path'])]
    def one(item):
        data=fetch('https://raw.githubusercontent.com/uzh-rpg/RVT/'+commit+'/'+item['path'])
        if hashlib.sha1(('blob %d\0'%len(data)).encode()+data).hexdigest()!=item['sha']:raise ValueError('Git blob mismatch')
        p=OUT/item['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        return dict(path=item['path'],git_blob_sha1=item['sha'],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,members))
    url='https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-t.ckpt'
    p=OUT/'weights/rvt-t-1mpx.ckpt';p.parent.mkdir(exist_ok=True)
    if not p.exists():
        print('Downloading official 1Mpx Tiny checkpoint',flush=True);p.write_bytes(fetch(url))
    data=p.read_bytes();md5=hashlib.md5(data).hexdigest()
    if not md5.startswith('5a3c78'):raise ValueError('Published checkpoint MD5 prefix mismatch')
    receipt=dict(repository='https://github.com/uzh-rpg/RVT',commit=commit,files=records,
        checkpoint=dict(path=p.relative_to(OUT).as_posix(),url=url,sha256=hashlib.sha256(data).hexdigest(),md5=md5,bytes=len(data),published_md5_prefix='5a3c78'))
    (OUT/'_UPSTREAM.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(commit=commit,files=len(records),checkpoint=receipt['checkpoint'])),flush=True)
if __name__=='__main__':main()
