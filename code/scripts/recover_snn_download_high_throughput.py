"""Reuse cached covering byte ranges; more concurrent direct HTTP without data changes."""
import bisect,functools,hashlib,os
from pathlib import Path
from snn_secondary_common import ROOT,OUT,read,sha
import recover_snn_download as recovery

class CoveringCache:
    def __init__(self,folder,etag):
        self.rows=[];self.verified={}
        for meta in Path(folder).glob('*.json'):
            r=read(meta);path=meta.with_suffix('.bin')
            if path.exists() and r['etag']==etag:
                self.rows.append((r['start'],r['end'],path,r['sha256']))
        self.rows.sort(key=lambda x:(x[0],x[1]));self.starts=[r[0] for r in self.rows]
    def get(self,start,end):
        pos=start;selected=[]
        while pos<end:
            hi=bisect.bisect_right(self.starts,pos)
            candidates=[r for r in self.rows[max(0,hi-8):hi] if r[0]<=pos<r[1]]
            if not candidates:return None
            row=max(candidates,key=lambda r:r[1]);stop=min(end,row[1]);selected.append((row,pos,stop));pos=stop
        result=[]
        for (a,b,path,digest),lo,hi in selected:
            stat=path.stat();identity=(stat.st_size,stat.st_mtime_ns)
            if self.verified.get(path)!=identity:
                if stat.st_size!=b-a or sha(path)!=digest:raise ValueError('Covering cache integrity mismatch')
                self.verified[path]=identity
            with path.open('rb') as f:f.seek(lo-a);result.append(f.read(hi-lo))
        data=b''.join(result)
        if len(data)!=end-start:raise ValueError('Truncated covering cache')
        return data

class IndexedRanges(recovery.PersistentPublicRanges):
    def __init__(self,expected):
        super().__init__(expected,workers=16)
        self.coverage=CoveringCache(self.cache,self.etag)
    def _fetch(self,interval):
        data=self.coverage.get(*interval)
        if data is not None:
            with self.stats_lock:self.cache_hits+=1
            return data
        return super()._fetch(interval)

def main():
    receipt=read(OUT/'transport_covering_cache_v05.json')
    for path,digest in receipt['sources'].items():assert sha(ROOT/path)==digest,path
    os.environ['NO_PROXY']='download.ifi.uzh.ch'
    recovery.PersistentPublicRanges=IndexedRanges
    recovery.resume_member=functools.partial(recovery.resume_member,block_size=64*1024**2)
    recovery.main()

if __name__=='__main__':main()
