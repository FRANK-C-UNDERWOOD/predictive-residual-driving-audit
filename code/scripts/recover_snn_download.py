"""Recover an exact official ZIP member without truncating the saved raw prefix."""
from pathlib import Path
import argparse, concurrent.futures, datetime, hashlib, io, json, os, threading, time, zipfile
import requests
from snn_secondary_common import ROOT, OUT, PARENT, read, save, sha

class PersistentPublicRanges(io.RawIOBase):
    """Persistent connections, bounded retries and strict publisher identity checks."""
    def __init__(self, expected, workers=8):
        super().__init__()
        from urllib.parse import urlparse
        self.url = expected['requested_url']
        if urlparse(self.url).hostname != 'download.ifi.uzh.ch':
            raise ValueError('Only the official public publisher is allowed')
        self.length, self.etag = expected['archive_bytes'], expected['etag']
        self.position = self.requests = self.transferred = self.retries = 0
        self.cache_hits = 0
        self.cache = OUT / 'download_recovery_chunks' / hashlib.sha256((self.url+'|'+self.etag).encode()).hexdigest()[:20]
        self.cache.mkdir(parents=True,exist_ok=True)
        self.local = threading.local()
        self.sessions, self.sessions_lock = [], threading.Lock()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.stats_lock = threading.Lock()
        self._head()

    def _session(self):
        if not hasattr(self.local, 'session'):
            session = requests.Session()
            session.headers.update({'User-Agent':'SCI-public-research/1.0','Accept-Encoding':'identity'})
            self.local.session = session
            with self.sessions_lock:
                self.sessions.append(session)
        return self.local.session

    def _head(self):
        for attempt in range(12):
            try:
                with self._session().head(self.url, timeout=(15,60), allow_redirects=False) as r:
                    r.raise_for_status()
                    if int(r.headers['Content-Length']) != self.length or r.headers.get('ETag') != self.etag:
                        raise ValueError('Publisher archive identity changed')
                return
            except requests.RequestException:
                if attempt == 11:raise
                time.sleep(min(30,2**attempt))

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self, offset, whence=0):
        pos = offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if pos<0:raise ValueError('Negative seek')
        self.position=pos
        return pos

    def _fetch(self, interval):
        start,end=interval
        cached=self.cache/(str(start)+'_'+str(end)+'.bin')
        metadata=cached.with_suffix('.json')
        if cached.exists() and metadata.exists():
            record=read(metadata);data=cached.read_bytes()
            if record['etag']!=self.etag or len(data)!=end-start or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise ValueError('Saved transport chunk failed integrity check')
            with self.stats_lock:self.cache_hits+=1
            return data
        headers={'Range':f'bytes={start}-{end-1}','If-Range':self.etag}
        for attempt in range(12):
            try:
                with self._session().get(self.url,headers=headers,stream=True,timeout=(15,60),allow_redirects=False) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end-1}/{self.length}' or r.headers.get('ETag')!=self.etag:
                        raise ValueError('Publisher range/identity mismatch; refusing fallback')
                    if r.headers.get('Content-Encoding','identity') not in ['identity','']:
                        raise ValueError('Unexpected HTTP content encoding')
                    data=r.content
                    if len(data)!=end-start:raise requests.ConnectionError('Truncated range')
                tmp=cached.with_suffix('.writing')
                tmp.write_bytes(data);tmp.replace(cached)
                save(metadata,dict(etag=self.etag,start=start,end=end,sha256=hashlib.sha256(data).hexdigest()))
                with self.stats_lock:
                    self.requests+=1;self.transferred+=len(data)
                return data
            except (requests.RequestException, OSError) as error:
                if attempt==11:raise
                with self.stats_lock:self.retries+=1
                # Renew only this thread's failed transport. Scientific inputs stay fixed.
                self._session().close()
                del self.local.session
                time.sleep(min(30,2**attempt))

    def read(self,size=-1):
        stop=self.length if size<0 else min(self.length,self.position+size)
        if stop<=self.position:return b''
        chunk=4*1024**2
        intervals=[(p,min(stop,p+chunk)) for p in range(self.position,stop,chunk)]
        pieces=list(self.pool.map(self._fetch,intervals))
        self.position=stop
        return b''.join(pieces)

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True)
            for session in self.sessions:session.close()
        super().close()

def resume_member(archive,info,target,expected_sha,allowed_root,progress=lambda value:None,block_size=32*1024**2):
    """Reinflate/compare saved prefix, append only verified new bytes, then seal."""
    target=Path(target).resolve();allowed_root=Path(allowed_root).resolve()
    target.relative_to(allowed_root)
    partial=target.with_suffix(target.suffix+'.part')
    if target.exists():
        if target.stat().st_size!=info.file_size or sha(target)!=expected_sha:
            raise ValueError('Existing completed file differs; refusing overwrite')
        return dict(path=target.relative_to(ROOT).as_posix(),bytes=target.stat().st_size,sha256=expected_sha,reused=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    existing=partial.stat().st_size if partial.exists() else 0
    if existing>info.file_size:raise ValueError('Saved prefix exceeds expected member')
    prefix_sha=sha(partial) if partial.exists() else None
    digest=hashlib.sha256();count=0
    # r+b never truncates; append starts only once every existing prefix byte matched.
    with archive.open(info) as source,partial.open('r+b' if partial.exists() else 'x+b') as output:
        while block:=source.read(block_size):
            compare=min(len(block),max(0,existing-count))
            if compare and output.read(compare)!=block[:compare]:
                raise ValueError(f'Saved prefix mismatch at raw offset {count}; file retained')
            if len(block)>compare:
                if count+compare<existing:raise AssertionError('Refusing write inside saved prefix')
                output.write(block[compare:]);output.flush()
            digest.update(block);count+=len(block)
            progress(dict(raw_verified_bytes=count,member_bytes=info.file_size,preserved_partial_bytes=existing,
                          appended_bytes=max(0,count-existing),phase='revalidating_existing_prefix' if count<existing else 'appending_missing_tail'))
        output.flush();os.fsync(output.fileno())
    # ZipExtFile checked the member CRC at EOF. SHA is the prior experiment's exact member identity.
    if count!=info.file_size or digest.hexdigest()!=expected_sha:
        raise ValueError('Completed member size/SHA mismatch; partial retained')
    partial.replace(target)
    return dict(path=target.relative_to(ROOT).as_posix(),bytes=count,sha256=digest.hexdigest(),
                zip_member=info.filename,zip_crc32=info.CRC,zip_crc_verified=True,
                preserved_partial_bytes=existing,original_partial_sha256=prefix_sha,appended_bytes=count-existing)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',default='thun_02_a');args=parser.parse_args()
    cfg=read(ROOT/'code/configs/snn_secondary_v01.json')
    if args.sequence not in cfg['sequences']:raise ValueError('Unregistered sequence')
    expected_path=PARENT/'data_receipts'/(args.sequence+'.json');expected=read(expected_path)
    expected_target=(ROOT/'datasets/dsec_confirmation/test'/args.sequence/'events/left/events.h5').resolve()
    if (ROOT/expected['event']['path']).resolve()!=expected_target:raise ValueError('Unexpected output path')
    amendment=read(OUT/'transport_recovery_v02.json')
    for path,digest in amendment['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Recovery transport source drift: '+path)
    lock=OUT/'recovery_download.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=time.monotonic();last=[0.]
    progress_path=OUT/'download_recovery_status.json'
    source=None
    try:
        partial=expected_target.with_suffix('.h5.part')
        save(progress_path,dict(state='running',sequence=args.sequence,pid=os.getpid(),phase='checking_publisher_and_saved_prefix',
             partial_bytes=partial.stat().st_size if partial.exists() else 0))
        source=PersistentPublicRanges(expected)
        def progress(value):
            if time.monotonic()-last[0]>=15 or value['raw_verified_bytes']==value['member_bytes']:
                save(progress_path,dict(state='running',sequence=args.sequence,pid=os.getpid(),elapsed_s=time.monotonic()-started,
                     network_bytes=source.transferred,retries=source.retries,**value))
                print(value['phase'],round(value['raw_verified_bytes']/1024**2),'/',round(value['member_bytes']/1024**2),'MiB; appended',round(value['appended_bytes']/1024**2),'MiB',flush=True)
                last[0]=time.monotonic()
        with zipfile.ZipFile(source) as archive:
            info=archive.getinfo(expected['event']['zip_member'])
            if info.file_size!=expected['event']['bytes'] or info.CRC!=expected['event']['zip_crc32']:
                raise ValueError('Named member size/CRC changed')
            asset=resume_member(archive,info,expected_target,expected['event']['sha256'],ROOT/'datasets/dsec_confirmation/test',progress)
        for record in expected['metadata']:
            if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Annotation/timestamp metadata changed')
        save(OUT/'data_receipts'/(args.sequence+'.json'),dict(expected,event=asset,secondary_reacquisition=True,
             original_receipt_sha256=sha(expected_path),HTTP_range_requests=source.requests,downloaded_bytes=source.transferred,
             recovery_transport='persistent concurrent ranges; raw prefix compared before append',transport_retries=source.retries,
             recovery_source_sha256=sha(Path(__file__).resolve())))
        save(progress_path,dict(state='complete',sequence=args.sequence,elapsed_s=time.monotonic()-started,
             exact_member_sha256=asset['sha256'],bytes=asset['bytes'],preserved_partial_bytes=asset.get('preserved_partial_bytes'),
             appended_bytes=asset.get('appended_bytes'),network_bytes=source.transferred,retries=source.retries,
             scope='Download recovery only; SNN inference has not been restarted'))
        print('Exact public event member recovered and verified',args.sequence,asset['bytes'],asset['sha256'],flush=True)
    except Exception as error:
        save(progress_path,dict(state='failed',sequence=args.sequence,error=repr(error),pid=os.getpid(),partial_retained=True))
        raise
    finally:
        if source is not None:source.close()
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
