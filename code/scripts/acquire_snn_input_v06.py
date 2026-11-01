"""Exact public inputs with resumable validated ranges; retain completed members."""
import argparse,os,shutil,time,zipfile
from snn_secondary_common import ROOT,OUT,PARENT,CONFIG,read,save,sha
from recover_snn_download import resume_member
from recover_snn_download_high_throughput import IndexedRanges

def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);args=p.parse_args()
    cfg=read(CONFIG);assert args.sequence in cfg['sequences']
    amendment=read(OUT/'continuation_v06.json')
    for path,digest in amendment['sources'].items():assert sha(ROOT/path)==digest,path
    expected_path=PARENT/'data_receipts'/(args.sequence+'.json');expected=read(expected_path)
    target=(ROOT/expected['event']['path']).resolve();allowed=(ROOT/'datasets/dsec_confirmation/test'/args.sequence/'events/left').resolve()
    assert target.parent==allowed and target.name=='events.h5'
    receipt=OUT/'data_receipts'/(args.sequence+'.json')
    if target.exists():
        assert sha(target)==expected['event']['sha256']
        if not receipt.exists():save(receipt,dict(expected,secondary_reacquisition=True,reused_complete_member=True,original_receipt_sha256=sha(expected_path)))
        print('Exact local SNN input ready',args.sequence,flush=True);return
    lock=OUT/'acquisition_locks'/(args.sequence+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as f:f.write(str(os.getpid()))
    status=OUT/'acquisition_status'/(args.sequence+'.json');source=None;started=time.monotonic();last=[0.]
    os.environ['NO_PROXY']='download.ifi.uzh.ch'
    try:
        partial=target.with_suffix('.h5.part');existing=partial.stat().st_size if partial.exists() else 0
        if shutil.disk_usage(ROOT).free<2*(expected['event']['bytes']-existing)+1024**3:
            raise RuntimeError('Insufficient raw+transport-cache reserve; old assets untouched')
        save(status,dict(state='running',sequence=args.sequence,raw_verified_bytes=0,member_bytes=expected['event']['bytes'],pid=os.getpid()))
        source=IndexedRanges(expected)
        def progress(v):
            if time.monotonic()-last[0]>=15 or v['raw_verified_bytes']==v['member_bytes']:
                save(status,dict(state='running',sequence=args.sequence,elapsed_s=time.monotonic()-started,retries=source.retries,**v))
                print('Exact input',args.sequence,round(v['raw_verified_bytes']/1024**2),'/',round(v['member_bytes']/1024**2),'MiB',flush=True);last[0]=time.monotonic()
        with zipfile.ZipFile(source) as archive:
            info=archive.getinfo(expected['event']['zip_member'])
            assert info.file_size==expected['event']['bytes'] and info.CRC==expected['event']['zip_crc32']
            asset=resume_member(archive,info,target,expected['event']['sha256'],ROOT/'datasets/dsec_confirmation/test',progress,64*1024**2)
        for r in expected['metadata']:assert sha(ROOT/r['path'])==r['sha256']
        save(receipt,dict(expected,event=asset,secondary_reacquisition=True,original_receipt_sha256=sha(expected_path),
             HTTP_range_requests=source.requests,downloaded_bytes=source.transferred,retries=source.retries,
             recovery_transport='v06 unchanged exact-member CRC/SHA; retained raw member and resumable compressed-range cache'))
        save(status,dict(state='complete',sequence=args.sequence,bytes=asset['bytes'],sha256=asset['sha256'],elapsed_s=time.monotonic()-started))
        print('Exact SNN input complete',args.sequence,asset['sha256'],flush=True)
    except Exception as error:
        save(status,dict(state='failed',sequence=args.sequence,error=repr(error),partial_retained=True));raise
    finally:
        if source is not None:source.close()
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
