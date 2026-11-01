"""Tail-only candidate recovery, followed by full exact-file validation before use."""
import argparse,datetime,hashlib,os,time,zipfile,zlib
from pathlib import Path
from snn_secondary_common import ROOT,OUT,PARENT,read,save,sha
from recover_snn_download import PersistentPublicRanges
from deflate_tail_recovery import validated_tail

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',default='thun_02_a');args=parser.parse_args()
    cfg=read(ROOT/'code/configs/snn_secondary_v01.json');assert args.sequence in cfg['sequences']
    amendment=read(OUT/'transport_tail_recovery_v04.json')
    for path,digest in amendment['sources'].items():assert sha(ROOT/path)==digest,path
    os.environ['NO_PROXY']='download.ifi.uzh.ch'
    receipt_path=PARENT/'data_receipts'/(args.sequence+'.json');expected=read(receipt_path)
    target=(ROOT/expected['event']['path']).resolve();allowed=(ROOT/'datasets/dsec_confirmation/test'/args.sequence/'events/left').resolve()
    assert target.parent==allowed and target.name=='events.h5'
    partial=target.with_suffix('.h5.part');assert partial.exists() and not target.exists()
    lock=OUT/'recovery_download.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    source=None;started=time.monotonic();last=[0.];status=OUT/'download_recovery_status.json'
    try:
        save(status,dict(state='running',sequence=args.sequence,pid=os.getpid(),phase='locating_tail_restart_anchor',preserved_partial_bytes=partial.stat().st_size))
        source=PersistentPublicRanges(expected)
        def progress(value):
            if time.monotonic()-last[0]>=15 or value['raw_verified_bytes']==value['member_bytes']:
                save(status,dict(state='running',sequence=args.sequence,pid=os.getpid(),elapsed_s=time.monotonic()-started,
                    network_bytes=source.transferred,retries=source.retries,**value))
                print('Candidate tail',round(value['validated_candidate_tail_bytes']/1024**2),'MiB; original prefix unchanged',flush=True)
                last[0]=time.monotonic()
        with zipfile.ZipFile(source) as archive:
            info=archive.getinfo(expected['event']['zip_member'])
            assert info.file_size==expected['event']['bytes'] and info.CRC==expected['event']['zip_crc32']
            recovered=validated_tail(source,info,partial,expected['event']['sha256'],progress)
        # Neither the original prefix nor metadata may change while the separate tail is recovered.
        assert partial.stat().st_size==recovered['preserved_partial_bytes']
        assert sha(partial)==recovered['original_partial_sha256']
        for r in expected['metadata']:assert sha(ROOT/r['path'])==r['sha256']
        candidate=Path(recovered['tail_path']).resolve();assert candidate.parent==allowed
        assert candidate.stat().st_size==recovered['tail_bytes']
        with partial.open('ab') as output,candidate.open('rb') as tail:
            while block:=tail.read(8*1024**2):output.write(block)
            output.flush();os.fsync(output.fileno())
        digest=hashlib.sha256();crc=0
        with partial.open('rb') as f:
            while block:=f.read(8*1024**2):digest.update(block);crc=zlib.crc32(block,crc)
        assert partial.stat().st_size==info.file_size and digest.hexdigest()==expected['event']['sha256'] and crc&0xffffffff==info.CRC
        partial.replace(target)
        event=dict(expected['event'],preserved_partial_bytes=recovered['preserved_partial_bytes'],appended_bytes=recovered['tail_bytes'],
             original_partial_sha256=recovered['original_partial_sha256'],recovery_candidate_tail_sha256=sha(candidate))
        save(OUT/'data_receipts'/(args.sequence+'.json'),dict(expected,event=event,secondary_reacquisition=True,
             original_receipt_sha256=sha(receipt_path),HTTP_range_requests=source.requests,downloaded_bytes=source.transferred,
             recovery_transport='Stored DEFLATE block restart with32KiB history; separate tail; complete CRC and SHA before commit plus full-file recheck after append',
             recovery_source_sha256=sha(Path(__file__).resolve()),tail_recovery=recovered,transport_retries=source.retries))
        save(status,dict(state='complete',sequence=args.sequence,bytes=event['bytes'],exact_member_sha256=event['sha256'],crc32=info.CRC,
             preserved_partial_bytes=recovered['preserved_partial_bytes'],appended_bytes=recovered['tail_bytes'],network_bytes=source.transferred,
             elapsed_s=time.monotonic()-started,retries=source.retries,scope='Download recovery only; SNN inference not restarted',
             recovery='Full CRC/SHA validated before and after appending separately stored tail'))
        print('Recovered exact official H5',event['bytes'],event['sha256'],flush=True)
    except Exception as error:
        save(status,dict(state='failed',sequence=args.sequence,error=repr(error),pid=os.getpid(),original_prefix_retained=True))
        raise
    finally:
        if source is not None:source.close()
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
