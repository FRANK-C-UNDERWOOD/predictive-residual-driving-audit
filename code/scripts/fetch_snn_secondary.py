"""Reacquire the exact public member, checking the sealed original receipt."""
import argparse, zipfile
from snn_secondary_common import ROOT, OUT, PARENT, read, save, sha
from fetch_driving_confirmation_events import PublicRangeFile, extract_member

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    args = parser.parse_args()
    cfg = read(ROOT / 'code/configs/snn_secondary_v01.json')
    assert args.sequence in cfg['sequences']
    expected = read(PARENT / 'data_receipts' / (args.sequence + '.json'))
    dest = OUT / 'data_receipts' / (args.sequence + '.json')
    target = ROOT / expected['event']['path']
    if dest.exists():
        assert target.exists() and sha(target) == expected['event']['sha256']
        print('Secondary public input already verified', args.sequence, flush=True)
        return
    source = PublicRangeFile(expected['requested_url'])
    assert source.length == expected['archive_bytes'] and source.etag == expected['etag']
    with zipfile.ZipFile(source) as archive:
        info = archive.getinfo(expected['event']['zip_member'])
        assert info.CRC == expected['event']['zip_crc32'] and info.file_size == expected['event']['bytes']
        asset = extract_member(archive, info, target)
    assert asset['sha256'] == expected['event']['sha256']
    for r in expected['metadata']:
        assert sha(ROOT / r['path']) == r['sha256']
    save(dest, dict(expected, event=asset, secondary_reacquisition=True,
         original_receipt_sha256=sha(PARENT / 'data_receipts' / (args.sequence + '.json')),
         HTTP_range_requests=source.requests, downloaded_bytes=source.transferred))
    print('Exact public member reacquired', args.sequence, asset['bytes'], flush=True)

if __name__ == '__main__':
    main()
