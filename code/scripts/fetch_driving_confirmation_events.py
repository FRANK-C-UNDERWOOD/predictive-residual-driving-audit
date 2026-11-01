"""Extract a named public ZIP member by HTTP ranges, without a duplicate ZIP copy."""
from pathlib import Path
import argparse, hashlib, io, json, shutil, time, urllib.request, zipfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/driving_confirmation_v02'
CFG = ROOT / 'code/configs/driving_confirmation_v02_run.json'
RESERVE = 2 * 1024**3

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()

class PublicRangeFile(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        if urlparse(url).hostname != 'download.ifi.uzh.ch':
            raise ValueError('This reader is limited to the official public publisher host')
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.length = int(r.headers['Content-Length'])
            self.etag = r.headers.get('ETag')
            self.modified = r.headers.get('Last-Modified')
        self.requests, self.transferred = 0, 0
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if target < 0:
            raise ValueError('Negative seek')
        self.position = target
        return target
    def read(self, size=-1):
        if self.position >= self.length:
            return b''
        stop = self.length if size < 0 else min(self.length, self.position + size)
        if stop <= self.position:
            return b''
        headers = {'Range': f'bytes={self.position}-{stop - 1}', 'User-Agent': 'SCI-public-research/1.0'}
        if self.etag:
            headers['If-Range'] = self.etag
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=90) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {self.position}-{stop - 1}/{self.length}':
                        raise ValueError('Publisher range/identity changed; refusing full-file fallback')
                    if self.etag and r.headers.get('ETag') != self.etag:
                        raise ValueError('Publisher ETag changed during read')
                    data = r.read(stop - self.position + 1)
                if len(data) != stop - self.position:
                    raise ValueError('Truncated HTTP range')
                self.requests += 1
                self.transferred += len(data)
                self.position = stop
                return data
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

def extract_member(archive, info, target):
    allowed = (ROOT / 'datasets/dsec_confirmation').resolve()
    target.resolve().relative_to(allowed)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != info.file_size:
            raise ValueError('Existing public asset has a different size')
        return dict(path=target.relative_to(ROOT).as_posix(), bytes=target.stat().st_size, sha256=sha(target),
                    zip_member=info.filename, zip_crc32=info.CRC, reused=True)
    if shutil.disk_usage(ROOT).free < info.file_size + RESERVE:
        raise RuntimeError('Insufficient space for this new event file plus2GiB reserve; old assets untouched')
    temporary = target.with_suffix(target.suffix + '.part')
    if temporary.exists():
        raise FileExistsError('Partial extraction retained; inspect before replacing')
    digest, count, tick = hashlib.sha256(), 0, time.monotonic()
    with archive.open(info) as source, temporary.open('xb') as output:
        while block := source.read(32 * 1024**2):
            output.write(block)
            digest.update(block)
            count += len(block)
            if time.monotonic() - tick >= 15:
                print('Extracting', info.filename, round(count / 1024**2), '/', round(info.file_size / 1024**2), 'MiB', flush=True)
                tick = time.monotonic()
    if count != info.file_size:
        raise ValueError('Extracted size differs')
    # ZipExtFile validates CRC at EOF; the raw member gets its own SHA256.
    temporary.replace(target)
    return dict(path=target.relative_to(ROOT).as_posix(), bytes=count, sha256=digest.hexdigest(),
                zip_member=info.filename, zip_crc32=info.CRC, zip_crc_verified=True)

def probe(cfg):
    results = []
    for kind in ['events', 'images']:
        url = cfg['new_sequence_' + kind + '_archive']
        source = PublicRangeFile(url)
        with zipfile.ZipFile(source) as archive:
            members = [dict(name=x.filename, bytes=x.file_size, compressed_bytes=x.compress_size,
                            crc32=x.CRC, compression=x.compress_type) for x in archive.infolist()]
            save(OUT / ('extra_' + kind + '_members.json'), dict(url=url, etag=source.etag, bytes=source.length, members=members))
            grid = [x for x in archive.infolist() if 'thun_02_a' in x.filename and
                    (x.filename.endswith('image_timestamps.txt') or x.filename.endswith('/images/timestamps.txt'))]
            for info in grid:
                target = ROOT / 'datasets/dsec_confirmation/test/thun_02_a/image_timestamps.txt'
                results.append(extract_member(archive, info, target))
        print('Official extra archive', kind, 'members', len(members), 'timestamp assets', len(grid), flush=True)
    save(OUT / 'extra_probe.json', dict(assets=results, no_model_outputs=True))

def fetch(cfg, seq):
    if seq not in cfg['sequences']:
        raise ValueError('Sequence was not selected')
    dest = OUT / 'data_receipts' / (seq + '.json')
    if dest.exists():
        record = json.loads(dest.read_text())
        if (ROOT / record['event']['path']).exists() and sha(ROOT / record['event']['path']) == record['event']['sha256']:
            print('Verified existing public input', seq, flush=True)
            return
        raise ValueError('Input receipt exists but local file is absent or changed')
    url = (cfg['new_sequence_events_archive'] if seq == 'thun_02_a' else
           f'https://download.ifi.uzh.ch/rpg/DSEC/test/{seq}/{seq}_events_left.zip')
    source = PublicRangeFile(url)
    with zipfile.ZipFile(source) as archive:
        candidates = [x for x in archive.infolist() if x.filename.endswith('events.h5')
                      and (seq != 'thun_02_a' or x.filename.endswith('/' + seq + '/events/left/events.h5'))]
        if len(candidates) != 1:
            raise ValueError('Ambiguous named event member: ' + str([x.filename for x in candidates]))
        target = ROOT / 'datasets/dsec_confirmation/test' / seq / 'events/left/events.h5'
        asset = extract_member(archive, candidates[0], target)
    folder = ROOT / 'datasets/dsec_confirmation/test' / seq
    paths = [folder / 'image_timestamps.txt', folder / 'object_detections/left/tracks.npy']
    record = dict(sequence=seq, event=asset,
          metadata=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in paths],
          requested_url=url, etag=source.etag, last_modified=source.modified, archive_bytes=source.length,
          HTTP_range_requests=source.requests, downloaded_bytes=source.transferred,
          checksum_scope='Local extracted member SHA256 and publisher ZIP CRC; not an independently published SHA256',
          storage='Direct range extraction of selected public member; no duplicate multi-GB ZIP retained')
    save(dest, record)
    print('Named complete event input ready', seq, asset['bytes'], asset['sha256'], flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['probe', 'fetch'], required=True)
    parser.add_argument('--sequence')
    args = parser.parse_args()
    cfg = json.loads(CFG.read_text())
    if args.phase == 'probe':
        probe(cfg)
    else:
        fetch(cfg, args.sequence)

if __name__ == '__main__':
    main()
