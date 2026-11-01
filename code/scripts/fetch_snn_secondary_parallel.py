"""Same official ZIP/member checks with concurrent disjoint HTTP ranges only."""
import concurrent.futures, runpy, time, urllib.request
from snn_secondary_common import ROOT, OUT, read, sha
import fetch_driving_confirmation_events as transport

class ConcurrentPublicRanges(transport.PublicRangeFile):
    def read(self, size=-1):
        if size < 2 * 1024**2:
            return super().read(size)
        stop = min(self.length, self.position + size) if size >= 0 else self.length
        if stop <= self.position:
            return b''
        intervals = [(start, min(stop, start + 1024**2)) for start in range(self.position, stop, 1024**2)]
        def get(interval):
            start, end = interval
            headers = {'Range': f'bytes={start}-{end - 1}', 'If-Range': self.etag, 'Accept-Encoding': 'identity', 'User-Agent': 'SCI-public-research/1.0'}
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=90) as response:
                        assert response.status == 206 and response.headers.get('Content-Range') == f'bytes {start}-{end - 1}/{self.length}'
                        assert response.headers.get('ETag') == self.etag
                        data = response.read(end - start + 1)
                    assert len(data) == end - start
                    return data
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            pieces = list(pool.map(get, intervals))
        self.position = stop
        self.requests += len(pieces)
        self.transferred += sum(map(len, pieces))
        return b''.join(pieces)

if __name__ == '__main__':
    receipt = read(OUT / 'transport_freeze.json')
    for path, digest in receipt['sources'].items():
        assert sha(ROOT / path) == digest, path
    transport.PublicRangeFile = ConcurrentPublicRanges
    runpy.run_path(str(ROOT / 'code/scripts/fetch_snn_secondary.py'), run_name='__main__')
