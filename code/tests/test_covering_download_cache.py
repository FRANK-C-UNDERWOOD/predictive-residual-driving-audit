import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/scripts'))
from recover_snn_download_high_throughput import CoveringCache

class CoveringCacheContracts(unittest.TestCase):
    def setUp(self):
        base=ROOT/'tmp/covering_cache_tests';base.mkdir(parents=True,exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=base)
        self.folder=Path(self.tmp.name).resolve();self.folder.relative_to(base.resolve())
    def tearDown(self):
        self.folder.relative_to((ROOT/'tmp/covering_cache_tests').resolve());self.tmp.cleanup()
    def store(self,start,data):
        end=start+len(data);p=self.folder/f'{start}_{end}.bin';p.write_bytes(data)
        p.with_suffix('.json').write_text(json.dumps(dict(start=start,end=end,etag='fixture',sha256=hashlib.sha256(data).hexdigest())))
    def test_stitch_misaligned_intervals(self):
        self.store(0,b'abcd');self.store(4,b'efgh');self.store(8,b'ijkl')
        self.assertEqual(CoveringCache(self.folder,'fixture').get(2,10),b'cdefghij')
    def test_cache_gap_returns_no_claimed_data(self):
        self.store(0,b'abcd');self.store(5,b'fgh');self.assertIsNone(CoveringCache(self.folder,'fixture').get(2,7))
    def test_detect_corrupted_cached_chunk(self):
        self.store(0,b'abcd');p=self.folder/'0_4.bin';p.write_bytes(b'bad!')
        with self.assertRaises(ValueError):CoveringCache(self.folder,'fixture').get(0,4)

if __name__=='__main__':unittest.main()
