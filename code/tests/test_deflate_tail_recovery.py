import hashlib,io,random,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/scripts'))
from deflate_tail_recovery import validated_tail

class TailRecoveryContracts(unittest.TestCase):
    def setUp(self):
        base=ROOT/'tmp/tail_recovery_tests';base.mkdir(parents=True,exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=base)
        self.root=Path(self.tmp.name).resolve();self.root.relative_to(base.resolve())
    def tearDown(self):
        self.root.relative_to((ROOT/'tmp/tail_recovery_tests').resolve());self.tmp.cleanup()
    def build(self,data,level):
        source=io.BytesIO()
        with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED,compresslevel=level) as z:z.writestr('events.h5',data)
        with zipfile.ZipFile(source) as z:info=z.getinfo('events.h5')
        return source,info
    def test_tail_after_stored_block_preserves_prefix(self):
        data=random.Random(1).randbytes(2*1024**2);source,info=self.build(data,0)
        p=self.root/'events.h5.part';prefix=data[:700001];p.write_bytes(prefix)
        result=validated_tail(source,info,p,hashlib.sha256(data).hexdigest())
        self.assertEqual(p.read_bytes(),prefix)
        self.assertEqual(prefix+Path(result['tail_path']).read_bytes(),data)
    def test_stored_anchor_then_dictionary_dependent_compressed_blocks(self):
        initial=random.Random(2).randbytes(1024**2);data=initial+initial[-32768:]*8
        source,info=self.build(data,6);p=self.root/'events.h5.part';prefix=data[:900007];p.write_bytes(prefix)
        result=validated_tail(source,info,p,hashlib.sha256(data).hexdigest())
        self.assertEqual(prefix+Path(result['tail_path']).read_bytes(),data)
        self.assertEqual(p.read_bytes(),prefix)
    def test_wrong_hash_never_changes_prefix(self):
        data=random.Random(3).randbytes(1024**2);source,info=self.build(data,0)
        p=self.root/'events.h5.part';prefix=data[:700001];p.write_bytes(prefix)
        with self.assertRaises(ValueError):validated_tail(source,info,p,'0'*64)
        self.assertEqual(p.read_bytes(),prefix)

if __name__=='__main__':unittest.main()
