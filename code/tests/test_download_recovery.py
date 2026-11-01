"""Offline recovery contracts for a DEFLATE member and an existing raw prefix."""
import hashlib,io,sys,tempfile,unittest,zipfile
from unittest.mock import MagicMock,patch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/scripts'))
from recover_snn_download import resume_member
from recover_snn_download import PersistentPublicRanges
import recover_snn_download

class DownloadRecoveryContracts(unittest.TestCase):
    def setUp(self):
        base=ROOT/'tmp/download_recovery_tests';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base)
        self.folder=Path(self.tmp.name).resolve()
        self.folder.relative_to(base.resolve())
        self.data=bytes(range(256))*8192+b'end'
        self.buffer=io.BytesIO()
        with zipfile.ZipFile(self.buffer,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('events.h5',self.data)
        self.digest=hashlib.sha256(self.data).hexdigest()

    def tearDown(self):
        self.folder.relative_to((ROOT/'tmp/download_recovery_tests').resolve())
        self.tmp.cleanup()

    def test_existing_prefix_preserved_and_tail_appended(self):
        target=self.folder/'events.h5';partial=target.with_suffix('.h5.part');prefix=self.data[:700001];partial.write_bytes(prefix)
        progress=[]
        with zipfile.ZipFile(self.buffer) as z:
            result=resume_member(z,z.getinfo('events.h5'),target,self.digest,self.folder,progress.append,131072)
        self.assertEqual(target.read_bytes(),self.data)
        self.assertEqual(result['preserved_partial_bytes'],len(prefix))
        self.assertEqual(result['appended_bytes'],len(self.data)-len(prefix))
        self.assertEqual(result['original_partial_sha256'],hashlib.sha256(prefix).hexdigest())
        self.assertFalse(partial.exists())
        self.assertTrue(any(r['phase']=='revalidating_existing_prefix' for r in progress))

    def test_mismatch_never_overwrites_prefix(self):
        target=self.folder/'events.h5';partial=target.with_suffix('.h5.part');wrong=b'bad-prefix';partial.write_bytes(wrong)
        with zipfile.ZipFile(self.buffer) as z:
            with self.assertRaises(ValueError):resume_member(z,z.getinfo('events.h5'),target,self.digest,self.folder,block_size=131072)
        self.assertEqual(partial.read_bytes(),wrong);self.assertFalse(target.exists())

    def test_bad_expected_hash_retains_completed_partial(self):
        target=self.folder/'events.h5'
        with zipfile.ZipFile(self.buffer) as z:
            with self.assertRaises(ValueError):resume_member(z,z.getinfo('events.h5'),target,'0'*64,self.folder,block_size=131072)
        self.assertFalse(target.exists());self.assertEqual(target.with_suffix('.h5.part').read_bytes(),self.data)

    def test_http_status_code_and_verified_chunk_cache(self):
        expected=dict(requested_url='https://download.ifi.uzh.ch/fixture.zip',archive_bytes=4,etag='"fixture"')
        response=MagicMock();response.__enter__.return_value=response
        response.status_code=206;response.headers={'Content-Range':'bytes 0-3/4','ETag':'"fixture"'};response.content=b'abcd'
        session=MagicMock();session.get.return_value=response
        with patch.object(recover_snn_download,'OUT',self.folder),patch.object(PersistentPublicRanges,'_head'):
            source=PersistentPublicRanges(expected,workers=1)
            try:
                with patch.object(source,'_session',return_value=session):
                    self.assertEqual(source.read(4),b'abcd');source.seek(0)
                    self.assertEqual(source.read(4),b'abcd')
                self.assertEqual(session.get.call_count,1);self.assertEqual(source.cache_hits,1)
            finally:source.close()

    def test_refuse_full_file_response_for_range(self):
        expected=dict(requested_url='https://download.ifi.uzh.ch/fixture.zip',archive_bytes=4,etag='"fixture"')
        response=MagicMock();response.__enter__.return_value=response;response.status_code=200;response.headers={}
        session=MagicMock();session.get.return_value=response
        with patch.object(recover_snn_download,'OUT',self.folder),patch.object(PersistentPublicRanges,'_head'):
            source=PersistentPublicRanges(expected,workers=1)
            try:
                with patch.object(source,'_session',return_value=session):
                    with self.assertRaises(ValueError):source.read(4)
            finally:source.close()

if __name__=='__main__':unittest.main()
