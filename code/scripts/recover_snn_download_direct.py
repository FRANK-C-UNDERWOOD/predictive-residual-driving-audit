"""Use direct public HTTPS for this process only; keep recovery bytes/checks unchanged."""
import os,runpy
from snn_secondary_common import ROOT,OUT,read,sha

receipt=read(OUT/'transport_direct_route_v03.json')
for path,digest in receipt['sources'].items():
    if sha(ROOT/path)!=digest:raise ValueError('Direct-route source drift: '+path)
os.environ['NO_PROXY']='download.ifi.uzh.ch'
runpy.run_path(str(ROOT/'code/scripts/recover_snn_download.py'),run_name='__main__')
