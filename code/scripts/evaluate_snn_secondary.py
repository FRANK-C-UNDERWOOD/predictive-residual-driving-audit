"""Run the unchanged actual-FP evaluator in the separate secondary output context."""
import runpy
from snn_secondary_common import ROOT, activate
activate()
runpy.run_path(str(ROOT / 'code/scripts/evaluate_driving_confirmation.py'), run_name='__main__')
