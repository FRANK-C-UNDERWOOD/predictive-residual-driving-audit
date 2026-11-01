import argparse
from summarize_snn_secondary import aggregate
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--sequence', required=True)
a = p.parse_args()
aggregate(a.model, a.sequence)
