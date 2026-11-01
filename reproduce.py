"""CPU reproduction of archived operating points, contrasts and descriptive omissions."""
from pathlib import Path
from fractions import Fraction
import argparse,json,csv,math,statistics,hashlib
from scipy.stats import t
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def summary(values):
 m=statistics.mean(values);h=float(t.ppf(.975,len(values)-1))*statistics.stdev(values)/math.sqrt(len(values))
 return dict(n=len(values),mean_pp=m,median_pp=statistics.median(values),nominal_CI95=[m-h,m+h])
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',default='results');parser.add_argument('--skip-hashes',action='store_true');args=parser.parse_args()
 if not args.skip_hashes:
  manifest=read('RELEASE_MANIFEST.json')
  for path,digest in manifest['files'].items():
   assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
 p=read('experiments/driving_confirmation_v02/primary_results.json');s=read('experiments/snn_secondary_v01/secondary_results.json')
 result={};rowsout=[]
 for model,rows in {'RVT':p['effects'],**{k:v['effects'] for k,v in s['models'].items()}}.items():
  values=[]
  for r in rows:
   a,b=('low','high') if model=='RVT' else ('small','large')
   ds=100*Fraction(r[a+'_tp_a']-r[a+'_tp_b'],r[a+'_gt']);dl=100*Fraction(r[b+'_tp_a']-r[b+'_tp_b'],r[b+'_gt']);v=float(ds-dl)
   assert abs(v-r['interaction_pp'])<1e-10
   values.append(v);rowsout.append(dict(carrier=model,recording=r['sequence'],difference_a_pp=float(ds),difference_b_pp=float(dl),interaction_pp=v,fp_a=r['fp_a'],fp_b=r['fp_b']))
  archived=p if model=='RVT' else s['models'][model]
  v=summary(values);assert abs(v['mean_pp']-archived['summary']['mean'])<1e-10
  assert max(abs(x-y) for x,y in zip(v['nominal_CI95'],archived['nominal_recording_t_CI95']))<1e-10
  v['leave_one_recording_out']=[dict(omitted=r['sequence'],**summary(values[:i]+values[i+1:])) for i,r in enumerate(rows)]
  result[model]=v
 totals={}
 stages=['fp_mechanism_v01','driving_confirmation_v02','snn_secondary_v01']
 for stage in stages:
  q=ROOT/'experiments'/stage/'operating_points.json'
  if q.exists():points=json.loads(q.read_text())
  else:points=[point for f in (ROOT/'experiments'/stage/'points').glob('*__*.json') for point in json.loads(f.read_text())['conditions'].values()]
  assert len(points)=={'fp_mechanism_v01':864,'driving_confirmation_v02':504,'snn_secondary_v01':1008}[stage]
  for point in points:
   assert point['queries']>0 and 0<=point['tp']<=point['gt']
   assert abs(point['recall']-point['tp']/point['gt'])<1e-12 if point['gt'] else point['recall'] is None
   if point['dense_tp']:assert abs(point['preservation']-point['kept']/point['dense_tp'])<1e-12
   assert abs(point['fppi']-point['fp']/point['queries'])<1e-12
  totals[stage]=len(points)
 result['operating_point_counts']=totals
 result['scope']='Arithmetic and aggregate replay, not new detections, raw-GT re-matching, training, hardware savings or independent confirmation.'
 out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
 (out/'reproduced.json').write_text(json.dumps(result,indent=2)+'\n')
 with (out/'interactions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rowsout[0]));w.writeheader();w.writerows(rowsout)
 for key,v in result.items():
  if isinstance(v,dict) and 'mean_pp' in v:print(key,round(v['mean_pp'],6),v['nominal_CI95'])
 print('PASS: 2376 operating points and all21 interactions rebuilt; outputs',out)
if __name__=='__main__':main()
