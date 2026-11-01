"""Read-only derivations for submission review; no new detector experiment."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];PARENT=ROOT/'experiments/fp_mechanism_v01';OUT=ROOT/'experiments/submission_audit_v01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(x):return '--' if x is None else f'{x:.2f}'
def make_table(caption,label,headers,rows,widths):
    return '\n'.join([r'\begin{table}[!htbp]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3.5pt}',r'\renewcommand{\arraystretch}{1.12}',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabularx}{\linewidth}{@{}'+widths+'@{}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(map(str,r))+r' \\' for r in rows],r'\bottomrule',r'\end{tabularx}',r'\end{table}'])+'\n'
def derive():
    cfg=read(ROOT/'code/configs/fp_mechanism_v01.json');points=read(PARENT/'operating_points.json');pairs=read(PARENT/'paired_fp_contrasts.json');models=cfg['models'];names={'spike_original':'S0','spike_adapted':'S1','rvt_stream':'RVT'};seqs=cfg['sequences'];transfer=[];budget=[];allmethods=[]
    for model in models:
        for seq in seqs:
            r=next(r for r in pairs if r['model']==model and r['sequence']==seq and r['target']==1. and r['regime']=='thun_calibrated' and r['method_a']=='ema/absolute' and r['method_b']=='current_activity');transfer.append(r)
        for target in cfg['fp_targets']:
            record=dict(model=model,target=target)
            for name,a,b in [('activity','ema/absolute','current_activity'),('lag','ema_lag3/current','ema/current')]:
                rr=[r for r in pairs if r['model']==model and r['target']==target and r['regime']=='retrospective_sequence_budget' and r['method_a']==a and r['method_b']==b];valid=[r for r in rr if r['target_matched']];values=[r['delta_recall_pp'] for r in valid];record[name]=dict(total=len(rr),matched=len(valid),positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),ties=sum(v==0 for v in values),median_delta_recall_pp=float(np.median(values)) if values else None,per_recording=[dict(sequence=r['sequence'],delta_recall_pp=r['delta_recall_pp'],matched=r['target_matched'],fp_a=r['fp_a'],fp_b=r['fp_b']) for r in rr])
            budget.append(record)
    for method in cfg['methods']:
        r=dict(method=method,carriers={})
        for model in models:
            rr=[r for r in points if r['model']==model and r['method']==method and r['target']==1. and r['regime']=='retrospective_sequence_budget'];matched=sum(v['under_budget'] and v['target_shortfall']<=max(1,int(np.floor(.01*v['allowed_fp']))) for v in rr);r['carriers'][model]=dict(recordings=len(rr),near_target=matched,median_recall=float(np.median([v['recall'] for v in rr])),min_fppi=min(v['fppi'] for v in rr),max_fppi=max(v['fppi'] for v in rr))
        allmethods.append(r)
    rows=[[names[r['model']],r['sequence'][12:14],fmt(100*r['recall_a']),fmt(100*r['recall_b']),f"{r['fppi_a']:.3f}",f"{r['fppi_b']:.3f}",'yes' if r['target_matched'] else 'no'] for r in transfer]
    tables={'transfer':make_table(r'Thun-only calibration transfer at the nominal 1 FP/query calibration target, 20\% site capacity. A is absolute EMA and B is current activity. Confidence thresholds are applied unchanged to each additional recording. Recall is complete-GT percent; FP/query is actually attained on that recording. None of these twelve pairs meets the declared evaluation-target matching rule. These are fixed-threshold transfer observations, not equal-FP superiority comparisons.','tab:transfer',['Carrier','Group',r'$R_A$',r'$R_B$',r'FP/q A',r'FP/q B','Matched'],rows,'ll'+r'*{5}{>{\raggedleft\arraybackslash}X}')}
    rows=[]
    for r in budget:
        a=r['activity'];b=r['lag'];rows.append([names[r['model']],f"{r['target']:g}",f"{a['matched']}/4",f"{a['positive']}/{a['negative']}/{a['ties']}",f"{a['median_delta_recall_pp']:+.2f}" if a['median_delta_recall_pp'] is not None else '--',f"{b['positive']}/{b['negative']}/{b['ties']}",f"{b['median_delta_recall_pp']:+.2f}" if b['median_delta_recall_pp'] is not None else '--'])
    tables['budget']=make_table(r'All four prespecified retrospective FP targets, holding site capacity at 20\%. A compares absolute EMA minus current activity; L compares lag3 minus EMA under current normalization. Signs are positive/negative/tied complete-GT recall differences across the four recording groups; medians are percentage points. Both contrasts attain target matching in all four groups at every listed target. These descriptive counts are not independent-frame significance tests.','tab:budget',['Carrier','FP/q','A matched','A signs',r'A $\Delta R$','L signs',r'L $\Delta R$'],rows,'ll'+r'*{5}{>{\raggedleft\arraybackslash}X}')
    labels={'dense':'Dense (all sites)','ema/current':'EMA current','ema/absolute':'EMA absolute','ema_lag3/current':'Lag3 current','current_activity':'Current activity','ema_activity':'Historical activity','persistence/absolute':'Persistence absolute','ema_roll/absolute':'Spatial-roll absolute','unsigned_residual':'Unsigned residual'}
    rows=[]
    for r in allmethods:
        rows.append([labels[r['method']]]+[fmt(100*r['carriers'][m]['median_recall'])+' ['+str(r['carriers'][m]['near_target'])+'/4]' for m in models])
    tables['allmethods']=make_table(r'Complete-GT recall medians (percent) over the same four recordings for all nine conditions at the retrospective 1 FP/query target. Brackets give the number of recordings attaining the declared integer target within tolerance. All conditions use 20\% site capacity except the full-site dense reference. These are separate per-carrier summaries, not a controlled comparison of detector architectures; paired recording-level counts and both threshold regimes remain archived.','tab:allmethods',['Condition','S0','S1','RVT'],rows,r'>{\raggedright\arraybackslash}p{.34\linewidth}*{3}{>{\raggedleft\arraybackslash}X}')
    return dict(transfer=transfer,budget=budget,allmethods=allmethods),tables
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'derived.json').exists():raise FileExistsError('Completed review derivations may not be overwritten')
    derived,tables=derive();inputs={str(p.relative_to(ROOT)):sha(p) for p in [PARENT/'artifact_manifest.json',PARENT/'operating_points.json',PARENT/'paired_fp_contrasts.json',ROOT/'code/configs/fp_mechanism_v01.json']}
    (OUT/'inputs.json').write_text(json.dumps(dict(role='Manuscript audit of existing verified outcomes; no new experiment or independent validation',inputs=inputs,script_sha256=sha(Path(__file__))),indent=2)+'\n');(OUT/'derived.json').write_text(json.dumps(derived,indent=2)+'\n')
    for name,source in tables.items():(OUT/(name+'.texfrag')).write_text(source,encoding='utf-8')
    print(json.dumps(dict(transfer_rows=len(derived['transfer']),budget_rows=len(derived['budget']),methods=len(derived['allmethods']),transfer_pairs_target_matched=sum(r['target_matched'] for r in derived['transfer']))))
if __name__=='__main__':main()
