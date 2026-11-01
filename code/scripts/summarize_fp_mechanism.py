"""Report both FP regimes and all named mechanism contrasts, including failures."""
from collections import defaultdict
import csv,json,shutil
import numpy as np
from confirmation_common import ROOT,read,save,sha
OUT=ROOT/'experiments/fp_mechanism_v01';PARENT=ROOT/'experiments/confirmation_v01'
def jsonlines(path):
    with path.open() as f:return [json.loads(x) for x in f]
def main():
    cfg=read(ROOT/'code/configs/fp_mechanism_v01.json');verify=read(OUT/'verification.json')
    if not verify['passed']:raise ValueError('Independent verification required')
    points=[];pairs=[];mechanisms=[];lookup={};thresholds=read(PARENT/'frozen_thresholds.json')
    for model in cfg['models']:
        for seq in cfg['sequences']:
            data=read(OUT/'points'/(model+'__'+seq+'.json'))['conditions'];lookup[(model,seq)]=data
            for p in data.values():points.append(dict(model=model,sequence=seq,**p))
            for regime in cfg['regimes']:
                for target in cfg['fp_targets']:
                    for a,b in cfg['contrasts']:
                        A=data[regime+'|'+a+'@'+str(target)];B=data[regime+'|'+b+'@'+str(target)];tol=max(1,int(np.floor(.01*A['allowed_fp'])));close=abs(A['fp']-B['fp'])<=tol;attained=A['under_budget'] and B['under_budget'] and A['target_shortfall']<=tol and B['target_shortfall']<=tol
                        pairs.append(dict(model=model,sequence=seq,regime=regime,target=target,method_a=a,method_b=b,fp_a=A['fp'],fp_b=B['fp'],fppi_a=A['fppi'],fppi_b=B['fppi'],threshold_a=A['threshold'],threshold_b=B['threshold'],allowed_fp=A['allowed_fp'],count_tolerance=tol,actual_fp_close=close,both_near_target=attained,target_matched=close and attained,
                            tp_a=A['tp'],tp_b=B['tp'],gt=A['gt'],recall_a=A['recall'],recall_b=B['recall'],delta_recall_pp=100*(A['recall']-B['recall']) if A['recall'] is not None and B['recall'] is not None else None,q_a=A['preservation'],q_b=B['preservation'],delta_q_pp=100*(A['preservation']-B['preservation']) if A['preservation'] is not None and B['preservation'] is not None else None))
            receipt=read(OUT/(model+'__'+seq+'.json'));tot=receipt['common_totals'];den=tot['dense']['dense_tp'];folder=OUT/'floor'/model/seq;qr=jsonlines(folder/'query_mechanism.jsonl');sr=jsonlines(folder/'dense_support.jsonl')
            def meanfield(k):
                v=[r[k] for r in qr if r[k] is not None];return float(np.mean(v)) if v else None
            def q(method):return tot[method]['kept']/den if den else None
            actual=q('ema/absolute')-q('current_activity') if den else None;coverage=(tot['ema/absolute']['source_covered']-tot['current_activity']['source_covered'])/den if den else None
            neg=meanfield('negative_fraction');allres=meanfield('residual_mean');op=meanfield('opposition_mean')
            row=dict(model=model,sequence=seq,dense_instances=den,actual_residual_minus_activity_pp=100*actual if den else None,source_coverage_residual_minus_activity_pp=100*coverage if den else None,candidate_postprocessing_difference_pp=100*(actual-coverage) if den else None,direction_explained=(np.sign(actual)==np.sign(coverage)) if den else None,
                activity_head_spearman_mean=meanfield('activity_head_spearman'),residual_head_spearman_mean=meanfield('residual_head_spearman'),activity_residual_spearman_mean=meanfield('activity_residual_spearman'),negative_fraction_mean=neg,unsigned_mask_equal_fraction=float(np.mean([r['unsigned_mask_same'] for r in qr])),opposed_term_fraction=2*op/allres if allres else None,
                aligned_minus_rolled_q_pp=100*(q('ema/absolute')-q('ema_roll/absolute')) if den else None,unsigned_minus_signed_q_pp=100*(q('unsigned_residual')-q('ema/absolute')) if den else None,unsigned_minus_signed_fp=tot['unsigned_residual']['fp']-tot['ema/absolute']['fp'],
                residual_q=q('ema/absolute'),activity_q=q('current_activity'),roll_q=q('ema_roll/absolute'),unsigned_q=q('unsigned_residual'),activity_selected_padding=sum(r['activity_padding_slots'] for r in qr),residual_selected_padding=sum(r['residual_padding_slots'] for r in qr),activity_gt_center_slots=sum(r['activity_gt_center_slots'] for r in qr),residual_gt_center_slots=sum(r['residual_gt_center_slots'] for r in qr),selected_slots=sum(r['selected_slots'] for r in qr),support_partition=receipt['mechanism_totals'])
            row['direction_explained']=bool(row['direction_explained']) if row['direction_explained'] is not None else None
            row['support_temporal_adjustment_by_selection']={}
            for name,act,res in [('residual_only',False,True),('activity_only',True,False),('both',True,True),('neither',False,False)]:
                v=[r for r in sr if r['activity_selected']==act and r['residual_selected']==res];row['support_temporal_adjustment_by_selection'][name]=dict(instances=len(v),mean_adjustment=float(np.mean([x['temporal_adjustment'] for x in v])) if v else None,mean_opposition=float(np.mean([x['opposition'] for x in v])) if v else None)
            mechanisms.append(row)
    for name,rows in [('operating_points.csv',points),('paired_fp_contrasts.csv',pairs)]:
        with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(OUT/'operating_points.json',points);save(OUT/'paired_fp_contrasts.json',pairs);save(OUT/'mechanism_summary.json',mechanisms)
    primary=[p for p in pairs if p['regime']=='retrospective_sequence_budget' and p['target']==cfg['primary_fp_target']];save(OUT/'primary_comparisons.json',primary)
    names={'spike_original':'SpikeFPN原始','spike_adapted':'SpikeFPN适配','rvt_stream':'RVT流式'};pct=lambda x:'undefined' if x is None else f'{100*x:.2f}%';num=lambda x:'undefined' if x is None else f'{x:+.2f}'
    lines=['# 最后两项追加分析：实际同误报工作点与排序反转机制','',
        '本轮是看到confirmation-v01结果后的回顾性追加分析，不是新的独立确认。保留此前四个同城DSEC训练录制组、三套冻结权重、全部6322查询／模型，以及Thun校准集。没有重新训练或新风险算法。所有网络计算和时序状态仍完整运行，仅限制NMS前候选。','',
        '## 同误报比较的口径','',
        '预定0.25/0.5/1/2 FP/query，主点为1。完整GT召回为主任务端点，Q仍以原固定dense-Thun阈值下的身份—查询为分母。阈值改变时dense自身Q也不必为100%。Thun选阈值直接迁移与评价片段标签选阈值的回顾性预算诊断分开；后者不能称可部署阈值的泛化性能。','',
        '每个报告阈值均实际重新运行上游NMS和一对一匹配；不假设含同分数时的NMS前缀不变。过预算的校准／回顾性点只依据实际FP分数提高阈值后重跑；Thun迁移点不使用新组标签纠正。若双方都在目标整数FP上限内、各自距上限不超过max(1, floor(1%目标FP))且两者FP差不超过同一容差，标target_matched=True。其余点仍列出，但不称目标误报点匹配。不插值产生分数检测。','',
        '## 主工作点：1 FP/query的回顾性配对','']
    for a,b in cfg['contrasts']:
        lines+=['','### '+a+' 对 '+b,'']
        for r in primary:
            if r['method_a']!=a or r['method_b']!=b:continue
            lines.append(f"- {r['sequence']}/{names[r['model']]}：FP {r['fp_a']}/{r['fp_b']}，允许{r['allowed_fp']}、容差{r['count_tolerance']}，匹配={r['target_matched']}；GT召回 {pct(r['recall_a'])}/{pct(r['recall_b'])}，差{num(r['delta_recall_pp'])}pp；Q {pct(r['q_a'])}/{pct(r['q_b'])}，差{num(r['delta_q_pp'])}pp。")
    lines+=['','## 四组方向（仅达到预定误报点的配对）','',
        '有效配对数量只描述匹配条件的覆盖，未匹配组在上面及CSV保留。组间不能凑成帧独立p值，两个Spike检查点也不是两个架构。','']
    group_stats=[]
    for model in cfg['models']:
        for a,b in cfg['contrasts']:
            allr=[r for r in primary if r['model']==model and r['method_a']==a and r['method_b']==b];eligible=[r for r in allr if r['target_matched']];vals=[r['delta_recall_pp'] for r in eligible if r['delta_recall_pp'] is not None];qq=[r['delta_q_pp'] for r in eligible if r['delta_q_pp'] is not None]
            row=dict(model=model,method_a=a,method_b=b,matched_recordings=len(eligible),unmatched_recordings=len(allr)-len(eligible),recall_positive=sum(v>0 for v in vals),recall_negative=sum(v<0 for v in vals),recall_ties=sum(v==0 for v in vals),median_delta_recall_pp=float(np.median(vals)) if vals else None,min_delta_recall_pp=min(vals) if vals else None,max_delta_recall_pp=max(vals) if vals else None,q_positive=sum(v>0 for v in qq),q_negative=sum(v<0 for v in qq));group_stats.append(row)
            lines.append(f"- {names[model]}：{a}−{b}，达到目标匹配{len(eligible)}/4组；召回差正/负/平{row['recall_positive']}/{row['recall_negative']}/{row['recall_ties']}，中位差{num(row['median_delta_recall_pp'])}pp，范围[{num(row['min_delta_recall_pp'])},{num(row['max_delta_recall_pp'])}]pp；Q差正/负{row['q_positive']}/{row['q_negative']}。")
    save(OUT/'primary_group_summaries.json',group_stats)
    lines+=['','## 反转机制：原dense阈值、固定候选的诊断','',
        '原dense TP框的原始anchor/site是源位置。源覆盖差直接描述两个同容量掩码的分歧；实际Q差减去源覆盖差，是替代候选／NMS／匹配造成的差额，不自动视为已识别的单一因果贡献。非dense支持位置不等于背景，GT框内中心占用和padding也只是几何描述。','']
    for r in mechanisms:
        lines.append(f"- {r['sequence']}/{names[r['model']]}：残差−活动实际Q差{num(r['actual_residual_minus_activity_pp'])}pp，源覆盖差{num(r['source_coverage_residual_minus_activity_pp'])}pp，候选／后处理差额{num(r['candidate_postprocessing_difference_pp'])}pp，同方向={r['direction_explained']}；活动/残差与head置信度平均秩相关 {num(r['activity_head_spearman_mean'])}/{num(r['residual_head_spearman_mean'])}。")
    lines+=['','## 历史对应与符号贡献的受控干预','',
        'N=sum|H−P|=M+B−2O：M=sum|H|、B=sum|P|、O为同符号重叠。无符号残差sum||H|−|P||删除相反符号项；空间滚动保留过去特征值分布但打破其位置对应。两项只改选择公式，不改检测器，不能解释成换一种训练后的SNN／ANN。变化同时列FP，不以更高Q自动称更优。','']
    for r in mechanisms:
        lines.append(f"- {r['sequence']}/{names[r['model']]}：负特征元素比例均值{pct(r['negative_fraction_mean'])}，相反符号项／残差均值比例{pct(r['opposed_term_fraction'])}；无符号与原掩码完全相同比例{pct(r['unsigned_mask_equal_fraction'])}；去符号Q差{num(r['unsigned_minus_signed_q_pp'])}pp、FP差{r['unsigned_minus_signed_fp']:+d}；aligned−spatial-roll Q差{num(r['aligned_minus_rolled_q_pp'])}pp。")
    lines+=['','## 软件证据与尚未解决的边界','',
        f"独立核对通过{verify['checks']}项，覆盖{verify['causal_mask_queries']}个因果掩码查询、{verify['point_query_rows']}条实际阈值评价行和{verify['floor_detections_independently_matched']}个最低阈值输出的独立匹配。原始NMS样例重放{verify['sampled_raw_nms_checks']}次，前缀不一致按实际NMS重放{verify['prefix_exceptions_replayed']}次；这些不是独立驾驶样本。",
        '四个组已暴露、同一城市及数据来源，原始／适配Spike属于同一架构且只有一个适配顺序种子。输入、分辨率、时序状态及预训练不同，不能把跨载体反转因果归于架构本体。符号或空间对应的干预若不能解释全部差异，剩余部分必须保留为未解释。',
        '回顾性同FP工作点用于排除部分误报混淆，不能代替独立数据上冻结策略的评价；完整GT召回和实际FP优先于有利的条件Q故事。没有新的危险标签、重要性修正、在线闭环门控、硬件算力／能耗节省或安全保证。','',
        '## 图与数据','',
        '- [实际FP—完整GT召回](E:/SCI/experiments/fp_mechanism_v01/fp_recall.png)；[SVG](E:/SCI/experiments/fp_mechanism_v01/fp_recall.svg)。',
        '- [源覆盖与机制干预](E:/SCI/experiments/fp_mechanism_v01/mechanism_reversal.png)；[SVG](E:/SCI/experiments/fp_mechanism_v01/mechanism_reversal.svg)。',
        '- [全部工作点](E:/SCI/experiments/fp_mechanism_v01/operating_points.csv)、[全部配对](E:/SCI/experiments/fp_mechanism_v01/paired_fp_contrasts.csv)、[机制数值](E:/SCI/experiments/fp_mechanism_v01/mechanism_summary.json)。']
    source=read(OUT/'source_survival/summary.json');source_check=read(OUT/'source_survival/verification.json')
    if not source_check['passed']:raise ValueError('Source survival refinement must pass')
    source_contrasts=[];source_lines=['','## 原始源、替代候选与NMS／匹配去向','',
        '这一步细化原计划的源覆盖—实际保留差额，只重放已经报告的共同阈值和回顾性1FP点，没有新增策略或阈值。有效源覆盖额外要求原始框达到该策略置信度。对每个原dense身份：Q = 有效源覆盖 + 原始源不可用时的替代恢复 − 有效源保留却未命中的损失。损失再区分NMS删除和匹配分配。替代命中不自动等于“原来被NMS抑制”，也可能来自原来未匹配的冗余框。','']
    for model in cfg['models']:
        for seq in cfg['sequences']:
            for regime in ['common_dense','retrospective_1fp']:
                a=next(r for r in source if r['model']==model and r['sequence']==seq and r['regime']==regime and r['method']=='ema/absolute');b=next(r for r in source if r['model']==model and r['sequence']==seq and r['regime']==regime and r['method']=='current_activity');den=a['dense_instances'];dc=a['effective_source_coverage']-b['effective_source_coverage'];da=(a['alternative_after_mask']+a['alternative_after_confidence'])-(b['alternative_after_mask']+b['alternative_after_confidence']);dl=(a['nms_loss']+a['assignment_loss'])-(b['nms_loss']+b['assignment_loss']);dq=a['kept']-b['kept']
                if dq!=dc+da-dl:raise AssertionError('Source survival decomposition failed')
                row=dict(model=model,sequence=seq,regime=regime,dense_instances=den,delta_kept=dq,delta_effective_source=dc,delta_alternative_recovery=da,delta_selected_source_loss=dl,delta_q_pp=100*dq/den if den else None,delta_source_pp=100*dc/den if den else None,delta_alternative_pp=100*da/den if den else None,delta_loss_pp=100*dl/den if den else None,residual_nms_loss=a['nms_loss'],activity_nms_loss=b['nms_loss'],residual_assignment_loss=a['assignment_loss'],activity_assignment_loss=b['assignment_loss']);source_contrasts.append(row)
                if model=='rvt_stream':source_lines.append(f"- {seq}/{regime}，残差−活动：有效源覆盖{dc:+d}，替代恢复{da:+d}，有效源损失{dl:+d}；保留差{dq:+d} = {dc:+d} + ({da:+d}) − ({dl:+d})，即{num(row['delta_q_pp'])}pp。NMS损失残差/活动={a['nms_loss']}/{b['nms_loss']}，匹配损失={a['assignment_loss']}/{b['assignment_loss']}。")
    source_lines+=['',f"该去向重放与既有端点逐帧相同，独立分类计数通过{source_check['conditions']}个条件、{source_check['identity_query_rows']}条身份—查询记录。完整分类及全部载体的分解保存在source_survival/和source_partition_contrasts.json。原始源格点不是唯一可用证据；这解释了部分源覆盖与最终身份保留的反向，但不把架构、分辨率或循环状态宣称为单一环境原因。"]
    save(OUT/'source_partition_contrasts.json',source_contrasts)
    lines+=source_lines
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles=[('dense','black','*'),('ema/absolute','#287747','s'),('current_activity','#777777','x'),('ema/current','#8748a5','o'),('ema_lag3/current','#d28b24','^')]
    fig,axes=plt.subplots(4,3,figsize=(6.4,8.5),layout='constrained',sharey=True)
    for row,seq in enumerate(cfg['sequences']):
        for col,model in enumerate(cfg['models']):
            ax=axes[row,col];data=lookup[(model,seq)]
            for method,color,marker in styles:
                pp=[data['retrospective_sequence_budget|'+method+'@'+str(t)] for t in cfg['fp_targets']];ax.plot([p['fppi'] for p in pp],[100*p['recall'] if p['recall'] is not None else np.nan for p in pp],marker=marker,color=color,label=method)
            ax.set_title(seq.replace('zurich_city_','Z')+' / '+{'spike_original':'S0','spike_adapted':'S1','rvt_stream':'RVT'}[model],fontsize=8);ax.grid(alpha=.2);ax.set_ylim(0,100);ax.set_xticks([.25,1,2]);ax.set_yticks([0,50,100]);ax.tick_params(labelsize=7)
            if col==0:ax.set_ylabel('Full-GT recall (%)',fontsize=7.5)
            if row==3:ax.set_xlabel('Actual FP / query',fontsize=7.5)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=7);fig.suptitle('Retrospective FP-budget points at 20% admission\nEvaluation-label thresholds; full detector computation\nLines guide the eye, not threshold interpolation',fontsize=9)
    for ext in ['png','svg']:fig.savefig(OUT/('fp_recall.'+ext),dpi=350)
    plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(6.4,6.6),layout='constrained');x=np.arange(4)
    for row,model in enumerate(cfg['models']):
        rr=[next(r for r in mechanisms if r['model']==model and r['sequence']==seq) for seq in cfg['sequences']]
        for col,series in enumerate([[('actual_residual_minus_activity_pp','Actual Q difference','#287747'),('source_coverage_residual_minus_activity_pp','Dense source coverage difference','#777777')],[('aligned_minus_rolled_q_pp','Aligned minus spatial roll','#d28b24'),('unsigned_minus_signed_q_pp','Unsigned minus signed residual','#377eb8')]]):
            ax=axes[row,col]
            for j,(field,label,color) in enumerate(series):ax.bar(x+(j-.5)*.35,[r[field] if r[field] is not None else np.nan for r in rr],.35,label=label,color=color)
            ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,['Z07','Z08','Z09','Z10']);ax.set_ylabel('Difference (pp)',fontsize=7.5);ax.tick_params(labelsize=7);ax.set_title({'spike_original':'S0','spike_adapted':'S1','rvt_stream':'RVT'}[model]+(' / residual minus activity' if col==0 else ' / score interventions'),fontsize=8);ax.grid(axis='y',alpha=.2)
    hh=[];ll=[]
    for ax in axes[0]:h,l=ax.get_legend_handles_labels();hh+=h;ll+=l
    fig.legend(hh,ll,loc='outside lower center',ncol=2,fontsize=7);fig.suptitle('Fixed candidates and original dense threshold\nOriginal source coverage and controlled score changes\nActual FP and matched-FP results reported separately',fontsize=9)
    for ext in ['png','svg']:fig.savefig(OUT/('mechanism_reversal.'+ext),dpi=350)
    plt.close(fig)
    save(OUT/'STATUS.json',dict(state='complete',verification_passed=True))
    sources=list(read(OUT/'freeze.json')['sources'])+['code/scripts/verify_fp_mechanism.py','code/scripts/summarize_fp_mechanism.py','code/scripts/trace_source_survival.py']
    for name in sources:
        p=OUT/'executed_sources'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    files=[p for p in OUT.rglob('*') if p.is_file() and 'executed_sources' not in p.parts and p.name not in ['artifact_manifest.json','pipeline_status.json','pipeline.lock']]
    save(OUT/'artifact_manifest.json',dict(sources=[dict(path=p,sha256=sha(ROOT/p)) for p in sources],files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files]));print('Final secondary analysis report and figures completed',flush=True)
if __name__=='__main__':main()
