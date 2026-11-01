"""Complete secondary size reporting, including opposite and undefined outcomes."""
import csv
import numpy as np
from snn_secondary_common import ROOT, OUT, CONFIG, PARENT, read, save, sha
from submission_audit_tables import make_table

def label(model):
    return 'S0' if model == 'spike_original' else 'S1'

def short(seq):
    from report_driving_confirmation_v02 import short as name
    return name(seq)

def number(value, signed=False):
    return '--' if value is None else f'{value:+.2f}' if signed else f'{value:.2f}'

def materials():
    cfg, result = read(CONFIG), read(OUT / 'secondary_results.json')
    assert read(OUT / 'verification.json')['passed']
    main_rows, coverage, budgets = [], [], []
    for model, data in result['models'].items():
        for r in data['effects']:
            main_rows.append([label(model), short(r['sequence']), f"{r['small_gt']}/{r['large_gt']}",
                number(r['small_difference_pp'], True), number(r['large_difference_pp'], True), number(r['interaction_pp'], True),
                f"{r['fp_a']}/{r['fp_b']}", 'yes' if r['matched'] else 'no'])
            coverage.append([label(model), short(r['sequence']), f"{r['fixed_dense_small']}/{r['small_gt']}",
                f"{r['fixed_dense_large']}/{r['large_gt']}",
                '/'.join(number(100 * r['small_preservation_' + m]) if r['small_preservation_' + m] is not None else '--' for m in ['a','b']),
                '/'.join(number(100 * r['large_preservation_' + m]) if r['large_preservation_' + m] is not None else '--' for m in ['a','b'])])
        for target, v in data['budgets'].items():
            s = v['summary']
            effects = [r for r in read(OUT / 'size_interactions.json') if r['model']==model and r['regime']=='retrospective_sequence_budget' and r['target']==float(target)]
            budgets.append([label(model), f'{float(target):g}', f"{sum(r['matched'] for r in effects)}/7", f"{s['defined']}/7", number(s['mean'], True), f"{s['negative']}/{s['positive']}/{s['tied']}"])
    tables = dict(
        primary=make_table(r'Completion of the specified SNN secondary size contrast on the same seven new DSEC recording groups. A/B is absolute EMA/current activity at 20\% site capacity and the actually replayed whole-recording 1 FP/query point. $\Delta R$ is complete-GT recall A minus B (percentage points); $J=\Delta R_{small}-\Delta R_{large}$. Small is at most 1\% native image area. GT counts identity/query instances; actual FP pairs and matching failures remain explicit. No subgroup threshold is fitted. S0/S1 are two checkpoints of one architecture on the same records, not independent architecture replications.',
             'tab:snn_secondary', ['Carrier','Record','GT S/L',r'$\Delta R_S$',r'$\Delta R_L$',r'$J$','FP A/B','Matched'], main_rows,
             'll'+r'*{6}{>{\raggedleft\arraybackslash}X}'),
        coverage=make_table(r'SNN secondary baseline observability and conditional evidence preservation. D/GT uses the original fixed Thun dense-reference threshold, not the retrospective dense 1 FP point. Q A/B is the fraction (percent) of that fixed reference preserved by absolute EMA/activity at their main whole-recording points. Empty reference denominators are undefined, never zero; high Q cannot repair baseline misses. Counts are identity/query instances rather than independent targets.',
             'tab:snn_secondary_coverage', ['Carrier','Record','D/GT small','D/GT large',r'$Q_S$ A/B (\%)',r'$Q_L$ A/B (\%)'], coverage,
             'll'+r'*{4}{>{\raggedleft\arraybackslash}X}'),
        budget=make_table(r'All four predeclared retrospective FP targets in the SNN secondary cohort, with capacity fixed at 20\%. Mean J equally weights recordings with defined size strata; defined and FP-matched counts are separately retained. Signs are negative/positive/tied. Only 1 FP/query is the main secondary contrast; other points are descriptive sensitivity analyses. Nominal intervals are secondary and not multiplicity controlled across checkpoints or targets.',
             'tab:snn_secondary_budgets',['Carrier','FP/query','FP matched','Defined','Mean J (pp)','Signs'],budgets,
             'll'+r'*{4}{>{\raggedleft\arraybackslash}X}'))
    return tables, result

def main():
    assert not (OUT / 'report_receipt.json').exists()
    tables, result = materials()
    for name, table in tables.items():
        (OUT / (name + '.texfrag')).write_text(table, encoding='utf-8')
    points = read(OUT / 'operating_points.json')
    with (OUT / 'operating_points.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.8))
    for ax,(model,data) in zip(axes,result['models'].items()):
        for i,r in enumerate(data['effects']):
            values=[r['small_difference_pp'],r['large_difference_pp']]
            if all(v is not None for v in values):ax.plot(values,[i,i],color='.65',linewidth=.8)
            for j,value in enumerate(values):
                if value is not None:ax.scatter(value,i,color=['#2166ac','#b2182b'][j],marker=['o','s'][j],s=25,zorder=3)
                else:ax.text(.98,i,('small' if j==0 else 'large')+' undefined',transform=ax.get_yaxis_transform(),ha='right',fontsize=6)
        ax.axvline(0,color='.4',linewidth=.7);ax.set_yticks(range(7),[short(r['sequence']) for r in data['effects']]);ax.invert_yaxis()
        ax.set_title(label(model)+' / '+('original' if label(model)=='S0' else 'selected adaptation'));ax.set_xlabel('Recall difference (pp)\nAbsolute EMA minus activity');ax.grid(axis='x',alpha=.18)
    handles=[Line2D([],[],color=c,marker=m,linestyle='None',label=l) for c,m,l in [('#2166ac','o','Small targets'),('#b2182b','s','Larger targets')]]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False);fig.subplots_adjust(left=.1,right=.98,bottom=.25,top=.9,wspace=.5)
    for ext in ['png','svg']:fig.savefig(OUT/('snn_size_secondary.'+ext),dpi=280,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    lines=['# 七组新录制数据：SNN 次级测试完整结果','','原始S0和selected116适配S1检查点固定；同一架构、相同七组，不能按十四个独立来源计算。RVT主结果已知，大小交互预测先前已提出，本轮在新SNN输出前冻结执行。','']
    for model,data in result['models'].items():
        s=data['summary'];ci=data['nominal_recording_t_CI95']
        lines.extend(['## '+label(model),f"J均值={number(s['mean'],True)}pp，中位数={number(s['median'],True)}pp；有定义{s['defined']}/7；负/正/平={s['negative']}/{s['positive']}/{s['tied']}；全部匹配={data['all_matched']}。",f'名义95%t区间={ci}。只作次级描述；共享录制/城市且检查点相关，未作多重比较控制。'])
        for r in data['effects']:lines.append(f"- {r['sequence']}：GT小/大={r['small_gt']}/{r['large_gt']}，召回差小/大={number(r['small_difference_pp'],True)}/{number(r['large_difference_pp'],True)}pp，J={number(r['interaction_pp'],True)}pp；FP={r['fp_a']}/{r['fp_b']}，matched={r['matched']}。")
        lines.append('')
    lines.extend(['## 范围与完整性','大小交互不等于大小的因果效应。全序列同误报不等于子组同误报；两套权重不是两次独立架构复现。全部1008点和40组分层、原dense覆盖/条件保留、源分解和连续损失保留；缺失/未匹配/反向结果不删除。','当前模型及神经元仍完整运行，掩码只影响候选准入。状态复用的新录制组效果、物理跳算、独立数据来源、Risk修正和安全收益尚未验证；旧Thun特征复用接口工程试验另存，不增加本轮独立样本。','独立复核从标注与原始候选重建匹配；上游NMS/模型输出仍共享，不增加驾驶样本数。事件成员按精确哈希重新取得。Thun01在早期存储方案下释放；恢复Thun02后磁盘容量允许保留后续六组原始成员和传输缓存。哪些资产已保留/释放分别有收据；缺失成员需重新获取，原RVT档案不改写。'])
    lines.extend(['', '## 数值兼容修复',
        'Interlaken01原始检查点回放曾因float32的0.01边界被float64校准验证拒绝而停止。原始概率与门控均未改写；版本化适配器只让该精确编码通过下限验证，返回阈值的排序/nextafter规则不变。继续前，全部72个旧SNN校准点和64组有效输入阈值回归精确通过，NaN及真正低于下限的值仍拒绝。失败记录、适配器、诊断和回归输出保留在numerical_floor_recovery_v07及numerical_floor_compat_v07.json。'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[OUT/name for name in ['secondary_results.json','conditions.json','size_interactions.json','operating_points.csv','REPORT.md','verification.json','primary.texfrag','coverage.texfrag','budget.texfrag','snn_size_secondary.png','snn_size_secondary.svg']]
    save(OUT/'report_receipt.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in files},source_sha256=sha(ROOT/'code/scripts/report_snn_secondary.py'),role='All planned secondary outcomes, not a new primary trial'))
    print('SNN secondary full report and scientific plot complete',flush=True)

if __name__=='__main__':main()
