"""Report all prespecified strata; produce manuscript tables from checked counts."""
from pathlib import Path
import json, shutil, sys
import numpy as np
from confirmation_common import ROOT, read, save, sha
from submission_audit_tables import make_table

OUT = ROOT / 'experiments/driving_factors_v01'
LABELS = {'all': 'All', 'small': 'Small', 'large': 'Larger', 'early': 'Early annotation',
          'established': 'Established', 'left_censored': 'Left-censored',
          'motion_low': 'Low image motion', 'motion_middle': 'Middle image motion',
          'motion_high': 'High image motion', 'motion_unknown': 'Motion unknown',
          'car': 'Car', 'pedestrian': 'Pedestrian', 'small_early': 'Small, early',
          'small_established': 'Small, established', 'large_early': 'Larger, early',
          'large_established': 'Larger, established'}

def materials():
    cfg = read(ROOT / 'code/configs/driving_factors_v01.json')
    if not read(OUT / 'verification.json')['passed']:
        raise ValueError('Independent subgroup verification required')
    summaries = read(OUT / 'group_summaries.json')
    pairs = read(OUT / 'pairs.json')
    continuity = read(OUT / 'continuity_summary.json')
    models = cfg['models']
    population, contrast, runrows = [], [], []
    lookup = {r['group']: r for r in summaries if r['model'] == models[0] and r['regime'] == 'retrospective_sequence_budget' and r['target'] == 1 and r['method_a'] == 'ema/absolute' and r['method_b'] == 'current_activity'}
    selected = {(r['model'], r['group']): r for r in summaries if r['regime'] == 'retrospective_sequence_budget' and r['target'] == 1 and r['method_a'] == 'ema/absolute' and r['method_b'] == 'current_activity'}
    for group in cfg['groups']:
        r = lookup[group]
        population.append([LABELS[group], r['gt_instances'], r['gt_tracks']] +
                          [selected[m, group]['dense_instances'] for m in models])
        values = []
        for model in models:
            v = selected[model, group]
            values.append(('--' if v['median_delta_recall_pp'] is None else f"{v['median_delta_recall_pp']:+.2f}") +
                          f" [{v['positive']}/{v['negative']}/{v['tied']}]")
        contrast.append([LABELS[group], *values])
    names = dict(zip(models, ['S0', 'S1', 'RVT']))
    run_lookup = {(r['model'], r['sequence'], r['method']): r for r in continuity if r['regime'] == 'retrospective_sequence_budget'}
    for model in models:
        for seq in cfg['sequences']:
            a, b = [run_lookup[model, seq, m] for m in ['ema/absolute', 'current_activity']]
            runrows.append([names[model], seq[12:14]] + [f'{a[k]}/{b[k]}' for k in ['lost_instances', 'longest_queries', 'tracks_with_losses', 'right_censored']])
    tables = {}
    tables['population'] = make_table(r'Annotation-defined subgroup populations pooled over the same four exposed recordings. GT and D columns count identity/query instances; tracks sum distinct identities within recordings, not independent driving trials. D is the original fixed dense-reference coverage for each carrier, independent of the new subgroup threshold analyses. Track memberships can overlap across strata as an object changes size or motion. Early/established exclude identities present at the first native timestamp. Small area is at most 1\% of the native image. Empty cells denote a genuinely empty population, not filtered failures.',
          'tab:driving_population', ['Stratum', 'GT', 'Tracks', r'$D_{S0}$', r'$D_{S1}$', r'$D_{RVT}$'],
          population, r'>{\raggedright\arraybackslash}p{.29\linewidth}'+r'*{5}{>{\raggedleft\arraybackslash}X}')
    tables['contrasts'] = make_table(r'Absolute EMA minus current activity: complete-GT recall differences at the already replayed retrospective 1 FP/query whole-recording points, 20\% site capacity. Each cell is the median percentage-point difference followed by positive/negative/tied recording counts. The whole-recording FP matching criterion holds in 4/4 groups; it does not establish subgroup FP matching. No subgroup threshold is fitted. All sixteen declared strata are shown; an undefined stratum contributes no sign. These are descriptive associations, not causal effects of object size, annotation age or image motion.',
          'tab:driving_contrasts', ['Stratum', r'S0 $\Delta R$ [+/--/0]', r'S1 $\Delta R$ [+/--/0]', r'RVT $\Delta R$ [+/--/0]'],
          contrast, r'>{\raggedright\arraybackslash}p{.29\linewidth}'+r'*{3}{>{\raggedleft\arraybackslash}X}')
    tables['continuity'] = make_table(r'Sampled evidence interruptions at the same whole-recording retrospective 1 FP/query points. A/B is absolute EMA/current activity. Lost counts fixed-reference dense identity/query instances not preserved; longest is the number of adjacent observed queries in a loss run. Tracks is the number of dense-reference tracks with losses; RC is the number of right-censored runs. Runs stop at a dense miss, GT absence, native-grid gap or sequence end, with observed recovery separated. These are conditional sampled interruptions, not physical blind intervals or an absolute detection deadline. Complete all-method/both-regime records and observed timestamp spans are archived.',
          'tab:driving_continuity', ['Carrier', 'Group', 'Lost A/B', 'Longest A/B', 'Tracks A/B', 'RC A/B'],
          runrows, 'll'+r'*{4}{>{\raggedleft\arraybackslash}X}')
    return tables, selected, pairs

def main():
    tables, selected, pairs = materials()
    previous = None
    if (OUT / 'report_receipt.json').exists():
        if '--refresh-layout' not in sys.argv or not (OUT / 'before_figure_layout.zip').exists():
            raise FileExistsError('Report already materialized')
        previous = read(OUT / 'report_receipt.json')
        for path, digest in previous['files'].items():
            if not path.endswith(('.png', '.svg')) and sha(ROOT / path) != digest:
                raise ValueError('A numerical/report artifact changed during visual refinement')
    for name, table in tables.items():
        (OUT / (name + '.texfrag')).write_text(table, encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['small', 'large', 'early', 'established', 'motion_low', 'motion_high']
    xlabels = ['Small', 'Larger', 'Early\nannotation', 'Established', 'Low\nimage motion', 'High\nimage motion']
    models = ['spike_original', 'spike_adapted', 'rvt_stream']
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.35), sharey=True)
    seqs = read(ROOT / 'code/configs/driving_factors_v01.json')['sequences']
    for ax, model, title in zip(axes, models, ['S0 / original SpikeFPN', 'S1 / adapted SpikeFPN', 'RVT / recurrent carrier']):
        for i, group in enumerate(groups):
            rr = [p for p in pairs if p['model'] == model and p['regime'] == 'retrospective_sequence_budget' and p['target'] == 1 and p['method_a'] == 'ema/absolute' and p['method_b'] == 'current_activity' and p['group'] == group]
            by_seq = {r['sequence']: r for r in rr}
            for j, seq in enumerate(seqs):
                ax.scatter(i + (j - 1.5) * .12, by_seq[seq]['delta_recall_pp'], s=22,
                           marker=['o', 's', '^', 'D'][j], color=['#2166ac', '#b2182b', '#4d9221', '#8c510a'][j],
                           label=seq[12:14] if i == 0 else None, zorder=3)
            ax.plot([i - .28, i + .28], [selected[model, group]['median_delta_recall_pp']] * 2, color='black', linewidth=1.7)
        ax.axhline(0, color='.45', linewidth=.7)
        ax.set_xticks(range(len(groups)), xlabels, rotation=65, ha='right')
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Recall difference (percentage points)\nAbsolute EMA minus activity')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Recording', ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.015), fontsize=7, title_fontsize=7, frameon=False)
    fig.subplots_adjust(left=.085, right=.99, bottom=.32, top=.77, wspace=.14)
    fig.savefig(OUT / 'driving_strata.png', dpi=260)
    fig.savefig(OUT / 'driving_strata.svg')
    plt.close(fig)
    populations = read(OUT / 'populations.json')
    early = sum(r['populations']['early']['instances'] for r in populations)
    report = ['# 驾驶因素分层：实测结果与边界', '',
              '本轮在已暴露的Zurich07—10结果上固定标注分组规则后连接策略输出。不是新的独立确认。未训练模型、未产生新阈值、未改变任何检测框或匹配。', '',
              '四组共6322查询、22841个GT身份—查询实例、320条组内独立身份轨迹；轨迹不是320个独立驾驶回合。三载体、九方法、四档全序列FP目标、两种阈值制度完整保留，形成13824个子组条件和4608个配对。', '',
              '## 分组先于子组结果连接', '',
              '配置为code/configs/driving_factors_v01.json。小目标为裁剪后面积≤图像1%；标注初期是身份在完整原生网格中的前三次标注，原生首帧已存在的身份左截断单列。图像位移以相邻原生标注框中心位移/秒/800像素对角线定义：低≤0.01，高>0.05，其余居中。缺少连续相邻标注记unknown，不跨缺口插值。它们不是距离、物理出现、自车运动或风险真值。', '',
              '冻结收据记录此前整体结果已知。所有子组沿用原全序列实际重放阈值，未按子组校准；全序列FP匹配不能称为各子组同误报。', '',
              '## 1FP主点的分层观察', '']
    for model in models:
        report.append('**' + model + '**：')
        for group in groups:
            r = selected[model, group]
            report.append(f"- {LABELS[group]}：完整GT召回配对差中位数{r['median_delta_recall_pp']:+.2f}pp；四组方向{r['positive']}正/{r['negative']}负/{r['tied']}平；GT={r['gt_instances']}，原dense证据={r['dense_instances']}，dense轨迹={r['dense_tracks']}。")
        report.append('')
    report.extend(['## 不能省略的解释限制', '',
        f'初期子组共有{early}个身份—查询实例，原始Spike/适配Spike/RVT的原dense证据分别只有14/13/57，即1.62%/1.51%/6.61%。不能用此小分母宣称新目标问题已经解决。', '',
        'Spike的较大目标收益明显高于小目标，但原dense对小目标的覆盖也明显更弱。此差异不自动说明门控歧视小目标；需要同时看全GT召回、条件Q与基线覆盖。运动、大小、标注年龄和类别相互关联，本轮没有做足以识别因果作用的独立干预。', '',
        'RVT低图像运动子组四组均偏向activity，召回差中位−9.46pp；高运动子组2正2负，中位+0.04pp。低运动子组的源覆盖差分别−201/−27/−15/−46，替代恢复差+111/+5/+1/−3，最终参考证据差−90/−22/−14/−49；均无不可恢复的有效源损失差。这个局部结构与整体Zurich09“源覆盖更高但替代恢复更差”的结构不同，不能用单一总量解释所有子组。', '',
        '## 连续性', '',
        '在相同主点，两个Spike检查点各四组的残差最长采样损失段均短于activity；RVT四组相反。RVT07为31步对3步，观察到的首末丢失跨度约1.50秒对0.10秒。它们以dense可检连续样本为条件，不是目标真实不可见时间，也不是200ms截止时间。全部216方法/制度连续性条件、29608个损失段及删失原因均保留。', '',
        '## 复核与下一步', '',
        '独立脚本重新从原始标注构建大小、年龄与位移，独立连接主点两制度全部九方法的341388个查询记录、3456个子组条件；核验80208个源证据实例及29608个连续损失段。32份旧首版数值收据不变。此为软件/算术复核，不增加独立驾驶记录。', '',
        '下一阶段的机制预测、混淆控制与数据准入门槛另见code/configs/driving_confirmation_v02.json及docs/36。该协议不能称为已执行的独立验证；现有四组不得再作为未见数据。'])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    files = [OUT / p for p in ['conditions.json', 'pairs.json', 'group_summaries.json', 'populations.json', 'continuity_summary.json', 'source_partitions.json', 'verification.json', 'REPORT.md', 'driving_strata.png', 'driving_strata.svg', 'population.texfrag', 'contrasts.texfrag', 'continuity.texfrag']]
    if previous:
        for path, digest in previous['files'].items():
            if not path.endswith(('.png', '.svg')) and sha(ROOT / path) != digest:
                raise AssertionError('Visual refinement altered numerical/report artifacts')
    save(OUT / 'report_receipt.json', dict(files={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
         reporter_sha256=sha(ROOT / 'code/scripts/report_driving_factors_v01.py'), role='Reporting checked retrospective results; no new scientific sample',
         visual_refinement_archive='before_figure_layout.zip' if previous else None))
    save(OUT / 'status.json', dict(state='complete', independent_verification='passed', reporting='complete'))
    print('Verified report, three measured tables and one scientific figure written', flush=True)

if __name__ == '__main__':
    main()
