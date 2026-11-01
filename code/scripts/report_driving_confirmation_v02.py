"""Complete primary-stage reporting, including contrary/undefined outcomes."""
from pathlib import Path
import csv, json
import numpy as np
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha
from submission_audit_tables import make_table

def short(seq):
    return {'thun_01_a': 'Thun01', 'thun_02_a': 'Thun02', 'interlaken_01_a': 'Inter01',
            'zurich_city_12_a': 'Z12', 'zurich_city_13_a': 'Z13', 'zurich_city_14_a': 'Z14', 'zurich_city_15_a': 'Z15'}[seq]

def number(value, signed=False):
    return '--' if value is None else f'{value:+.2f}' if signed else f'{value:.2f}'

def materials():
    cfg, result = read(CONFIG), read(OUT / 'primary_results.json')
    if not read(OUT / 'verification.json')['passed']:
        raise ValueError('Independent primary-stage reconstruction must pass first')
    rows = read(OUT / 'conditions.json')
    lookup = {(r['sequence'], r['regime'], r['target'], r['method'], r['group']): r for r in rows}
    points = {seq: read(OUT / 'points' / ('rvt_stream__' + seq + '.json'))['conditions'] for seq in cfg['sequences']}
    primary = [[short(r['sequence']), r['low_gt'], number(r['low_difference_pp'], True),
                r['high_gt'], number(r['high_difference_pp'], True), number(r['interaction_pp'], True),
                f"{r['fp_a']}/{r['fp_b']}", 'yes' if r['matched'] else 'no'] for r in result['effects']]
    tables = {'primary': make_table(r'Prospectively specified RVT primary contrast on seven named new DSEC recording groups. At 20\% site capacity and the actually replayed retrospective whole-recording 1 FP/query point, $\Delta R$ is absolute EMA minus current activity complete-GT recall (percentage points); $I=\Delta R_{low}-\Delta R_{high}$. Low/high are annotation-center image-motion strata, not ego-motion or risk labels. GT counts identity/query instances; FP A/B are complete-recording counts. Matching applies to the whole recording, with the same declared integer tolerance; no subgroup threshold is fitted. All seven planned groups are retained.',
         'tab:new_primary', ['Record', r'GT$_L$', r'$\Delta R_L$', r'GT$_H$', r'$\Delta R_H$', r'$I$', 'FP A/B', 'Matched'],
         primary, 'l'+r'*{7}{>{\raggedleft\arraybackslash}X}')}
    budget = []
    budget_records = []
    for target in cfg['fp_targets']:
        interactions, low, high, matched = [], [], [], 0
        for seq in cfg['sequences']:
            regime = 'retrospective_sequence_budget'
            a, b = [points[seq][regime + '|' + method + '@' + str(target)] for method in ['ema/absolute', 'current_activity']]
            tolerance = max(1, a['allowed_fp'] // 100)
            pair_matched = (0 <= a['allowed_fp'] - a['fp'] <= tolerance and 0 <= b['allowed_fp'] - b['fp'] <= tolerance and abs(a['fp'] - b['fp']) <= tolerance)
            matched += pair_matched
            diffs = []
            for group in ['motion_low', 'motion_high']:
                aa, bb = [lookup[seq, regime, target, method, group] for method in ['ema/absolute', 'current_activity']]
                diffs.append(100 * (aa['tp'] - bb['tp']) / aa['gt'] if aa['gt'] else None)
            value = diffs[0] - diffs[1] if all(v is not None for v in diffs) else None
            budget_records.append(dict(sequence=seq, target=target, low_difference_pp=diffs[0], high_difference_pp=diffs[1],
                                       interaction_pp=value, matched=pair_matched, fp_a=a['fp'], fp_b=b['fp']))
            if value is not None:
                low.append(diffs[0]); high.append(diffs[1]); interactions.append(value)
        budget.append([f'{target:g}', f'{matched}/7', f'{len(interactions)}/7',
                       number(float(np.mean(interactions)) if interactions else None, True),
                       f'{sum(v < 0 for v in interactions)}/{sum(v > 0 for v in interactions)}/{sum(v == 0 for v in interactions)}'])
    tables['budget'] = make_table(r'All four predeclared retrospective whole-recording FP targets in the new RVT cohort. The mean interaction equally weights recordings with defined low/high GT support; missing groups remain counted as undefined. Signs are negative/positive/tied interactions. Only the 1 FP/query target has the frozen primary interval/verdict; the others are descriptive sensitivity analyses. A defined point is not automatically FP-matched, so target attainment is reported separately.',
         'tab:new_budget', ['FP/query', 'FP matched', 'Defined', r'Mean $I$ (pp)', 'Negative/positive/tied'],
         budget, 'l'+r'*{4}{>{\raggedleft\arraybackslash}X}')
    cross = read(OUT / 'cross_cells.json')
    weights = result['adjustment']['weights']
    adjustment = []
    all_cells = list(dict.fromkeys(r['cell'] for r in cross))
    for cell in all_cells:
        pp = [r for r in cross if r['cell'] == cell]
        la, lb, lg, ha, hb, hg = [sum(r[k] for r in pp) for k in ['low_a', 'low_b', 'low_gt', 'high_a', 'high_b', 'high_gt']]
        value = 100 * (la - lb) / lg - 100 * (ha - hb) / hg if lg and hg else None
        label = cell.replace('left_censored', 'left').replace('established', 'est.').replace('pedestrian', 'ped.').replace('|', ', ')
        adjustment.append([label, f"{100 * weights[cell]:.2f}" if cell in weights else 'outside', lg, hg,
                           number(value, True), sum(r['interaction_pp'] is not None for r in pp)])
    tables['adjustment'] = make_table(r'Size/class/annotation-stage common support in the new cohort. Weights (percent) were fixed from development annotations before any new prediction; outside denotes zero-development-common-support cells, which remain tabulated and are not imputed. GT counts pool identity/query instances; cell $I$ is a pooled conditional contrast, not the primary equal-recording mean. Defined gives recording counts with both motion populations. Any absent positive-weight cell makes that recording standardization undefined; weights are never silently renormalized. Early means first three annotations, not physical onset.',
          'tab:new_adjustment', ['Cell', r'Weight(\%)', r'GT$_L$', r'GT$_H$', r'Pooled $I$', 'Defined/7'],
          [[*r[:-1], f'{r[-1]}/7'] for r in adjustment], r'>{\raggedright\arraybackslash}p{.34\linewidth}'+r'*{5}{>{\raggedleft\arraybackslash}X}')
    source = read(OUT / 'source_partitions.json')
    source_rows = []
    for seq in cfg['sequences']:
        pp = [r for r in source if r['sequence'] == seq and r['regime'] == 'retrospective_sequence_budget' and r['group'] == 'motion_low']
        aa = next((r for r in pp if r['method'] == 'ema/absolute'), {})
        bb = next((r for r in pp if r['method'] == 'current_activity'), {})
        values = [aa.get(k, 0) - bb.get(k, 0) for k in ['coverage', 'alternative_recovery', 'selected_source_loss', 'kept']]
        source_rows.append([short(seq), aa.get('dense_instances', bb.get('dense_instances', 0)), *[f'{v:+d}' for v in values]])
    tables['source'] = make_table(r'New-recording RVT low-image-motion source paths at the primary retrospective point. Differences are residual minus activity in integer identity/query counts. D is the fixed original dense-reference evidence denominator. Every row obeys $\Delta kept=\Delta C+\Delta A-\Delta L$, where C is effective original-source coverage, A alternative recovery and L unrecovered effective-source loss. This identity is accounting; the predeclared secondary prediction concerns the signs and compensation pattern, not whether the identity holds. Zero-D rows cannot support a retention-rate claim.',
         'tab:new_source', ['Record', 'D', r'$\Delta C$', r'$\Delta A$', r'$\Delta L$', r'$\Delta kept$'],
         source_rows, 'l'+r'*{5}{>{\raggedleft\arraybackslash}X}')
    return tables, result, budget_records, cross, rows, points

def main():
    if (OUT / 'report_receipt.json').exists():
        import sys, zipfile
        if sys.argv[1:] != ['--layout-refine'] or (OUT / 'artifact_manifest.json').exists():
            raise FileExistsError('Primary report already materialized; sealed results cannot be overwritten')
        previous = read(OUT / 'report_receipt.json')
        for path, digest in previous['files'].items():
            assert sha(ROOT / path) == digest, path
        with zipfile.ZipFile(ROOT / 'provenance/before_v18_figure_layout.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in list(previous['files']) + [(OUT / 'report_receipt.json').relative_to(ROOT).as_posix()]:
                archive.write(ROOT / path, path)
    tables, result, budget, cross, rows, points = materials()
    cfg = read(CONFIG)
    for name, value in tables.items():
        (OUT / (name + '.texfrag')).write_text(value, encoding='utf-8')
    save(OUT / 'budget_sensitivity.json', budget)
    operating = [dict(model='rvt_stream', sequence=seq, **p) for seq, conditions in points.items() for p in conditions.values()]
    with (OUT / 'operating_points.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(operating[0])); writer.writeheader(); writer.writerows(operating)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.8), gridspec_kw={'width_ratios': [1., 1.25]})
    ax = axes[0]
    for i, r in enumerate(result['effects']):
        values = [r['low_difference_pp'], r['high_difference_pp']]
        if all(v is not None for v in values):
            ax.plot(values, [i, i], color='.65', linewidth=.8)
        for j, value in enumerate(values):
            if value is not None:
                ax.scatter(value, i, color=['#2166ac', '#b2182b'][j], marker=['o', 's'][j], s=27,
                           label=['Low image motion', 'High image motion'][j] if i == 0 else None, zorder=3)
        if any(value is None for value in values):
            missing = '/'.join(name for name, value in zip(['L', 'H'], values) if value is None)
            ax.text(.98, i, missing + ' undefined', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6)
    ax.axvline(0, color='.4', linewidth=.7)
    ax.set_yticks(range(7), [short(r['sequence']) for r in result['effects']])
    ax.invert_yaxis(); ax.set_xlabel('Recall difference (pp)\nAbsolute EMA minus activity')
    ax.set_title('New recording contrasts at 1 FP/query', fontsize=8)
    ax.grid(axis='x', alpha=.18)
    handles = [Line2D([], [], color=color, marker=marker, linestyle='None', label=label)
               for color, marker, label in [('#2166ac', 'o', 'Low image motion'), ('#b2182b', 's', 'High image motion')]]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.24), frameon=False, fontsize=6.5)
    keys = list(result['adjustment']['weights'])
    support = np.array([[min(r['low_gt'], r['high_gt']) for seq in cfg['sequences']
                         for r in cross if r['sequence'] == seq and r['cell'] == key] for key in keys])
    plotted = np.ma.masked_where(support == 0, support)
    cmap = plt.colormaps['Blues'].copy(); cmap.set_bad('#eeeeee')
    im = axes[1].imshow(plotted, aspect='auto', cmap=cmap, norm=LogNorm(vmin=1, vmax=max(2, int(support.max()))))
    for i, j in zip(*np.where(support == 0)):
        axes[1].text(j, i, 'x', ha='center', va='center', fontsize=7, color='.3')
    labels = [k.replace('left_censored', 'left').replace('established', 'est.').replace('pedestrian', 'ped.').replace('|', '/') for k in keys]
    axes[1].set_yticks(range(len(keys)), labels, fontsize=6.5)
    axes[1].set_xticks(range(7), [short(s) for s in cfg['sequences']], rotation=60, ha='right', fontsize=6.5)
    axes[1].set_title('Fixed-weight cell support: min(low GT, high GT)', fontsize=7.5)
    fig.colorbar(im, ax=axes[1], fraction=.045, pad=.03).ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=.09, right=.98, bottom=.28, top=.87, wspace=1.0)
    fig.savefig(OUT / 'new_recording_validation.png', dpi=280, bbox_inches='tight', pad_inches=.12)
    fig.savefig(OUT / 'new_recording_validation.svg', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    summary, adjustment = result['summary'], result['adjustment']
    report = ['# 七个新录制组：RVT主验证完整结果', '',
              '本轮在新预测前固定七个记录组、统计量、八个开发共同支持权重和源码。原始数据来自官方公开DSEC检测test；不是官方挑战提交、新数据集或独立传感器验证。当前只执行RVT主阶段，Spike新组次级实验未执行。', '',
              '## 主判定', '',
              f"预定判定：`{result['verdict']}`。七组中主效应有定义{summary['defined']}/7，全序列FP全部匹配={result['all_seven_whole_recording_FP_matched']}。",
              f"记录组等权均值I={number(summary['mean'], True)}pp，中位数{number(summary['median'], True)}pp；名义95%t区间{result['nominal_recording_t_CI95']}。",
              '这个区间依赖记录组独立/可交换假设，七组共享三个城市。它不是逐帧置信区间、安全概率或宽泛人群结论。', '']
    if summary['defined'] != 7:
        report.extend(['有定义组不足七个时，该数值区间只描述有定义子集；完整主判定仍为不确定，不把缺失组填零。', ''])
    for r in result['effects']:
        report.append(f"- {r['sequence']}：低/高GT={r['low_gt']}/{r['high_gt']}，低/高召回差={number(r['low_difference_pp'], True)}/{number(r['high_difference_pp'], True)}pp，I={number(r['interaction_pp'], True)}pp；FP={r['fp_a']}/{r['fp_b']}，匹配={r['matched']}；标准化I={number(r['adjusted_interaction_pp'], True)}pp。")
    report.extend(['', '## 混淆调整与不能越过的边界', '',
        f"开发固定权重下，七组中可计算完整记录级标准化效应的数量为{adjustment['per_recording_defined']}/7；合并标准化点估计={number(adjustment['pooled_point_pp'], True)}pp。",
        f"5000次记录组bootstrap中，缺共同支持的抽样={adjustment['undefined_bootstrap_draws']}；按冻结规则，调整后区间为{adjustment['CI95_if_all_draws_defined']}，判定`{adjustment['verdict']}`。",
        '缺少共同支持不是零效应，也不能通过删掉单元或按新结果重新归一化权重解决。即使原始I方向得到支持，不能据此认定运动具有独立因果作用。', '',
        '## 稳健性与完整数据', '',
        f"等城市描述均值={number(result['equal_city_mean'], True)}pp；各城市均值={result['city_means']}；留一城市={result['leave_one_city_out']}。",
        '四档FP目标、两套阈值制度和全部九条件保留在operating_points.csv；目标未达到、置信度下界受限和实际FP均保留。子组内不调阈值，不把全序列匹配称为子组匹配。source_partitions.json保留全部源路径，continuity_summary.json及逐段记录保留采样连续性与删失。', '',
        '## 复现边界', '',
        '三个载体在旧Thun00各225查询的原始预测/九掩码精确回归；RVT36校准FP点精确回归。新模型仍完整计算，掩码只影响候选准入。原始公开事件成员为节省本机磁盘，在完整缓存/实际点生成后释放；URL、ETag、成员CRC/SHA及释放记录完整保留，重做完整前向需重新取得同一成员。旧数据/检查点未删除。', '',
        '主验证的软件复核见verification.json，不能增加独立记录组数。新的SNN大小交互尚未执行，真实自车运动、碰撞风险、物理跳算或节能均未得到本轮验证。'])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    artifacts = [OUT / name for name in ['primary_results.json', 'conditions.json', 'cross_cells.json',
        'continuity_summary.json', 'source_partitions.json', 'verification.json', 'budget_sensitivity.json',
        'operating_points.csv', 'REPORT.md', 'primary.texfrag', 'budget.texfrag', 'adjustment.texfrag',
        'source.texfrag', 'new_recording_validation.png', 'new_recording_validation.svg']]
    save(OUT / 'report_receipt.json', dict(files={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
         source_sha256=sha(ROOT / 'code/scripts/report_driving_confirmation_v02.py'),
         scope='All planned primary-stage outcomes, including contrary and undefined; secondarySNNnotexecuted'))
    if 'previous' in locals():
        for path, digest in previous['files'].items():
            if not path.endswith(('.png', '.svg')):
                assert sha(ROOT / path) == digest, 'Layout-only revision changed numeric/report material: ' + path
    print('Complete new-recording primary report written', flush=True)

if __name__ == '__main__':
    main()
