"""Predeclared primary new-recording estimands, with undefined cells retained."""
from collections import defaultdict
import itertools, json
import numpy as np
from scipy.stats import t as student_t
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha
from riskaware.driving_factors import annotation_factors
from riskaware.validation_estimands import motion_interaction, weighted_cell_interaction, recording_summary
from riskaware.evidence_runs import evidence_runs

def cell_key(row):
    return row['size'] + '|' + ('car' if row['cls'] == 0 else 'pedestrian') + '|' + row['stage']

def annotation_cohort(sequence, metadata, cfg):
    folder = ROOT / 'datasets/dsec_confirmation/test' / sequence
    grid = np.loadtxt(folder / 'image_timestamps.txt', dtype=np.int64)
    labels = np.load(folder / 'object_detections/left/tracks.npy', allow_pickle=False)
    factors_cfg = read(ROOT / cfg['factors_config'])
    factors, gap = annotation_factors(grid, labels, factors_cfg)
    by_t = defaultdict(list)
    for (t, identity), row in factors.items():
        by_t[t].append(row)
    cells = ['|'.join(values) for values in itertools.product(['small', 'large'], ['car', 'pedestrian'], ['early', 'established', 'left_censored'])]
    groups = factors_cfg['groups'] + ['cell/' + cell + '/' + motion for cell in cells for motion in ['motion_low', 'motion_high']]
    frames, lookup, population = [], {}, {g: dict(gt=0, tracks=set()) for g in groups}
    for q, record in enumerate(metadata['frames']):
        rows = by_t[record['time_us']]
        identities = {g: set() for g in groups}
        for row in rows:
            tags = list(row['groups'])
            if row['motion'] in ['motion_low', 'motion_high']:
                tags.append('cell/' + cell_key(row) + '/' + row['motion'])
            lookup[q, row['identity']] = dict(**row, analysis_groups=tags)
            for group in tags:
                identities[group].add(row['identity'])
                population[group]['gt'] += 1
                population[group]['tracks'].add(row['identity'])
        frames.append(dict(q=q, time_us=record['time_us'], grid_index=int(np.searchsorted(grid, record['time_us'])),
                           gt_ids=sorted(row['identity'] for row in rows), groups=identities))
    return frames, lookup, groups, cells, gap, population

def aggregate_one(model, seq, cfg):
    meta = read(OUT / 'carriers' / (model + '__' + seq + '.json'))
    points = read(OUT / 'points' / (model + '__' + seq + '.json'))['conditions']
    frames, lookup, groups, cells, gap, population = annotation_cohort(seq, meta, cfg)
    totals = {key: {group: defaultdict(int) for group in groups} for key in points}
    trajectories = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    histories, endpoint_totals = defaultdict(list), defaultdict(lambda: defaultdict(int))
    with (OUT / 'points' / model / (seq + '.jsonl')).open() as f:
        for line in f:
            r = json.loads(line)
            p = points[r['key']]
            if r['attempt'] != p['attempt']:
                continue
            frame = frames[r['q']]
            gt, dense, detected = set(frame['gt_ids']), set(r['dense_ids']), set(r['matched_ids'])
            assert dense <= gt and detected <= gt and len(detected) == r['tp']
            for k, value in [('queries', 1), ('tp', len(detected)), ('fp', r['fp']), ('gt', len(gt)),
                             ('dense_tp', len(dense)), ('kept', len(dense & detected)), ('gained', len(detected - dense))]:
                endpoint_totals[r['key']][k] += value
            for group, members in frame['groups'].items():
                ref, hit = dense & members, detected & members
                for k, value in [('gt', len(members)), ('tp', len(hit)), ('dense_tp', len(ref)),
                                 ('kept', len(ref & hit)), ('lost', len(ref - hit)), ('gained', len(hit - ref))]:
                    totals[r['key']][group][k] += value
                for identity in ref:
                    tr = trajectories[r['key'], group][identity]
                    tr[0] += 1
                    tr[1] += identity in hit
            if p['target'] == 1.0:
                histories[r['key']].append(dict(q=frame['q'], time_us=frame['time_us'], grid_index=frame['grid_index'],
                                              gt_ids=sorted(gt), dense_ids=sorted(dense), matched_ids=sorted(detected)))
    conditions = []
    for key, p in points.items():
        assert all(endpoint_totals[key][k] == p[k] for k in ['queries', 'gt', 'tp', 'fp', 'dense_tp', 'kept', 'gained'])
        for group, v in totals[key].items():
            v = dict(v)
            tr = trajectories[key, group]
            assert v['gt'] == population[group]['gt']
            conditions.append(dict(model=model, sequence=seq, key=key, method=p['method'], regime=p['regime'], target=p['target'],
                group=group, **v, gt_tracks=len(population[group]['tracks']), dense_tracks=len(tr),
                recall=v['tp'] / v['gt'] if v['gt'] else None,
                preservation=v['kept'] / v['dense_tp'] if v['dense_tp'] else None,
                track_equal_preservation=float(np.mean([b / a for a, b in tr.values()])) if tr else None))
    continuity = []
    runpath = OUT / 'continuity' / (model + '__' + seq + '.jsonl')
    runpath.parent.mkdir(parents=True, exist_ok=True)
    with runpath.open('w') as f:
        for key, history in histories.items():
            history.sort(key=lambda r: r['q'])
            runs, stats = evidence_runs(history, gap, 1.5)
            assert stats['lost_instances'] == totals[key]['all']['lost']
            p = points[key]
            continuity.append(dict(model=model, sequence=seq, key=key, method=p['method'], regime=p['regime'], **stats))
            for r in runs:
                f.write(json.dumps(dict(key=key, **r), separators=(',', ':')) + '\n')
    partitions = defaultdict(lambda: defaultdict(int))
    with (OUT / 'source_survival' / (model + '__' + seq + '.jsonl')).open() as f:
        for line in f:
            r = json.loads(line)
            p = points[r['key']]
            if r['attempt'] != p['attempt']:
                continue
            effective = r['mask_selected'] and r['above_threshold']
            for group in lookup[r['q'], r['identity']]['analysis_groups']:
                v = partitions[r['key'], group]
                v['dense_instances'] += 1
                v['coverage'] += effective
                v['alternative_recovery'] += r['identity_detected'] and not effective
                v['selected_source_loss'] += effective and not r['identity_detected']
                v['kept'] += r['identity_detected']
                v[r['category']] += 1
    source = []
    for (key, group), v in partitions.items():
        assert v['kept'] == v['coverage'] + v['alternative_recovery'] - v['selected_source_loss']
        p = points[key]
        source.append(dict(model=model, sequence=seq, key=key, group=group, method=p['method'], regime=p['regime'], **v))
    return conditions, continuity, source, cells

def main():
    cfg, plan = read(CONFIG), read(OUT / 'statistics_plan.json')
    if (OUT / 'primary_results.json').exists():
        raise FileExistsError('Primary results already complete')
    model = 'rvt_stream'
    rows, continuity, partitions, point_lookup = [], [], [], {}
    for seq in cfg['sequences']:
        cond, runs, sources, cells = aggregate_one(model, seq, cfg)
        rows.extend(cond)
        continuity.extend(runs)
        partitions.extend(sources)
        point_lookup[seq] = read(OUT / 'points' / (model + '__' + seq + '.json'))['conditions']
        print('New-record factors joined', seq, flush=True)
    lookup = {(r['sequence'], r['regime'], r['target'], r['method'], r['group']): r for r in rows}
    effects, cross_cells, pooled_cells = [], [], {key: defaultdict(int) for key in cells}
    for seq in cfg['sequences']:
        regime, target = 'retrospective_sequence_budget', 1.
        def group(method, name):
            return lookup[seq, regime, target, method, name]
        la, lb, ha, hb = [group(method, name) for method, name in [('ema/absolute', 'motion_low'), ('current_activity', 'motion_low'),
                                                                  ('ema/absolute', 'motion_high'), ('current_activity', 'motion_high')]]
        interaction = motion_interaction(la['tp'], lb['tp'], la['gt'], ha['tp'], hb['tp'], ha['gt'])
        a, b = [point_lookup[seq][regime + '|' + method + '@1.0'] for method in ['ema/absolute', 'current_activity']]
        tolerance = max(1, a['allowed_fp'] // 100)
        matched = 0 <= a['allowed_fp'] - a['fp'] <= tolerance and 0 <= b['allowed_fp'] - b['fp'] <= tolerance and abs(a['fp'] - b['fp']) <= tolerance
        current_cells = {}
        for cell in cells:
            lo_a, lo_b, hi_a, hi_b = [group(method, 'cell/' + cell + '/' + motion) for method, motion in
                                     [('ema/absolute', 'motion_low'), ('current_activity', 'motion_low'),
                                      ('ema/absolute', 'motion_high'), ('current_activity', 'motion_high')]]
            values = dict(low_a=lo_a['tp'], low_b=lo_b['tp'], low_gt=lo_a['gt'], high_a=hi_a['tp'], high_b=hi_b['tp'], high_gt=hi_a['gt'])
            current_cells[cell] = values
            for k, value in values.items():
                pooled_cells[cell][k] += value
            cross_cells.append(dict(sequence=seq, cell=cell, **values,
                                    interaction_pp=motion_interaction(**values), development_weight=plan['common_support_weights'].get(cell, 0.)))
        adjusted = weighted_cell_interaction(current_cells, plan['common_support_weights'])
        effects.append(dict(sequence=seq, city=seq.split('_')[0], interaction_pp=interaction, adjusted_interaction_pp=adjusted,
                            matched=matched, low_gt=la['gt'], high_gt=ha['gt'], low_tp_a=la['tp'], low_tp_b=lb['tp'],
                            high_tp_a=ha['tp'], high_tp_b=hb['tp'], fp_a=a['fp'], fp_b=b['fp'], queries=a['queries'],
                            low_difference_pp=100 * (la['tp'] - lb['tp']) / la['gt'] if la['gt'] else None,
                            high_difference_pp=100 * (ha['tp'] - hb['tp']) / ha['gt'] if ha['gt'] else None))
    summary = recording_summary([r['interaction_pp'] for r in effects])
    defined = [r['interaction_pp'] for r in effects if r['interaction_pp'] is not None]
    ci = None
    if len(defined) >= 2:
        halfwidth = float(student_t.ppf(.975, len(defined) - 1) * np.std(defined, ddof=1) / np.sqrt(len(defined)))
        ci = [summary['mean'] - halfwidth, summary['mean'] + halfwidth]
    all_matched = all(r['matched'] for r in effects)
    verdict = ('inconclusive' if not summary['all_planned_defined'] or not all_matched or ci is None else
               'supports_declared_negative_interaction' if ci[1] < 0 else
               'contrary_positive_interaction' if ci[0] > 0 else 'inconclusive')
    cities = sorted({r['city'] for r in effects})
    city_means = {city: float(np.mean([r['interaction_pp'] for r in effects if r['city'] == city and r['interaction_pp'] is not None])) for city in cities}
    leave_city = {city: float(np.mean([r['interaction_pp'] for r in effects if r['city'] != city and r['interaction_pp'] is not None])) for city in cities}
    weights = plan['common_support_weights']
    pooled_adjusted = weighted_cell_interaction(pooled_cells, weights)
    by_seq_cell = {(r['sequence'], r['cell']): r for r in cross_cells}
    rng, boot, undefined = np.random.default_rng(20261031), [], 0
    for indices in rng.integers(0, len(effects), size=(5000, len(effects))):
        values = {key: defaultdict(int) for key in cells}
        for i in indices:
            seq = effects[int(i)]['sequence']
            for cell in cells:
                for key in ['low_a', 'low_b', 'low_gt', 'high_a', 'high_b', 'high_gt']:
                    values[cell][key] += by_seq_cell[seq, cell][key]
        value = weighted_cell_interaction(values, weights)
        if value is None:
            undefined += 1
        else:
            boot.append(value)
    adjusted_ci = [float(v) for v in np.quantile(boot, [.025, .975])] if undefined == 0 else None
    save(OUT / 'conditions.json', rows)
    save(OUT / 'continuity_summary.json', continuity)
    save(OUT / 'source_partitions.json', partitions)
    save(OUT / 'cross_cells.json', cross_cells)
    save(OUT / 'primary_results.json', dict(effects=effects, summary=summary, nominal_recording_t_CI95=ci,
         all_seven_whole_recording_FP_matched=all_matched, verdict=verdict,
         city_means=city_means, equal_city_mean=float(np.mean(list(city_means.values()))), leave_one_city_out=leave_city,
         adjustment=dict(weights=weights, pooled_point_pp=pooled_adjusted, per_recording_defined=sum(r['adjusted_interaction_pp'] is not None for r in effects),
                         bootstrap_draws=5000, undefined_bootstrap_draws=undefined, CI95_if_all_draws_defined=adjusted_ci,
                         verdict='inconclusive' if pooled_adjusted is None or adjusted_ci is None else 'descriptive_adjusted_estimate',
                         scope='Restricted development-common-support population; observational, not causal.'),
         interpretation='Nominal t interval assumes independent/exchangeable recording groups; shared cities/sessions andN7limit population inference. New records within DSEC, no new sensor or safety claim.',
         secondary_Spike_status='Not executed in this primary-stage run; no new-recording SNN size-confirmation claim.',
         source_sha256=sha(ROOT / 'code/scripts/summarize_driving_confirmation.py')))
    print('Primary new-recording estimate', summary, 'CI', ci, 'verdict', verdict, flush=True)

if __name__ == '__main__':
    main()
