"""All predeclared secondary size contrasts; no relabeling as a new primary."""
from collections import defaultdict
import numpy as np
from scipy.stats import t as student_t
from snn_secondary_common import ROOT, OUT, CONFIG, PARENT, read, save, sha, activate

def nominal_summary(effects, field='interaction_pp'):
    from riskaware.validation_estimands import recording_summary
    summary = recording_summary([r[field] for r in effects])
    values = [r[field] for r in effects if r[field] is not None]
    ci = None
    if len(values) >= 2:
        half = float(student_t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
        ci = [summary['mean'] - half, summary['mean'] + half]
    return dict(summary=summary, nominal_recording_t_CI95=ci, all_matched=all(r['matched'] for r in effects),
                scope='Secondary descriptive interval, no multiplicity-controlled claim; same seven recording units for both checkpoints')

def aggregate(model, sequence):
    activate()
    from summarize_driving_confirmation import aggregate_one
    cfg = read(CONFIG)
    dest = OUT / 'aggregates' / (model + '__' + sequence + '.json')
    if dest.exists():
        return read(dest)
    rows, runs, sources, cells = aggregate_one(model, sequence, cfg)
    result = dict(conditions=rows, continuity=runs, source_partitions=sources, cells=cells)
    save(dest, result)
    print('SNN annotation groups joined', model, sequence, flush=True)
    return result

def main():
    cfg = read(CONFIG)
    assert not (OUT / 'secondary_results.json').exists()
    conditions, runs, sources, results, all_points, interactions = [], [], [], {}, [], []
    for model in cfg['models']:
        effects, budgets, transfer = [], [], []
        for seq in cfg['sequences']:
            data = aggregate(model, seq)
            conditions.extend(data['conditions']); runs.extend(data['continuity']); sources.extend(data['source_partitions'])
            lookup = {(r['regime'], r['target'], r['method'], r['group']): r for r in data['conditions']}
            points = read(OUT / 'points' / (model + '__' + seq + '.json'))['conditions']
            all_points.extend(dict(model=model, sequence=seq, **p) for p in points.values())
            for regime in ['retrospective_sequence_budget', 'thun_calibrated']:
                for target in cfg['fp_targets']:
                    a, b = [points[regime + '|' + method + '@' + str(target)] for method in ['ema/absolute', 'current_activity']]
                    tol = max(1, a['allowed_fp'] // 100)
                    matched = (0 <= a['allowed_fp'] - a['fp'] <= tol and 0 <= b['allowed_fp'] - b['fp'] <= tol and abs(a['fp'] - b['fp']) <= tol)
                    small_a, small_b, large_a, large_b = [lookup[regime, target, method, group] for method, group in
                        [('ema/absolute', 'small'), ('current_activity', 'small'), ('ema/absolute', 'large'), ('current_activity', 'large')]]
                    ds = 100 * (small_a['tp'] - small_b['tp']) / small_a['gt'] if small_a['gt'] else None
                    dl = 100 * (large_a['tp'] - large_b['tp']) / large_a['gt'] if large_a['gt'] else None
                    js = ds - dl if ds is not None and dl is not None else None
                    entry = dict(model=model, sequence=seq, city=seq.split('_')[0], regime=regime, target=target,
                         interaction_pp=js, small_difference_pp=ds, large_difference_pp=dl, matched=matched,
                         small_gt=small_a['gt'], large_gt=large_a['gt'], small_tp_a=small_a['tp'], small_tp_b=small_b['tp'],
                         large_tp_a=large_a['tp'], large_tp_b=large_b['tp'], fp_a=a['fp'], fp_b=b['fp'], queries=a['queries'],
                         fixed_dense_small=small_a['dense_tp'], fixed_dense_large=large_a['dense_tp'],
                         small_kept_a=small_a['kept'], small_kept_b=small_b['kept'], large_kept_a=large_a['kept'], large_kept_b=large_b['kept'],
                         small_preservation_a=small_a['preservation'], small_preservation_b=small_b['preservation'],
                         large_preservation_a=large_a['preservation'], large_preservation_b=large_b['preservation'],
                         all_gt=a['gt'], all_tp_a=a['tp'], all_tp_b=b['tp'], all_dense=a['dense_tp'])
                    interactions.append(entry)
                    if regime == 'retrospective_sequence_budget':
                        budgets.append(entry)
                        if target == 1.:
                            effects.append(entry)
                    else:
                        transfer.append(entry)
        result = nominal_summary(effects)
        cities = sorted({r['city'] for r in effects})
        city = {c: [r['interaction_pp'] for r in effects if r['city'] == c and r['interaction_pp'] is not None] for c in cities}
        result.update(effects=effects, budgets={str(t): nominal_summary([r for r in budgets if r['target'] == t]) for t in cfg['fp_targets']},
                      transfer=transfer, city_means={c: float(np.mean(v)) if v else None for c, v in city.items()},
                      interpretation='Size association under fixed carrier/admission policy; no causal size mechanism or independent architecture replication')
        results[model] = result
    for name, data in [('conditions', conditions), ('continuity_summary', runs), ('source_partitions', sources), ('operating_points', all_points), ('size_interactions', interactions)]:
        save(OUT / (name + '.json'), data)
    save(OUT / 'secondary_results.json', dict(models=results, planned_groups=7,
         role='Completion of previously specified SNN secondary size test after RVT primary results; no new primary inference',
         parent_primary_results_sha256=sha(PARENT / 'primary_results.json'),
         uncertainty='Nominal descriptive recording t intervals, correlated checkpoints/shared cities; no frame-independent or multiplicity-controlled significance',
         stateful_reuse='not_executed', source_sha256=sha(ROOT / 'code/scripts/summarize_snn_secondary.py')))
    print('SNN secondary results complete', {m: r['summary'] for m, r in results.items()}, flush=True)

if __name__ == '__main__':
    main()
