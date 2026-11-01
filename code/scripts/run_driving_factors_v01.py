"""Freeze label-defined strata, then join unchanged, actually replayed detections."""
from collections import defaultdict
from pathlib import Path
import argparse, datetime, json, sys
import numpy as np
from confirmation_common import ROOT, read, save, sha
from riskaware.driving_factors import annotation_factors, group_identities
from riskaware.evidence_runs import evidence_runs

CONFIG = ROOT / 'code/configs/driving_factors_v01.json'
OUT = ROOT / 'experiments/driving_factors_v01'
PARENT = ROOT / 'experiments/fp_mechanism_v01'
META = ROOT / 'experiments/confirmation_v01'
SOURCES = ['code/configs/driving_factors_v01.json', 'code/riskaware/driving_factors.py',
           'code/scripts/run_driving_factors_v01.py', 'code/tests/test_driving_factors.py',
           'code/riskaware/evidence_runs.py']

def prepare(cfg):
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('Annotation-stratum freeze already exists')
    if not read(ROOT / 'provenance/v16_paper_check.json')['passed']:
        raise ValueError('Prior manuscript/numeric audit must pass')
    paths = [PARENT / p for p in ['artifact_manifest.json', 'operating_points.json', 'paired_fp_contrasts.json']]
    for seq in cfg['sequences']:
        folder = ROOT / 'datasets/dsec_pilot/train' / seq
        paths.extend([folder / 'image_timestamps.txt', folder / 'object_detections/left/tracks.npy'])
        for model in cfg['models']:
            paths.extend([META / (model + '__' + seq + '.json'),
                          PARENT / 'points' / (model + '__' + seq + '.json'),
                          PARENT / 'points' / model / (seq + '.jsonl'),
                          PARENT / 'source_survival' / (model + '__' + seq + '.jsonl')])
    save(OUT / 'freeze.json', dict(config=cfg,
         frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         sources={p: sha(ROOT / p) for p in SOURCES},
         inputs={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
         prior_overall_outcomes_known=True, before_new_subgroup_outcome_joins=True,
         role='Local label-rule freeze only; not untouched confirmation or external preregistration'))
    populations = []
    for seq in cfg['sequences']:
        folder = ROOT / 'datasets/dsec_pilot/train' / seq
        grid = np.loadtxt(folder / 'image_timestamps.txt', dtype=np.int64)
        labels = np.load(folder / 'object_detections/left/tracks.npy')
        factors, gap = annotation_factors(grid, labels, cfg)
        metadata = [read(META / (m + '__' + seq + '.json')) for m in cfg['models']]
        times = [r['time_us'] for r in metadata[0]['frames']]
        if any(times != [r['time_us'] for r in m['frames']] for m in metadata[1:]):
            raise AssertionError('Carrier query cohorts differ')
        by_time = defaultdict(list)
        for (t, identity), row in factors.items():
            by_time[t].append(row)
        frames, rows = [], []
        population = {g: dict(instances=0, tracks=set(), queries=0) for g in cfg['groups']}
        for q, t in enumerate(times):
            current = sorted(by_time[t], key=lambda r: r['identity'])
            frames.append(dict(q=q, time_us=t, grid_index=int(np.searchsorted(grid, t)),
                               gt_ids=[r['identity'] for r in current]))
            for r in current:
                rows.append(dict(q=q, **r))
            for group, ids in group_identities(current, cfg['groups']).items():
                population[group]['instances'] += len(ids)
                population[group]['tracks'].update(ids)
                population[group]['queries'] += bool(ids)
        dest = OUT / 'factors' / (seq + '.jsonl')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w') as f:
            for r in rows:
                f.write(json.dumps(r, separators=(',', ':')) + '\n')
        save(OUT / 'cohorts' / (seq + '.json'), dict(sequence=seq, frames=frames, median_grid_gap_us=gap,
             factor_sha256=sha(dest), native_grid_start_us=int(grid[0]),
             populations={g: dict(instances=v['instances'], tracks=len(v['tracks']), queries=v['queries'])
                          for g, v in population.items()}))
        populations.append(dict(sequence=seq, queries=len(times),
                           populations=read(OUT / 'cohorts' / (seq + '.json'))['populations']))
        print('Annotation factors frozen', seq, 'queries', len(times), 'GT instances', len(rows), flush=True)
    save(OUT / 'populations.json', populations)
    save(OUT / 'prepare_complete.json', dict(passed=True, outcome_joins_not_started=True,
         prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

def frozen(cfg):
    record = read(OUT / 'freeze.json')
    if cfg != record['config']:
        raise ValueError('Stratum configuration drift')
    for path, digest in record['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    for path, digest in record['inputs'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen input changed: ' + path)
    return record

def cohort(seq, cfg):
    manifest = read(OUT / 'cohorts' / (seq + '.json'))
    path = OUT / 'factors' / (seq + '.jsonl')
    if sha(path) != manifest['factor_sha256']:
        raise ValueError('Annotation factors changed')
    by_q, lookup = defaultdict(list), {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            by_q[r['q']].append(r)
            lookup[r['q'], r['identity']] = r
    groups = [group_identities(by_q[q], cfg['groups']) for q in range(len(manifest['frames']))]
    return manifest, groups, lookup

def evaluate(cfg):
    if (OUT / 'complete.json').exists():
        raise FileExistsError('Completed analysis may not be overwritten')
    frozen(cfg)
    if not read(OUT / 'prepare_complete.json')['passed']:
        raise ValueError('Label factors not prepared')
    for model in cfg['models']:
        for seq in cfg['sequences']:
            dest = OUT / 'groups' / (model + '__' + seq + '.json')
            if dest.exists():
                continue
            cm, groups, factor_lookup = cohort(seq, cfg)
            plans = read(PARENT / 'points' / (model + '__' + seq + '.json'))['conditions']
            totals = {key: {g: defaultdict(int) for g in cfg['groups']} for key in plans}
            trajectories = defaultdict(lambda: defaultdict(lambda: [0, 0]))
            full = {key: defaultdict(int) for key in plans}
            histories = defaultdict(list)
            point_rows = 0
            with (PARENT / 'points' / model / (seq + '.jsonl')).open() as f:
                for line in f:
                    r = json.loads(line)
                    key = r['key']
                    plan = plans[key]
                    if r['attempt'] != plan['attempt']:
                        continue
                    q = r['q']
                    frame = cm['frames'][q]
                    if r['time_us'] != frame['time_us'] or r['threshold'] != plan['threshold']:
                        raise AssertionError('Actual point query or threshold changed')
                    gt, dense, detected = set(frame['gt_ids']), set(r['dense_ids']), set(r['matched_ids'])
                    if not dense <= gt or not detected <= gt or len(detected) != r['tp'] or len(gt) != r['gt']:
                        raise AssertionError('Identity membership inconsistent')
                    v = full[key]
                    v['queries'] += 1
                    for name, value in [('gt', len(gt)), ('tp', len(detected)), ('fp', r['fp']),
                                        ('dense_tp', len(dense)), ('kept', len(dense & detected)),
                                        ('gained', len(detected - dense))]:
                        v[name] += value
                    for group, ids in groups[q].items():
                        g = totals[key][group]
                        ref, hits = dense & ids, detected & ids
                        for name, value in [('gt', len(ids)), ('tp', len(hits)), ('dense_tp', len(ref)),
                                            ('kept', len(ref & hits)), ('lost', len(ref - hits)),
                                            ('gained', len(hits - ref))]:
                            g[name] += value
                        for identity in ref:
                            tr = trajectories[key, group][identity]
                            tr[0] += 1
                            tr[1] += identity in hits
                    if plan['target'] == cfg['primary_fp_target']:
                        histories[key].append(dict(**frame, dense_ids=sorted(dense), matched_ids=sorted(detected)))
                    point_rows += 1
            conditions = []
            for key, plan in plans.items():
                if any(full[key][k] != plan[k] for k in ['queries', 'gt', 'tp', 'fp', 'dense_tp', 'kept', 'gained']):
                    raise AssertionError('Full sequence failed protected endpoint regression: ' + key)
                for group, v in totals[key].items():
                    v = dict(v)
                    pop = cm['populations'][group]
                    if v['gt'] != pop['instances'] or v['tp'] != v['kept'] + v['gained'] or v['dense_tp'] != v['kept'] + v['lost']:
                        raise AssertionError('Subgroup partition failed')
                    tracks = trajectories[key, group]
                    v.update(gt_tracks=pop['tracks'], dense_tracks=len(tracks),
                             tracks_with_loss=sum(b < a for a, b in tracks.values()),
                             recall=v['tp'] / v['gt'] if v['gt'] else None,
                             preservation=v['kept'] / v['dense_tp'] if v['dense_tp'] else None,
                             track_equal_preservation=float(np.mean([b / a for a, b in tracks.values()])) if tracks else None)
                    conditions.append(dict(model=model, sequence=seq, key=key, group=group,
                         method=plan['method'], regime=plan['regime'], target=plan['target'],
                         threshold=plan['threshold'], full_sequence_fp=plan['fp'],
                         full_sequence_queries=plan['queries'], full_sequence_allowed_fp=plan['allowed_fp'], **v))
            run_summary, all_runs = [], []
            for key, frames in histories.items():
                frames.sort(key=lambda r: r['q'])
                if len(frames) != len(cm['frames']):
                    raise AssertionError('Continuity cohort incomplete')
                runs, stats = evidence_runs(frames, cm['median_grid_gap_us'], cfg['gap_factor'])
                if stats['lost_instances'] != totals[key]['all']['lost']:
                    raise AssertionError('Continuity loss totals differ')
                plan = plans[key]
                run_summary.append(dict(model=model, sequence=seq, key=key,
                                        method=plan['method'], regime=plan['regime'], target=plan['target'], **stats))
                all_runs.extend(dict(model=model, sequence=seq, key=key, **r) for r in runs)
            save(dest, dict(model=model, sequence=seq, conditions=conditions,
                            protected_points=len(plans), joined_point_query_rows=point_rows))
            save(OUT / 'continuity' / (model + '__' + seq + '.json'), run_summary)
            runpath = OUT / 'continuity' / (model + '__' + seq + '.jsonl')
            with runpath.open('w') as f:
                for r in all_runs:
                    f.write(json.dumps(r, separators=(',', ':')) + '\n')
            source_totals = defaultdict(lambda: defaultdict(int))
            with (PARENT / 'source_survival' / (model + '__' + seq + '.jsonl')).open() as f:
                for line in f:
                    r = json.loads(line)
                    factor = factor_lookup[r['q'], r['identity']]
                    eligible = r['mask_selected'] and r['above_threshold']
                    for group in factor['groups']:
                        v = source_totals[r['regime'], r['method'], group]
                        v['dense_instances'] += 1
                        v['coverage'] += eligible
                        v['alternative_recovery'] += r['identity_detected'] and not eligible
                        v['selected_source_loss'] += eligible and not r['identity_detected']
                        v['kept'] += r['identity_detected']
                        v[r['category']] += 1
            partitions = []
            for (regime, method, group), v in source_totals.items():
                if v['kept'] != v['coverage'] + v['alternative_recovery'] - v['selected_source_loss']:
                    raise AssertionError('Source partition failed')
                partitions.append(dict(model=model, sequence=seq, regime=regime, method=method, group=group, **v))
            save(OUT / 'source_partitions' / (model + '__' + seq + '.json'), partitions)
            save(OUT / 'status.json', dict(state='running', completed_model=model, completed_sequence=seq))
            print('Subgroup joins complete', model, seq, 'points', len(plans), 'rows', point_rows, flush=True)
    summarize(cfg)

def summarize(cfg):
    rows, continuity, source = [], [], []
    for model in cfg['models']:
        for seq in cfg['sequences']:
            stem = model + '__' + seq + '.json'
            rows.extend(read(OUT / 'groups' / stem)['conditions'])
            continuity.extend(read(OUT / 'continuity' / stem))
            source.extend(read(OUT / 'source_partitions' / stem))
    lookup = {(r['model'], r['sequence'], r['regime'], r['target'], r['method'], r['group']): r for r in rows}
    pairs = []
    for model in cfg['models']:
        for seq in cfg['sequences']:
            for regime in cfg['regimes']:
                for target in cfg['fp_targets']:
                    for ma, mb in cfg['contrasts']:
                        for group in cfg['groups']:
                            a, b = [lookup[model, seq, regime, target, m, group] for m in [ma, mb]]
                            if a['gt'] != b['gt'] or a['dense_tp'] != b['dense_tp']:
                                raise AssertionError('Pair denominators differ')
                            allowed = a['full_sequence_allowed_fp']
                            tolerance = max(1, allowed // 100)
                            matched = (0 <= allowed - a['full_sequence_fp'] <= tolerance
                                       and 0 <= allowed - b['full_sequence_fp'] <= tolerance
                                       and abs(a['full_sequence_fp'] - b['full_sequence_fp']) <= tolerance)
                            pairs.append(dict(model=model, sequence=seq, regime=regime, target=target,
                                  method_a=ma, method_b=mb, group=group,
                                  full_sequence_target_matched=matched,
                                  gt=a['gt'], gt_tracks=a['gt_tracks'], dense_tp=a['dense_tp'], dense_tracks=a['dense_tracks'],
                                  tp_a=a['tp'], tp_b=b['tp'], kept_a=a['kept'], kept_b=b['kept'],
                                  full_sequence_fp_a=a['full_sequence_fp'], full_sequence_fp_b=b['full_sequence_fp'],
                                  delta_recall_pp=100 * (a['recall'] - b['recall']) if a['gt'] else None,
                                  delta_preservation_pp=100 * (a['preservation'] - b['preservation']) if a['dense_tp'] else None))
    groups = []
    for model in cfg['models']:
        for regime in cfg['regimes']:
            for target in cfg['fp_targets']:
                for ma, mb in cfg['contrasts']:
                    for group in cfg['groups']:
                        pp = [p for p in pairs if (p['model'], p['regime'], p['target'], p['method_a'], p['method_b'], p['group']) == (model, regime, target, ma, mb, group)]
                        valid = [p for p in pp if p['delta_recall_pp'] is not None]
                        values = [p['delta_recall_pp'] for p in valid]
                        groups.append(dict(model=model, regime=regime, target=target, method_a=ma, method_b=mb,
                             group=group, recordings=len(pp), defined_recordings=len(valid),
                             full_sequence_matched=sum(p['full_sequence_target_matched'] for p in pp),
                             positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values), tied=sum(v == 0 for v in values),
                             median_delta_recall_pp=float(np.median(values)) if values else None,
                             min_delta_recall_pp=min(values) if values else None, max_delta_recall_pp=max(values) if values else None,
                             gt_instances=sum(p['gt'] for p in pp), gt_tracks=sum(p['gt_tracks'] for p in pp),
                             dense_instances=sum(p['dense_tp'] for p in pp), dense_tracks=sum(p['dense_tracks'] for p in pp)))
    save(OUT / 'conditions.json', rows)
    save(OUT / 'pairs.json', pairs)
    save(OUT / 'group_summaries.json', groups)
    save(OUT / 'continuity_summary.json', continuity)
    save(OUT / 'source_partitions.json', source)
    save(OUT / 'complete.json', dict(state='complete', conditions=len(rows), contrasts=len(pairs),
         summaries=len(groups), continuity_conditions=len(continuity),
         no_new_thresholds=True, no_new_detector_inference=True, independent_verification_pending=True))
    save(OUT / 'status.json', dict(state='complete', verification_pending=True))
    print('All driving-factor joins complete', len(rows), 'group conditions', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    cfg = read(CONFIG)
    if args.phase == 'prepare':
        prepare(cfg)
    else:
        evaluate(cfg)

if __name__ == '__main__':
    main()
