"""Separate integer reconstruction of annotation strata and primary-point joins."""
from collections import defaultdict, Counter
import json, math
import numpy as np
from confirmation_common import ROOT, read, save, sha

OUT = ROOT / 'experiments/driving_factors_v01'
PARENT = ROOT / 'experiments/fp_mechanism_v01'

def main():
    cfg = read(ROOT / 'code/configs/driving_factors_v01.json')
    frozen = read(OUT / 'freeze.json')
    hashes = 0
    for mapping in [frozen['sources'], frozen['inputs']]:
        for path, digest in mapping.items():
            assert sha(ROOT / path) == digest, path
            hashes += 1
    groups = cfg['groups']
    factors_by_seq, cohorts = {}, {}
    factor_count = 0
    for seq in cfg['sequences']:
        cm = read(OUT / 'cohorts' / (seq + '.json'))
        cohorts[seq] = cm
        folder = ROOT / 'datasets/dsec_pilot/train' / seq
        grid = np.loadtxt(folder / 'image_timestamps.txt', dtype=np.int64)
        grid_index = {int(t): i for i, t in enumerate(grid)}
        gap = float(np.median(np.diff(grid)))
        labels = np.load(folder / 'object_detections/left/tracks.npy')
        trajectories = defaultdict(list)
        for r in labels:
            if str(int(r['class_id'])) not in cfg['class_map']:
                continue
            box = (max(0., float(r['x'])), max(0., float(r['y'])),
                   min(cfg['width'], float(r['x'] + r['w'])),
                   min(cfg['height'], float(r['y'] + r['h'])))
            if box[2] > box[0] and box[3] > box[1]:
                trajectories[int(r['track_id'])].append((int(r['t']), box, int(r['class_id'])))
        independent = {}
        for identity, track in trajectories.items():
            track.sort()
            censored = track[0][0] == int(grid[0])
            for i, (t, box, cls) in enumerate(track):
                area = (box[2] - box[0]) * (box[3] - box[1]) / (cfg['width'] * cfg['height'])
                stage = 'left_censored' if censored else 'early' if i < 3 else 'established'
                size = 'small' if area <= .01 else 'large'
                speed = None
                if i:
                    pt, pb, _ = track[i - 1]
                    if grid_index[t] - grid_index[pt] == 1 and t - pt <= 1.5 * gap:
                        dx = (box[0] + box[2] - pb[0] - pb[2]) / 2
                        dy = (box[1] + box[3] - pb[1] - pb[3]) / 2
                        speed = math.sqrt(dx * dx + dy * dy) * 1e6 / (t - pt) / 800
                motion = ('motion_unknown' if speed is None else 'motion_low' if speed <= .01
                          else 'motion_high' if speed > .05 else 'motion_middle')
                tags = ['all', size, stage, motion, 'car' if cls == 2 else 'pedestrian']
                if not censored:
                    tags.append(size + '_' + stage)
                independent[t, identity] = (box, tags, speed, i + 1)
        by_q = defaultdict(dict)
        path = OUT / 'factors' / (seq + '.jsonl')
        assert sha(path) == cm['factor_sha256']
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                box, tags, speed, number = independent[r['time_us'], r['identity']]
                assert set(tags) == set(r['groups']) and list(box) == r['box']
                assert number == r['appearance_number']
                assert ((speed is None and r['motion_diagonal_per_second'] is None)
                        or (speed is not None and abs(speed - r['motion_diagonal_per_second']) < 1e-12))
                by_q[r['q']][r['identity']] = tags
                factor_count += 1
        for frame in cm['frames']:
            assert set(by_q[frame['q']]) == set(frame['gt_ids'])
        for group in groups:
            members = [(q, identity) for q, rows in by_q.items() for identity, tags in rows.items() if group in tags]
            pop = cm['populations'][group]
            assert pop['instances'] == len(members)
            assert pop['tracks'] == len({identity for q, identity in members})
        factors_by_seq[seq] = by_q

    rows = read(OUT / 'conditions.json')
    row_lookup = {(r['model'], r['sequence'], r['key'], r['group']): r for r in rows}
    pair_lookup = {(r['model'], r['sequence'], r['regime'], r['target'], r['method'], r['group']): r for r in rows}
    primary_rows, primary_groups, run_count, source_rows = 0, 0, 0, 0
    for model in cfg['models']:
        for seq in cfg['sequences']:
            cm = cohorts[seq]
            factors = factors_by_seq[seq]
            plans = read(PARENT / 'points' / (model + '__' + seq + '.json'))['conditions']
            totals = defaultdict(lambda: {g: [0] * 6 for g in groups})
            track_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
            primary = {}
            with (PARENT / 'points' / model / (seq + '.jsonl')).open() as f:
                for line in f:
                    r = json.loads(line)
                    p = plans[r['key']]
                    if p['target'] != 1.0 or p['attempt'] != r['attempt']:
                        continue
                    q, key = r['q'], r['key']
                    detected, dense = set(r['matched_ids']), set(r['dense_ids'])
                    primary[key, q] = (dense, detected)
                    for identity, tags in factors[q].items():
                        hit, ref = identity in detected, identity in dense
                        values = [int(hit), int(ref), int(hit and ref), int(ref and not hit), int(hit and not ref), 1]
                        for group in tags:
                            v = totals[key][group]
                            for j, value in enumerate(values):
                                v[j] += value
                            if ref:
                                tr = track_counts[key, group][identity]
                                tr[0] += 1
                                tr[1] += hit
                    primary_rows += 1
            for key in {key for key, q in primary}:
                for group, v in totals[key].items():
                    r = row_lookup[model, seq, key, group]
                    assert v == [r[k] for k in ['tp', 'dense_tp', 'kept', 'lost', 'gained', 'gt']]
                    tracks = track_counts[key, group]
                    assert len(tracks) == r['dense_tracks']
                    eq = sum(b / a for a, b in tracks.values()) / len(tracks) if tracks else None
                    assert ((eq is None and r['track_equal_preservation'] is None)
                            or (eq is not None and abs(eq - r['track_equal_preservation']) < 1e-12))
                    primary_groups += 1
            # Independently verify maximal sampled loss intervals and all boundaries.
            time_to_q = {r['time_us']: r['q'] for r in cm['frames']}
            seen_losses, run_stats = defaultdict(set), defaultdict(Counter)
            def state(key, q, identity):
                if q < 0:
                    return 'sequence_start'
                if q >= len(cm['frames']):
                    return 'sequence_end'
                dense, detected = primary[key, q]
                return ('gt_absence' if identity not in factors[q] else 'dense_miss' if identity not in dense
                        else 'preserved' if identity in detected else 'lost')
            def gap_at(q):
                if not 0 < q < len(cm['frames']):
                    return False
                a, b = cm['frames'][q - 1:q + 1]
                return b['grid_index'] - a['grid_index'] != 1 or b['time_us'] - a['time_us'] > cfg['gap_factor'] * cm['median_grid_gap_us']
            with (OUT / 'continuity' / (model + '__' + seq + '.jsonl')).open() as f:
                for line in f:
                    r = json.loads(line)
                    key, identity = r['key'], r['track_id']
                    start, end = time_to_q[r['first_loss_us']], time_to_q[r['last_loss_us']]
                    assert r['queries'] == end - start + 1
                    assert r['observed_span_us'] == r['last_loss_us'] - r['first_loss_us']
                    for q in range(start, end + 1):
                        assert state(key, q, identity) == 'lost' and (q == start or not gap_at(q))
                        assert (q, identity) not in seen_losses[key]
                        seen_losses[key].add((q, identity))
                    left = 'query_gap' if gap_at(start) else state(key, start - 1, identity)
                    following = state(key, end + 1, identity)
                    right = ('sequence_end' if following == 'sequence_end' else 'query_gap' if gap_at(end + 1)
                             else 'recovered' if following == 'preserved' else following)
                    assert left == r['left_cause'] and right == r['end_cause']
                    assert r['left_censored'] == (left != 'preserved') and r['right_censored'] == (right != 'recovered')
                    if right == 'recovered':
                        assert r['recovery_time_us'] == cm['frames'][end + 1]['time_us']
                        assert r['observed_to_recovery_us'] == r['recovery_time_us'] - r['first_loss_us']
                    else:
                        assert r['recovery_time_us'] is None and r['observed_to_recovery_us'] is None
                    stats = run_stats[key]
                    stats['runs'] += 1
                    stats['longest_queries'] = max(stats['longest_queries'], r['queries'])
                    stats['right_censored'] += r['right_censored']
                    stats['left_censored'] += r['left_censored']
                    run_count += 1
            for r in read(OUT / 'continuity' / (model + '__' + seq + '.json')):
                key = r['key']
                expected = {(q, identity) for (k, q), (dense, detected) in primary.items() if k == key for identity in dense - detected}
                assert seen_losses[key] == expected
                assert len(expected) == r['lost_instances']
                for k in ['runs', 'longest_queries', 'right_censored', 'left_censored']:
                    assert run_stats[key][k] == r[k]
            tally = defaultdict(Counter)
            with (PARENT / 'source_survival' / (model + '__' + seq + '.jsonl')).open() as f:
                for line in f:
                    r = json.loads(line)
                    selected = r['mask_selected'] and r['above_threshold']
                    for group in factors[r['q']][r['identity']]:
                        v = tally[r['regime'], r['method'], group]
                        v['dense_instances'] += 1
                        v['coverage'] += selected
                        v['alternative_recovery'] += r['identity_detected'] and not selected
                        v['selected_source_loss'] += selected and not r['identity_detected']
                        v['kept'] += r['identity_detected']
                        v[r['category']] += 1
                    source_rows += 1
            for r in read(OUT / 'source_partitions' / (model + '__' + seq + '.json')):
                v = tally[r['regime'], r['method'], r['group']]
                for k, value in v.items():
                    assert r[k] == value
                assert r['kept'] == r['coverage'] + r['alternative_recovery'] - r['selected_source_loss']
            print('Independent factors, primary joins, runs and sources passed', model, seq, flush=True)
    for r in rows:
        assert r['tp'] == r['kept'] + r['gained'] and r['dense_tp'] == r['kept'] + r['lost']
        assert r['recall'] == (r['tp'] / r['gt'] if r['gt'] else None)
        assert r['preservation'] == (r['kept'] / r['dense_tp'] if r['dense_tp'] else None)
    pairs = read(OUT / 'pairs.json')
    for p in pairs:
        a, b = [pair_lookup[p['model'], p['sequence'], p['regime'], p['target'], m, p['group']] for m in [p['method_a'], p['method_b']]]
        expected = 100 * (a['tp'] - b['tp']) / a['gt'] if a['gt'] else None
        assert (expected is None and p['delta_recall_pp'] is None) or (expected is not None and abs(expected - p['delta_recall_pp']) < 1e-10)
    for s in read(OUT / 'group_summaries.json'):
        pp = [p for p in pairs if all(p[k] == s[k] for k in ['model', 'regime', 'target', 'method_a', 'method_b', 'group'])]
        values = [p['delta_recall_pp'] for p in pp if p['delta_recall_pp'] is not None]
        assert s['gt_instances'] == sum(p['gt'] for p in pp)
        assert s['dense_instances'] == sum(p['dense_tp'] for p in pp)
        assert s['positive'] == sum(v > 0 for v in values) and s['negative'] == sum(v < 0 for v in values)
        assert s['median_delta_recall_pp'] == (float(np.median(values)) if values else None)
    initial = read(ROOT / 'provenance/fp_mechanism_v01_initial_preservation.json')
    for r in initial['protected_files']:
        assert sha(ROOT / r['path']) == r['sha256']
        hashes += 1
    save(OUT / 'verification.json', dict(passed=True, checked_input_hashes=hashes,
         annotation_instances=factor_count, primary_point_query_rows=primary_rows,
         independently_rebuilt_primary_group_conditions=primary_groups,
         source_identity_query_rows=source_rows, independently_checked_loss_runs=run_count,
         all_group_conditions=len(rows), all_contrasts=len(pairs),
         protected_initial_numeric_receipts=len(initial['protected_files']),
         verifier_sha256=sha(ROOT / 'code/scripts/verify_driving_factors_v01.py'),
         scope='Independent annotation factors and all nine methods/both regimes at1FP; all point arithmetic, source partitions and continuity. No new detector inference or independent external data.'))
    print(json.dumps(read(OUT / 'verification.json')), flush=True)

if __name__ == '__main__':
    main()
