"""Independent primary raw replay/matching and annotation-stratum reconstruction."""
from collections import defaultdict
import json, math
import numpy as np
from scipy.stats import t as student_t
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha
from run_fp_mechanism import canonical

def independent_annotations(seq, meta):
    folder = ROOT / 'datasets/dsec_confirmation/test' / seq
    grid = np.loadtxt(folder / 'image_timestamps.txt', dtype=np.int64)
    labels = np.load(folder / 'object_detections/left/tracks.npy', allow_pickle=False)
    index = {int(t): i for i, t in enumerate(grid)}
    gap = float(np.median(np.diff(grid)))
    by_track, native_gt = defaultdict(list), defaultdict(list)
    for r in labels:
        cls = int(r['class_id'])
        if cls not in [0, 2]:
            continue
        box = [max(0., float(r['x'])), max(0., float(r['y'])),
               min(640., float(r['x'] + r['w'])), min(480., float(r['y'] + r['h']))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        timestamp, identity = int(r['t']), int(r['track_id'])
        by_track[identity].append((timestamp, box, cls))
        native_gt[timestamp].append(dict(box=box, cls=0 if cls == 2 else 1, identity=identity))
    tags = {}
    for identity, trajectory in by_track.items():
        trajectory.sort()
        censored = trajectory[0][0] == int(grid[0])
        for i, (t, box, cls) in enumerate(trajectory):
            stage = 'left_censored' if censored else 'early' if i < 3 else 'established'
            size = 'small' if (box[2] - box[0]) * (box[3] - box[1]) <= .01 * 640 * 480 else 'large'
            speed = None
            if i:
                pt, pb, _ = trajectory[i - 1]
                if index[t] - index[pt] == 1 and t - pt <= 1.5 * gap:
                    dx, dy = (box[0] + box[2] - pb[0] - pb[2]) / 2, (box[1] + box[3] - pb[1] - pb[3]) / 2
                    speed = math.hypot(dx, dy) / ((t - pt) / 1e6) / 800
            motion = 'motion_unknown' if speed is None else 'motion_low' if speed <= .01 else 'motion_high' if speed > .05 else 'motion_middle'
            clsname = 'car' if cls == 2 else 'pedestrian'
            current = ['all', size, stage, motion, clsname]
            if not censored:
                current.append(size + '_' + stage)
            if motion in ['motion_low', 'motion_high']:
                current.append('cell/' + size + '|' + clsname + '|' + stage + '/' + motion)
            tags[t, identity] = current
    frames = []
    for q, r in enumerate(meta['frames']):
        gt = native_gt[r['time_us']]
        frames.append(dict(q=q, time_us=r['time_us'], grid_index=index[r['time_us']], gt=gt,
                           tags={g['identity']: tags[r['time_us'], g['identity']] for g in gt}))
    return frames

def independent_match(filtered, gt):
    boxes, scores, classes, anchors = filtered
    used, matched = set(), {}
    for di in sorted(range(len(scores)), key=lambda i: -float(scores[i])):
        b = list(map(float, boxes[di]))
        winner, best = None, .5
        for gi, g in enumerate(gt):
            if gi in used or int(classes[di]) != g['cls']:
                continue
            a = g['box']
            intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
            area = max(0., b[2] - b[0]) * max(0., b[3] - b[1]) + (a[2] - a[0]) * (a[3] - a[1]) - intersection
            value = intersection / area if area > 0 else 0.
            if value >= best:
                best, winner = value, gi
        if winner is not None:
            used.add(winner)
            matched[gt[winner]['identity']] = int(anchors[di])
    return matched, len(scores) - len(matched)

def main():
    cfg, frozen, results = read(CONFIG), read(OUT / 'freeze.json'), read(OUT / 'primary_results.json')
    hashes = 0
    for group in ['sources', 'inputs']:
        for path, digest in frozen[group].items():
            assert sha(ROOT / path) == digest, path
            hashes += 1
    rows = read(OUT / 'conditions.json')
    lookup = {(r['sequence'], r['key'], r['group']): r for r in rows}
    source_expectations = {}
    primary_rows = raw_replays = sampled_replays = source_rows = checked_runs = 0
    total_point_rows = 0
    recomputed_effects = []
    for seq in cfg['sequences']:
        meta = read(OUT / 'carriers' / ('rvt_stream__' + seq + '.json'))
        assert meta['complete'] and all(r['all_open_exact'] and r['explicit_state_replay_exact'] for r in meta['alignment'])
        frames = independent_annotations(seq, meta)
        result = read(OUT / 'points' / ('rvt_stream__' + seq + '.json'))
        points = result['conditions']
        primary = {key for key, p in points.items() if p['target'] == 1.}
        sample_q = {0, len(frames) // 2, len(frames) - 1}
        actual = defaultdict(lambda: defaultdict(int))
        groups = defaultdict(lambda: defaultdict(lambda: [0] * 6))
        point_rows = defaultdict(dict)
        primary_states = {}
        trace = OUT / 'points/rvt_stream' / (seq + '.jsonl')
        assert sha(trace) == result['trace_sha256']
        with trace.open() as f:
            for line in f:
                r = json.loads(line)
                p, key = points[r['key']], r['key']
                if r['attempt'] != p['attempt']:
                    continue
                frame = frames[r['q']]
                gt, dense, detected = set(frame['tags']), set(r['dense_ids']), set(r['matched_ids'])
                assert dense <= gt and detected <= gt and len(detected) == r['tp'] and len(gt) == r['gt']
                assert r['time_us'] == frame['time_us'] and r['threshold'] == p['threshold']
                for name, value in [('queries', 1), ('gt', len(gt)), ('tp', len(detected)), ('fp', r['fp']),
                                    ('dense_tp', len(dense)), ('kept', len(dense & detected)), ('gained', len(detected - dense))]:
                    actual[key][name] += value
                if key in primary:
                    primary_states[key, r['q']] = (dense, detected)
                    for identity, tags in frame['tags'].items():
                        hit, ref = identity in detected, identity in dense
                        values = [1, int(hit), int(ref), int(hit and ref), int(ref and not hit), int(hit and not ref)]
                        for group in tags:
                            v = groups[key][group]
                            for i, value in enumerate(values):
                                v[i] += value
                    primary_rows += 1
                if (key in primary and p['method'] in ['ema/absolute', 'current_activity']) or r['q'] in sample_q:
                    point_rows[r['q']][key] = r
                total_point_rows += 1
        for key, p in points.items():
            assert all(actual[key][k] == p[k] for k in ['queries', 'gt', 'tp', 'fp', 'dense_tp', 'kept', 'gained'])
        for key in primary:
            for group in {r['group'] for r in rows if r['sequence'] == seq and r['key'] == key}:
                expected = lookup[seq, key, group]
                assert groups[key][group] == [expected[k] for k in ['gt', 'tp', 'dense_tp', 'kept', 'lost', 'gained']]
        # Maximal observed-loss intervals are checked against all18 primary-point
        # method/regime histories, without calling the production run builder.
        times = {r['time_us']: r['q'] for r in frames}
        seen = defaultdict(set)
        run_tally = defaultdict(lambda: defaultdict(int))
        def state(key, q, identity):
            if q < 0:
                return 'sequence_start'
            if q >= len(frames):
                return 'sequence_end'
            dense, detected = primary_states[key, q]
            return ('gt_absence' if identity not in frames[q]['tags'] else 'dense_miss' if identity not in dense
                    else 'preserved' if identity in detected else 'lost')
        grid_gap = float(np.median(np.diff(np.loadtxt(ROOT / 'datasets/dsec_confirmation/test' / seq / 'image_timestamps.txt'))))
        def gap_at(q):
            if not 0 < q < len(frames):
                return False
            a, b = frames[q - 1:q + 1]
            return b['grid_index'] - a['grid_index'] != 1 or b['time_us'] - a['time_us'] > 1.5 * grid_gap
        with (OUT / 'continuity' / ('rvt_stream__' + seq + '.jsonl')).open() as f:
            for line in f:
                r = json.loads(line)
                start, end = times[r['first_loss_us']], times[r['last_loss_us']]
                key, identity = r['key'], r['track_id']
                assert r['queries'] == end - start + 1 and r['observed_span_us'] == r['last_loss_us'] - r['first_loss_us']
                for q in range(start, end + 1):
                    assert state(key, q, identity) == 'lost' and (q == start or not gap_at(q))
                    assert (q, identity) not in seen[key]
                    seen[key].add((q, identity))
                left = 'query_gap' if gap_at(start) else state(key, start - 1, identity)
                right = state(key, end + 1, identity)
                right = 'query_gap' if gap_at(end + 1) else 'recovered' if right == 'preserved' else right
                assert left == r['left_cause'] and right == r['end_cause']
                assert r['left_censored'] == (left != 'preserved') and r['right_censored'] == (right != 'recovered')
                run_tally[key]['runs'] += 1
                run_tally[key]['longest_queries'] = max(run_tally[key]['longest_queries'], r['queries'])
                run_tally[key]['right_censored'] += r['right_censored']
                checked_runs += 1
        continuity = [r for r in read(OUT / 'continuity_summary.json') if r['sequence'] == seq]
        for r in continuity:
            key = r['key']
            losses = {(q, identity) for (k, q), (dense, detected) in primary_states.items() if k == key for identity in dense - detected}
            assert seen[key] == losses and len(losses) == r['lost_instances']
            for k in ['runs', 'longest_queries', 'right_censored']:
                assert run_tally[key][k] == r[k]
        n = sum(g['height'] * g['width'] for g in meta['geometry'])
        # Use the shared upstream NMS, but independently rebuild GT matching from
        # raw candidate outputs. All primary A/B queries plus all points at3queries.
        for q, requested in sorted(point_rows.items()):
            record = meta['frames'][q]
            path = ROOT / record['path']
            assert sha(path) == record['sha256']
            hashes += 1
            with np.load(path) as z:
                raw, masks = dict(boxes=z['boxes'], scores=z['scores']), np.unpackbits(z['masks'], axis=1)[:, :n].astype(bool)
            frame = frames[q]
            reference_theta = read(ROOT / 'experiments/confirmation_v01/frozen_thresholds.json')['rvt_stream']['dense_threshold']
            f0, _, _ = canonical(raw, masks[0], reference_theta, meta, [])
            ref, _ = independent_match(f0, frame['gt'])
            for key, r in requested.items():
                p = points[key]
                mask = masks[cfg['methods'].index(p['method'])]
                filtered, _, _ = canonical(raw, mask, p['threshold'], meta, [])
                matched, fp = independent_match(filtered, frame['gt'])
                assert set(matched) == set(r['matched_ids']) and fp == r['fp']
                assert set(ref) == set(r['dense_ids'])
                if p['target'] == 1. and p['method'] in ['ema/absolute', 'current_activity']:
                    raw_replays += 1
                    survived = set(map(int, filtered[3]))
                    for identity, anchor in ref.items():
                        selected = bool(mask[anchor])
                        above = bool(raw['scores'][anchor].max() >= p['threshold'])
                        effective, detected = selected and above, identity in matched
                        category = ('original_match' if detected and matched[identity] == anchor else
                                    'admitted_alternative' if detected and effective else
                                    'alternative_after_mask' if detected and not selected else
                                    'alternative_after_confidence' if detected else
                                    'nms_loss' if effective and anchor not in survived else 'assignment_loss' if effective else
                                    'mask_miss' if not selected else 'confidence_miss')
                        source_expectations[seq, q, key, identity] = (category, selected, above, detected)
                elif q in sample_q:
                    sampled_replays += 1
        sourcepath = OUT / 'source_survival' / ('rvt_stream__' + seq + '.jsonl')
        assert sha(sourcepath) == result['source_trace_sha256']
        with sourcepath.open() as f:
            for line in f:
                r = json.loads(line)
                if r['attempt'] != points[r['key']]['attempt']:
                    continue
                expected = source_expectations.pop((seq, r['q'], r['key'], r['identity']))
                assert expected == (r['category'], r['mask_selected'], r['above_threshold'], r['identity_detected'])
                source_rows += 1
        regime = 'retrospective_sequence_budget'
        ka, kb = [regime + '|' + m + '@1.0' for m in ['ema/absolute', 'current_activity']]
        la, lb, ha, hb = [lookup[seq, key, group] for key, group in [(ka, 'motion_low'), (kb, 'motion_low'), (ka, 'motion_high'), (kb, 'motion_high')]]
        effect = (100 * (la['tp'] - lb['tp']) / la['gt'] - 100 * (ha['tp'] - hb['tp']) / ha['gt']) if la['gt'] and ha['gt'] else None
        saved = next(r for r in results['effects'] if r['sequence'] == seq)
        assert (effect is None and saved['interaction_pp'] is None) or abs(effect - saved['interaction_pp']) < 1e-10
        recomputed_effects.append(effect)
        pa, pb = points[ka], points[kb]
        tolerance = max(1, pa['allowed_fp'] // 100)
        matched = (0 <= pa['allowed_fp'] - pa['fp'] <= tolerance and 0 <= pb['allowed_fp'] - pb['fp'] <= tolerance
                   and abs(pa['fp'] - pb['fp']) <= tolerance)
        assert matched == saved['matched']
        receipt = read(OUT / 'data_receipts' / (seq + '.json'))
        eventpath = ROOT / receipt['event']['path']
        if eventpath.exists():
            assert sha(eventpath) == receipt['event']['sha256']
        else:
            released = read(OUT / 'released_public_inputs' / (seq + '.json'))
            assert released['sha256'] == receipt['event']['sha256'] and released['source_url'] == receipt['requested_url']
        print('Independent new-record raw/matching/subgroup reconstruction passed', seq, flush=True)
    assert not source_expectations
    defined = [v for v in recomputed_effects if v is not None]
    if defined:
        mean = sum(defined) / len(defined)
        assert abs(mean - results['summary']['mean']) < 1e-10
        if len(defined) >= 2:
            sd = math.sqrt(sum((v - mean) ** 2 for v in defined) / (len(defined) - 1))
            half = float(student_t.ppf(.975, len(defined) - 1) * sd / math.sqrt(len(defined)))
            assert np.allclose([mean - half, mean + half], results['nominal_recording_t_CI95'], rtol=0, atol=1e-10)
    # Independently reconstruct fixed-weight adjustment and its support failures.
    cross = read(OUT / 'cross_cells.json')
    weights = read(OUT / 'statistics_plan.json')['common_support_weights']
    cells_by_seq, pooled = defaultdict(dict), defaultdict(lambda: defaultdict(int))
    for r in cross:
        values = {k: r[k] for k in ['low_a', 'low_b', 'low_gt', 'high_a', 'high_b', 'high_gt']}
        ka, kb = ['retrospective_sequence_budget|' + m + '@1.0' for m in ['ema/absolute', 'current_activity']]
        source = [lookup[r['sequence'], key, 'cell/' + r['cell'] + '/' + motion] for key, motion in
                  [(ka, 'motion_low'), (kb, 'motion_low'), (ka, 'motion_high'), (kb, 'motion_high')]]
        assert list(values.values()) == [source[0]['tp'], source[1]['tp'], source[0]['gt'], source[2]['tp'], source[3]['tp'], source[2]['gt']]
        cells_by_seq[r['sequence']][r['cell']] = values
        for k, value in values.items():
            pooled[r['cell']][k] += value
    def adjustment(values):
        answer = 0.
        for cell, weight in weights.items():
            v = values[cell]
            if not v['low_gt'] or not v['high_gt']:
                return None
            answer += weight * (100 * (v['low_a'] - v['low_b']) / v['low_gt'] - 100 * (v['high_a'] - v['high_b']) / v['high_gt'])
        return answer
    adjusted = adjustment(pooled)
    saved_adjusted = results['adjustment']['pooled_point_pp']
    assert (adjusted is None and saved_adjusted is None) or abs(adjusted - saved_adjusted) < 1e-10
    for r in results['effects']:
        value = adjustment(cells_by_seq[r['sequence']])
        assert (value is None and r['adjusted_interaction_pp'] is None) or abs(value - r['adjusted_interaction_pp']) < 1e-10
    undefined = 0
    for indices in np.random.default_rng(20261031).integers(0, 7, size=(5000, 7)):
        values = defaultdict(lambda: defaultdict(int))
        for i in indices:
            for cell, v in cells_by_seq[cfg['sequences'][int(i)]].items():
                for k, value in v.items():
                    values[cell][k] += value
        undefined += adjustment(values) is None
    assert undefined == results['adjustment']['undefined_bootstrap_draws']
    ci = results['nominal_recording_t_CI95']
    expected_verdict = ('inconclusive' if len(defined) != 7 or not all(r['matched'] for r in results['effects']) or ci is None else
                        'supports_declared_negative_interaction' if ci[1] < 0 else 'contrary_positive_interaction' if ci[0] > 0 else 'inconclusive')
    assert results['verdict'] == expected_verdict
    initial = read(ROOT / 'provenance/fp_mechanism_v01_initial_preservation.json')
    for r in initial['protected_files']:
        assert sha(ROOT / r['path']) == r['sha256']
        hashes += 1
    save(OUT / 'verification.json', dict(passed=True, checked_hashes=hashes, total_final_point_query_rows=total_point_rows,
          independent_primary_group_query_rows=primary_rows, actual_primary_raw_replays=raw_replays,
          other_sampled_raw_replays=sampled_replays, exact_source_identity_query_rows=source_rows,
          independently_checked_loss_runs=checked_runs,
          recording_groups=len(cfg['sequences']), models_new_data=['rvt_stream'],
          secondary_SNN_new_data='not_executed', protected_old_numeric_receipts=32,
          verifier_sha256=sha(ROOT / 'code/scripts/verify_driving_confirmation_v02.py'),
          scope='Frozen-input integrity; separately reconstructed annotation strata; all primary A/B rawNMS with independent GT matching, all-method primary subgroup counts and all-point integer totals. Upstream NMS/raw model outputs shared; not independent hardware/sensor evidence.'))
    print('Independent primary-stage verification passed', flush=True)

if __name__ == '__main__':
    main()
