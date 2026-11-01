"""Separately rebuild annotations, raw matching, provenance and secondary arithmetic."""
from collections import defaultdict
import argparse, json, math
import numpy as np
from scipy.stats import t as student_t
from snn_secondary_common import ROOT, OUT, CONFIG, PARENT, read, save, sha, assert_frozen
from verify_driving_confirmation_v02 import independent_annotations, independent_match
from run_fp_mechanism import canonical

def native_filtered(filtered):
    # Invert the fixed256 letterbox: scale in float32, then offsets in float64.
    boxes, scores, classes, anchors = filtered
    native = (boxes * 256).astype(np.float64)
    native[:, [0, 2]] /= .4
    native[:, [1, 3]] = (native[:, [1, 3]] - 32.) / .4
    return native, scores, classes, anchors

def verify_one(model, seq):
    cfg = read(CONFIG)
    assert model in cfg['models'] and seq in cfg['sequences']
    aggregate = read(OUT / 'aggregates' / (model + '__' + seq + '.json'))
    rows = aggregate['conditions']
    lookup = {(r['sequence'], r['key'], r['group']): r for r in rows}
    # Keep read-only review independent of final result summarization timing.
    point_file = read(OUT / 'points' / (model + '__' + seq + '.json'))
    point_lookup = point_file['conditions']
    ka,kb = ['retrospective_sequence_budget|' + m + '@1.0' for m in ['ema/absolute','current_activity']]
    la,lb,ha,hb = [lookup[seq,key,group] for key,group in [(ka,'small'),(kb,'small'),(ka,'large'),(kb,'large')]]
    val = 100*(la['tp']-lb['tp'])/la['gt'] - 100*(ha['tp']-hb['tp'])/ha['gt'] if la['gt'] and ha['gt'] else None
    pa,pb=point_lookup[ka],point_lookup[kb];tol=max(1,pa['allowed_fp']//100)
    match = 0<=pa['allowed_fp']-pa['fp']<=tol and 0<=pb['allowed_fp']-pb['fp']<=tol and abs(pa['fp']-pb['fp'])<=tol
    results={'effects':[dict(sequence=seq,interaction_pp=val,matched=match)]}
    hashes=primary_rows=raw_replays=sampled_replays=source_rows=checked_runs=total_point_rows=0
    source_expectations={};recomputed_effects=[]
    meta = read(OUT / 'carriers' / (model + '__' + seq + '.json'))
    assert meta['complete'] and all(r['all_open_exact'] for r in meta['alignment'])
    frames = independent_annotations(seq, meta)
    result = read(OUT / 'points' / (model + '__' + seq + '.json'))
    points = result['conditions']
    primary = {key for key, p in points.items() if p['target'] == 1.}
    sample_q = {0, len(frames) // 2, len(frames) - 1}
    actual = defaultdict(lambda: defaultdict(int))
    groups = defaultdict(lambda: defaultdict(lambda: [0] * 6))
    point_rows = defaultdict(dict)
    primary_states = {}
    trace = OUT / 'points' / model / (seq + '.jsonl')
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
    with (OUT / 'continuity' / (model + '__' + seq + '.jsonl')).open() as f:
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
    continuity = aggregate['continuity']
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
        reference_theta = read(ROOT / 'experiments/confirmation_v01/frozen_thresholds.json')[model]['dense_threshold']
        f0, _, _ = canonical(raw, masks[0], reference_theta, meta, [])
        ref, _ = independent_match(native_filtered(f0), frame['gt'])
        for key, r in requested.items():
            p = points[key]
            mask = masks[cfg['methods'].index(p['method'])]
            filtered, _, _ = canonical(raw, mask, p['threshold'], meta, [])
            matched, fp = independent_match(native_filtered(filtered), frame['gt'])
            assert set(matched) == set(r['matched_ids']) and fp == r['fp']
            assert set(ref) == set(r['dense_ids'])
            if p['target'] == 1. and p['method'] in ['ema/absolute', 'current_activity']:
                raw_replays += 1
                survived = set(map(int, filtered[3]))
                for identity, anchor in ref.items():
                    selected = bool(mask[anchor // meta['model_provenance']['anchors']])
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
    sourcepath = OUT / 'source_survival' / (model + '__' + seq + '.jsonl')
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
    la, lb, ha, hb = [lookup[seq, key, group] for key, group in [(ka, 'small'), (kb, 'small'), (ka, 'large'), (kb, 'large')]]
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
        assert released['sha256'] == receipt['event']['sha256']
    print('Independent new-record raw/matching/subgroup reconstruction passed', seq, flush=True)
    assert not source_expectations
    paths = [OUT/'carriers'/(model+'__'+seq+'.json'), OUT/'points'/(model+'__'+seq+'.json'), OUT/'aggregates'/(model+'__'+seq+'.json')]
    receipt=dict(passed=True,model=model,sequence=seq,interaction_pp=recomputed_effects[0],matched=match,
        checked_hashes=hashes,total_final_point_query_rows=total_point_rows,independent_primary_group_query_rows=primary_rows,
        actual_primary_raw_replays=raw_replays,other_sampled_raw_replays=sampled_replays,
        exact_source_identity_query_rows=source_rows,independently_checked_loss_runs=checked_runs,
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        verifier_sha256=sha(ROOT/'code/scripts/verify_snn_secondary.py'),
        annotation_matcher_source_sha256=sha(ROOT/'code/scripts/verify_driving_confirmation_v02.py'))
    save(OUT/'checks'/(model+'__'+seq+'.json'),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--sequence');args=parser.parse_args()
    assert_frozen();cfg=read(CONFIG)
    if args.model:
        verify_one(args.model,args.sequence);return
    results=read(OUT/'secondary_results.json');receipts=[]
    for model in cfg['models']:
        values=[]
        for seq in cfg['sequences']:
            p=OUT/'checks'/(model+'__'+seq+'.json')
            receipt=read(p) if p.exists() else verify_one(model,seq)
            assert receipt['passed'] and receipt['verifier_sha256']==sha(ROOT/'code/scripts/verify_snn_secondary.py')
            for path,digest in receipt['inputs'].items():assert sha(ROOT/path)==digest,path
            saved=next(r for r in results['models'][model]['effects'] if r['sequence']==seq)
            value=receipt['interaction_pp'];assert (value is None and saved['interaction_pp'] is None) or abs(value-saved['interaction_pp'])<1e-10
            assert receipt['matched']==saved['matched'];values.append(value);receipts.append(receipt)
        defined=[v for v in values if v is not None];summary=results['models'][model]['summary']
        assert summary['defined']==len(defined) and summary['planned']==7
        assert summary['negative']==sum(v<0 for v in defined) and summary['positive']==sum(v>0 for v in defined)
        if defined:
            mean=sum(defined)/len(defined);assert abs(mean-summary['mean'])<1e-10
            if len(defined)>=2:
                half=float(student_t.ppf(.975,len(defined)-1)*math.sqrt(sum((v-mean)**2 for v in defined)/(len(defined)-1))/math.sqrt(len(defined)))
                assert np.allclose([mean-half,mean+half],results['models'][model]['nominal_recording_t_CI95'],rtol=0,atol=1e-10)
    assert results['parent_primary_results_sha256']==sha(PARENT/'primary_results.json')
    initial=read(ROOT/'provenance/fp_mechanism_v01_initial_preservation.json')
    for row in initial['protected_files']:assert sha(ROOT/row['path'])==row['sha256']
    counts=['checked_hashes','total_final_point_query_rows','independent_primary_group_query_rows','actual_primary_raw_replays','other_sampled_raw_replays','exact_source_identity_query_rows','independently_checked_loss_runs']
    save(OUT/'verification.json',dict(passed=True,recording_groups=7,checkpoints=cfg['models'],pair_receipts=len(receipts),
        **{key:sum(r[key] for r in receipts) for key in counts},protected_old_numeric_receipts=32,
        verifier_sha256=sha(ROOT/'code/scripts/verify_snn_secondary.py'),scope='Independent annotation/matching reconstruction, all point integer totals, all primary group/source/loss paths. Shared raw model outputs/upstreamNMS. Two checkpoints are not independent architectures.'))
    print('SNN secondary independent reconstruction passed',flush=True)

if __name__=='__main__':main()
