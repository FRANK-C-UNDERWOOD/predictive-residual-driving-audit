"""Actual whole-recording FP replay on compact, pinned new-recording outputs."""
from collections import defaultdict
import argparse, json, time
import numpy as np
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha, assert_frozen
from run_fp_mechanism import canonical
from riskaware.criterion_audit import ground_truth
from riskaware.calibration_audit import threshold_at_fppi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    cfg = read(CONFIG)
    preflight = args.preflight
    if preflight and args.sequence != 'thun_00_a':
        raise ValueError('Preflight uses old calibration data only')
    if not preflight:
        assert_frozen()
    m = read(OUT / ('preflight' if preflight else 'carriers') / (args.model + '__' + args.sequence + '.json'))
    destination = OUT / ('preflight_evaluation' if preflight else 'points') / (args.model + '__' + args.sequence + '.json')
    if destination.exists():
        if not read(destination)['complete']:
            raise ValueError('Incomplete actual-point result')
        print('Actual points already complete', args.model, args.sequence, flush=True)
        return
    root = ROOT / ('datasets/dsec_pilot/train' if preflight else 'datasets/dsec_confirmation/test') / args.sequence
    labels = np.load(root / 'object_detections/left/tracks.npy', allow_pickle=False)
    factor_cfg = read(ROOT / cfg['factors_config'])
    gts = {int(t): ground_truth(labels[labels['t'] == t], factor_cfg['class_map'], factor_cfg['small_area_fraction'])[0] for t in np.unique(labels['t'])}
    frozen_thresholds = read(ROOT / 'experiments/confirmation_v01/frozen_thresholds.json')
    original_theta = frozen_thresholds[args.model]['dense_threshold']
    calibration = read(ROOT / 'experiments/fp_mechanism_v01/calibration_points.json')[args.model]
    n = sum(g['height'] * g['width'] for g in m['geometry'])
    cache, floor_fp = [], defaultdict(list)
    start = time.monotonic()
    for q, record in enumerate(m['frames']):
        path = ROOT / record['path']
        if sha(path) != record['sha256']:
            raise ValueError('Compact raw cache hash changed')
        with np.load(path) as z:
            raw = dict(boxes=z['boxes'], scores=z['scores'])
            masks = np.unpackbits(z['masks'], axis=1)[:, :n].astype(bool)
        gt = gts.get(record['time_us'], [])
        f, matched, identities = canonical(raw, masks[0], original_theta, m, gt)
        refs = {gt[gi].track_id: int(f[3][di]) for gi, di in matched.items()}
        cache.append(dict(raw=raw, masks=masks, gt=gt, time_us=record['time_us'], sources=refs))
        memo = {}
        for mi, method in enumerate(cfg['methods']):
            key = masks[mi].tobytes()
            if key not in memo:
                memo[key] = canonical(raw, masks[mi], cfg['confidence_floor'], m, gt)
            ff, mm, ids = memo[key]
            floor_fp[method].extend(float(s) for s in ff[1][ids < 0])
        if (q + 1) % 100 == 0:
            print('Actual FP floors', args.model, args.sequence, q + 1, '/', len(m['frames']), round(time.monotonic() - start, 1), 's', flush=True)
    plans = []
    for method in cfg['methods']:
        for target in cfg['fp_targets']:
            if preflight:
                original = calibration[method + '@' + str(target)]
                plans.append(dict(original))
            else:
                for regime in ['thun_calibrated', 'retrospective_sequence_budget']:
                    theta = (calibration[method + '@' + str(target)]['threshold'] if regime == 'thun_calibrated' else
                             threshold_at_fppi(floor_fp[method], len(cache), target, cfg['confidence_floor']))
                    plans.append(dict(key=regime + '|' + method + '@' + str(target), method=method, target=target,
                         regime=regime, allowed_fp=int(np.floor(target * len(cache))), threshold=theta))
    trace = OUT / ('preflight_evaluation' if preflight else 'points') / args.model / (args.sequence + '.jsonl')
    trace.parent.mkdir(parents=True, exist_ok=True)
    source_trace = OUT / 'source_survival' / (args.model + '__' + args.sequence + '.jsonl')
    source_trace.parent.mkdir(parents=True, exist_ok=True)
    result, attempt, pending = {}, 0, plans
    with trace.open('w') as handle, source_trace.open('w') as source_file:
        while pending:
            totals = {p['key']: defaultdict(int) for p in pending}
            scores = defaultdict(list)
            for q, frame in enumerate(cache):
                raw, gt, masks = frame['raw'], frame['gt'], frame['masks']
                reference = set(frame['sources'])
                memo = {}
                for plan in pending:
                    mi = cfg['methods'].index(plan['method'])
                    mask, theta = masks[mi], plan['threshold']
                    memo_key = (mask.tobytes(), theta)
                    if memo_key not in memo:
                        memo[memo_key] = canonical(raw, mask, theta, m, gt)
                    filtered, matches, labels = memo[memo_key]
                    ids = set(map(int, labels[labels >= 0]))
                    values = dict(queries=1, gt=len(gt), tp=len(matches), fp=len(labels) - len(matches),
                                  dense_tp=len(reference), kept=len(ids & reference), gained=len(ids - reference))
                    for k, value in values.items():
                        totals[plan['key']][k] += value
                    scores[plan['key']].extend(float(s) for s in filtered[1][labels < 0])
                    handle.write(json.dumps(dict(q=q, time_us=frame['time_us'], key=plan['key'], attempt=attempt,
                        threshold=theta, tp=values['tp'], fp=values['fp'], gt=len(gt),
                        dense_ids=sorted(reference), matched_ids=sorted(ids)), separators=(',', ':')) + '\n')
                    if not preflight and plan['target'] == 1.0 and plan['method'] in ['ema/absolute', 'current_activity']:
                        surviving = set(map(int, filtered[3]))
                        matched_anchor = {gt[gi].track_id: int(filtered[3][di]) for gi, di in matches.items()}
                        for identity, anchor in frame['sources'].items():
                            selected = bool(mask[anchor // m['model_provenance']['anchors']])
                            above = bool(raw['scores'][anchor].max() >= theta)
                            effective, detected = selected and above, identity in ids
                            category = ('original_match' if detected and matched_anchor[identity] == anchor else
                                        'admitted_alternative' if detected and effective else
                                        'alternative_after_mask' if detected and not selected else
                                        'alternative_after_confidence' if detected else
                                        'nms_loss' if effective and anchor not in surviving else
                                        'assignment_loss' if effective else 'mask_miss' if not selected else 'confidence_miss')
                            source_file.write(json.dumps(dict(q=q, time_us=frame['time_us'], key=plan['key'], attempt=attempt,
                                identity=identity, method=plan['method'], regime=plan['regime'],
                                original_anchor=anchor, mask_selected=selected, above_threshold=above,
                                original_survives_nms=anchor in surviving, identity_detected=detected,
                                matched_anchor=matched_anchor.get(identity), category=category), separators=(',', ':')) + '\n')
                if (q + 1) % 100 == 0:
                    print('Actual new-record FP replay', args.model, args.sequence, q + 1, '/', len(cache), 'points', len(pending), 'attempt', attempt, flush=True)
            retry = []
            for plan in pending:
                v = dict(totals[plan['key']])
                if plan['regime'] != 'thun_calibrated' and v['fp'] > plan['allowed_fp']:
                    theta = threshold_at_fppi(scores[plan['key']], len(cache), plan['target'], cfg['confidence_floor'])
                    retry.append(dict(plan, threshold=max(theta, float(np.nextafter(np.float32(plan['threshold']), np.float32(np.inf))))))
                    continue
                v.update(recall=v['tp'] / v['gt'] if v['gt'] else None,
                         preservation=v['kept'] / v['dense_tp'] if v['dense_tp'] else None,
                         fppi=v['fp'] / v['queries'],
                         precision=v['tp'] / (v['tp'] + v['fp']) if v['tp'] + v['fp'] else None)
                if preflight:
                    expected = calibration[plan['key']]
                    if any(v[k] != expected[k] for k in ['queries', 'gt', 'tp', 'fp', 'dense_tp', 'kept', 'gained']):
                        raise AssertionError('Protected actual calibration point differs: ' + plan['key'])
                result[plan['key']] = dict(plan, **v, attempt=attempt,
                       under_budget=v['fp'] <= plan['allowed_fp'], target_shortfall=plan['allowed_fp'] - v['fp'],
                       floor_limited=plan['threshold'] == cfg['confidence_floor'] and v['fp'] < plan['allowed_fp'])
            pending, attempt = retry, attempt + 1
            if attempt > 32:
                raise RuntimeError('Actual FP replay did not converge')
    save(destination, dict(complete=True, model=args.model, sequence=args.sequence, conditions=result,
         source_sha256=sha(ROOT / 'code/scripts/evaluate_driving_confirmation.py'),
         trace_sha256=sha(trace), source_trace_sha256=sha(source_trace),
         preflight_protected_points=len(result) if preflight else 0,
         new_data_outputs=not preflight))
    print('Completed actual FP points', args.model, args.sequence, len(result), flush=True)

if __name__ == '__main__':
    main()
