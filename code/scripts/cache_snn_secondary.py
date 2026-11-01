"""Exact batch-one SNN forwards; share raw input reading across two checkpoints."""
import argparse, time
import numpy as np
import torch
from snn_secondary_common import ROOT, OUT, CONFIG, PARENT, read, save, sha, assert_frozen, activate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    common = activate()
    from confirmation_common import Scores, variants
    from run_diverse_adaptation import selected_model
    from riskaware.torch_spikefpn import NAMES, replay_dense
    cfg, parent = read(CONFIG), read(ROOT / 'code/configs/confirmation_v01.json')
    if args.preflight:
        assert args.sequence == 'thun_00_a'
    else:
        assert_frozen()
        assert args.sequence in cfg['sequences']
        receipt = read(OUT / 'data_receipts' / (args.sequence + '.json'))
        for r in [receipt['event'], *receipt['metadata']]:
            assert sha(ROOT / r['path']) == r['sha256']
    dests = {model: OUT / ('preflight' if args.preflight else 'carriers') / (model + '__' + args.sequence + '.json') for model in cfg['models']}
    complete = [p.exists() and read(p)['complete'] for p in dests.values()]
    if all(complete):
        print('Both SNN caches already complete', args.sequence, flush=True)
        return
    assert not any(complete), 'Partial pair needs explicit reconciliation; never overwrite completed receipts'
    reader = common.ValidationSequence(args.sequence)
    grid, times = common.query_times(reader, cfg)
    shapes = common.geometry('spike_original')
    models, captured, hooks, predictors, rows, align, errors, expected = {}, {}, [], {}, {}, {}, {}, {}
    for model in cfg['models']:
        models[model] = selected_model('original' if model == 'spike_original' else 'full_conv', read(ROOT / 'code/configs/diverse_adaptation_v01.json'))
        captured[model] = {s: [] for s in [8, 16, 32]}
        for s in [8, 16, 32]:
            def hook(mod, a, model=model, s=s):
                captured[model][s].append(a[0].detach().clone())
            hooks.append(getattr(models[model].feature, NAMES[s][0]).register_forward_pre_hook(hook))
        predictors[model] = Scores(parent, {g['stride']: (g['height'], g['width']) for g in shapes})
        rows[model], align[model] = [], []
        errors[model] = {p: dict(absolute=0., elements=0) for p in parent['predictors']}
        if args.preflight:
            expected[model] = read(PARENT / 'preflight' / (model + '__thun_00_a.json'))
            assert times == [r['time_us'] for r in expected[model]['all_frames']]
    started = time.monotonic()
    with torch.inference_mode():
        for i, t in enumerate(times):
            # Input encoding is identical, read once; both models still forward separately.
            x, _, counts = reader.signed_stack(t, return_counts=True)
            xx = torch.from_numpy(x).cuda()
            for model, net in models.items():
                for v in captured[model].values():
                    v.clear()
                output = net(xx)
                shared = {s: list(captured[model][s]) for s in [8, 16, 32]}
                assert all(len(v) == 3 for v in shared.values())
                h = {}
                for s, v in shared.items():
                    z = torch.stack(v)[:, 0].cpu().numpy()
                    h[s] = np.ascontiguousarray(z.mean(0).transpose(1, 2, 0).reshape(-1, z.shape[1]))
                boxes = output[3].clamp(0, 1)[0].cpu().numpy()
                scores = (output[1].softmax(-1) * output[0].sigmoid())[0].cpu().numpy()
                if i in [0, len(times) // 2, len(times) - 1]:
                    replay, _, _ = replay_dense(net, shared)
                    exact = all(torch.equal(a, b) for a, b in zip(output, replay))
                    assert exact, 'Full-open branch replay failed'
                    align[model].append(dict(time_us=t, all_open_exact=exact))
                predictor = predictors[model]
                prior = predictor.history[-1] if predictor.history else h
                arrays, error = predictor.observe(h, counts, 0, int(np.searchsorted(grid, t)))
                packed = common.compact_masks(h, predictor, arrays, variants(parent), cfg, shapes, prior)
                if i >= cfg['warmup']:
                    for name, values in error.items():
                        for k, value in values.items():
                            errors[model][name][k] += value
                if args.preflight:
                    with np.load(ROOT / expected[model]['all_frames'][i]['path']) as old:
                        assert np.array_equal(boxes, old['boxes']) and np.array_equal(scores, old['scores'])
                        assert np.array_equal(packed, old['masks'])
                path = OUT / 'cache' / model / args.sequence / (str(t) + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    # A restarted incomplete pair must reproduce already written arrays.
                    with np.load(path) as old:
                        assert all(np.array_equal(value, old[key]) for key, value in [('boxes', boxes), ('scores', scores), ('masks', packed)])
                else:
                    np.savez_compressed(path, boxes=boxes, scores=scores, masks=packed, time_us=t)
                rows[model].append(dict(time_us=t, path=path.relative_to(ROOT).as_posix(), sha256=sha(path), warmup=i < cfg['warmup']))
            if (i + 1) % 25 == 0:
                elapsed = time.monotonic() - started
                save(OUT / 'cache_progress.json', dict(sequence=args.sequence, queries=i + 1, total=len(times), both_models=True, elapsed_s=elapsed))
                print('SNN pair', args.sequence, i + 1, '/', len(times), round(elapsed, 1), 's', flush=True)
    for hook in hooks:
        hook.remove()
    reader.close()
    for model in cfg['models']:
        save(dests[model], dict(complete=True, model=model, sequence=args.sequence, geometry=shapes,
             model_provenance=dict(variant=model, checkpoint='same pinned original / selected116 adaptation', anchors=3,
                 coordinates='spike256_normalized', state='reset every180ms query; predictor full-grid across queries'),
             all_frames=rows[model], frames=rows[model][cfg['warmup']:], prediction_error=errors[model], alignment=align[model],
             protected_Thun_raw_and_mask_queries=len(times) if args.preflight else 0,
             cache_scope='Each original batch-one SNN forward; only identical input reading shared; no feature/output reuse',
             execution_sources={p: sha(ROOT / p) for p in ['code/scripts/cache_snn_secondary.py', 'code/scripts/snn_secondary_common.py']}))
    print('Complete SNN pair', args.sequence, 'evaluated', len(times) - cfg['warmup'], flush=True)

if __name__ == '__main__':
    main()
