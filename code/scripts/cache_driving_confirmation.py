"""Pinned detector semantics, compact raw candidates and masks on new records."""
import argparse, time
import numpy as np
import torch
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha, assert_frozen, geometry, compact_masks, ValidationSequence, query_times
from confirmation_common import Scores, variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['rvt_stream', 'spike_original', 'spike_adapted'])
    parser.add_argument('--sequence', required=True)
    args = parser.parse_args()
    cfg, parent = read(CONFIG), read(ROOT / 'code/configs/confirmation_v01.json')
    preflight = args.sequence == 'thun_00_a'
    if not preflight:
        assert_frozen()
        if args.sequence not in cfg['sequences']:
            raise ValueError('Unregistered new sequence')
        receipt = read(OUT / 'data_receipts' / (args.sequence + '.json'))
        for r in [receipt['event'], *receipt['metadata']]:
            if sha(ROOT / r['path']) != r['sha256']:
                raise ValueError('New raw input hash differs')
    dest = OUT / ('preflight' if preflight else 'carriers') / (args.model + '__' + args.sequence + '.json')
    if dest.exists():
        if not read(dest)['complete']:
            raise ValueError('Existing worker is incomplete')
        print('Completed cache already exists', args.model, args.sequence, flush=True)
        return
    shapes = geometry(args.model)
    predictor = Scores(parent, {g['stride']: (g['height'], g['width']) for g in shapes})
    reader = ValidationSequence(args.sequence)
    grid, times = query_times(reader, cfg)
    expected = read(ROOT / 'experiments/confirmation_v01' / (args.model + '__thun_00_a.json')) if preflight else None
    if preflight and times != [r['time_us'] for r in expected['all_frames']]:
        raise AssertionError('Thun input/query convention changed')
    old_masks = None
    if preflight:
        with np.load(ROOT / 'experiments/fp_mechanism_v01/floor' / args.model / 'thun_00_a/masks.npz') as z:
            old_masks = z['masks']
    captured, hooks, states = [], [], None
    if args.model == 'rvt_stream':
        import run_rvt_reference as original
        from cache_confirmation_rvt import clone_state, equal_state
        import torch.nn.functional as F
        net, provenance = original.load_model()
        rep = original.StackedHistogram(10, 480, 640, count_cutoff=255, fastmode=False)
        hooks.append(net.yolox_head.register_forward_pre_hook(lambda mod, a: captured.append([x.detach().clone() for x in a[0]])))
    else:
        from run_diverse_adaptation import selected_model
        from riskaware.torch_spikefpn import NAMES, replay_dense
        net = selected_model('original' if args.model == 'spike_original' else 'full_conv', read(ROOT / 'code/configs/diverse_adaptation_v01.json'))
        provenance = dict(variant=args.model, checkpoint='same pinned original / selected116 adaptation',
                          selected_checkpoint_receipt_sha256=sha(ROOT / 'experiments/diverse_adaptation_v01/full_conv_training.json'))
        captured = {s: [] for s in [8, 16, 32]}
        for s in [8, 16, 32]:
            def hook(mod, a, s=s):
                captured[s].append(a[0].detach().clone())
            hooks.append(getattr(net.feature, NAMES[s][0]).register_forward_pre_hook(hook))
    provenance.update(anchors=1 if args.model == 'rvt_stream' else 3,
                      coordinates='native' if args.model == 'rvt_stream' else 'spike256_normalized')
    rows, alignment, previous_t, regression = [], [], None, 0
    errors = {p: dict(absolute=0., elements=0) for p in parent['predictors']}
    folder = OUT / 'cache' / args.model / args.sequence
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with torch.inference_mode():
        for i, t in enumerate(times):
            diagnostic = i in [0, len(times) // 2, len(times) - 1]
            if args.model == 'rvt_stream':
                duration = 50000 if previous_t is None else t - previous_t
                events, _ = reader.read_window(t, duration)
                tensors = [torch.from_numpy(events[k].astype(np.int64)).cuda() for k in ['x', 'y', 'p', 't']]
                x = F.pad(rep.construct(*tensors).float().unsqueeze(0), (0, 0, 0, 32))
                captured.clear()
                previous_states = clone_state(states) if diagnostic else None
                output, _, next_states = net(x, previous_states=states)
                feature = list(captured[0])
                if len(captured) != 1:
                    raise AssertionError('RVT seam capture differs')
                if diagnostic:
                    head, _ = net.yolox_head(feature)
                    head_exact = torch.equal(output, head)
                    net(torch.zeros_like(x), previous_states=None)
                    replay, _, replay_states = net(x, previous_states=previous_states)
                    state_exact = torch.equal(output, replay) and equal_state(next_states, replay_states)
                    alignment.append(dict(time_us=t, all_open_exact=head_exact, explicit_state_replay_exact=state_exact))
                    if not head_exact or not state_exact:
                        raise AssertionError('RVT head/state alignment failed')
                states, previous_t = next_states, t
                h = {s: np.ascontiguousarray(v[0].permute(1, 2, 0).cpu().numpy().reshape(-1, v.shape[1])) for s, v in zip([8, 16, 32], feature)}
                counts = {s: np.bincount((events['y'].astype(np.int64) // s) * (640 // s) + events['x'].astype(np.int64) // s,
                                         minlength=g['height'] * g['width']) for s, g in zip([8, 16, 32], shapes)}
                out = output[0].cpu().numpy()
                boxes = np.concatenate((out[:, :2] - out[:, 2:4] / 2, out[:, :2] + out[:, 2:4] / 2), 1)
                raw = out[:, 4:5] * out[:, 5:8]
                classes = raw.argmax(1)
                scores = np.zeros((len(out), 2), np.float32)
                valid = classes != 1
                scores[np.flatnonzero(valid), np.where(classes[valid] == 2, 0, 1)] = raw[valid, classes[valid]]
            else:
                for v in captured.values():
                    v.clear()
                x, _, counts = reader.signed_stack(t, return_counts=True)
                output = net(torch.from_numpy(x).cuda())
                shared = {s: list(captured[s]) for s in [8, 16, 32]}
                if any(len(v) != 3 for v in shared.values()):
                    raise AssertionError('Three native internal-step tensors required')
                h = {}
                for s, v in shared.items():
                    z = torch.stack(v)[:, 0].cpu().numpy()
                    h[s] = np.ascontiguousarray(z.mean(0).transpose(1, 2, 0).reshape(-1, z.shape[1]))
                boxes = output[3].clamp(0, 1)[0].cpu().numpy()
                scores = (output[1].softmax(-1) * output[0].sigmoid())[0].cpu().numpy()
                if diagnostic:
                    replay, _, _ = replay_dense(net, shared)
                    exact = all(torch.equal(a, b) for a, b in zip(output, replay))
                    alignment.append(dict(time_us=t, all_open_exact=exact))
                    if not exact:
                        raise AssertionError('Spike full-open graph alignment failed')
            prior = predictor.history[-1] if predictor.history else h
            arrays, error = predictor.observe(h, counts, 0, int(np.searchsorted(grid, t)))
            packed = compact_masks(h, predictor, arrays, variants(parent), cfg, shapes, prior)
            if i >= cfg['warmup']:
                for name, values in error.items():
                    for k, value in values.items():
                        errors[name][k] += value
                if preflight:
                    with np.load(ROOT / expected['frames'][i - cfg['warmup']]['path']) as z:
                        if not np.array_equal(boxes, z['boxes']) or not np.array_equal(scores, z['scores']):
                            raise AssertionError('Protected Thun raw predictions differ')
                    if not np.array_equal(packed, old_masks[i - cfg['warmup']]):
                        raise AssertionError('Protected Thun nine-method masks differ')
                    regression += 1
            path = folder / (str(t) + '.npz')
            np.savez_compressed(path, boxes=boxes, scores=scores, masks=packed, time_us=t)
            rows.append(dict(time_us=t, path=path.relative_to(ROOT).as_posix(), sha256=sha(path), warmup=i < cfg['warmup']))
            if (i + 1) % 50 == 0:
                print('Compact cache', args.model, args.sequence, i + 1, '/', len(times), round(time.monotonic() - started, 1), 's', flush=True)
    for hook in hooks:
        hook.remove()
    reader.close()
    save(dest, dict(complete=True, model=args.model, sequence=args.sequence, geometry=shapes,
         model_provenance=provenance, all_frames=rows, frames=rows[cfg['warmup']:],
         prediction_error=errors, alignment=alignment, protected_Thun_raw_and_mask_queries=regression,
         cache_scope='Raw candidates and all nine exact masks; no new features or state intervention',
         execution_sources={p: sha(ROOT / p) for p in ['code/scripts/cache_driving_confirmation.py',
              'code/scripts/driving_confirmation_common.py', 'code/scripts/confirmation_common.py', 'code/riskaware/dsec_pilot.py']}))
    print('Completed compact carrier', args.model, args.sequence, 'evaluated', len(rows) - cfg['warmup'], flush=True)

if __name__ == '__main__':
    main()
