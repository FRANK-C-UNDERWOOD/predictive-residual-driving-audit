"""Fix the previously specified secondary endpoint before new SNN outputs."""
import datetime
import numpy as np
from scipy.stats import t as student_t
from snn_secondary_common import ROOT, OUT, CONFIG, PARENT, read, save, sha

def main():
    assert not (OUT / 'freeze.json').exists()
    assert not list((OUT / 'carriers').glob('*.json')) and not list((OUT / 'points').glob('*__*.json'))
    cfg = read(CONFIG)
    preflight = {}
    for model in cfg['models']:
        p = OUT / 'preflight' / (model + '__thun_00_a.json')
        m = read(p)
        assert m['complete'] and m['protected_Thun_raw_and_mask_queries'] == 235
        assert len(m['frames']) == 225 and all(r['all_open_exact'] for r in m['alignment'])
        e = OUT / 'preflight_evaluation' / (model + '__thun_00_a.json')
        assert read(e)['preflight_protected_points'] == 36
        preflight[model] = dict(raw=sha(p), calibration=sha(e), raw_queries=235, eval_queries=225, actual_points=36)
    development = read(ROOT / 'experiments/driving_factors_v01/pairs.json')
    planning = {}
    for model in cfg['models']:
        values = []
        for seq in ['zurich_city_07_a', 'zurich_city_08_a', 'zurich_city_09_a', 'zurich_city_10_a']:
            rows = {r['group']: r for r in development if r['model'] == model and r['sequence'] == seq and r['regime'] == 'retrospective_sequence_budget'
                    and r['target'] == 1 and r['method_a'] == 'ema/absolute' and r['method_b'] == 'current_activity'}
            values.append(dict(sequence=seq, interaction_pp=rows['small']['delta_recall_pp'] - rows['large']['delta_recall_pp']))
        sd = float(np.std([r['interaction_pp'] for r in values], ddof=1))
        planning[model] = dict(development=values, sd_pp=sd, nominal_n7_halfwidth_pp=float(student_t.ppf(.975, 6) * sd / np.sqrt(7)))
    plan = dict(fixed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), planned_groups=7,
         estimand='J_size=(R_EMA-R_activity)_small-(R_EMA-R_activity)_large', expected_direction='negative',
         role='Previously specified secondary prediction; RVT seven-group outcomes known before this stage. No new primary trial or external preregistration.',
         endpoint='Complete-GT recall, whole-recording actual1FP/query,20% capacity; small<=1%native area; no subgroup thresholds',
         statistics='Each checkpoint: seven equal recording weights, all per-record effects and nominal two-sided95%t interval. Secondary descriptive intervals not adjusted for two checkpoint tests; no family-level significance verdict.',
         dependence='Two checkpoints of the same architecture and the same seven recordings; no14-group sample or causal architecture claim. Three already encountered cities.',
         missing='Retain missing GT strata as undefined; full planned estimand not established if any group undefined/unmatched. Never zero-impute or silently subset.',
         matching='Use original integer tolerance max(1,floor(allowedFP/100)); report exact actual FP counts, floor limits and failed matching. Whole-recording matching is not subgroup FP matching.',
         all_conditions='All9methods x4FPtargets x2thresholdregimes x2checkpoints x7groups =1008 actual points; original Thun thresholds unchanged. All16marginal+24motion-crosscell groups, source traces, baseline coverage and loss runs retained.',
         planning=planning, precision_scope='Development-only SD from four exposed groups; fixed seven-source census, no guaranteed power or outcome-dependent extension.',
         stateful_reuse='Separate subsequent experiment, not supplied by this stage',
         storage='Reacquire one exact public event member at a time; SHA/CRC/ETag must match sealed primary. Release only exact reacquired file after both checkpoints complete; do not alter parent outputs/old data.')
    save(OUT / 'statistics_plan.json', plan)
    inherited = read(PARENT / 'freeze.json')
    paths = set(inherited['sources'])
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'code/riskaware').glob('*.py'))
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'vendor/spikefpn_snapshot').rglob('*.py'))
    paths.update(['code/configs/snn_secondary_v01.json', 'code/scripts/snn_secondary_common.py', 'code/scripts/cache_snn_secondary.py',
         'code/scripts/evaluate_snn_secondary.py', 'code/scripts/fetch_snn_secondary.py', 'code/scripts/summarize_snn_secondary.py',
         'code/scripts/join_snn_secondary.py', 'code/scripts/run_snn_secondary.py', 'code/scripts/freeze_snn_secondary.py',
         'code/scripts/run_diverse_adaptation.py', 'code/scripts/run_backbone_adaptation.py', 'code/scripts/run_smoothing_adaptation.py',
         'code/scripts/run_readout_adaptation.py', 'code/configs/diverse_adaptation_v01.json', 'docs/39_SNN次级验证执行协议.md',
         'experiments/snn_secondary_v01/statistics_plan.json'])
    inputs = dict(inherited['inputs'])
    for p in [PARENT / 'artifact_manifest.json', PARENT / 'primary_results.json', PARENT / 'execution_scope.json',
              ROOT / 'experiments/driving_factors_v01/pairs.json', ROOT / 'vendor/spikefpn_snapshot/weights/gad_weight.pth',
              ROOT / 'experiments/diverse_adaptation_v01/full_conv_training.json']:
        inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    training = read(ROOT / 'experiments/diverse_adaptation_v01/full_conv_training.json')
    inputs[training['checkpoint']] = training['checkpoint_sha256']
    for model in cfg['models']:
        for folder in ['preflight', 'preflight_evaluation']:
            p = OUT / folder / (model + '__thun_00_a.json')
            inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    for seq in cfg['sequences']:
        p = PARENT / 'data_receipts' / (seq + '.json')
        inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    for path, digest in inputs.items():
        assert sha(ROOT / path) == digest, path
    save(OUT / 'freeze.json', dict(config=cfg, frozen_utc=plan['fixed_utc'], sources={p: sha(ROOT / p) for p in sorted(paths)},
         inputs=inputs, preflight=preflight, new_SNN_outputs_not_started=True, prior_RVT_results_known=True,
         paper_before_sha256=sha(ROOT / 'paper/main.tex'), parent_manifest_sha256=sha(PARENT / 'artifact_manifest.json')))
    print('Secondary execution frozen before new SNN predictions', len(paths), 'sources', flush=True)

if __name__ == '__main__':
    main()
