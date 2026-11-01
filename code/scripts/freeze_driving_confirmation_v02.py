"""Freeze exact input identities, estimands and execution before new predictions."""
import datetime
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha

def main():
    if (OUT / 'freeze.json').exists():
        raise FileExistsError('New-recording execution is already frozen')
    if list((OUT / 'carriers').glob('*.json')) or list((OUT / 'points').glob('*__*.json')):
        raise ValueError('Cannot claim a pre-output freeze after new outputs')
    cfg, plan = read(CONFIG), read(OUT / 'statistics_plan.json')
    assert plan['planned_recording_groups'] == len(cfg['sequences']) == 7
    preflight = {}
    for model in cfg['models']:
        path = OUT / 'preflight' / (model + '__thun_00_a.json')
        m = read(path)
        assert m['protected_Thun_raw_and_mask_queries'] == 225
        assert all(r['all_open_exact'] and r.get('explicit_state_replay_exact', True) for r in m['alignment'])
        preflight[model] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), queries=225)
    assert read(OUT / 'preflight_evaluation/rvt_stream__thun_00_a.json')['preflight_protected_points'] == 36
    scope = dict(stage='primary_RVT_seven_complete_recordings', models_executed_now=['rvt_stream'],
         secondary_models_pending=['spike_original', 'spike_adapted'],
         scope_fixed_before_new_outputs=True,
         reason='Execute the declared primary RVT interaction first. Three-carrier old-data alignment passed; SNN secondary new-recording predictions are not reported as executed.',
         storage='Only new public event members under datasets/dsec_confirmation/test may be removed after a complete primary raw-cache and actual-FP receipt exists, with exact member SHA/CRC/URL archived for reacquisition. Old datasets/results/checkpoints never removed. No duplicate multi-GBZIP retained.')
    save(OUT / 'execution_scope.json', scope)
    sources = [
        'code/configs/driving_confirmation_v02_run.json', 'code/configs/driving_confirmation_v02.json',
        'code/configs/driving_factors_v01.json', 'code/scripts/driving_confirmation_common.py',
        'code/scripts/cache_driving_confirmation.py', 'code/scripts/evaluate_driving_confirmation.py',
        'code/scripts/summarize_driving_confirmation.py', 'code/scripts/run_driving_confirmation_v02.py',
        'code/scripts/fetch_driving_confirmation_events.py', 'code/scripts/freeze_driving_confirmation_v02.py',
        'code/scripts/prepare_driving_confirmation_plan.py', 'code/riskaware/validation_estimands.py',
        'code/tests/test_validation_estimands.py', 'code/riskaware/driving_factors.py',
        'code/scripts/confirmation_common.py', 'code/scripts/run_rvt_reference.py',
        'code/scripts/cache_confirmation_rvt.py', 'code/scripts/run_fp_mechanism.py',
        'code/scripts/confirmation_evaluate.py', 'code/riskaware/criterion_audit.py',
        'code/riskaware/metrics.py', 'code/riskaware/calibration_audit.py', 'code/riskaware/evidence_runs.py',
        'code/riskaware/dsec_pilot.py', 'vendor/spikefpn_snapshot/utils/tools.py',
        'code/configs/rvt_reference_v01.json', 'code/configs/confirmation_v01.json',
        'experiments/driving_confirmation_v02/statistics_plan.json',
        'experiments/driving_confirmation_v02/selection.json', 'experiments/driving_confirmation_v02/execution_scope.json']
    inputs = dict(plan['data_metadata_sha256'])
    inputs.update(plan['development_sources_sha256'])
    for path in [OUT / 'asset_inspection.json', OUT / 'extra_events_members.json', OUT / 'extra_images_members.json',
                 OUT / 'preflight_evaluation/rvt_stream__thun_00_a.json',
                 ROOT / 'experiments/fp_mechanism_v01/calibration_points.json',
                 ROOT / 'experiments/confirmation_v01/frozen_thresholds.json',
                 ROOT / 'experiments/driving_factors_v01/artifact_manifest.json']:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for model in cfg['models']:
        inputs[preflight[model]['path']] = preflight[model]['sha256']
    save(OUT / 'freeze.json', dict(config=cfg, frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         sources={p: sha(ROOT / p) for p in sources}, inputs=inputs, preflight=preflight,
         primary_new_outputs_not_started=True, new_annotation_metadata_already_inspected=True,
         local_freeze_not_external_preregistration=True, paper_before_sha256=sha(ROOT / 'paper/main.tex'),
         original_checkpoint_sha256=sha(ROOT / 'vendor/spikefpn_snapshot/weights/gad_weight.pth'),
         rvt_checkpoint_sha256=sha(ROOT / 'vendor/rvt_snapshot/weights/rvt-t-1mpx.ckpt'),
         adapted_training_receipt_sha256=sha(ROOT / 'experiments/diverse_adaptation_v01/full_conv_training.json')))
    print('Execution frozen before new predictions: seven named groups; primaryRVT; SNNsecondary pending', flush=True)

if __name__ == '__main__':
    main()
