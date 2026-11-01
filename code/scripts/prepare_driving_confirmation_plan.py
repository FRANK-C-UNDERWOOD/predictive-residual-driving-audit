"""Development-only precision calculation, fixed weights and test annotation census."""
from collections import defaultdict
import datetime
import numpy as np
from scipy.stats import t as student_t
from driving_confirmation_common import ROOT, OUT, CONFIG, read, save, sha
from riskaware.driving_factors import annotation_factors

def main():
    if (OUT / 'statistics_plan.json').exists():
        raise FileExistsError('Pre-output statistics plan already fixed')
    cfg = read(CONFIG)
    factors = read(ROOT / cfg['factors_config'])
    pairs = read(ROOT / 'experiments/driving_factors_v01/pairs.json')
    interactions = []
    counts = defaultdict(lambda: dict(low=0, high=0))
    development_sources = {}
    for seq in factors['sequences']:
        selected = [r for r in pairs if r['model'] == 'rvt_stream' and r['sequence'] == seq
                    and r['regime'] == 'retrospective_sequence_budget' and r['target'] == 1
                    and r['method_a'] == 'ema/absolute' and r['method_b'] == 'current_activity']
        values = {r['group']: r['delta_recall_pp'] for r in selected}
        interactions.append(dict(sequence=seq, interaction_pp=values['motion_low'] - values['motion_high']))
        path = ROOT / 'experiments/driving_factors_v01/factors' / (seq + '.jsonl')
        development_sources[path.relative_to(ROOT).as_posix()] = sha(path)
        with path.open() as f:
            for line in f:
                import json
                row = json.loads(line)
                if row['motion'] not in ['motion_low', 'motion_high']:
                    continue
                key = row['size'] + '|' + ('car' if row['cls'] == 0 else 'pedestrian') + '|' + row['stage']
                counts[key]['low' if row['motion'] == 'motion_low' else 'high'] += 1
    active = {k: v for k, v in counts.items() if v['low'] > 0 and v['high'] > 0}
    denominator = sum(v['low'] + v['high'] for v in active.values())
    weights = {k: (v['low'] + v['high']) / denominator for k, v in active.items()}
    values = [r['interaction_pp'] for r in interactions]
    sd = float(np.std(values, ddof=1))
    n = len(cfg['sequences'])
    halfwidth = float(student_t.ppf(.975, n - 1) * sd / np.sqrt(n))
    metadata, populations = {}, []
    for seq in cfg['sequences']:
        folder = ROOT / 'datasets/dsec_confirmation/test' / seq
        grid_path, label_path = folder / 'image_timestamps.txt', folder / 'object_detections/left/tracks.npy'
        grid = np.loadtxt(grid_path, dtype=np.int64)
        labels = np.load(label_path, allow_pickle=False)
        rows, gap = annotation_factors(grid, labels, factors)
        motion = defaultdict(int)
        for row in rows.values():
            motion[row['motion']] += 1
        populations.append(dict(sequence=seq, native_queries=len(grid), eligible_native_gt=len(rows),
                           native_annotation_motion_counts=dict(motion), median_grid_gap_us=gap,
                           role='Annotation-only census; actual full180ms support and warmup cohort fixed after raw event input verification'))
        for p in [grid_path, label_path]:
            metadata[p.relative_to(ROOT).as_posix()] = sha(p)
    save(OUT / 'statistics_plan.json', dict(fixed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         planned_recording_groups=n, selection='Fixed census of seven eligible unused official-test groups, no outcome-dependent extension or replacement',
         development_interactions=interactions, development_recording_sd_pp=sd,
         working_precision_halfwidth_pp=5., projected_n7_t_halfwidth_pp=halfwidth,
         projected_halfwidth_with_1point5_sd_inflation_pp=1.5 * halfwidth,
         precision_scope='Approximation from only four exposed development groups; not a guaranteed power calculation. CensusN7 is feasible available-source scope; inflation shows substantial uncertainty.',
         primary=dict(carrier='rvt_stream', target=1., capacity=.2, estimand='Equal-recording mean of low-minus-high residual/activity recall contrasts',
                      expected_direction='negative', interval='Two-sided95% Student-t across recording-group effects when all seven are defined. Assumes independent/exchangeable group effects; report shared-city/session limitations.',
                      verdict='Supported only if all seven defined and upper95%t limit<0; contrary if lower limit>0; otherwise inconclusive. No practical or safety equivalence claim.',
                      missing='Retain all planned groups; undefined groups make complete-primary verdict inconclusive. Describe defined subset separately, never impute zero.'),
         sensitivity='All effects by recording; equal-city means and leave-one-city-out descriptive analyses. Cities are three shared locations, not seven independent environments. No frame-independent p-values.',
         common_support_weights=weights, development_cell_counts=dict(counts),
         development_common_support_instances=denominator,
         adjustment_scope='Weights fixed only from development annotated low/high populations. Zero-development-support cells are outside this explicitly restricted estimand and still tabulated. Missing positive-weight new cells =>undefined, never silently dropped or renormalized.',
         adjusted_analysis='Report raw cross-cells and per-recording standardized interactions. Also pooled standardized point estimate and5000 recording-cluster bootstrap draws, seed20261031. Publish undefined-draw count; no adjusted CI unless all draws have support. No causal claim from observational adjustment.',
         secondary='Spike size interaction and RVT low-motion source deficits; descriptive full outputs. Four FP targets/two regimes remain reported without selecting a favorable target.',
         subgroup_FP='Whole-recording FP matching only; no subgroup threshold fitting or claim of subgroup FP matching',
         data_metadata_sha256=metadata, development_sources_sha256=development_sources,
         source_sha256=sha(ROOT / 'code/scripts/prepare_driving_confirmation_plan.py'),
         annotation_only_census=populations, new_detector_outputs_read=False))
    print('Prospective statistics/weights fixed. Planned groups', n, 'projected halfwidth', halfwidth, 'active cells', len(weights), flush=True)

if __name__ == '__main__':
    main()
