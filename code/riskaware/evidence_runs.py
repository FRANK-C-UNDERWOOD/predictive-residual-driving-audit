"""Sampled loss runs relative to a fixed dense detector, with explicit boundaries."""
from collections import Counter
import numpy as np


def evidence_runs(frames, median_grid_gap_us, gap_factor=1.5):
    if median_grid_gap_us <= 0 or gap_factor < 1:
        raise ValueError('Invalid observation-grid rule')
    previous_t=previous_i=None
    tracks=set()
    for f in frames:
        if previous_t is not None and (f['time_us']<=previous_t or f['grid_index']<=previous_i):
            raise ValueError('Queries must be strictly ordered')
        gt=set(f['gt_ids']);dense=set(f['dense_ids']);selected=set(f['matched_ids'])
        if len(gt)!=len(f['gt_ids']) or not dense<=gt or not selected<=gt:
            raise ValueError('Invalid or duplicated identity membership')
        tracks.update(gt);previous_t=f['time_us'];previous_i=f['grid_index']
    results=[]
    for track in sorted(tracks):
        active=None;previous=None;previous_state='sequence_start'
        for frame in frames:
            state=('gt_absence' if track not in frame['gt_ids'] else
                   'dense_miss' if track not in frame['dense_ids'] else
                   'preserved' if track in frame['matched_ids'] else 'lost')
            gap=previous is not None and (frame['grid_index']!=previous['grid_index']+1 or
                 frame['time_us']-previous['time_us']>gap_factor*median_grid_gap_us)
            if active is not None and (gap or state!='lost'):
                cause='query_gap' if gap else 'recovered' if state=='preserved' else state
                active.update(end_cause=cause,right_censored=cause!='recovered',
                    recovery_time_us=frame['time_us'] if cause=='recovered' else None,
                    observed_to_recovery_us=frame['time_us']-active['first_loss_us'] if cause=='recovered' else None)
                results.append(active);active=None
            if state=='lost':
                if active is None:
                    left='query_gap' if gap else previous_state
                    active=dict(track_id=track,first_loss_us=frame['time_us'],last_loss_us=frame['time_us'],
                                queries=0,observed_span_us=0,left_cause=left,left_censored=left!='preserved')
                active['queries']+=1;active['last_loss_us']=frame['time_us']
                active['observed_span_us']=active['last_loss_us']-active['first_loss_us']
            previous=frame;previous_state=state
        if active is not None:
            active.update(end_cause='sequence_end',right_censored=True,recovery_time_us=None,observed_to_recovery_us=None)
            results.append(active)
    dense=sum(len(f['dense_ids']) for f in frames)
    lost=sum(len(set(f['dense_ids'])-set(f['matched_ids'])) for f in frames)
    if sum(r['queries'] for r in results)!=lost:raise AssertionError('Loss-run partition failed')
    return results,dict(gt_instances=sum(len(f['gt_ids']) for f in frames),dense_instances=dense,
        dense_misses=sum(len(f['gt_ids'])-len(f['dense_ids']) for f in frames),lost_instances=lost,
        runs=len(results),runs_at_least_2=sum(r['queries']>=2 for r in results),
        runs_at_least_3=sum(r['queries']>=3 for r in results),
        longest_queries=max((r['queries'] for r in results),default=0),
        max_observed_span_us=max((r['observed_span_us'] for r in results),default=0),
        observed_recoveries=sum(not r['right_censored'] for r in results),
        right_censored=sum(r['right_censored'] for r in results),
        left_censored=sum(r['left_censored'] for r in results),
        end_causes=dict(Counter(r['end_cause'] for r in results)),
        tracks_with_dense_evidence=len({i for f in frames for i in f['dense_ids']}),
        tracks_with_losses=len({r['track_id'] for r in results}),
        median_grid_gap_us=float(median_grid_gap_us))
