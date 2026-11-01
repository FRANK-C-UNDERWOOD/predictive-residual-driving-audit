"""Label geometry and event eligibility for the bounded adaptation experiment."""
from __future__ import annotations
import numpy as np
from .criterion_audit import ground_truth
from .metrics import KeyObjectEvent


def training_labels(rows, class_map, small_fraction=.01):
    gts,ignored=ground_truth(rows,class_map,small_fraction)
    labels=[]
    for g in gts:
        x1,y1,x2,y2=g.box
        labels.append([g.cls,round(.4*x1),round(.4*y1+32),round(.4*(x2-x1)),round(.4*(y2-y1))])
    tiny=sum(r[3]<1 or r[4]<1 for r in labels)
    return labels,dict(supported=len(gts),ignored=ignored,resized_tiny=tiny)


def eligible_events(tracks, first_evaluated, last_evaluated, horizon=.5, max_gap=.075):
    events,excluded=[],[]
    for (seq,obj),track in tracks.items():
        episodes=[];current=[]
        for t in sorted(track):
            if current and t-current[-1]>max_gap+1e-12:
                episodes.append(current);current=[]
            current.append(t)
        if current: episodes.append(current)
        for episode in episodes:
            onset=episode[0]
            if (onset<=first_evaluated[seq] or episode[-1]-onset<horizon-1e-12
                or onset+horizon>last_evaluated[seq]+1e-12):
                excluded.append(dict(sequence=seq,track_id=obj,onset=onset,
                    reason='left-censored-or-incomplete-visible/input-horizon'))
            else:
                events.append(KeyObjectEvent(seq,obj,onset,episode[-1],
                    meta={'event_id':seq+':'+str(obj)+':'+str(onset)}))
    return events,excluded
