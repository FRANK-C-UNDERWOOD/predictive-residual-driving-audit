"""Offline, detector-independent size-onset diagnosis in integer timestamps."""


def size_onsets(tracks, first_evaluated, last_evaluated, minimum_input_side,
                resize_ratio=.4,horizon_us=500000,max_gap_us=75000):
    if minimum_input_side<0 or resize_ratio<=0: raise ValueError('Invalid size convention')
    events=[];excluded=[]
    for (seq,obj),track in tracks.items():
        episodes=[];block=[]
        for t in sorted(track):
            if block and t-block[-1]>max_gap_us: episodes.append(block);block=[]
            block.append(t)
        if block: episodes.append(block)
        for block in episodes:
            candidates=[]
            for t in block:
                x1,y1,x2,y2=track[t].box
                if min(x2-x1,y2-y1)*resize_ratio>=minimum_input_side: candidates.append(t)
            if not candidates:
                excluded.append(dict(sequence=seq,track_id=obj,reason='never-size-eligible'));continue
            onset=candidates[0]
            if onset<=first_evaluated[seq] or block[-1]<onset+horizon_us or last_evaluated[seq]<onset+horizon_us:
                excluded.append(dict(sequence=seq,track_id=obj,onset_us=onset,reason='left-censored-or-incomplete-horizon'));continue
            x1,y1,x2,y2=track[onset].box
            events.append(dict(sequence=seq,track_id=obj,onset_us=onset,end_us=block[-1],episode_start_us=block[0],
                small_at_onset=(x2-x1)*(y2-y1)/(640*480)<=.01))
    return events,excluded


def new_run_delay_us(times,hits,obj,onset_us,horizon_us=500000,max_gap_us=75000,run_length=3):
    run=0;previous=None
    for t in times:
        if t<onset_us: continue
        if t>onset_us+horizon_us: break
        if previous is not None and t-previous>max_gap_us: run=0
        run=run+1 if obj in hits.get(t,set()) else 0
        previous=t
        if run>=run_length: return t-onset_us
    return None
