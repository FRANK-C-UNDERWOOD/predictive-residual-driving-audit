"""Identity-based allocation audit and GT-free exchange sampling."""
import hashlib
import numpy as np
from .criterion_audit import SCALES

def evidence_partition(dense,chosen,valid):
    dense,chosen,valid=set(dense),set(chosen),set(valid)
    if not dense<=valid or not chosen<=valid:raise ValueError('Matched identity outside valid GT population')
    kept=dense&chosen;lost=dense-chosen;gained=chosen-dense
    return dict(dense_tp=len(dense),tp=len(chosen),kept=len(kept),lost=len(lost),gained=len(gained),
        net_tp=len(gained)-len(lost),preservation=len(kept)/len(dense) if dense else None,
        lost_ids=sorted(lost),gained_ids=sorted(gained),kept_ids=sorted(kept))

class CausalScores:
    def __init__(self,alpha=.8):
        self.alpha=alpha;self.pred={};self.previous={}
    def observe(self,features,counts,seed,sequence_index,frame_index):
        # Legacy replay used npz_array.copy(): a C-contiguous float32 reduction.
        # Layout affects channel-sum rounding and top-k ties; make it explicit.
        features={s:np.ascontiguousarray(h,dtype=np.float32) for s,h in features.items()}
        result={p:{} for p in ['random','event_rate','current_activity','past_activity','residual_ema','residual_persistence']}
        error=dict(ema=0.,persistence=0.,zero=0.,elements=0)
        for s in SCALES:
            h=np.asarray(features[s])
            if s not in self.pred:self.pred[s]=h.copy();self.previous[s]=h.copy()
            pred,previous=self.pred[s],self.previous[s];denom=np.abs(h).sum(1)+1e-8
            rng=np.random.default_rng(seed+sequence_index*100000+frame_index*100+s)
            values=dict(random=rng.random(len(h)),event_rate=counts[s],current_activity=np.abs(h).mean(1),
                past_activity=np.abs(previous).mean(1),residual_ema=np.abs(h-pred).sum(1)/denom,
                residual_persistence=np.abs(h-previous).sum(1)/denom)
            for p in result:result[p][s]=values[p]
            error['ema']+=float(np.abs(h-pred).sum(dtype=np.float64));error['persistence']+=float(np.abs(h-previous).sum(dtype=np.float64))
            error['zero']+=float(np.abs(h).sum(dtype=np.float64));error['elements']+=h.size
        # Current scores above use only prior prediction states; commit all observations last.
        for s,h in features.items():
            self.pred[s]=self.alpha*self.pred[s]+(1-self.alpha)*h
            self.previous[s]=h.copy()
        return result,error

def deterministic_rng(seed,*parts):
    digest=hashlib.sha256(('|'.join(map(str,(seed,)+parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'little'))

def exchange_pairs(selected,rank,current,stratum,count,pool_size,rng):
    selected=np.asarray(selected,dtype=bool);rank=np.asarray(rank);current=np.asarray(current)
    if len(rank)!=len(selected) or len(current)!=len(selected):raise ValueError('Geometry mismatch')
    outgoing=np.flatnonzero(selected);incoming=np.flatnonzero(~selected)
    if stratum=='rank_boundary':
        outgoing=outgoing[np.lexsort((outgoing,rank[outgoing]))[:pool_size]]
        incoming=incoming[np.lexsort((incoming,-rank[incoming]))[:pool_size]]
    elif stratum=='current_evidence':
        outgoing=outgoing[np.lexsort((outgoing,current[outgoing]))[:pool_size]]
        incoming=incoming[np.lexsort((incoming,-current[incoming]))[:pool_size]]
    elif stratum!='uniform':raise ValueError('Unknown sampling stratum')
    n=len(outgoing)*len(incoming)
    if n==0:return []
    indices=rng.choice(n,size=min(count,n),replace=False)
    return [(int(outgoing[v//len(incoming)]),int(incoming[v%len(incoming)])) for v in indices]

def swap(mask,outgoing,incoming,offsets):
    mask=np.asarray(mask,dtype=bool)
    if not 0<=outgoing<len(mask) or not 0<=incoming<len(mask):raise ValueError('Index out of bounds')
    if not mask[outgoing] or mask[incoming]:raise ValueError('Exchange must remove selected and admit omitted')
    level=lambda i: int(np.searchsorted(offsets[1:],i,side='right'))
    if level(outgoing)!=level(incoming):raise ValueError('Cross-scale exchange forbidden')
    changed=mask.copy();changed[outgoing]=False;changed[incoming]=True
    return changed
