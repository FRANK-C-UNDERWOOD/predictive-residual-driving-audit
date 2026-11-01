"""Causal predictor-by-normalizer controls; no detector outputs or GT inputs."""
from collections import deque
import numpy as np
from .criterion_audit import SCALES

class MechanismScores:
    def __init__(self,alpha=.8,lag=3):
        self.alpha=alpha;self.lag=lag;self.history=deque(maxlen=lag+1);self.previous=None
    def observe(self,features,counts,seed,sequence_index,frame_index):
        h={s:np.ascontiguousarray(v,dtype=np.float32) for s,v in features.items()}
        prior=self.history[-1] if self.history else {s:v.copy() for s,v in h.items()}
        delayed=self.history[-(self.lag+1)] if len(self.history)>self.lag else (self.history[0] if self.history else prior)
        previous=self.previous if self.previous is not None else h
        scores={};mae={};zero={}
        for s,x in h.items():
            n,c=x.shape;side=int(round(np.sqrt(n)))
            if side*side!=n:raise ValueError('Expected square per-scale feature map')
            rolled=np.roll(prior[s].reshape(side,side,c),(side//3,side//3),axis=(0,1)).reshape(n,c)
            broadcast=np.broadcast_to(prior[s].mean(0,keepdims=True),x.shape)
            predictors=dict(zero=np.zeros_like(x),persistence=previous[s],ema=prior[s],ema_lag3=delayed[s],ema_roll=rolled,ema_global=broadcast)
            magnitude=np.abs(x).sum(1);zero[s]=magnitude==0
            for name,p in predictors.items():
                delta=np.abs(x-p);numerator=delta.sum(1)
                values=dict(absolute=delta.mean(1),current=numerator/(magnitude+1e-8),
                    symmetric=numerator/(magnitude+np.abs(p).sum(1)+1e-8))
                for norm,v in values.items():scores.setdefault(name+'/'+norm,{})[s]=v
                error=mae.setdefault(name,dict(absolute=0.,elements=0));error['absolute']+=float(delta.sum(dtype=np.float64));error['elements']+=x.size
            refs=dict(current_activity=np.abs(x).mean(1),ema_activity=np.abs(prior[s]).mean(1),event_rate=counts[s],
                random=np.random.default_rng(seed+sequence_index*100000+frame_index*100+s).random(n))
            for name,v in refs.items():scores.setdefault(name,{})[s]=v
        # No current observation enters the predictions used above.
        self.history.append({s:self.alpha*prior[s]+(1-self.alpha)*x for s,x in h.items()})
        self.previous={s:x.copy() for s,x in h.items()}
        return scores,mae,zero
