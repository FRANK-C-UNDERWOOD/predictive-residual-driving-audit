"""Explicit pre-smooth feature refresh/hold; this does not skip a physical kernel."""
import torch

class FeatureReuse:
    def __init__(self):
        self.state, self.ages, self.last_query = {}, {}, None

    def reset(self):
        self.state.clear(); self.ages.clear(); self.last_query = None

    def update(self, fresh, masks, query):
        if self.last_query is not None and query <= self.last_query:
            raise ValueError('Queries must advance; reset at a new sequence')
        if set(fresh) != set(masks) or (self.state and set(fresh) != set(self.state)):
            raise ValueError('Fixed scales required')
        result = {}
        for stride, values in fresh.items():
            stacked = torch.stack(values).detach()
            if stacked.ndim != 5:
                raise ValueError('Expected internal-step,batch,channel,height,width')
            mask = masks[stride]
            if mask.dtype != torch.bool or tuple(mask.shape) != tuple(stacked.shape[-2:]) or mask.device != stacked.device:
                raise ValueError('Boolean spatial masks on the feature device required')
            if stride not in self.state:
                if not bool(mask.all()):
                    raise ValueError('Initialize every region explicitly before selective updates')
                self.state[stride] = stacked.clone()
                self.ages[stride] = torch.zeros_like(mask, dtype=torch.int64)
            else:
                if self.state[stride].shape != stacked.shape or self.state[stride].dtype != stacked.dtype:
                    raise ValueError('Feature geometry/dtype changed')
                self.state[stride] = torch.where(mask[None,None,None], stacked, self.state[stride])
                self.ages[stride] = torch.where(mask, 0, self.ages[stride] + 1)
            result[stride] = list(self.state[stride].unbind(0))
        self.last_query = query
        return result
