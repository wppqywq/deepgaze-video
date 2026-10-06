"""Population gaze-history GRU and Gaussian-mixture cell probabilities."""
import numpy as np
import torch
from torch import nn

class Population(nn.Module):
    def __init__(self, history_dimensions=5):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(2048,32), nn.Tanh())
        self.context = nn.Sequential(nn.Linear(4*34,32), nn.Tanh())
        self.gru = nn.GRUCell(history_dimensions+32,32)
        self.head = nn.Linear(32,25)

    def forward(self, h, hm, v, vm, vt):
        encoded = self.visual(v)*vm[...,None]
        context = self.context(torch.cat([encoded, vm[...,None], vt[...,None]],-1).flatten(1))
        hidden = torch.zeros(len(h),32,device=h.device)
        for i in range(4):
            update = self.gru(torch.cat([h[:,i],context],-1),hidden)
            hidden = torch.where(hm[:,i,None].bool(),update,hidden)
        return self.head(hidden).reshape(-1,5,5)


def parameters(output):
    return output[...,0].softmax(-1), output[...,1:3].sigmoid(), .02+.48*output[...,3:5].sigmoid()


def cell_log_mass(output, cells):
    # Log-domain tails prevent a narrow mixture assigning numerical zero mass.
    output = output.double()
    weights, means, scales = parameters(output)
    def interval(lo,hi):
        a=(lo-means)/scales; b=(hi-means)/scales
        low, high = torch.where(a > 0, -b, a), torch.where(a > 0, -a, b)
        lower, upper = torch.special.log_ndtr(low), torch.special.log_ndtr(high)
        return upper + torch.log(-torch.expm1(lower-upper))
    cells = cells.double()
    mass = interval(cells[:,None,:]/100,(cells[:,None,:]+1)/100).sum(-1)
    screen = interval(torch.zeros_like(means),torch.ones_like(means)).sum(-1)
    log_weights = output[...,0].log_softmax(-1)
    return torch.logsumexp(log_weights+mass,-1)-torch.logsumexp(log_weights+screen,-1)


def tensors(rows, cache):
    h=np.zeros((len(rows),4,5),np.float32); hm=np.zeros((len(rows),4),np.float32)
    v=np.zeros((len(rows),4,2048),np.float32); vm=hm.copy(); vt=hm.copy()
    for n,r in enumerate(rows):
        for i,e in enumerate(r['input']['history']):
            if e is not None:
                h[n,i]=e['xy']+[e['start_relative_seconds'],e['end_relative_seconds'],e['raw_relative_seconds']];hm[n,i]=1
        for i,f in enumerate(r['input']['frames']):
            if f is not None:
                v[n,i]=cache[(r['video'],f['index'])];vm[n,i]=1;vt[n,i]=f['relative_seconds']
    train=np.array([r['split']=='train' for r in rows])
    history=h[train][hm[train].astype(bool)]; visual=v[train][vm[train].astype(bool)]
    norm={'hm':history.mean(0),'hs':np.maximum(history.std(0),1e-3),
          'vm':visual.mean(0),'vs':np.maximum(visual.std(0),1e-3)}
    h=((h-norm['hm'])/norm['hs'])*hm[...,None]
    v=((v-norm['vm'])/norm['vs'])*vm[...,None]
    arrays=[h,hm,v,vm,vt,np.array([r['target']['cell'] for r in rows],np.float32)]
    return [torch.from_numpy(a) for a in arrays],norm,train
