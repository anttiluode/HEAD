"""Matched continuous HEAD: center-contrast signal and a potentially sign-trapped decoder.

No 8-cue table, no probe. Each query is continuous on S^1 and indicates which of
8 correlated coordinates carries the signal. The forward pass sees all values.
The auxiliary is batch-local and turns off after an early training window.

Run: python continuous_centered.py --json centered.json
"""
import argparse
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from continuous_unsigned import get_true_keys, conditional_correlation

SLOTS = 8

class CenteredHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(SLOTS, 2)*.10)
        self.output = nn.Linear(2,1)
    def forward(self,q,x):
        first=x.mean(1,keepdim=True)
        logits=(q@self.keys.T)*6
        second=(logits.softmax(1)*x).sum(1,keepdim=True)
        return self.output(torch.cat((first,second),1)).squeeze(1),logits

def sample(n, true_keys, generator, polarity):
    angle=torch.rand(n,generator=generator)*2*math.pi
    q=torch.stack((angle.cos(),angle.sin()),1)
    noise=torch.randn(n,SLOTS,generator=generator)
    detail=noise-noise.mean(1,keepdim=True)
    c=torch.randn(n,1,generator=generator)
    x=c+detail
    chosen=(q @ true_keys.T).argmax(1)
    target=polarity*detail.gather(1,chosen[:,None]).squeeze(1)
    return q,x,target,chosen

@torch.no_grad()
def teacher(mode,q,x,y,pred,rng):
    centered=x-x.mean(1,keepdim=True)
    if mode=='instant':
        return centered.abs().argmax(1)
    response=(y-pred).detach()
    if mode in ('target','signed_target'):
        response=y.detach()
    if mode=='shuffled':
        response=response[torch.randperm(len(response),generator=rng)]
    scores=conditional_correlation(q,centered,response)
    if mode in ('signed','signed_target'):
        sim=20*(q@q.T)
        sim.fill_diagonal_(-1e9)
        w=sim.softmax(1)
        mu_v=w@centered
        mu_r=w@response
        cov=w@(centered*response[:,None])-mu_v*mu_r[:,None]
        scores=cov.sign()*scores.sqrt()
    return scores.argmax(1)

@torch.no_grad()
def evaluation(model,seed,truth,polarity,n=8192):
    q,x,y,j=sample(n,truth,torch.Generator().manual_seed(17000+seed),polarity)
    pred,logits=model(q,x)
    return {'route_acc':(logits.argmax(1)==j).float().mean().item(),
            'nmse':((pred-y).square().mean()/y.square().mean()).item(),
            'second_weight':model.output.weight[0,1].item()}

def run(seed,mode,polarity=1,steps=1400,window=350,weight=.25,batch=192):
    truth=get_true_keys(seed)
    torch.manual_seed(seed)
    model=CenteredHead()
    w0=model.output.weight[0,1].item()
    opt=torch.optim.Adam(model.parameters(),lr=.012)
    batches=torch.Generator().manual_seed(7000+seed)
    rng=torch.Generator().manual_seed(67000+seed)
    for step in range(1,steps+1):
        q,x,y,_=sample(batch,truth,batches,polarity)
        pred,logits=model(q,x)
        loss=F.mse_loss(pred,y)
        if mode!='baseline' and step<=window:
            pseudo=teacher(mode,q,x,y,pred,rng)
            loss=loss+weight*F.cross_entropy(logits,pseudo)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step==window:
            cut=evaluation(model,seed,truth,polarity,2048)
    return {'seed':seed,'mode':mode,'polarity':polarity,'initial_second_weight':w0,
            'at_cut':cut,'final':evaluation(model,seed,truth,polarity)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5,6,7,8,9,10])
    ap.add_argument('--modes',nargs='+',default=['baseline','residual','target','shuffled','instant'],
                    choices=['baseline','residual','target','shuffled','instant','signed','signed_target'])
    ap.add_argument('--polarities',type=int,nargs='+',default=[1,-1])
    ap.add_argument('--steps',type=int,default=1400)
    ap.add_argument('--window',type=int,default=350)
    ap.add_argument('--batch',type=int,default=192)
    ap.add_argument('--weight',type=float,default=.25)
    ap.add_argument('--json',type=Path)
    a=ap.parse_args()
    torch.set_num_threads(1)
    rows=[]
    for polarity in a.polarities:
        for mode in a.modes:
            for seed in a.seeds:
                row=run(seed,mode,polarity,a.steps,a.window,a.weight,a.batch)
                rows.append(row)
                print(polarity,mode,seed,'start',f"{row['initial_second_weight']:+.3f}",
                      'route',f"{row['final']['route_acc']:.3f}",
                      'nmse',f"{row['final']['nmse']:.4f}",flush=True)
    data={'design':'HEAD centered values with 2D continuous routing queries; no cue table or extra coordinate probes',
          'torch':torch.__version__,'steps':a.steps,'batch':a.batch,'window':a.window,'weight':a.weight,
          'passes_if':'route_acc >= .95 AND nmse < .10', 'rows':rows}
    if a.json:
        a.json.write_text(json.dumps(data,indent=2)+'\n')
    for polarity in a.polarities:
        for mode in a.modes:
            sub=[r for r in rows if r['polarity']==polarity and r['mode']==mode]
            passed=sum(r['final']['route_acc']>=.95 and r['final']['nmse']<.10 for r in sub)
            print('SUMMARY',polarity,mode,passed,'/',len(sub),'routes',
                  round(sum(r['final']['route_acc'] for r in sub)/len(sub),3),'nmse',
                  round(sum(r['final']['nmse'] for r in sub)/len(sub),4))

if __name__=='__main__': main()

