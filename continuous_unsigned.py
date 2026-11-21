"""HEAD follow-up: continuous-context, full-read unsigned attention pilot.

Research experiment, not an established improvement. No cue buckets, probe calls,
or routing labels. Queries are real-valued unit vectors. Candidate values are
already in the forward pass. All auxiliary statistics are batch-local.

Usage: python continuous_unsigned.py --seeds 1 2 3 --steps 900 --modes ...
"""
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SLOTS = 8


class Head(nn.Module):
    def __init__(self, slots=SLOTS):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(slots, 2) * 0.10)
        self.decoder = nn.Linear(1, 1)

    def forward(self, query, values):
        logits = (query @ self.keys.T) * 6.0
        attention = logits.softmax(dim=1)
        read = (attention * values).sum(1, keepdim=True)
        return self.decoder(read).squeeze(1), logits


def get_true_keys(seed, slots=SLOTS):
    generator = torch.Generator().manual_seed(1000 + seed)
    permutation = torch.randperm(slots, generator=generator)
    angles = 2 * math.pi * permutation.float() / slots
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def sample(n, true_keys, generator, polarity=1):
    angle = torch.rand(n, generator=generator) * (2 * math.pi)
    query = torch.stack((angle.cos(), angle.sin()), dim=1)
    values = torch.randn(n, len(true_keys), generator=generator)
    chosen = (query @ true_keys.T).argmax(dim=1)
    target = polarity * values.gather(1, chosen[:, None]).squeeze(1)
    return query, values, target, chosen


@torch.no_grad()
def conditional_correlation(query, values, response, sharpness=20.0):
    """Leave-self-out Nadaraya-Watson moments; squared conditional correlation.

    Compute per-query weights from continuous query similarity. The cost is
    O(batch^2 * slots), unlike the original O(cue * slot) cumulative table.
    No model parameters or target addresses are used here.
    """
    sim = sharpness * (query @ query.T)
    sim.fill_diagonal_(-1e9)  # no sample can certify its own relevance
    weights = sim.softmax(1)
    m_v = weights @ values
    m_r = weights @ response
    cov = weights @ (values * response[:, None]) - m_v * m_r[:, None]
    var_v = (weights @ values.square() - m_v.square()).clamp_min(1e-8)
    var_r = (weights @ response.square() - m_r.square()).clamp_min(1e-8)
    return (cov.square() / (var_v * var_r[:, None])).clamp(0, 1)


def aux_labels(mode, query, values, target, prediction, rng):
    if mode == 'instant':
        # This deliberately implements the false shortcut in Neon's proposal.
        error = (target - prediction).detach()
        return ((values * error[:, None])**2).argmax(1)
    if mode == 'shuffled':
        perm = torch.randperm(len(target), generator=rng)
        response = (target - prediction).detach()[perm]
    elif mode == 'corr_residual' or mode == 'signed_residual':
        response = (target - prediction).detach()
    elif mode == 'corr_target' or mode == 'signed_target':
        response = target.detach()
    else:
        raise ValueError(mode)
    corr2 = conditional_correlation(query, values, response)
    if mode.startswith('signed_'):
        # Derive sign by covariance, not by naively sqrt(corr^2).
        sim = 20.0 * (query @ query.T)
        sim.fill_diagonal_(-1e9)
        weights = sim.softmax(1)
        mv, mr = weights @ values, weights @ response
        covariance = weights @ (values * response[:, None]) - mv * mr[:, None]
        return (covariance.sign() * corr2.sqrt()).argmax(1)
    return corr2.argmax(1)


@torch.no_grad()
def evaluate(model, true_keys, seed, polarity, count=8192):
    q, v, y, true_slot = sample(count, true_keys,
                               torch.Generator().manual_seed(seed+17000), polarity)
    pred, logits = model(q, v)
    return {'nmse': ((pred-y).square().mean() / y.square().mean()).item(),
            'route_acc': (logits.argmax(1)==true_slot).float().mean().item(),
            'decoder_weight': model.decoder.weight.item()}


def train(seed, mode, polarity=1, steps=900, window=225, weight=.25, batch=192):
    true_keys = get_true_keys(seed)
    torch.manual_seed(seed)
    model = Head()
    batches = torch.Generator().manual_seed(7000+seed)
    shuffle_rng = torch.Generator().manual_seed(67000+seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    start_decoder = model.decoder.weight.item()
    accuracy_at_cut = None
    for step in range(1, steps+1):
        q, v, y, _ = sample(batch, true_keys, batches, polarity)
        prediction, logits = model(q, v)
        loss = F.mse_loss(prediction, y)
        if mode != 'baseline' and step <= window:
            pseudo = aux_labels(mode, q, v, y, prediction, shuffle_rng)
            loss = loss + weight * F.cross_entropy(logits, pseudo)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == window:
            accuracy_at_cut = evaluate(model, true_keys, seed, polarity, count=2048)
    return {'seed': seed, 'mode': mode, 'polarity': polarity, 'start_decoder': start_decoder,
            'at_cut': accuracy_at_cut, 'final': evaluate(model, true_keys, seed, polarity)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,2,3])
    parser.add_argument('--modes', nargs='+', default=['baseline', 'instant', 'corr_residual', 'corr_target', 'shuffled'],
                        choices=['baseline','instant','corr_residual','corr_target','shuffled','signed_residual','signed_target'])
    parser.add_argument('--polarities', nargs='+', type=int, choices=[-1, 1], default=[1, -1])
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--window', type=int, default=225)
    parser.add_argument('--batch', type=int, default=192)
    parser.add_argument('--weight', type=float, default=.25)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for polarity in args.polarities:
        for mode in args.modes:
            for seed in args.seeds:
                row = train(seed,mode,polarity,args.steps,args.window,args.weight,args.batch)
                rows.append(row)
                print('polarity',polarity,'mode',mode,'seed',seed,
                      'routing',f"{row['final']['route_acc']:.3f}",
                      'nmse',f"{row['final']['nmse']:.4f}", flush=True)
    result = {'torch': torch.__version__, 'design': '8 slots; 2D continuous queries on unit circle; fixed hidden random-permuted compass; full existing values; batch-local leave-one-out correlation',
              'steps':args.steps,'window':args.window,'weight':args.weight,'batch':args.batch,
              'success': 'route_acc > .95 and nmse < .05', 'runs':rows}
    if args.json:
        args.json.write_text(json.dumps(result,indent=2)+'\n')
    for polarity in args.polarities:
        for mode in args.modes:
            subset=[r for r in rows if r['polarity']==polarity and r['mode']==mode]
            passes=sum(r['final']['route_acc']>.95 and r['final']['nmse']<.05 for r in subset)
            print('SUMMARY',polarity,mode,passes,'/',len(subset),
                  'mean_route',round(sum(r['final']['route_acc'] for r in subset)/len(subset),3),
                  'mean_nmse',round(sum(r['final']['nmse'] for r in subset)/len(subset),4))

if __name__=='__main__':
    main()

