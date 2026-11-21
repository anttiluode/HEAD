"""Trace credit in the original HEAD reader and test a decoder-independent repair."""
import argparse
import copy
import json
from itertools import permutations
from pathlib import Path

import torch
from torch.nn import functional as F

from head import Reader, nmse, sample

N = 8
LEVELS = torch.linspace(-1, 1, N)
CHECKPOINTS = {0, 1, 5, 20, 100, 500, 1400}


def cue_bin(c):
    return (c.reshape(-1, 1) - LEVELS).abs().argmin(1)


def audit(model, data, mapping):
    x, y, address = data
    pred = model(x)
    params = [model.keys, *model.controller.parameters()]
    grad = torch.autograd.grad(F.mse_loss(pred, y), params)
    with torch.no_grad():
        query = model.controller(LEVELS[:, None]).tanh()
        gates = (query @ model.keys.T).softmax(1)
        hits = gates[torch.arange(N), mapping]
        return {"nmse": nmse(pred, y), "beta": model.output.weight[0, 1].item(),
                "router_grad_norm": sum(g.square().sum() for g in grad).sqrt().item(),
                "target_weight": hits.mean().item(),
                "top_correct": int((gates.argmax(1) == mapping).sum()),
                "bottom_correct": int((gates.argmin(1) == mapping).sum()),
                "mean_effective_slots": (1 / gates.square().sum(1)).mean().item()}


def update(model, opt, x, y):
    opt.zero_grad()
    F.mse_loss(model(x), y).backward()
    opt.step()


def read_orders(x, indices):
    """F reflects around the mean; B copies the probed coordinate to every slot."""
    mean = x.mean(1, keepdim=True)
    reflect_then_copy = (2*mean - x).gather(1, indices[:, None]).expand_as(x)
    copied = x.gather(1, indices[:, None]).expand_as(x)
    copy_then_reflect = 2*copied.mean(1, keepdim=True) - copied
    return (copy_then_reflect.mean(1) - reflect_then_copy.mean(1)) / 2


def fit_routes(cue, probes, readings, targets):
    """Estimate a conditional cross-moment from one probed coordinate per example."""
    cells = N * cue_bin(cue) + probes
    counts = torch.bincount(cells, minlength=N*N)
    if torch.any(counts == 0):
        raise ValueError("probe calibration did not cover every cue and actuator")
    sums = torch.bincount(cells, weights=readings * targets, minlength=N*N)
    return (sums/counts).reshape(N, N).argmax(1)


@torch.no_grad()
def install(model, routes, cue, probes, readings, targets):
    """Revise only the keys and affine decoder. Keep the learned controller intact."""
    basis = model.controller(LEVELS[:, None]).tanh().double()
    desired = 8 * F.one_hot(routes, N).double()
    keys = model.keys.double()
    # Minimum-norm correction at the eight cue levels. Start from the existing keys.
    delta = torch.linalg.pinv(basis) @ (desired - basis @ keys.T)
    model.keys.add_(delta.T.float())
    # Use only trials where the random actuator happened to match the inferred route.
    # Softmax with one logit of 8 mixes that actuator's read with the uniform read.
    selected = probes == routes[cue_bin(cue)]
    gain = (torch.exp(torch.tensor(8.)) - 1) / (torch.exp(torch.tensor(8.)) + N - 1)
    second = cue[selected] + gain * readings[selected]
    fit_decoder(model, cue[selected], second, targets[selected])


@torch.no_grad()
def fit_decoder(model, first, second, targets):
    design = torch.stack((first, second, torch.ones_like(first)), dim=1).double()
    coef = torch.linalg.lstsq(design, targets.double()).solution.float()
    model.output.weight.copy_(coef[:2][None, :])
    model.output.bias.copy_(coef[2:])


def seed_run(seed):
    rng = lambda offset: torch.Generator().manual_seed(offset + seed)
    mapping = torch.randperm(N, generator=rng(1000))
    torch.manual_seed(seed)
    model = Reader(N)
    opt = torch.optim.Adam(model.parameters(), lr=0.012)
    batches = rng(7000)
    held = sample(8192, mapping, rng(17000), "cpu")
    # Fixed held-out examples for loss and gradient diagnostics. Never used to update.
    trace = {}
    for t in range(1401):
        if t in CHECKPOINTS:
            trace[t] = audit(model, held, mapping)
        if t < 1400:
            x, y, _ = sample(192, mapping, batches, "cpu")
            update(model, opt, x, y)

    # Calibrate using exactly 48 fresh batches, 9216 examples and one task outcome each.
    generator = rng(31000)
    x, y, _ = sample(48*192, mapping, generator, "cpu")
    probes = torch.randint(N, (len(y),), generator=rng(32000))
    cue = x.mean(1)
    readings = read_orders(x, probes)
    direct = x.gather(1, probes[:, None]).flatten() - cue
    routes = fit_routes(cue, probes, readings, y)
    passive = fit_routes(cue, probes, direct, y)
    # Shuffle labels within each cue, preserving each cue's empirical target distribution.
    shuffled = y.clone()
    shuffle_rng = rng(33000)
    bins = cue_bin(cue)
    for j in range(N):
        rows = torch.where(bins == j)[0]
        shuffled[rows] = y[rows[torch.randperm(len(rows), generator=shuffle_rng)]]
    bogus = fit_routes(cue, probes, readings, shuffled)

    # Post-hoc: does avoidance already identify the permutation? Decode all 8! candidates,
    # assuming a bijection and using no calibration outcomes. This is an audit, not training.
    with torch.no_grad():
        logits = model.controller(LEVELS[:, None]).tanh() @ model.keys.T
        perm_bank = torch.tensor(list(permutations(range(N))))
        costs = logits[torch.arange(N), perm_bank].sum(1)
        avoid = perm_bank[costs.argmin()]
        prefer = perm_bank[costs.argmax()]
        decoded = avoid if model.output.weight[0, 1].item() < 0 else prefer
        rank_score = (logits.gather(1, mapping[:, None]) > logits).sum(1)
        saved_route = {"sign_oriented_assignment_correct": int((decoded == mapping).sum()),
                       "true_slot_mean_rank_from_lowest": 1 + rank_score.float().mean().item(),
                       "inferred_assignment": decoded.tolist()}
    treatments = {}
    for name, table in (("order_probe", routes), ("direct_probe", passive),
                        ("shuffled_outcomes", bogus)):
        candidate = copy.deepcopy(model)
        install(candidate, table, cue, probes, direct if name == "direct_probe" else readings,
                shuffled if name == "shuffled_outcomes" else y)
        treatments[name] = audit(candidate, held, mapping)
    # Give the existing route the best affine readout on the same calibration examples.
    candidate = copy.deepcopy(model)
    with torch.no_grad():
        gates = (candidate.controller(cue[:, None]).tanh() @ candidate.keys.T).softmax(1)
        fit_decoder(candidate, cue, (gates*x).sum(1), y)
    treatments["refit_decoder_only"] = audit(candidate, held, mapping)
    # Adam retains its momentum for the budget-matched continuation.
    for chunk, target in zip(x.split(192), y.split(192)):
        update(model, opt, chunk, target)
    treatments["continue_adam"] = audit(model, held, mapping)
    return {"seed": seed, "route_audit": saved_route, "trace": trace, "after": treatments,
            "order_direct_max_difference": (readings - direct).abs().max().item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [seed_run(seed) for seed in range(1, 11)]
    result = {"torch": str(torch.__version__), "fit_examples": 48*192, "runs": rows}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["seed"], row["trace"][1400], {k: v["top_correct"] for k, v in row["after"].items()})
