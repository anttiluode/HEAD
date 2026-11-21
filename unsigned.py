"""One random local probe per training example during an early auxiliary window."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from head import Reader, nmse, sample


SLOTS = 8
LEVELS = torch.linspace(-1, 1, SLOTS)


class ProbeScores:
    def __init__(self):
        # count, sum(x), sum(e), sum(x²), sum(e²), sum(xe), per cue and probe.
        self.sums = torch.zeros(6, SLOTS*SLOTS, dtype=torch.float64)

    @torch.no_grad()
    def collect(self, cue_ids, probe_ids, reading, residual):
        cells = SLOTS*cue_ids + probe_ids
        x, e = reading.double(), residual.double()
        terms = torch.stack((torch.ones_like(x), x, e, x*x, e*e, x*e))
        self.sums.scatter_add_(1, cells.expand_as(terms), terms)

    def correlation(self):
        n, x, e, xx, ee, xe = self.sums
        n_safe = n.clamp_min(1)
        cov = xe - x*e/n_safe
        xvar = (xx-x*x/n_safe).clamp_min(0)
        evar = (ee-e*e/n_safe).clamp_min(0)
        denom = (xvar*evar).sqrt()
        corr = torch.where((n >= 3) & (denom > 1e-12), cov/denom.clamp_min(1e-12), 0.)
        return corr.reshape(SLOTS, SLOTS)


def shuffle_within_cue(errors, cue_ids, generator):
    result = errors.clone()
    for group in cue_ids.unique():
        ix = torch.where(cue_ids == group)[0]
        result[ix] = errors[ix[torch.randperm(len(ix), generator=generator)]]
    return result


@torch.no_grad()
def evaluate(model, held, mapping, polarity):
    x, y, _ = held
    gates = (model.controller(LEVELS[:, None]).tanh() @ model.keys.T).softmax(1)
    return {"routes": int((gates.argmax(1) == mapping).sum()),
            "target_weight": gates[torch.arange(SLOTS), mapping].mean().item(),
            "nmse": nmse(model(x), polarity*y),
            "second_coefficient": model.output.weight[0, 1].item()}


def run(seed, mode, polarity=1, weight=0.1):
    rng = lambda offset: torch.Generator().manual_seed(offset+seed)
    mapping = torch.randperm(SLOTS, generator=rng(1000))
    batches, probing, shuffling = rng(7000), rng(57000), rng(67000)
    torch.manual_seed(seed)
    model = Reader(SLOTS)
    opt = torch.optim.Adam(model.parameters(), lr=0.012)
    held = sample(8192, mapping, rng(17000), "cpu")
    stats = ProbeScores()
    teacher_routes = None
    for step in range(1, 1401):
        x, target, _ = sample(192, mapping, batches, "cpu")
        y = polarity*target
        pred = model(x)
        loss = F.mse_loss(pred, y)
        if mode != "baseline" and step <= 350:
            mean = x.mean(1)
            cues = (mean[:, None]-LEVELS).abs().argmin(1)
            probes = torch.randint(SLOTS, (len(x),), generator=probing)
            reading = x[torch.arange(len(x)), probes] - mean
            errors = (y-pred).detach()
            if mode == "shuffled":
                errors = shuffle_within_cue(errors, cues, shuffling)
            stats.collect(cues, probes, reading, errors)
            if step > 8:
                scores = stats.correlation()
                if mode != "signed":
                    scores = scores.square()
                labels = scores.argmax(1)
                # Recompute only at the eight legal cue levels for the auxiliary loss.
                logits = model.controller(LEVELS[:, None]).tanh() @ model.keys.T
                loss = loss + weight*F.cross_entropy(logits[cues], labels[cues])
                if step == 350:
                    teacher_routes = labels.tolist()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 350:
            at_cut = evaluate(model, held, mapping, polarity)
    return {"seed": seed, "mode": mode, "polarity": polarity, "at_cut": at_cut,
            "final": evaluate(model, held, mapping, polarity),
            "mapping": mapping.tolist(), "auxiliary_routes_at_cut": teacher_routes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--modes", nargs="+", choices=["baseline","unsigned","shuffled","signed"],
                        default=["baseline","unsigned","shuffled","signed"])
    parser.add_argument("--polarity", type=int, choices=[-1,1], default=1)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [run(seed, mode, args.polarity, args.weight) for mode in args.modes for seed in args.seeds]
    result = {"torch": str(torch.__version__), "device": "cpu", "steps":1400,
              "batch":192, "window":350, "weight":args.weight, "runs": rows}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for mode in args.modes:
        final = [r["final"] for r in rows if r["mode"] == mode]
        print(mode, sum(f["routes"] == 8 and f["nmse"] < .001 for f in final), "/", len(final))
