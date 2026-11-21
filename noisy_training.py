"""Paired rerun of HEAD with independent noise on the second scalar meter."""
import argparse
import json
from pathlib import Path

import torch

from head import sample, train


@torch.no_grad()
def assess(model, mapping, seed, sd):
    rng = torch.Generator().manual_seed(17000 + seed)
    x, y, address = sample(8192, mapping, rng, "cpu")
    cue = x.mean(1, keepdim=True)
    access = (model.controller(cue).tanh() @ model.keys.T).softmax(1)
    beta = model.output.weight[0, 1].item()
    clean_mse = (model(x) - y).square().mean().item()
    target_power = y.square().mean().item()
    return {
        "correct_top_slot_fraction": (access.argmax(1) == mapping[address]).float().mean().item(),
        "mean_target_weight": access.gather(1, mapping[address, None]).mean().item(),
        "second_decoder_weight": beta,
        "clean_nmse": clean_mse / target_power,
        "expected_noisy_nmse": (clean_mse + (sd * beta)**2) / target_power,
    }


def experiment():
    rows = []
    for sd in (0., 0.05, 0.2, 0.5, 1.0):
        for exact in ([False, True] if sd else [False]):
            for seed in range(1, 11):
                mapping = torch.randperm(8, generator=torch.Generator().manual_seed(seed + 1000))
                model, initial = train(seed, mapping, "mask", 1400, "cpu", "default", sd, exact)
                rows.append({"sd": sd, "loss": "exact" if exact else "sampled",
                             "seed": seed, "initial_weight": initial, **assess(model, mapping, seed, sd)})
    return {"torch": str(torch.__version__), "device": "cpu", "slots": 8, "parameters": 323,
            "steps": 1400, "train_batch": 192, "held_out": 8192, "runs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = experiment()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for sd, loss in dict.fromkeys((r["sd"], r["loss"]) for r in result["runs"]):
        rows = [r for r in result["runs"] if (r["sd"], r["loss"]) == (sd, loss)]
        print(sd, loss, "seeds routing every item correctly:",
              [r["seed"] for r in rows if r["correct_top_slot_fraction"] == 1])
