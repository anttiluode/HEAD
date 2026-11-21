"""Two scalar observations of the same vector. PyTorch 2.5+."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Reader(nn.Module):
    def __init__(self, slots, width=32):
        super().__init__()
        self.controller = nn.Linear(1, width)
        self.keys = nn.Parameter(torch.randn(slots, width) * 0.05)
        self.output = nn.Linear(2, 1)

    def forward(self, values, mode="mask", cue_override=None):
        first = values.mean(dim=1, keepdim=True)
        cue = first if cue_override is None else cue_override
        query = self.controller(cue).tanh()
        if mode == "attention":
            # A single query over learned positional keys and scalar token values.
            second = F.scaled_dot_product_attention(
                query[:, None, None, :], self.keys[None, None, :, :],
                values[:, None, :, None], scale=1.0).reshape(-1, 1)
        elif mode == "mask":
            access = (query @ self.keys.T).softmax(dim=-1)
            # Scratch view only. The payload is available to a later reader.
            second = (values.shape[1] * access * values).mean(dim=1, keepdim=True)
        else:
            raise ValueError(mode)
        return self.output(torch.cat((first, second), dim=1)).squeeze(1)


def sample(count, mapping, generator, device):
    slots = len(mapping)
    address = torch.randint(slots, (count,), generator=generator, device=device)
    cue = (2 * address / (slots - 1) - 1).unsqueeze(1)
    noise = torch.randn(count, slots, generator=generator, device=device)
    detail = noise - noise.mean(dim=1, keepdim=True)
    values = cue + detail
    target = detail.gather(1, mapping[address, None]).squeeze(1)
    return values, target, address


def train(seed, mapping, mode, steps, device, initial_sign):
    torch.manual_seed(seed)
    model = Reader(len(mapping)).to(device)
    start_weight = model.output.weight[0, 1]
    if initial_sign != "default":
        with torch.no_grad():
            start_weight.copy_(start_weight.abs() if initial_sign == "positive" else -start_weight.abs())
    starting_value = start_weight.item()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    batches = torch.Generator(device=device).manual_seed(7000 + seed)
    for _ in range(steps):
        x, y, _ = sample(192, mapping, batches, device)
        prediction = model(x, mode)
        loss = F.mse_loss(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval(), starting_value


def nmse(pred, y):
    return ((pred - y).square().mean() / y.square().mean()).item()


@torch.no_grad()
def evaluate(mask, attention, mapping, seed, device):
    batches = torch.Generator(device=device).manual_seed(seed + 17000)
    x, y, address = sample(8192, mapping, batches, device)
    cue = x.mean(1, keepdim=True)
    # Break only the routing signal. Preserve both the payload and decoder's first read.
    permutation = torch.randperm(len(x), generator=batches, device=device)
    gates = (mask.controller(cue).tanh() @ mask.keys.T).softmax(1)
    pred = mask(x)
    # Best two fixed orthogonal contrast directions with address supplied free.
    # Choose one 2-D plane in the contrast space. Isotropy makes all choices tie in expectation.
    slots = len(mapping)
    offsets = torch.arange(slots, device=device) * (2 * torch.pi / slots)
    plane = torch.stack((offsets.cos(), offsets.sin()), dim=1) * (2 / slots)**0.5
    projection = ((x - cue) @ plane) @ plane.T
    oracle = projection.gather(1, mapping[address, None]).squeeze(1)
    return {
        "seed": seed, "slot_for_each_cue": mapping.tolist(),
        "examples": len(x), "mask_nmse": nmse(pred, y),
        "attention_nmse": nmse(attention(x, "attention"), y),
        "same_weights_max_abs_error": (mask(x, "attention") - pred).abs().max().item(),
        "shuffled_cue_nmse": nmse(mask(x, cue_override=cue[permutation]), y),
        "fixed_two_plus_free_cue_nmse": nmse(oracle, y),
        "correct_top_slot_fraction": (gates.argmax(1) == mapping[address]).float().mean().item(),
        "output_weights_first_second": mask.output.weight.flatten().tolist(),
        "parameters_per_learned_model": sum(p.numel() for p in mask.parameters()),
    }


def run(seeds, slots, steps, device, initial_sign):
    rows = []
    for seed in seeds:
        perm = torch.randperm(slots, generator=torch.Generator().manual_seed(seed+1000)).to(device)
        mask, starting_value = train(seed, perm, "mask", steps, device, initial_sign)
        attention, _ = train(seed, perm, "attention", steps, device, initial_sign)
        row = evaluate(mask, attention, perm, seed, device)
        row["initial_second_read_weight"] = starting_value
        rows.append(row)
    return {"initial_second_read_sign": initial_sign, "slots": slots, "steps": steps, "batch": 192, "device": device,
            "torch": str(torch.__version__), "fixed_two_population_nmse": 1 - 2/(slots-1),
            "runs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial-sign", choices=["default", "positive", "negative"], default="default")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.slots < 3 or args.steps < 1:
        parser.error("use at least three slots and one optimization step")
    torch.set_num_threads(1)
    result = run(args.seeds, args.slots, args.steps, args.device, args.initial_sign)
    report = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(report, encoding="utf-8")
    print(report, end="")
