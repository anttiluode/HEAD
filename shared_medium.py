"""A persistent write in a shared dynamical state, with overlapping source codes."""
import argparse
import json
import math
from pathlib import Path

import torch

from credit import LEVELS, N, fit_routes


def codebook(dim, rng):
    if dim == N:
        frame = torch.eye(N, dtype=torch.float64)
    elif dim == 4:
        angles = 2 * math.pi * torch.arange(N, dtype=torch.float64) / N
        frame = torch.stack((angles.cos(), angles.sin(),
                             (2*angles).cos(), (2*angles).sin())) / math.sqrt(2)
    else:
        raise ValueError("supported dimensions: 4 and 8")
    rotation, _ = torch.linalg.qr(torch.randn(dim, dim, generator=rng, dtype=torch.float64))
    return rotation @ frame


class Medium:
    """A mark latches a direction. Subsequent ticks attenuate the field along it."""
    def __init__(self, field):
        self.field = field.clone()
        self.mark = torch.zeros_like(field)

    def write(self, direction):
        self.mark = direction.clone()

    def tick(self):
        reading = (self.field*self.mark).sum(1)
        self.field -= 0.5 * reading[:, None] * self.mark
        return reading


def batch(n, mapping, book, rng):
    address = torch.randint(N, (n,), generator=rng)
    sources = torch.randn(n, N, generator=rng, dtype=book.dtype)
    field = sources @ book.T
    target = sources.gather(1, mapping[address, None]).flatten()
    return address, field, target


def run(dim, seed):
    rng = torch.Generator().manual_seed(91000 + seed)
    book = codebook(dim, rng)
    mapping = torch.randperm(N, generator=rng)
    address, field, y = batch(9216, mapping, book, rng)
    probes = torch.randint(N, (len(y),), generator=rng)
    medium = Medium(field)
    medium.write(book.T[probes])
    first_tick = medium.tick()
    # A tick before the first mark always reads zero. One prepared run suffices.
    routes = fit_routes(LEVELS[address], probes, first_tick, y)
    chosen = probes == routes[address]
    gain = (first_tick[chosen] @ y[chosen]) / first_tick[chosen].square().sum()

    address, field, y = batch(8192, mapping, book, rng)
    indices = routes[address]
    direction = book.T[indices]
    medium = Medium(field)
    medium.write(direction)
    output = medium.tick()
    # Replace the latch in both paths. Any difference left now lives in the field.
    others = (indices + torch.randint(1, N, (len(y),), generator=rng)) % N
    common = book.T[others]
    medium.write(common)
    later = medium.tick()
    sham = Medium(field)
    sham.write(common)
    unchanged = sham.tick()
    delta = later - unchanged
    expected_delta = -0.5 * output * (direction*common).sum(1)

    # Equal-field source twins, constructed by projecting e_0 into the nullspace.
    v = torch.eye(N, dtype=book.dtype)[:, 0]
    invisible = v - torch.linalg.pinv(book) @ (book @ v)
    return {"dimension": dim, "seed": seed, "correct_routes": int((routes == mapping).sum()),
            "nmse": ((gain*output-y).square().mean()/y.square().mean()).item(),
            "learned_gain": gain.item(), "population_bayes_nmse": 1-dim/N,
            "field_change_visible_to_other_probe_rms": delta.square().mean().sqrt().item(),
            "stateful_witness_max_formula_error": (delta-expected_delta).abs().max().item(),
            "first_tick_max_difference_from_passive_projection":
                (output - (field*direction).sum(1)).abs().max().item(),
            "null_twin_field_difference": (book @ invisible).norm().item(),
            "null_twin_source0_difference": invisible[0].item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [run(dim, seed) for dim in (8, 4) for seed in range(1, 11)]
    report = json.dumps({"torch": str(torch.__version__), "runs": rows}, indent=2) + "\n"
    if args.json:
        args.json.write_text(report, encoding="utf-8")
    print(report, end="")
