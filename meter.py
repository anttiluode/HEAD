"""Compare exact readouts that open the requested slot or close it completely."""
import json

import torch

from head import sample


def compare(slots=8, examples=32768, sigma=0.05, seed=119):
    generator = torch.Generator().manual_seed(seed)
    x, target, j = sample(examples, torch.arange(slots), generator, "cpu")
    x = x.double()
    s = x.mean(1)
    selected = torch.nn.functional.one_hot(j, slots).double()
    keep = (x * selected).sum(1)
    omit = (x * (1 - selected)).sum(1) / (slots - 1)
    error = sigma * torch.randn(examples, dtype=x.dtype, generator=generator)
    # Common noise realization isolates the gain of the reconstruction.
    direct_error = keep + error - s - target
    complement_error = (slots - 1) * (s - (omit + error)) - target
    direct_mse = direct_error.square().mean().item()
    complement_mse = complement_error.square().mean().item()
    return {
        "slots": slots, "examples": examples, "seed": seed, "second_read_noise_sd": sigma,
        "noiseless_max_disagreement": ((keep - s) - (slots-1)*(s-omit)).abs().max().item(),
        "direct_mse": direct_mse, "complement_mse": complement_mse,
        "mse_ratio": complement_mse / direct_mse, "predicted_ratio": (slots-1)**2,
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
