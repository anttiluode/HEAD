import unittest

import torch

from head import Reader, sample
from meter import compare


class AccessTests(unittest.TestCase):
    def test_two_shadows_and_fixed_query(self):
        x = torch.tensor([[0., 1.], [1., 0.], [2., -7.]], dtype=torch.float64)
        copy_left = torch.tensor([[1., 0.], [1., 0.]], dtype=x.dtype)
        meter = torch.ones(2, dtype=x.dtype)
        s = x @ meter
        v = (x @ copy_left.T) @ meter
        torch.testing.assert_close(s - v, x[:, 1] - x[:, 0])
        torch.testing.assert_close(v, x @ (copy_left.T @ meter))

    def test_same_values_and_gradients(self):
        torch.manual_seed(41)
        model = Reader(7, width=9).double()
        x = torch.randn(31, 7, dtype=torch.float64, requires_grad=True)
        mask = model(x)
        attn = model(x, "attention")
        torch.testing.assert_close(mask, attn, rtol=1e-11, atol=1e-12)
        variables = (x, *model.parameters())
        gradients = [torch.autograd.grad(out.square().sum(), variables) for out in (mask, attn)]
        for left, right in zip(*gradients):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)

    def test_noise_penalty_matches_both_meter_perturbations(self):
        torch.manual_seed(60)
        model = Reader(8).double()
        x = torch.randn(13, 8, dtype=torch.float64)
        y = torch.randn(13, dtype=torch.float64)
        scalars = []
        hook = model.output.register_forward_pre_hook(lambda _, args: scalars.append(args[0]))
        p = model(x)
        hook.remove()
        delta = torch.tensor([0.0, 0.2], dtype=x.dtype)
        losses = [(model.output(scalars[0] + sign*delta).flatten()-y).square().mean()
                  for sign in (-1, 1)]
        expected = (p-y).square().mean() + (model.output.weight[0, 1]*0.2).square()
        torch.testing.assert_close((losses[0] + losses[1])/2, expected)

    def test_meter_noise_gain(self):
        for slots in (3, 8, 11):
            result = compare(slots=slots, examples=1024)
            self.assertLess(result["noiseless_max_disagreement"], 1e-12)
            self.assertAlmostEqual(result["mse_ratio"] / (slots-1)**2, 1.0, places=5)

    def test_bottleneck_and_target(self):
        perm = torch.tensor([2, 1, 3, 0])
        x, y, a = sample(57, perm, torch.Generator().manual_seed(43), "cpu")
        c = x.mean(1)
        torch.testing.assert_close(c, 2*a/3 - 1)
        torch.testing.assert_close(y, x[torch.arange(len(x)), perm[a]] - c)


if __name__ == "__main__":
    unittest.main()
