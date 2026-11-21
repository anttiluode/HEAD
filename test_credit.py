import unittest

import torch

from credit import LEVELS, N, fit_routes, read_orders
from head import Reader
from shared_medium import Medium, codebook


class CreditAndMedium(unittest.TestCase):
    def test_order_probe_and_target_moment(self):
        # All cue/probe pairs with ideal centered one-hot inputs as calibration examples.
        cue, probe, coord = torch.cartesian_prod(*[torch.arange(N)] * 3).T
        detail = torch.nn.functional.one_hot(coord, N).float() - 1/N
        x = detail + LEVELS[cue, None]
        y = detail.gather(1, ((cue + 3) % N)[:, None]).flatten()
        readings = read_orders(x, probe)
        torch.testing.assert_close(readings, detail.gather(1, probe[:, None]).flatten())
        torch.testing.assert_close(fit_routes(LEVELS[cue], probe, readings, y),
                                   (torch.arange(N) + 3) % N)

    def test_uniform_gate_gradient_reverses(self):
        # Centered ensemble, with fixed cue and zero-mean target. At uniform attention
        # the forward prediction is identical while the decoder sign reverses credit.
        x = torch.eye(N) - 1/N
        model = Reader(N)
        with torch.no_grad():
            model.keys.zero_()
            model.output.weight.zero_()
            model.output.bias.zero_()
        grads = []
        for beta in (-1., 0., 1.):
            with torch.no_grad():
                model.output.weight[0, 1] = beta
            loss = (model(x)-x[:, 0]).square().mean()
            grads.append(torch.autograd.grad(loss, model.keys)[0])
        self.assertGreater(grads[2].norm().item(), 0)
        torch.testing.assert_close(grads[0], -grads[2])
        self.assertEqual(grads[1].norm().item(), 0)

    def test_overlapping_code_and_persistent_effect(self):
        book = codebook(4, torch.Generator().manual_seed(52))
        torch.testing.assert_close(book @ book.T, 2 * torch.eye(4, dtype=book.dtype))
        x = book[:, 0][None, :]
        a, b = Medium(x), Medium(x)
        a.write(x)
        self.assertAlmostEqual(a.tick().item(), 1.)
        self.assertEqual(b.tick().item(), 0.)
        for medium in (a, b):
            medium.write(book[:, 1][None, :])
        self.assertAlmostEqual(a.tick().item(), b.tick().item()/2)
        # No shared mutable tensor between the initial field and either specimen.
        torch.testing.assert_close(x, book[:, 0][None, :])


if __name__ == "__main__":
    unittest.main()
