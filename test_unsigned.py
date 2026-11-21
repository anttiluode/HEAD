import unittest

import torch

from unsigned import ProbeScores, SLOTS, shuffle_within_cue


class UnsignedFeedback(unittest.TestCase):
    def test_local_statistics_agree_with_direct_correlation_and_ignore_sign(self):
        rng = torch.Generator().manual_seed(34)
        cue = torch.arange(SLOTS).repeat_interleave(SLOTS*6)
        probes = torch.arange(SLOTS).repeat_interleave(6).repeat(SLOTS)
        x = torch.randn(len(cue), generator=rng)
        e = torch.randn(len(cue), generator=rng)
        a, b = ProbeScores(), ProbeScores()
        for ix in torch.arange(len(cue)).split(23):
            a.collect(cue[ix], probes[ix], x[ix], e[ix])
            b.collect(cue[ix], probes[ix], x[ix], -e[ix])
        score = a.correlation()
        torch.testing.assert_close(score, -b.correlation())
        torch.testing.assert_close(score.square(), b.correlation().square())
        for c in range(SLOTS):
            for i in range(SLOTS):
                ix = (cue == c) & (probes == i)
                expected = torch.corrcoef(torch.stack((x[ix], e[ix])).double())[0, 1]
                torch.testing.assert_close(score[c, i], expected)

    def test_shuffle_preserves_each_groups_outcomes(self):
        cue = torch.arange(48) % SLOTS
        e = torch.arange(48)
        shuffled = shuffle_within_cue(e, cue, torch.Generator().manual_seed(3))
        self.assertFalse(torch.equal(e, shuffled))
        for c in range(SLOTS):
            torch.testing.assert_close(e[cue == c], shuffled[cue == c].sort().values)

    def test_empty_and_constant_cells_have_zero_signal(self):
        stats = ProbeScores()
        ix = torch.zeros(5, dtype=torch.long)
        stats.collect(ix, ix, torch.ones(5), torch.arange(5))
        torch.testing.assert_close(stats.correlation(), torch.zeros(SLOTS, SLOTS, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
