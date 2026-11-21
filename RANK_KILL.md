# Decoder-rank kill test (2026-11-21)

Paired ablation on the original HEAD data generator, without any unsigned auxiliary, with seeds 1–10 under both target polarities, 1,400 Adam updates (lr .012), 192 training examples per update, and 8,192 held-out examples per seed. Full pass = all eight cue routes correct and NMSE < .001.

| Model | Positive target | Reversed target | Total |
| --- | ---: | ---: | ---: |
| Original one-head scalar output | 2/10 | 7/10 | 9/20 |
| Two-unit *linear* decoder (2→2→1) | 2/10 | 7/10 | 9/20 |
| Two-unit paired-ReLU contrast decoder | 2/10 | 7/10 | 9/20 |
| Two independent attention reads + linear output | 6/10 | 8/10 | 14/20 |

Every modified model started with the same predictions as the baseline. Four unit tests passed, including an exact check of the fixed-route Shadow Valley formulas.

**Interpretation:** Widening the decoder over the *same scalar read* did not remove the sign-trap (even with two nonlinear units). Adding a genuinely independent read helped but did not cure it. The two-head arm gives more information and parameters, so it is not a decoder-only control. None of these results establishes a transformer-scale improvement from unsigned routing.

For the zero-sum Gaussian details in HEAD, if attention excludes the target exactly and puts weights α on seven wrong slots, define m_eff = 1/Σα_i². Conditional optimum: β*=−m_eff/(8−m_eff), NMSE*=1−m_eff/[7(8−m_eff)]. This is a fixed-route conditional identity, not a proof of a global optimization barrier.

**Reproducible local archive and full 80-run receipts** are delivered with the ChatGPT experiment as `HEAD_decoder_rank_kill.zip` (files: `rank_kill.py`, `test_rank_kill.py`, `RANK_KILL.md`, `rank_kill_results.json`, `rank_kill.log`). The full multi-dimensional value-projection test remains open.

Original `main` unchanged.
