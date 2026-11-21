# HEAD: Continuous unsigned routing — first tests (2026-11-21)

This is a **research pilot**, not a validated transformer/MoE improvement.

## Established HEAD result being extended

See [HEAD](https://github.com/anttiluode/HEAD), `UNSIGNED.md`: eight discrete cues;
training-only cue/slot correlation table and one random probe per training example;
weight .25 yielded 10/10 successes on the original task and reversed polarity.
That result is **not** a result on continuous token representations.

## Exact trap in the proposed shortcut

For scalar error `e` and scalar candidate values `v_i` on one example,
`argmax_i (v_i * e)^2 == argmax_i v_i^2` except in the degenerate case `e=0`.
No error-target information influences which candidate wins.
`instant` implements this negative control, and a unit test checks its
invariance to shuffling outcomes.

## Replacements tested

1. `residual`: continuous query-local Pearson correlation squared between
   **centered candidate values** and **detached prediction residuals** from
   nearby training examples.
2. `target`: the same batch-local estimate, but with the available training
   target instead of the prediction residual.
3. `signed`: signed conditional correlation with the residual.
4. `shuffled`: squared conditional correlation with randomly permuted residuals.
5. `instant`: magnitude-only single-example pseudo-label.
6. `baseline`: ordinary end-to-end Adam.

Both unsigned variants only use existing query/value matrices and existing
training labels; **no discrete cue ID statistics or extra coordinate reads**.
These are still supervised training objectives, not unsupervised attention.

## Architecture and data

- 8 candidate scalar slots; continuous two-dimensional unit-circle query `q`.
- Per seed, 8 hidden unit vectors at compass angles with a fixed random
  permutation. The target slot is `argmax(q @ hidden_keys.T)`.
- Matched CENTERED task: Gaussian detail `d = noise - mean(noise)`, common
  Gaussian level `c`, input slot values `x_i = c + d_i`, target `+d_j` or `-d_j`.
- Model sees `q` and all `x`, but not hidden keys or target slots.
- Learned attention logits `6 * q @ model_keys.T`, attention-weighted second
  read, and affine decoder on `[mean(x), second_read]`.
- 19 trainable parameters, not the original HEAD's 323.
- Adam learning rate .012; batches 192; 1,400 updates; training-only auxiliary
  at weight .25 during updates 1–350; later 1,050 updates without auxiliary.
- Held-out: 8,192 fresh examples/seed. Five initial seeds 1–5, five additional
  seeds 21–25, both target polarities. Same data RNG across methods per seed.
- Auxiliary uses leave-one-out query-neighborhood weighted covariance computed
  **inside the minibatch**. Cosine kernel `softmax(20 * q @ q.T)` with self blocked.
  It costs O(B^2 * slots) operations and O(B^2) similarity memory, so it is **not
  a negligible-cost drop-in for large attention models**.
- Routing criterion: >=95% of test queries routed to correct slot.
  Full-pass criterion, fixed before full runs: routing criterion **and** NMSE <0.10.

## Complete CENTERED results

Across both polarities, 10 seeds × 2 = 20 runs per method:

| method | route passes | full passes | mean route accuracy | mean NMSE |
|---|---:|---:|---:|---:|
| baseline | 10/20 | 7/20 | 0.491 | 0.458 |
| residual | 20/20 | 0/20 | 0.982 | 0.115 |
| target | 20/20 | 6/20 | 0.982 | 0.102 |
| shuffled | 10/20 | 0/20 | 0.491 | 0.469 |
| signed | 14/20 | 0/20 | 0.687 | 0.327 |

Polarity split, route passes:

| method | positive | negative |
|---|---:|---:|
| baseline | 3/10 | 7/10 |
| residual | 10/10 | 10/10 |
| target | 10/10 | 10/10 |
| shuffled | 4/10 | 6/10 |
| signed | 10/10 | 4/10 |

**Interpretation:** a table-free, sign-invariant, batch-local signal can rescue
routing under sign-trapped initialization and generalizes to five untouched
seeds in this toy setting. It did **not** improve the predeclared complete
predictive-success count (best 6/20 vs baseline 7/20), and residual-guided
models retained substantial prediction error even when their top slots were
almost always correct. Do not claim a useful neural-network improvement yet.

## Easier control task

`continuous_unsigned.py` uses independent Gaussian slot values and a one-read
decoder, rather than common mean plus centered contrast. The ordinary model
already reached ~98% routing accuracy in both polarities. With 5 seeds per
polarity at 900 updates, adding any tested auxiliary **worsened** mean NMSE:

| mode | overall mean NMSE (two polarities) |
|---|---:|
| baseline | 0.0939 |
| instant | 0.1080 |
| corr_residual | 0.1065 |
| corr_target | 0.0958 |
| shuffled | 0.1105 |

No method passed the predeclared NMSE <0.05 on this easier pilot.

## Reproduce (PyTorch only, CPU)

```
python -m unittest -v test_continuous_unsigned test_continuous_centered
python continuous_unsigned.py --seeds 1 2 3 4 5 --polarities 1 -1 --steps 900 --window 225 --json continuous_5seed_900.json
python continuous_centered.py --seeds 1 2 3 4 5 --modes baseline residual target shuffled signed --polarities 1 -1 --steps 1400 --window 350 --json centered_5seed_1400.json
python continuous_centered.py --seeds 21 22 23 24 25 --modes baseline residual target shuffled signed --polarities 1 -1 --steps 1400 --window 350 --json centered_fresh_5seed_1400.json
```

## What would genuinely move the needle

- Measure *end-to-end loss*, not just address accuracy, on a task where the
  baseline commonly gets trapped, with repeated unseen seeds.
- Test against simple interventions such as polarity-safe initialization,
  sign-constrained decoder, key pretraining, and standard supervised routing.
- Scale beyond 2D queries with close neighbors: query-local squared correlation
  is O(batch^2 * slots), may fail in high-dimensional token spaces, and needs
  large enough conditional sample support to be meaningful.
- Test on a standard small attention or sparse MoE benchmark, with throughput
  and wall-clock training time reported as well as accuracy.
- Avoid presenting this as a new general defect in backpropagation: it is a
  specific coupled optimization failure, and signed/unsigned relevance
  objectives have existing methodological relatives.