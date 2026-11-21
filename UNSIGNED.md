# Can unsigned feedback rescue a repulsive router during training?

**At auxiliary weight 0.25 the original seeds went from 2/10 to 10/10 and an untouched ten seeds went from 3/10 to 10/10.** Both signed and unsigned local feedback rescued the positive-target task. Only unsigned feedback preserved that gain under target reversal.

The original HEAD reader has 323 trainable parameters. Its decoder sometimes supplies a gradient that steers the router away from the informative slot. Here an auxiliary objective gives the router a second training signal while the downstream coefficient is still learning its sign. The inference code and parameter count stay the same.

## Method

On each of the first 350 existing minibatches, choose one slot uniformly per example, using a separate training RNG. Measure its value minus the first read (the mean). Associate this measurement with the detached scalar prediction error for the same example. For each of the eight observable cue levels and eight possible probes, keep the six running sums needed for Pearson correlation. After eight batches of collection, use the slot with largest squared correlation for that cue as a provisional target for cross entropy on the routing logits. Update the decoder with ordinary MSE throughout. Starting at update 351, optimize only MSE for the remaining 1,050 updates.

The auxiliary uses the existing task outcome and no target address. Cue bins are supplied from the known discrete input design. Squaring correlation makes the local score invariant to reversing the sign of its residual samples. The complete optimizer trajectory can still depend on sign. The local objective brings a stronger training interface: one extra coordinate observation per example in the early window, 384 training-only statistics, and extra training computation. There are exactly 268,800 labelled training examples per fit and zero extra labels, updates or trainable parameters. A randomized probe need not be available or cheap in other architectures.

[First-run protocol](UNSIGNED_PROTOCOL.md) · [Training code](unsigned.py)

## Results

A pass requires every cue's top slot correct **and** held-out NMSE below 0.001, after the auxiliary has been off for 1,050 updates. Evaluation uses the same 8,192 fresh inputs per seed as HEAD. The signed control picks the largest Pearson correlation; the shuffled control permutes the detached residuals within each cue and then uses squared correlation. Both get the same access, auxiliary weight and duration. A dash denotes a condition not rerun in that sweep.

| Seeds and target | Weight | Task loss alone | Unsigned | Signed | Shuffled |
|---|---:|---:|---:|---:|---:|
| Original 1–10 | 0.10 | 2/10 | 8/10 | 8/10 | 0/10 |
| Fresh 11–20 | 0.10 | 4/10 | 9/10 | 9/10 | 0/10 |
| Reversed target 1–10 | 0.10 | — | 9/10 | 2/10 | — |
| Original 1–10 | 0.25 | 2/10 | 10/10 | 10/10 | 0/10 |
| Fresh 21–30 | 0.25 | 3/10 | 10/10 | 10/10 | 0/10 |
| Reversed target 1–10 | 0.25 | 7/10 | 10/10 | 0/10 | 0/10 |

At the initial weight of 0.10, original seeds 9 and 10 failed even though each probe table identified all eight addresses at the cutoff. Their networks still gave the requested slot the highest weight on zero cues. With the usual positive target polarity the signed control matched the unsigned pass count on both groups. Reversing every target exposed the difference between positive correlation and correlation of either sign. Routing success alone overstates prediction success in some control arms.

Weight 0.25 was selected **after** observing the 0.10 result. Seeds 21–30 were untouched when that choice was made. The binning rule, thresholds, training window and optimizer remained fixed. The experiments measure this eight-slot supervised routing task. They do not establish generality for transformer layers, sparse expert routers, continuous contexts, or language modeling.

| Unsigned run | Median held-out NMSE | Range | Runs with all routes correct |
|---|---:|---:|---:|
| Original 1–10, 0.10 | 0.000483 | 0.000378–0.537780 | 8/10 |
| Fresh 11–20, 0.10 | 0.000475 | 0.000322–0.658786 | 9/10 |
| Reversed target 1–10, 0.10 | 0.000505 | 0.000424–0.548512 | 9/10 |
| Original 1–10, 0.25 | 0.000467 | 0.000399–0.000854 | 10/10 |
| Fresh 21–30, 0.25 | 0.000503 | 0.000299–0.000792 | 10/10 |
| Reversed target 1–10, 0.25 | 0.000517 | 0.000402–0.000852 | 10/10 |

## Reproduce

```sh
python unsigned.py --json results/unsigned_original.json
python unsigned.py --seeds 11 12 13 14 15 16 17 18 19 20 --json results/unsigned_fresh.json
python unsigned.py --polarity -1 --modes unsigned signed --json results/unsigned_reversed.json
python unsigned.py --weight .25 --json results/unsigned_followup_original.json
python unsigned.py --weight .25 --seeds 21 22 23 24 25 26 27 28 29 30 --json results/unsigned_followup_fresh.json
python unsigned.py --weight .25 --polarity -1 --modes baseline unsigned signed shuffled --json results/unsigned_followup_reversed.json
python -m unittest -v
```

Each receipt stores the teacher's inferred mapping at cutoff and the model's routing, NMSE and decoder coefficient at cutoff and at the end. The original ten baseline NMSEs reproduce the earlier run exactly. All reported runs used PyTorch 2.5.1 on CPU.
