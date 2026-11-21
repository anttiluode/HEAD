# HEAD

**Can a head that only reads an average learn to expose a hidden contrast?** Sometimes. On an eight-slot synthetic memory, two of ten default training seeds learned the address and subtraction from target error alone. A matched dot-product attention reader gave the same result. There is also an exact reader that recovers the requested slot's contrast while giving that slot **zero weight in its second reading**.

## Where the idea came from

In [SpaceScreensaver.py](https://github.com/anttiluode/phi-world-theory/blob/main/SpaceScreensaver.py), each singularity has a field. Its mean contributes to the displayed density. Pairwise field dot products, passed through tanh and averaged, modulate force. These are different readings of the same numbers. The code does not implement learned query/key/value attention.

[Sihti](https://github.com/anttiluode/Sihti) retains the difference from a coarse reading. [SaturdayAfterRHproof](https://github.com/anttiluode/SaturdayAfterRHproof) supplied the single-specimen example. Read the sum of two channels, copy the left one into both slots, and read again. Subtraction recovers the original right-minus-left contrast. The meter gained information because the copy operation can address the left slot. A permutation of the slots alone leaves the sum unchanged.

The biological analogy is access. Repositioning nucleosomes can change which parts of DNA are available to binding proteins. HEAD borrows the distinction between storing content and changing access to it. The experiment is entirely computational. [Accessing the Genome](https://www.ncbi.nlm.nih.gov/books/NBK21137/)

## The fixed-fold identity

For a linear meter w and a fixed linear intervention M, the second read obeys

    wᵀ M x = (Mᵀ w)ᵀ x.

A programmer with unrestricted access to the vector can compile the intervention into another static reader. On a physical device with a fixed meter, the intervention may still be the only available way to obtain that reading.

What if the first read chooses the intervention? In this experiment the first scalar is s = mean(x). A controller maps that scalar to normalized nonnegative access weights a(s). Multiplying each slot by n·aᵢ(s) and reusing the mean gives

    mean(n · a(s) ⊙ x) = Σᵢ aᵢ(s) xᵢ.

Choose a(s) = softmax(K tanh(Ws+b)), and the right side is exactly positional-key dot-product attention with scalar values. The mask changes a private view. The stored vector stays available to other readers. The two learned networks implement opposite sides of this equation, one with a scratch mask and the other with PyTorch's scaled dot-product attention at scale 1. They have identical initial parameters, training samples, mathematical reductions and decoder. Kernel execution details differ. This comparison establishes equivalence for this head. Dense transformer blocks and hardware throughput are separate questions.

## Learning task

Draw eight Gaussian values and subtract their mean to get a detail vector d. Draw one of eight cue levels c uniformly and construct xᵢ = c + dᵢ. Each cue selects a slot j through a random permutation fixed within a run and changed between seeds. The target is dⱼ. Each model receives only x. The controller sees its scalar mean, the reader gets a second scalar, and a learned affine decoder combines them. Training supplies the target value only. No slot label, attention map, intervention operator beyond diagonal gating, or subtraction rule is supplied as a supervision target. The architecture does supply labelled positions, a mean, and a bank of learnable positional keys.

Each run trained both models independently for 1,400 Adam updates, 192 new examples per update. Each evaluation uses 8,192 fresh examples under the same permutation. Normalized MSE divides squared error by mean squared target. Zero prediction scores 1. Numbers are CPU results with PyTorch 2.5.1.

| Runs | Scratch mask NMSE, mean | Dot-product NMSE, mean | Fixed two projections + free cue | Runs below 0.001 |
|---|---:|---:|---:|---:|
| First run, default seeds 1–5 | 0.817802 | 0.817802 | 0.713867 | 0/5 |
| Follow-up, default seeds 6–10 | 0.487638 | 0.487638 | 0.711391 | 2/5 |
| Follow-up, seeds 1–5, start second decoder coefficient positive | 0.334394 | 0.334394 | 0.713867 | 3/5 |

The first five defaults all happened to start with a negative second decoder coefficient. Reversing its initial sign rescued seeds 1, 2 and 5. The coefficient was free to change during training. Seeds 3 and 4 still failed. The sign intervention and extra seeds were investigated after seeing the first result. The below-0.001 column describes the observed split. In the five successful fits across the three rows the highest-weight slot matched the requested slot on every evaluation example. Shuffling only the cue to the router raised their NMSE to roughly 2. In the failed fits, top-slot agreement was zero and NMSE stayed around 0.8. A failed fit should not be credited with having learned the exact complement solution below.

The fixed baseline gets the address for free and two fixed linear measurements of the detail. The detail is isotropic Gaussian in the seven-dimensional zero-sum subspace. Conditional expectation given two orthogonal projections is their reconstruction in that plane. Averaged over uniformly selected target coordinates, any fixed plane recovers 2/7 of the variance, leaving population NMSE 5/7. The baseline uses this optimal decoder. It even has an extra free cue relative to the learned reader. The bound applies to fixed linear measurements on this distribution.

[First runs](results/head.json) · [Additional seeds](results/followup.json) · [Paired sign intervention](results/positive.json)

## A head can read a slot by ignoring it

Suppose the correct slot j is given and the initial mean s is retained. Two ideal accessibility masks give the same answer on every input.

| Second read | Decode the contrast xⱼ − s |
|---|---|
| Open j only, giving v = xⱼ | v − s |
| Close j and average the other n−1 slots, giving u | (n−1)(s − u) |

These hard masks are analytic controls. Finite softmax can also express two opposing exact solutions with strictly positive weights. Mix either ideal mask halfway toward the uniform mask and double its decoder gain. Both still give the same contrast. A heatmap of the second read, viewed without the first reading and decoder, can tell opposite stories about the same computation.

Now let the second scalar reading have additive error ε while the first remains exact. The direct estimate acquires error ε and the complement estimate acquires error −(n−1)ε. For eight slots the squared-error ratio is **49**. A CPU check with 32,768 examples and the same noise realization in both paths produced MSE **0.00246 vs 0.12070**. Their noiseless answers agreed to within 9e−16. This ratio is for these two ideal masks and noise placed after the second read. Noise already present in the common stored vector is a different experiment. [Meter results](results/meter.json)

## Reproduce

Python with PyTorch 2.5 or later. On Windows a local Python installation with torch is enough. Run from this directory.

```sh
python head.py --json results/head.json
python head.py --seeds 6 7 8 9 10 --json results/followup.json
python head.py --initial-sign positive --json results/positive.json
python meter.py
python -m unittest -v
```

CPU is the default. The training script also accepts `--device cuda`. All reported measurements used CPU. Tests check the two-channel identity, targets, reader outputs and their gradients with respect to inputs and every parameter, and the noise prediction.

## Related attention work

Adaptive selection of later observations is established in [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247). [EfficientViT](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.pdf) already passes one head's output to a later head. HEAD isolates a narrower question about access to scalar measurements. The algebra explains the absence of an architectural advantage here. It also gives a diagnostic for other experiments: ask whether changing the substrate does anything a different query on the original substrate could not reproduce under the actual access and noise constraints.

## Does noisy training make the head select its target?

We reran the 323-parameter mask reader with the original ten seeds, slot permutations, initializations, clean first read, minibatch streams and 1,400-step training budget. Independent Gaussian noise with standard deviation σ enters the second scalar after its read and before the decoder. Target power is 7/8 in expectation. The sign of the second coefficient remains free. We trained once with sampled perturbations and once with their exact expected loss, sharing the clean baseline. Each cell counts seeds whose top-weight slot was correct on all 8,192 held-out items. This measures routing separately from reconstruction accuracy.

| Training noise SD | Perfect routing, sampled loss | Perfect routing, exact loss | Mean clean NMSE, sampled fit | Mean noisy NMSE, sampled fit |
|---|---:|---:|---:|---:|
| 0 | 2/10 | 2/10 | 0.6527 | 0.6527 |
| 0.05 | 2/10 | 2/10 | 0.6534 | 0.6582 |
| 0.2 | 2/10 | 2/10 | 0.6762 | 0.7124 |
| 0.5 | 2/10 | 2/10 | 0.7506 | 0.8083 |
| 1 | 1/10 | 2/10 | 0.8176 | 0.8834 |

The same two seeds (6 and 8) routed every item correctly from σ=0 through 0.5. Seed 1 illustrates how the other runs usually responded: its second-read coefficient shrank from −1.39 to −0.33 at σ=0.5, with zero correct top-slot choices in both cases. At σ=1 sampled noise did produce a partial change in that seed (about 75% correct), while seed 6 lost perfect routing. Noise alone did not reliably select the direct route.

Noisy NMSE averages over held-out examples and integrates over independent evaluation noise at the corresponding training SD. For an affine decoder with second coefficient b, independent zero-mean meter error adds exactly σ²b² to expected MSE. This term penalizes both signs equally. An ideal direct read has b=1 and the ideal eight-slot complement has b=−7, which explains their 49× noise cost. The optimizer can also reduce its use of the second meter. A failed fit in this comparison is not evidence that it learned the exact complement. These runs measure routing at the end of training, with no claim about convergence at an unlimited training budget.

The biological comparison concerns how separately delivered signals can change a cell's response. In mouse auditory cortex, Zhang et al. found preferential encoding of inferred rule in apical dendrites and sensory stimuli in somata, and found that dendritic inhibition impaired rule-dependent somatic choice coding. HEAD receives its cue as an exact scalar mean, and implements neither dendrites nor feedback spikes. [Zhang et al., Science](https://doi.org/10.1126/science.adz5323)

Run `python noisy_training.py --json results/noisy_training.json` for the full CPU sweep. [Per-seed measurements](results/noisy_training.json)
