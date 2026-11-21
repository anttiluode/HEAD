# What happened to the eight failed heads?

All ten original default-seed endpoints reproduce. The eight failures retained a trainable second reading. A decoder-independent calibration repaired all eight. A simple scalar projection supplied the same information as the order probe. In a separate simulation the read wrote a persistent change into a field shared by two readers. In the compressed field, identifying the correct route still left half the source variance inaccessible.

## 1. The router was turning away

We replayed the clean, 323-parameter experiment with the original ten seeds and minibatches. At seven checkpoints we inspected the decoder coefficient β, gate weights, effective number of slots, and held-out MSE gradient with respect to all key and controller parameters. These diagnostics never update the model. The eight failures are the original endpoint group (seeds 1–5, 7, 9 and 10). Their means are:

| Updates | Decoder β | Gate weight on requested slot | Router gradient norm |
|---:|---:|---:|---:|
| 0 | −0.267 | 0.1241 | 0.0373 |
| 5 | −0.272 | 0.0912 | 0.0327 |
| 20 | −0.317 | 0.0326 | 0.0199 |
| 100 | −0.621 | 0.0062 | 0.0416 |
| 1,400 | −1.396 | 0.0107 | 0.0892 |

Uniform attention gives the target 0.125. An exact complement reader would give the target zero, average the seven remaining slots equally, and use β=−7. The failures averaged about five effective slots (inverse squared-weight sum). Strong meter noise in the separate noisy-training run attenuated β, but that does not explain these clean failures.

There is a simple reason a negative coefficient can teach avoidance. Write u for uniform weights, a for this cue's attention, and eⱼ for the target selector. The data are isotropic Gaussian detail in the zero-sum subspace. The population MSE contribution from the detail, conditional on the cue, is exactly

    || β(a − u) − (eⱼ − u) ||².

At uniform attention the derivative with respect to a is −2β(eⱼ−u). Changing β's sign reverses the direction of credit while leaving the initial detail estimate at zero in either case. β=0 really does block the router gradient. The measured trajectories instead move toward active, imperfect exclusion. This calculation describes the initial direction and the data geometry. It does not establish an inescapable local minimum or a biological mechanism.

A post-hoc check tried to recover the mapping from avoidance alone, using the known bijection. Among all 8! assignments choose the one with minimum total logit for a negative decoder (maximum for positive). This recovered 0/8 cue assignments in each failed run, 8/8 in the two successful ones. A low average weight at the true slot does not give a reliable inverse decoder.

## 2. Can a measurement before the decoder repair the original head?

Each checkpoint got the same 9,216 new labelled examples. A uniformly randomized actuator interrogated **one** coordinate per calibration example. For the order probe, F reflects the vector about its mean and B copies the selected coordinate to all slots. Subtract the two final meter readings in opposite orders and divide by two. For every vector this is exactly xᵢ−mean(x), also available by one direct coordinate read and the existing mean. The order calculation uses two copies of each example; the direct control needs only one.

Group trials by observed scalar cue and chosen actuator. In each group estimate mean(probe response × task outcome). Its population value is 7/8 for the target direction, −1/8 for any other. Selecting the largest empirical moment gives the routing table. Calibration uses the ordinary scalar task target; the fitter receives neither target index nor the seed's permutation. Without outcomes, the inputs are identical in distribution under every possible assignment.

We installed this table into each existing head with a least-squares correction to its positional keys at the eight supplied cue levels. Its trained controller stayed intact. We refit the affine decoder on calibration examples whose random probe matched the inferred route. Both changes use linear algebra without gradient updates. The hard-coded finite table representation and calibration protocol supply inductive bias. This is a supervised, task-specific repair, not spontaneous recovery by Adam.

| Treatment on the eight failures | Fully repaired seeds | Held-out normalized MSE, mean |
|---|---:|---:|
| Continue original Adam with retained momentum | 0/8 | 0.817 |
| Refit affine decoder alone with original routes | 0/8 | 0.816 |
| Local order probe, then fit keys and decoder | 8/8 | 1.28 × 10⁻¹⁰ |
| Direct local projection, same repair | 8/8 | 1.28 × 10⁻¹⁰ |
| Shuffle calibration outcomes within cue | 0/8 | 1.027 |

Full repair requires all eight held-out cue routes correct and NMSE below 0.001. Continuation and calibration share labelled examples, but have different observation access and different computational costs. The final learned network gets only its usual reads. No test labels enter calibration. [Code](credit.py) · [Full traces and controls](results/credit.json)

## 3. A read that changes someone else's future measurement

Eight independent unit-variance Gaussian source amplitudes h generate a field z = B h. The actuator directions are the unit columns of B. The 8D control uses a rotated orthogonal basis. The smeared 4D condition uses two Fourier sine/cosine pairs followed by an orthogonal rotation, giving overlapping directions with BBᵀ=2I. The scalar address cue is supplied separately, with the same random unknown cue-to-source mapping as above. This simulation tests a measurement protocol; it does not train the 323-parameter network in the new medium.

A write latches an actuator direction b. On each tick the meter returns r=bᵀz and the field updates to z−0.5rb. A neutral initial latch returns zero, so write-then-tick is distinguishable from tick-then-write and one trajectory suffices for this calibration. The same cross-moment method chooses a direction. A scalar gain is estimated from the calibration outcomes.

The first tick is exactly reproducible by a passive projection. To check for an actual side effect, replace the latch with a common **different** direction k in two copies of the specimen. One previously underwent the first read; the other did not. The later readings differ by −0.5r kᵀb. Both specimens now have identical latch settings. Any discrepancy is in the shared field. The specific write and attenuation are engineered, and the experiment tests their consequences.

| Mean over ten seeds | 8D orthogonal | 4D overlapping |
|---|---:|---:|
| Correct routes per seed | 8/8 | 8/8 |
| Held-out source NMSE | 2.1 × 10⁻³¹ | 0.4996 |
| Optimal population NMSE given the entire field | 0 | 0.5 |
| RMS change at the subsequent different probe | 1.6 × 10⁻¹⁶ | 0.2682 |

In 4D, the conditional expectation is Bᵀ(BBᵀ)⁻¹z = Bᵀz/2. Every unit source coordinate has conditional variance 0.5. For an explicit witness, add (I−B⁺B)e₀ to the original source vector. Its first source changes by 0.5 while its field stays the same to numerical precision. Given the same cue, any protocol operating only on this field and its latch has the same possible future observations in both cases. This bound follows from compression before the read. It differs from the original 49× result about noise added after a scalar measurement. Overlap alone need not destroy information if the encoding remains invertible.

[Shared-medium code](shared_medium.py) · [Measurements and nullspace witnesses](results/shared_medium.json)

## Reproduce

```sh
python credit.py --json results/credit.json
python shared_medium.py --json results/shared_medium.json
python -m unittest -v
```
