# Stage 3 Report — P1-P8 verdicts

**Evaluation protocol.** W is measured against a single frozen, calibrated scripted opponent (EliteScriptedTeam), identical for every cell. This replaces the design's original self-play W — a deliberate deviation, adopted because a fixed opponent removes co-evolution as a confound: every condition is graded against the same yardstick, so between-condition differences are attributable to the communication channel rather than to divergent opponent curricula. The cost is that W says nothing about inter-condition head-to-head play. Stacks are trained at dt_coord=250 ms and evaluated across the rate sweep without per-rate fine-tuning. 500 episodes/cell, bootstrap 95% CIs (10k resamples).

## W(condition, dt_coord)

| condition | 50 ms | 100 ms | 250 ms | 500 ms | 1000 ms | 2000 ms | zeroed@250 |
|---|---|---|---|---|---|---|---|
| A | 0.782 [0.746,0.818] | 0.770 [0.732,0.806] | 0.814 [0.780,0.848] | 0.820 [0.786,0.854] | 0.846 [0.814,0.878] | 0.740 [0.702,0.778] | 0.594 [0.552,0.636] |
| B | 0.672 [0.632,0.714] | 0.724 [0.684,0.764] | 0.694 [0.654,0.734] | 0.696 [0.656,0.736] | 0.666 [0.624,0.708] | 0.680 [0.638,0.720] | 0.528 [0.484,0.572] |
| C | 0.916 [0.890,0.940] | 0.930 [0.908,0.952] | 0.916 [0.890,0.940] | 0.922 [0.898,0.944] | 0.894 [0.866,0.920] | 0.866 [0.836,0.894] | 0.424 [0.382,0.468] |
| NC (trained deaf) | — | — | — | — | — | — | 0.734 [0.696,0.772] |

**No-communication baseline.** NC is the same FF listener architecture and training budget as condition A, trained with the channel zeroed from the first update. Contrast with the eval-time ablation (zeroed@250 column): the ablation understates deaf performance because those policies expect messages. Gap analysis (condition live − NC = value of having a channel; condition zeroed − NC = protocol-dependence penalty):

- A: channel value +0.080 (**CONFIRMED** (p=0.0013)); ablation-vs-NC -0.140
- B: channel value -0.040 (UNDERPOWERED (p=0.9237)); ablation-vs-NC -0.206
- C: channel value +0.182 (**CONFIRMED** (p=0.0000)); ablation-vs-NC -0.310

## Verdicts

**P1** W(C,250) > W(B,250) > W(A,250), evaluated across 3 independent training seeds each: **PARTIALLY CONFIRMED, REPLICATED** — C>B holds in 3/3 independent seeds; B>A is **REFUTED**, reversing (A>B) in 3/3 seeds. (Seed-1 point estimate: C=0.916, B=0.694, A=0.814; single-seed bootstrap C>B **CONFIRMED** (p=0.0000), B>A UNDERPOWERED (p=1.0000) — see the per-seed table below for why a pooled bootstrap is not the right statistic here.)

### P1 multi-seed replication

Win rate (live, 250 ms) and causal drop (pts) per seed. Seed is treated as the unit of replication: each seed is one independent training run, so its 500-episode win rate is one data point, not pooled with the others — with only 2-3 seeds the honest summary is the ordering's consistency across runs, not a bootstrap p-value over pooled episodes.

| seed | W(cond=A)/drop | W(cond=B)/drop | W(cond=C)/drop |
|---|---|---|---|
| 1 | 0.814 [0.780,0.848] / +22.0 | 0.694 [0.654,0.734] / +16.6 | 0.916 [0.890,0.940] / +49.2 |
| 2 | 0.952 [0.932,0.970] / +29.8 | 0.764 [0.728,0.802] / +17.4 | 0.880 [0.852,0.908] / +85.6 |
| 3 | 0.854 [0.824,0.884] / +9.8 | 0.722 [0.684,0.760] / -15.6 | 0.738 [0.700,0.776] / +69.2 |

Across 3 fully-evaluated seeds: C>B held in 3/3; B>A held in 0/3. Per-seed detail: seed1: C>B=True, B>A=False; seed2: C>B=True, B>A=False; seed3: C>B=True, B>A=False.


Per-seed G6 gate (pre-registered win/diversity/causal-drop bar), disclosed regardless of outcome — 4/8 seed-stacks failed it even though all seeds are included in the ordering above: As1=PASS, As2=FAIL, As3=FAIL, Bs2=PASS, Bs3=FAIL, Cs1=PASS, Cs2=PASS, Cs3=FAIL. A seed failing G6 (e.g. a negative causal drop) is a real training-run outcome, not grounds for exclusion or a rerun-until-it-passes policy.


**P2** argmax_r W(C,·) ∈ [250, 1000] ms: observed argmax at 100 ms → **REFUTED** (point estimate; CI overlap caveat applies).

**P3** argmax W(B,·) at smaller rate than argmax W(C,·): B* = 100 ms vs C* = 100 ms → TIED — NOT CONFIRMED (point estimate).

**P4** BPW(C, C*) < BPW(B, B*): 2,813 vs 5,746 bits/win → **CONFIRMED** (point estimate).

**P5** Γ self/world discrimination: **DROPPED** — the Γ instrument (contact vs non-contact denoising-error analysis) was not built; the prediction is withdrawn from this study's claims rather than deferred.

**P6** W(C,50) ≈ W(B,50): observed 0.916 vs 0.672 (two-sided bootstrap p=0.0000) → **REFUTED** (difference persists at 50 ms).

**P7** window-timing proxy: mean window-open fraction by rate (condition C): 50ms=0.89, 100ms=0.89, 250ms=0.89, 500ms=0.90, 1000ms=0.90, 2000ms=0.95. Full e_w (cue-to-open latency) instrument not built; **PARTIAL** — see caveats.

**P8** protocol learnability (fresh listeners BC'd on K episodes, evaluated with the original coordinator):

| condition | K | clone team win rate |
|---|---|---|
| C | 10 | 0.907 |
| C | 50 | 0.847 |
| B | 10 | 0.667 |
| B | 50 | 0.653 |

C−B recovery gap: K=10 → +0.240, K=50 → +0.193. P8 (C recovers more, gap wider at small K): **CONFIRMED** (point estimates, N=150 eval eps).

## Caveats (read before citing)

- Fixed-opponent W, not self-play W: every condition faces the same frozen scripted opponent. Removes co-evolution confounds; does not measure inter-condition head-to-head.
- 3 training seed(s) per condition evaluated at 250 ms (P1); the design's 3-seed requirement is met for the headline ordering. The rate sweep (P2, P3, P6, P7) and BPW (P4) remain seed-1 only — replicating those across seeds was out of scope for this pass.
- Stacks trained at 250 ms only; rate sweep is evaluation-time (per-rate fine-tunes were cut per the scope ladder).
- P5 dropped, not deferred — no instrument was built and none is claimed. P7 proxy only.
- Diffusion conditions use the sample-and-select architecture (frozen prior + learned scorer) — G4's pre-registered fallback.
