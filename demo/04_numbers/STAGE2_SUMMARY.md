# Stage 2 cross-condition summary (bring-up, pre-Stage-3)

All numbers: deterministic/greedy evaluation, frozen EliteScriptedTeam
opponent, 200 episodes, seeds 930000+, deployment noise 0.40 for diffusion
stacks. NOT the measured experiment (that is Stage 3); bring-up results.

FINAL (2026-07-05, sample-and-select architecture for diffusion
conditions — G4's pre-registered fallback, invoked after direct
fine-tuning failed to scale; frozen distilled prior + learned scorer):

| Stack | Win (z_g on) | Win (z_g off) | Causal drop | G6 |
|---|---|---|---|---|
| FF + language (pathfinder) | 0.92 | 0.745 | 17.5 | PASS |
| A: FF + embedding | 0.825 | 0.555 | 27.0 | PASS |
| B: diffusion + embedding (SAS) | 0.75 | 0.55 | 20.0 | PASS |
| C: diffusion + language (SAS) | 0.92 | 0.45 | 47.0 | PASS |

Bring-up ordering: C ≥ (FF+lang) > A > B — language beats embedding
within BOTH policy classes. (Historical direct-fine-tuning attempts and
the superseded B waiver are preserved in git-less run dirs and the
handoff; the earlier table below is retained for the record.)

Superseded (direct fine-tuning era):

| Stack | Win (z_g on) | Win (z_g off) | Causal drop | G6 |
|---|---|---|---|---|
| C: diffusion + language (direct) | 0.79 | 0.095 | 69.5 | PASS (superseded) |
| B: diffusion + embedding (direct) | 0.59 | 0.33 | 26.5 | waiver (superseded) |

Emerging observations (single-seed, bring-up grade):
1. Language > embedding in team competence at matched 40 bits, in BOTH
   policy classes (0.92>0.825 FF; 0.79>0.59 diffusion) — P1 direction.
2. Diffusion stacks are far more communication-dependent than FF stacks
   (69.5 vs 17.5 within language) — P5 territory.
3. Mechanism candidates for (1), observed during bring-up:
   discrete messages snap-to-vocabulary under imperfect imitation
   (continuous near-misses confused frozen listeners: B run-1 baseline
   0.25 vs listener skill 0.59, fixed only partially by jitter training);
   listeners discriminate token-chunk z-geometry more easily than a
   smooth rank-5 manifold.
4. Diffusion sampling diversity is load-bearing: condition-C win peaks at
   deployment noise 0.4 (0.84 val) and degrades when made deterministic
   (0.555 at 0.05); causal drop GROWS with diversity (77 pts) — the
   diversity is exploited through coordination, not chaos.

Known issue: rare MuJoCo QACC instability warning (1 episode in ~400,
tanks wedging under thrust) — benign so far, watch in Stage 3.
