# Dreaming Together — Status Report

**2v2 physics combat testbed where team communication is provably causally
necessary, built to compare linguistic vs continuous coordination at exactly
matched bandwidth (40 bits/frame).**

Status as of 2026-07-05: **apparatus fully certified; preliminary results
favor language.** The measured experiment (Stage 3) has not yet run.

---

## Executive summary

Two teams of two tanks (a shield-bearer and a shotgun-bearer each) fight in
a walled MuJoCo arena. The only cross-agent channel is a per-team
coordinator emitting exactly 40 bits every 250 ms — as **8 discrete tokens
from a 32-word vocabulary** (condition C) or as a **quantized 5-d continuous
vector** (conditions A/B). The shield agent cannot observe its teammate's
fire-readiness, so window timing must travel through the channel: silencing
it costs **17.5–47 win-rate points** depending on the stack.

| Gate | What it certifies | Status |
|---|---|---|
| G0 | Physics parity: 100+ unit tests, video per test | PASS |
| G1 | Perception: rendered-POV tests + encoder 98.5% pixel-acc | PASS |
| G2 | Combat signal across the spawn distribution (200 oracle episodes) | PASS |
| G3 | Every reward term's gradient reaches policy parameters | PASS |
| G4 | Diffusion policy class (incl. RL go/no-go) | PASS |
| G6 | Coordination + causal necessity, per condition | PASS ×3 |

## The certified grid (Stage 2 bring-up)

All numbers: greedy evaluation vs a frozen calibrated opponent, 200
episodes. Diffusion conditions use the sample-and-select architecture
(frozen distilled diffusion prior, K=8 candidates, learned scorer) — the
pre-registered G4 fallback, invoked after direct fine-tuning failed to
scale (a methodological finding in itself).

| Condition | Stack | Win (channel live) | Win (silenced) | Causal drop |
|---|---|---|---|---|
| A | feedforward + embedding | 0.825 | 0.555 | 27.0 pts |
| B | diffusion + embedding | 0.75 | 0.55 | 20.0 pts |
| C | diffusion + language | **0.92** | 0.45 | **47.0 pts** |

> **Caveats box — read before quoting.** Single-seed bring-up results
> against a scripted opponent, obtained BEFORE Stage 3 self-play and the
> pre-registered evaluation grid. Directional signals, not conclusions.
> The measured experiment (≥1000 episodes per condition × rate cell,
> bootstrap CIs, P1–P8 verdicts) is the roadmap item after this report.

## Demo index

| Folder | What you'll see |
|---|---|
| `demo/01_what_is_this/` | A scripted 2v2 episode (MP4) + 9 arrow-labeled frames: shields blocking volleys, the timed firing-window mechanic, an elimination |
| `demo/02_learning_to_fight/` | The first learned behavior (aim task), then the vision shotgun's learning arc: update 0 (6% wins, fumbling) → update 61 (34%) → final (71%, camera-guided hunting) |
| `demo/03_communication_matters/` | The killer pair: **same spawn**, coordinated team wins in 4.4 s (`comm_on.mp4`); with the channel zeroed the same team grinds to defeat (`comm_off.mp4`) |
| `demo/04_numbers/` | Stage 2 cross-condition summary with emerging findings |

## Emerging findings (bring-up grade)

1. **Language beats embedding within both policy classes** at identical
   bandwidth (FF: 0.92 vs 0.825; diffusion: 0.92 vs 0.75) — the direction
   of prediction P1, observed twice.
2. **Diffusion policies are more communication-dependent** than
   feedforward ones (47 vs 17.5-pt causal drops within the language
   condition) — prediction P5's structural claim materializing.
3. **Discreteness confers protocol robustness**: imperfectly imitated
   discrete messages snap back to the vocabulary; continuous near-misses
   confused listeners trained on exact vectors.
4. **Diffusion sampling diversity is load-bearing**: win rate peaks at
   deployment noise 0.4 and degrades when made deterministic, while the
   causal drop grows with diversity — the stochasticity is exploited
   through coordination, not despite it.
5. **Method: freeze the generator, learn a selector.** Direct RL
   fine-tuning of diffusion policies collapsed under multi-agent variance
   across 12+ attempts; sample-and-select passed both conditions on the
   first try.

## Methodology (how this was built without burning weeks)

The project runs on a failure ledger (R1–R10) distilled from two failed
predecessor codebases, enforced as executable gates: no training run
before its gate passes, every camera has a rendered-POV test, every reward
term has a gradient-flow test, every physics test renders an MP4, kill
criteria abort doomed runs in minutes. The ledger caught real bugs on
first contact throughout (world-frame thrust, segmentation-invisible
shields, reward hacking spotted by video review, near-clip blindness from
parked projectile bodies). See `DESIGN_OF_EXPERIMENT.md` for the ledger,
gates, and the frozen experimental spec; `HANDOFF.md` for the running
build journal.

## Roadmap

1. **Stage 3**: self-play training per condition at dt_coord = 250 ms.
2. **Rate sweep**: dt_coord ∈ {50, 100, 250, 500, 1000, 2000} ms.
3. **Evaluation grid**: ≥1000 episodes per (condition × rate) cell,
   bootstrap 95% CIs, Welch tests.
4. **REPORT.md**: formal verdicts on predictions P1–P8 (including P8, the
   protocol-learnability/transmission probe added to test the
   evolutionary claim).
