# Design of Experiment v2 — "Dreaming Together" Tank Combat

**Repo:** `physics-engine-multi-agent-team-collaboration-v2`
**Status:** TRAINING-READY for Stage 1 (proprio pathfinder). 100 tests pass;
gates G0 (physics parity), G1 (rendered-POV half), G2 (combat signal, 200
oracle episodes) and G3 (per-term gradient flow) PASS; ladder T0/T1 PASS.
Still pending before the full A/B/C experiment: Stage 0 vision pretraining
(G1 second half), diffusion policy + G4, coordinators. This file is
authoritative — it supersedes the v1 `claude_code_prompt.md` wherever they
disagree. Current state and next steps: `HANDOFF.md`.

**Lineage:**

| Generation | Repo | Outcome |
|---|---|---|
| v0 humanoid | `../physics-engine-multi-agent-team-collaboration-vla` | Abandoned. Trained before validating; agents fell, camera blinded by own shield, spawns out of weapon range. All training data deleted. |
| v1 tank physics | `../physics-engine-multi-agent-team-collaboration-vla-simplified` | Physics layer validated (61 tests pass, video per test, interactive UI). Never reached training. Three late design decisions never landed in code. |
| **v2 (this repo)** | here | Full experiment, built on the validated tank physics, under the gate discipline below. |

---

## 1. Failure ledger — what went wrong and the rule each failure produces

Every rule below is binding. They are ordered by cost of the original failure.

| # | Failure (observed) | Root cause | Rule for v2 |
|---|---|---|---|
| F1 | Multi-hour humanoid training produced agents that fell at step ~10; data deleted | Training launched before any physics validation; gait reference used `base=zeros` (bent knees) | **R1 — No training run may start before its gate (§6) passes.** Gates are executable scripts, not judgment calls. |
| F2 | Zero combat signal for an entire training run | Spawn separation 6 m, shotgun range 3 m; nobody could ever hit anyone | **R2 — "Combat is possible" is a unit test**: scripted oracle vs scripted oracle must produce hits, HP loss, and non-draw outcomes before any learning code runs. |
| F3 | Shield agent effectively blind; discovered by watching a post-training video | Camera placed without rendering what the agent actually sees | **R3 — Every camera has a rendered-POV unit test**: enemy visible at spawn, own shield occlusion below threshold, in canonical scenes. |
| F4 | Height-bonus reward never influenced behavior | Stage-1 loss was imitation-only; env-return gradient reached only the critic | **R4 — Reward plumbing test per term**: each reward term needs (a) a value-correctness unit test and (b) an end-to-end test that a synthetic nonzero advantage changes policy parameters. |
| F5 | Pellets tunneled through the shield (looked like "shield doesn't work") | Default MuJoCo contact stiffness can't stop a 0.005 kg pellet at 30 m/s; contact flips to back face past the midpoint | **R5 — Keep the validated contact recipe** (§7): `SHIELD_THICKNESS=0.25`, `solref="0.002 1"`, `solimp="0.9 0.99 0.001"`. Anti-tunneling test stays in the suite. |
| F6 | "Pellet hit hull" detector fired at step 1 forever | Hull always touches the floor; `any_contact_with(hull)` is always true | **R6 — Contact queries are always pairwise** (`contacts_between`), never "any contact". |
| F7 | Bugs (arm/pellet mismatch, dead-agent-standing, shield clipping body) found only by human video review, late | No render-first tooling early on | **R7 — Every physics test renders an MP4.** Video is a first-class test artifact, generated from day 1, reviewed at every gate. Port `render_videos.py` + tracer visuals + interactive UI. |
| F8 | Elbow removal, seg-only vision, and camera redesign were decided but never reached code; build prompt went stale | Design discussion not tied to code change | **R8 — A design decision is complete only when code + tests + this document are updated in the same session.** This document carries a Frozen Spec (§4–5); changes to the frozen spec require an explicit changelog entry. |
| F9 | Killed a 2 h run manually after asking "how many epochs, is data saved?" | No checkpoint cadence, no kill criteria, no pilot | **R9 — Pilot-then-commit**: every stage runs a ≤30 min pilot with checkpoints + auto-rendered behavior video for human review before the long run. Kill criteria (§6) are written down in advance and monitored automatically. |
| F10 | Whole locomotion curriculum built, debugged for days, then discarded as irrelevant | Complexity not required by the research claim | **R10 — Nothing enters the environment that the hypothesis does not need.** Tanks, 2-DOF arm, 1-channel vision. Any proposed addition must name the prediction (P1–P8) it serves. |

Process rules from the same history: no emoji in output; no trailing summaries; when the user says "CRITICAL: Respond with TEXT ONLY" — no tool calls.

---

## 2. Research question (unchanged — this is the part that survives)

Does **linguistic encoding** in a team coordinator outperform **continuous embedding
encoding at matched information bandwidth** (40 bits/frame both) in a 2v2 physics
combat task where inter-agent communication is causally necessary?

### Conditions

| Condition | Per-agent policy (core self) | Per-team coordinator (extended self) |
|---|---|---|
| A | Feedforward MLP | Embedding MLP |
| B | Diffusion policy (DDPM) | Embedding MLP |
| C | Diffusion policy (DDPM) | Language transformer (discrete tokens) |

- A vs B isolates the diffusion body prior.
- B vs C isolates linguistic structure at matched bandwidth.

### Independent variables

1. Condition ∈ {A, B, C}
2. Coordinator period `dt_coord` ∈ {50, 100, 250, 500, 1000, 2000} ms
   (train at 250 ms, evaluate the sweep, fine-tune per-rate variants).

### Primary metric

Win rate `W(condition, dt_coord)` over ≥1000 eval episodes per cell,
bootstrap 95% CIs (N=10,000).

### Falsifiable predictions (log everything needed for all of them)

- **P1**: W(C,250) > W(B,250) > W(A,250), p < 0.05.
- **P2**: argmax over dt_coord of W(C,·) ∈ [250, 1000] ms.
- **P3**: argmax of W(B,·) at smaller dt_coord than argmax of W(C,·).
- **P4**: bits-per-win BPW(C, dt*_C) < BPW(B, dt*_B).
- **P5**: self/world discrimination Γ differs between contact and non-contact frames
  for B and C (Welch t, p < 0.01), no significant difference for A.
- **P6**: W(C,50) ≈ W(B,50).
- **P7**: window timing error e_w minimized near the shotgun fire-cycle Nyquist period
  (~300–600 ms), coinciding with the W-maximizing dt_coord per condition.
- **P8 (transmission/learnability — the evolutionary claim)**: a fresh,
  randomly initialized listener behavior-cloned on K logged (obs, z_g, action)
  episodes recovers more team performance under condition C than under B, and
  the gap widens as K shrinks. Measured by the learnability probe (§6, G7);
  motivated by iterated-learning results (compositional protocols survive
  transmission bottlenecks; entangled ones do not).

Statistical machinery (bootstrap, Welch tests, BPW formula, Γ definition) is unchanged
from `docs/dreaming_math.pdf` in the v1 repo; the kinematics/action-space sections of
that PDF are stale and this document overrides them.

---

## 3. Environment (validated tank physics + three landed design decisions)

Arena, loadout, window mechanic, rewards, and clocks carry over from the tank edition;
the deltas below are the v1 design decisions that never reached code, now frozen.

### 3.1 Carried over unchanged

- 8 m × 8 m open arena, 1.2 m walls, mirror-symmetric spawns RED (−3, ±0.6) /
  BLUE (+3, ∓0.6) facing center, `symmetry_check` invariance under 180° rotation.
- Tank hull 0.60 × 0.40 × 0.25 m, differential drive (`left_track`, `right_track`,
  max ~1.5 m/s), no balance problem, no fall, no locomotion curriculum.
- Roles: agent 0 SHIELD (1.2 × 1.0 m shield plate, zero damage by construction),
  agent 1 SHOTGUN (8 pellets/shot, 6 HP/pellet, 8° cone, range < 3 m, period 1.2 s).
- Window mechanic: continuous, emergent. `W_q` = fraction of pellet-cone rays to the
  nearest opponent unobstructed by own shield; `C_r` = fraction of opponent
  muzzle-to-own-team rays blocked by own shield; `window_open ⇔ W_q > 0.25`.
- Friendly fire real; HP-only elimination; HP 100; one agent dead ⇒ team loses
  (decided in v0: no zombie standing bodies); 30 s episode cap ⇒ draw, penalty −5.
- Rewards: identical structure to the tank prompt (time-alive, win/loss/draw, damage,
  advance, action-rate; role-shaped shield rewards for blocking/window-timing/assist;
  shotgun rewards for kill/hit/close-range/window/friendly-fire penalty). No knockdown
  term (nothing falls). The 14 role-shaped + 4 terminal reward functions already exist
  and are unit-tested in the v1 repo — port them.
- Clocks: `dt_phys = 2 ms` < `dt_policy = 50 ms` (25 substeps) < `dt_coord`.

### 3.2 Delta 1 — Arm is 2-DOF: the elbow is removed

Decision from v1 session ("What is the point of having elbow? … Remove it."), never
applied to `tank.py`. In v2 the arm is:

- `arm_pan`: yaw about vertical, ±60°, Kp=150, Kd=15.
- `arm_tilt`: pitch about lateral, −30° to +70°, Kp=150, Kd=15.
- Fixed end-effector link (`END_EFFECTOR_LEN = 0.15 m`) carrying shield plate or muzzle.

Consequences to verify by unit test (Gate G1):
- Shield can still fully open (`W_q > 0.5`) and re-close (`W_q < 0.1`) within 400 ms
  using pan alone at combat-relevant geometries.
- Shield can still cover both teammates in column formation (`C_r > 0.5`).
- IK expert for aiming reduces to 2 joints — closed-form pan/tilt from a target point;
  `ik_expert.py` becomes ~10 lines. Muzzle-to-target alignment test: < 2 cm at ≤3 m.

### 3.3 Delta 2 — Vision is segmentation-only, 1 channel

Decision from v1 design doc `docs/design/agent-visual-observations/`:

- 64×64 × 1 channel int8 segmentation. 6 classes: background, red hull, blue hull,
  shield, pellet, arm.
- **No depth channel** — hull size is fixed, so apparent pixel area is a monotonic
  function of distance; the policy learns range implicitly.
- **No RGB** — texture/lighting carry no task signal.
- Stage 0 visual pre-training loss becomes segmentation-only (drop the
  depth-regression head); acceptance stays pixel-acc > 90% on 5k held-out frames.

### 3.4 Delta 3 — Cameras are hull-fixed; shield agent gets two

Decision from v1 design doc `docs/design/agent-cameras/`:

| Agent | Cameras | Mount | Purpose |
|---|---|---|---|
| SHOTGUN | 1 | Hull front face, z ≈ 0.35 m, forward, 90° HFOV | Locate + aim at enemy |
| SHIELD | 2 | Hull front + hull rear, z ≈ 0.35 m, 90° HFOV | Front: see threat, position shield. Rear: verify ally inside coverage cone |

- **No camera rotates with the arm.** Arm-mounted cameras create a circular dependency
  (moving the arm changes the observation → policy chases its own arm). Fixed mounts
  make credit assignment clean: pixel (u,v) → arm angle.
- Both roles' observation formats are identical *across conditions* (integrity
  constraint); they may differ *between roles* (shield has a second view).
  Encoder runs per view; shield concatenates two 256-d encodings.
- The perceptual-occlusion premise survives: cameras are 90° forward/rear only, and in
  column formation the teammate's shield fills the shotgun's forward view (HANDOFF
  detector and occlusion metrics carry over unchanged).

### 3.5 Frozen action space (5-D, uniform across roles)

`a ∈ [−1,1]^5 = [left_track, right_track, arm_pan, arm_tilt, trigger]`

- Tracks map to ±max speed; pan/tilt map to their radian ranges; `trigger > 0.5`
  fires iff cooldown elapsed; trigger is ignored for the shield role (uniform shape
  keeps the policy classes identical across roles and conditions).
- Diffusion horizon H=8 ⇒ `a^0 ∈ R^{8×5}`. Feedforward outputs the same 8×5 horizon.

### 3.6 Frozen proprioception (16-D)

| Slice | Dims |
|---|---|
| Hull heading (sinθ, cosθ) | 2 |
| Hull linear velocity, body frame (vx, vy) | 2 |
| Hull yaw rate | 1 |
| Hull position (x, y) | 2 |
| Arm joint angles (pan, tilt) | 2 |
| Arm joint velocities | 2 |
| Implement state — SHIELD: [shield_plane_angle_norm, window_open, C_r_recent]; SHOTGUN: [cooldown_norm, window_open, W_q_recent] | 3 |
| Game: [t_remaining/30, own_HP/100] | 2 |
| **Total** | **16** |

Plus role token `r ∈ {[1,0],[0,1]}` and coordinator goal `z_g ∈ R^256` (the only
cross-agent channel). Exact layout lives in `obs.py` and is asserted by a shape test.

### 3.7 Models

- Visual encoder: ResNet-10-style CNN, 1×64×64 → R^256, shared. Slightly smaller
  first conv (1 input channel).
- Diffusion policy (B, C): 1-D temporal U-Net over the horizon, hidden 128, 4 levels,
  FiLM on (k-embedding, conditioning), K=100 cosine, DDIM 8 steps at deployment,
  EMA 0.9999. ~4M params.
- Feedforward (A): MLP [c → 256 → 256 → 8×5], tanh. ~1M params.
- Coordinators: unchanged. Language: 40-token vocab, 4-layer transformer d=128,
  40 bits/frame. Embedding: d_bottleneck=5 × 8-bit = 40 bits/frame.
  `assert bits_lang == bits_embed == 40` at startup.
- Coordinator input `s_τ = MLP_in([v_0, x_prop_0, v_1, x_prop_1, game])`; for the
  shield agent `v_0` is the concatenated two-view encoding passed through a linear
  projection back to 256 so `s_τ` width is role-independent.

---

## 4. What to port from `../physics-engine-multi-agent-team-collaboration-vla-simplified`

Port and adapt — do not rewrite from scratch. Status 2026-07-03: tank.py,
rewards.py, projectiles.py (now fully implemented, ray-swept), helpers, and all
61 tests are ported and green; render_videos.py and interactive_ui.py remain.

| Asset | Adaptation needed |
|---|---|
| `dreaming_together/envs/tank.py` (MJCF builder, FK, constants) | Remove elbow joint/actuator; add hull-front camera (both roles) + hull-rear camera (shield); keep contact recipe verbatim |
| `dreaming_together/envs/rewards.py` (14 + 4 functions) | None expected; re-run its 18 tests |
| `dreaming_together/envs/projectiles.py` (ray-traced anti-tunneling pellets) | None |
| `tests/` (61 tests: reward 18, movement 11, shield 27, shooting 5) | Update every arm-pose call site from (pan, tilt, elbow) to (pan, tilt); re-derive shield poses used in fixtures; all 61 must pass again before Gate G1 closes |
| `tests/render_videos.py` (28 MP4s, `PelletTrail` tracer) | Same arm-signature update |
| `tools/interactive_ui.py` (Gradio live tuning UI) | Remove elbow slider; keep the single-background-thread renderer architecture (EGL is thread-local — see §7) |
| `tests/helpers.py` scene builders (`build_opposing_pair`, `build_friendly_pair`, `build_column_with_enemy`, `build_interception_range`) | Arm-signature update; `build_interception_range` shield yaw stays π (facing shooter) |
| `docs/dreaming_math.pdf` | Carry over; §2–§4 and the Stage-1 section are stale (humanoid) — this document overrides |

---

## 5. Repository layout

```
dreaming_together/
  envs/            combat_env.py, tank.py, projectiles.py, obs.py, rewards.py
  vision/          encoder.py, pretrain_stage0.py
  policies/        diffusion_policy.py, ddim.py, schedules.py, ff_policy.py
  coordinators/    vocab.py, language_coord.py, embedding_coord.py, bandwidth.py
  oracle/          scripted_shield.py, scripted_shotgun.py, scripted_coordinator.py
  training/        stage1_combat.py, stage2_coordination.py, stage3_selfplay.py,
                   awr_diffusion.py, ppo.py
  evaluation/      eval_winrate.py, eval_gamma.py, eval_handoff.py, eval_silencing.py,
                   eval_bpw.py
  replay/          recorder.py, episode_index.py
  analysis/        figures.py
  configs/         base.yaml, condition_{A,B,C}.yaml, rates.yaml
  tools/           budget_check.py, symmetry_check.py, smoke_test.py, replay.py,
                   ik_expert.py, interactive_ui.py
tests/             ported 61 + new gate tests (below); every physics test renders MP4
docs/              this file's supporting design docs, dreaming_math.pdf
```

Stack: Python 3.11, PyTorch ≥ 2.4 cu128 (RTX 5070 Ti, sm_120, 16 GB), mujoco ≥ 3.1.
Hyperparameters in YAML only. Pin `requirements.txt`.

---

## 6. Gated milestone plan

This replaces the old milestone list. Each gate is an executable script
(`tools/gate_gN.py` or a pytest marker) whose passing output is committed. **No stage
may consume more than 30 minutes of compute before its gate has passed (R1, R9).**

### G0 — Port gate (physics parity)
- All 61 ported tests pass with the 2-DOF arm.
- New tests: 2-DOF window open/close timing (§3.2); zero-control hull stability
  (1 mm / 10 s); arm PD hold (1° / 10 s); differential-drive turning-radius formula;
  10k-trial anti-tunneling; deterministic resets; replay bit-exact round trip.
- MP4 rendered for every physics test; human review of the video set.

### G1 — Perception gate (kills F3)
- Rendered-POV tests for all three camera mounts in canonical scenes:
  (a) enemy hull ≥ N pixels at spawn separation from the shotgun front cam;
  (b) own shield occludes < 30% of shotgun front view in column formation *when the
  window is open*, and the HANDOFF detector fires when it is closed;
  (c) shield rear cam sees ally hull in column formation;
  (d) segmentation classes present and correctly coded in all scenes.
- Stage 0 pretraining passes: seg pixel-acc > 90% on 5k held-out frames.

### G2 — Combat-signal gate (kills F2)
- Scripted oracle vs scripted oracle, 200 episodes: pellet hits > 0 in ≥ 90% of
  episodes; mean HP damage per episode > 30; non-draw rate ≥ 60%; both teams win ≥ 20%
  (symmetry sanity). Spawn-separation randomization `U(1.5, 5.0) m` stays inside
  weapon-range reachability by construction — asserted, not assumed.

### G3 — Reward-plumbing gate (kills F4)
- Per-term value tests (ported, 18) all pass.
- Gradient-flow test: for each reward term, inject a synthetic episode where only that
  term is nonzero, run one update, assert policy (not just critic) parameters change.
- Reward-scale audit: log per-term episode sums from G2 oracle play; no single shaped
  term may exceed 30% of |terminal reward| on average (prevents shaping domination).

### G4 — Policy-class gate
- Diffusion policy on toy tasks (drive to waypoint, point arm at target): smooth
  samples, 8-step DDIM within MSE 1e-3 of 100-step DDPM, Π_k monotone.
- Canary tests: overfit a single batch; cheat-channel test (policy given the answer
  must exploit it) — both per policy class, `action_dim=5`.
- **Diffusion-under-RL go/no-go**: the diffusion policy must improve from RL
  signal on the T1 aim task at a rate comparable to the feedforward policy.
  If it cannot, adopt the sample-and-select fallback (draw N action candidates
  from the diffusion policy, train a lightweight critic-based selector with
  RL) — this preserves the diffusion body prior that A-vs-B tests while
  giving RL a tractable optimization surface. Decide here, not in Stage 3.

### G5 — Stage 1: individual combat skills (was humanoid Stage 2)
- **Protocol seeding (closes the Stage 1→2 interface gap)**: all scripted
  cues (OPEN/CLOSE etc.) are delivered through the real z_g pathway by the
  scripted coordinator, emitting real vocabulary tokens (condition C) or
  the corresponding scripted embedding vectors (conditions A/B). The
  listener interface learned in Stage 1 is exactly the interface the
  learned coordinator will drive in Stage 2. Seeding must be equivalent
  across conditions (integrity constraint).
- **Pilot first (R9)**: 30 min, checkpoints every 5 min, auto-rendered video of current
  behavior at each checkpoint; human review before the full run.
- SHIELD: ≥70% projectiles blocked over 200 episodes; open/close cue response within
  400 ms in ≥80% of trials (2-DOF criteria of §3.2).
- SHOTGUN (BC from 2-DOF IK expert, then RL): ≥70% wins vs scripted bot from <3 m;
  ≥60% of shots through a scripted-open window land ≥4 pellets; friendly-block < 25%.
- **Kill criteria** (auto-monitored, stop + diagnose if hit): win/block rate vs
  scripted flat at 0 after 25% of budget; entropy collapse (H < 0.1·H_max); NaN/inf in
  any loss; policy action saturation > 95% for 10 consecutive updates.

### G6 — Stage 2: pair coordination vs scripted team
- Pilot-first as above. dt_coord = 250 ms.
- Accept: team win vs scripted duo ≥75%; coordinator entropy H > 0.5·H_max;
  friendly-fire pellets < 5%.
- **Alternating freezing**: Stage 2 opens with the coordinator training
  against frozen (Stage-1) listeners, then unfreezes — two stationary
  problems alternated instead of one non-stationary joint problem.
- **Counterfactual influence auxiliary reward** for the coordinator: KL
  between the listener's action distribution under the actual vs a
  marginalized message. Capped, identical coefficient and schedule for all
  conditions, annealed to zero before Stage 3 evaluation (scaffolds get the
  protocol born; the experiment measures which encoding lets it live).
- **Hindsight message relabeling**: episodes where the joint window event
  fired by luck are relabeled with the message that should have announced
  it, giving the coordinator dense supervised (state, message) pairs.
- **Causal-necessity gate (mandatory before any A/B/C comparison):** zeroing z_g drops
  win rate ≥ 15 points. If this fails, the communication channel is not causally
  necessary and the experiment is meaningless — redesign before proceeding.

### G7 — Stage 3: self-play + rate sweep
- Pilot-first. BC anchor on. Train at 250 ms; short per-rate fine-tunes.
- Opponent pool, checkpoint cadence, and eval-seed disjointness asserted in code.
- **Learnability probe (P8, cheap)**: periodically clone a fresh listener,
  behavior-clone it on K logged (obs, z_g, action) episodes, and measure
  recovered team performance. Identical procedure for B and C. (The
  expensive generational variant — actually replacing listeners during
  training — is a pre-registered follow-up experiment, not a core condition.)

### G8 — Evaluation
- ≥1000 episodes per (condition × rate) cell (scope ladder below if over budget).
- Produce `results/winrate.csv`, all metric figures, bootstrap CIs, Welch tests,
  `REPORT.md` with explicit P1–P7 verdicts (confirmed / refuted / underpowered).

### Trainability ladder (run inside the gates above)

Each rung is the smallest end-to-end learning problem that can expose the next
class of failure; a flat curve at rung Tn localizes the bug class. Condition A
(feedforward) is the pathfinder at every rung; matched A/B/C runs follow.

| Rung | Task | Status |
|---|---|---|
| T0 | BC-clone the closed-form IK expert; fire in real physics | **PASS** (97.5% held-out physical hit rate) |
| T1 | RL from scratch on the aim task (REINFORCE, ray-swept hits) | **PASS** (100% deterministic hit rate, 38k episodes, 31 s) |
| T2 | Moving target; shield blocking vs scripted shooter | pending |
| T3 | Pair vs scripted duo, scripted coordinator driving z_g | pending |
| T4 | Learned coordinator vs frozen listeners, then unfreeze | pending (the research risk lives here) |
| T5 | Self-play | pending |

### Compute budget

Tank envs: ~200–400 policy-steps/s/env/core, 16 envs ⇒ 2,000–6,000 steps/s aggregate.
GPU < 1.5 GB of 16 GB; replay buffer 9–18 GB CPU RAM (pinned, memmap fallback).
Wall clock ≈ 30 h total. `tools/budget_check.py` extrapolates from the pilot; if the
extrapolation exceeds budget, apply the pre-committed ladder **in order**:
1. Seeds 3 → 2.  2. Rates → {100, 250, 500, 1000}.  3. Eval 1000 → 500.
4. Condition A to 1 seed.  5. Vision 64 → 32.

---

## 7. Engineering notes (hard-won; do not re-learn)

1. **Shield contact recipe.** `SHIELD_THICKNESS = 0.25 m`, `solref="0.002 1"`,
   `solimp="0.9 0.99 0.001"` on the shield geom. Default `solref="0.02 1"` gives ~15 N
   on a 0.005 kg pellet at 30 m/s — the pellet crosses the midplane, MuJoCo flips the
   active contact to the back face (normal −x, zero restoring force), and it exits.
   Stiff `solref` = dt applies ~full velocity correction in one step; `solimp` caps it
   at 0.99·v_approach. MuJoCo combines `solimp` element-wise MAX and `solref`
   element-wise MIN across geom pairs, so setting the shield alone suffices.
2. **Pellet speed vs timestep.** 30 m/s at dt=2 ms ⇒ 0.06 m/step. Any collider thinner
   than ~2 steps of travel needs the ray-traced anti-tunneling projectile, not naive
   contacts.
3. **Contact queries.** Hull-floor contacts are always active (4 corners). Use
   pairwise `contacts_between(pellet_g, hull_g)`; `any_contact_with` is a footgun.
4. **EGL is thread-local.** One background thread owns `mujoco.Renderer` and does both
   stepping and rendering; UI callbacks only read the latest frame and set signal
   flags. Violating this yields `EGLError(EGL_BAD_ACCESS)` / GLX `BadAccess`.
5. **Offscreen framebuffer.** Set `model.vis.global_.offwidth` (and `offheight`)
   before constructing `mujoco.Renderer` for widths > 640.
6. **Scene-builder conventions.** Shield agent in `build_interception_range` must have
   yaw = π (arm extends toward the shooter). Shooter at x=0 yaw=0, target at
   x=distance yaw=π in `build_opposing_pair`.
7. **Pellet visibility.** Tracer-ammunition rendering (`PelletTrail`, 12-point fading
   tail) — a bare 30 m/s pellet is invisible at video frame rates and every shooting
   bug in v0/v1 was found visually.
8. **Ray-swept hits are authoritative (v2).** The contact recipe (note 1) stops
   near-normal pellets but let 1/200 through at oblique incidence (found by the
   randomized G0 gate). `ProjectileManager` ray-casts each pellet's per-step
   travel segment; the first crossed geom is the hit and the pellet retires.
   Tunneling is impossible by construction. Contacts remain as physical backup.
9. **The shield must be opaque (rgba alpha 1.0).** MuJoCo's segmentation
   renderer skips transparent geoms — at the v1 alpha of 0.8 the shield was
   absent from segmentation observations and agents could see straight through
   their own shield. Found by the G1 rendered-POV gate on its first run.
10. **Pin `<statistic extent>` in every model.** Parked pellet bodies at
    z=100 inflate `stat.extent` to ~100 m, and the camera near-clip plane is
    `map.znear × extent` ≈ 1 m — everything closer is silently clipped,
    including the shield in front of the shield agent's own camera. Set
    `<statistic extent="8" center="0 0 0.5"/>` and `map.znear="0.005"`.
12. **Track thrust must use site transmission.** A freejoint's translational
    dof axes are world-aligned, so joint-transmission motor gears push along
    world +x regardless of hull heading — every tank piled onto the east wall
    within 2 s of the first e2e episode. Tracks are `<velocity>` actuators on
    a hull site: the site frame rotates with the hull, gear (1,0,0,0,0,±r)
    projects true per-track surface speed, ctrl is m/s (capped at
    MAX_TRACK_SPEED), forcerange caps thrust. Regression tests drive at
    yaw 90/180/−135°.
13. **Volley pellets must not interact with each other.** 8 pellets spawn
    coincident at the muzzle: pellet-pellet contact must be masked off
    (contype 2 / conaffinity 5) and the ray sweep must skip sibling pellet
    geoms, or the volley self-collides and every pellet "hits" a pellet at
    spawn. Also: W_q counts only the shield geom as a blocker — in a tight
    on-axis column an "open" window can still put the teammate's HULL in the
    lane (red1 shot red0 in the back in an early e2e run). Scripted oracles
    offset the shield laterally from the firing lane; learned policies are
    expected to discover the same via the friendly-fire penalty.
11. **RL shaping/eval lessons from T1** (each stalled a run): shaping must be
    unsaturated everywhere (linear −miss, not `max(0, 1 − miss/0.5)`);
    normalize advantages per batch rather than using a global EMA baseline
    when episode difficulty is randomized; evaluate gates on the
    deterministic (mean) policy — stochastic eval caps measured performance
    at the exploration-noise level no matter how good the mean is.

---

## 8. Scientific integrity constraints (unchanged, restated)

- Conditions must never differ in: environment steps, network update counts, eval
  episode counts, observation content, `d_goal` as seen by the policy, bits/frame.
- Seeds ≥ 3 per condition (ladder rung 1 may reduce to 2); eval seeds disjoint from
  train seeds; asserted in code.
- `python tools/smoke_test.py` must run the whole pipeline end-to-end (tiny sizes)
  before any long run.
- Log Π_k, score norms, coordinator entropy, Γ, and per-term reward sums throughout.
- No claims about consciousness; structural/functional hypotheses only.

---

## Changelog

- 2026-07-03 — v2.4. TRAINING-READY. CombatEnv (game layer: HP, cooldowns,
  one-death-team-loss, full RewardContext plumbing, U(1.5,5.0) mirror spawns),
  scripted team oracles (drive/screen/aim/timed-windows through the real 5-D
  action interface), G2 gate PASS (200 episodes: 100% hit episodes, 181 HP
  mean damage, 78% decided, 38/40 win symmetry), G3 gradient-flow tests PASS
  (all 14 reward terms reach policy parameters), tank-collision test added.
  100 tests total.
- 2026-07-03 — v2.3. End-to-end visual test PASS (`tools/e2e_visual_test.py`,
  `videos/e2e_scripted_combat.mp4`): scripted 2v2 episode with column advance,
  86 shield blocks, timed open-fire-close windows, elimination. Found and
  fixed: world-frame track thrust (→ site-transmission velocity servos,
  notes 12), pellet volley self-collision (note 13). Arena builder
  (`envs/arena.py`) now exists.
- 2026-07-03 — v2.2. Physics layer ported and gated: 2-DOF arm live in code,
  73 tests green. New engineering notes 8–11 (ray-swept hits authoritative,
  opaque shield for segmentation, statistic-extent/znear pin, T1 RL lessons).
  T0 and T1 trainability rungs PASS; first learned-behavior video rendered.
- 2026-07-03 — v2.1. Amendments from the trainability discussion: P8
  learnability prediction, protocol seeding in G5, alternating freezing +
  influence reward + hindsight relabeling in G6, learnability probe in G7,
  diffusion-under-RL go/no-go with sample-and-select fallback in G4,
  trainability ladder T0–T5.
- 2026-07-03 — v2.0. Created from v0/v1 post-mortem. Landed the three orphaned v1
  decisions (2-DOF arm, seg-only vision, hull-fixed dual cameras) into the frozen
  spec; replaced milestone list with gates G0–G8; added failure ledger R1–R10.
