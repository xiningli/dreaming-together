# Dreaming Together — tank-combat testbed for language vs embedding coordination

A 2v2 MuJoCo combat environment where **team communication is provably
causally necessary**, built to test whether **linguistic encoding beats
continuous embedding encoding at exactly matched bandwidth** (40
bits/frame). Experimental companion to the ALIFE 2026 paper *"Dreaming
Itself."*

## See it

**The game** — shields screen volleys until the coordinator calls a firing
window (scripted demo; labeled frame-by-frame in `demo/01_what_is_this/`):

<img src="demo/media/e2e_combat.gif" width="480" alt="scripted 2v2 episode"/>

*How to read it: **red is the learned team** (the one under study);
**blue is a fixed scripted opponent**, identical in every clip. The tall
plates are shields that physically stop pellets; yellow streaks are
shotgun volleys; floating green bars are health. A team loses when either
of its tanks is destroyed.*

**Communication is causally necessary.** Below is the fully trained
language-coordinated team (diffusion policies + an 8-token transformer
coordinator) playing the **same spawn against the same opponent twice**.
The only difference: on the left **red's** 40-bit language channel is
live — you can read red's coordinator messages at the bottom of the clip
(`OPEN_WINDOW READY RIGHT NEAR EXPOSED` …). On the right red's channel is
zeroed out. Blue never has a learned channel; its scripted behavior is
identical in both clips.

| **Channel live — wins 92% of episodes** (this one in 4.4 s) | **Channel silenced — wins only 45%** (this one: a losing grind) |
|---|---|
| <img src="demo/media/comm_on.gif" width="440" alt="coordinated team wins 92%"/> | <img src="demo/media/comm_off.gif" width="440" alt="silenced team wins 45%"/> |

Win rates are from 200 evaluation episodes each against the frozen
calibrated opponent — silencing the channel costs this team **47
win-rate points**, because the shield-bearer cannot otherwise know when
its shotgun is ready to fire through a briefly-opened window.

Full-resolution MP4s: [coordinated](demo/03_communication_matters/comm_on.mp4) ·
[silenced](demo/03_communication_matters/comm_off.mp4) ·
[scripted episode](demo/01_what_is_this/e2e_scripted_combat.mp4) —
and the whole learning arc in [`demo/`](demo/).

**Start here:**

- [`REPORT_STATUS.md`](REPORT_STATUS.md) — current status, certified
  results, demo index, roadmap (2-minute read).
- [`demo/`](demo/) — watch it: the game explained in labeled frames, a
  policy learning to fight from a camera, and the same team winning with
  its channel live vs losing with it silenced.
- [`DESIGN_OF_EXPERIMENT.md`](DESIGN_OF_EXPERIMENT.md) — the failure
  ledger, gates, frozen spec, and hard-won engineering notes.
- [`results/STAGE2_SUMMARY.md`](results/STAGE2_SUMMARY.md) — the numbers.

**Reproduce:**

```bash
python -m pytest tests/ -q                        # 100+ physics/env/gate tests
python tools/e2e_visual_test.py                   # scripted 2v2 episode + video
python tools/gate_g2.py --episodes 200            # combat-signal gate
python -m dreaming_together.training.stage1_combat --workers 8   # Stage 1
python -m dreaming_together.training.stage2_coordination --condition A --workers 8
python -m dreaming_together.training.stage2_diffusion --condition C --seed 1 --workers 8
```

Stack: Python 3.11+, MuJoCo ≥ 3.9, PyTorch ≥ 2.4 (CUDA), single
consumer GPU (developed on an RTX 5070 Ti, <2 GB VRAM used).

Training checkpoints and raw progress videos are not tracked in git
(`runs/`, `videos/`); the curated `demo/` folder carries the presentable
artifacts.
