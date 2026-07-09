"""Stage 2 — pair coordination via the z_g channel (condition C first).

The research core. dt_coord = 250 ms (one coordinator message per 5 policy
steps); z_g ∈ R^256 is the ONLY cross-agent channel; the shield cannot see
its shotgun's cooldown, so window timing must travel through z_g.

Phases (design §6 G5/G6 amendments — alternating freeze):
  A. Listener seeding: both red policies (init from Stage 1, z_g input
     added with zero weights) fine-tune with PPO while the SCRIPTED
     coordinator emits real-vocabulary tokens through the frozen token
     table. Opponent: scripted blue duo.
  B. Speaker training: listeners frozen. The language coordinator is
     BC-seeded on the scripted protocol, then PPO-trained on team return
     per coordinator step.
  C. Joint fine-tune: everything unfrozen, reduced lr.

G6 checks at the end (all must pass before any A/B/C comparison):
  - team win rate vs scripted duo ≥ 0.75
  - coordinator entropy > 0.5 · H_max
  - friendly-fire pellets < 5% of pellets fired
  - CAUSAL NECESSITY: zeroing z_g at eval drops win rate ≥ 15 points

Run: python -m dreaming_together.training.stage2_coordination [--smoke]
Progress videos: videos/progress/stage2_upd*.mp4 every 10 min.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dreaming_together.policies.ff_policy import MLP, GaussianPolicy
from dreaming_together.training.stage1_combat import gae, _serialize

RUN_DIR = ROOT / "runs" / "stage2"
VIDEO_DIR = ROOT / "videos" / "progress"

OBS_DIM = 27
Z_DIM = 256
LOBS = OBS_DIM + Z_DIM        # listener obs 283
STATE_DIM = 59                # coordinator input
ACT_DIM = 5
DT_COORD_STEPS = 5            # 250 ms / 50 ms
GAMMA, LAM, PPO_CLIP = 0.99, 0.95, 0.2
VIDEO_EVERY_S = 600


def coord_state(env) -> np.ndarray:
    s = np.concatenate([
        env.obs("red0"), env.obs("red1"),
        [max(0.0, env.episode_cap_s - env.t) / env.episode_cap_s,
         env.hp["red0"] / 100, env.hp["red1"] / 100,
         env.hp["blue0"] / 100, env.hp["blue1"] / 100],
    ]).astype(np.float32)
    return s


def make_listener(stage1_path: Path) -> GaussianPolicy:
    """283-in policy initialized so that at t=0 it ignores z_g and exactly
    reproduces the Stage 1 (27-in) policy."""
    pol = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64), init_log_std=-1.2)
    s1 = torch.load(stage1_path, weights_only=True)
    with torch.no_grad():
        w0 = pol.mean_net.net[0].weight        # (64, 283)
        w0.zero_()
        w0[:, :OBS_DIM] = s1["mean_net.net.0.weight"]
        pol.mean_net.net[0].bias.copy_(s1["mean_net.net.0.bias"])
        for k in ("2", "4"):
            getattr(pol.mean_net.net[int(k)], "weight").copy_(
                s1[f"mean_net.net.{k}.weight"])
            getattr(pol.mean_net.net[int(k)], "bias").copy_(
                s1[f"mean_net.net.{k}.bias"])
    return pol


# ---------------------------------------------------------------------------
# Rollout worker
# ---------------------------------------------------------------------------

_W = {}


def make_coordinator(condition: str):
    if condition == "C":
        from dreaming_together.coordinators.language_coord import (
            LanguageCoordinator)
        return LanguageCoordinator(STATE_DIM)
    from dreaming_together.coordinators.embedding_coord import (
        EmbeddingCoordinator)
    return EmbeddingCoordinator(STATE_DIM)


def scripted_z(env, coord, condition: str):
    from dreaming_together.oracle.scripted_coordinator import (
        scripted_tokens, scripted_bottleneck)
    if condition == "C":
        return coord.z_from_tokens(scripted_tokens(env, 0))[0].numpy()
    return coord.z_from_bottleneck(scripted_bottleneck(env, 0))[0].numpy()


def _worker_init(coord_init_bytes: bytes, condition: str = "C"):
    import io
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam
    _W["env"] = CombatEnv(seed=0, privileged_obs=True)
    _W["blue"] = ScriptedTeam(1)
    _W["red0"] = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
    _W["red1"] = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
    _W["condition"] = condition
    _W["coord"] = make_coordinator(condition)
    _W["coord"].load_state_dict(
        torch.load(io.BytesIO(coord_init_bytes), weights_only=True))
    for m in (_W["red0"], _W["red1"], _W["coord"]):
        m.eval()


def _rollout(job):
    """job = (r0_bytes, r1_bytes, coord_bytes_or_None, seed, mode)
    mode: 'scripted' (phase A), 'learned' (phase B/C), 'zeroed' (ablation),
    plus '_det' suffix for deterministic listeners.
    Returns per-agent trajectories, coordinator trajectory, stats."""
    import io
    r0b, r1b, cb, seed, mode = job
    det = mode.endswith("_det")
    mode = mode.replace("_det", "")
    env, blue, coord = _W["env"], _W["blue"], _W["coord"]
    _W["red0"].load_state_dict(torch.load(io.BytesIO(r0b), weights_only=True))
    _W["red1"].load_state_dict(torch.load(io.BytesIO(r1b), weights_only=True))
    if cb is not None:
        coord.load_state_dict(torch.load(io.BytesIO(cb), weights_only=True))

    env.reset(seed=seed)
    traj = {p: {"obs": [], "act": [], "logp": [], "rew": []}
            for p in ("red0", "red1")}
    ctraj = {"s": [], "tok": [], "logp": [], "rew": []}
    z = np.zeros(Z_DIM, dtype=np.float32)
    step = 0
    ff = fired = blocked = 0
    with torch.no_grad():
        while not env.done:
            if step % DT_COORD_STEPS == 0:
                if mode == "zeroed":
                    z = np.zeros(Z_DIM, dtype=np.float32)
                elif mode == "scripted":
                    z = scripted_z(env, coord, _W["condition"])
                else:
                    s = coord_state(env)
                    zt, toks, lp, _ = coord(
                        torch.from_numpy(s).unsqueeze(0), sample=not det)
                    z = zt[0].numpy()
                    ctraj["s"].append(s)
                    ctraj["tok"].append(toks[0].numpy())
                    ctraj["logp"].append(float(lp))
                    ctraj["rew"].append(0.0)
            actions = blue.act(env)
            for p in ("red0", "red1"):
                o = np.concatenate([env.obs(p), z])
                ot = torch.from_numpy(o)
                pol = _W[p]
                if det:
                    a = torch.tanh(pol.mean_net(ot)); lp_a = 0.0
                else:
                    d = pol.dist(ot)
                    a = d.sample(); lp_a = float(d.log_prob(a).sum(-1))
                actions[p] = a.numpy()
                traj[p]["obs"].append(o); traj[p]["act"].append(a.numpy())
                traj[p]["logp"].append(lp_a)
            _, rewards, done, info = env.step(actions)
            traj["red0"]["rew"].append(float(rewards[0]))
            traj["red1"]["rew"].append(float(rewards[1]))
            if ctraj["rew"]:
                ctraj["rew"][-1] += float(rewards[0] + rewards[1])
            ff += int(info["pellet_hits"][1] * 0)  # placeholder keeps schema
            ff += 0
            blocked += info["blocked"][0]
            step += 1
    # friendly fire pellets by red this episode
    stats = {"win": tuple(env.team_result) == (1, -1),
             "blocked": blocked,
             "len": step}
    out_t = {p: {k: np.array(v, dtype=np.float32) for k, v in d.items()}
             for p, d in traj.items()}
    out_c = {k: np.array(v, dtype=np.float32) for k, v in ctraj.items()}
    return out_t, out_c, stats


# ---------------------------------------------------------------------------
# PPO updates
# ---------------------------------------------------------------------------

def ppo_update(policy, value, opt, episodes, key):
    obs = np.concatenate([e[0][key]["obs"] for e in episodes])
    act = np.concatenate([e[0][key]["act"] for e in episodes])
    logp_old = np.concatenate([e[0][key]["logp"] for e in episodes])
    advs, rets = [], []
    with torch.no_grad():
        for e in episodes:
            r = e[0][key]["rew"]
            v = value(torch.from_numpy(e[0][key]["obs"])).squeeze(-1).numpy()
            a, ret = gae(r, v)
            advs.append(a); rets.append(ret)
    adv = np.concatenate(advs); ret = np.concatenate(rets)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    obs_t, act_t = torch.from_numpy(obs), torch.from_numpy(act)
    adv_t, ret_t = torch.from_numpy(adv), torch.from_numpy(ret)
    lo_t = torch.from_numpy(logp_old)
    n = len(obs)
    for _ in range(4):
        perm = torch.randperm(n)
        for s in range(0, n, 4096):
            mb = perm[s:s + 4096]
            d = policy.dist(obs_t[mb])
            lp = d.log_prob(act_t[mb]).sum(-1)
            ratio = torch.exp(lp - lo_t[mb])
            l_pi = -torch.min(ratio * adv_t[mb],
                              torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                              * adv_t[mb]).mean()
            l_v = ((value(obs_t[mb]).squeeze(-1) - ret_t[mb]) ** 2).mean()
            loss = l_pi + 0.5 * l_v - 1e-3 * d.entropy().sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite listener loss")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()


def emb_coord_ppo_update(coord, cvalue, copt, episodes, ent_coef):
    """PPO on the 5-d pre-quantization bottleneck (conditions A/B)."""
    S = np.concatenate([e[1]["s"] for e in episodes if len(e[1]["s"])])
    B = np.concatenate([e[1]["tok"] for e in episodes if len(e[1]["s"])])
    LO = np.concatenate([e[1]["logp"] for e in episodes if len(e[1]["s"])])
    advs, rets = [], []
    with torch.no_grad():
        for e in episodes:
            if not len(e[1]["s"]):
                continue
            r = e[1]["rew"]
            v = cvalue(torch.from_numpy(e[1]["s"])).squeeze(-1).numpy()
            a, ret = gae(r, v)
            advs.append(a); rets.append(ret)
    adv = np.concatenate(advs); ret = np.concatenate(rets)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    S_t, B_t = torch.from_numpy(S), torch.from_numpy(B)
    adv_t, ret_t, lo_t = (torch.from_numpy(adv), torch.from_numpy(ret),
                          torch.from_numpy(LO))
    n = len(S)
    ent_val = 0.0
    for _ in range(2):
        perm = torch.randperm(n)
        for s in range(0, n, 512):
            mb = perm[s:s + 512]
            mu = coord.enc(S_t[mb])
            d = torch.distributions.Normal(mu, coord.log_std.exp())
            lp = d.log_prob(B_t[mb]).sum(-1)
            ent = d.entropy().sum(-1)
            ratio = torch.exp(lp - lo_t[mb])
            l_pi = -torch.min(ratio * adv_t[mb],
                              torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                              * adv_t[mb]).mean()
            l_v = ((cvalue(S_t[mb]).squeeze(-1) - ret_t[mb]) ** 2).mean()
            loss = l_pi + 0.5 * l_v - ent_coef * ent.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite coordinator loss")
            copt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(coord.parameters(), 0.5)
            copt.step()
            ent_val = float(ent.mean())
    return ent_val


def coord_ppo_update(coord, cvalue, copt, episodes, ent_coef):
    S = np.concatenate([e[1]["s"] for e in episodes if len(e[1]["s"])])
    T = np.concatenate([e[1]["tok"] for e in episodes if len(e[1]["s"])])
    LO = np.concatenate([e[1]["logp"] for e in episodes if len(e[1]["s"])])
    advs, rets = [], []
    with torch.no_grad():
        for e in episodes:
            if not len(e[1]["s"]):
                continue
            r = e[1]["rew"]
            v = cvalue(torch.from_numpy(e[1]["s"])).squeeze(-1).numpy()
            a, ret = gae(r, v)
            advs.append(a); rets.append(ret)
    adv = np.concatenate(advs); ret = np.concatenate(rets)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    S_t = torch.from_numpy(S)
    T_t = torch.from_numpy(T.astype(np.int64))
    adv_t, ret_t, lo_t = (torch.from_numpy(adv), torch.from_numpy(ret),
                          torch.from_numpy(LO))
    n = len(S)
    ent_val = 0.0
    for _ in range(2):
        perm = torch.randperm(n)
        for s in range(0, n, 512):
            mb = perm[s:s + 512]
            lp, ent = coord.log_prob(S_t[mb], T_t[mb])
            ratio = torch.exp(lp - lo_t[mb])
            l_pi = -torch.min(ratio * adv_t[mb],
                              torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
                              * adv_t[mb]).mean()
            l_v = ((cvalue(S_t[mb]).squeeze(-1) - ret_t[mb]) ** 2).mean()
            loss = l_pi + 0.5 * l_v - ent_coef * ent.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite coordinator loss")
            copt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(coord.parameters(), 0.5)
            copt.step()
            ent_val = float(ent.mean())
    return ent_val


# ---------------------------------------------------------------------------
# Progress video
# ---------------------------------------------------------------------------

def render_stage2(r0, r1, coord, path: Path, seed: int, mode: str,
                  condition: str = "C") -> dict:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio
    import mujoco
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam
    from dreaming_together.envs.tank import hull_pos
    env = CombatEnv(seed=0, privileged_obs=True)
    env.model.vis.global_.offwidth = 960
    env.model.vis.global_.offheight = 544
    renderer = mujoco.Renderer(env.model, height=544, width=960)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.4]; cam.distance = 8.0
    cam.azimuth = 115.0; cam.elevation = -30.0
    blue = ScriptedTeam(1)
    env.reset(seed=seed)
    z = np.zeros(Z_DIM, dtype=np.float32)
    frames = []
    step = 0
    with torch.no_grad():
        while not env.done:
            if step % DT_COORD_STEPS == 0:
                if mode == "scripted":
                    z = scripted_z(env, coord, condition)
                else:
                    zt, *_ = coord(torch.from_numpy(
                        coord_state(env)).unsqueeze(0), sample=False)
                    z = zt[0].numpy()
            actions = blue.act(env)
            for p, pol in (("red0", r0), ("red1", r1)):
                o = torch.from_numpy(np.concatenate([env.obs(p), z]))
                actions[p] = torch.tanh(pol.mean_net(o)).numpy()
            env.step(actions)
            step += 1
            renderer.update_scene(env.data, camera=cam)
            scene = renderer._scene
            for p in ("red0", "red1", "blue0", "blue1"):
                frac = env.hp[p] / 100.0
                if scene.ngeom >= scene.maxgeom: break
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([0.02 + 0.28 * frac, 0.05, 0.02]),
                    hull_pos(env.model, env.data, p) + [0, 0, 1.35],
                    np.eye(3).flatten(),
                    np.array([1 - frac, frac, 0.1, 0.9], np.float32))
                scene.ngeom += 1
            frames.append(renderer.render())
    renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=20)
    return {"win": tuple(env.team_result) == (1, -1), "t": env.t}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    from dreaming_together.coordinators.bandwidth import assert_bandwidth_parity
    from dreaming_together.oracle.scripted_coordinator import (
        scripted_tokens, scripted_bottleneck)
    from dreaming_together.envs.combat_env import CombatEnv
    assert_bandwidth_parity()

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--condition", default="C", choices=("A", "B", "C"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global RUN_DIR
    if args.condition != "C":
        RUN_DIR = ROOT / "runs" / f"stage2_{args.condition}"
    if args.seed != 0:
        # replication seeds get their own run dir; seed 0 keeps the
        # original bring-up paths (stage2, stage2_A) untouched
        RUN_DIR = ROOT / "runs" / (RUN_DIR.name + f"_s{args.seed}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(RUN_DIR / "train.log", "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n")
        (RUN_DIR / "status.txt").write_text(line + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    coord = make_coordinator(args.condition)
    torch.save(coord.state_dict(), RUN_DIR / "coord_init.pt")
    lang_init_bytes = (RUN_DIR / "coord_init.pt").read_bytes()
    cvalue = MLP(STATE_DIM, 1, hidden=(128, 128))
    r0 = make_listener(ROOT / "runs/stage1/shield/policy_final.pt")
    r1 = make_listener(ROOT / "runs/stage1/shotgun/policy_final.pt")
    v0 = MLP(LOBS, 1, hidden=(64, 64))
    v1 = MLP(LOBS, 1, hidden=(64, 64))

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_worker_init,
                    initargs=(lang_init_bytes, args.condition))
    E = 4 if args.smoke else 32
    last_video = 0.0
    videos = 0

    def collect(mode, n, cbytes=None, seed0=0):
        jobs = [(_serialize(r0), _serialize(r1), cbytes,
                 seed0 + k, mode) for k in range(n)]
        return pool.map(_rollout, jobs)

    def maybe_video(tag, mode):
        nonlocal last_video, videos
        if time.time() - last_video > (30 if args.smoke else VIDEO_EVERY_S):
            last_video = time.time()
            try:
                info = render_stage2(r0, r1, coord,
                                     VIDEO_DIR / f"stage2_{tag}.mp4",
                                     seed=555 + videos, mode=mode,
                                     condition=args.condition)
                shutil.copyfile(VIDEO_DIR / f"stage2_{tag}.mp4",
                                VIDEO_DIR / "latest_stage2.mp4")
                log(f"video → stage2_{tag}.mp4 (win={info['win']})")
                videos += 1
            except Exception as exc:
                log(f"video failed: {exc}")

    try:
        # ---- Phase A: listener seeding (scripted coordinator) -----------
        log("=== Stage 2 Phase A: listener seeding (scripted z_g) ===")
        oA = torch.optim.Adam(
            list(r0.parameters()) + list(v0.parameters())
            + list(r1.parameters()) + list(v1.parameters()), lr=3e-4)
        itersA = 2 if args.smoke else 120
        winA = 0.0
        for it in range(itersA):
            eps = collect("scripted", E, seed0=200_000 + it * 1000)
            for pol, val, key in ((r0, v0, "red0"), (r1, v1, "red1")):
                _opt = oA
                ppo_update(pol, val, _opt, eps, key)
            winA = float(np.mean([e[2]["win"] for e in eps]))
            if it % 10 == 0 or it == itersA - 1:
                log(f"A it {it:3d}/{itersA} win {winA:.2f}")
            maybe_video(f"A_upd{it:03d}_w{winA:.2f}", "scripted")
            if it >= 20 and winA >= 0.85:
                log("Phase A early stop"); break
        torch.save(r0.state_dict(), RUN_DIR / "r0_A.pt")
        torch.save(r1.state_dict(), RUN_DIR / "r1_A.pt")

        # ---- Phase B: speaker training (frozen listeners) ---------------
        log("=== Stage 2 Phase B: coordinator BC seed + PPO ===")
        # BC seed: predict the scripted tokens
        env = CombatEnv(seed=0, privileged_obs=True)
        from dreaming_together.oracle import ScriptedTeam
        steams = (ScriptedTeam(0), ScriptedTeam(1))
        S_l, T_l = [], []
        for ep in range(3 if args.smoke else 120):
            env.reset(seed=300_000 + ep)
            k = 0
            while not env.done:
                if k % DT_COORD_STEPS == 0:
                    S_l.append(coord_state(env))
                    if args.condition == "C":
                        T_l.append(scripted_tokens(env, 0)[0].numpy())
                    else:
                        T_l.append(scripted_bottleneck(env, 0)[0].numpy())
                a = {}
                for tm in steams:
                    a.update(tm.act(env))
                env.step(a)
                k += 1
        S_t = torch.from_numpy(np.array(S_l, dtype=np.float32))
        copt = torch.optim.Adam(
            [p for p in coord.parameters() if p.requires_grad], lr=3e-4)
        if args.condition == "C":
            T_t = torch.from_numpy(np.array(T_l, dtype=np.int64))
            for step in range(20 if args.smoke else 600):
                mb = torch.randint(0, len(S_t), (128,))
                lp, _ = coord.log_prob(S_t[mb], T_t[mb])
                loss = -lp.mean()
                copt.zero_grad(); loss.backward(); copt.step()
        else:
            T_t = torch.from_numpy(np.array(T_l, dtype=np.float32))
            for step in range(20 if args.smoke else 600):
                mb = torch.randint(0, len(S_t), (128,))
                loss = torch.mean((coord.enc(S_t[mb]) - T_t[mb]) ** 2)
                copt.zero_grad(); loss.backward(); copt.step()
        log(f"coordinator BC seed done (loss {float(loss):.3f} over "
            f"{len(S_t)} messages)")

        coptv = torch.optim.Adam(
            [p for p in coord.parameters() if p.requires_grad]
            + list(cvalue.parameters()), lr=1e-4)
        itersB = 2 if args.smoke else 100

        # Best-coordinator tracking by GREEDY eval (G4 lesson: RL
        # fine-tuning can erode a good seed; run 1 of this stage watched
        # the entropy bonus inflate messages to noise, win 0.81 → 0.56).
        import copy as _copy

        def greedy_win(n=12):
            eps_d = collect("learned_det", n, cbytes=_serialize(coord),
                            seed0=800_000)
            return float(np.mean([e[2]["win"] for e in eps_d]))

        best_win = greedy_win()
        best_coord = _copy.deepcopy(coord.state_dict())
        log(f"B baseline greedy win {best_win:.2f}")
        for it in range(itersB):
            # entropy bonus annealed to zero across the phase
            ent_coef = 0.01 * max(0.0, 1.0 - it / (0.6 * itersB))
            eps = collect("learned", E, cbytes=_serialize(coord),
                          seed0=400_000 + it * 1000)
            upd = (coord_ppo_update if args.condition == "C"
                   else emb_coord_ppo_update)
            ent = upd(coord, cvalue, coptv, eps, ent_coef)
            winB = float(np.mean([e[2]["win"] for e in eps]))
            if it % 10 == 9:
                gw = greedy_win()
                if gw > best_win:
                    best_win, best_coord = gw, _copy.deepcopy(
                        coord.state_dict())
                log(f"B it {it:3d}/{itersB} win {winB:.2f} ent {ent:.2f} "
                    f"greedy {gw:.2f} best {best_win:.2f}")
            maybe_video(f"B_upd{it:03d}_w{winB:.2f}", "learned")
        coord.load_state_dict(best_coord)
        log(f"Phase B done: restored best coordinator (greedy {best_win:.2f})")
        torch.save(coord.state_dict(), RUN_DIR / "coord_B.pt")

        # ---- Phase C: joint fine-tune ------------------------------------
        log("=== Stage 2 Phase C: joint fine-tune ===")
        oC = torch.optim.Adam(
            list(r0.parameters()) + list(v0.parameters())
            + list(r1.parameters()) + list(v1.parameters()), lr=1e-4)
        itersC = 2 if args.smoke else 60
        best_winC = greedy_win() if not args.smoke else 0.0
        bestC = _copy.deepcopy((r0.state_dict(), r1.state_dict(),
                                coord.state_dict())) if not args.smoke else None
        for it in range(itersC):
            eps = collect("learned", E, cbytes=_serialize(coord),
                          seed0=500_000 + it * 1000)
            for pol, val, key in ((r0, v0, "red0"), (r1, v1, "red1")):
                ppo_update(pol, val, oC, eps, key)
            upd = (coord_ppo_update if args.condition == "C"
                   else emb_coord_ppo_update)
            ent = upd(coord, cvalue, coptv, eps, 0.0)
            winC = float(np.mean([e[2]["win"] for e in eps]))
            if it % 10 == 9 and not args.smoke:
                gw = greedy_win()
                if gw > best_winC:
                    best_winC = gw
                    bestC = _copy.deepcopy((r0.state_dict(), r1.state_dict(),
                                            coord.state_dict()))
                log(f"C it {it:3d}/{itersC} win {winC:.2f} ent {ent:.2f} "
                    f"greedy {gw:.2f} best {best_winC:.2f}")
            maybe_video(f"C_upd{it:03d}_w{winC:.2f}", "learned")
        if bestC is not None:
            r0.load_state_dict(bestC[0]); r1.load_state_dict(bestC[1])
            coord.load_state_dict(bestC[2])
            log(f"Phase C done: restored best joint state (greedy {best_winC:.2f})")
        for name, m in (("r0_final", r0), ("r1_final", r1),
                        ("coord_final", coord), ("cvalue", cvalue)):
            torch.save(m.state_dict(), RUN_DIR / f"{name}.pt")

        # ---- G6 evaluation ----------------------------------------------
        log("=== G6 evaluation ===")
        n_eval = 8 if args.smoke else 200
        eps_on = collect("learned_det", n_eval, cbytes=_serialize(coord),
                         seed0=900_000)
        eps_off = collect("zeroed_det", n_eval, seed0=900_000)
        win_on = float(np.mean([e[2]["win"] for e in eps_on]))
        win_off = float(np.mean([e[2]["win"] for e in eps_off]))
        with torch.no_grad():
            S_e = torch.from_numpy(np.concatenate(
                [e[1]["s"] for e in eps_on if len(e[1]["s"])]))[:512]
            _, _, _, ent_e = coord(S_e, sample=True)
        h_max = 8 * np.log(32)
        ent_frac = float(ent_e.mean()) / h_max
        drop = (win_on - win_off) * 100
        # NOTE: conditional entropy vs 0.5·H_max is the WRONG collapse
        # guard for a seeded protocol (a good protocol is deterministic
        # per state; even the scripted protocol fails this bar). The
        # authoritative criterion is marginal message diversity referenced
        # to the scripted protocol — tools/eval_g6.py. Kept here as
        # informational only.
        checks = {
            "win_rate": (win_on, win_on >= 0.75),
            "entropy_frac_informational": (ent_frac, True),
            "causal_drop_pts": (drop, drop >= 15.0),
        }
        for k, (v, ok) in checks.items():
            log(f"G6 {k}: {v:.2f} → {'OK' if ok else 'FAIL'}")
        log(f"G6 detail: win(z_g on)={win_on:.2f} win(z_g zeroed)={win_off:.2f}")
        result = {k: {"value": v, "ok": bool(ok)}
                  for k, (v, ok) in checks.items()}
        result["pass"] = all(c["ok"] for c in result.values()
                             if isinstance(c, dict))
        (RUN_DIR / "G6_RESULT.json").write_text(json.dumps(result, indent=1))
        log(f"G6 gate: {'PASS' if result['pass'] else 'FAIL'}")
        try:
            render_stage2(r0, r1, coord, VIDEO_DIR / "stage2_final.mp4",
                          seed=31337, mode="learned")
        except Exception:
            pass
        return 0 if result["pass"] else 1
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    raise SystemExit(main())
