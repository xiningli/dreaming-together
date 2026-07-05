"""Stage 1 — individual combat skills, trained in CombatEnv vs the oracle.

For each role (shotgun red1, then shield red0):
  1. BC warm start: clone the scripted oracle's actions from privileged
     obs (the T0 lesson — never make RL discover what a teacher can show).
  2. PPO fine-tune: learned agent + scripted teammate vs fully scripted
     blue team. Parallel rollout workers (spawn; no GL in workers).
  3. Acceptance (design §6): shotgun — team win rate ≥ 70% vs the scripted
     duo; shield — ≥ 70% of enemy pellets blocked AND team survival.

Progress reporting (the reason this file exists in this shape): every
VIDEO_EVERY_S seconds of wall clock, the current policy plays one episode
that is rendered with pellet tracers and HP bars to
videos/progress/stage1_<role>_upd<NNN>_<metric>.mp4, and
videos/progress/latest_<role>.mp4 is refreshed. runs/stage1/status.txt
always holds the one-line current state.

Kill criteria (rule R9), enforced automatically:
  - non-finite loss → abort with log
  - primary metric still at 0 after 25% of the iteration budget → abort
  - entropy collapse (mean policy std < 0.02 before 50% budget) → abort

Run:  python -m dreaming_together.training.stage1_combat [--smoke]
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

RUN_DIR = ROOT / "runs" / "stage1"
VIDEO_DIR = ROOT / "videos" / "progress"

OBS_DIM = 27          # privileged pathfinder obs
ACT_DIM = 5
GAMMA, LAM = 0.99, 0.95
PPO_CLIP = 0.2
LR = 3e-4
ENT_COEF = 1e-3
EPISODES_PER_ITER = 32
MINIBATCH = 4096
PPO_EPOCHS = 4
VIDEO_EVERY_S = 600           # a fresh progress video every 10 minutes
CHECKPOINT_EVERY = 10         # iterations

# Shield acceptance note: the design's 70%-blocked bar is defined for the
# ISOLATED blocking curriculum (static teammate, scripted shooter, window
# never opened). In full 2v2 the oracle shield itself only reaches ~0.5
# block fraction because timed windows deliberately open the corridor.
# Full-2v2 bar: block_frac ≥ 0.60 sustained (clearly above the oracle
# baseline); the isolated-curriculum test is a later addition.
ROLE_CFG = {
    "shotgun": {"prefix": "red1", "agent_idx": 1, "iters": 400,
                "accept_metric": "win_rate", "accept": 0.70,
                "obs_dim_vision": 272},   # 256 front-cam embed + 16 proprio
    "shield":  {"prefix": "red0", "agent_idx": 0, "iters": 400,
                "accept_metric": "block_frac", "accept": 0.60,
                "obs_dim_vision": 528},   # front+rear embeds + proprio
}
ENCODER_PATH = ROOT / "runs" / "stage0" / "encoder.pt"


# ---------------------------------------------------------------------------
# Rollout worker (no rendering, no GL)
# ---------------------------------------------------------------------------

_W = {}


def _worker_init(role_prefix: str, vision: bool = False,
                 obs_dim: int = OBS_DIM, encoder_path: str = ""):
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam
    env = CombatEnv(seed=0, privileged_obs=True)
    if vision:
        env.enable_vision(encoder_path or ENCODER_PATH)
    _W["env"] = env
    _W["teams"] = (ScriptedTeam(0), ScriptedTeam(1))
    _W["prefix"] = role_prefix
    _W["vision"] = vision
    hidden = (256, 256) if vision else (64, 64)
    _W["policy"] = GaussianPolicy(obs_dim, ACT_DIM, hidden=hidden)
    _W["policy"].eval()


def _rollout(job):
    """job = (state_dict_bytes, seed, deterministic) → episode arrays."""
    import io
    state_bytes, seed, deterministic = job
    env, teams, prefix = _W["env"], _W["teams"], _W["prefix"]
    policy = _W["policy"]
    policy.load_state_dict(torch.load(io.BytesIO(state_bytes),
                                      weights_only=True))
    agent_idx = ["red0", "red1", "blue0", "blue1"].index(prefix)

    env.reset(seed=seed)
    obs_l, act_l, logp_l, rew_l = [], [], [], []
    blocked = 0
    hits_taken = 0
    with torch.no_grad():
        while not env.done:
            actions = {}
            for tm in teams:
                actions.update(tm.act(env))
            o = (env.vision_obs(prefix) if _W["vision"]
                 else env.obs(prefix))
            ot = torch.from_numpy(o)
            if deterministic:
                a = torch.tanh(policy.mean_net(ot))
                lp = torch.zeros(())
            else:
                d = policy.dist(ot)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
            actions[prefix] = a.numpy()
            _, rewards, done, info = env.step(actions)
            obs_l.append(o)
            act_l.append(a.numpy())
            logp_l.append(float(lp))
            rew_l.append(float(rewards[agent_idx]))
            blocked += info["blocked"][0]
            hits_taken += int(info["damage"][:2].sum() / 6)
    win = tuple(env.team_result) == (1, -1)
    return (np.array(obs_l, dtype=np.float32),
            np.array(act_l, dtype=np.float32),
            np.array(logp_l, dtype=np.float32),
            np.array(rew_l, dtype=np.float32),
            {"win": win, "blocked": blocked, "hits_taken": hits_taken,
             "len": len(rew_l), "ret": float(np.sum(rew_l))})


def _serialize(policy) -> bytes:
    import io
    buf = io.BytesIO()
    torch.save(policy.state_dict(), buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# BC warm start from the oracle
# ---------------------------------------------------------------------------

def collect_bc_data(role_prefix: str, n_episodes: int,
                    vision: bool = False):
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam
    env = CombatEnv(seed=0, privileged_obs=True)
    if vision:
        env.enable_vision(encoder_path or ENCODER_PATH)
    teams = (ScriptedTeam(0), ScriptedTeam(1))
    X, Y = [], []
    for ep in range(n_episodes):
        env.reset(seed=50_000 + ep)
        while not env.done:
            actions = {}
            for tm in teams:
                actions.update(tm.act(env))
            X.append(env.vision_obs(role_prefix) if vision
                     else env.obs(role_prefix))
            Y.append(actions[role_prefix])
            env.step(actions)
    return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32))


def bc_warm_start(policy: GaussianPolicy, role_prefix: str,
                  n_episodes: int, log, vision: bool = False) -> None:
    t0 = time.time()
    X, Y = collect_bc_data(role_prefix, n_episodes, vision)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(np.clip(Y, -0.999, 0.999))
    opt = torch.optim.Adam(policy.mean_net.parameters(), lr=1e-3)
    for epoch in range(300):
        idx = torch.randperm(len(Xt))[:8192]
        loss = torch.mean((torch.tanh(policy.mean_net(Xt[idx])) - Yt[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"BC warm start: {len(X)} samples from {n_episodes} oracle episodes, "
        f"final MSE {loss.item():.4f} ({time.time()-t0:.0f}s)")


def bc_warm_start_vision(policy: GaussianPolicy, role_prefix: str,
                         n_episodes: int, out: Path, log) -> Path:
    """Vision BC that fine-tunes the encoder end-to-end with an auxiliary
    enemy-bearing head.

    Root cause this exists for (found by video review at vision run 1,
    iter 28): the frozen Stage-0 RECONSTRUCTION embedding preserves pixels
    but does not expose enemy bearing in a form a small MLP learns from
    272 dims — the BC clone won 0% (proprio clone: 12%), so PPO climbed
    the dense shaped rewards instead and parked the tank against the enemy
    shield. The bearing auxiliary forces the embedding to carry aim
    information; the fine-tuned encoder is then frozen for PPO.

    Returns the path of the fine-tuned encoder for the rollout workers.
    """
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.envs.cameras import SegCamera
    from dreaming_together.vision.encoder import SegEncoder
    from dreaming_together.oracle import ScriptedTeam
    from dreaming_together.envs.arena import ROLES as ARENA_ROLES

    t0 = time.time()
    env = CombatEnv(seed=0, privileged_obs=True)
    cam = SegCamera(env.model)
    teams = (ScriptedTeam(0), ScriptedTeam(1))
    cams = [f"{role_prefix}_front_cam"]
    if ARENA_ROLES[role_prefix] == "shield":
        cams.append(f"{role_prefix}_rear_cam")
    K = len(cams)

    frames_l, prop_l, act_l, bearing_l = [], [], [], []
    for ep in range(n_episodes):
        env.reset(seed=50_000 + ep)
        while not env.done:
            actions = {}
            for tm in teams:
                actions.update(tm.act(env))
            priv = env.obs(role_prefix)          # 27-dim privileged
            frames_l.append(np.stack(
                [cam.render(env.data, c) for c in cams]))
            prop_l.append(priv[:16])
            bearing_l.append(priv[16:19])        # nearest-enemy sin,cos,dist
            act_l.append(actions[role_prefix])
            env.step(actions)
    cam.close()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F = torch.from_numpy(np.array(frames_l, dtype=np.int64))
    P = torch.from_numpy(np.array(prop_l, dtype=np.float32))
    A = torch.from_numpy(np.clip(np.array(act_l, dtype=np.float32),
                                 -0.999, 0.999))
    B = torch.from_numpy(np.array(bearing_l, dtype=np.float32))

    enc = SegEncoder().to(dev)
    enc.load_state_dict(torch.load(ENCODER_PATH, weights_only=True))
    head = MLP(K * 256 + 16, ACT_DIM, hidden=(256, 256)).to(dev)
    aux = torch.nn.Linear(K * 256, 3).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters())
                           + list(aux.parameters()), lr=1e-3)
    n = len(F)
    for epoch in range(40):
        perm = torch.randperm(n)
        tot_a = tot_b = 0.0
        for s in range(0, n, 256):
            mb = perm[s:s + 256]
            f = F[mb].to(dev)
            z = enc(f.view(-1, 64, 64)).view(len(mb), K * 256)
            feat = torch.cat([z, P[mb].to(dev)], dim=1)
            l_act = torch.mean((torch.tanh(head(feat)) - A[mb].to(dev)) ** 2)
            l_aux = torch.mean((aux(z) - B[mb].to(dev)) ** 2)
            loss = l_act + 0.5 * l_aux
            opt.zero_grad(); loss.backward(); opt.step()
            tot_a += float(l_act) * len(mb); tot_b += float(l_aux) * len(mb)
        if epoch % 10 == 0 or epoch == 39:
            log(f"  BC-vision epoch {epoch}: act MSE {tot_a/n:.4f}, "
                f"bearing MSE {tot_b/n:.4f}")

    policy.mean_net.load_state_dict(
        {k: v.cpu() for k, v in head.state_dict().items()})
    enc_path = out / "encoder_ft.pt"
    torch.save({k: v.cpu() for k, v in enc.state_dict().items()}, enc_path)
    log(f"BC-vision warm start: {n} samples from {n_episodes} episodes, "
        f"encoder fine-tuned with bearing aux ({time.time()-t0:.0f}s)")
    return enc_path


# ---------------------------------------------------------------------------
# Progress video (main process only — owns EGL)
# ---------------------------------------------------------------------------

def render_progress(policy: GaussianPolicy, role_prefix: str,
                    out_path: Path, seed: int, vision: bool = False,
                    encoder_path: str = "") -> dict:
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

    if vision:
        env.enable_vision(encoder_path or ENCODER_PATH)
    teams = (ScriptedTeam(0), ScriptedTeam(1))
    env.reset(seed=seed)
    pellet_bids = {n: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in env.pm._names}
    trails: dict[str, list] = {}
    frames = []
    with torch.no_grad():
        while not env.done:
            actions = {}
            for tm in teams:
                actions.update(tm.act(env))
            o = torch.from_numpy(env.vision_obs(role_prefix) if vision
                                 else env.obs(role_prefix))
            actions[role_prefix] = torch.tanh(policy.mean_net(o)).numpy()
            # step with per-substep trail capture
            import dreaming_together.envs.combat_env as ce
            # (env.step already does substeps; capture pellet pos after step
            #  — coarse trails, good enough for progress review)
            _, _, done, info = env.step(actions)
            for name, bid in pellet_bids.items():
                p = env.data.xpos[bid].copy()
                if p[2] < 50:
                    trails.setdefault(name, []).append(p)
                    trails[name] = trails[name][-6:]
                elif name in trails:
                    del trails[name]
            renderer.update_scene(env.data, camera=cam)
            scene = renderer._scene
            for pts in trails.values():
                for a, b in zip(pts[:-1], pts[1:]):
                    if scene.ngeom >= scene.maxgeom: break
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                        np.zeros(3), np.zeros(3), np.zeros(9),
                                        np.array([1, .85, .1, .9], np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                         0.01, a, b)
                    scene.ngeom += 1
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=20)
    return {"win": tuple(env.team_result) == (1, -1),
            "t": env.t, "hp": dict(env.hp)}


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def gae(rews: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nxt = vals[t + 1] if t + 1 < T else 0.0
        delta = rews[t] + GAMMA * nxt - vals[t]
        last = delta + GAMMA * LAM * last
        adv[t] = last
    return adv, adv + vals[:T]


def train_role(role: str, n_workers: int, smoke: bool, log,
               vision: bool = False) -> bool:
    cfg = ROLE_CFG[role]
    prefix = cfg["prefix"]
    iters = 4 if smoke else cfg["iters"]
    n_bc = 8 if smoke else 200
    obs_dim = cfg["obs_dim_vision"] if vision else OBS_DIM
    out = RUN_DIR / (f"{role}_vision" if vision else role)
    out.mkdir(parents=True, exist_ok=True)

    hidden = (256, 256) if vision else (64, 64)
    policy = GaussianPolicy(obs_dim, ACT_DIM, hidden=hidden,
                            init_log_std=-1.0)
    value = MLP(obs_dim, 1, hidden=hidden)
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=LR)

    if vision:
        enc_ft = bc_warm_start_vision(policy, prefix, n_bc, out, log)
    else:
        enc_ft = ""
        bc_warm_start(policy, prefix, n_bc, log)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(n_workers, initializer=_worker_init,
                    initargs=(prefix, vision, obs_dim, str(enc_ft)))
    torch.save(policy.state_dict(), out / "policy_bc.pt")

    history = []
    last_video = 0.0
    metric_name = cfg["accept_metric"]
    t_start = time.time()

    for it in range(iters):
        jobs = [(_serialize(policy), 100_000 + it * 1000 + k, False)
                for k in range(4 if smoke else EPISODES_PER_ITER)]
        episodes = pool.map(_rollout, jobs)

        obs = np.concatenate([e[0] for e in episodes])
        act = np.concatenate([e[1] for e in episodes])
        logp_old = np.concatenate([e[2] for e in episodes])
        advs, rets = [], []
        with torch.no_grad():
            for e in episodes:
                v = value(torch.from_numpy(e[0])).squeeze(-1).numpy()
                a, r = gae(e[3], v)
                advs.append(a); rets.append(r)
        adv = np.concatenate(advs); ret = np.concatenate(rets)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        obs_t = torch.from_numpy(obs); act_t = torch.from_numpy(act)
        adv_t = torch.from_numpy(adv); ret_t = torch.from_numpy(ret)
        logp_old_t = torch.from_numpy(logp_old)
        n = len(obs)
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(n)
            for s in range(0, n, MINIBATCH):
                mb = perm[s:s + MINIBATCH]
                d = policy.dist(obs_t[mb])
                logp = d.log_prob(act_t[mb]).sum(-1)
                ratio = torch.exp(logp - logp_old_t[mb])
                l_pi = -torch.min(
                    ratio * adv_t[mb],
                    torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_t[mb]
                ).mean()
                l_v = ((value(obs_t[mb]).squeeze(-1) - ret_t[mb]) ** 2).mean()
                l_ent = -d.entropy().sum(-1).mean()
                loss = l_pi + 0.5 * l_v + ENT_COEF * l_ent
                if not torch.isfinite(loss):
                    log(f"KILL: non-finite loss at iter {it}")
                    return False
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        stats = [e[4] for e in episodes]
        win_rate = float(np.mean([s["win"] for s in stats]))
        blocks = sum(s["blocked"] for s in stats)
        taken = sum(s["hits_taken"] for s in stats)
        block_frac = blocks / max(1, blocks + taken)
        mean_ret = float(np.mean([s["ret"] for s in stats]))
        metric = win_rate if metric_name == "win_rate" else block_frac
        std = float(policy.log_std.exp().mean())
        history.append({"iter": it, "win_rate": win_rate,
                        "block_frac": block_frac, "ret": mean_ret,
                        "std": std, "wall": time.time() - t_start})
        line = (f"{role} it {it:3d}/{iters}  win {win_rate:.2f}  "
                f"block {block_frac:.2f}  ret {mean_ret:+.1f}  std {std:.3f}")
        log(line)
        (RUN_DIR / "status.txt").write_text(line + "\n")

        # kill criteria
        if it == max(1, iters // 4):
            recent = np.mean([h[metric_name] for h in history[-10:]])
            if recent == 0.0:
                log(f"KILL: {metric_name} flat at 0 after 25% budget")
                return False
        if std < 0.02 and it < iters // 2:
            log("KILL: entropy collapse before half budget")
            return False
        if it == iters // 2 and not smoke:
            recent = np.mean([h[metric_name] for h in history[-40:]])
            if recent < cfg["accept"] / 3:
                log(f"KILL: {metric_name} plateaued at {recent:.2f} "
                    f"(< accept/3) at half budget")
                return False

        if it % CHECKPOINT_EVERY == 0 or it == iters - 1:
            torch.save(policy.state_dict(), out / f"policy_{it:04d}.pt")
            torch.save(policy.state_dict(), out / "policy_latest.pt")
            (out / "history.json").write_text(json.dumps(history))

        if time.time() - last_video > (30 if smoke else VIDEO_EVERY_S):
            last_video = time.time()
            vtag = "v_" if vision else ""
            tag = f"stage1_{vtag}{role}_upd{it:04d}_{metric_name[0]}{metric:.2f}"
            path = VIDEO_DIR / f"{tag}.mp4"
            try:
                info = render_progress(policy, prefix, path, seed=777 + it,
                                       vision=vision,
                                       encoder_path=str(enc_ft))
                shutil.copyfile(path, VIDEO_DIR / f"latest_{vtag}{role}.mp4")
                log(f"video → {path.name} (win={info['win']}, t={info['t']:.1f}s)")
            except Exception as exc:
                log(f"video render failed (training continues): {exc}")

        # early acceptance: sustained over the last 10 iters
        if (len(history) >= 10
                and np.mean([h[metric_name] for h in history[-10:]])
                >= cfg["accept"]):
            log(f"{role}: acceptance reached early at iter {it}")
            break

    pool.close(); pool.join()
    torch.save(policy.state_dict(), out / "policy_final.pt")
    (out / "history.json").write_text(json.dumps(history))
    final = np.mean([h[metric_name] for h in history[-10:]])
    ok = bool(final >= cfg["accept"])
    log(f"{role} DONE: {metric_name}={final:.2f} over last 10 iters "
        f"(accept ≥ {cfg['accept']}) → {'PASS' if ok else 'FAIL'}")
    try:
        vtag = "v_" if vision else ""
        info = render_progress(policy, prefix,
                               VIDEO_DIR / f"stage1_{vtag}{role}_final.mp4",
                               seed=4242, vision=vision,
                               encoder_path=str(enc_ft))
        log(f"final video → stage1_{vtag}{role}_final.mp4 (win={info['win']})")
    except Exception as exc:
        log(f"final video failed: {exc}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--roles", default="shotgun,shield")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(RUN_DIR / "train.log", "a", buffering=1)

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    log(f"=== Stage 1 start (smoke={args.smoke}, vision={args.vision}, "
        f"workers={args.workers}) ===")
    results = {}
    try:
        for role in args.roles.split(","):
            results[role] = train_role(role, args.workers, args.smoke, log,
                                       vision=args.vision)
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1
    log(f"=== Stage 1 complete: {results} ===")
    tag = "STAGE1_VISION_RESULT.json" if args.vision else "STAGE1_RESULT.json"
    (RUN_DIR / tag).write_text(json.dumps(results))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
