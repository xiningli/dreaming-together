"""Stage 2 with DIFFUSION listeners — true conditions B and C.

Same three-phase structure as stage2_coordination.py, adapted to the
G4-certified diffusion policy class:

  A. Listener BC: both red listeners are DiffusionPolicy nets
     (cond = 27-d privileged obs + 256-d z_g = 283; 8×5 action horizon,
     receding-horizon execution of a[0]) BC-trained on scripted-coordinator
     oracle episodes — z delivered through the condition's frozen channel.
  B. Speaker training: listeners frozen; coordinator BC-seeded on the
     scripted protocol then PPO-trained (language PPO for C, bottleneck
     Gaussian PPO for B) with entropy anneal + best-checkpoint tracking.
  C. Joint: alternating elite self-imitation on the listeners (the G4
     method — PPO has no tractable log-prob through a DDIM sampler) and
     coordinator PPO, with best-joint tracking.

G6 verdict: tools/eval_g6.py is FF-listener-specific; this script writes
its own G6_RESULT_v2.json using the same frozen elite opponent, corrected
diversity criterion, and 15-pt causal bar.

Run: python -m dreaming_together.training.stage2_diffusion --condition B|C
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dreaming_together.policies.diffusion_policy import DiffusionPolicy
from dreaming_together.policies.ff_policy import MLP
from dreaming_together.training.stage1_combat import gae
from dreaming_together.training.stage2_coordination import (
    LOBS, STATE_DIM, DT_COORD_STEPS, Z_DIM, ACT_DIM,
    coord_state, make_coordinator, scripted_z,
    coord_ppo_update, emb_coord_ppo_update,
)

VIDEO_DIR = ROOT / "videos" / "progress"
H = 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import os
NOISE_EXPLORE = 1.0
# overridable so eval scripts can sweep deployment-time sampling noise
# (spawned workers re-import this module, so an env var is the only
# channel that reaches them)
# 0.40 selected by validation-seed sweep on condition C (win peaks
# there and the causal drop grows with diversity); applied uniformly
# to every diffusion condition.
NOISE_EVAL = float(os.environ.get("DIFF_NOISE_EVAL", "0.40"))
# Checkpoint-SELECTION noise during bring-up. Deployment noise 0.40 is
# right for the final stack but devastates weak early listeners — using
# it for training-phase evals made A2 selection random and collapsed the
# C parity re-run (0.79 → 0.29) and handicapped every condition-B
# bring-up. Selection replays the certified pathway at 0.15; only the
# official G6/deployment evals use 0.40.
SELECT_NOISE = 0.15


K_CANDIDATES = 8


class Scorer(torch.nn.Module):
    """Sample-and-select head (G4's pre-registered fallback, invoked
    2026-07-05 after 12+ direct fine-tuning attempts failed to scale to
    2v2): scores first-actions of K frozen-diffusion candidates; trained
    by return-to-go regression on chosen actions. The diffusion prior
    stays frozen — the body-prior property under test is untouched;
    learning variance lives in this small critic."""

    def __init__(self):
        super().__init__()
        self.net = MLP(LOBS + ACT_DIM, 1, hidden=(128, 128))

    def forward(self, obs, acts):
        # obs (K, LOBS) tiled, acts (K, ACT_DIM) → (K,)
        return self.net(torch.cat([obs, acts], dim=-1)).squeeze(-1)


def sas_act(diff, scorer, o_np, det):
    """Sample K horizons from the frozen prior, score first actions,
    pick argmax (det) or softmax sample (explore). Returns (action5,
    chosen_horizon40)."""
    o = torch.from_numpy(o_np).unsqueeze(0).expand(K_CANDIDATES, -1)
    hor = diff.ddim_sample(o, 8, noise_scale=1.0)
    firsts = hor.view(K_CANDIDATES, H, ACT_DIM)[:, 0]
    q = scorer(o, firsts)
    if det:
        i = int(q.argmax())
    else:
        i = int(torch.distributions.Categorical(logits=q * 2.0).sample())
    return firsts[i].numpy(), hor[i].numpy()


def make_listener() -> DiffusionPolicy:
    return DiffusionPolicy(cond_dim=LOBS, act_dim=ACT_DIM, horizon=H,
                           hidden=256)


# ---------------------------------------------------------------------------
# Rollout worker (CPU DDIM sampling)
# ---------------------------------------------------------------------------

_W = {}


def _winit(condition: str, coord_init: bytes):
    import io
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam, EliteScriptedTeam
    _W["env"] = CombatEnv(seed=0, privileged_obs=True)
    _W["blue_std"] = ScriptedTeam(1)
    _W["blue_elite"] = EliteScriptedTeam(1)
    _W["condition"] = condition
    _W["r0"] = make_listener()
    _W["r1"] = make_listener()
    _W["q0"] = Scorer()
    _W["q1"] = Scorer()
    _W["coord"] = make_coordinator(condition)
    _W["coord"].load_state_dict(
        torch.load(io.BytesIO(coord_init), weights_only=True))
    for m in (_W["r0"], _W["r1"], _W["coord"]):
        m.eval()


def _scripted_z_train(env, coord, condition):
    """Scripted z for LISTENER-TRAINING rollouts. For the continuous
    channel (B), jitter the bottleneck with N(0, 0.05) so listeners
    tolerate the near-miss messages a BC-trained coordinator emits —
    run 1's frozen listeners only understood bit-exact scripted vectors
    (B baseline 0.25 vs listener skill 0.59)."""
    if condition == "C":
        return scripted_z(env, coord, condition)
    from dreaming_together.oracle.scripted_coordinator import (
        scripted_bottleneck)
    b = scripted_bottleneck(env, 0) + torch.randn(1, 5) * 0.05
    return coord.z_from_bottleneck(b)[0].numpy()


def _roll(job):
    """job = (r0b, r1b, cb, seed, mode) — mode in
    scripted | learned | zeroed  (+'_det')  (+'_elite')."""
    import io
    r0b, r1b, cb, seed, mode = job
    elite = "_elite" in mode
    det = "_det" in mode
    mode = mode.replace("_elite", "").replace("_det", "")
    env = _W["env"]
    blue = _W["blue_elite"] if elite else _W["blue_std"]
    coord = _W["coord"]
    d0, q0b = r0b
    d1, q1b = r1b
    if d0 is not None:
        _W["r0"].load_state_dict(torch.load(io.BytesIO(d0),
                                            weights_only=True))
        _W["r1"].load_state_dict(torch.load(io.BytesIO(d1),
                                            weights_only=True))
    _W["q0"].load_state_dict(torch.load(io.BytesIO(q0b), weights_only=True))
    _W["q1"].load_state_dict(torch.load(io.BytesIO(q1b), weights_only=True))
    if cb is not None:
        coord.load_state_dict(torch.load(io.BytesIO(cb), weights_only=True))

    env.reset(seed=seed)
    traj = {p: {"obs": [], "act": [], "ret": 0.0}
            for p in ("red0", "red1")}
    ctraj = {"s": [], "tok": [], "logp": [], "rew": []}
    z = np.zeros(Z_DIM, dtype=np.float32)
    step = 0
    if det:
        noise = NOISE_EVAL if elite else SELECT_NOISE
    else:
        noise = NOISE_EXPLORE
    with torch.no_grad():
        while not env.done:
            if step % DT_COORD_STEPS == 0:
                if mode == "zeroed":
                    z = np.zeros(Z_DIM, dtype=np.float32)
                elif mode == "scripted":
                    z = _scripted_z_train(env, coord, _W["condition"])
                elif mode == "scripted_exact":
                    z = scripted_z(env, coord, _W["condition"])
                else:
                    s = coord_state(env)
                    zt, toks, lp, _ = coord(
                        torch.from_numpy(s).unsqueeze(0), sample=not det)
                    z = zt[0].numpy()
                    ctraj["s"].append(s)
                    ctraj["tok"].append(np.asarray(toks[0]))
                    ctraj["logp"].append(float(lp))
                    ctraj["rew"].append(0.0)
            actions = blue.act(env)
            for p, dkey, qkey in (("red0", "r0", "q0"),
                                  ("red1", "r1", "q1")):
                o = np.concatenate([env.obs(p), z]).astype(np.float32)
                a, hor = sas_act(_W[dkey], _W[qkey], o, det)
                actions[p] = a
                traj[p]["obs"].append(o)
                traj[p]["act"].append(a)
            _, rewards, done, info = env.step(actions)
            traj["red0"].setdefault("rew", []).append(float(rewards[0]))
            traj["red1"].setdefault("rew", []).append(float(rewards[1]))
            traj["red0"]["ret"] += float(rewards[0])
            traj["red1"]["ret"] += float(rewards[1])
            if ctraj["rew"]:
                ctraj["rew"][-1] += float(rewards[0] + rewards[1])
            step += 1
    stats = {"win": tuple(env.team_result) == (1, -1), "len": step}
    out_t = {p: {"obs": np.array(d["obs"], dtype=np.float32),
                 "act": np.array(d["act"], dtype=np.float32),
                 "rew": np.array(d.get("rew", []), dtype=np.float32),
                 "ret": d["ret"]} for p, d in traj.items()}
    out_c = {k: np.array(v, dtype=np.float32 if k != "tok" else None)
             for k, v in ctraj.items()}
    return out_t, out_c, stats


def _ser(m) -> bytes:
    import io
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.getvalue()


# ---------------------------------------------------------------------------

def collect_distill(condition: str, n_episodes: int,
                    coord_init: Path | None = None):
    """Distillation data: the CERTIFIED FF Stage-2 listeners playing
    under the scripted coordinator. A far stronger, far less noisy
    teacher than the oracle: the multi-seed grid showed BC-from-oracle +
    elite-SI is a brittle foundation (6/6 seeds ≤ 0.40), while the FF
    listeners carry Stage-1 PPO grounding. Uniform across B and C
    (language-side FF stack for C, embedding-side for B)."""
    import io
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam
    from dreaming_together.policies.ff_policy import GaussianPolicy
    ff_run = ROOT / "runs" / ("stage2" if condition == "C" else "stage2_A")
    ff0 = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
    ff1 = GaussianPolicy(LOBS, ACT_DIM, hidden=(64, 64))
    ff0.load_state_dict(torch.load(ff_run / "r0_final.pt",
                                   weights_only=True))
    ff1.load_state_dict(torch.load(ff_run / "r1_final.pt",
                                   weights_only=True))
    # the distillation z MUST come through the SAME frozen channel the
    # run trains/evaluates with — a different seed's token→z table would
    # teach the listeners a foreign vocabulary
    if coord_init is None:
        coord_init = ROOT / "runs" / f"stage2_{condition}_diff_s1" \
            / "coord_init.pt"
    fcoord = make_coordinator(condition)
    fcoord.load_state_dict(torch.load(coord_init, weights_only=True))
    env = CombatEnv(seed=0, privileged_obs=True)
    blue = ScriptedTeam(1)
    data = {p: {"C": [], "X": []} for p in ("red0", "red1")}
    with torch.no_grad():
        for ep in range(n_episodes):
            env.reset(seed=70_000 + ep)
            obs_l = {p: [] for p in ("red0", "red1")}
            act_l = {p: [] for p in ("red0", "red1")}
            z = np.zeros(Z_DIM, dtype=np.float32)
            k = 0
            while not env.done:
                if k % DT_COORD_STEPS == 0:
                    z = scripted_z(env, fcoord, condition)
                actions = blue.act(env)
                for p, pol in (("red0", ff0), ("red1", ff1)):
                    o = np.concatenate([env.obs(p), z]).astype(np.float32)
                    a = torch.tanh(pol.mean_net(
                        torch.from_numpy(o))).numpy()
                    # small action jitter → horizons carry local diversity
                    a = np.clip(a + np.random.randn(ACT_DIM) * 0.05, -1, 1)
                    actions[p] = a
                    obs_l[p].append(o)
                    act_l[p].append(a)
                env.step(actions)
                k += 1
            for p in ("red0", "red1"):
                for tt in range(len(obs_l[p]) - H):
                    data[p]["C"].append(obs_l[p][tt])
                    data[p]["X"].append(
                        np.concatenate(act_l[p][tt:tt + H]))
    out = {}
    for p in ("red0", "red1"):
        out[p] = (torch.tensor(np.array(data[p]["C"]), dtype=torch.float32),
                  torch.tensor(np.clip(np.array(data[p]["X"]), -.999, .999),
                               dtype=torch.float32))
    return out


def collect_bc(condition: str, n_episodes: int, coord):
    """Oracle episodes with scripted z; horizons for both listeners."""
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam
    env = CombatEnv(seed=0, privileged_obs=True)
    teams = (ScriptedTeam(0), ScriptedTeam(1))
    data = {p: {"C": [], "X": []} for p in ("red0", "red1")}
    with torch.no_grad():
        for ep in range(n_episodes):
            env.reset(seed=60_000 + ep)
            obs_l = {p: [] for p in ("red0", "red1")}
            act_l = {p: [] for p in ("red0", "red1")}
            z = np.zeros(Z_DIM, dtype=np.float32)
            k = 0
            while not env.done:
                if k % DT_COORD_STEPS == 0:
                    if condition == "C":
                        z = scripted_z(env, coord, condition)
                    else:
                        from dreaming_together.oracle.scripted_coordinator \
                            import scripted_bottleneck
                        b = (scripted_bottleneck(env, 0)
                             + torch.randn(1, 5) * 0.05)
                        z = coord.z_from_bottleneck(b)[0].numpy()
                a = {}
                for tm in teams:
                    a.update(tm.act(env))
                for p in ("red0", "red1"):
                    obs_l[p].append(np.concatenate([env.obs(p), z]))
                    act_l[p].append(a[p])
                env.step(a)
                k += 1
            for p in ("red0", "red1"):
                for t in range(len(obs_l[p]) - H):
                    data[p]["C"].append(obs_l[p][t])
                    data[p]["X"].append(
                        np.concatenate(act_l[p][t:t + H]))
    out = {}
    for p in ("red0", "red1"):
        out[p] = (torch.tensor(np.array(data[p]["C"]), dtype=torch.float32),
                  torch.tensor(np.clip(np.array(data[p]["X"]), -.999, .999),
                               dtype=torch.float32))
    return out


def bc_train(model: DiffusionPolicy, C, X, steps, log, tag):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    C, X = C.to(DEV), X.to(DEV)
    for s in range(steps):
        mb = torch.randint(0, len(C), (256,), device=DEV)
        loss = model.loss(X[mb], C[mb])
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 3000 == 0:
            log(f"  {tag} BC step {s}: loss {float(loss):.4f}")
    model.cpu()
    return model


def elite_si_update(model: DiffusionPolicy, buf, episodes, key, log,
                    lr=3e-4, steps=200):
    """Accumulate top-K episodes by that listener's return; weighted BC."""
    for e in episodes:
        buf.append((e[0][key]["ret"], e[0][key]["obs"], e[0][key]["act"]))
    buf.sort(key=lambda x: x[0], reverse=True)
    del buf[150:]
    C = torch.tensor(np.concatenate([b[1] for b in buf]),
                     dtype=torch.float32, device=DEV)
    X = torch.tensor(np.concatenate([b[2] for b in buf]),
                     dtype=torch.float32, device=DEV)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        mb = torch.randint(0, len(C), (256,), device=DEV)
        loss = model.loss(X[mb].clamp(-.999, .999), C[mb])
        opt.zero_grad(); loss.backward(); opt.step()
    model.cpu()
    return model


def fit_scorer(scorer: Scorer, episodes, key, gamma=0.99):
    """Return-to-go regression on chosen actions (z-scored per batch)."""
    O, A, G = [], [], []
    for e in episodes:
        r = e[0][key]["rew"]
        g = np.zeros(len(r), dtype=np.float32)
        acc = 0.0
        for i in reversed(range(len(r))):
            acc = r[i] + gamma * acc
            g[i] = acc
        O.append(e[0][key]["obs"]); A.append(e[0][key]["act"]); G.append(g)
    O = torch.tensor(np.concatenate(O), device=DEV)
    A = torch.tensor(np.concatenate(A), device=DEV)
    G = torch.tensor(np.concatenate(G), device=DEV)
    G = (G - G.mean()) / (G.std() + 1e-6)
    scorer = scorer.to(DEV)
    opt = torch.optim.Adam(scorer.parameters(), lr=3e-4)
    for _ in range(150):
        mb = torch.randint(0, len(O), (512,), device=DEV)
        loss = ((scorer(O[mb], A[mb]) - G[mb]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return scorer.cpu()


def main() -> int:
    from dreaming_together.coordinators.bandwidth import assert_bandwidth_parity
    from dreaming_together.oracle.scripted_coordinator import (
        scripted_tokens, scripted_bottleneck)
    from dreaming_together.envs.combat_env import CombatEnv
    from dreaming_together.oracle import ScriptedTeam
    assert_bandwidth_parity()

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=("B", "C"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = ROOT / "runs" / f"stage2_{args.condition}_diff_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train.log", "a", buffering=1)

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line + "\n")
        (run / "status.txt").write_text(line + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    coord = make_coordinator(args.condition)
    torch.save(coord.state_dict(), run / "coord_init.pt")
    cvalue = MLP(STATE_DIM, 1, hidden=(128, 128))
    r0, r1 = make_listener(), make_listener()
    q0, q1 = Scorer(), Scorer()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_winit,
                    initargs=(args.condition,
                              (run / "coord_init.pt").read_bytes()))
    E = 4 if args.smoke else 32

    def collect(mode, n, cb=None, seed0=0, send_diff=True):
        jobs = [((_ser(r0) if send_diff else None, _ser(q0)),
                 (_ser(r1) if send_diff else None, _ser(q1)),
                 cb, seed0 + k, mode) for k in range(n)]
        return pool.map(_roll, jobs)

    def greedy_win(cb, n=24, elite=False):
        eps = collect("learned_det" + ("_elite" if elite else ""), n,
                      cb=cb, seed0=850_000)
        return float(np.mean([e[2]["win"] for e in eps]))

    try:
        # Phase A: diffusion listener BC on scripted-z oracle episodes
        log(f"=== D-Stage2 {args.condition} Phase A: diffusion listener "
            f"BC (distilled from certified FF stack) ===")
        data = collect_distill(args.condition, 4 if args.smoke else 150,
                               coord_init=run / "coord_init.pt")
        for p, m in (("red0", r0), ("red1", r1)):
            C, X = data[p]
            log(f"  {p}: {len(C)} horizon samples")
            bc_train(m, C, X, 300 if args.smoke else 15000, log, p)
        eps = collect("scripted_det", E, seed0=210_000)
        winA = float(np.mean([e[2]["win"] for e in eps]))
        log(f"Phase A done: BC listeners win {winA:.2f} (scripted z, det)")

        # Phase A2: elite-SI listener fine-tune under the scripted
        # coordinator. Pure-BC diffusion listeners plateau ~0.2 win — a
        # ceiling no amount of speaker training can fix (run 1 baseline
        # 0.25). This is the diffusion analogue of the FF pathway's
        # Phase-A PPO.
        log("=== Phase A2: SCORER fitting (frozen prior, scripted z) ===")
        import copy as _copy2
        bestA = winA
        bestA_state = _copy2.deepcopy((q0.state_dict(), q1.state_dict()))
        itersA2 = 2 if args.smoke else 150
        since_best = 0
        for it in range(itersA2):
            eps = collect("scripted", E, seed0=220_000 + it * 1000)
            q0.load_state_dict(fit_scorer(q0, eps, "red0").state_dict())
            q1.load_state_dict(fit_scorer(q1, eps, "red1").state_dict())
            if it % 5 == 4:
                d = collect("scripted_det", 24, seed0=230_000)
                gw = float(np.mean([e[2]["win"] for e in d]))
                if gw > bestA:
                    bestA = gw
                    bestA_state = _copy2.deepcopy(
                        (q0.state_dict(), q1.state_dict()))
                    since_best = 0
                else:
                    since_best += 5
                log(f"A2 it {it}/{itersA2} det win {gw:.2f} best {bestA:.2f}")
                if bestA >= 0.85 or since_best >= 40:
                    log("Phase A2 early stop")
                    break
        q0.load_state_dict(bestA_state[0])
        q1.load_state_dict(bestA_state[1])
        log(f"Phase A2 done: listeners at {bestA:.2f} (scripted z)")
        torch.save(r0.state_dict(), run / "r0_A.pt")
        torch.save(r1.state_dict(), run / "r1_A.pt")
        torch.save(q0.state_dict(), run / "q0_A.pt")
        torch.save(q1.state_dict(), run / "q1_A.pt")

        # Phase B: coordinator BC seed + PPO vs frozen diffusion listeners
        log("=== Phase B: coordinator BC seed + PPO ===")
        env = CombatEnv(seed=0, privileged_obs=True)
        steams = (ScriptedTeam(0), ScriptedTeam(1))
        S_l, T_l = [], []
        for ep in range(3 if args.smoke else 120):
            env.reset(seed=310_000 + ep)
            k = 0
            while not env.done:
                if k % DT_COORD_STEPS == 0:
                    S_l.append(coord_state(env))
                    T_l.append((scripted_tokens(env, 0)
                                if args.condition == "C"
                                else scripted_bottleneck(env, 0))[0].numpy())
                a = {}
                for tm in steams:
                    a.update(tm.act(env))
                env.step(a); k += 1
        S_t = torch.from_numpy(np.array(S_l, dtype=np.float32))
        copt = torch.optim.Adam(
            [p for p in coord.parameters() if p.requires_grad], lr=3e-4)
        if args.condition == "C":
            T_t = torch.from_numpy(np.array(T_l, dtype=np.int64))
            for _ in range(20 if args.smoke else 600):
                mb = torch.randint(0, len(S_t), (128,))
                lp, _ = coord.log_prob(S_t[mb], T_t[mb])
                loss = -lp.mean()
                copt.zero_grad(); loss.backward(); copt.step()
        else:
            T_t = torch.from_numpy(np.array(T_l, dtype=np.float32))
            for s in range(20 if args.smoke else 3000):
                if s == 1500:
                    for g in copt.param_groups:
                        g["lr"] = 1e-4
                mb = torch.randint(0, len(S_t), (128,))
                loss = torch.mean((coord.enc(S_t[mb]) - T_t[mb]) ** 2)
                copt.zero_grad(); loss.backward(); copt.step()
        log(f"coordinator BC seed done (loss {float(loss):.3f})")

        import copy as _copy
        coptv = torch.optim.Adam(
            [p for p in coord.parameters() if p.requires_grad]
            + list(cvalue.parameters()), lr=1e-4)
        best = greedy_win(_ser(coord))
        best_state = _copy.deepcopy(coord.state_dict())
        log(f"B baseline greedy win {best:.2f}")
        itersB = 2 if args.smoke else 30
        upd = coord_ppo_update if args.condition == "C" else emb_coord_ppo_update
        for it in range(itersB):
            ent_coef = 0.01 * max(0.0, 1.0 - it / (0.6 * itersB))
            eps = collect("learned", E, cb=_ser(coord),
                          seed0=410_000 + it * 1000)
            ent = upd(coord, cvalue, coptv, eps, ent_coef)
            if it % 10 == 9:
                gw = greedy_win(_ser(coord))
                if gw > best:
                    best, best_state = gw, _copy.deepcopy(coord.state_dict())
                log(f"B it {it}/{itersB} ent {ent:.2f} greedy {gw:.2f} "
                    f"best {best:.2f}")
        coord.load_state_dict(best_state)
        log(f"Phase B done (best greedy {best:.2f})")

        # Phase C: coordinator-only polish, LISTENERS FROZEN.
        # Run 2 taught: joint elite-SI on diffusion listeners under a
        # shifting coordinator catastrophically drifts them (0.67 → 0.00);
        # unlike low-lr PPO on FF listeners, elite-SI retraining is not a
        # gentle joint update. The coordinator keeps adapting to the
        # frozen (Phase-A2-competent) listeners.
        log("=== Phase C: coord PPO + GENTLE listener adaptation ===")
        # run-2 (C) lesson: lr 3e-4 / 200-step elite-SI here is
        # catastrophic (0.67→0.00). 5e-5 / 60 steps with best-joint
        # tracking lets listeners adapt to the learned coordinator's
        # near-miss messages without drifting (needed for the continuous
        # channel — run 1 of B: frozen listeners only read exact vectors).
        bestC = greedy_win(_ser(coord))
        bestC_state = _copy.deepcopy((q0.state_dict(), q1.state_dict(),
                                      coord.state_dict()))
        itersC = 2 if args.smoke else 25
        for it in range(itersC):
            eps = collect("learned", E, cb=_ser(coord),
                          seed0=510_000 + it * 1000)
            q0.load_state_dict(fit_scorer(q0, eps, "red0").state_dict())
            q1.load_state_dict(fit_scorer(q1, eps, "red1").state_dict())
            ent = upd(coord, cvalue, coptv, eps, 0.0)
            if it % 5 == 4:
                gw = greedy_win(_ser(coord))
                if gw > bestC:
                    bestC = gw
                    bestC_state = _copy.deepcopy(
                        (q0.state_dict(), q1.state_dict(),
                         coord.state_dict()))
                log(f"C it {it}/{itersC} greedy {gw:.2f} best {bestC:.2f}")
        q0.load_state_dict(bestC_state[0])
        q1.load_state_dict(bestC_state[1])
        coord.load_state_dict(bestC_state[2])
        for n, m in (("r0_final", r0), ("r1_final", r1),
                     ("q0_final", q0), ("q1_final", q1),
                     ("coord_final", coord)):
            torch.save(m.state_dict(), run / f"{n}.pt")

        # G6 (frozen elite opponent)
        log("=== G6 (elite opponent) ===")
        n_eval = 8 if args.smoke else 200
        on = collect("learned_det_elite", n_eval, cb=_ser(coord),
                     seed0=930_000)
        off = collect("zeroed_det_elite", n_eval, seed0=930_000)
        win_on = float(np.mean([e[2]["win"] for e in on]))
        win_off = float(np.mean([e[2]["win"] for e in off]))
        msgs = np.concatenate([e[1]["tok"] for e in on if len(e[1]["s"])])
        # diversity vs scripted reference
        S_ref, T_ref = [], []
        env.reset(seed=990_000)
        envs = CombatEnv(seed=0, privileged_obs=True)
        for ep in range(20):
            envs.reset(seed=990_000 + ep)
            k = 0
            while not envs.done:
                if k % DT_COORD_STEPS == 0:
                    T_ref.append((scripted_tokens(envs, 0)
                                  if args.condition == "C"
                                  else scripted_bottleneck(envs, 0))[0].numpy())
                a = {}
                for tm in steams:
                    a.update(tm.act(envs))
                envs.step(a); k += 1
        sys.path.insert(0, str(ROOT / "tools"))
        from eval_g6 import marginal_entropy
        if args.condition == "C":
            msgs = msgs.astype(np.int64)
        h_l = marginal_entropy(msgs)
        h_s = marginal_entropy(np.array(T_ref))
        drop = (win_on - win_off) * 100
        checks = {"win_rate": (win_on, win_on >= 0.75),
                  "diversity_ratio": (h_l / max(h_s, 1e-9),
                                      h_l / max(h_s, 1e-9) >= 0.5),
                  "causal_drop_pts": (drop, drop >= 15.0)}
        for k2, (v, ok) in checks.items():
            log(f"G6 {k2}: {v:.2f} → {'OK' if ok else 'FAIL'}")
        log(f"G6 detail: on={win_on:.2f} off={win_off:.2f}")
        res = {k2: {"value": float(v), "ok": bool(ok)}
               for k2, (v, ok) in checks.items()}
        res["pass"] = all(c["ok"] for c in res.values() if isinstance(c, dict))
        (run / "G6_RESULT_v2.json").write_text(json.dumps(res, indent=1))
        log(f"G6 gate: {'PASS' if res['pass'] else 'FAIL'}")
        return 0 if res["pass"] else 1
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    raise SystemExit(main())
