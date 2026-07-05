"""CombatEnv — the 2v2 game layer over the validated physics substrate.

Design §3.1 and the G2 gate. Per policy step (50 ms = 25 physics substeps):
  - actions: {prefix: a ∈ [-1,1]^5} = [left_track, right_track, arm_pan,
    arm_tilt, trigger]; trigger fires iff shotgun role, alive, cooldown
    elapsed, and trigger > 0.5. Arm channels are PD targets (set_arm_target,
    never teleport). Dead agents' controls are zeroed; hulls remain as
    obstacles.
  - hits: ray-swept ProjectileManager (authoritative); 6 HP per hull hit.
  - termination: any agent at 0 HP → that team loses (one death = team
    loss); episode cap 30 s → draw.
  - rewards: envs/rewards.py compute_rewards over a fully populated
    RewardContext (every field driven by real events — rule R4).
  - spawns: mirror-symmetric, separation ~ U(1.5, 5.0) m, red at −sep/2
    facing +x, blue at +sep/2 facing −x, agents at y = ±0.6 (rule R2 — the
    v0 fixed 6 m spawn vs 3 m weapon range produced zero combat signal).

Observations are the 16-dim proprio vector of design §3.6. Vision (the
SegCamera channel) is wired in at Stage 0/T3; scripted oracles act on
privileged state and do not use obs.

Privileged mode (`privileged_obs=True`): appends 11 dims of state the
design expects vision to carry — per-enemy body-frame bearing (sin, cos),
distance, alive flag, and teammate bearing/distance. Used ONLY by
pathfinder training runs before the vision encoder exists; the A/B/C
experiment itself must use identical (vision) observations across
conditions.
"""
from __future__ import annotations

import numpy as np
import mujoco

from dreaming_together.envs.arena import build_combat_arena, ROLES
from dreaming_together.envs.projectiles import (
    ProjectileManager, sample_pellet_directions,
)
from dreaming_together.envs.rewards import RewardContext, compute_rewards
from dreaming_together.envs.tank import (
    HULL_Z, N_PELLETS, PELLET_SPEED,
    ARM_PAN_RANGE, ARM_TILT_RANGE,
    set_track_ctrl, set_arm_target, arm_angles,
    muzzle_pos, muzzle_dir, hull_pos, hull_yaw,
    window_quality, residual_coverage,
)

PREFIXES = ("red0", "red1", "blue0", "blue1")   # index order is fixed
TEAM_OF = {"red0": 0, "red1": 0, "blue0": 1, "blue1": 1}
SHOTGUN_OF_TEAM = {0: "red1", 1: "blue1"}
SHIELD_OF_TEAM = {0: "red0", 1: "blue0"}

DT_POLICY = 0.05
SUBSTEPS = 25
EPISODE_CAP_S = 30.0
FIRE_PERIOD_S = 1.2
HP_MAX = 100
HP_PER_PELLET = 6
SHOTGUN_RANGE = 3.0
WINDOW_OPEN_WQ = 0.25
ASSIST_WINDOW_AGE_S = 1.5

OBS_DIM = 16
OBS_DIM_PRIV = 27


def _norm(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def _denorm(a, lo, hi):
    return lo + (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)


class CombatEnv:
    """2v2 combat episode. Not vectorized; wrap for parallelism."""

    def __init__(self, seed: int = 0,
                 spawn_sep_range: tuple[float, float] = (1.5, 5.0),
                 episode_cap_s: float = EPISODE_CAP_S,
                 privileged_obs: bool = False):
        self.rng = np.random.default_rng(seed)
        self.spawn_sep_range = spawn_sep_range
        self.episode_cap_s = episode_cap_s
        self.privileged_obs = privileged_obs
        self.model, self.data = build_combat_arena(n_pellets=32)
        self._hull_qadr = {}
        for p in PREFIXES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{p}_hull")
            jid = self.model.body_jntadr[bid]
            self._hull_qadr[p] = (self.model.jnt_qposadr[jid],
                                  self.model.jnt_dofadr[jid])
        self.pm = ProjectileManager(self.model, self.data)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.pm = ProjectileManager(self.model, self.data)

        sep = float(self.rng.uniform(*self.spawn_sep_range))
        self.spawn_separation = sep
        spawn = {
            "red0":  (-sep / 2,  0.6, 0.0),
            "red1":  (-sep / 2, -0.6, 0.0),
            "blue0": ( sep / 2, -0.6, np.pi),
            "blue1": ( sep / 2,  0.6, np.pi),
        }
        for p, (x, y, yaw) in spawn.items():
            qadr, vadr = self._hull_qadr[p]
            self.data.qpos[qadr:qadr + 3] = [x, y, HULL_Z]
            self.data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0,
                                                 np.sin(yaw / 2)]
            self.data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.t = 0.0
        self.hp = {p: HP_MAX for p in PREFIXES}
        self.cooldown = {p: 0.0 for p in PREFIXES}
        self.prev_action = np.zeros((4, 5))
        self.prev_dist = self._dists_to_nearest_opp()
        self.window_open_since = {0: None, 1: None}
        self.done = False
        self.team_result = np.zeros(2, dtype=int)
        self.episode_events: list[str] = []
        return self._obs_all()

    # ------------------------------------------------------------------
    def _alive(self, p: str) -> bool:
        return self.hp[p] > 0

    def _nearest_opp(self, p: str) -> str:
        opps = [o for o in PREFIXES if TEAM_OF[o] != TEAM_OF[p]]
        alive = [o for o in opps if self._alive(o)] or opps
        pos = hull_pos(self.model, self.data, p)[:2]
        return min(alive, key=lambda o: np.linalg.norm(
            hull_pos(self.model, self.data, o)[:2] - pos))

    def _dists_to_nearest_opp(self) -> np.ndarray:
        out = np.zeros(4)
        for i, p in enumerate(PREFIXES):
            o = self._nearest_opp(p)
            out[i] = np.linalg.norm(
                hull_pos(self.model, self.data, p)[:2]
                - hull_pos(self.model, self.data, o)[:2])
        return out

    def _team_wq(self, team: int) -> float:
        return window_quality(self.model, self.data,
                              SHOTGUN_OF_TEAM[team], SHIELD_OF_TEAM[team])

    def _team_cr(self, team: int) -> float:
        return residual_coverage(self.model, self.data,
                                 SHOTGUN_OF_TEAM[1 - team],
                                 SHIELD_OF_TEAM[team])

    # ------------------------------------------------------------------
    def step(self, actions: dict[str, np.ndarray]):
        assert not self.done, "episode is done — call reset()"
        act_mat = np.zeros((4, 5))

        # apply controls and fire
        for i, p in enumerate(PREFIXES):
            a = np.clip(np.asarray(actions[p], dtype=float), -1.0, 1.0)
            act_mat[i] = a
            if not self._alive(p):
                set_track_ctrl(self.model, self.data, p, 0.0, 0.0)
                continue
            set_track_ctrl(self.model, self.data, p, a[0], a[1])
            set_arm_target(self.model, self.data, p,
                           _denorm(a[2], *ARM_PAN_RANGE),
                           _denorm(a[3], *ARM_TILT_RANGE))
            self.cooldown[p] = max(0.0, self.cooldown[p] - DT_POLICY)
            if (ROLES[p] == "shotgun" and a[4] > 0.5
                    and self.cooldown[p] <= 0.0):
                origin = muzzle_pos(self.model, self.data, p)
                dirs = sample_pellet_directions(
                    muzzle_dir(self.model, self.data, p), N_PELLETS,
                    seed=int(self.rng.integers(2 ** 31)))
                for d in dirs:
                    self.pm.spawn(origin, d * PELLET_SPEED, shooter=p)
                self.cooldown[p] = FIRE_PERIOD_S
                self.episode_events.append(f"{self.t:.2f} {p} fires")

        # physics
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            self.pm.step()
        self.t += DT_POLICY

        # resolve hits
        damage = np.zeros(4)
        pellet_hits_opp = np.zeros(4, dtype=int)
        friendly_pellets = np.zeros(4, dtype=int)
        kills = np.zeros(4, dtype=int)
        killed = np.zeros(4, dtype=bool)
        blocked_by = {0: 0, 1: 0}
        landed_on_opp_by_team = {0: 0, 1: 0}
        for h in self.pm.drain_hits():
            shooter_i = PREFIXES.index(h.shooter) if h.shooter in PREFIXES else -1
            if h.geom_name.endswith("_hull_g"):
                victim = h.geom_name[:-len("_hull_g")]
                vi = PREFIXES.index(victim)
                if self.hp[victim] > 0:
                    self.hp[victim] = max(0, self.hp[victim] - HP_PER_PELLET)
                    damage[vi] += HP_PER_PELLET
                    if shooter_i >= 0:
                        if TEAM_OF[h.shooter] == TEAM_OF[victim]:
                            friendly_pellets[shooter_i] += 1
                        else:
                            pellet_hits_opp[shooter_i] += 1
                            landed_on_opp_by_team[TEAM_OF[h.shooter]] += 1
                    if self.hp[victim] == 0:
                        killed[vi] = True
                        if shooter_i >= 0 and TEAM_OF[h.shooter] != TEAM_OF[victim]:
                            kills[shooter_i] += 1
                        self.episode_events.append(
                            f"{self.t:.2f} {victim} eliminated by {h.shooter}")
            elif h.geom_name.endswith("_shield_g"):
                owner = h.geom_name[:-len("_shield_g")]
                blocked_by[TEAM_OF[owner]] += 1

        # window bookkeeping per team
        wq = {t: self._team_wq(t) for t in (0, 1)}
        cr = {t: self._team_cr(t) for t in (0, 1)}
        window_open = np.zeros(2, dtype=bool)
        window_assist = np.zeros(2, dtype=bool)
        shot_4plus = np.zeros(2, dtype=bool)
        for team in (0, 1):
            open_now = wq[team] > WINDOW_OPEN_WQ
            window_open[team] = open_now
            if open_now and self.window_open_since[team] is None:
                self.window_open_since[team] = self.t
            if not open_now:
                self.window_open_since[team] = None
            if open_now and landed_on_opp_by_team[team] >= 1:
                age = self.t - (self.window_open_since[team] or self.t)
                window_assist[team] = age < ASSIST_WINDOW_AGE_S
            shot_4plus[team] = open_now and landed_on_opp_by_team[team] >= 4

        # termination
        alive = np.array([self._alive(p) for p in PREFIXES])
        team_dead = np.array([killed[0] or killed[1] or not (alive[0] and alive[1]),
                              killed[2] or killed[3] or not (alive[2] and alive[3])])
        self.team_result = np.zeros(2, dtype=int)
        if team_dead.any():
            self.done = True
            if team_dead.all():
                self.team_result[:] = 0          # mutual destruction: draw
            else:
                loser = 0 if team_dead[0] else 1
                self.team_result[loser] = -1
                self.team_result[1 - loser] = 1
        elif self.t >= self.episode_cap_s:
            self.done = True                      # draw: result stays 0

        # rewards
        dists = self._dists_to_nearest_opp()
        opp_in_reload = np.array([
            self.cooldown[SHOTGUN_OF_TEAM[1]] > 0.15,
            self.cooldown[SHOTGUN_OF_TEAM[0]] > 0.15,
        ])
        died_before_teammate = np.array([
            killed[i] and self._teammate_alive(i) for i in range(4)])
        ctx = RewardContext(
            dt=DT_POLICY,
            alive=alive | killed,   # agents that died this step still get terminal
            damage_taken=damage,
            killed_this_step=killed,
            dist_to_nearest_opp=dists,
            prev_dist_to_nearest_opp=self.prev_dist,
            action=act_mat,
            prev_action=self.prev_action,
            episode_done=self.done,
            team_result=self.team_result,
            shield_blocks_line=np.array([cr[0] > 0.5, cr[1] > 0.5]),
            window_open=window_open,
            opp_in_reload=opp_in_reload,
            window_assist=window_assist,
            pellet_hits_opp=pellet_hits_opp,
            friendly_pellets=friendly_pellets,
            kills=kills,
            within_3m=dists < SHOTGUN_RANGE,
            shot_through_window_4plus=shot_4plus,
            died_before_teammate=died_before_teammate,
        )
        rewards = compute_rewards(ctx)

        self.prev_action = act_mat.copy()
        self.prev_dist = dists
        info = {
            "hp": dict(self.hp),
            "t": self.t,
            "team_result": self.team_result.copy(),
            "blocked": blocked_by,
            "pellet_hits": pellet_hits_opp.copy(),
            "wq": wq, "cr": cr,
            "damage": damage.copy(),
        }
        return self._obs_all(), rewards, self.done, info

    def _teammate_alive(self, i: int) -> bool:
        mate = i + 1 if i % 2 == 0 else i - 1
        return self._alive(PREFIXES[mate])

    # ------------------------------------------------------------------
    def obs(self, p: str) -> np.ndarray:
        """16-dim proprio vector (design §3.6)."""
        i = PREFIXES.index(p)
        team = TEAM_OF[p]
        yaw = hull_yaw(self.model, self.data, p)
        c, s = np.cos(yaw), np.sin(yaw)
        qadr, vadr = self._hull_qadr[p]
        vx_w, vy_w = self.data.qvel[vadr:vadr + 2]
        wz = self.data.qvel[vadr + 5]
        pos = hull_pos(self.model, self.data, p)
        pan, tilt = arm_angles(self.model, self.data, p)
        pan_v, tilt_v = 0.0, 0.0
        for jname, slot in (("arm_pan", 0), ("arm_tilt", 1)):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{p}_{jname}")
            v = self.data.qvel[self.model.jnt_dofadr[jid]]
            if slot == 0: pan_v = v
            else: tilt_v = v

        wq = self._team_wq(team)
        window = float(wq > WINDOW_OPEN_WQ)
        if ROLES[p] == "shield":
            imp = [_norm(pan, *ARM_PAN_RANGE), window, self._team_cr(team)]
        else:
            imp = [self.cooldown[p] / FIRE_PERIOD_S, window, wq]

        base = np.array([
            s, c,
            c * vx_w + s * vy_w, -s * vx_w + c * vy_w,
            wz,
            pos[0] / 4.0, pos[1] / 4.0,
            _norm(pan, *ARM_PAN_RANGE), _norm(tilt, *ARM_TILT_RANGE),
            pan_v / 5.0, tilt_v / 5.0,
            *imp,
            max(0.0, self.episode_cap_s - self.t) / self.episode_cap_s,
            self.hp[p] / HP_MAX,
        ], dtype=np.float32)
        if not self.privileged_obs:
            return base

        def rel(other: str) -> list[float]:
            v = hull_pos(self.model, self.data, other)[:2] - pos[:2]
            d = float(np.linalg.norm(v))
            bearing = np.arctan2(v[1], v[0]) - yaw
            return [np.sin(bearing), np.cos(bearing), d / 4.0]

        enemies = sorted(
            (o for o in PREFIXES if TEAM_OF[o] != team),
            key=lambda o: np.linalg.norm(
                hull_pos(self.model, self.data, o)[:2] - pos[:2]))
        mate = [o for o in PREFIXES
                if TEAM_OF[o] == team and o != p][0]
        extra = (rel(enemies[0]) + [float(self._alive(enemies[0]))]
                 + rel(enemies[1]) + [float(self._alive(enemies[1]))]
                 + rel(mate))
        return np.concatenate([base, np.array(extra, dtype=np.float32)])

    def _obs_all(self) -> dict[str, np.ndarray]:
        return {p: self.obs(p) for p in PREFIXES}

    # ------------------------------------------------------------------
    # Vision observations (Stage 0 onward). Lazy init so the SegCamera's
    # EGL context is created inside whichever process actually renders —
    # EGL contexts are thread/process-local (engineering note 4).
    # ------------------------------------------------------------------
    def enable_vision(self, encoder_path: str) -> None:
        self._vision_path = str(encoder_path)
        self._seg_cam = None
        self._encoder = None

    def vision_obs(self, p: str) -> np.ndarray:
        """Fused observation: frozen-encoder embeddings of the agent's
        camera(s) + the 16-dim proprio. Shotgun: 256+16=272 dims;
        shield (front+rear): 512+16=528 dims."""
        import torch
        if self._encoder is None:
            from dreaming_together.vision.encoder import SegEncoder
            from dreaming_together.envs.cameras import SegCamera
            self._encoder = SegEncoder()
            self._encoder.load_state_dict(
                torch.load(self._vision_path, weights_only=True))
            self._encoder.eval()
            self._seg_cam = SegCamera(self.model)
        cams = [f"{p}_front_cam"]
        if ROLES[p] == "shield":
            cams.append(f"{p}_rear_cam")
        frames = np.stack([self._seg_cam.render(self.data, c) for c in cams])
        with torch.no_grad():
            z = self._encoder(torch.from_numpy(frames)).flatten().numpy()
        # proprio without the privileged extras, regardless of env mode
        was = self.privileged_obs
        self.privileged_obs = False
        base = self.obs(p)
        self.privileged_obs = was
        return np.concatenate([z.astype(np.float32), base])
