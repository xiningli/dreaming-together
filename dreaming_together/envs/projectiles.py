"""Projectiles: cone spread sampling and the ray-swept pellet manager.

Anti-tunneling (rule R5)
------------------------
At 30 m/s and dt=2 ms a pellet moves 0.06 m per step — comparable to the
thickness of every collider in the arena. Discrete contact detection alone
is probabilistic at oblique incidence (found by the randomized G0 trials:
1/200 pellets slipped through the raised shield). Authoritative hit
detection is therefore a continuous ray sweep: each step, the segment from
the pellet's previous position to its new position is ray-cast; the first
geom the segment crosses is the hit. Tunneling is impossible by
construction. The pellet body still carries contact physics so stray
pellets never rest inside geometry, but game-layer hits come from the
sweep, not from the contact list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from dreaming_together.envs.tank import CONE_HALF_ANGLE

# Parked pellets live here, far above the arena, until spawned.
PARK_POS = np.array([0.0, 0.0, 100.0])


def sample_pellet_directions(muzzle_dir: np.ndarray,
                              n: int,
                              seed: int | None = None) -> np.ndarray:
    """Return (n, 3) unit vectors spread within CONE_HALF_ANGLE of muzzle_dir.

    Each direction is a unit vector; all are within tank.CONE_HALF_ANGLE degrees
    of muzzle_dir.  seed is optional for reproducibility in tests.

    Algorithm: uniform sampling on the spherical cap within CONE_HALF_ANGLE.
    Uses rejection-free method: sample (phi, z) where z is cos of elevation.
    """
    rng = np.random.default_rng(seed)
    half_angle_rad = np.radians(CONE_HALF_ANGLE)

    # Sample azimuth uniformly and elevation uniformly in solid angle.
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(np.cos(half_angle_rad), 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)

    # Directions in a frame where muzzle_dir = +z.
    local = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta,
    ])

    # Rotate local frame to world frame so local +z aligns with muzzle_dir.
    d = muzzle_dir / np.linalg.norm(muzzle_dir)
    # Build rotation matrix from +z to d.
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        # d is already ±z; handle parallel/anti-parallel cases.
        rot = np.eye(3) if d[2] > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(z, d), -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return (rot @ local.T).T


@dataclass
class PelletHit:
    """One resolved pellet impact."""
    pellet_body: str      # e.g. "pellet_3"
    geom_id: int          # geom the sweep crossed
    geom_name: str
    point: np.ndarray     # world-frame impact point
    shooter: str | None   # prefix passed at spawn time (for attribution)


class ProjectileManager:
    """Manages a pre-allocated pool of pellet bodies with ray-swept hits.

    Usage:
        pm = ProjectileManager(model, data)          # discovers pellet_* bodies
        pm.spawn(pos, vel, shooter="red1")
        for _ in range(n):
            mujoco.mj_step(model, data)
            pm.step()                                # sweep + retire
        hits = pm.drain_hits()                       # list[PelletHit]
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 max_range: float = 20.0):
        self.model = model
        self.data = data
        self.max_range = max_range

        self._names: list[str] = []
        self._bids: list[int] = []
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith("pellet_"):
                self._names.append(name)
                self._bids.append(bid)
        if not self._bids:
            raise ValueError("Model has no pellet_* bodies to manage")

        n = len(self._bids)
        self._active = np.zeros(n, dtype=bool)
        self._prev_pos = np.zeros((n, 3))
        self._shooter: list[str | None] = [None] * n
        self._hits: list[PelletHit] = []
        self._pellet_bids = set(self._bids)

    # ------------------------------------------------------------------
    def _qadr(self, slot: int) -> tuple[int, int]:
        bid = self._bids[slot]
        jid = self.model.body_jntadr[bid]
        return self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid]

    def _park(self, slot: int) -> None:
        qadr, vadr = self._qadr(slot)
        self.data.qpos[qadr:qadr + 3] = PARK_POS + np.array([0.0, 0.0, slot * 0.5])
        self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[vadr:vadr + 6] = 0.0
        self._active[slot] = False
        self._shooter[slot] = None

    # ------------------------------------------------------------------
    def spawn(self, pos: np.ndarray, vel: np.ndarray,
              shooter: str | None = None) -> str | None:
        """Claim a free pellet slot; returns the body name, or None if the
        pool is exhausted (caller may drop the pellet — log it upstream)."""
        free = np.flatnonzero(~self._active)
        if free.size == 0:
            return None
        slot = int(free[0])
        qadr, vadr = self._qadr(slot)
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[vadr:vadr + 3] = vel
        self.data.qvel[vadr + 3:vadr + 6] = 0.0
        self._active[slot] = True
        self._prev_pos[slot] = np.asarray(pos, dtype=float)
        self._shooter[slot] = shooter
        return self._names[slot]

    def _sweep(self, origin: np.ndarray, d: np.ndarray,
               seg_len: float, bid: int) -> tuple[int, float] | None:
        """First *colliding* geom crossed by the segment, or None.

        mj_ray intersects every geom geometrically, including visual-only
        ones (contype=0 conaffinity=0 arm capsules and barrels) that pellets
        physically fly through, and sibling pellets (an 8-pellet cone spawns
        coincident at the muzzle) — re-cast just past both.
        """
        geomid = np.array([-1], dtype=np.int32)
        travelled = 0.0
        for _ in range(12):
            dist = mujoco.mj_ray(self.model, self.data,
                                 origin + d * travelled, d,
                                 None, 1, bid, geomid)
            if dist < 0.0 or geomid[0] < 0:
                return None
            travelled += dist
            if travelled > seg_len:
                return None
            gid = int(geomid[0])
            solid = (self.model.geom_contype[gid]
                     or self.model.geom_conaffinity[gid])
            if solid and int(self.model.geom_bodyid[gid]) not in self._pellet_bids:
                return gid, travelled
            travelled += 1e-4   # step past non-colliding geoms and pellets

    def step(self) -> None:
        """Call once after each mj_step: ray-sweep every active pellet over
        the segment it just travelled; register the first crossing as a hit
        and retire the pellet. Also retires out-of-range pellets."""
        for slot in np.flatnonzero(self._active):
            bid = self._bids[slot]
            new_pos = self.data.xpos[bid].copy()
            seg = new_pos - self._prev_pos[slot]
            seg_len = float(np.linalg.norm(seg))
            if seg_len > 1e-9:
                d = seg / seg_len
                hit = self._sweep(self._prev_pos[slot], d, seg_len, bid)
                if hit is not None:
                    gid, dist = hit
                    gname = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                    point = self._prev_pos[slot] + d * dist
                    self._hits.append(PelletHit(
                        pellet_body=self._names[slot],
                        geom_id=gid, geom_name=gname,
                        point=point, shooter=self._shooter[slot]))
                    self._park(slot)
                    continue
            self._prev_pos[slot] = new_pos
            if np.linalg.norm(new_pos[:2]) > self.max_range or new_pos[2] < -1.0:
                self._park(slot)

    def drain_hits(self) -> list[PelletHit]:
        """Return accumulated hits and clear the buffer."""
        out = self._hits
        self._hits = []
        return out

    @property
    def n_active(self) -> int:
        return int(self._active.sum())
