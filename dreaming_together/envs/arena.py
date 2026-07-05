"""Full combat arena builder: 8×8 m walled arena, four tanks, pellet pool.

First real composition of the validated pieces (design §3.1). Used by the
end-to-end visual test and, next, by CombatEnv and the G2 combat-signal
gate. Spawn positions default to the mirror-symmetric spec but are
overridable so scripted scenarios can choreograph formations.
"""
from __future__ import annotations

import numpy as np
import mujoco

from dreaming_together.envs.tank import tank_body_xml, actuator_xml

ARENA_HALF   = 4.0    # m — 8×8 arena
WALL_HEIGHT  = 1.2    # m
WALL_THICK   = 0.10   # m

TEAM_RGBA = {
    "red0":  "0.85 0.30 0.25 1",
    "red1":  "0.75 0.15 0.15 1",
    "blue0": "0.25 0.45 0.90 1",
    "blue1": "0.15 0.25 0.80 1",
}

ROLES = {"red0": "shield", "red1": "shotgun",
         "blue0": "shield", "blue1": "shotgun"}

DEFAULT_SPAWNS = {
    "red0":  ((-3.0,  0.6), 0.0),
    "red1":  ((-3.0, -0.6), 0.0),
    "blue0": (( 3.0, -0.6), np.pi),
    "blue1": (( 3.0,  0.6), np.pi),
}

# Header duplicated from tests/helpers.py PHYSICS_XML_HEADER on purpose:
# envs must not import from tests. Keep the two in sync.
ARENA_XML_HEADER = """\
<option timestep="0.002" iterations="20" solver="Newton"
        cone="elliptic" gravity="0 0 -9.81"/>
<compiler angle="radian" autolimits="true"/>
<!-- Pin extent: parked pellets at z=100 otherwise inflate stat.extent and
     the camera near-clip plane (map.znear x extent) clips close geometry. -->
<statistic extent="8" center="0 0 0.5"/>
<visual><map znear="0.005"/></visual>
<default>
  <default class="tank">
    <geom condim="3" friction="0.8 0.02 0.01"/>
    <joint damping="2" armature="0.02" limited="true"/>
  </default>
</default>
"""


def _walls_xml() -> str:
    a, h, t = ARENA_HALF, WALL_HEIGHT / 2, WALL_THICK / 2
    walls = [
        ("wall_n", (0,  a + t), (a + 2 * t, t)),
        ("wall_s", (0, -a - t), (a + 2 * t, t)),
        ("wall_e", ( a + t, 0), (t, a)),
        ("wall_w", (-a - t, 0), (t, a)),
    ]
    out = []
    for name, (x, y), (sx, sy) in walls:
        out.append(
            f'<geom name="{name}" type="box" pos="{x} {y} {h}" '
            f'size="{sx} {sy} {h}" rgba="0.45 0.45 0.5 1" '
            f'contype="1" conaffinity="7"/>')
    return "\n    ".join(out)


def _pellet_pool_xml(n: int) -> str:
    out = []
    for i in range(n):
        out.append(
            f'<body name="pellet_{i}" pos="0 0 {100 + i * 0.5}">'
            f'<freejoint name="pellet_{i}_jnt"/>'
            # conaffinity 5 (=1|4): collides with hulls/walls/shields but
            # NOT with other pellets (contype 2 & 5 = 0) — an 8-pellet cone
            # spawns coincident at the muzzle and must not self-explode.
            f'<geom name="pellet_{i}_g" type="sphere" size="0.01" mass="0.005"'
            f' contype="2" conaffinity="5" rgba="1 0.8 0 1"/></body>')
    return "\n    ".join(out)


def build_combat_arena(spawns: dict | None = None,
                       n_pellets: int = 16
                       ) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Four tanks in the walled arena. spawns: {prefix: ((x, y), yaw)}."""
    spawns = spawns or DEFAULT_SPAWNS
    bodies, acts = [], []
    for prefix, ((x, y), yaw) in spawns.items():
        bodies.append(tank_body_xml(prefix, TEAM_RGBA[prefix], (x, y), yaw,
                                    ROLES[prefix]))
        acts.append(actuator_xml(prefix))
    xml = f"""<mujoco model="combat_arena">
  {ARENA_XML_HEADER}
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="-4 -4 6" dir="0.5 0.5 -0.7" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="{ARENA_HALF + 1} {ARENA_HALF + 1} 0.1"
          friction="0.5 0.01 0.001" rgba="0.55 0.57 0.55 1"/>
    {_walls_xml()}
    {chr(10).join(bodies)}
    {_pellet_pool_xml(n_pellets)}
  </worldbody>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
