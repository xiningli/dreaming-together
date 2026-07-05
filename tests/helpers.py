"""Shared MuJoCo model builders and low-level helpers for all physics unit tests.

Design principle
----------------
Each test builds the *minimum* model that exercises one concern:

  build_solo_tank(role, prefix)
      One tank agent on a minimal floor.  Used by movement tests and arm FK
      tests.  No target, no opponent.

  build_shooting_range(target_distance)
      One shotgun tank (red1) at origin facing +x, plus a static paper target
      at (target_distance, 0, TARGET_Z).  Used by shooting tests.  No shield.

  build_interception_range(shield_distance, target_distance)
      A fixed world-frame muzzle origin, one shield tank (red0) at
      shield_distance along x, and a paper target at target_distance.
      Used by shield / window tests.  No shooter tank body needed — rays are
      cast from the fixed muzzle origin directly.

All models use dt_phys = 2 ms, identical to the full arena.

Paper target
------------
A static rigid body welded to the world, named "paper_target".  Geom name
"paper_target_g".  Dimensions TARGET_W × TARGET_W × TARGET_DEPTH (face plate
in the YZ plane, normal pointing along -x so shots arrive head-on).

Muzzle origin (interception range only)
----------------------------------------
A world site named "shooter_muzzle" at (0, 0, MUZZLE_Z).  No body — the site
is used as the source point for mj_ray calls and pellet spawning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dreaming_together.envs.tank import (
    tank_body_xml,
    actuator_xml,
    HULL_Z,
    ARM_MOUNT_LOCAL,
    ARM_LEN,
    END_EFFECTOR_LEN,
)

# Paper target dimensions
TARGET_W     = 0.50   # m — width and height of the target face
TARGET_DEPTH = 0.02   # m — thickness
TARGET_Z     = HULL_Z + ARM_MOUNT_LOCAL[2] + 0.05   # approximately muzzle height

# Fixed muzzle origin height for interception tests (world site)
MUZZLE_Z     = TARGET_Z

PHYSICS_XML_HEADER = """\
<option timestep="0.002" iterations="20" solver="Newton"
        cone="elliptic" gravity="0 0 -9.81"/>
<compiler angle="radian" autolimits="true"/>
<!-- Pin extent: parked pellets at z=100 otherwise inflate stat.extent to
     ~100 m, pushing the camera near-clip plane (znear = map.znear x extent)
     past 1 m and silently clipping everything close - including the shield
     in front of the shield agent's own camera. -->
<statistic extent="8" center="0 0 0.5"/>
<visual><map znear="0.005"/></visual>
<default>
  <default class="tank">
    <geom condim="3" friction="0.8 0.02 0.01"/>
    <joint damping="2" armature="0.02" limited="true"/>
  </default>
</default>
"""


def _floor_xml() -> str:
    # friction="0.5 ..." keeps effective contact friction at hull-class value (0.8)
    # because MuJoCo contact friction = max(geom1, geom2); floor default (1.0)
    # was overriding the hull friction and making the tank unable to drive.
    return '<geom name="floor" type="plane" size="10 10 0.1" friction="0.5 0.01 0.001" rgba="0.6 0.6 0.6 1"/>'


def _paper_target_xml(distance: float) -> str:
    """Static paper target in the YZ plane at x=distance."""
    hw = TARGET_W / 2
    return (
        f'<body name="paper_target" pos="{distance} 0 {TARGET_Z}">\n'
        f'  <geom name="paper_target_g" type="box"'
        f' size="{TARGET_DEPTH/2} {hw} {hw}"'
        f' rgba="1 0.9 0.7 1" contype="1" conaffinity="7"/>\n'
        f'</body>'
    )


def _shooter_muzzle_site_xml() -> str:
    """World-frame site used as mj_ray origin in interception tests."""
    return f'<site name="shooter_muzzle" pos="0 0 {MUZZLE_Z}" size="0.01"/>'


def build_solo_tank(role: str, prefix: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Minimal single-agent model: one tank on a 10×10 m floor.

    Used for: movement tests, arm FK tests, shield geometry tests (solo).
    """
    body  = tank_body_xml(prefix, "0.8 0.3 0.3 1", (0.0, 0.0), 0.0, role)
    acts  = actuator_xml(prefix)
    xml = f"""<mujoco model="solo_{role}">
  {PHYSICS_XML_HEADER}
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {_floor_xml()}
    {body}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def build_shooting_range(target_distance: float = 2.5) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Shotgun tank (red1) at origin facing +x, paper target at target_distance.

    Includes one pre-allocated pellet body (pellet_0) for projectile tests.
    No shield agent.
    """
    body = tank_body_xml("red1", "0.8 0.3 0.3 1", (0.0, 0.0), 0.0, "shotgun")
    acts = actuator_xml("red1")
    xml = f"""<mujoco model="shooting_range">
  {PHYSICS_XML_HEADER}
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {_floor_xml()}
    {body}
    {_paper_target_xml(target_distance)}
    {_pellet_pool_xml(1)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def build_interception_range(
    shield_distance: float = 1.5,
    target_distance: float = 3.0,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Fixed muzzle origin + shield tank (red0) + paper target.

    Layout along world +x:
      x=0          world site "shooter_muzzle" at (0, 0, MUZZLE_Z)
      x=shield_distance   shield tank (red0), facing +x, arm centred
      x=target_distance   paper target (TARGET_W × TARGET_W face in YZ plane)

    Used for: shield-blocks-ray, window-quality, residual-coverage tests.
    The shooter is just a point — no hull body — so the shield test is
    isolated from any shooter-arm FK dependencies.
    """
    # yaw=π: shield agent faces shooter (local +x → world -x), so arm extends
    # toward shooter and shield sits between shooter and agent hull.
    body = tank_body_xml("red0", "0.3 0.3 0.8 1", (shield_distance, 0.0), np.pi, "shield")
    acts = actuator_xml("red0")
    xml = f"""<mujoco model="interception_range">
  {PHYSICS_XML_HEADER}
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {_floor_xml()}
    {_shooter_muzzle_site_xml()}
    {body}
    {_paper_target_xml(target_distance)}
    {_pellet_pool_xml(1)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ---------------------------------------------------------------------------
# Low-level helpers used directly in tests
# ---------------------------------------------------------------------------

def geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def contacts_between(data: mujoco.MjData,
                     model: mujoco.MjModel,
                     geom_name_a: str,
                     geom_name_b: str) -> bool:
    """Return True if any active contact involves both named geoms."""
    ga = geom_id(model, geom_name_a)
    gb = geom_id(model, geom_name_b)
    for c in range(data.ncon):
        con = data.contact[c]
        if (con.geom1 == ga and con.geom2 == gb) or \
           (con.geom1 == gb and con.geom2 == ga):
            return True
    return False


def any_contact_with(data: mujoco.MjData,
                     model: mujoco.MjModel,
                     geom_name: str) -> bool:
    """Return True if the named geom has any active contact."""
    gid = geom_id(model, geom_name)
    for c in range(data.ncon):
        con = data.contact[c]
        if con.geom1 == gid or con.geom2 == gid:
            return True
    return False


def cast_ray(model: mujoco.MjModel,
             data: mujoco.MjData,
             origin: np.ndarray,
             direction: np.ndarray,
             exclude_body: int = -1) -> tuple[float, int]:
    """Wrapper around mj_ray. Returns (distance, geom_id); distance=-1 if no hit."""
    d = direction / np.linalg.norm(direction)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, origin, d,
                         None, 1, exclude_body, geomid)
    return dist, int(geomid[0])


def _pellet_pool_xml(n: int = 1) -> str:
    """Pre-allocate n free-body pellets at a safe "parked" height (z=100 m).

    Each body gets a freejoint and a sphere geom.  They are parked far above
    the arena so they do not collide with anything at rest.  spawn_pellet()
    moves one into position when a shot is fired.
    """
    lines = []
    for i in range(n):
        lines.append(
            f'<body name="pellet_{i}" pos="0 0 100">'
            f'  <freejoint name="pellet_{i}_jnt"/>'
            f'  <geom name="pellet_{i}_g" type="sphere" size="0.01"'
            f' mass="0.005" contype="2" conaffinity="7" rgba="1 0.8 0 1"/>'
            f'</body>'
        )
    return "\n".join(lines)


def build_opposing_pair(
    shooter_prefix: str,
    shooter_role: str,
    target_prefix: str,
    target_role: str,
    distance: float = 2.5,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Two tanks: shooter at origin facing +x, target at (distance, 0) facing -x.

    Includes one pre-allocated pellet body (pellet_0) for physical projectile tests.

    shooter_prefix / target_prefix determine team membership by naming convention:
      "red0" / "red1" — red team
      "blue0" / "blue1" — blue team
    The test is responsible for which hit constitutes friendly vs hostile fire.
    """
    shooter_body = tank_body_xml(shooter_prefix, "0.8 0.3 0.3 1",
                                 (0.0, 0.0), 0.0, shooter_role)
    target_body  = tank_body_xml(target_prefix,  "0.3 0.3 0.8 1",
                                 (distance, 0.0), np.pi, target_role)
    shooter_acts = actuator_xml(shooter_prefix)
    target_acts  = actuator_xml(target_prefix)
    xml = f"""<mujoco model="opposing_pair">
  {PHYSICS_XML_HEADER}
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {_floor_xml()}
    {shooter_body}
    {target_body}
    {_pellet_pool_xml(1)}
  </worldbody>
  <actuator>{shooter_acts}{target_acts}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def build_friendly_pair(distance: float = 2.5) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Friendly pair: red1 (shotgun) at origin, red0 (shield) at distance facing -x.

    Used for friendly-fire tests: pellet from red1 travels toward red0 (same team).
    """
    return build_opposing_pair(
        shooter_prefix="red1", shooter_role="shotgun",
        target_prefix="red0",  target_role="shield",
        distance=distance,
    )


def build_column_with_enemy(
    shield_distance: float = 1.5,
    enemy_distance: float = 3.0,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Column formation: red1 (shotgun) at origin, red0 (shield) ahead, blue1 (enemy) further.

    Tests that a friendly shield blocking the corridor stops red1's pellets
    before they reach blue1.  This is the core "own shield blocks own shot"
    scenario (window must be open for red1 to fire through).

    Layout along world +x:
      x=0              red1 (shotgun), facing +x
      x=shield_distance  red0 (shield), facing +x
      x=enemy_distance   blue1 (shotgun), facing -x
    """
    red1_body  = tank_body_xml("red1",  "0.8 0.3 0.3 1", (0.0, 0.0), 0.0, "shotgun")
    red0_body  = tank_body_xml("red0",  "0.8 0.5 0.3 1", (shield_distance, 0.0), 0.0, "shield")
    blue1_body = tank_body_xml("blue1", "0.3 0.3 0.8 1", (enemy_distance, 0.0), np.pi, "shotgun")
    acts = actuator_xml("red1") + actuator_xml("red0") + actuator_xml("blue1")
    xml = f"""<mujoco model="column_with_enemy">
  {PHYSICS_XML_HEADER}
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {_floor_xml()}
    {red1_body}
    {red0_body}
    {blue1_body}
    {_pellet_pool_xml(1)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def spawn_pellet(model: mujoco.MjModel,
                 data: mujoco.MjData,
                 pellet_body_name: str,
                 pos: np.ndarray,
                 vel: np.ndarray) -> None:
    """Set a pre-allocated pellet body to the given world position and velocity.

    The pellet body must exist in the model XML (pre-allocated pool).
    Call mj_forward() after to update xpos.
    """
    bid = body_id(model, pellet_body_name)
    jid = model.body_jntadr[bid]
    qadr = model.jnt_qposadr[jid]
    vadr = model.jnt_dofadr[jid]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]   # upright quat
    data.qvel[vadr:vadr + 3] = vel
    data.qvel[vadr + 3:vadr + 6] = [0.0, 0.0, 0.0]
