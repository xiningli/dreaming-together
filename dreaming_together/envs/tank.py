"""Tank agent: MJCF builder, geometric constants, forward kinematics helpers.

This is NOT AI code. It is the physics substrate — the MuJoCo model that the
policies will eventually control.

v2 changes vs the vla-simplified repo (see DESIGN_OF_EXPERIMENT.md §3):
  - Arm is 2-DOF (pan + tilt). The elbow joint is removed; the old
    upper-arm + forearm chain is a single ARM_LEN segment.
  - Hull-fixed observation cameras: {prefix}_front_cam on both roles,
    {prefix}_rear_cam on the shield role only. Neither rotates with the arm.
  - set_arm_target() added: PD-target-only variant of set_arm_ctrl for
    tests that measure arm settling time through real physics.

Agent roles
-----------
shield  (prefix red0 / blue0) — 2-DOF arm carries a 1.2 m × 1.0 m shield box.
shotgun (prefix red1 / blue1) — 2-DOF arm carries a muzzle site; fires pellets.

Hull geometry (both roles identical)
-------------------------------------
Box: 0.60 m long × 0.40 m wide × 0.25 m tall.
Hull centre sits at z = HULL_Z when on the ground plane.
Arm mount: top-front of hull, local position ARM_MOUNT_LOCAL.

Arm kinematic chain
--------------------
arm_pan   — hinge around hull-vertical (Z), range ±60°.
arm_tilt  — hinge around lateral (Y) after pan, range −30° to +70°.
Arm segment: ARM_LEN. End-effector (shield face / muzzle site) at arm tip,
END_EFFECTOR_LEN beyond.

Locomotion
----------
Differential drive via two named motor actuators per tank:
  {prefix}_left_track  — normalised ±1 → body-frame force + yaw torque
  {prefix}_right_track — normalised ±1 → body-frame force + yaw torque

Physical velocity mapping (approximate — depends on friction/mass):
  v_forward ≈ (left + right) / 2 * MAX_TRACK_SPEED
  omega_yaw  ≈ (right - left) / TRACK_WIDTH

Motor gear encodes track position so that equal-and-opposite inputs produce
pure rotation, equal same-sign inputs produce forward/backward translation.

Arm actuators (PD position control via MuJoCo `position` actuator)
--------------------------------------------------------------------
  act_{prefix}_arm_pan   — pan joint target (rad)
  act_{prefix}_arm_tilt  — tilt joint target (rad)

Cameras
-------
  {prefix}_cam        — chase camera behind the tank, for video rendering only.
  {prefix}_front_cam  — hull front face, looks along body +x, fovy 90°.
  {prefix}_rear_cam   — shield role only; hull rear face, looks along body -x.
Observation cameras are rendered at 64×64 segmentation (1 channel).

Arm mass and gravity
--------------------
All arm geoms have explicit low mass so that kp=150 Nm/rad holds the arm at
any target angle without significant gravity droop. Appropriate for unit
tests; production tuning should re-derive kp from the end-effector load.
"""
from __future__ import annotations

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Geometric constants (authoritative — do not change without updating tests)
# ---------------------------------------------------------------------------

HULL_LENGTH       = 0.60    # m, full length along hull +x axis
HULL_WIDTH        = 0.40    # m, full width along hull ±y axis
HULL_HEIGHT       = 0.25    # m, total vertical extent
HULL_Z            = HULL_HEIGHT / 2          # 0.125 m — hull centre above floor

TRACK_WIDTH       = 0.36    # m, lateral separation between left and right tracks
MAX_TRACK_SPEED   = 1.5     # m/s, normalised ±1 maps to ±MAX_TRACK_SPEED

# Arm mount in hull-local frame (origin at hull centre)
ARM_MOUNT_LOCAL   = np.array([0.10, 0.0, HULL_HEIGHT / 2])   # top-front of hull

ARM_LEN           = 0.50    # m — single 2-DOF arm segment (was upper 0.25 + forearm 0.25)
END_EFFECTOR_LEN  = 0.15    # m — from arm tip to shield face / muzzle

# Arm joint ranges (radians)
ARM_PAN_RANGE     = (-1.047, 1.047)    # ±60°
ARM_TILT_RANGE    = (-0.524, 1.222)    # −30° to +70°

# Shield geometry (shield role only)
SHIELD_WIDTH      = 1.20    # m (y-extent of shield box)
SHIELD_HEIGHT     = 1.00    # m (z-extent of shield box)
# 0.25 m gives ~4 contact steps at 30 m/s / dt=0.002 s (step=0.06 m),
# enough for the soft constraint to decelerate the pellet to rest before exit.
SHIELD_THICKNESS  = 0.25    # m

# Projectile parameters
PELLET_SPEED      = 30.0    # m/s
N_PELLETS         = 8
CONE_HALF_ANGLE   = 8.0     # degrees half-angle of scatter cone

# Arm joint group name (used by actuator builders)
ARM_JOINTS = [
    ("arm_pan",  "arm"),
    ("arm_tilt", "arm"),
]

# PD gains for arm joints
ARM_KP = 150.0
ARM_KD = 15.0

# Track velocity-servo gain (N per m/s of track-speed error)
KV_TRACK = 5000.0

# Observation camera geometry (hull-local)
FRONT_CAM_LOCAL   = np.array([HULL_LENGTH / 2, 0.0, 0.225])  # world z ≈ 0.35 m
REAR_CAM_LOCAL    = np.array([-HULL_LENGTH / 2, 0.0, 0.225])
CAM_FOVY          = 90.0    # degrees; square render → 90° HFOV too
CAM_RES           = 64     # observation render resolution (square)

# Drive force per track at normalised control = 1.0
# Must exceed static friction torque ≈ 170 Nm for pivot turn (μ=0.8, N=589N, r=0.36m, 4 contacts).
# At 1500 N: applied torque = 1500*0.18 = 270 Nm >> 170 Nm; forward force = 2*1500*0.5 = 1500N.
_TRACK_FORCE_MAX = 1500.0   # N per track


# ---------------------------------------------------------------------------
# MJCF builders
# ---------------------------------------------------------------------------

def tank_body_xml(prefix: str, rgba: str, pos: tuple, yaw: float,
                  role: str) -> str:
    """Return the MJCF XML string for one tank agent.

    prefix : "red0", "red1", "blue0", or "blue1"
    rgba   : hull colour string, e.g. "0.8 0.3 0.3 1"
    pos    : (x, y) world position; hull spawns at z = HULL_Z
    yaw    : heading in radians (0 = facing world +x)
    role   : "shield" or "shotgun"
    """
    x, y = pos

    hw = HULL_LENGTH / 2    # 0.300
    hd = HULL_WIDTH  / 2    # 0.200
    hh = HULL_HEIGHT / 2    # 0.125

    mx, my, mz = ARM_MOUNT_LOCAL   # 0.100, 0.000, 0.125

    lo_pan,  hi_pan  = ARM_PAN_RANGE
    lo_tilt, hi_tilt = ARM_TILT_RANGE

    if role == "shield":
        # Team-tinted, and OPAQUE: the segmentation renderer skips
        # transparent geoms entirely (alpha < 1 made the shield invisible
        # in agent observations — G1 gate catch).
        shield_rgba = ("0.75 0.35 0.30 1" if prefix.startswith("red")
                       else "0.30 0.40 0.90 1")
        ee_xml = f"""
          <geom name="{prefix}_shield_g" type="box"
                size="{SHIELD_THICKNESS/2:.4f} {SHIELD_WIDTH/2:.4f} {SHIELD_HEIGHT/2:.4f}"
                pos="{END_EFFECTOR_LEN:.4f} 0 {SHIELD_HEIGHT/2 - mz:.4f}"
                mass="1.0"
                rgba="{shield_rgba}"
                contype="1" conaffinity="7"
                solimp="0.9 0.99 0.001" solref="0.002 1"/>"""
        rear_cam_xml = (
            f'\n  <camera name="{prefix}_rear_cam" '
            f'pos="{REAR_CAM_LOCAL[0]:.4f} 0 {REAR_CAM_LOCAL[2]:.4f}" '
            f'xyaxes="0 1 0 0 0 1" fovy="{CAM_FOVY:.0f}"/>'
        )
    else:   # shotgun
        ee_xml = f"""
          <geom name="{prefix}_barrel_g" type="cylinder"
                fromto="0 0 0 {END_EFFECTOR_LEN:.4f} 0 0" size="0.015"
                mass="0.10"
                rgba="0.30 0.30 0.30 1"
                contype="0" conaffinity="0"/>
          <site name="{prefix}_muzzle" pos="{END_EFFECTOR_LEN:.4f} 0 0"
                size="0.01" rgba="1 0.5 0 1"/>"""
        rear_cam_xml = ""

    return f"""<body name="{prefix}_hull" pos="{x:.4f} {y:.4f} {HULL_Z:.4f}" euler="0 0 {yaw:.6f}">
  <freejoint name="{prefix}_root"/>
  <geom name="{prefix}_hull_g" type="box"
        size="{hw:.4f} {hd:.4f} {hh:.4f}"
        rgba="{rgba}" contype="1" conaffinity="1" class="tank"/>
  <site name="{prefix}_thrust" pos="0 0 0" size="0.01" rgba="0 0 0 0"/>
  <camera name="{prefix}_cam" pos="-1.5 0 0.8" euler="0 20 180" fovy="70"/>
  <camera name="{prefix}_front_cam" pos="{FRONT_CAM_LOCAL[0]:.4f} 0 {FRONT_CAM_LOCAL[2]:.4f}" xyaxes="0 -1 0 0 0 1" fovy="{CAM_FOVY:.0f}"/>{rear_cam_xml}
  <body name="{prefix}_arm_pan_lnk" pos="{mx:.4f} {my:.4f} {mz:.4f}">
    <inertial mass="0.01" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>
    <joint name="{prefix}_arm_pan" type="hinge" axis="0 0 1"
           range="{lo_pan:.4f} {hi_pan:.4f}" damping="{ARM_KD}" armature="0.05"/>
    <body name="{prefix}_arm_tilt_lnk">
      <joint name="{prefix}_arm_tilt" type="hinge" axis="0 -1 0"
             range="{lo_tilt:.4f} {hi_tilt:.4f}" damping="{ARM_KD}" armature="0.05"/>
      <geom name="{prefix}_arm_g" type="capsule"
            fromto="0 0 0 {ARM_LEN:.4f} 0 0" size="0.025"
            mass="0.25" rgba="0.55 0.55 0.55 1" contype="0" conaffinity="0"/>
      <body name="{prefix}_end_eff" pos="{ARM_LEN:.4f} 0 0">
        {ee_xml}
      </body>
    </body>
  </body>
</body>"""


def actuator_xml(prefix: str) -> str:
    """Return MJCF <actuator> children for one tank agent.

    Produces:
      - Two track velocity servos: {prefix}_left_track, {prefix}_right_track.
        SITE transmission, not joint: a freejoint's translational dof axes
        are WORLD-aligned, so a joint-transmission gear always pushes along
        world +x regardless of heading (found by the e2e visual test: every
        tank piled onto the east wall). The site frame rotates with the
        hull, and gear (1, 0, 0, 0, 0, ±r) projects the site's 6D velocity
        onto exactly the left/right track surface speed (v_x ∓ r·ω_z), so a
        <velocity> actuator servos true track speed in the hull frame.
        ctrl is the commanded track speed in m/s (±MAX_TRACK_SPEED);
        forcerange caps thrust at the motor's physical limit.
      - Two arm position actuators: act_{prefix}_arm_pan / act_{prefix}_arm_tilt
    """
    F   = _TRACK_FORCE_MAX
    r   = TRACK_WIDTH / 2          # 0.18 m
    v   = MAX_TRACK_SPEED

    lo_pan,  hi_pan  = ARM_PAN_RANGE
    lo_tilt, hi_tilt = ARM_TILT_RANGE

    return f"""
    <velocity name="{prefix}_left_track"  site="{prefix}_thrust"
              gear="1 0 0 0 0 {-r:.4f}" kv="{KV_TRACK}"
              ctrlrange="{-v} {v}" forcerange="{-F} {F}"/>
    <velocity name="{prefix}_right_track" site="{prefix}_thrust"
              gear="1 0 0 0 0 {r:.4f}" kv="{KV_TRACK}"
              ctrlrange="{-v} {v}" forcerange="{-F} {F}"/>
    <position name="act_{prefix}_arm_pan"   joint="{prefix}_arm_pan"
              kp="{ARM_KP}" ctrlrange="{lo_pan:.4f} {hi_pan:.4f}"/>
    <position name="act_{prefix}_arm_tilt"  joint="{prefix}_arm_tilt"
              kp="{ARM_KP}" ctrlrange="{lo_tilt:.4f} {hi_tilt:.4f}"/>"""


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def muzzle_pos(model: mujoco.MjModel, data: mujoco.MjData,
               prefix: str) -> np.ndarray:
    """Return world-frame position of the muzzle site for a shotgun tank."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_muzzle")
    return data.site_xpos[sid].copy()


def muzzle_dir(model: mujoco.MjModel, data: mujoco.MjData,
               prefix: str) -> np.ndarray:
    """Return unit direction vector the muzzle points in world frame.

    The muzzle site's local +x axis (first column of site_xmat) is the
    direction the barrel points.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_muzzle")
    mat = data.site_xmat[sid].reshape(3, 3)
    return mat[:, 0].copy()   # world-frame muzzle direction (local +x)


def shield_face_normal(model: mujoco.MjModel, data: mujoco.MjData,
                       prefix: str) -> np.ndarray:
    """Return the outward normal of the shield face in world frame.

    The shield geom's local +x axis points outward from the face.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_shield_g")
    bid = model.geom_bodyid[gid]
    mat = data.xmat[bid].reshape(3, 3)
    return mat[:, 0].copy()


# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

def set_track_ctrl(model: mujoco.MjModel, data: mujoco.MjData,
                   prefix: str, left: float, right: float) -> None:
    """Set left and right track speed targets. Values in [-1, 1] (normalised).

    Normalised inputs map to ±MAX_TRACK_SPEED m/s commanded track speed;
    the velocity servos produce the force (capped at the motor limit).
    """
    left_aid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   f"{prefix}_left_track")
    right_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                   f"{prefix}_right_track")
    data.ctrl[left_aid]  = float(np.clip(left,  -1.0, 1.0)) * MAX_TRACK_SPEED
    data.ctrl[right_aid] = float(np.clip(right, -1.0, 1.0)) * MAX_TRACK_SPEED


def set_arm_ctrl(model: mujoco.MjModel, data: mujoco.MjData,
                 prefix: str,
                 pan_rad: float, tilt_rad: float) -> None:
    """Teleport arm joints to target angles and set PD ctrl targets.

    Sets both qpos (immediate position) and ctrl (PD target) so that
    calling mj_forward after set_arm_ctrl immediately reflects the new
    arm pose in data.site_xpos. This is the correct semantics for unit
    tests that call set_arm_ctrl → mj_forward → muzzle_pos.
    """
    targets = [
        ("arm_pan",  pan_rad),
        ("arm_tilt", tilt_rad),
    ]
    for jname, target_rad in targets:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"{prefix}_{jname}")
        if jid >= 0:
            lo, hi = model.jnt_range[jid]
            clamped = float(np.clip(target_rad, lo, hi))
            data.qpos[model.jnt_qposadr[jid]] = clamped
            data.qvel[model.jnt_dofadr[jid]]  = 0.0

        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                 f"act_{prefix}_{jname}")
        if aid >= 0:
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(target_rad, lo, hi))


def set_arm_target(model: mujoco.MjModel, data: mujoco.MjData,
                   prefix: str,
                   pan_rad: float, tilt_rad: float) -> None:
    """Set PD ctrl targets only — the arm must physically move there.

    Unlike set_arm_ctrl this does NOT teleport qpos. Use for tests that
    measure real arm settling time (window open/close timing) and for
    policies, which command targets rather than states.
    """
    for jname, target_rad in [("arm_pan", pan_rad), ("arm_tilt", tilt_rad)]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                 f"act_{prefix}_{jname}")
        if aid >= 0:
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(target_rad, lo, hi))


def arm_angles(model: mujoco.MjModel, data: mujoco.MjData,
               prefix: str) -> tuple[float, float]:
    """Return current (pan, tilt) joint angles in radians."""
    out = []
    for jname in ("arm_pan", "arm_tilt"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"{prefix}_{jname}")
        out.append(float(data.qpos[model.jnt_qposadr[jid]]))
    return out[0], out[1]


def hull_pos(model: mujoco.MjModel, data: mujoco.MjData,
             prefix: str) -> np.ndarray:
    """Return world-frame (x, y, z) of the hull centre body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_hull")
    return data.xpos[bid].copy()


def hull_yaw(model: mujoco.MjModel, data: mujoco.MjData,
             prefix: str) -> float:
    """Return yaw angle (radians, world frame) of the hull.

    Extracted from the hull body's rotation matrix:
      yaw = atan2(R[1,0], R[0,0]) = atan2(sin ψ, cos ψ)
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_hull")
    mat = data.xmat[bid].reshape(3, 3)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


# ---------------------------------------------------------------------------
# Window / coverage ray computation
# ---------------------------------------------------------------------------

def _shooter_pose(model: mujoco.MjModel, data: mujoco.MjData,
                  prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, local-x direction) for a world site or tank muzzle site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix)
    if sid < 0:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_muzzle")
    if sid < 0:
        raise ValueError(f"No site found for shooter/attacker prefix '{prefix}'")
    origin = data.site_xpos[sid].copy()
    direction = data.site_xmat[sid].reshape(3, 3)[:, 0].copy()
    return origin, direction


def _cone_rays(base_dir: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Return (n, 3) unit vectors uniformly distributed within CONE_HALF_ANGLE of base_dir."""
    rng = np.random.default_rng(seed)
    half = np.radians(CONE_HALF_ANGLE)
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(np.cos(half), 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    local = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    d = base_dir / np.linalg.norm(base_dir)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        rot = np.eye(3) if d[2] > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(z, d), -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return (rot @ local.T).T


def window_quality(model: mujoco.MjModel, data: mujoco.MjData,
                   shooter_prefix: str, shield_prefix: str,
                   n_rays: int = 16) -> float:
    """W_q ∈ [0,1]: fraction of muzzle cone rays not blocked by own shield.

    Cast n_rays within CONE_HALF_ANGLE from the shooter's muzzle toward the
    shield team's region.  Counts rays NOT hitting the shield geom.
    shooter_prefix may be a site name (e.g. "shooter_muzzle") or a tank prefix.
    Rays that hit the floor are also treated as blocked (no window below floor).
    """
    origin, base_dir = _shooter_pose(model, data, shooter_prefix)
    gid_shield = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{shield_prefix}_shield_g")
    gid_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    directions = _cone_rays(base_dir, n_rays, seed=0)
    geomid = np.array([-1], dtype=np.int32)
    free_rays = 0
    for d in directions:
        mujoco.mj_ray(model, data, origin, d, None, 1, -1, geomid)
        gid = int(geomid[0])
        if gid != gid_shield and gid != gid_floor:
            free_rays += 1
    return free_rays / n_rays


def residual_coverage(model: mujoco.MjModel, data: mujoco.MjData,
                      attacker_prefix: str, shield_prefix: str,
                      n_rays: int = 16) -> float:
    """C_r ∈ [0,1]: fraction of rays from attacker blocked by the shield."""
    origin, base_dir = _shooter_pose(model, data, attacker_prefix)
    gid_shield = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{shield_prefix}_shield_g")

    directions = _cone_rays(base_dir, n_rays, seed=0)
    geomid = np.array([-1], dtype=np.int32)
    shield_hits = 0
    for d in directions:
        mujoco.mj_ray(model, data, origin, d, None, 1, -1, geomid)
        if int(geomid[0]) == gid_shield:
            shield_hits += 1
    return shield_hits / n_rays
