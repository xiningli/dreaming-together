"""Segmentation observation rendering — the only visual channel agents get.

Design (DESIGN_OF_EXPERIMENT.md §3.3–3.4):
  - 1 channel, int8 class ids, 64×64.
  - Classes: 0 background, 1 red hull, 2 blue hull, 3 shield, 4 pellet, 5 arm.
  - Cameras are hull-fixed: {prefix}_front_cam (both roles),
    {prefix}_rear_cam (shield role only). They never rotate with the arm.

Renderer usage note: EGL contexts are thread-local. Create the Renderer in
the thread that uses it, and set model.vis.global_.offwidth/offheight before
construction if rendering wider than 640 px (not needed at 64×64).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from dreaming_together.envs.tank import CAM_RES

SEG_BACKGROUND = 0
SEG_RED_HULL   = 1
SEG_BLUE_HULL  = 2
SEG_SHIELD     = 3
SEG_PELLET     = 4
SEG_ARM        = 5

N_SEG_CLASSES  = 6


def geom_class_map(model: mujoco.MjModel) -> np.ndarray:
    """Return (ngeom,) int8 array mapping geom id → segmentation class."""
    classes = np.zeros(model.ngeom, dtype=np.int8)
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.endswith("_shield_g"):
            classes[gid] = SEG_SHIELD
        elif name.endswith("_hull_g"):
            classes[gid] = SEG_RED_HULL if name.startswith("red") else SEG_BLUE_HULL
        elif name.startswith("pellet_"):
            classes[gid] = SEG_PELLET
        elif name.endswith("_arm_g") or name.endswith("_barrel_g"):
            classes[gid] = SEG_ARM
        # everything else (floor, walls, targets) stays background
    return classes


class SegCamera:
    """Renders 1-channel segmentation class maps from a named model camera."""

    def __init__(self, model: mujoco.MjModel, res: int = CAM_RES):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=res, width=res)
        self.renderer.enable_segmentation_rendering()
        self._classes = geom_class_map(model)

    def render(self, data: mujoco.MjData, cam_name: str) -> np.ndarray:
        """Return (res, res) int8 class-id image from the named camera."""
        self.renderer.update_scene(data, camera=cam_name)
        seg = self.renderer.render()          # (H, W, 2): [objid, objtype]
        objid, objtype = seg[..., 0], seg[..., 1]
        out = np.zeros(objid.shape, dtype=np.int8)
        geom_mask = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
        valid = geom_mask & (objid >= 0) & (objid < self.model.ngeom)
        out[valid] = self._classes[objid[valid].astype(int)]
        return out

    def close(self) -> None:
        self.renderer.close()
