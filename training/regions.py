"""The test report's regions, as Milestone 2 drew them: a ray through each
pixel's center, classified by what it hits first.

Milestone 2's masks (docs/research/2026-10-11-restir-pt-phase1.md, "Equal
frame time") were "by a ray cast through each pixel's center: glass and metal
marbles (209k pixels), porcelain (81k), the table (1.77M)", made by a
diagnostic that was not committed. They are made again here, from the scene
file, on the CPU: the camera's pinhole ray through each pixel's center
(core/contracts/camera.h: d = forward + (2 px / W - 1) right + (1 - 2 py / H)
up, rows from the top, the lens's radius ignored, as a ray through the
center of the lens), against the scene's spheres and axis-aligned boxes;
the first hit's material decides:

  glass and metal  a dielectric (clear, tinted, the cat's-eyes' shells) or a
                   conductor (brass, steel, copper);
  porcelain        a coated material;
  table            the table's material (its top, edges and legs);
  whole            every pixel.

Fireflies move and are not shapes; they are in no region but the whole, as
the scene's static masks were. The counts are checked against Milestone 2's
(tests/test_training.py).
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np

REGIONS = ("whole", "glass_metal", "porcelain", "table")


def camera_basis(camera: dict, width: int, height: int):
    """(origin, forward, right, up) of the pinhole, `right` and `up` scaled as camera.h has them."""
    origin = np.array(camera["position"], dtype=np.float64)
    look_at = np.array(camera["look_at"], dtype=np.float64)
    up_hint = np.array(camera.get("up", [0.0, 1.0, 0.0]), dtype=np.float64)
    forward = look_at - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half_height = math.tan(math.radians(camera["vertical_fov_degrees"]) / 2.0)
    half_width = half_height * width / height
    return origin, forward, right * half_width, up * half_height


def view_axes(scene_file: Path) -> np.ndarray:
    """(3, 3): the camera's unit right, up and backward (toward the viewer) axes, as rows."""
    scene = tomllib.loads(Path(scene_file).read_text())
    _, forward, right, up = camera_basis(scene["camera"], 2, 1)
    return np.stack([right / np.linalg.norm(right), up / np.linalg.norm(up), -forward])


def masks(scene_file: Path, width: int, height: int) -> dict[str, np.ndarray]:
    """Each region's (H, W) bool mask for `scene_file` seen at width x height."""
    scene = tomllib.loads(Path(scene_file).read_text())
    origin, forward, right, up = camera_basis(scene["camera"], width, height)
    px = (np.arange(width) + 0.5)[None, :]
    py = (np.arange(height) + 0.5)[:, None]
    d = (forward[None, None, :] + (2 * px / width - 1)[..., None] * right[None, None, :]
         + (1 - 2 * py / height)[..., None] * up[None, None, :])
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)

    nearest = np.full((height, width), np.inf)
    material = np.full((height, width), "", dtype=object)
    for shape in scene["shapes"]:
        if shape["kind"] == "sphere":
            c = np.array(shape["center"]) - origin
            r = shape["radius"]
            b = d @ c
            disc = b * b - (c @ c - r * r)
            hit = disc >= 0
            root = np.sqrt(np.where(hit, disc, 0.0))
            t0, t1 = b - root, b + root
            t = np.where(t0 > 1e-6, t0, t1)
            hit &= t > 1e-6
        elif shape["kind"] == "box":
            lo = np.array(shape["min"]) - origin
            hi = np.array(shape["max"]) - origin
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / d
                ta, tb = lo * inv, hi * inv
            tmin = np.nanmax(np.minimum(ta, tb), axis=-1)
            tmax = np.nanmin(np.maximum(ta, tb), axis=-1)
            hit = (tmax >= tmin) & (tmax > 1e-6)
            t = np.where(tmin > 1e-6, tmin, tmax)
        else:
            raise ValueError(f"{scene_file}: a shape of kind {shape['kind']!r}, which regions.py does not trace")
        closer = hit & (t < nearest)
        nearest = np.where(closer, t, nearest)
        material = np.where(closer, shape["material"], material)

    kinds = {name: m["kind"] for name, m in scene["materials"].items()}
    kind = np.vectorize(lambda name: kinds.get(name, ""), otypes=[object])(material)
    return {
        "whole": np.ones((height, width), dtype=bool),
        "glass_metal": (kind == "dielectric") | (kind == "conductor"),
        "porcelain": kind == "coated",
        "table": material == "table",
    }
