"""Dense display geometry, independent of the policy's discrete sensor rays."""

from __future__ import annotations

import math

import numpy as np

from entities import RAY_LENGTH, ray_aabb_distances


def visibility_polygon(
    origin: np.ndarray,
    boxes: np.ndarray,
    *,
    angle: float = 0.0,
    spread: float = math.tau,
    radius: float = RAY_LENGTH,
) -> np.ndarray:
    """Return ordered wall-clipped arc points (world coordinates).

    Uniform angular samples keep range arcs smooth. Rays on both sides of every
    wall corner preserve sharp occlusion boundaries without cutting across thin
    obstacles. For a cone, prepend the origin when filling the polygon.
    """
    half = spread / 2.0
    offsets = np.linspace(-half, half, max(2, math.ceil(spread / math.radians(0.75)) + 1))
    if spread < math.tau:
        # Subpixel angular bands let the renderer feather the cone's sides.
        edge_width = min(half, math.radians(2.5))
        edge = np.linspace(0, edge_width, 33)
        offsets = np.concatenate((offsets, -half + edge, half - edge))
    if len(boxes):
        corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
        delta = corners - origin
        corner_angles = np.arctan2(delta[:, 1], delta[:, 0])
        relative = (corner_angles - angle + math.pi) % math.tau - math.pi
        around_corners = (relative[:, None] + np.array([-1e-5, 0.0, 1e-5])).ravel()
        offsets = np.concatenate((offsets, around_corners[(around_corners >= -half) & (around_corners <= half)]))
    angles = np.unique(offsets) + angle
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    distances = np.minimum(ray_aabb_distances(origin, directions, boxes), radius)
    return origin + directions * distances[:, None]
