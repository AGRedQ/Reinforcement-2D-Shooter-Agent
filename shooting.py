import math
from typing import TYPE_CHECKING

from pygame import Vector2

from entities import Agent

if TYPE_CHECKING:
    from environment import Environment


def fire_beam(
    start: Vector2,
    end: Vector2,
    shooter: Agent,
    environment: "Environment",
) -> tuple[Vector2, Vector2] | None:
    """Trigger every intersected body except the shooter, then return the segment.

    The segment is clipped to the world bounds without changing direction.
    Return None when it never enters the world. A zero-length segment tests
    the start point. Drawing the returned segment is the renderer's job.
    """
    if not all(math.isfinite(value) for value in (*start, *end, *environment.bounds)):
        raise ValueError("Beam coordinates and bounds must be finite.")
    if environment.bounds.x <= 0 or environment.bounds.y <= 0:
        raise ValueError("World bounds must be positive.")

    direction = end - start
    enter, exit = 0.0, 1.0
    for origin, component, limit in (
        (start.x, direction.x, environment.bounds.x),
        (start.y, direction.y, environment.bounds.y),
    ):
        if component == 0:
            if not 0 <= origin <= limit:
                return None
            continue
        near, far = sorted((-origin / component, (limit - origin) / component))
        enter = max(enter, near)
        exit = min(exit, far)
        if enter > exit:
            return None

    clipped_start = start + direction * enter
    clipped_end = start + direction * exit
    segment = clipped_end - clipped_start
    length_squared = segment.length_squared()
    hits = []
    for entity in environment.entities:
        if entity is shooter:
            continue
        fraction = 0.0
        if length_squared > 0:
            fraction = max(
                0.0,
                min(1.0, (entity.position - clipped_start).dot(segment) / length_squared),
            )
        closest = clipped_start + segment * fraction
        if entity.position.distance_squared_to(closest) <= entity.radius ** 2:
            hits.append(entity)

    for target in hits:
        environment.on_hit(shooter, target)
    return clipped_start, clipped_end
