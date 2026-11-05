"""
Note to AI Agents: This is a source of truth code file.
If changes are needed for the user's requirements, ask them first.
"""

import math
import pygame
from pygame import Vector2
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Environment


# ============= Classes ============= #

@dataclass(frozen=True)
class Action:
    """Control movement, optional absolute aim, or a discrete turn command."""

    move_x: int
    move_y: int
    aim_x: float = 0.0
    aim_y: float = 0.0
    fire: bool = False
    turn: int | None = None

class Wall:
    def __init__(self, start: Vector2, end: Vector2):
        self.start = start
        self.end = end

class Entity:
    def __init__(self, position: Vector2):
        self.position = position
        self.radius = 10

class Threat(Entity):
    def __init__(self, position: Vector2):
        super().__init__(position)

class Agent(Entity):
    def __init__(self, position: Vector2):
        super().__init__(position)
        self.shoot_cooldown = 0.0
        self.aim_lerp_speed = 10.0
        self.slow_turn_rate = 30.0
        self.fast_turn_rate = 180.0
        self.aim_x = 1.0
        self.aim_y = 0.0
        self.shot_start = position.copy()
        self.shot_end = position.copy()
        self.beam_remaining = 0.0

    def update_aim(self, aim_x: float, aim_y: float, dt: float):
        """Lerp along the shortest arc, with a time-based exponential weight."""
        if not math.isfinite(aim_x) or not math.isfinite(aim_y):
            raise ValueError("Aim components must be finite.")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative.")
        if aim_x == 0 and aim_y == 0:
            return
        target_degree = math.degrees(math.atan2(aim_y, aim_x)) % 360
        current_degree = math.degrees(math.atan2(self.aim_y, self.aim_x)) % 360
        difference = (target_degree - current_degree + 180) % 360 - 180
        weight = -math.expm1(-self.aim_lerp_speed * dt)
        new_degree = (current_degree + difference * weight) % 360
        self.aim_x = math.cos(math.radians(new_degree))
        self.aim_y = math.sin(math.radians(new_degree))

    def update_turn(self, turn: int, dt: float) -> None:
        """Turn at a fixed discrete rate: slow (-1/1) or fast (-2/2)."""
        if turn not in (-2, -1, 0, 1, 2):
            raise ValueError("turn must be -2, -1, 0, 1, or 2.")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative.")
        rate = self.fast_turn_rate if abs(turn) == 2 else self.slow_turn_rate
        heading = math.atan2(self.aim_y, self.aim_x) + math.radians(turn * rate * dt)
        self.aim_x = math.cos(heading)
        self.aim_y = math.sin(heading)


class Player(Agent):
    def __init__(self, position: Vector2):
        super().__init__(position)

    def player_action(self) -> Action:
        """Read testing controls without changing simulation state."""
        keys = pygame.key.get_pressed()
        offset = Vector2(pygame.mouse.get_pos()) - self.position
        aim = Vector2(self.aim_x, self.aim_y)
        if offset.length_squared() > 0:
            aim = offset.normalize()
        return Action(
            move_x=int(keys[pygame.K_d]) - int(keys[pygame.K_a]),
            move_y=int(keys[pygame.K_s]) - int(keys[pygame.K_w]),
            aim_x=aim.x,
            aim_y=aim.y,
            fire=bool(pygame.mouse.get_pressed()[0]),
        )

# ============= Support Defs ============= #



def turning_logic(position: Vector2, aim_x: float, aim_y: float, point: Vector2) -> Vector2:
    angle = math.degrees(math.atan2(aim_y, aim_x)) % 360
    return position + point.rotate(angle)


def move_with_walls(
    position: Vector2, displacement: Vector2, radius: float, walls: list[Wall]
) -> Vector2:
    """Sweep a circular body against segments, sliding along contact normals."""
    position = position.copy()
    remaining = displacement.copy()
    for _ in range(8):
        if remaining.length_squared() < 1e-18:
            break
        contacts: list[tuple[float, Vector2]] = []
        for wall in walls:
            edge = wall.end - wall.start
            length = edge.length()
            if length > 0:
                tangent = edge / length
                normal = Vector2(-tangent.y, tangent.x)
                distance = (position - wall.start).dot(normal)
                velocity = remaining.dot(normal)
                if abs(velocity) > 1e-12:
                    for sign in (-1, 1):
                        time = (sign * radius - distance) / velocity
                        contact_normal = normal * sign
                        along = (position + remaining * time - wall.start).dot(tangent)
                        if (
                            -1e-10 <= time <= 1
                            and 0 <= along <= length
                            and remaining.dot(contact_normal) < -1e-12
                        ):
                            contacts.append((max(0.0, time), contact_normal))
            for endpoint in (wall.start, wall.end):
                offset = position - endpoint
                a = remaining.length_squared()
                b = offset.dot(remaining)
                c = offset.length_squared() - radius * radius
                discriminant = b * b - a * c
                if discriminant < 0:
                    continue
                time = (-b - math.sqrt(discriminant)) / a
                if -1e-10 <= time <= 1:
                    normal = position + remaining * time - endpoint
                    if normal.length_squared() > 0:
                        normal.normalize_ip()
                        if remaining.dot(normal) < -1e-12:
                            contacts.append((max(0.0, time), normal))
        if not contacts:
            position += remaining
            break
        time = min(contact[0] for contact in contacts)
        position += remaining * time
        remaining *= 1 - time
        normals = [normal for t, normal in contacts if abs(t - time) < 1e-9]
        # Two nonparallel contacts form a corner; do not project through either wall.
        if any(abs(normals[0].cross(normal)) > 1e-8 for normal in normals[1:]):
            break
        for normal in normals:
            inward = remaining.dot(normal)
            if inward < 0:
                remaining -= normal * inward
    return position


def wall_intersection_fraction(start: Vector2, end: Vector2, wall: Wall) -> float | None:
    """Return the first intersection along a beam, including collinear overlap."""
    direction = end - start
    edge = wall.end - wall.start
    offset = wall.start - start
    denominator = direction.cross(edge)
    if denominator != 0:
        time = offset.cross(edge) / denominator
        along = offset.cross(direction) / denominator
        return time if 0 <= time <= 1 and 0 <= along <= 1 else None
    length_squared = direction.length_squared()
    if length_squared == 0:
        if edge.length_squared() == 0:
            return 0.0 if start == wall.start else None
        along = (start - wall.start).dot(edge) / edge.length_squared()
        return 0.0 if offset.cross(edge) == 0 and 0 <= along <= 1 else None
    if offset.cross(direction) != 0:
        return None
    near, far = sorted((
        offset.dot(direction) / length_squared,
        (wall.end - start).dot(direction) / length_squared,
    ))
    return max(0.0, near) if max(0.0, near) <= min(1.0, far) else None


def is_in_field_of_view(
    origin: Vector2,
    direction: Vector2,
    target: Vector2,
    half_angle_degrees: float,
) -> bool:
    """Return whether a point is in an unlimited-range cone around direction."""
    if not math.isfinite(half_angle_degrees) or not 0 <= half_angle_degrees <= 180:
        raise ValueError("Field-of-view half-angle must be finite and in [0, 180].")
    offset = target - origin
    if offset.length_squared() == 0:
        return True
    if direction.length_squared() == 0:
        return False
    threshold = math.cos(math.radians(half_angle_degrees))
    magnitude = direction.length() * offset.length()
    return direction.dot(offset) + 1e-12 * magnitude >= threshold * magnitude


def has_line_of_sight(start: Vector2, end: Vector2, walls: list[Wall]) -> bool:
    """Return whether the segment between two points intersects no wall."""
    return all(
        wall_intersection_fraction(start, end, wall) is None
        for wall in walls
    )


def fire_beam(
    start: Vector2,
    end: Vector2,
    shooter: Agent,
    environment: "Environment",
) -> tuple[Vector2, Vector2] | None:
    """Trigger every intersected body except the shooter, then return the segment.

    The segment is clipped to the world bounds and nearest wall.
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
    wall_times = [
        time for wall in environment.walls
        if (time := wall_intersection_fraction(clipped_start, clipped_end, wall)) is not None
    ]
    if wall_times:
        clipped_end = clipped_start + (clipped_end - clipped_start) * min(wall_times)
    segment = clipped_end - clipped_start
    length_squared = segment.length_squared()
    hits = []
    for entity in environment.entities:
        if entity is shooter:
            continue
        if wall_times and (entity.position - clipped_end).dot(direction) >= 0:
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
