"""Game entities: walls, bullets and agents, plus the geometry they rely on.

Coordinates use screen space (x right, y down). An angle ``a`` points along
``(cos a, sin a)``, so increasing the angle turns the agent clockwise on screen
(i.e. to its right).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

NUM_RAYS = 32
VISION_CONE_DEG = 60.0
RAY_LENGTH = 600.0

RAY_OFFSETS = (np.arange(NUM_RAYS) * 2.0 * np.pi / NUM_RAYS + np.pi) % (2.0 * np.pi) - np.pi
"""Ray angles relative to the aim direction, in [-pi, pi). Index 0 points forward."""

CONE_RAY_INDICES = np.flatnonzero(np.abs(RAY_OFFSETS) <= math.radians(VISION_CONE_DEG) / 2.0 + 1e-9)
"""Rays that additionally detect enemies (they overlap the regular wall rays)."""

NUM_CONE_RAYS = len(CONE_RAY_INDICES)
BULLET_FEATURE_SIZE = 4
ENEMY_REGION_EDGES = np.radians([-30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0])
NUM_ENEMY_REGIONS = len(ENEMY_REGION_EDGES) - 1

_DIAG = math.sqrt(0.5)
MOVE_DIRECTIONS = np.array(
    [
        (0.0, 0.0),  # 0 stand still
        (1.0, 0.0),  # 1 forward
        (_DIAG, _DIAG),  # 2 forward-right
        (0.0, 1.0),  # 3 right
        (-_DIAG, _DIAG),  # 4 backward-right
        (-1.0, 0.0),  # 5 backward
        (-_DIAG, -_DIAG),  # 6 backward-left
        (0.0, -1.0),  # 7 left
        (_DIAG, -_DIAG),  # 8 forward-left
    ]
)
"""Movement actions as (forward, right) components in the agent's own frame."""

NUM_MOVE_ACTIONS = len(MOVE_DIRECTIONS)
NUM_AIM_ACTIONS = 3  # 0 turn left, 1 hold, 2 turn right
NUM_FIRE_ACTIONS = 2  # 0 hold fire, 1 fire
ACTION_DIMS = (NUM_MOVE_ACTIONS, NUM_AIM_ACTIONS, NUM_FIRE_ACTIONS)
_ACTION_ONE_HOT_OFFSETS = np.cumsum((0,) + ACTION_DIMS[:-1])

OBSERVATION_SIZE = NUM_RAYS + 2 * NUM_CONE_RAYS + 2 + 1 + 1 + sum(ACTION_DIMS)
"""Wall rays, enemy (flag, distance) per cone ray, local velocity, cooldown, health, previous action."""
ANGULAR_OBSERVATION_SIZE = OBSERVATION_SIZE - 2 * NUM_CONE_RAYS + 2 * NUM_ENEMY_REGIONS + 2
"""Six region flags/distances plus a separate center-ray flag/distance."""


# --------------------------------------------------------------------------- geometry


def ray_aabb_distances(origin: np.ndarray, directions: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Distance along each unit ray to the nearest box, ``inf`` if none is hit.

    ``directions`` has shape (N, 2) and ``boxes`` has shape (M, 4) as
    ``(xmin, ymin, xmax, ymax)``. Rays starting inside a box return 0.
    """
    if len(boxes) == 0:
        return np.full(len(directions), np.inf)
    d = np.where(np.abs(directions) < 1e-12, 1e-12, directions)
    inv = 1.0 / d[:, None, :]  # (N, 1, 2)
    t_lo = (boxes[None, :, :2] - origin) * inv  # (N, M, 2)
    t_hi = (boxes[None, :, 2:] - origin) * inv
    t_enter = np.minimum(t_lo, t_hi).max(axis=2)
    t_exit = np.maximum(t_lo, t_hi).min(axis=2)
    hit = (t_enter <= t_exit) & (t_exit >= 0.0)
    t = np.where(hit, np.maximum(t_enter, 0.0), np.inf)
    return t.min(axis=1)


def ray_circle_distances(origin: np.ndarray, directions: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Distance along each unit ray to a circle, ``inf`` if missed. Rays starting inside return 0."""
    oc = origin - center
    b = directions @ oc
    c = float(oc @ oc) - radius * radius
    if c <= 0.0:
        return np.zeros(len(directions))
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0.0) & (t >= 0.0), t, np.inf)


def circle_overlaps_boxes(center: np.ndarray, radius: float, boxes: np.ndarray) -> bool:
    closest = np.clip(center, boxes[:, :2], boxes[:, 2:])
    return bool((np.sum((closest - center) ** 2, axis=1) < radius * radius).any())


# --------------------------------------------------------------------------- entities


@dataclass(frozen=True)
class Wall:
    """Axis-aligned rectangular wall; blocks movement, bullets and all rays."""

    x: float
    y: float
    width: float
    height: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def walls_to_array(walls: list[Wall]) -> np.ndarray:
    return np.array([w.bounds for w in walls], dtype=np.float64).reshape(-1, 4)


def resolve_wall_collisions(
    position: np.ndarray, velocity: np.ndarray, radius: float, boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Push a circle out of overlapping walls and remove the velocity component into them."""
    closest_all = np.clip(position, boxes[:, :2], boxes[:, 2:])
    touching = np.sum((closest_all - position) ** 2, axis=1) < radius * radius
    for xmin, ymin, xmax, ymax in boxes[touching]:
        closest = np.array([min(max(position[0], xmin), xmax), min(max(position[1], ymin), ymax)])
        delta = position - closest
        dist = float(np.hypot(*delta))
        if dist >= radius:
            continue
        if dist > 1e-9:
            normal = delta / dist
            position = closest + normal * radius
        else:  # center inside the box: push out along the shallowest side
            pushes = [
                (position[0] - xmin, np.array([-1.0, 0.0])),
                (xmax - position[0], np.array([1.0, 0.0])),
                (position[1] - ymin, np.array([0.0, -1.0])),
                (ymax - position[1], np.array([0.0, 1.0])),
            ]
            depth, normal = min(pushes, key=lambda p: p[0])
            position = position + normal * (depth + radius)
        into_wall = float(velocity @ normal)
        if into_wall < 0.0:
            velocity = velocity - normal * into_wall
    return position, velocity


@dataclass
class Bullet:
    position: np.ndarray
    velocity: np.ndarray
    damage: float
    max_range: float
    traveled: float = 0.0
    alive: bool = True

    def update(self, dt: float, boxes: np.ndarray, targets: list[Threat | Agent]) -> Threat | Agent | None:
        """Advance with swept collision. Returns the target hit, if any."""
        speed = float(np.linalg.norm(self.velocity))
        step = min(speed * dt, max(0.0, self.max_range - self.traveled))
        direction = (self.velocity / speed)[None, :]
        wall_t = float(ray_aabb_distances(self.position, direction, boxes)[0])

        target, target_t = None, math.inf
        for candidate in targets:
            if not candidate.alive:
                continue
            t = float(ray_circle_distances(self.position, direction, candidate.position, candidate.radius)[0])
            if t < target_t:
                target, target_t = candidate, t

        if target is not None and target_t <= step and target_t < wall_t:
            self.position = self.position + direction[0] * target_t
            self.alive = False
            return target
        if wall_t <= step:
            self.position = self.position + direction[0] * wall_t
            self.alive = False
            return None

        self.position = self.position + direction[0] * step
        self.traveled += step
        if self.traveled >= self.max_range:
            self.alive = False
        return None


@dataclass
class Agent:
    agent_id: int = 0
    angular_regions: bool = False
    color: tuple[int, int, int] = (80, 160, 255)
    radius: float = 14.0
    max_speed: float = 220.0
    acceleration: float = 12.0
    """Velocity smoothing rate (1/s) towards the commanded velocity."""
    turn_rate: float = math.radians(200.0)
    max_health: float = 100.0
    fire_cooldown: float = 0.35
    bullet_speed: float = 900.0
    bullet_damage: float = 25.0
    bullet_range: float = 900.0

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    health: float = 0.0
    cooldown: float = 0.0
    last_action: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.int64))

    # Latest sensor readings, in world units (kept for observations and rendering).
    wall_distances: np.ndarray = field(default_factory=lambda: np.full(NUM_RAYS, RAY_LENGTH))
    enemy_detected: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CONE_RAYS, dtype=bool))
    enemy_distances: np.ndarray = field(default_factory=lambda: np.full(NUM_CONE_RAYS, RAY_LENGTH))
    center_detected: bool = False
    center_distance: float = RAY_LENGTH
    region_points: np.ndarray = field(default_factory=lambda: np.zeros((NUM_ENEMY_REGIONS, 2)))

    def reset(self, position: np.ndarray, angle: float) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros(2)
        self.angle = angle
        self.health = self.max_health
        self.cooldown = 0.0
        self.last_action = np.array([0, 1, 0], dtype=np.int64)

    @property
    def alive(self) -> bool:
        return self.health > 0.0

    @property
    def forward(self) -> np.ndarray:
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    @property
    def right(self) -> np.ndarray:
        return np.array([-math.sin(self.angle), math.cos(self.angle)])

    def ray_directions(self) -> np.ndarray:
        angles = self.angle + RAY_OFFSETS
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # ------------------------------------------------------------------ dynamics

    def apply_action(self, action, dt: float, boxes: np.ndarray) -> Bullet | None:
        """Apply ``(move, aim, fire)``. Returns a new bullet if one was fired."""
        move, aim, fire = (int(a) for a in action)
        if not (0 <= move < NUM_MOVE_ACTIONS and 0 <= aim < NUM_AIM_ACTIONS and 0 <= fire < NUM_FIRE_ACTIONS):
            raise ValueError(f"Invalid action {action!r}; expected values within {ACTION_DIMS}")
        self.last_action = np.array([move, aim, fire], dtype=np.int64)

        self.angle = (self.angle + (aim - 1) * self.turn_rate * dt + math.pi) % (2.0 * math.pi) - math.pi

        local = MOVE_DIRECTIONS[move]
        target = (local[0] * self.forward + local[1] * self.right) * self.max_speed
        self.velocity = self.velocity + (target - self.velocity) * min(1.0, self.acceleration * dt)
        self._move(dt, boxes)

        self.cooldown = max(0.0, self.cooldown - dt)
        if fire and self.cooldown <= 0.0:
            self.cooldown = self.fire_cooldown
            return Bullet(
                position=self.position + self.forward * (self.radius + 2.0),
                velocity=self.forward * self.bullet_speed,
                damage=self.bullet_damage,
                max_range=self.bullet_range,
            )
        return None

    def _move(self, dt: float, boxes: np.ndarray) -> None:
        self.position, self.velocity = resolve_wall_collisions(
            self.position + self.velocity * dt, self.velocity, self.radius, boxes
        )

    def take_damage(self, amount: float) -> float:
        """Apply damage and return the amount actually removed from health."""
        dealt = min(self.health, amount)
        self.health -= dealt
        return dealt

    # ------------------------------------------------------------------ sensing

    def observe_bullets(self, boxes: np.ndarray, bullets: list[Bullet]) -> np.ndarray:
        """Visible hostile projectiles as local position and relative velocity.

        Visibility covers the continuous viewing cone, not just its discrete rays.
        Velocity is bounded componentwise using the projectile's speed plus ours.
        """
        features = []
        forward, right = self.forward, self.right
        cos_half_cone = math.cos(math.radians(VISION_CONE_DEG) / 2.0)
        for bullet in bullets:
            if not bullet.alive:
                continue
            offset = bullet.position - self.position
            distance = float(np.linalg.norm(offset))
            if distance > RAY_LENGTH:
                continue
            if distance > 1e-9:
                direction = offset / distance
                if float(direction @ forward) < cos_half_cone - 1e-9:
                    continue
                if float(ray_aabb_distances(self.position, direction[None, :], boxes)[0]) <= distance:
                    continue
            relative_velocity = bullet.velocity - self.velocity
            velocity_scale = max(float(np.linalg.norm(bullet.velocity)) + self.max_speed, 1e-9)
            features.append([
                float(offset @ forward) / RAY_LENGTH,
                float(offset @ right) / RAY_LENGTH,
                float(relative_velocity @ forward) / velocity_scale,
                float(relative_velocity @ right) / velocity_scale,
            ])
        return np.asarray(features, dtype=np.float32).reshape(-1, BULLET_FEATURE_SIZE)

    def sense(self, boxes: np.ndarray, enemies: list[Threat | Agent]) -> None:
        directions = self.ray_directions()
        self.wall_distances = np.minimum(ray_aabb_distances(self.position, directions, boxes), RAY_LENGTH)
        if self.angular_regions:
            self._sense_regions(boxes, enemies)
            return

        cone_dirs = directions[CONE_RAY_INDICES]
        enemy_t = np.full(NUM_CONE_RAYS, np.inf)
        for enemy in enemies:
            if enemy.alive:
                enemy_t = np.minimum(enemy_t, ray_circle_distances(self.position, cone_dirs, enemy.position, enemy.radius))
        # Enemies are only visible if nothing (wall) blocks the line of sight.
        self.enemy_detected = (enemy_t < self.wall_distances[CONE_RAY_INDICES]) & (enemy_t < RAY_LENGTH)
        self.enemy_distances = np.where(self.enemy_detected, enemy_t, RAY_LENGTH)

    def _sense_regions(self, boxes: np.ndarray, enemies: list[Threat | Agent]) -> None:
        self.enemy_detected = np.zeros(NUM_ENEMY_REGIONS, dtype=bool)
        self.enemy_distances = np.full(NUM_ENEMY_REGIONS, RAY_LENGTH)
        self.region_points = np.zeros((NUM_ENEMY_REGIONS, 2))
        self.center_detected, self.center_distance = False, RAY_LENGTH
        # Wall silhouettes change at corners. Partition there and at target
        # tangents, rather than sampling a fixed set of rays with blind gaps.
        corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2) - self.position
        corner_angles = (np.arctan2(corners[:, 1], corners[:, 0]) - self.angle + np.pi) % (2 * np.pi) - np.pi
        for enemy in enemies:
            if not enemy.alive:
                continue
            offset = enemy.position - self.position
            distance = float(np.linalg.norm(offset))
            if distance - enemy.radius >= RAY_LENGTH:
                continue
            bearing = (math.atan2(float(offset[1]), float(offset[0])) - self.angle + math.pi) % (2 * math.pi) - math.pi
            half_width = math.asin(min(1.0, enemy.radius / max(distance, 1e-12)))
            center_t = float(ray_circle_distances(self.position, self.forward[None, :], enemy.position, enemy.radius)[0])
            if center_t < min(self.wall_distances[0], self.center_distance):
                self.center_detected, self.center_distance = True, center_t
            for i, (left, right) in enumerate(zip(ENEMY_REGION_EDGES[:-1], ENEMY_REGION_EDGES[1:])):
                lo, hi = max(left, bearing - half_width), min(right, bearing + half_width)
                if distance <= enemy.radius:
                    lo, hi = left, right
                if lo > hi:
                    continue
                cuts = np.unique(np.concatenate(([lo, hi], corner_angles[(corner_angles > lo) & (corner_angles < hi)])))
                angles = np.concatenate((cuts, (cuts[:-1] + cuts[1:]) / 2, [np.clip(bearing, lo, hi)]))
                # Check immediately beside occluding corners for nearest visible surfaces.
                angles = np.clip(np.concatenate((angles, cuts - 1e-8, cuts + 1e-8)), lo, hi)
                world_angles = angles + self.angle
                rays = np.column_stack((np.cos(world_angles), np.sin(world_angles)))
                target_t = ray_circle_distances(self.position, rays, enemy.position, enemy.radius)
                wall_t = ray_aabb_distances(self.position, rays, boxes)
                visible_t = np.where((target_t < wall_t) & (target_t < RAY_LENGTH), target_t, np.inf)
                nearest = int(np.argmin(visible_t))
                if visible_t[nearest] < self.enemy_distances[i]:
                    self.enemy_detected[i] = True
                    self.enemy_distances[i] = visible_t[nearest]
                    self.region_points[i] = self.position + rays[nearest] * visible_t[nearest]

    def observation(self) -> np.ndarray:
        """Egocentric observation built from the latest :meth:`sense` call."""
        local_velocity = np.array([self.velocity @ self.forward, self.velocity @ self.right]) / self.max_speed
        previous_action = np.zeros(sum(ACTION_DIMS))
        previous_action[_ACTION_ONE_HOT_OFFSETS + self.last_action] = 1.0
        return np.concatenate(
            [
                self.wall_distances / RAY_LENGTH,
                self.enemy_detected.astype(np.float64),
                self.enemy_distances / RAY_LENGTH,
                *([[float(self.center_detected), self.center_distance / RAY_LENGTH]] if self.angular_regions else []),
                np.clip(local_velocity, -1.0, 1.0),
                [self.cooldown / self.fire_cooldown, self.health / self.max_health],
                previous_action,
            ]
        ).astype(np.float32)


@dataclass
class Threat:
    """Rule-based moving target that does not shoot back.

    It wanders between random waypoints it has a clear straight line to, steers away
    from nearby walls with short whisker rays and pauses briefly at each waypoint.
    With ``evade`` enabled it strafes sideways while the agent aims at it.
    """

    color: tuple[int, int, int] = (255, 110, 90)
    radius: float = 14.0
    max_health: float = 100.0
    speed: float = 0.0
    evade: bool = False
    whisker_length: float = 50.0

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    health: float = 0.0
    waypoint: np.ndarray | None = None
    pause: float = 0.0
    waypoint_timer: float = 0.0

    def reset(self, position: np.ndarray, speed: float, evade: bool) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros(2)
        self.health = self.max_health
        self.speed = speed
        self.evade = evade
        self.waypoint = None
        self.pause = 0.0
        self.waypoint_timer = 0.0

    @property
    def alive(self) -> bool:
        return self.health > 0.0

    def take_damage(self, amount: float) -> float:
        dealt = min(self.health, amount)
        self.health -= dealt
        return dealt

    def is_targeted_by(self, agent: Agent, boxes: np.ndarray, margin_deg: float = 8.0) -> bool:
        """True if the agent aims (roughly) at this threat with a clear line of sight."""
        offset = self.position - agent.position
        dist = float(np.hypot(*offset))
        if dist < 1e-9:
            return True
        direction = offset / dist
        half_width = math.atan2(self.radius, dist) + math.radians(margin_deg)
        if math.acos(float(np.clip(direction @ agent.forward, -1.0, 1.0))) > half_width:
            return False
        return float(ray_aabb_distances(agent.position, direction[None, :], boxes)[0]) > dist - self.radius

    def update(
        self,
        dt: float,
        boxes: np.ndarray,
        rng: np.random.Generator,
        sample_position: Callable[[], np.ndarray],
        agent: Agent,
    ) -> None:
        if self.speed <= 0.0:
            self.velocity = np.zeros(2)
            return

        desired = np.zeros(2)
        if self.evade and self.is_targeted_by(agent, boxes):
            away = self.position - agent.position
            lateral = np.array([-away[1], away[0]]) / max(float(np.hypot(*away)), 1e-9)
            if float(lateral @ self.velocity) < 0.0:
                lateral = -lateral  # keep strafing the same way instead of jittering
            desired = lateral
            self.pause = 0.0
        elif self.pause > 0.0:
            self.pause -= dt
        else:
            if self.waypoint is None or self.waypoint_timer <= 0.0:
                self._pick_waypoint(boxes, sample_position)
            to_waypoint = self.waypoint - self.position
            dist = float(np.hypot(*to_waypoint))
            if dist < self.radius:
                self.waypoint = None
                self.pause = float(rng.uniform(0.3, 1.2))
            else:
                desired = to_waypoint / dist
                self.waypoint_timer -= dt

        if desired.any():
            desired = self._steer(desired, boxes)
        target_velocity = desired * self.speed
        self.velocity = self.velocity + (target_velocity - self.velocity) * min(1.0, 10.0 * dt)
        self.position, self.velocity = resolve_wall_collisions(
            self.position + self.velocity * dt, self.velocity, self.radius, boxes
        )

    def _pick_waypoint(self, boxes: np.ndarray, sample_position: Callable[[], np.ndarray]) -> None:
        candidate = sample_position()
        for _ in range(30):
            offset = candidate - self.position
            dist = float(np.hypot(*offset))
            if dist > 1e-9 and float(ray_aabb_distances(self.position, (offset / dist)[None, :], boxes)[0]) > dist:
                break
            candidate = sample_position()
        self.waypoint = candidate
        # Give up on waypoints that take far longer than expected (e.g. stuck behind a corner).
        self.waypoint_timer = 2.0 * float(np.hypot(*(candidate - self.position))) / self.speed + 1.0

    def _steer(self, desired: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        angles = np.arange(8) * (np.pi / 4.0)
        whiskers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        distances = np.minimum(ray_aabb_distances(self.position, whiskers, boxes), self.whisker_length)
        clearance = np.maximum(0.0, 1.0 - (distances - self.radius) / (self.whisker_length - self.radius))
        steer = desired - 1.5 * (whiskers * clearance[:, None]).sum(axis=0)
        norm = float(np.hypot(*steer))
        return steer / norm if norm > 1e-6 else desired


@dataclass
class ShootingThreat(Threat):
    """Moving threat that fires at the agent when it has a clear line of sight."""

    color: tuple[int, int, int] = (220, 80, 180)
    fire_cooldown: float = 0.8
    bullet_speed: float = 450.0
    bullet_damage: float = 25.0
    bullet_range: float = 600.0
    cooldown: float = 0.0

    def reset(self, position: np.ndarray, speed: float, evade: bool) -> None:
        super().reset(position, speed, evade)
        self.cooldown = self.fire_cooldown

    def shoot(self, dt: float, boxes: np.ndarray, agent: Agent) -> Bullet | None:
        self.cooldown = max(0.0, self.cooldown - dt)
        if not self.alive or not agent.alive or self.cooldown > 1e-9:
            return None
        offset = agent.position - self.position
        distance = float(np.linalg.norm(offset))
        if distance <= self.radius + 2.0 or distance - agent.radius > self.bullet_range:
            return None
        direction = offset / distance
        wall_distance = float(ray_aabb_distances(self.position, direction[None, :], boxes)[0])
        if wall_distance <= distance - agent.radius:
            return None
        self.cooldown = self.fire_cooldown
        return Bullet(
            position=self.position + direction * (self.radius + 2.0),
            velocity=direction * self.bullet_speed,
            damage=self.bullet_damage,
            max_range=self.bullet_range,
        )


@dataclass
class Player(ShootingThreat):
    """Human-controlled opponent with Agent stats and shared combat mechanics."""

    radius: float = Agent.radius
    max_health: float = Agent.max_health
    max_speed: float = Agent.max_speed
    acceleration: float = Agent.acceleration
    fire_cooldown: float = Agent.fire_cooldown
    bullet_speed: float = Agent.bullet_speed
    bullet_damage: float = Agent.bullet_damage
    bullet_range: float = Agent.bullet_range
    angle: float = 0.0
    movement: np.ndarray = field(default_factory=lambda: np.zeros(2))
    aim_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    firing: bool = False

    def reset(self, position: np.ndarray, speed: float, evade: bool) -> None:
        super().reset(position, self.max_speed, False)
        self.cooldown = 0.0
        self.angle = 0.0
        self.movement = np.zeros(2)
        self.aim_position = self.position + np.array([1.0, 0.0])
        self.firing = False

    def set_controls(self, movement: np.ndarray, aim_position: np.ndarray, firing: bool) -> None:
        if movement.shape != (2,) or aim_position.shape != (2,):
            raise ValueError("Player movement and aim_position must be two-dimensional")
        if not np.isfinite(movement).all() or not np.isfinite(aim_position).all():
            raise ValueError("Player controls must be finite")
        self.movement = movement / max(1.0, float(np.linalg.norm(movement)))
        self.aim_position = aim_position.copy()
        self.firing = firing

    def update(
        self, dt: float, boxes: np.ndarray, rng: np.random.Generator,
        sample_position: Callable[[], np.ndarray], agent: Agent,
    ) -> None:
        if not self.alive:
            return
        target_velocity = self.movement * self.max_speed
        self.velocity += (target_velocity - self.velocity) * min(1.0, self.acceleration * dt)
        self.position, self.velocity = resolve_wall_collisions(
            self.position + self.velocity * dt, self.velocity, self.radius, boxes,
        )
        offset = self.aim_position - self.position
        if float(np.linalg.norm(offset)) > 1e-9:
            self.angle = math.atan2(float(offset[1]), float(offset[0]))

    def shoot(self, dt: float, boxes: np.ndarray, agent: Agent) -> Bullet | None:
        self.cooldown = max(0.0, self.cooldown - dt)
        if not self.alive or not agent.alive or not self.firing or self.cooldown > 1e-9:
            return None
        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        self.cooldown = self.fire_cooldown
        # Start at the body center so the muzzle cannot bypass an adjacent wall.
        return Bullet(self.position.copy(), direction * self.bullet_speed, self.bullet_damage, self.bullet_range)

    def can_see(self, position: np.ndarray, boxes: np.ndarray) -> bool:
        offset = position - self.position
        distance = float(np.linalg.norm(offset))
        if distance > RAY_LENGTH:
            return False
        if distance <= 1e-9:
            return True
        direction = offset / distance
        forward = np.array([math.cos(self.angle), math.sin(self.angle)])
        if float(direction @ forward) < math.cos(math.radians(VISION_CONE_DEG) / 2.0) - 1e-9:
            return False
        return float(ray_aabb_distances(self.position, direction[None, :], boxes)[0]) > distance
