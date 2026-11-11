"""Hunting and opt-in combat environments sharing maps, mechanics and curriculum."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from entities import (
    ACTION_DIMS,
    ANGULAR_OBSERVATION_SIZE,
    BULLET_FEATURE_SIZE,
    OBSERVATION_SIZE,
    Agent,
    Bullet,
    ShootingThreat,
    Threat,
    Wall,
    circle_overlaps_boxes,
    ray_aabb_distances,
    walls_to_array,
)


@dataclass
class MapLayout:
    width: float
    height: float
    walls: list[Wall]
    boxes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.boxes = walls_to_array(self.walls)


def boundary_walls(width: float, height: float, thickness: float) -> list[Wall]:
    return [
        Wall(0, 0, width, thickness),
        Wall(0, height - thickness, width, thickness),
        Wall(0, 0, thickness, height),
        Wall(width - thickness, 0, thickness, height),
    ]


def default_map() -> MapLayout:
    width, height, t = 800.0, 800.0, 20.0
    walls = boundary_walls(width, height, t) + [
        Wall(370, 370, 60, 60),
        Wall(170, 160, 140, 30),
        Wall(490, 610, 140, 30),
        Wall(610, 170, 30, 140),
        Wall(160, 490, 30, 140),
        Wall(375, 120, 50, 90),
        Wall(375, 590, 50, 90),
    ]
    return MapLayout(width, height, walls)


@dataclass
class RewardConfig:
    hit: float = 1.0
    kill_bonus: float = 3.0
    """Added on top of ``hit`` for the killing shot."""
    wasted_shot: float = 0.1
    """Subtracted when a bullet is fired while no enemy ray sees the threat."""
    damage_taken: float = 0.04
    """Penalty per health point lost in combat mode."""
    death: float = 3.0
    """Additional penalty when the agent dies in combat mode."""


@dataclass(frozen=True)
class StageConfig:
    threat_speed: float
    spawn_half_angle_deg: float
    """Spawn within this angle of the agent's aim; 180 means anywhere."""
    spawn_distance: tuple[float, float]
    require_line_of_sight: bool = False
    prefer_hidden: bool = False
    evade: bool = False


STAGES: tuple[StageConfig, ...] = (
    StageConfig(0.0, 10.0, (80.0, 250.0), require_line_of_sight=True),
    StageConfig(60.0, 30.0, (100.0, 350.0), require_line_of_sight=True),
    StageConfig(120.0, 90.0, (150.0, 500.0)),
    StageConfig(180.0, 180.0, (200.0, 700.0)),
    StageConfig(180.0, 180.0, (250.0, 900.0), prefer_hidden=True, evade=True),
)


class Curriculum:
    """Promotes to the next stage once the mean reward of the last ``window`` episodes
    played at the current stage (excluding replays of earlier stages) reaches ``threshold``."""

    def __init__(self, num_stages: int = len(STAGES), threshold: float = 35.0, window: int = 100, stage: int = 0):
        self.num_stages = num_stages
        self.threshold = threshold
        self.window = window
        self.stage = stage
        self.rewards: deque[float] = deque(maxlen=window)

    @property
    def is_final_stage(self) -> bool:
        return self.stage >= self.num_stages - 1

    def record(self, episode: dict) -> bool:
        """Record a finished episode; returns True if this caused a promotion."""
        if episode["replay"] or int(episode["stage"]) != self.stage or self.is_final_stage:
            return False
        self.rewards.append(float(episode["reward"]))
        if len(self.rewards) == self.window and float(np.mean(self.rewards)) >= self.threshold:
            self.stage += 1
            self.rewards.clear()
            return True
        return False

    def state_dict(self) -> dict:
        return {"stage": self.stage, "rewards": list(self.rewards)}

    def load_state_dict(self, state: dict) -> None:
        self.stage = int(state["stage"])
        self.rewards = deque(state["rewards"], maxlen=self.window)


class HuntEnv(gym.Env):
    """The agent hunts a rule-based threat that respawns whenever it dies.

    By default the agent cannot die and episodes truncate after ``max_steps``.
    With ``shoot_back=True``, a ``ShootingThreat`` damages the agent and death
    terminates the episode. Enemy bullets already in flight survive threat respawns.
    ``map_seeds`` opts into generated layouts, cycling once per reset. An explicit
    reset seed restarts the map cycle as well as the spawn RNG; threat respawns
    do not change maps. Without ``map_seeds``, the fixed/custom layout is preserved.
    Observation: egocentric vector of size ``OBSERVATION_SIZE`` (see ``entities.Agent.observation``).
    ``bullet_sensing=True`` instead returns a dict with that ``state`` vector and
    a variable-length ``bullets`` array (N, 4), limited by cone, range and walls.
    ``angular_regions=True`` replaces enemy rays with six continuous sectors and
    a separate center-ray flag/distance, giving ``ANGULAR_OBSERVATION_SIZE`` state values.
    Action: ``MultiDiscrete([9, 3, 2])`` = (move, aim, fire).
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
        self,
        layout: MapLayout | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 1000,
        dt: float = 1.0 / 30.0,
        stage: int = 0,
        replay_probability: float = 0.2,
        shoot_back: bool = False,
        map_seeds: Sequence[int] | None = None,
        bullet_sensing: bool = False,
        angular_regions: bool = False,
    ):
        super().__init__()
        if map_seeds is not None:
            if layout is not None:
                raise ValueError("Choose either a fixed layout or map_seeds, not both.")
            if isinstance(map_seeds, range):
                if not map_seeds or min(map_seeds[0], map_seeds[-1]) < 0:
                    raise ValueError("map_seeds must contain nonnegative integers.")
            else:
                map_seeds = tuple(map_seeds)
                if not map_seeds or any(type(seed) is not int or seed < 0 for seed in map_seeds):
                    raise ValueError("map_seeds must be a nonempty sequence of nonnegative integers.")
        self.map_seeds = map_seeds
        self.map_seed: int | None = None
        self._map_index = 0
        self.layout = layout or default_map()
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.dt = dt
        self.replay_probability = replay_probability
        self.shoot_back = shoot_back
        self.bullet_sensing = bullet_sensing
        self.set_stage(stage)

        obs_size = ANGULAR_OBSERVATION_SIZE if angular_regions else OBSERVATION_SIZE
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(obs_size,), dtype=np.float32)
        if bullet_sensing:
            self.observation_space = gym.spaces.Dict({
                "state": self.observation_space,
                "bullets": gym.spaces.Sequence(
                    gym.spaces.Box(-1.0, 1.0, shape=(BULLET_FEATURE_SIZE,), dtype=np.float32),
                    stack=True,
                ),
            })
        self.action_space = gym.spaces.MultiDiscrete(ACTION_DIMS)

        self.agent = Agent(agent_id=0, angular_regions=angular_regions)
        self.threat = ShootingThreat() if shoot_back else Threat()
        self.bullets: list[Bullet] = []
        self.threat_bullets: list[Bullet] = []
        self.steps = 0
        self.episode_stage = stage
        self.episode_stats: dict[str, float] = {}
        self._spawn_step = 0
        self._threat_detected = False
        self._detection_delays: list[int] = []

    def set_stage(self, stage: int) -> None:
        """Set the curriculum stage used from the next reset (and its threat respawns)."""
        if not 0 <= stage < len(STAGES):
            raise ValueError(f"Stage must be in [0, {len(STAGES) - 1}], got {stage}")
        self.stage = stage

    @property
    def stage_config(self) -> StageConfig:
        return STAGES[self.episode_stage]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.map_seeds is not None:
            # MapLayout lives here, so import the generator after module initialization.
            from map_generation import generate_map

            if seed is not None:
                self._map_index = 0
            self.map_seed = self.map_seeds[self._map_index % len(self.map_seeds)]
            self.layout = generate_map(self.map_seed)
            self._map_index += 1
        options = options or {}
        replay = False
        if "stage" in options:
            self.episode_stage = int(options["stage"])
        elif self.stage > 0 and self.np_random.random() < self.replay_probability:
            self.episode_stage = int(self.np_random.integers(0, self.stage))
            replay = True
        else:
            self.episode_stage = self.stage

        self.bullets = []
        self.threat_bullets = []
        self.steps = 0
        self._detection_delays = []
        self.episode_stats = {
            "reward": 0.0,
            "kills": 0.0,
            "hits": 0.0,
            "shots": 0.0,
            "wasted_shots": 0.0,
            "threat_shots": 0.0,
            "damage_taken": 0.0,
            "deaths": 0.0,
            "distance": 0.0,
            "stage": float(self.episode_stage),
            "replay": float(replay),
        }
        if self.map_seed is not None:
            self.episode_stats["map_seed"] = self.map_seed
        cfg = self.stage_config
        for _ in range(50):
            self.agent.reset(self._random_free_position(), float(self.np_random.uniform(-math.pi, math.pi)))
            threat_position = self._threat_spawn_position(cfg, strict=True)
            if threat_position is not None:
                break
        else:
            threat_position = self._threat_spawn_position(cfg)
        self._spawn_threat(threat_position)
        self.agent.sense(self.layout.boxes, [self.threat])
        self._update_detection()
        return self._observation(), ({"map_seed": self.map_seed} if self.map_seed is not None else {})

    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        state = self.agent.observation()
        if not self.bullet_sensing:
            return state
        return {
            "state": state,
            "bullets": self.agent.observe_bullets(self.layout.boxes, self.threat_bullets),
        }

    def step(self, action):
        if self.steps >= self.max_steps or (self.shoot_back and not self.agent.alive):
            raise RuntimeError("Episode is over; call reset() before step().")
        cfg = self.reward_config
        stats = self.episode_stats
        boxes = self.layout.boxes
        reward = 0.0

        threat_visible = bool(self.agent.enemy_detected.any())
        previous_position = self.agent.position
        bullet = self.agent.apply_action(action, self.dt, boxes)
        if bullet is not None:
            self.bullets.append(bullet)
            stats["shots"] += 1
            if not threat_visible:
                stats["wasted_shots"] += 1
                reward -= cfg.wasted_shot
        stats["distance"] += float(np.hypot(*(self.agent.position - previous_position)))

        self.threat.update(self.dt, boxes, self.np_random, self._random_free_position, self.agent)

        for b in self.bullets:
            target = b.update(self.dt, boxes, [self.threat])
            if target is None:
                continue
            target.take_damage(b.damage)
            stats["hits"] += 1
            reward += cfg.hit
            if not self.threat.alive:
                stats["kills"] += 1
                reward += cfg.kill_bonus
                self.bullets = []
                self._spawn_threat()
                break
        self.bullets = [b for b in self.bullets if b.alive]

        if isinstance(self.threat, ShootingThreat):
            enemy_bullet = self.threat.shoot(self.dt, boxes, self.agent)
            if enemy_bullet is not None:
                self.threat_bullets.append(enemy_bullet)
                stats["threat_shots"] += 1
        for b in self.threat_bullets:
            if b.update(self.dt, boxes, [self.agent]) is not None:
                damage = self.agent.take_damage(b.damage)
                stats["damage_taken"] += damage
                reward -= cfg.damage_taken * damage
        self.threat_bullets = [b for b in self.threat_bullets if b.alive]
        terminated = self.shoot_back and not self.agent.alive
        if terminated:
            stats["deaths"] = 1.0
            reward -= cfg.death
        self.steps += 1

        self.agent.sense(boxes, [self.threat])
        self._update_detection()

        stats["reward"] += reward
        truncated = self.steps >= self.max_steps
        info: dict = {}
        if terminated or truncated:
            delays = self._detection_delays + ([] if self._threat_detected else [self.steps - self._spawn_step])
            stats["detect_steps"] = float(np.mean(delays))
            stats["length"] = float(self.steps)
            info["episode"] = dict(stats)
        return self._observation(), float(reward), terminated, truncated, info

    def _update_detection(self) -> None:
        if not self._threat_detected and self.agent.enemy_detected.any():
            self._threat_detected = True
            self._detection_delays.append(self.steps - self._spawn_step)

    def _spawn_threat(self, position: np.ndarray | None = None) -> None:
        cfg = self.stage_config
        if position is None:
            position = self._threat_spawn_position(cfg)
        self.threat.reset(position, cfg.threat_speed, cfg.evade)
        self._spawn_step = self.steps
        self._threat_detected = False

    def _threat_spawn_position(self, cfg: StageConfig, strict: bool = False) -> np.ndarray | None:
        """Sample a spawn satisfying the stage rules. Unless ``strict``, the rules are relaxed
        when the agent's surroundings make them impossible (e.g. facing a nearby wall)."""
        attempts = [(cfg.spawn_half_angle_deg, cfg.require_line_of_sight, cfg.prefer_hidden)]
        if not strict:
            attempts += [(180.0, cfg.require_line_of_sight, cfg.prefer_hidden), (180.0, False, False)]
        for half_angle, need_los, need_hidden in attempts:
            for _ in range(300):
                position = self._sample_relative_position(half_angle, cfg.spawn_distance)
                if position is None:
                    continue
                visible = self._line_of_sight(position)
                if (need_los and not visible) or (need_hidden and visible):
                    continue
                return position
        return None if strict else self._random_free_position()

    def _sample_relative_position(self, half_angle_deg: float, distance: tuple[float, float]) -> np.ndarray | None:
        angle = self.agent.angle + math.radians(float(self.np_random.uniform(-half_angle_deg, half_angle_deg)))
        dist = float(self.np_random.uniform(*distance))
        position = self.agent.position + dist * np.array([math.cos(angle), math.sin(angle)])
        r = self.threat.radius + 4.0
        if not (r <= position[0] <= self.layout.width - r and r <= position[1] <= self.layout.height - r):
            return None
        if circle_overlaps_boxes(position, r, self.layout.boxes):
            return None
        return position

    def _line_of_sight(self, position: np.ndarray) -> bool:
        offset = position - self.agent.position
        dist = float(np.hypot(*offset))
        wall = float(ray_aabb_distances(self.agent.position, (offset / dist)[None, :], self.layout.boxes)[0])
        return wall > dist - self.threat.radius

    def _random_free_position(self) -> np.ndarray:
        radius = self.agent.radius + 4.0
        for _ in range(1000):
            position = self.np_random.uniform([radius, radius], [self.layout.width - radius, self.layout.height - radius])
            if not circle_overlaps_boxes(position, radius, self.layout.boxes):
                return position
        raise RuntimeError("Could not find a free position; the map may be too crowded.")
