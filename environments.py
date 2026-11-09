"""Hunting environment: fixed map, moving threat, rewards and the curriculum."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from entities import (
    ACTION_DIMS,
    OBSERVATION_SIZE,
    Agent,
    Bullet,
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


def default_map() -> MapLayout:
    width, height, t = 800.0, 800.0, 20.0
    walls = [
        Wall(0, 0, width, t),
        Wall(0, height - t, width, t),
        Wall(0, 0, t, height),
        Wall(width - t, 0, t, height),
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

    The agent cannot die, so episodes always end by truncation after ``max_steps``.
    Observation: egocentric vector of size ``OBSERVATION_SIZE`` (see ``entities.Agent.observation``).
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
    ):
        super().__init__()
        self.layout = layout or default_map()
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.dt = dt
        self.replay_probability = replay_probability
        self.set_stage(stage)

        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(ACTION_DIMS)

        self.agent = Agent(agent_id=0)
        self.threat = Threat()
        self.bullets: list[Bullet] = []
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
        self.steps = 0
        self._detection_delays = []
        self.episode_stats = {
            "reward": 0.0,
            "kills": 0.0,
            "hits": 0.0,
            "shots": 0.0,
            "wasted_shots": 0.0,
            "distance": 0.0,
            "stage": float(self.episode_stage),
            "replay": float(replay),
        }
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
        return self.agent.observation(), {}

    def step(self, action):
        if self.steps >= self.max_steps:
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
            if b.update(self.dt, boxes, [self.threat]) is None:
                continue
            self.threat.take_damage(b.damage)
            stats["hits"] += 1
            reward += cfg.hit
            if not self.threat.alive:
                stats["kills"] += 1
                reward += cfg.kill_bonus
                self.bullets = []
                self._spawn_threat()
                break
        self.bullets = [b for b in self.bullets if b.alive]
        self.steps += 1

        self.agent.sense(boxes, [self.threat])
        self._update_detection()

        stats["reward"] += reward
        truncated = self.steps >= self.max_steps
        info: dict = {}
        if truncated:
            delays = self._detection_delays + ([] if self._threat_detected else [self.steps - self._spawn_step])
            stats["detect_steps"] = float(np.mean(delays))
            stats["length"] = float(self.steps)
            info["episode"] = dict(stats)
        return self.agent.observation(), float(reward), False, truncated, info

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
