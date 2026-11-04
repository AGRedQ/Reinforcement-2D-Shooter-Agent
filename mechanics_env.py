"""Gymnasium task for training with movement, walls, line of sight, and vision."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pygame import Vector2

from entities import (
    Action,
    Agent,
    Entity,
    Wall,
    has_line_of_sight,
    is_in_field_of_view,
)
from environment import Environment
from renderer import Renderer


class MechanicsEnv(gym.Env):
    """Navigate the interactive-test layout and eliminate its stationary target."""

    metadata = {"render_modes": ["human"], "render_fps": 60}
    dt = 1 / 60
    half_angle_degrees = 45.0
    hit_reward = 1.0
    movement_reward = 0.2
    movement_reward_distance = 75.0
    missed_shot_penalty = 0.001
    step_penalty = 0.001
    spawn_separation = 100.0
    spawn_attempts = 1_000
    wall_segments = (
        ((300.0, 200.0), (500.0, 200.0)),
        ((500.0, 200.0), (500.0, 400.0)),
    )
    observation_size = 17

    def __init__(self, render_mode: str | None = None, max_steps: int = 600):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.simulation = Renderer() if render_mode == "human" else Environment()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        low = np.full(self.observation_size, -1.0, dtype=np.float32)
        high = np.ones(self.observation_size, dtype=np.float32)
        low[6:9] = 0.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.agent = Agent(Vector2())
        self.target = Entity(Vector2())
        self.agent_spawn = Vector2()
        self.shots = 0
        self.movement_reward_earned = False
        self._needs_reset = True

    def _wall_observation(self) -> list[float]:
        bounds = self.simulation.bounds
        return [
            coordinate
            for start, end in self.wall_segments
            for x, y in (start, end)
            for coordinate in (2 * x / bounds.x - 1, 2 * y / bounds.y - 1)
        ]

    def _target_visible(self) -> bool:
        return (
            self.target in self.simulation.entities
            and is_in_field_of_view(
                self.agent.position,
                Vector2(self.agent.aim_x, self.agent.aim_y),
                self.target.position,
                self.half_angle_degrees,
            )
            and has_line_of_sight(
                self.agent.position, self.target.position, self.simulation.walls
            )
        )

    @staticmethod
    def _distance_to_wall(position: Vector2, wall: Wall) -> float:
        edge = wall.end - wall.start
        if edge.length_squared() == 0:
            return position.distance_to(wall.start)
        fraction = max(
            0.0,
            min(1.0, (position - wall.start).dot(edge) / edge.length_squared()),
        )
        return position.distance_to(wall.start + fraction * edge)

    def _sample_spawn(
        self, radius: float, minimum_distance_from: Vector2 | None = None
    ) -> Vector2:
        bounds = self.simulation.bounds
        for _ in range(self.spawn_attempts):
            position = Vector2(
                self.np_random.uniform(radius, bounds.x - radius),
                self.np_random.uniform(radius, bounds.y - radius),
            )
            if all(
                self._distance_to_wall(position, wall) >= radius
                for wall in self.simulation.walls
            ) and (
                minimum_distance_from is None
                or position.distance_to(minimum_distance_from) >= self.spawn_separation
            ):
                return position
        raise RuntimeError("Could not sample a valid spawn position.")

    def _observation(self) -> np.ndarray:
        bounds = self.simulation.bounds
        target_visible = self._target_visible()
        offset = (
            (self.target.position - self.agent.position) / bounds.length()
            if target_visible
            else Vector2()
        )
        return np.array(
            [
                2 * self.agent.position.x / bounds.x - 1,
                2 * self.agent.position.y / bounds.y - 1,
                self.agent.aim_x,
                self.agent.aim_y,
                offset.x,
                offset.y,
                float(target_visible),
                min(1.0, self.agent.shoot_cooldown / self.simulation.fire_interval),
                max(0.0, 1 - self.simulation.step_count / self.max_steps),
                *self._wall_observation(),
            ],
            dtype=np.float32,
        )

    def _info(self, success: bool) -> dict[str, Any]:
        return {
            "is_success": success,
            "target_visible": self._target_visible(),
            "shots": self.shots,
            "movement_reward_earned": self.movement_reward_earned,
            "elapsed_time": self.simulation.step_count * self.dt,
            "episode_steps": self.simulation.step_count,
        }

    @staticmethod
    def _movement_component(value: float) -> int:
        if value <= -1 / 3:
            return -1
        if value >= 1 / 3:
            return 1
        return 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("MechanicsEnv does not support reset options.")
        self.simulation.walls = [
            Wall(Vector2(start), Vector2(end))
            for start, end in self.wall_segments
        ]
        self.agent_spawn = self._sample_spawn(Agent(Vector2()).radius)
        self.agent = Agent(self.agent_spawn)
        self.target = Entity(
            self._sample_spawn(
                Entity(Vector2()).radius, minimum_distance_from=self.agent.position
            )
        )
        heading = self.np_random.uniform(0, 2 * np.pi)
        self.agent.aim_x = float(np.cos(heading))
        self.agent.aim_y = float(np.sin(heading))
        self.simulation.entities = [self.agent, self.target]
        self.simulation.step_count = 0
        self.shots = 0
        self.movement_reward_earned = False
        if isinstance(self.simulation, Renderer):
            self.simulation.enable_field_of_view(
                self.agent, half_angle_degrees=self.half_angle_degrees
            )
        self._needs_reset = False
        return self._observation(), self._info(False)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping or after an episode ends.")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must contain five finite values in [-1, 1].")
        request_fire = bool(action[4] > 0)
        ready = self.agent.shoot_cooldown - self.dt <= 0
        fired = request_fire and ready
        core_action = Action(
            self._movement_component(float(action[0])),
            self._movement_component(float(action[1])),
            float(action[2]),
            float(action[3]),
            request_fire,
        )
        self.simulation.step([core_action], dt=self.dt)
        self.shots += int(fired)

        terminated = self.target not in self.simulation.entities
        truncated = not terminated and self.simulation.step_count >= self.max_steps
        earned_movement_reward = (
            not self.movement_reward_earned
            and self.agent.position.distance_to(self.agent_spawn)
            >= self.movement_reward_distance
        )
        self.movement_reward_earned |= earned_movement_reward
        reward = (
            self.hit_reward * terminated
            + self.movement_reward * earned_movement_reward
            - self.missed_shot_penalty * (fired and not terminated)
            - self.step_penalty
        )
        self._needs_reset = terminated or truncated
        return self._observation(), reward, terminated, truncated, self._info(terminated)

    def render(self) -> None:
        if isinstance(self.simulation, Renderer):
            self.simulation.render()

    def close(self) -> None:
        if isinstance(self.simulation, Renderer):
            self.simulation.close()
