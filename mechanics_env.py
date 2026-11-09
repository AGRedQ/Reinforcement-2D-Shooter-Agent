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
    """Navigate the interactive-test layout using discrete movement and turns."""

    metadata = {"render_modes": ["human"], "render_fps": 60}
    dt = 1 / 60
    half_angle_degrees = 45.0
    hit_reward = 1.0
    movement_reward = 0.2
    movement_reward_distance = 75.0
    target_visibility_reward = 0.1
    target_alignment_reward = 0.1
    missed_shot_penalty = 0.001
    step_penalty = 0.001
    target_memory_duration = 2.0
    spawn_separation = 100.0
    spawn_attempts = 1_000
    wall_segments = (
        ((300.0, 200.0), (500.0, 200.0)),
        ((500.0, 200.0), (500.0, 400.0)),
    )
    observation_size = 21
    movement_actions = (
        (0, 0),
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    )
    turn_actions = (-2, -1, 0, 1, 2)

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 600,
        use_walls: bool = True,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if not isinstance(use_walls, bool):
            raise ValueError("use_walls must be a boolean.")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.use_walls = use_walls
        self.simulation = Renderer() if render_mode == "human" else Environment()
        self.action_space = spaces.MultiDiscrete([len(self.movement_actions), 5, 2])
        low = np.full(self.observation_size, -1.0, dtype=np.float32)
        high = np.ones(self.observation_size, dtype=np.float32)
        low[6:9] = 0.0
        low[11:13] = 0.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.agent = Agent(Vector2())
        self.target = Entity(Vector2())
        self.agent_spawn = Vector2()
        self.last_seen_target_position: Vector2 | None = None
        self.last_seen_target_age = self.target_memory_duration
        self.shots = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.movement_reward_earned = False
        self.target_visibility_reward_earned = False
        self.target_alignment_reward_earned = False
        self._needs_reset = True

    def _map_observation(self) -> list[float]:
        if not self.use_walls:
            return [0.0] * 8
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
            and self._target_in_view()
        )

    def _target_in_view(self) -> bool:
        return is_in_field_of_view(
            self.agent.position,
            Vector2(self.agent.aim_x, self.agent.aim_y),
            self.target.position,
            self.half_angle_degrees,
        ) and has_line_of_sight(
            self.agent.position, self.target.position, self.simulation.walls
        )

    def _target_aligned(self) -> bool:
        offset = self.target.position - self.agent.position
        direction = Vector2(self.agent.aim_x, self.agent.aim_y)
        if direction.length_squared() == 0:
            return False
        fraction = max(0.0, offset.dot(direction) / direction.length_squared())
        closest_point = self.agent.position + direction * fraction
        return self.target.position.distance_squared_to(closest_point) <= self.target.radius ** 2

    def _update_target_memory(self, target_visible: bool) -> None:
        if target_visible:
            self.last_seen_target_position = self.target.position.copy()
            self.last_seen_target_age = 0.0
        elif self.last_seen_target_position is not None:
            self.last_seen_target_age += self.dt
            if self.last_seen_target_age + 1e-12 >= self.target_memory_duration:
                self.last_seen_target_position = None
                self.last_seen_target_age = self.target_memory_duration

    def _target_memory_observation(self) -> tuple[Vector2, float, float]:
        if self.last_seen_target_position is None:
            return Vector2(), 0.0, 1.0
        offset = (
            self.last_seen_target_position - self.agent.position
        ) / self.simulation.bounds.length()
        return (
            offset,
            1.0,
            self.last_seen_target_age / self.target_memory_duration,
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
        remembered_offset, has_target_memory, memory_age = self._target_memory_observation()
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
                remembered_offset.x,
                remembered_offset.y,
                has_target_memory,
                memory_age,
                *self._map_observation(),
            ],
            dtype=np.float32,
        )

    def _info(self, success: bool) -> dict[str, Any]:
        return {
            "is_success": success,
            "target_visible": self._target_visible(),
            "shots": self.shots,
            "episode_reward": self.episode_reward,
            "last_reward": self.last_reward,
            "movement_reward_earned": self.movement_reward_earned,
            "target_visibility_reward_earned": self.target_visibility_reward_earned,
            "target_alignment_reward_earned": self.target_alignment_reward_earned,
            "has_target_memory": self.last_seen_target_position is not None,
            "last_seen_target_age": self.last_seen_target_age,
            "elapsed_time": self.simulation.step_count * self.dt,
            "episode_steps": self.simulation.step_count,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("MechanicsEnv does not support reset options.")
        self.simulation.walls = [
            Wall(Vector2(start), Vector2(end))
            for start, end in self.wall_segments
        ] if self.use_walls else []
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
        self.last_seen_target_position = None
        self.last_seen_target_age = self.target_memory_duration
        self._update_target_memory(self._target_visible())
        self.shots = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.movement_reward_earned = False
        self.target_visibility_reward_earned = False
        self.target_alignment_reward_earned = False
        if isinstance(self.simulation, Renderer):
            self.simulation.enable_field_of_view(
                self.agent, half_angle_degrees=self.half_angle_degrees
            )
        self._needs_reset = False
        return self._observation(), self._info(False)

    def _movement_bonus(self) -> float:
        earned = (
            not self.movement_reward_earned
            and self.agent.position.distance_to(self.agent_spawn)
            >= self.movement_reward_distance
        )
        self.movement_reward_earned |= earned
        return self.movement_reward * earned

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping or after an episode ends.")
        action = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action):
            raise ValueError("Action must be movement, turn, and fire indices.")
        move_x, move_y = self.movement_actions[int(action[0])]
        turn = self.turn_actions[int(action[1])]
        request_fire = bool(action[2])
        ready = self.agent.shoot_cooldown - self.dt <= 0
        fired = request_fire and ready
        core_action = Action(
            move_x, move_y, fire=request_fire, turn=turn
        )
        target_was_present = self.target in self.simulation.entities
        self.simulation.step([core_action], dt=self.dt)
        self.shots += int(fired)

        terminated = self.target not in self.simulation.entities
        truncated = not terminated and self.simulation.step_count >= self.max_steps
        movement_bonus = self._movement_bonus()
        target_visible = target_was_present and self._target_in_view()
        self._update_target_memory(target_visible)
        earned_target_visibility_reward = (
            target_visible
            and not terminated
            and not self.target_visibility_reward_earned
        )
        self.target_visibility_reward_earned |= earned_target_visibility_reward
        earned_target_alignment_reward = (
            target_visible
            and not terminated
            and self._target_aligned()
            and not self.target_alignment_reward_earned
        )
        self.target_alignment_reward_earned |= earned_target_alignment_reward
        reward = (
            self.hit_reward * terminated
            + movement_bonus
            + self.target_visibility_reward * earned_target_visibility_reward
            + self.target_alignment_reward * earned_target_alignment_reward
            - self.missed_shot_penalty * (fired and not terminated)
            - self.step_penalty
        )
        self.last_reward = reward
        self.episode_reward += reward
        self._needs_reset = terminated or truncated
        return self._observation(), reward, terminated, truncated, self._info(terminated)

    def _score_hud(self) -> tuple[str, ...]:
        return (
            f"Score (episode reward): {self.episode_reward:+.3f}",
            f"Last reward: {self.last_reward:+.3f} | Step: {self.simulation.step_count}/{self.max_steps}",
        )

    def render(self) -> None:
        if isinstance(self.simulation, Renderer):
            self.simulation.render(hud_lines=self._score_hud())

    def close(self) -> None:
        if isinstance(self.simulation, Renderer):
            self.simulation.close()
