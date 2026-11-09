"""Movement-only exploration; shared physics, finite rays, episodic tile novelty."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pygame import Vector2

from entities import Action, Agent, Wall, wall_intersection_fraction
from environment import Environment
from map_generation import generate_map
from renderer import Renderer


@dataclass(frozen=True)
class ExplorationSettings:
    map_mode: str
    map_seed: int
    random_spawn: bool
    columns: int
    rows: int
    max_walls: int
    ray_range: float
    tile_size: int
    new_tile_reward: float
    reward_decay: float
    step_penalty: float
    max_steps: int

    def __post_init__(self) -> None:
        if self.map_mode not in ("fixed", "random"):
            raise ValueError("map_mode must be fixed or random.")
        if type(self.random_spawn) is not bool:
            raise ValueError("random_spawn must be a boolean.")
        for name, minimum in (
            ("map_seed", 0), ("columns", 2), ("rows", 2),
            ("max_walls", 0), ("tile_size", 1), ("max_steps", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("ray_range", "new_tile_reward", "reward_decay", "step_penalty"):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.ray_range == 0 or self.new_tile_reward == 0:
            raise ValueError("ray_range and new_tile_reward must be positive.")


class ExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    dt = 1 / 60
    ray_count = 16
    movement_actions = (
        (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
    )

    def __init__(
        self, settings: ExplorationSettings, render_mode: str | None = None,
        *, seed_log: Path | None = None,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human.")
        self.settings = settings
        self.render_mode = render_mode
        # Validate geometry before opening a rendering window.
        simulation = Environment()
        radius = Agent(Vector2()).radius
        for extent, cells in (
            (simulation.bounds.x, settings.columns),
            (simulation.bounds.y, settings.rows),
        ):
            if extent % (cells * settings.tile_size) != 0:
                raise ValueError("tile_size must divide each generated map cell dimension.")
        if settings.tile_size <= 2 * radius:
            raise ValueError("tile_size must exceed the agent diameter.")
        self.simulation = Renderer() if render_mode == "human" else simulation
        self.agent = Agent(Vector2())
        self.simulation.entities = [self.agent]
        self.action_space = spaces.Discrete(len(self.movement_actions))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(18,), dtype=np.float32)
        self.ray_directions = tuple(
            Vector2(1, 0).rotate(index * 360 / self.ray_count)
            for index in range(self.ray_count)
        )
        width, height = self.simulation.bounds
        self.boundaries = [
            Wall(Vector2(0, 0), Vector2(width, 0)),
            Wall(Vector2(width, 0), Vector2(width, height)),
            Wall(Vector2(width, height), Vector2(0, height)),
            Wall(Vector2(0, height), Vector2(0, 0)),
        ]
        self.tile_columns = int(width / settings.tile_size)
        self.tile_rows = int(height / settings.tile_size)
        self.total_tiles = self.tile_columns * self.tile_rows
        self.visited: set[tuple[int, int]] = set()
        self.map_seed = settings.map_seed
        self.episode_reward = 0.0
        self.seed_log = seed_log
        self.episode_index = 0
        self._needs_reset = True

    def _tile(self) -> tuple[int, int]:
        return (
            int(self.agent.position.x // self.settings.tile_size),
            int(self.agent.position.y // self.settings.tile_size),
        )

    def ray_distances(self) -> np.ndarray:
        walls = self.simulation.walls + self.boundaries
        origin = self.agent.position
        values = []
        for direction in self.ray_directions:
            end = origin + direction * self.settings.ray_range
            nearest = 1.0
            for wall in walls:
                fraction = wall_intersection_fraction(origin, end, wall)
                if fraction is not None:
                    nearest = min(nearest, fraction)
            values.append(nearest)
        return np.asarray(values, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        return np.concatenate((
            self.ray_distances(),
            np.asarray([
                self.agent.position.x / self.simulation.bounds.x,
                self.agent.position.y / self.simulation.bounds.y,
            ], dtype=np.float32),
        ))

    def _info(self, bonus: float = 0.0) -> dict[str, Any]:
        return {
            "map_seed": self.map_seed,
            "tiles_visited": len(self.visited),
            "new_tiles_visited": len(self.visited) - 1,
            "tile_coverage": len(self.visited) / self.total_tiles,
            "exploration_reward": bonus,
            "episode_steps": self.simulation.step_count,
            "episode_reward": self.episode_reward,
        }

    def _random_spawn(self) -> Vector2:
        radius = self.agent.radius
        for _ in range(10_000):
            point = Vector2(
                self.np_random.uniform(radius, self.simulation.bounds.x - radius),
                self.np_random.uniform(radius, self.simulation.bounds.y - radius),
            )
            for wall in self.simulation.walls:
                edge = wall.end - wall.start
                fraction = max(0.0, min(1.0, (point - wall.start).dot(edge) / edge.length_squared()))
                if point.distance_to(wall.start + fraction * edge) < radius:
                    break
            else:
                return point
        raise RuntimeError("Could not find a collision-free random spawn.")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("ExplorationEnv does not support reset options.")
        self.map_seed = (
            int(self.np_random.integers(0, 2**31))
            if self.settings.map_mode == "random" else self.settings.map_seed
        )
        layout = generate_map(
            self.simulation.bounds, agent_radius=self.agent.radius,
            seed=self.map_seed, max_walls=self.settings.max_walls,
            columns=self.settings.columns, rows=self.settings.rows,
        )
        self.simulation.walls = layout.walls
        spawn = (
            self._random_spawn() if self.settings.random_spawn
            else layout.nodes[layout.agent_spawn].copy()
        )
        self.agent = Agent(spawn)
        self.simulation.entities = [self.agent]
        self.simulation.step_count = 0
        self.visited = {self._tile()}
        self.episode_reward = 0.0
        self._needs_reset = False
        if self.seed_log is not None:
            with self.seed_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "episode_index": self.episode_index,
                    "map_seed": self.map_seed, "spawn": list(spawn),
                }) + "\n")
        self.episode_index += 1
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset before stepping or after an episode ends.")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 8.")
        move_x, move_y = self.movement_actions[int(action)]
        self.simulation.step([Action(move_x, move_y, turn=0)], dt=self.dt)
        tile = self._tile()
        bonus = 0.0
        if tile not in self.visited:
            previous_new_tiles = len(self.visited) - 1
            progress = previous_new_tiles / (self.total_tiles - 1)
            bonus = self.settings.new_tile_reward / (1 + self.settings.reward_decay * progress)
            self.visited.add(tile)
        reward = bonus - self.settings.step_penalty
        self.episode_reward += reward
        truncated = self.simulation.step_count >= self.settings.max_steps
        self._needs_reset = truncated
        return self._observation(), reward, False, truncated, self._info(bonus)

    def render(self) -> None:
        if isinstance(self.simulation, Renderer):
            self.simulation.render(
                draw_overlay=self._draw_overlay,
                hud_lines=(
                    f"Map {self.map_seed} | Tiles {len(self.visited)}/{self.total_tiles}",
                    f"Step {self.simulation.step_count}/{self.settings.max_steps}"
                    f" | Reward {self.episode_reward:+.3f}",
                ),
            )

    def _draw_overlay(self, screen: pygame.Surface) -> None:
        size = self.settings.tile_size
        for column, row in self.visited:
            pygame.draw.rect(screen, (35, 65, 50), (column * size, row * size, size, size))
        for column in range(self.tile_columns + 1):
            pygame.draw.line(screen, (45, 50, 60), (column * size, 0),
                             (column * size, self.simulation.bounds.y))
        for row in range(self.tile_rows + 1):
            pygame.draw.line(screen, (45, 50, 60), (0, row * size),
                             (self.simulation.bounds.x, row * size))
        for direction, distance in zip(self.ray_directions, self.ray_distances()):
            end = self.agent.position + direction * float(distance) * self.settings.ray_range
            pygame.draw.line(screen, (85, 165, 210), self.agent.position, end)

    def close(self) -> None:
        if isinstance(self.simulation, Renderer):
            self.simulation.close()
