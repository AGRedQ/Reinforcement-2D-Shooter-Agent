"""Fixed generated-map task with observable first-visit exploration state."""

from typing import Any

import numpy as np
from gymnasium import spaces
from pygame import Vector2

from map_generation import generate_map
from mechanics_env import MechanicsEnv
from renderer import Renderer, draw_navigation


class WallgenNavpointEnv(MechanicsEnv):
    columns = 4
    rows = 3
    node_count = columns * rows
    connection_count = rows * (columns - 1) + columns * (rows - 1)
    observation_size = 13 + 3 * node_count + connection_count
    exploration_budget = 0.2
    node_reward = exploration_budget / (node_count - 1)

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 600,
        map_seed: int = 0,
        max_walls: int = 4,
    ):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.max_walls = max_walls
        try:
            self.set_map(map_seed)
        except ValueError:
            self.close()
            raise
        low = np.full(self.observation_size, -1.0, dtype=np.float32)
        low[6:9] = 0.0
        low[11:13] = 0.0
        low[15:13 + 3 * self.node_count:3] = 0.0
        low[13 + 3 * self.node_count:] = 0.0
        self.observation_space = spaces.Box(
            low=low, high=np.ones(self.observation_size, dtype=np.float32),
            dtype=np.float32,
        )
        self.visited: set[int] = set()
        self.new_nodes_visited = 0

    def set_map(self, map_seed: int) -> None:
        """Replace task geometry before resetting the shared simulation."""
        self.layout = generate_map(
            self.simulation.bounds,
            agent_radius=max(self.agent.radius, self.target.radius),
            seed=map_seed, max_walls=self.max_walls,
            columns=self.columns, rows=self.rows,
        )
        self.map_seed = map_seed
        self.wall_segments = tuple(
            ((wall.start.x, wall.start.y), (wall.end.x, wall.end.y))
            for wall in self.layout.walls
        )
        open_edges = set(self.layout.edges)
        self.connections: list[float] = []
        for row in range(self.rows):
            for column in range(self.columns):
                index = row * self.columns + column
                if column + 1 < self.columns:
                    self.connections.append(float((index, index + 1) in open_edges))
                if row + 1 < self.rows:
                    self.connections.append(float((index, index + self.columns) in open_edges))

    def _sample_spawn(
        self, radius: float, minimum_distance_from: Vector2 | None = None
    ) -> Vector2:
        index = (
            self.layout.agent_spawn
            if minimum_distance_from is None
            else self.layout.threat_spawn
        )
        return self.layout.nodes[index].copy()

    def _map_observation(self) -> list[float]:
        bounds = self.simulation.bounds
        values = [
            value
            for index, node in enumerate(self.layout.nodes)
            for value in (
                (node.x - self.agent.position.x) / bounds.x,
                (node.y - self.agent.position.y) / bounds.y,
                float(index in self.visited),
            )
        ]
        return values + self.connections

    def _movement_bonus(self) -> float:
        reached = {
            index for index, node in enumerate(self.layout.nodes)
            if self.agent.position.distance_squared_to(node) <= self.agent.radius ** 2
        }
        self.new_nodes_visited = len(reached - self.visited)
        self.visited.update(reached)
        return self.node_reward * self.new_nodes_visited

    def _info(self, success: bool) -> dict[str, Any]:
        info = super()._info(success)
        del info["movement_reward_earned"]
        info.update(
            map_seed=self.map_seed,
            nodes_visited=len(self.visited),
            new_nodes_visited=self.new_nodes_visited,
            node_coverage=len(self.visited) / self.node_count,
            exploration_reward=self.new_nodes_visited * self.node_reward,
        )
        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.visited = {self.layout.agent_spawn}
        self.new_nodes_visited = 0
        return super().reset(seed=seed, options=options)

    def render(self) -> None:
        if isinstance(self.simulation, Renderer):
            renderer = self.simulation
            renderer.render(
                draw_overlay=lambda screen: draw_navigation(
                    screen, self.layout, self.visited, renderer.vision_font
                ),
                hud_lines=self._score_hud(),
            )
