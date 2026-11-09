"""Seeded walls and connected navigation shared by playground and task wrappers."""

import math
import random
from collections import deque
from dataclasses import dataclass

from pygame import Vector2

from entities import Wall


@dataclass
class NavigationMap:
    walls: list[Wall]
    nodes: tuple[Vector2, ...]
    edges: tuple[tuple[int, int], ...]
    agent_spawn: int
    threat_spawn: int


def generate_map(
    bounds: Vector2,
    *,
    agent_radius: float,
    seed: int = 0,
    max_walls: int = 16,
    columns: int = 8,
    rows: int = 6,
) -> NavigationMap:
    """Keep a random spanning tree open, then wall off other grid connections.

    The cap counts individual interior wall segments, not the arena boundary.
    Cell centers and every open edge have more than agent_radius clearance.
    """
    for name, value, minimum in (
        ("seed", seed, 0),
        ("max_walls", max_walls, 0),
        ("columns", columns, 2),
        ("rows", rows, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not all(math.isfinite(value) and value > 0 for value in bounds):
        raise ValueError("bounds must be finite and positive.")
    if not math.isfinite(agent_radius) or agent_radius <= 0:
        raise ValueError("agent_radius must be finite and positive.")
    width, height = bounds.x / columns, bounds.y / rows
    if min(width, height) <= 2 * agent_radius:
        raise ValueError("Grid cells must be wider and taller than the agent diameter.")

    rng = random.Random(seed)
    nodes = tuple(
        Vector2((column + 0.5) * width, (row + 0.5) * height)
        for row in range(rows)
        for column in range(columns)
    )
    connections: list[tuple[int, int, Wall]] = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if column + 1 < columns:
                x = (column + 1) * width
                connections.append((
                    index,
                    index + 1,
                    Wall(Vector2(x, row * height), Vector2(x, (row + 1) * height)),
                ))
            if row + 1 < rows:
                y = (row + 1) * height
                connections.append((
                    index,
                    index + columns,
                    Wall(Vector2(column * width, y), Vector2((column + 1) * width, y)),
                ))

    parents = list(range(len(nodes)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    order = list(range(len(connections)))
    rng.shuffle(order)
    optional_walls: list[int] = []
    for index in order:
        first, second, _ = connections[index]
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[first_root] = second_root
        else:
            optional_walls.append(index)

    blocked = set(rng.sample(optional_walls, min(max_walls, len(optional_walls))))
    walls = [wall for index, (_, _, wall) in enumerate(connections) if index in blocked]
    edges = tuple(
        (first, second)
        for index, (first, second, _) in enumerate(connections)
        if index not in blocked
    )
    neighbors: list[list[int]] = [[] for _ in nodes]
    for first, second in edges:
        neighbors[first].append(second)
        neighbors[second].append(first)

    agent_spawn = rng.randrange(len(nodes))
    distances = {agent_spawn: 0}
    pending = deque([agent_spawn])
    while pending:
        current = pending.popleft()
        for neighbor in neighbors[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                pending.append(neighbor)
    furthest = max(distances.values())
    threat_spawn = rng.choice([
        index for index, distance in distances.items() if distance == furthest
    ])
    return NavigationMap(walls, nodes, edges, agent_spawn, threat_spawn)
