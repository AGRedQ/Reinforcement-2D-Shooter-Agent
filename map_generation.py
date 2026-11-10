"""Seeded arena generation and CLI map-seed selection.

Algorithm: stratified random rectangular obstacles. Choose 6-9 of nine grid
slots, then randomize each rectangle's size and position inside its slot.
A 48-unit slot margin leaves at least 96 units between obstacles and 48 units
to the boundary walls, keeping free space connected for the default bodies.
There is no maze solver, rejection loop, or additional dependency.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from entities import Wall
from environments import MapLayout, boundary_walls


MAP_SEEDS_HELP = (
    "map seed cycle: N plays 0..N inclusive; '[N]' plays only N; "
    "'[N,M,...]' loops those seeds in order (default: 9, meaning 0..9)"
)


def parse_map_seeds(value: str) -> range | tuple[int, ...]:
    try:
        seeds = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or a JSON array, e.g. 5 or '[2,7,11]'.") from error
    if type(seeds) is int and seeds >= 0:
        return range(seeds + 1)
    if isinstance(seeds, list) and seeds and all(type(seed) is int and seed >= 0 for seed in seeds):
        return tuple(seeds)
    raise argparse.ArgumentTypeError("Map seeds must be a nonnegative integer or a nonempty array of nonnegative integers.")


def generate_map(seed: int) -> MapLayout:
    if type(seed) is not int or seed < 0:
        raise ValueError("Map seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    width, height, thickness = 800.0, 800.0, 20.0
    walls = boundary_walls(width, height, thickness)
    slot_size = (width - 2 * thickness) / 3
    margin = 48.0
    slots = rng.choice(9, size=int(rng.integers(6, 10)), replace=False)
    for slot in slots:
        wall_width, wall_height = rng.uniform(40.0, 150.0, size=2)
        origin = thickness + np.array([slot % 3, slot // 3]) * slot_size
        low = origin + margin
        high = origin + slot_size - margin - np.array([wall_width, wall_height])
        x, y = rng.uniform(low, high)
        walls.append(Wall(float(x), float(y), float(wall_width), float(wall_height)))
    return MapLayout(width, height, walls)
