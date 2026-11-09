import io
import os
import unittest
from collections import defaultdict, deque
from unittest.mock import patch

import pygame
from pygame import Vector2

from tests import interactive_test
from entities import Action, Agent, Player
from environment import Environment
from map_generation import NavigationMap, generate_map


def graph_distances(layout: NavigationMap, start: int) -> dict[int, int]:
    neighbors: list[list[int]] = [[] for _ in layout.nodes]
    for first, second in layout.edges:
        neighbors[first].append(second)
        neighbors[second].append(first)
    distances = {start: 0}
    pending = deque([start])
    while pending:
        current = pending.popleft()
        for neighbor in neighbors[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                pending.append(neighbor)
    return distances


def snapshot(layout: NavigationMap) -> tuple:
    return (
        tuple((tuple(wall.start), tuple(wall.end)) for wall in layout.walls),
        tuple(tuple(node) for node in layout.nodes),
        layout.edges,
        layout.agent_spawn,
        layout.threat_spawn,
    )


class MapGenerationTests(unittest.TestCase):
    def test_wall_caps_connectivity_and_spawns_across_seeds(self):
        for seed in range(100):
            for cap in (0, 1, 16, 35, 100):
                with self.subTest(seed=seed, cap=cap):
                    layout = generate_map(
                        Vector2(800, 600), agent_radius=10, seed=seed, max_walls=cap
                    )
                    self.assertEqual(len(layout.nodes), 48)
                    self.assertEqual(len(layout.walls), min(cap, 35))
                    self.assertLessEqual(len(layout.walls), cap)
                    self.assertEqual(len(layout.edges) + len(layout.walls), 82)
                    self.assertEqual(len(set(layout.edges)), len(layout.edges))
                    distances = graph_distances(layout, layout.agent_spawn)
                    self.assertEqual(set(distances), set(range(len(layout.nodes))))
                    self.assertNotEqual(layout.agent_spawn, layout.threat_spawn)
                    self.assertEqual(
                        distances[layout.threat_spawn], max(distances.values())
                    )
                    for node in layout.nodes:
                        self.assertTrue(10 < node.x < 790)
                        self.assertTrue(10 < node.y < 590)
                        for wall in layout.walls:
                            edge = wall.end - wall.start
                            fraction = max(
                                0.0, min(1.0, (node - wall.start).dot(edge) / edge.length_squared())
                            )
                            self.assertGreater(
                                node.distance_to(wall.start + fraction * edge), 10
                            )

    def test_every_edge_is_traversable_both_ways_using_core_movement(self):
        env = Environment()
        agent = Agent(Vector2())
        env.entities = [agent]
        for seed in range(12):
            for radius in (10, 49):
                for cap in (0, 16, 100):
                    with self.subTest(seed=seed, radius=radius, cap=cap):
                        layout = generate_map(
                            env.bounds, agent_radius=radius, seed=seed, max_walls=cap
                        )
                        env.walls = layout.walls
                        agent.radius = radius
                        for first, second in layout.edges:
                            for start, end in ((first, second), (second, first)):
                                agent.position = layout.nodes[start].copy()
                                displacement = layout.nodes[end] - agent.position
                                direction = displacement.normalize()
                                env.step(
                                    [Action(round(direction.x), round(direction.y))],
                                    dt=displacement.length() / env.move_speed,
                                )
                                self.assertLess(
                                    agent.position.distance_to(layout.nodes[end]), 1e-7
                                )

    def test_reproducibility_and_seed_variation(self):
        first = generate_map(Vector2(800, 600), agent_radius=10, seed=42)
        again = generate_map(Vector2(800, 600), agent_radius=10, seed=42)
        other = generate_map(Vector2(800, 600), agent_radius=10, seed=43)
        self.assertEqual(snapshot(first), snapshot(again))
        self.assertNotEqual(snapshot(first), snapshot(other))

    def test_non_default_grids_and_bounds(self):
        for columns, rows in ((2, 2), (3, 7), (10, 4)):
            layout = generate_map(
                Vector2(931, 719), agent_radius=20, seed=17,
                columns=columns, rows=rows, max_walls=10_000,
            )
            self.assertEqual(len(layout.nodes), columns * rows)
            self.assertEqual(len(layout.walls), (columns - 1) * (rows - 1))
            self.assertEqual(len(layout.edges), len(layout.nodes) - 1)
            self.assertEqual(len(graph_distances(layout, 0)), len(layout.nodes))

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (
            {"max_walls": -1}, {"max_walls": 1.5}, {"max_walls": True},
            {"seed": -1}, {"seed": False}, {"seed": 0.5},
            {"columns": 1}, {"columns": 2.5}, {"rows": 0}, {"rows": True},
            {"agent_radius": 0}, {"agent_radius": -1},
            {"agent_radius": float("nan")}, {"agent_radius": float("inf")},
            {"agent_radius": 50},
            {"bounds": Vector2(0, 600)}, {"bounds": Vector2(-1, 600)},
            {"bounds": Vector2(float("nan"), 600)},
            {"bounds": Vector2(800, float("inf"))},
        ):
            arguments = {"bounds": Vector2(800, 600), "agent_radius": 10}
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_map(**arguments)


class NavigationPlaygroundTests(unittest.TestCase):
    def test_generated_demo_renders_nodes_and_runs_player_controls(self):
        original_flip = pygame.display.flip
        original_step = Environment.step

        def finish_after_one_frame():
            original_flip()
            pygame.event.post(pygame.event.Event(pygame.QUIT))

        with (
            patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}),
            patch.object(pygame.key, "get_pressed", return_value=defaultdict(bool, {pygame.K_d: True})),
            patch.object(pygame.mouse, "get_pos", return_value=(400, 300)),
            patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)),
            patch.object(pygame.display, "flip", side_effect=finish_after_one_frame),
            patch.object(Environment, "step", autospec=True, side_effect=original_step) as step,
            patch.object(
                interactive_test, "draw_navigation", wraps=interactive_test.draw_navigation
            ) as draw,
            patch("sys.argv", ["interactive_test.py", "--seed", "7", "--max-walls", "3"]),
        ):
            interactive_test.main()
            env = step.call_args.args[0]
            player = next(entity for entity in env.entities if isinstance(entity, Player))
            _, layout, visited, _ = draw.call_args.args
            self.assertEqual(snapshot(layout), snapshot(generate_map(
                env.bounds, agent_radius=player.radius, seed=7, max_walls=3
            )))
            self.assertEqual(visited, {layout.agent_spawn})
            self.assertEqual(len(env.entities), 3)
            self.assertEqual(len(env.walls), 3)
            self.assertGreater(player.position.x, layout.nodes[layout.agent_spawn].x)
            self.assertFalse(env.running)
            self.assertFalse(pygame.display.get_init())

    def test_overlay_colors_visited_and_unvisited_nodes(self):
        pygame.font.init()
        layout = generate_map(Vector2(800, 600), agent_radius=10)
        surface = pygame.Surface((800, 600))
        visited = {0, 1}
        interactive_test.draw_navigation(
            surface, layout, visited, pygame.font.Font(None, 24)
        )
        for index, node in enumerate(layout.nodes):
            expected = (100, 235, 150) if index in visited else (230, 190, 80)
            self.assertEqual(surface.get_at((round(node.x), round(node.y)))[:3], expected)
        self.assertEqual(visited, {0, 1})

    def test_cli_rejects_negative_limit_and_fixed_layout_over_budget(self):
        for args in (["--max-walls", "-1"], ["--fixed-layout", "--max-walls", "1"]):
            with (
                self.subTest(args=args),
                patch("sys.argv", ["interactive_test.py", *args]),
                patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit) as raised,
            ):
                interactive_test.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
