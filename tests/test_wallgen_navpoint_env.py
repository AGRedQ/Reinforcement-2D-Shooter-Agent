import os
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from pygame import Vector2
from stable_baselines3.common.env_checker import check_env

from entities import has_line_of_sight
from mechanics_env import MechanicsEnv
from wallgen_navpoint_env import WallgenNavpointEnv


class WallgenNavpointTests(unittest.TestCase):
    def setUp(self):
        self.env = WallgenNavpointEnv()
        self.addCleanup(self.env.close)

    def suppress_combat_bonuses(self):
        self.env.target_visibility_reward_earned = True
        self.env.target_alignment_reward_earned = True

    def test_sb3_contract_and_exact_observation_layout(self):
        check_env(self.env)
        observation, info = self.env.reset(seed=0)
        self.assertEqual(observation.shape, (66,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(observation))
        nodes = observation[13:49].reshape(12, 3)
        for index, node in enumerate(self.env.layout.nodes):
            np.testing.assert_allclose(nodes[index, :2], (
                (node.x - self.env.agent.position.x) / 800,
                (node.y - self.env.agent.position.y) / 600,
            ))
            self.assertEqual(nodes[index, 2], index == self.env.layout.agent_spawn)
        expected = []
        edges = set(self.env.layout.edges)
        for row in range(3):
            for column in range(4):
                index = row * 4 + column
                if column < 3:
                    expected.append(float((index, index + 1) in edges))
                if row < 2:
                    expected.append(float((index, index + 4) in edges))
        np.testing.assert_array_equal(observation[49:], expected)
        self.assertEqual(info["nodes_visited"], 1)
        self.assertEqual(info["exploration_reward"], 0)
        self.assertNotIn("movement_reward_earned", info)

    def test_fixed_map_fixed_spawns_random_seeded_heading_and_larger_walls(self):
        first, _ = self.env.reset(seed=1)
        walls = [(wall.start.copy(), wall.end.copy()) for wall in self.env.simulation.walls]
        self.assertEqual(len(walls), 4)
        for start, end in walls:
            self.assertEqual(start.distance_to(end), 200)
        self.env.visited.update(range(12))
        second, _ = self.env.reset(seed=2)
        self.assertEqual(self.env.visited, {self.env.layout.agent_spawn})
        self.assertEqual(self.env.agent.position, self.env.layout.nodes[self.env.layout.agent_spawn])
        self.assertEqual(self.env.target.position, self.env.layout.nodes[self.env.layout.threat_spawn])
        self.assertEqual(walls, [(wall.start, wall.end) for wall in self.env.simulation.walls])
        self.assertFalse(np.array_equal(first[2:4], second[2:4]))
        repeated, _ = self.env.reset(seed=1)
        np.testing.assert_array_equal(first, repeated)

    def test_visit_updates_observation_reward_once_and_never_earns_spawn_bonus(self):
        self.env.reset(seed=0)
        self.suppress_combat_bonuses()
        action = np.array([0, 2, 0])
        _, reward, _, _, info = self.env.step(action)
        self.assertAlmostEqual(reward, -self.env.step_penalty)
        self.assertEqual(info["new_nodes_visited"], 0)
        index = next(index for index in range(12) if index not in self.env.visited)
        node = self.env.layout.nodes[index]
        self.env.agent.position = node + Vector2(self.env.agent.radius + 0.01, 0)
        _, reward, _, _, _ = self.env.step(action)
        self.assertAlmostEqual(reward, -self.env.step_penalty)
        self.env.agent.position = node + Vector2(self.env.agent.radius, 0)
        observation, reward, _, _, info = self.env.step(action)
        self.assertEqual(observation[13 + 3 * index + 2], 1)
        self.assertAlmostEqual(reward, self.env.node_reward - self.env.step_penalty)
        self.assertEqual(info["new_nodes_visited"], 1)
        self.assertAlmostEqual(info["exploration_reward"], self.env.node_reward)
        self.env.agent.position = self.env.agent_spawn.copy()
        self.env.step(action)
        self.env.agent.position = node.copy()
        _, reward, _, _, info = self.env.step(action)
        self.assertAlmostEqual(reward, -self.env.step_penalty)
        self.assertEqual(info["new_nodes_visited"], 0)

    def test_total_exploration_budget_and_reset(self):
        self.env.reset(seed=0)
        self.suppress_combat_bonuses()
        total = 0.0
        for node in self.env.layout.nodes:
            self.env.agent.position = node.copy()
            _, reward, _, _, _ = self.env.step(np.array([0, 2, 0]))
            total += reward + self.env.step_penalty
        self.assertAlmostEqual(total, 0.2)
        self.assertLess(total, self.env.hit_reward)
        self.assertEqual(len(self.env.visited), 12)
        observation, info = self.env.reset(seed=0)
        self.assertEqual(info["nodes_visited"], 1)
        self.assertEqual(info["exploration_reward"], 0)
        self.assertEqual(sum(observation[15:49:3]), 1)

    def test_hidden_target_is_not_leaked_through_navigation(self):
        self.env.reset(seed=0)
        self.env.agent.position = Vector2(100, 100)
        self.env.agent.aim_x, self.env.agent.aim_y = -1, 0
        self.env.last_seen_target_position = None
        self.env.target.position = Vector2(600, 300)
        first = self.env._observation()
        self.env.target.position = Vector2(700, 500)
        second = self.env._observation()
        self.assertEqual(first[6], 0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[4:6], [0, 0])

    def test_relative_node_offsets_follow_movement(self):
        before, _ = self.env.reset(seed=0)
        self.env.agent.position += Vector2(20, 10)
        after = self.env._observation()
        np.testing.assert_allclose(
            after[13:49].reshape(12, 3)[:, :2],
            before[13:49].reshape(12, 3)[:, :2] - [20 / 800, 10 / 600],
            atol=1e-7,
        )
        np.testing.assert_array_equal(before[49:], after[49:])

    def test_generated_walls_use_shared_collision_and_beam_blocking(self):
        self.env.reset(seed=0)
        wall = self.env.simulation.walls[0]
        midpoint = (wall.start + wall.end) / 2
        normal = (wall.end - wall.start).rotate(90).normalize()
        self.env.agent.position = midpoint + normal * 12
        self.env.target.position = midpoint - normal * 50
        self.env.agent.aim_x, self.env.agent.aim_y = -normal.x, -normal.y
        movement = self.env.movement_actions.index((round(-normal.x), round(-normal.y)))
        _, _, terminated, _, _ = self.env.step(np.array([movement, 2, 1]))
        self.assertFalse(terminated)
        self.assertAlmostEqual(self.env._distance_to_wall(self.env.agent.position, wall), 10)
        self.assertLess(self.env.agent.shot_end.distance_to(midpoint), 1e-7)
        self.assertFalse(has_line_of_sight(
            self.env.agent.position, self.env.target.position, self.env.simulation.walls
        ))

    def test_hit_and_timeout_preserve_episode_semantics(self):
        self.env.reset(seed=0)
        self.env.agent.position = Vector2(50, 50)
        self.env.target.position = Vector2(150, 50)
        self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
        _, reward, terminated, truncated, info = self.env.step(np.array([0, 2, 1]))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertAlmostEqual(reward, self.env.hit_reward - self.env.step_penalty)
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.step(np.array([0, 2, 0]))
        env = WallgenNavpointEnv(max_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(np.array([0, 2, 0]))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_invalid_map_options_fail_and_original_task_stays_separate(self):
        for map_seed, max_walls in ((-1, 4), (0, -1), (0, True)):
            with self.subTest(map_seed=map_seed, max_walls=max_walls), self.assertRaises(ValueError):
                WallgenNavpointEnv(map_seed=map_seed, max_walls=max_walls)
        original = MechanicsEnv()
        self.addCleanup(original.close)
        observation, _ = original.reset(seed=0)
        self.assertEqual(observation.shape, (21,))
        self.assertEqual(len(original.simulation.walls), 2)

    def test_human_render_draws_shared_navigation_overlay(self):
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            env = WallgenNavpointEnv(render_mode="human")
        try:
            env.reset(seed=0)
            env.render()
            self.assertEqual(env.simulation.step_count, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
