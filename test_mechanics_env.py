import unittest

import gymnasium as gym
import numpy as np
from pygame import Vector2
from stable_baselines3.common.env_checker import check_env

from mechanics_env import MechanicsEnv


class MechanicsEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = MechanicsEnv()
        self.addCleanup(self.env.close)

    def test_gymnasium_and_sb3_contracts(self):
        check_env(self.env, skip_render_check=True)

    def test_reset_uses_seeded_random_spawns_and_heading(self):
        observation, info = self.env.reset(seed=12)
        first_agent_position = self.env.agent.position.copy()
        first_target_position = self.env.target.position.copy()
        first_heading = (self.env.agent.aim_x, self.env.agent.aim_y)
        self.env.reset(seed=12)
        self.assertEqual(len(self.env.simulation.walls), 2)
        self.assertEqual(self.env.agent.position, first_agent_position)
        self.assertEqual(self.env.target.position, first_target_position)
        self.assertEqual((self.env.agent.aim_x, self.env.agent.aim_y), first_heading)
        self.assertGreaterEqual(
            self.env.agent.position.distance_to(self.env.target.position),
            self.env.spawn_separation,
        )
        for entity in (self.env.agent, self.env.target):
            for wall in self.env.simulation.walls:
                self.assertGreaterEqual(
                    self.env._distance_to_wall(entity.position, wall), entity.radius
                )
        self.assertAlmostEqual(
            Vector2(self.env.agent.aim_x, self.env.agent.aim_y).length(), 1
        )
        self.assertEqual(observation.shape, (17,))
        self.assertTrue(self.env.observation_space.contains(observation))

    def test_target_is_observed_when_inside_cone_and_clear_of_walls(self):
        self.env.reset(seed=12)
        self.env.agent.position = Vector2(600, 150)
        self.env.target.position = Vector2(400, 150)
        self.env.agent.aim_x, self.env.agent.aim_y = -1, 0
        observation = self.env._observation()
        self.assertTrue(observation[6])
        self.assertNotEqual(observation[4:6].tolist(), [0.0, 0.0])

    def test_all_action_fields_reach_the_core_simulation(self):
        self.env.reset(seed=1)
        self.env.agent.position = Vector2(200, 300)
        self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
        action = np.array([4, 4, 0], dtype=np.int64)
        self.env.step(action)
        self.assertAlmostEqual(self.env.agent.position.x, 200 + 200 / 60)
        self.assertAlmostEqual(self.env.agent.position.y, 300)
        self.assertGreater(self.env.agent.aim_y, 0)
        self.assertEqual(self.env.shots, 0)

    def test_walls_block_movement_and_shots_using_core_mechanics(self):
        self.env.reset(seed=1)
        self.env.agent.position = Vector2(400, 212)
        self.env.target.position = Vector2(400, 150)
        self.env.agent.aim_x, self.env.agent.aim_y = 0, -1
        self.env.step(np.array([1, 2, 1], dtype=np.int64))
        self.assertAlmostEqual(self.env.agent.position.y, 210)
        self.assertIn(self.env.target, self.env.simulation.entities)
        self.assertEqual(self.env.shots, 1)

    def test_visible_target_can_be_eliminated_and_ends_episode(self):
        self.env.reset(seed=1)
        self.env.agent.position = Vector2(600, 150)
        self.env.target.position = Vector2(400, 150)
        self.env.agent.aim_x, self.env.agent.aim_y = -1, 0
        self.env.movement_reward_earned = True
        observation, reward, terminated, truncated, info = self.env.step(
            np.array([0, 2, 1], dtype=np.int64)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 1 - 0.001)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["shots"], 1)
        self.assertFalse(observation[6])
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.step(np.zeros(3, dtype=np.int64))

    def test_timeout_truncates_episode(self):
        env = MechanicsEnv(max_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=1)
        _, reward, terminated, truncated, info = env.step(
            np.array([0, 2, 0], dtype=np.int64)
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, -env.step_penalty)
        self.assertFalse(info["is_success"])

    def test_movement_reward_is_earned_once_after_crossing_threshold(self):
        self.env.reset(seed=1)
        self.env.agent_spawn = Vector2(200, 300)
        self.env.agent.position = self.env.agent_spawn.copy()
        action = np.array([4, 2, 0], dtype=np.int64)
        for _ in range(22):
            _, reward, _, _, _ = self.env.step(action)
            self.assertEqual(reward, -self.env.step_penalty)

        _, reward, _, _, info = self.env.step(action)
        self.assertAlmostEqual(
            reward, self.env.movement_reward - self.env.step_penalty
        )
        self.assertTrue(info["movement_reward_earned"])

        _, reward, _, _, info = self.env.step(action)
        self.assertEqual(reward, -self.env.step_penalty)
        self.assertTrue(info["movement_reward_earned"])

    def test_missed_shot_has_small_penalty(self):
        self.env.reset(seed=1)
        _, reward, _, _, _ = self.env.step(
            np.array([0, 2, 1], dtype=np.int64)
        )
        self.assertAlmostEqual(
            reward, -self.env.missed_shot_penalty - self.env.step_penalty
        )


if __name__ == "__main__":
    unittest.main()
