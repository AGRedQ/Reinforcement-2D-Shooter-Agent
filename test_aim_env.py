import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pygame
import torch
from gymnasium.utils.env_checker import check_env as check_gym_env
from pygame import Vector2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from aim_env import AimEnv
from train import evaluate, evaluate_policy, main, scripted_action


class AimEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = AimEnv()
        self.addCleanup(self.env.close)

    def test_gymnasium_and_sb3_contracts(self):
        check_gym_env(self.env, skip_render_check=True)
        check_env(self.env)

    def test_headless_environment_does_not_initialize_display(self):
        with patch.object(pygame.display, "set_mode") as create_window:
            env = AimEnv()
            try:
                env.reset(seed=4)
                env.step(np.zeros(3, dtype=np.float32))
                env.render()
            finally:
                env.close()
        create_window.assert_not_called()

    def test_reset_is_seeded_and_restores_all_episode_state(self):
        observation, info = self.env.reset(seed=42)
        initial_position = self.env.target.position.copy()
        old_target = self.env.target
        self.env.simulation.entities.remove(old_target)
        self.env.agent.shoot_cooldown = 0.2
        self.env.agent.beam_remaining = 1
        self.env.agent.shot_end = Vector2(800, 600)
        self.env.simulation.step_count = 299
        self.env.shots = 10
        repeated, repeated_info = self.env.reset(seed=42)
        np.testing.assert_array_equal(observation, repeated)
        self.assertEqual(info, repeated_info)
        self.assertIsNot(self.env.target, old_target)
        self.assertEqual(self.env.target.position, initial_position)
        self.assertEqual(self.env.simulation.entities, [self.env.agent, self.env.target])
        self.assertEqual(self.env.agent.shoot_cooldown, 0)
        self.assertEqual(self.env.agent.beam_remaining, 0)
        self.assertEqual(self.env.agent.shot_start, self.env.agent.position)
        self.assertEqual(self.env.agent.shot_end, self.env.agent.position)
        different, _ = self.env.reset(seed=43)
        self.assertFalse(np.array_equal(observation, different))

    def test_spawn_is_inside_bounds_separated_and_aim_is_unit_length(self):
        for seed in range(100):
            self.env.reset(seed=seed)
            target = self.env.target
            self.assertGreaterEqual(target.position.distance_to(self.env.agent.position), 80)
            self.assertGreaterEqual(target.position.x, target.radius)
            self.assertGreaterEqual(target.position.y, target.radius)
            self.assertLessEqual(target.position.x, self.env.simulation.bounds.x - target.radius)
            self.assertLessEqual(target.position.y, self.env.simulation.bounds.y - target.radius)
            self.assertAlmostEqual(Vector2(self.env.agent.aim_x, self.env.agent.aim_y).length(), 1)

    def test_stationary_task_and_observation_fields(self):
        self.env.reset(seed=12)
        agent_position = self.env.agent.position.copy()
        target_position = self.env.target.position.copy()
        observation, _, terminated, truncated, _ = self.env.step(
            np.array([0, 1, -1], dtype=np.float32)
        )
        self.assertEqual(self.env.agent.position, agent_position)
        self.assertEqual(self.env.target.position, target_position)
        self.assertFalse(terminated or truncated)
        offset = (target_position - agent_position) / self.env.simulation.bounds.length()
        np.testing.assert_allclose(observation[:2], [offset.x, offset.y])
        np.testing.assert_allclose(observation[2:4], [self.env.agent.aim_x, self.env.agent.aim_y])
        self.assertEqual(observation[4], 0)
        self.assertAlmostEqual(float(observation[5]), 299 / 300)
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(observation))

    def test_hit_reward_and_success_on_last_step_take_priority_over_timeout(self):
        env = AimEnv(max_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=1)
        env.agent.aim_x, env.agent.aim_y = 1, 0
        env.target.position = env.agent.position + Vector2(100, 0)
        observation, reward, terminated, truncated, info = env.step(
            np.array([1, 0, 1], dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 1 - 0.01 - 0.001 + 0.0005)
        self.assertEqual(info["shots"], 1)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["elapsed_time"], 1 / 60)
        self.assertEqual(observation[5], 0)

    def test_penalizes_actual_shots_not_fire_requests_during_cooldown(self):
        self.env.reset(seed=1)
        self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
        self.env.target.position = self.env.agent.position + Vector2(0, 100)
        action = np.array([1, 0, 1], dtype=np.float32)
        observation, reward, _, _, info = self.env.step(action)
        self.assertAlmostEqual(reward, -0.01075)
        self.assertEqual(observation[4], 1)
        self.assertEqual(info["shots"], 1)
        _, reward, _, _, info = self.env.step(action)
        self.assertAlmostEqual(reward, -0.00075)
        self.assertEqual(info["shots"], 1)
        self.env.agent.shoot_cooldown = self.env.dt
        _, reward, _, _, info = self.env.step(action)
        self.assertAlmostEqual(reward, -0.01075)
        self.assertEqual(info["shots"], 2)

    def test_timeout_and_reset_required(self):
        with self.assertRaises(gym.error.ResetNeeded):
            self.env.step(np.zeros(3, dtype=np.float32))
        env = AimEnv(max_steps=2)
        self.addCleanup(env.close)
        _, info = env.reset(seed=1)
        env.agent.aim_x, env.agent.aim_y = 1, 0
        env.target.position = env.agent.position + Vector2(-100, 0)
        for index in range(2):
            _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == 1)
            self.assertAlmostEqual(reward, -0.001)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["shots"], 0)
        self.assertEqual(info["elapsed_time"], 2 / 60)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(np.zeros(3, dtype=np.float32))
        env.reset()
        env.step(np.zeros(3, dtype=np.float32))

    def test_invalid_actions_fail_without_advancing(self):
        self.env.reset(seed=1)
        for action in ([0, 0], [0, 0, 0, 0], [2, 0, 0], [0, np.nan, 0], [0, 0, np.inf]):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    self.env.step(np.array(action, dtype=np.float32))
                self.assertEqual(self.env.simulation.step_count, 0)
                self.assertEqual(self.env.shots, 0)

    def test_diagnostics_measure_post_turn_heading_and_actual_shots(self):
        self.env.reset(seed=1)
        self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
        angle = 90 * -math.expm1(-10 * self.env.dt)
        self.env.target.position = self.env.agent.position + Vector2(100, 0).rotate(angle)
        _, reward, terminated, _, info = self.env.step(
            np.array([0, 1, 1], dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward, 0.9895)
        self.assertAlmostEqual(info["angular_error_sum_degrees"], 0)
        self.assertEqual(info["aligned_steps"], 1)
        self.assertEqual(info["aligned_shots"], 1)
        self.assertEqual(info["ready_aligned_steps"], 1)
        self.assertEqual(info["missed_fire_opportunities"], 0)

    def test_diagnostics_distinguish_cooldown_from_missed_opportunity(self):
        self.env.reset(seed=1)
        self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
        self.env.target.position = self.env.agent.position + Vector2(100, 0)
        _, _, _, _, info = self.env.step(np.zeros(3, dtype=np.float32))
        self.assertEqual(info["aligned_steps"], 1)
        self.assertEqual(info["missed_fire_opportunities"], 1)
        self.env.agent.shoot_cooldown = 0.2
        _, reward, _, _, info = self.env.step(np.array([0, 0, 1], dtype=np.float32))
        self.assertAlmostEqual(reward, -0.0005)
        self.assertEqual(info["aligned_steps"], 2)
        self.assertEqual(info["aligned_shots"], 0)
        self.assertEqual(info["shots"], 0)
        self.assertEqual(info["ready_aligned_steps"], 1)
        self.assertEqual(info["missed_fire_opportunities"], 1)
        _, info = self.env.reset(seed=1)
        for key in ("angular_error_sum_degrees", "aligned_steps", "aligned_shots",
                    "ready_aligned_steps", "missed_fire_opportunities"):
            self.assertEqual(info[key], 0)

    def test_alignment_uses_target_radius_distance_and_forward_direction(self):
        for offset, expected_aligned, expected_error in (
            ((100, 10), True, math.degrees(math.atan2(10, 100))),
            ((100, 11), False, math.degrees(math.atan2(11, 100))),
            ((-100, 0), False, 180),
            ((0, 100), False, 90),
        ):
            with self.subTest(offset=offset):
                self.env.reset(seed=1)
                self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
                self.env.target.position = self.env.agent.position + Vector2(offset)
                _, _, terminated, _, info = self.env.step(
                    np.array([0, 0, 1], dtype=np.float32)
                )
                self.assertEqual(terminated, expected_aligned)
                self.assertEqual(info["aligned_steps"], int(expected_aligned))
                self.assertEqual(info["aligned_shots"], int(expected_aligned))
                self.assertAlmostEqual(info["angular_error_sum_degrees"], expected_error)

    def test_aim_bonus_increases_with_accuracy_not_physical_distance(self):
        for distance in (100, 200):
            for angle, bonus in ((0, 0.0005), (45, 0.000375), (90, 0.00025), (180, 0)):
                with self.subTest(distance=distance, angle=angle):
                    self.env.reset(seed=1)
                    self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
                    self.env.agent.shoot_cooldown = 0.2
                    self.env.target.position = (
                        self.env.agent.position + Vector2(distance, 0).rotate(angle)
                    )
                    _, reward, terminated, truncated, _ = self.env.step(
                        np.zeros(3, dtype=np.float32)
                    )
                    self.assertAlmostEqual(reward, -0.001 + bonus)
                    self.assertFalse(terminated or truncated)

    def test_missed_opportunity_penalty_only_when_aligned_ready_and_not_firing(self):
        for fire, cooldown, offset, expected in (
            (-1, 0, (100, 0), -0.0025),
            (0, 0, (100, 0), -0.0025),
            (1, 0, (100, 0), 0.9895),
            (-1, 0.2, (100, 0), -0.0005),
            (1, 0.2, (100, 0), -0.0005),
            (-1, 0, (0, 100), -0.00075),
            (1, 0, (0, 100), -0.01075),
        ):
            with self.subTest(fire=fire, cooldown=cooldown, offset=offset):
                self.env.reset(seed=1)
                self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
                self.env.agent.shoot_cooldown = cooldown
                self.env.target.position = self.env.agent.position + Vector2(offset)
                _, reward, _, _, _ = self.env.step(
                    np.array([0, 0, fire], dtype=np.float32)
                )
                self.assertAlmostEqual(reward, expected)

    def test_perfect_aim_without_firing_still_has_negative_episode_return(self):
        self.env.reset(seed=1)
        self.env.agent.aim_x, self.env.agent.aim_y = 1, 0
        self.env.target.position = self.env.agent.position + Vector2(100, 0)
        total = 0.0
        terminated = truncated = False
        for _ in range(300):
            _, reward, terminated, truncated, _ = self.env.step(
                np.zeros(3, dtype=np.float32)
            )
            self.assertLess(reward, 0)
            total += reward
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(total, -0.75)

    def test_shaping_uses_post_turn_alignment_on_timeout_step(self):
        env = AimEnv(max_steps=1)
        self.addCleanup(env.close)
        env.reset(seed=1)
        env.agent.aim_x, env.agent.aim_y = 1, 0
        angle = 90 * -math.expm1(-10 * env.dt)
        env.target.position = env.agent.position + Vector2(100, 0).rotate(angle)
        _, reward, terminated, truncated, info = env.step(
            np.array([0, 1, -1], dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, -0.0025)
        self.assertEqual(info["missed_fire_opportunities"], 1)

    def test_invalid_constructor_arguments(self):
        for max_steps in (0, -1, True):
            with self.subTest(max_steps=max_steps):
                with self.assertRaises(ValueError):
                    AimEnv(max_steps=max_steps)
        with self.assertRaises(ValueError):
            AimEnv(render_mode="rgb_array")

    def test_config_loading_does_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                env = AimEnv()
                try:
                    env.reset(seed=5)
                finally:
                    env.close()
            finally:
                os.chdir(previous)

    def test_human_rendering_uses_same_simulation_and_closes(self):
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            env = AimEnv(render_mode="human")
        try:
            env.reset(seed=1)
            env.render()
            env.step(scripted_action(env._observation()))
            self.assertTrue(pygame.display.get_init())
        finally:
            env.close()
        self.assertFalse(pygame.display.get_init())


class WatchTests(unittest.TestCase):
    def run_watch(self, *arguments: str, close_early: bool = False):
        torch.set_num_threads(1)
        training_env = AimEnv()
        self.addCleanup(training_env.close)
        model = PPO("MlpPolicy", training_env, n_steps=8, batch_size=4, seed=5)
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            env = AimEnv(render_mode="human", max_steps=3)
        self.addCleanup(env.close)
        actions = []
        predict = model.predict
        original_render = env.render

        def record_prediction(observation, *, deterministic):
            action, state = predict(observation, deterministic=deterministic)
            actions.append((deterministic, action.copy()))
            return action, state

        def render():
            if close_early:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            original_render()

        rng_state = torch.get_rng_state().clone()
        with (
            patch.object(sys, "argv", ["train.py", "watch", "--episodes", "2", *arguments]),
            patch("train.PPO.load", return_value=model),
            patch("train.AimEnv", return_value=env),
            patch.object(model, "predict", side_effect=record_prediction),
            patch.object(env, "render", side_effect=render),
            patch.object(env, "reset", wraps=env.reset) as reset,
        ):
            main()
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        self.assertFalse(pygame.display.get_init())
        return actions, reset.call_args_list

    def test_default_watch_is_deterministic_and_scenario_seeds_are_preserved(self):
        actions, resets = self.run_watch()
        self.assertTrue(actions)
        self.assertTrue(all(deterministic for deterministic, _ in actions))
        self.assertEqual([call.kwargs["seed"] for call in resets], [10000, 10001])
        different_seed, _ = self.run_watch("--sampling-seed", "9")
        np.testing.assert_array_equal(
            np.stack([action for _, action in actions]),
            np.stack([action for _, action in different_seed]),
        )

    def test_stochastic_watch_is_repeatable_and_sampling_seed_changes_actions(self):
        first, first_resets = self.run_watch("--stochastic", "--sampling-seed", "7")
        repeated, _ = self.run_watch("--stochastic", "--sampling-seed", "7")
        different, different_resets = self.run_watch("--stochastic", "--sampling-seed", "8")
        self.assertTrue(first)
        self.assertTrue(all(not deterministic for deterministic, _ in first))
        np.testing.assert_array_equal(
            np.stack([action for _, action in first]),
            np.stack([action for _, action in repeated]),
        )
        self.assertFalse(np.array_equal(first[0][1], different[0][1]))
        self.assertEqual(first_resets, different_resets)

    def test_watch_closes_before_predicting_when_window_quits(self):
        actions, _ = self.run_watch("--stochastic", close_early=True)
        self.assertEqual(actions, [])

    def test_watch_rejects_negative_sampling_seed_before_loading_model(self):
        with (
            patch.object(sys, "argv", ["train.py", "watch", "--sampling-seed", "-1"]),
            patch("train.PPO.load") as load,
            patch("sys.stderr"),
        ):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()


class TrainingTests(unittest.TestCase):
    def test_scripted_baseline_hits_all_held_out_scenarios_with_one_shot(self):
        report = evaluate_policy(scripted_action, episodes=100, seed=10_000)
        self.assertEqual(report["success_rate"], 1)
        self.assertEqual(report["mean_shots_on_success"], 1)
        hit_time = report["mean_time_to_hit_seconds"]
        assert hit_time is not None
        self.assertLess(hit_time, 1)
        self.assertEqual(report["aligned_shot_fraction"], 1)
        self.assertEqual(report["total_shots"], 100)
        self.assertEqual(report["aligned_shots"], 100)
        self.assertEqual(report["episodes_without_shots"], 0)
        aligned_steps = report["aligned_steps"]
        total_steps = report["total_steps"]
        assert aligned_steps is not None and total_steps is not None
        self.assertEqual(report["aligned_step_fraction"], aligned_steps / total_steps)

    def test_failed_policy_metrics_are_explicit_and_repeatable(self):
        policy = lambda _observation: np.zeros(3, dtype=np.float32)
        first = evaluate_policy(policy, episodes=2, seed=10_000)
        self.assertEqual(first, evaluate_policy(policy, episodes=2, seed=10_000))
        self.assertEqual(first["success_rate"], 0)
        self.assertEqual(first["mean_shots"], 0)
        self.assertIsNone(first["mean_shots_on_success"])
        self.assertIsNone(first["mean_time_to_hit_seconds"])
        self.assertIsNone(first["aligned_shot_fraction"])
        self.assertEqual(first["episodes_without_shots"], 2)
        self.assertEqual(first["no_shot_episode_fraction"], 1)
        self.assertEqual(first["total_steps"], 600)
        self.assertEqual(first["total_shots"], 0)

    def test_empty_firing_opportunities_are_null_and_metrics_are_step_weighted(self):
        class FacingAwayEnv(AimEnv):
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed, options=options)
                self.agent.aim_x, self.agent.aim_y = 1, 0
                self.target.position = self.agent.position + Vector2(-100, 0)
                return self._observation(), self._info(False)

        with patch("train.AimEnv", FacingAwayEnv):
            report = evaluate_policy(
                lambda _obs: np.zeros(3, dtype=np.float32), episodes=2
            )
        self.assertEqual(report["mean_angular_error_degrees"], 180)
        self.assertEqual(report["aligned_step_fraction"], 0)
        self.assertEqual(report["ready_aligned_steps"], 0)
        self.assertIsNone(report["missed_fire_opportunity_fraction"])

    def test_aggregation_weights_steps_not_episode_means(self):
        class UnequalEpisodesEnv(AimEnv):
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed, options=options)
                assert seed is not None
                self.max_steps = 1 if seed % 2 == 0 else 2
                self.agent.aim_x, self.agent.aim_y = 1, 0
                self.target.position = self.agent.position + Vector2(
                    -100 if self.max_steps == 1 else 100, 0
                )
                return self._observation(), self._info(False)

        with patch("train.AimEnv", UnequalEpisodesEnv):
            report = evaluate_policy(
                lambda _obs: np.zeros(3, dtype=np.float32), episodes=2
            )
        self.assertEqual(report["total_steps"], 3)
        self.assertEqual(report["mean_angular_error_degrees"], 60)
        self.assertEqual(report["aligned_step_fraction"], 2 / 3)
        self.assertEqual(report["missed_fire_opportunity_fraction"], 1)

    def test_ppo_learn_save_load_and_evaluate_round_trip(self):
        torch.set_num_threads(1)
        env = AimEnv(max_steps=8)
        try:
            model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=1, seed=0)
            model.learn(total_timesteps=32)
            observation, _ = env.reset(seed=8)
            expected, _ = model.predict(observation, deterministic=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "smoke.zip"
                model.save(path)
                restored = PPO.load(path, device="cpu")
                actual, _ = restored.predict(observation, deterministic=True)
                np.testing.assert_array_equal(expected, actual)
                report = evaluate_policy(
                    lambda obs: restored.predict(obs, deterministic=True)[0],
                    episodes=2,
                )
                self.assertEqual(report["episodes"], 2)
                rng_state = torch.get_rng_state().clone()
                first = evaluate(restored, episodes=2, seed=30, sampling_seed=7)
                self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
                self.assertEqual(
                    first, evaluate(restored, episodes=2, seed=30, sampling_seed=7)
                )
                self.assertIn("ppo_stochastic", first)
                self.assertEqual(first["sampling_seed"], 7)
                self.assertEqual(first["ppo"]["episodes"], 2)
                self.assertEqual(first["ppo_stochastic"]["episodes"], 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
