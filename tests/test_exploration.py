import csv
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

import training_random_map
from entities import Bullet
from environments import Curriculum, HuntEnv, MapLayout
from models import ppo
from self_play import SelfPlayEnv


class ExplorationTests(unittest.TestCase):
    def make_env(self, enabled=True, **kwargs):
        env = HuntEnv(
            layout=MapLayout(2000.0, 2000.0, []), exploration_bonus=enabled,
            replay_probability=0.0, **kwargs,
        )
        env.reset(seed=4)
        env.agent.reset(np.array([500.0, 500.0]), 0.0)
        env.threat.reset(np.array([200.0, 500.0]), 0.0, False)
        env.agent.sense(env.layout.boxes, [env.threat])
        if enabled:
            env._reset_exploration_anchor()
        return env

    def test_outward_progress_backtracking_and_distance_limit(self):
        env = self.make_env()
        for x, expected in [(575, .0375), (525, 0), (575, 0), (600, .0125), (650, 0)]:
            env.agent.position[0] = x
            self.assertAlmostEqual(env._exploration_reward(searching=True), expected)
        self.assertAlmostEqual(env.episode_stats["exploration_reward"], .05)
        np.testing.assert_array_equal(env._exploration_anchor, [500, 500])
        env.agent.angle = 2.0
        self.assertEqual(env._exploration_reward(searching=True), 0)

    def test_combat_progress_cannot_be_claimed_later(self):
        env = self.make_env()
        env.agent.position[0] = 575
        self.assertEqual(env._exploration_reward(searching=False), 0)
        self.assertEqual(env._exploration_reward(searching=True), 0)
        env.agent.position[0] = 600
        self.assertAlmostEqual(env._exploration_reward(searching=True), .0125)

    def test_interval_resets_after_five_simulated_seconds(self):
        env = self.make_env(dt=.1)
        env.agent.position[0] = 600
        env.steps = 49
        env._exploration_reward(searching=True)
        np.testing.assert_array_equal(env._exploration_anchor, [500, 500])
        env.steps = 50
        self.assertEqual(env._exploration_reward(searching=True), 0)
        np.testing.assert_array_equal(env._exploration_anchor, [600, 500])
        env.agent.position[0] = 500
        self.assertAlmostEqual(env._exploration_reward(searching=True), .05)

    def test_episode_budget_and_reset(self):
        env = self.make_env()
        total = 0.0
        for _ in range(15):
            env.agent.position[0] += 100
            total += env._exploration_reward(searching=True)
            env._reset_exploration_anchor()
        self.assertAlmostEqual(total, .5)
        self.assertLessEqual(env.episode_stats["exploration_reward"], .5)
        env.reset(seed=7)
        self.assertEqual(env.episode_stats["exploration_reward"], 0)
        np.testing.assert_array_equal(env._exploration_anchor, env.agent.position)

    def test_step_reward_observations_and_metrics(self):
        enabled, baseline = self.make_env(max_steps=1), self.make_env(False, max_steps=1)
        obs, reward, _, _, info = enabled.step((1, 1, 0))
        old_obs, old_reward, _, _, _ = baseline.step((1, 1, 0))
        np.testing.assert_array_equal(obs, old_obs)
        self.assertGreater(reward, old_reward)
        self.assertAlmostEqual(reward - old_reward, info["episode"]["exploration_reward"])
        metrics = ppo.episode_metrics(Curriculum(), deque([info["episode"]]), combat=False)
        self.assertEqual(metrics["mean_exploration_reward"], reward)
        for visible_before in (False, True):
            env = self.make_env()
            env.threat.position = np.array([800.0, 500.0])
            if visible_before:
                env.agent.sense(env.layout.boxes, [env.threat])
            self.assertEqual(env.step((1, 1, 0))[1], 0)

    def test_kill_reanchors_without_resetting_episode_budget(self):
        env = self.make_env()
        env.episode_stats["exploration_reward"] = .2
        env.agent.position[0] = 550
        env.threat.health = 25
        env.bullets.append(Bullet(env.threat.position.copy(), np.array([900., 0.]), 25., 900.))
        env.step((0, 1, 0))
        self.assertEqual(env.episode_stats["kills"], 1)
        self.assertEqual(env.episode_stats["exploration_reward"], .2)
        np.testing.assert_array_equal(env._exploration_anchor, env.agent.position)

    def test_terminal_damage_does_not_earn_exploration(self):
        env = self.make_env(shoot_back=True)
        env.agent.health = 25
        env.threat_bullets.append(
            Bullet(np.array([490., 500.]), np.array([900., 0.]), 25., 900.)
        )
        _, _, terminated, _, info = env.step((1, 1, 0))
        self.assertTrue(terminated)
        self.assertEqual(info["episode"]["exploration_reward"], 0)

    def test_self_play_stays_unshaped(self):
        env = SelfPlayEnv(exploration_bonus=True)
        env.reset(seed=3)
        self.assertFalse(env.exploration_bonus)
        self.assertNotIn("exploration_reward", env.episode_stats)
        with self.assertRaises(SystemExit):
            training_random_map.parse_args(["--self-play", "--exploration-bonus"])

    def test_default_and_disabled_sensing_and_legacy_loading(self):
        for flags, bullets, regions, model in [
            ([], True, True, "angular_deep_sets_ppo.pt"),
            (["--no-angular-regions"], True, False, "deep_sets_ppo_random_map.pt"),
            (["--no-bullet-sensing"], False, True, "angular_mlp_ppo.pt"),
            (["--no-angular-regions", "--no-bullet-sensing"], False, False,
             training_random_map.DEFAULT_MODEL_NAME),
        ]:
            args = training_random_map.parse_args(flags)
            self.assertEqual((args.bullet_sensing, args.angular_regions, args.model),
                             (bullets, regions, model))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = training_random_map.parse_args([
                "--no-angular-regions", "--no-bullet-sensing", "--model", "legacy",
                "--no-dashboard", "--num-envs", "1", "--max-steps", "2",
                "--total-episodes", "2", "--rollout-steps", "2", "--epochs", "1",
                "--minibatch-size", "2", "--hidden-size", "16",
                "--exploration-bonus", "--report-dir", str(root / "reports"),
            ])
            with patch.object(ppo, "CHECKPOINTS_DIR", root):
                training_random_map.train(args)
                resumed = training_random_map.parse_args(["--model", "legacy"])
                for key in ("no_dashboard", "num_envs", "max_steps", "rollout_steps", "epochs",
                            "minibatch_size", "report_dir", "exploration_bonus"):
                    setattr(resumed, key, getattr(args, key))
                resumed.total_episodes = 4
                training_random_map.train(resumed)
                checkpoint = ppo.load_checkpoint(root / "legacy.pt")
                self.assertEqual(checkpoint["architecture"], "mlp")
                self.assertEqual(checkpoint["enemy_sensing"], "rays_v1")
                with self.assertRaisesRegex(ValueError, "not a Deep Sets"):
                    training_random_map.train(training_random_map.parse_args([
                        "--model", "legacy", "--bullet-sensing",
                    ]))
            for folder in (root / "reports").iterdir():
                with (folder / "metrics.csv").open(newline="") as source:
                    self.assertIn("mean_exploration_reward", next(csv.DictReader(source)))


if __name__ == "__main__":
    unittest.main()
