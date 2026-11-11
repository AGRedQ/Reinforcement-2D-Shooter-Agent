import csv
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from entities import Agent, Bullet
from environments import Curriculum, MapLayout, boundary_walls
from models import ppo
from models.mlp import ActorCritic
from self_play import SelfPlayEnv, SelfPlayRenderer
import training_random_map

HOLD = np.array([[0, 1, 0], [0, 1, 0]])


class DuelTests(unittest.TestCase):
    def env(self, **kwargs):
        env = SelfPlayEnv(
            layout=MapLayout(800, 800, boundary_walls(800, 800, 20)),
            bullet_sensing=True, angular_regions=True, **kwargs,
        )
        env.reset(seed=7)
        env.agent.reset(np.array([200.0, 200.0]), 0.0)
        env.threat.reset(np.array([400.0, 200.0]), math.pi)
        env.agent.sense(env.layout.boxes, [env.threat])
        env.threat.sense(env.layout.boxes, [env.agent])
        return env

    def test_equal_stats_local_observations_and_own_bullet_exclusion(self):
        env = self.env()
        for field in ("radius", "max_health", "max_speed", "acceleration", "turn_rate",
                      "fire_cooldown", "bullet_speed", "bullet_damage", "bullet_range"):
            self.assertEqual(getattr(env.agent, field), getattr(env.threat, field))
            self.assertEqual(getattr(env.agent, field), getattr(Agent(), field))
        env.bullets = [Bullet(np.array([300., 200.]), np.array([900., 0.]), 25, 900)]
        learner, enemy = env._observation(), env.opponent_observation()
        self.assertEqual(learner["state"].shape, (64,))
        self.assertEqual(learner["bullets"].shape, (0, 4))
        self.assertEqual(enemy["bullets"].shape, (1, 4))
        self.assertGreater(enemy["bullets"][0, 0], 0)
        self.assertLess(enemy["bullets"][0, 2], 0)
        env.agent.health = 25
        self.assertFalse(np.array_equal(env._observation()["state"], enemy["state"]))

    def test_both_actions_move_fire_and_share_wall_collisions(self):
        env = self.env()
        env.agent.reset(np.array([35., 200.]), math.pi)
        env.threat.reset(np.array([765., 200.]), 0)
        for _ in range(10):
            env.step(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertGreaterEqual(env.agent.position[0], 34)
        self.assertLessEqual(env.threat.position[0], 766)
        self.assertEqual(env.episode_stats["shots"], env.episode_stats["threat_shots"])
        self.assertEqual(env.agent.health, 100)
        self.assertEqual(env.threat.health, 100)

    def test_win_loss_mutual_kill_and_timeout(self):
        for learner_dead, enemy_dead in ((False, True), (True, False), (True, True)):
            with self.subTest(learner_dead=learner_dead, enemy_dead=enemy_dead):
                env = self.env()
                if enemy_dead:
                    env.threat.health = 25
                    env.bullets = [
                        Bullet(env.threat.position - (20, 0), np.array([900., 0.]), 25, 900)
                    ]
                if learner_dead:
                    env.agent.health = 25
                    env.threat_bullets = [
                        Bullet(env.agent.position + (20, 0), np.array([-900., 0.]), 25, 900)
                    ]
                _, reward, terminated, truncated, info = env.step(HOLD)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                stats = info["episode"]
                self.assertEqual(stats["win"], float(enemy_dead and not learner_dead))
                self.assertEqual(stats["loss"], float(learner_dead and not enemy_dead))
                self.assertEqual(stats["draw"], float(learner_dead and enemy_dead))
                self.assertAlmostEqual(reward, 4 * (int(enemy_dead) - int(learner_dead)))
                with self.assertRaises(RuntimeError):
                    env.step(HOLD)
                env.reset()
                self.assertEqual(env.agent.health, 100)
                self.assertEqual(env.threat.health, 100)
        env = self.env(max_steps=1)
        _, _, terminated, truncated, info = env.step(HOLD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["episode"]["draw"], 1)

    def test_rendered_transitions_match_headless_and_map_cycle(self):
        plain = SelfPlayEnv(map_seeds=(2, 7), bullet_sensing=True, angular_regions=True, max_steps=3)
        rendered = SelfPlayRenderer(
            map_seeds=(2, 7), bullet_sensing=True, angular_regions=True, max_steps=3, fps=0
        )
        try:
            for seed in (9, None):
                a, _ = plain.reset(seed=seed)
                b, _ = rendered.reset(seed=seed)
                self.assertEqual(plain.map_seed, rendered.map_seed)
                np.testing.assert_array_equal(a["state"], b["state"])
                for _ in range(3):
                    a = plain.step(np.array([[1, 2, 1], [3, 0, 1]]))
                    b = rendered.step(np.array([[1, 2, 1], [3, 0, 1]]))
                    np.testing.assert_array_equal(a[0]["state"], b[0]["state"])
                    self.assertEqual(a[1:], b[1:])
                    np.testing.assert_array_equal(
                        plain.opponent_observation()["state"], rendered.opponent_observation()["state"]
                    )
        finally:
            rendered.close()


class SelfPlayTrainingTests(unittest.TestCase):
    def test_finetune_refresh_resume_report_and_watch_preserve_source(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "angular_deep_sets_ppo.pt"
            policy = ActorCritic(hidden_size=16, bullet_sensing=True, angular_regions=True)
            ppo.save_checkpoint(
                source, policy, torch.optim.Adam(policy.parameters()),
                Curriculum(stage=4), {"global_step": 1000, "episodes": 100, "updates": 50}, 16,
            )
            original = source.read_bytes()
            args = training_random_map.parse_args([
                "--self-play", "--num-envs", "2", "--max-steps", "4", "--rollout-steps", "2",
                "--epochs", "1", "--minibatch-size", "4", "--total-episodes", "12",
                "--opponent-refresh", "2", "--no-dashboard",
                "--report-dir", str(root / "reports"), "--map-seeds", "[4]",
            ])
            with patch.object(ppo, "CHECKPOINTS_DIR", root):
                # The frozen model must remain untouched until the refresh boundary.
                original_update = ppo.ppo_update
                snapshots = []
                original_act = ActorCritic.act

                def tracked_act(model, obs, deterministic=False):
                    if not next(model.parameters()).requires_grad:
                        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
                    return original_act(model, obs, deterministic)

                with patch.object(ActorCritic, "act", tracked_act), patch.object(
                    ppo, "ppo_update", wraps=original_update
                ) as update:
                    training_random_map.train(args)
                    self.assertGreater(update.call_count, 2)
                for key in snapshots[0]:
                    torch.testing.assert_close(snapshots[0][key], snapshots[3][key], rtol=0, atol=0)
                self.assertTrue(any(
                    not torch.equal(snapshots[0][key], snapshots[4][key]) for key in snapshots[0]
                ))
                checkpoint = ppo.load_checkpoint(root / args.model)
                self.assertEqual(checkpoint["episodes"], 12)
                self.assertEqual(checkpoint["self_play"]["generation"], checkpoint["updates"] // 2)
                self.assertTrue(any(
                    not torch.equal(checkpoint["model"][key], policy.state_dict()[key])
                    for key in checkpoint["model"]
                ))
                args.total_episodes = 16
                args.render = True
                args.no_dashboard = False
                dashboards = []

                class LocalDashboard(ppo.MetricsDashboard):
                    def start(self):
                        dashboards.append(self)
                        return self

                with patch.object(ppo, "MetricsDashboard", LocalDashboard):
                    training_random_map.train(args)
                history = dashboards[0].snapshot()["history"]
                self.assertTrue(history["win_rate"])
                self.assertTrue(history["draw_rate"])
                self.assertTrue(history["opponent_generation"])
                resumed = ppo.load_checkpoint(root / args.model)
                self.assertGreater(resumed["updates"], checkpoint["updates"])
                args.episodes = 1
                args.fps = 0
                before_watch = (root / args.model).read_bytes()
                training_random_map.watch(args)
                self.assertEqual((root / args.model).read_bytes(), before_watch)
            self.assertEqual(source.read_bytes(), original)
            reports = list((root / "reports").glob("*/metrics.csv"))
            self.assertEqual(len(reports), 2)
            with reports[0].open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["run_episodes"] for row in rows], ["10", "12"])
            self.assertAlmostEqual(sum(float(rows[-1][key]) for key in ("win_rate", "loss_rate", "draw_rate")), 1)

    def test_refuse_overwriting_combat_and_missing_initialization(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)):
            args = training_random_map.parse_args(["--self-play", "--no-dashboard"])
            with self.assertRaises(FileNotFoundError):
                training_random_map.train(args)
            policy = ActorCritic(hidden_size=16, bullet_sensing=True, angular_regions=True)
            path = Path(directory) / "combat.pt"
            ppo.save_checkpoint(
                path, policy, torch.optim.Adam(policy.parameters()), Curriculum(),
                {"global_step": 0, "episodes": 0, "updates": 0}, 16,
            )
            before = path.read_bytes()
            args.model = "combat.pt"
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                training_random_map.train(args)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
