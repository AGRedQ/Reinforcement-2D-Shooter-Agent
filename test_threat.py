import math
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from models import ppo
import training_random_map
from entities import OBSERVATION_SIZE, Agent, Bullet, ShootingThreat, Threat
from environments import Curriculum, HuntEnv, MapLayout
from renderer import MetricsDashboard, Renderer


EMPTY_BOXES = np.empty((0, 4))
IDLE = np.array([0, 1, 0])


def combat_env(**kwargs) -> HuntEnv:
    env = HuntEnv(layout=MapLayout(800.0, 800.0, []), shoot_back=True, replay_probability=0.0, **kwargs)
    env.reset(seed=7)
    env.agent.reset(np.array([100.0, 100.0]), 0.0)
    env.threat.reset(np.array([200.0, 100.0]), 0.0, False)
    env.agent.sense(env.layout.boxes, [env.threat])
    return env


class ShootingThreatTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()
        self.agent.reset(np.array([200.0, 100.0]), math.pi)
        self.threat = ShootingThreat()
        self.threat.reset(np.array([100.0, 100.0]), 0.0, False)

    def test_fire_delay_cooldown_and_reset(self):
        self.assertIsNone(self.threat.shoot(0.4, EMPTY_BOXES, self.agent))
        bullet = self.threat.shoot(0.4, EMPTY_BOXES, self.agent)
        if bullet is None:
            self.fail("Threat did not fire after its initial delay")
        self.assertIs(bullet.update(1.0, EMPTY_BOXES, [self.agent]), self.agent)
        self.assertFalse(bullet.alive)
        self.assertIsNone(self.threat.shoot(0.1, EMPTY_BOXES, self.agent))
        self.threat.reset(np.array([100.0, 100.0]), 0.0, False)
        self.assertIsNone(self.threat.shoot(0.1, EMPTY_BOXES, self.agent))

    def test_walls_range_and_dead_entities_prevent_shooting(self):
        wall = np.array([[140.0, 50.0, 160.0, 150.0]])
        self.assertIsNone(self.threat.shoot(1.0, wall, self.agent))
        self.agent.position = np.array([750.0, 100.0])
        self.assertIsNone(self.threat.shoot(1.0, EMPTY_BOXES, self.agent))
        self.agent.position = np.array([200.0, 100.0])
        self.agent.health = 0.0
        self.assertIsNone(self.threat.shoot(1.0, EMPTY_BOXES, self.agent))
        self.agent.health = self.agent.max_health
        self.threat.health = 0.0
        self.assertIsNone(self.threat.shoot(1.0, EMPTY_BOXES, self.agent))

    def test_projectiles_stop_at_walls_and_max_range(self):
        wall = np.array([[140.0, 50.0, 160.0, 150.0]])
        for boxes, max_range, expected_x in [(wall, 600.0, 140.0), (EMPTY_BOXES, 30.0, 130.0)]:
            with self.subTest(max_range=max_range):
                bullet = Bullet(np.array([100.0, 100.0]), np.array([450.0, 0.0]), 10.0, max_range)
                self.assertIsNone(bullet.update(1.0, boxes, [self.agent]))
                self.assertFalse(bullet.alive)
                self.assertAlmostEqual(bullet.position[0], expected_x)


class CombatEnvironmentTests(unittest.TestCase):
    def test_gym_api_and_seeded_rollouts(self):
        check_env(HuntEnv(shoot_back=True, max_steps=10), skip_render_check=True)
        for stage in range(5):
            first = HuntEnv(shoot_back=True, stage=stage, max_steps=30)
            second = HuntEnv(shoot_back=True, stage=stage, max_steps=30)
            np.testing.assert_array_equal(first.reset(seed=13)[0], second.reset(seed=13)[0])
            for _ in range(30):
                left, right = first.step(IDLE), second.step(IDLE)
                np.testing.assert_array_equal(left[0], right[0])
                self.assertEqual(left[1:], right[1:])
                self.assertTrue(first.observation_space.contains(left[0]))
                if left[2] or left[3]:
                    break

    def test_shots_damage_death_terminal_stats_and_reset(self):
        env = combat_env(max_steps=1000)
        total_reward = 0.0
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(IDLE)
            total_reward += reward
            if terminated:
                break
        else:
            self.fail("A stationary agent should die from repeated enemy fire")
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.agent.health, 0.0)
        self.assertEqual(obs.shape, (OBSERVATION_SIZE,))
        self.assertEqual(info["episode"]["damage_taken"], 100.0)
        self.assertEqual(info["episode"]["deaths"], 1.0)
        self.assertEqual(
            info["episode"]["threat_shots"],
            math.ceil(env.agent.max_health / env.threat.bullet_damage),
        )
        self.assertAlmostEqual(total_reward, -7.0)
        self.assertAlmostEqual(info["episode"]["reward"], total_reward)
        with self.assertRaises(RuntimeError):
            env.step(IDLE)
        env.reset(seed=7)
        self.assertEqual(env.agent.health, env.agent.max_health)
        self.assertEqual(env.threat_bullets, [])
        self.assertEqual(env.episode_stats["damage_taken"], 0.0)

    def test_cover_blocks_in_flight_enemy_bullets(self):
        env = combat_env(max_steps=20)
        env.threat_bullets.append(Bullet(np.array([180.0, 100.0]), np.array([-450.0, 0.0]), 10.0, 600.0))
        env.layout.boxes = np.array([[140.0, 50.0, 160.0, 150.0]])
        for _ in range(10):
            env.step(IDLE)
        self.assertEqual(env.agent.health, env.agent.max_health)
        self.assertEqual(env.threat_bullets, [])

    def test_kills_respawn_threat_without_erasing_in_flight_enemy_shots(self):
        env = combat_env(max_steps=20)
        env.threat.health = env.agent.bullet_damage
        env.threat_bullets.append(Bullet(np.array([700.0, 700.0]), np.array([0.0, -450.0]), 10.0, 600.0))
        for _ in range(5):
            env.step(np.array([0, 1, 1]))
            if env.episode_stats["kills"]:
                break
        self.assertEqual(env.episode_stats["kills"], 1.0)
        self.assertEqual(env.threat.health, env.threat.max_health)
        self.assertEqual(env.threat_bullets[0].traveled, env.steps * 15.0)
        if not isinstance(env.threat, ShootingThreat):
            self.fail("Combat mode must use a shooting threat")
        self.assertGreater(env.threat.cooldown, 0.0)

    def test_default_hunt_remains_passive_and_time_limited(self):
        env = HuntEnv(max_steps=40, replay_probability=0.0)
        env.reset(seed=7)
        self.assertIs(type(env.threat), Threat)
        for _ in range(39):
            _, _, terminated, truncated, info = env.step(IDLE)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        _, _, terminated, truncated, info = env.step(IDLE)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.agent.health, env.agent.max_health)
        self.assertEqual(env.threat_bullets, [])
        self.assertEqual(info["episode"]["damage_taken"], 0.0)


class CombatTrainingTests(unittest.TestCase):
    def test_random_map_trainer_uses_combat_checkpoint_and_mode(self):
        args = training_random_map.parse_args([])
        self.assertEqual(args.model, training_random_map.DEFAULT_MODEL_NAME)
        with (
            patch.object(training_random_map, "train_ppo") as train,
            patch.object(training_random_map, "watch_ppo") as watch,
        ):
            training_random_map.train(args)
            training_random_map.watch(args)
        train.assert_called_once_with(args, shoot_back=True, map_seeds=args.map_seeds)
        watch.assert_called_once_with(args, shoot_back=True, map_seeds=args.map_seeds)

    def test_dashboard_metrics_work_in_both_modes(self):
        env = combat_env(max_steps=1)
        episode = env.step(IDLE)[4]["episode"]
        for metrics in [ppo.EPISODE_METRICS, {**ppo.EPISODE_METRICS, **ppo.COMBAT_METRICS}]:
            dashboard = MetricsDashboard(metrics)
            ppo.log_episode_metrics(dashboard, {"global_step": 1}, Curriculum(), deque([episode]))

    def test_autoreset_on_death_preserves_terminal_observation(self):
        env = combat_env(max_steps=100)
        env.agent.health = 10.0
        env.threat_bullets.append(Bullet(np.array([120.0, 100.0]), np.array([-450.0, 0.0]), 10.0, 600.0))
        obs, _, terminated, truncated, info, final_obs = ppo._step_autoreset(env, IDLE)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["episode"]["deaths"], 1.0)
        self.assertEqual(env.agent.health, env.agent.max_health)
        if final_obs is None:
            self.fail("Autoreset must return the final observation")
        self.assertFalse(np.array_equal(obs, final_obs))

    def test_headless_renderer_handles_death(self):
        with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            env = Renderer(shoot_back=True, fps=0, max_steps=10)
            try:
                env.reset(seed=7)
                env.agent.health = 10.0
                env.threat_bullets.append(
                    Bullet(env.agent.position.copy(), np.array([450.0, 0.0]), 10.0, 600.0)
                )
                self.assertTrue(env.step(IDLE)[2])
            finally:
                env.close()

    def test_multiprocess_workers_shoot_and_autoreset_on_death(self):
        envs = ppo.VecEnv(
            2, {"shoot_back": True, "max_steps": 400, "replay_probability": 0.0}, seed=7
        )
        finished = set()
        try:
            for _ in range(400):
                _, _, terminated, _, infos, final_obs = envs.step(np.stack([IDLE, IDLE]))
                for i, info in enumerate(infos):
                    if "episode" not in info:
                        continue
                    self.assertTrue(terminated[i])
                    self.assertEqual(info["episode"]["deaths"], 1.0)
                    self.assertEqual(info["episode"]["damage_taken"], 100.0)
                    self.assertGreaterEqual(
                        info["episode"]["threat_shots"],
                        math.ceil(Agent().max_health / ShootingThreat().bullet_damage),
                    )
                    self.assertIsNotNone(final_obs[i])
                    finished.add(i)
                if len(finished) == 2:
                    break
            self.assertEqual(finished, {0, 1})
        finally:
            envs.close()

    def test_multiprocess_training_saves_and_resumes_real_updates(self):
        args = training_random_map.parse_args([
            "--no-dashboard", "--num-envs", "2", "--max-steps", "12",
            "--rollout-steps", "4", "--minibatch-size", "8", "--hidden-size", "16",
            "--epochs", "1", "--total-episodes", "4", "--save-every", "1",
        ])
        with tempfile.TemporaryDirectory() as directory, patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)):
            training_random_map.train(args)
            model_path = ppo.resolve_model_path(args.model)
            checkpoint = ppo.load_checkpoint(model_path)
            self.assertGreater(checkpoint["updates"], 0)
            self.assertGreaterEqual(checkpoint["episodes"], 4)
            self.assertTrue(all(torch.isfinite(value).all() for value in checkpoint["model"].values()))
            args.total_episodes = checkpoint["episodes"] + 2
            training_random_map.train(args)
            resumed = ppo.load_checkpoint(model_path)
            self.assertGreater(resumed["updates"], checkpoint["updates"])
            self.assertGreater(resumed["global_step"], checkpoint["global_step"])
            with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
                args.episodes = 1
                args.fps = 0
                training_random_map.watch(args)


if __name__ == "__main__":
    unittest.main()
