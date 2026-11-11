import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pygame
import torch
from gymnasium.utils.env_checker import check_env

import training_random_map
from entities import (
    ANGULAR_OBSERVATION_SIZE, OBSERVATION_SIZE, Agent, Threat,
)
from environments import HuntEnv, MapLayout
from models import ppo
from models.mlp import ActorCritic
from renderer import Renderer


BOXES = np.empty((0, 4))


class AngularRegionTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(angular_regions=True)
        self.agent.reset(np.array([100.0, 100.0]), 0.0)
        self.target = Threat()

    def target_at(self, angle, distance=400.0):
        radians = math.radians(angle) + self.agent.angle
        self.target.reset(
            self.agent.position + distance * np.array([math.cos(radians), math.sin(radians)]),
            0.0, False,
        )

    def test_sweep_has_no_gaps_and_keeps_wall_rays(self):
        legacy = Agent()
        legacy.reset(self.agent.position, self.agent.angle)
        for angle in np.linspace(-30.0, 30.0, 241):
            self.target_at(float(angle), 550.0)
            self.agent.sense(BOXES, [self.target])
            self.assertTrue(self.agent.enemy_detected.any(), angle)
            legacy.sense(BOXES, [self.target])
            np.testing.assert_array_equal(self.agent.wall_distances, legacy.wall_distances)
        self.target_at(5.625, 550.0)
        self.agent.sense(BOXES, [self.target])
        legacy.sense(BOXES, [self.target])
        self.assertTrue(self.agent.enemy_detected.any())
        self.assertFalse(legacy.enemy_detected.any())

    def test_region_order_footprint_and_precise_center(self):
        for angle, index in [(-23, 0), (-10, 1), (-2.5, 2), (2.5, 3), (10, 4), (23, 5)]:
            self.target_at(angle, 550.0)
            self.agent.sense(BOXES, [self.target])
            self.assertEqual(np.flatnonzero(self.agent.enemy_detected).tolist(), [index])
            self.assertFalse(self.agent.center_detected)
        self.target_at(0.0)
        self.agent.sense(BOXES, [self.target])
        self.assertEqual(np.flatnonzero(self.agent.enemy_detected).tolist(), [2, 3])
        self.assertTrue(self.agent.center_detected)
        self.assertAlmostEqual(self.agent.center_distance, 400 - self.target.radius)
        # A shot can hit the body's edge without aiming at its exact center.
        self.target_at(1.0)
        self.agent.sense(BOXES, [self.target])
        self.assertTrue(self.agent.center_detected)
        self.target_at(4.0)
        self.agent.sense(BOXES, [self.target])
        self.assertFalse(self.agent.center_detected)

    def test_walls_partial_visibility_range_and_dead_targets(self):
        self.target_at(0.0)
        full_wall = np.array([[200.0, 50.0, 220.0, 150.0]])
        self.agent.sense(full_wall, [self.target])
        self.assertFalse(self.agent.enemy_detected.any())
        self.assertFalse(self.agent.center_detected)
        # A narrow pillar hides the center, but not the visible sides.
        pillar = np.array([[200.0, 99.0, 210.0, 101.0]])
        self.agent.sense(pillar, [self.target])
        self.assertTrue(self.agent.enemy_detected[2:4].all())
        self.assertFalse(self.agent.center_detected)
        self.target_at(40.0)
        self.agent.sense(BOXES, [self.target])
        self.assertFalse(self.agent.enemy_detected.any())
        self.target_at(0.0, 620.0)
        self.agent.sense(BOXES, [self.target])
        self.assertFalse(self.agent.enemy_detected.any())
        self.target_at(0.0, 610.0)
        self.agent.sense(BOXES, [self.target])
        self.assertTrue(self.agent.enemy_detected.any())
        self.target.health = 0.0
        self.agent.sense(BOXES, [self.target])
        self.assertFalse(self.agent.enemy_detected.any())
        np.testing.assert_array_equal(self.agent.enemy_distances, np.full(6, 600.0))

    def test_nearest_enemy_and_egocentric_rotation(self):
        self.target_at(10.0, 300.0)
        near = self.target
        far = Threat()
        far.reset(near.position + (near.position - self.agent.position), 0.0, False)
        self.agent.sense(BOXES, [far, near])
        expected = self.agent.observation()
        self.assertAlmostEqual(self.agent.enemy_distances[4], 286.0)
        for actor in (self.agent, near, far):
            actor.position = np.array([-actor.position[1], actor.position[0]]) + 800
        self.agent.angle += math.pi / 2
        self.agent.sense(BOXES, [near, far])
        np.testing.assert_allclose(self.agent.observation(), expected, atol=1e-6)
        self.assertEqual(expected.shape, (ANGULAR_OBSERVATION_SIZE,))


class AngularIntegrationTests(unittest.TestCase):
    def test_gym_api_and_legacy_size(self):
        for bullets in (False, True):
            env = HuntEnv(angular_regions=True, bullet_sensing=bullets, shoot_back=True, max_steps=10)
            check_env(env, skip_render_check=True)
        legacy = HuntEnv()
        observation = legacy.reset(seed=7)[0]
        assert isinstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(ANGULAR_OBSERVATION_SIZE, 64)

    def test_center_alignment_does_not_gate_shooting(self):
        env = HuntEnv(angular_regions=True, layout=MapLayout(800.0, 800.0, []))
        env.reset(seed=7)
        env.agent.reset(np.array([100.0, 100.0]), 0.0)
        angle = math.radians(20)
        env.threat.reset(
            env.agent.position + 400 * np.array([math.cos(angle), math.sin(angle)]), 0.0, False,
        )
        env.agent.sense(env.layout.boxes, [env.threat])
        self.assertTrue(env.agent.enemy_detected.any())
        self.assertFalse(env.agent.center_detected)
        env.step(np.array([0, 1, 1]))
        self.assertEqual(env.episode_stats["shots"], 1.0)
        self.assertEqual(env.episode_stats["wasted_shots"], 0.0)

    def test_training_resume_render_watch_and_play(self):
        args = training_random_map.parse_args([
            "--angular-regions", "--bullet-sensing", "--no-dashboard",
            "--num-envs", "2", "--max-steps", "12", "--rollout-steps", "4",
            "--minibatch-size", "8", "--hidden-size", "16", "--epochs", "1",
            "--total-episodes", "4", "--map-seeds", "[2,7]",
        ])
        self.assertEqual(args.model, "angular_deep_sets_ppo.pt")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)),
            patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}),
        ):
            training_random_map.train(args)
            path = ppo.resolve_model_path(args.model)
            first = ppo.load_checkpoint(path)
            self.assertEqual(first["obs_size"], 64)
            self.assertEqual(first["enemy_sensing"], "angular_regions_v1")
            self.assertGreater(first["updates"], 0)
            args.angular_regions = False
            args.bullet_sensing = False
            args.total_episodes = first["episodes"] + 2
            args.render = True
            training_random_map.train(args)
            self.assertGreater(ppo.load_checkpoint(path)["updates"], first["updates"])
            args.episodes, args.fps = 1, 0
            training_random_map.watch(args)
            args.play_against, args.player_hacks = True, True
            training_random_map.watch(args)

    def test_flat_angular_checkpoint_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)):
            path = Path(directory) / "legacy.pt"
            legacy = ActorCritic(hidden_size=16)
            ppo.save_checkpoint(
                path, legacy, torch.optim.Adam(legacy.parameters()), ppo.Curriculum(),
                {"global_step": 0, "episodes": 0, "updates": 0}, 16,
            )
            # Checkpoints predating sensing metadata still mean legacy rays.
            saved = torch.load(path, weights_only=False)
            saved.pop("enemy_sensing")
            torch.save(saved, path)
            self.assertEqual(ppo.load_checkpoint(path)["obs_size"], 60)
            args = training_random_map.parse_args(["--angular-regions", "--model", "legacy"])
            with self.assertRaisesRegex(ValueError, "legacy enemy rays"):
                training_random_map.train(args)
            with self.assertRaisesRegex(ValueError, "legacy checkpoints retain rays"):
                training_random_map.watch(args)
            policy = ActorCritic(hidden_size=16, angular_regions=True)
            ppo.save_checkpoint(
                path, policy, torch.optim.Adam(policy.parameters()), ppo.Curriculum(),
                {"global_step": 0, "episodes": 0, "updates": 0}, 16,
            )
            checkpoint = ppo.load_checkpoint(path)
            restored = ActorCritic(hidden_size=16, angular_regions=True)
            restored.load_state_dict(checkpoint["model"])
            obs = torch.zeros(2, 64)
            torch.testing.assert_close(policy.value(obs), restored.value(obs))

    def test_renderer_draws_regions_and_center(self):
        with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            env = Renderer(angular_regions=True, layout=MapLayout(800.0, 800.0, []), fps=0)
            try:
                env.reset(seed=7)
                env.agent.reset(np.array([100.0, 100.0]), 0.0)
                env.threat.reset(np.array([400.0, 100.0]), 0.0, False)
                env.agent.sense(env.layout.boxes, [env.threat])
                with patch("pygame.draw.line", wraps=pygame.draw.line) as line:
                    env._draw_rays()
                    # 32 wall rays, seven boundaries, two contacts, one center ray.
                    self.assertEqual(line.call_count, 42)
                env.render()
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
