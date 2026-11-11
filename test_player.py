import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pygame
import torch

import training_random_map
from entities import Agent, Bullet, Player, Wall
from environments import MapLayout
from models import ppo
from models.mlp import ActorCritic
from renderer import BACKGROUND, Renderer


BOXES = np.empty((0, 4))
IDLE = np.array([0, 1, 0])


class PlayerTests(unittest.TestCase):
    def setUp(self):
        self.player = Player()
        self.player.reset(np.array([100.0, 100.0]), 0.0, True)
        self.agent = Agent()
        self.agent.reset(np.array([500.0, 500.0]), 0.0)

    def update(self, boxes=BOXES):
        self.player.update(0.1, boxes, np.random.default_rng(7), lambda: np.zeros(2), self.agent)

    def test_stats_and_acceleration_match_agent(self):
        for name in (
            "radius", "max_health", "max_speed", "acceleration", "fire_cooldown",
            "bullet_speed", "bullet_damage", "bullet_range", "cooldown",
        ):
            self.assertEqual(getattr(self.player, name), getattr(self.agent, name), name)
        self.player.set_controls(np.array([1.0, 0.0]), np.array([500.0, 100.0]), False)
        self.player.update(1 / 30, BOXES, np.random.default_rng(7), lambda: np.zeros(2), self.agent)
        self.agent.apply_action(np.array([1, 1, 0]), 1 / 30, BOXES)
        np.testing.assert_allclose(self.player.velocity, self.agent.velocity)

    def test_world_movement_normalized_diagonals_and_mouse_aim(self):
        self.player.set_controls(np.array([1.0, -1.0]), np.array([100.0, 400.0]), False)
        self.update()
        self.assertAlmostEqual(float(np.linalg.norm(self.player.velocity)), self.player.max_speed)
        self.assertGreater(self.player.position[0], 100.0)
        self.assertLess(self.player.position[1], 100.0)
        self.assertGreater(self.player.angle, math.pi / 2)
        wall = np.array([[115.0, 0.0, 140.0, 300.0]])
        self.player.reset(np.array([100.0, 100.0]), 0.0, False)
        self.player.set_controls(np.array([1.0, 0.0]), np.array([500.0, 100.0]), False)
        self.update(wall)
        self.assertLessEqual(self.player.position[0], 115.0 - self.player.radius)

    def test_manual_fire_direction_cooldown_walls_and_reset(self):
        self.assertIsNone(self.player.shoot(1.0, BOXES, self.agent))
        self.player.set_controls(np.zeros(2), np.array([500.0, 100.0]), True)
        self.update()
        shot = self.player.shoot(0.1, BOXES, self.agent)
        assert shot is not None
        np.testing.assert_array_equal(shot.velocity, [self.agent.bullet_speed, 0.0])
        self.assertEqual(shot.max_range, self.agent.bullet_range)
        self.assertEqual(shot.damage, 25.0)
        self.assertIsNone(self.player.shoot(0.1, BOXES, self.agent))
        wall = np.array([[115.0, 0.0, 140.0, 300.0]])
        shot.update(1.0, wall, [self.agent])
        self.assertFalse(shot.alive)
        self.assertEqual(shot.position[0], 115.0)
        self.player.reset(np.array([100.0, 100.0]), 0.0, True)
        self.assertFalse(self.player.firing)
        np.testing.assert_array_equal(self.player.movement, [0.0, 0.0])
        self.assertEqual(self.player.speed, self.player.max_speed)
        self.assertFalse(self.player.evade)

    def test_visibility_cone_range_and_occlusion(self):
        self.assertTrue(self.player.can_see(np.array([200.0, 100.0]), BOXES))
        self.assertFalse(self.player.can_see(np.array([0.0, 100.0]), BOXES))
        self.assertFalse(self.player.can_see(np.array([701.0, 100.0]), BOXES))
        wall = np.array([[140.0, 0.0, 160.0, 300.0]])
        self.assertFalse(self.player.can_see(np.array([200.0, 100.0]), wall))
        with self.assertRaises(ValueError):
            self.player.set_controls(np.array([np.nan, 0.0]), np.zeros(2), False)


class PlayerRendererTests(unittest.TestCase):
    def setUp(self):
        dummy = patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"})
        dummy.start()
        self.addCleanup(dummy.stop)

    def env(self, **kwargs):
        env = Renderer(human_opponent=True, fps=0, **kwargs)
        self.addCleanup(env.close)
        env.reset(seed=7)
        return env

    def test_hacks_reveal_hidden_agent_without_changing_observation(self):
        env = self.env(layout=MapLayout(800.0, 800.0, [Wall(180.0, 0.0, 20.0, 800.0)]),
                       bullet_sensing=True)
        env.agent.reset(np.array([300.0, 100.0]), math.pi)
        env.threat.reset(np.array([100.0, 100.0]), 0.0, False)
        env.agent.sense(env.layout.boxes, [env.threat])
        before = env._observation()
        env.render()
        screen = pygame.display.get_surface()
        assert screen is not None
        self.assertEqual(tuple(screen.get_at((300, 100)))[:3], BACKGROUND)
        self.assertFalse(env.show_rays)
        for key in [pygame.K_r, pygame.K_t]:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        env.render()
        self.assertFalse(env.show_rays)
        self.assertFalse(env.show_trails)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
        env.render()
        self.assertTrue(env.player_hacks)
        self.assertTrue(env.show_rays)
        self.assertNotEqual(tuple(screen.get_at((300, 100)))[:3], BACKGROUND)
        after = env._observation()
        np.testing.assert_array_equal(before["state"], after["state"])
        np.testing.assert_array_equal(before["bullets"], after["bullets"])
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
        env.render()
        self.assertFalse(env.show_rays)

    def test_keyboard_mouse_input_and_shared_damage_respawn(self):
        env = self.env(layout=MapLayout(800.0, 800.0, []), max_steps=100)
        player = env.threat
        assert isinstance(player, Player)
        env.agent.reset(np.array([500.0, 100.0]), 0.0)
        player.reset(np.array([100.0, 100.0]), 0.0, False)
        player.cooldown = 0.0
        keys = {key: key == pygame.K_d for key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]}
        with (
            patch("pygame.key.get_pressed", return_value=keys),
            patch("pygame.mouse.get_pos", return_value=(500, 100)),
            patch("pygame.mouse.get_pressed", return_value=(True, False, False)),
        ):
            env.step(IDLE)
        self.assertGreater(player.position[0], 100.0)
        self.assertEqual(env.episode_stats["threat_shots"], 1.0)
        self.assertAlmostEqual(player.angle, 0.0)
        # A player-fired projectile uses the environment's normal damage path.
        env.threat_bullets[0].position = env.agent.position.copy()
        env.step(IDLE)
        self.assertEqual(env.agent.health, 75.0)
        player.health = 25.0
        env.bullets.append(Bullet(player.position.copy(), np.array([900.0, 0.0]), 25.0, 900.0))
        env.step(IDLE)
        self.assertIs(env.threat, player)
        self.assertEqual(player.health, 100.0)
        self.assertEqual(env.episode_stats["kills"], 1.0)

    def test_real_checkpoint_play_mode_inference_only(self):
        args = training_random_map.parse_args([
            "--play-against", "--bullet-sensing", "--map-seeds", "[4]",
            "--episodes", "1", "--max-steps", "2", "--fps", "0", "--player-hacks",
        ])
        created = []

        class RecordingRenderer(Renderer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                created.append(self)

        with tempfile.TemporaryDirectory() as directory, patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)):
            path = ppo.resolve_model_path(args.model)
            policy = ActorCritic(hidden_size=16, bullet_sensing=True)
            ppo.save_checkpoint(
                path, policy, torch.optim.Adam(policy.parameters()), ppo.Curriculum(),
                {"global_step": 0, "episodes": 0, "updates": 0}, 16,
            )
            before = path.read_bytes()
            with patch.object(ppo, "Renderer", RecordingRenderer):
                training_random_map.watch(args)
            self.assertEqual(path.read_bytes(), before)
        self.assertIsInstance(created[0].threat, Player)
        self.assertTrue(created[0].player_hacks)
        self.assertFalse(created[0].window_open)


if __name__ == "__main__":
    unittest.main()
