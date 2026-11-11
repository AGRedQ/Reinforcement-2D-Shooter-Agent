import unittest
from unittest.mock import patch

import numpy as np
import pygame

from entities import Bullet, Wall
from environments import HuntEnv, MapLayout
from renderer import BACKGROUND, BULLET_COLOR, THREAT_BULLET_COLOR, WALL_COLOR, Renderer


IDLE = np.array([0, 1, 0])


class RendererTests(unittest.TestCase):
    def setUp(self):
        dummy = patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"})
        dummy.start()
        self.addCleanup(dummy.stop)

    def renderer(self, **kwargs):
        env = Renderer(fps=0, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_rendered_and_plain_environments_have_identical_transitions(self):
        kwargs = dict(map_seeds=(2, 7), shoot_back=True, max_steps=20, replay_probability=0.0)
        plain = HuntEnv(**kwargs)
        self.addCleanup(plain.close)
        rendered = self.renderer(**kwargs)
        for episode in range(2):
            left = plain.reset(seed=7 if episode == 0 else None)
            right = rendered.reset(seed=7 if episode == 0 else None)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1], right[1])
            for step in range(20):
                if step == 5:
                    for key in (pygame.K_r, pygame.K_t, pygame.K_v):
                        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
                action = np.array([3, 2, step % 2])
                left, right = plain.step(action), rendered.step(action)
                np.testing.assert_array_equal(left[0], right[0])
                self.assertEqual(left[1:], right[1:])
                np.testing.assert_array_equal(plain.agent.position, rendered.agent.position)
                np.testing.assert_array_equal(plain.threat.position, rendered.threat.position)
                self.assertEqual(len(plain.threat_bullets), len(rendered.threat_bullets))
                if left[2] or left[3]:
                    break

    def test_no_filled_sight_and_both_bullet_colors(self):
        env = self.renderer(layout=MapLayout(800, 800, []), show_rays=False)
        env.reset(seed=7)
        env.agent.reset(np.array([100., 100.]), 0.)
        env.threat.reset(np.array([700., 700.]), 0., False)
        env.bullets = [Bullet(np.array([200., 100.]), np.array([900., 0.]), 25., 900.)]
        env.threat_bullets = [Bullet(np.array([300., 100.]), np.array([-450., 0.]), 10., 600.)]
        env.agent.sense(env.layout.boxes, [env.threat])
        env.render()
        screen = pygame.display.get_surface()
        self.assertEqual(tuple(screen.get_at((250, 110)))[:3], BACKGROUND)
        self.assertEqual(tuple(screen.get_at((200, 100)))[:3], BULLET_COLOR)
        self.assertEqual(tuple(screen.get_at((300, 100)))[:3], THREAT_BULLET_COLOR)
        self.assertFalse(hasattr(env, "show_visibility"))
        for key in (pygame.K_r, pygame.K_t, pygame.K_f):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        env.render()
        self.assertTrue(env.show_rays)
        self.assertFalse(env.show_trails)
        self.assertTrue(env.cap_fps)
        env.step(IDLE)
        self.assertTrue(env._trails[0])
        env.reset()
        self.assertTrue(all(not trail for trail in env._trails))

    def test_cached_walls_refresh_when_generated_map_changes(self):
        env = self.renderer(map_seeds=(2, 7), show_rays=False)
        env.reset(seed=7)
        old_walls = env._walls
        env.reset()
        self.assertEqual(env.map_seed, 7)
        self.assertIsNot(old_walls, env._walls)
        for wall in env.layout.walls:
            center = env._point((wall.x + wall.width / 2, wall.y + wall.height / 2))
            self.assertEqual(tuple(env._walls.get_at(center))[:3], WALL_COLOR)

    def test_angular_fills_toggle_clip_and_do_not_change_observations(self):
        env = self.renderer(
            angular_regions=True, layout=MapLayout(800, 800, [Wall(250, 50, 20, 100)]),
            show_rays=False,
        )
        env.reset(seed=7)
        env.agent.reset(np.array([100., 100.]), 0.)
        env.threat.reset(np.array([700., 700.]), 0., False)
        env.agent.sense(env.layout.boxes, [env.threat])
        before = env.agent.observation().copy()
        env.render()
        screen = pygame.display.get_surface()
        visible_pixel = (200, 125)
        shadow_pixel = (350, 125)
        outside_pixel = (150, 200)
        unfilled = tuple(screen.get_at(visible_pixel))
        shadow = tuple(screen.get_at(shadow_pixel))
        outside = tuple(screen.get_at(outside_pixel))
        env.show_rays = True
        env.render()
        self.assertNotEqual(tuple(screen.get_at(visible_pixel)), unfilled)
        self.assertEqual(tuple(screen.get_at(shadow_pixel)), shadow)
        self.assertEqual(tuple(screen.get_at(outside_pixel)), outside)
        np.testing.assert_array_equal(env.agent.observation(), before)
        env.show_rays = False
        env.render()
        self.assertEqual(tuple(screen.get_at(visible_pixel)), unfilled)


if __name__ == "__main__":
    unittest.main()
