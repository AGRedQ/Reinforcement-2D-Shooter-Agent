"""Geometry and renderer integration checks; run with unittest discovery."""

import math
import os
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import pygame

from entities import RAY_LENGTH
from environments import HuntEnv
from renderer import Renderer
from visibility import visibility_polygon


def contains(points, point):
    """Rasterize at 2x to check the actual polygon fill used by the renderer."""
    surface = pygame.Surface((1600, 1600))
    pygame.draw.polygon(surface, "white", np.round(points * 2).astype(int))
    return surface.get_at(tuple(np.round(np.asarray(point) * 2).astype(int)))[0] > 0


class VisibilityTests(unittest.TestCase):
    def test_range_is_smooth_and_bounded(self):
        origin = np.array([100., 100.])
        points = visibility_polygon(origin, np.empty((0, 4)))
        np.testing.assert_allclose(np.linalg.norm(points - origin, axis=1), RAY_LENGTH)
        self.assertGreater(len(points), 400)
        self.assertLess(np.linalg.norm(np.diff(points, axis=0), axis=1).max(), 8)

    def test_corner_samples_preserve_a_thin_shadow(self):
        origin = np.array([20., 200.])
        boxes = np.array([[200., 201., 201., 202.]])
        points = visibility_polygon(origin, boxes)
        self.assertFalse(contains(points, (400, 203)))
        self.assertTrue(contains(points, (400, 190)))
        self.assertTrue(contains(points, (150, 201)))

    def test_cone_crosses_angle_wrap_and_clips_to_wall(self):
        origin = np.array([400., 400.])
        boxes = np.array([[200., 200., 220., 600.]])
        arc = visibility_polygon(origin, boxes, angle=math.pi - 0.01, spread=math.pi / 3)
        points = np.vstack((origin, arc))
        self.assertTrue(contains(points, (300, 400)))
        self.assertFalse(contains(points, (100, 400)))
        self.assertFalse(contains(points, (500, 400)))
        self.assertFalse(contains(points, (350, 250)))

    def test_rendering_preserves_environment_behavior_and_toggles(self):
        env = HuntEnv(replay_probability=0)
        rendered = Renderer(fps=0, replay_probability=0, render_every=4, show_rays=False, show_visibility=True)
        try:
            np.testing.assert_array_equal(env.reset(seed=17)[0], rendered.reset(seed=17)[0])
            rng = np.random.default_rng(42)
            for _ in range(40):
                action = np.array([rng.integers(9), rng.integers(3), rng.integers(2)])
                plain_result = env.step(action)
                rendered_result = rendered.step(action)
                np.testing.assert_array_equal(plain_result[0], rendered_result[0])
                self.assertEqual(plain_result[1:], rendered_result[1:])
            self.assertEqual(env.episode_stats, rendered.episode_stats)
            for key in (pygame.K_r, pygame.K_v, pygame.K_t, pygame.K_f):
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            rendered.render()
            self.assertTrue(rendered.show_rays)
            self.assertFalse(rendered.show_visibility)
            self.assertFalse(rendered.show_trails)
            self.assertTrue(rendered.cap_fps)
        finally:
            rendered.close()
            env.close()

    def test_feathered_light_does_not_bleed_through_walls(self):
        env = Renderer(fps=0, replay_probability=0, show_visibility=True)
        try:
            env.reset(seed=7)
            env.agent.reset(np.array([310., 400.]), 0.)
            env.agent.sense(env.layout.boxes, [env.threat])
            env.render()
            # The central wall occupies x=370..430, y=370..430.
            self.assertGreater(env._light_layer.get_at((350 * 2, 400 * 2)).a, 0)
            self.assertEqual(env._light_layer.get_at((470 * 2, 400 * 2)).a, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
