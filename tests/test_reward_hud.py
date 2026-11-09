import os
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from pygame import Vector2

from entities import Wall
from mechanics_env import MechanicsEnv
from renderer import Renderer
from wallgen_navpoint_env import WallgenNavpointEnv


class RewardScoreTests(unittest.TestCase):
    def test_score_is_exact_return_and_resets_for_both_tasks(self):
        for task in (MechanicsEnv, WallgenNavpointEnv):
            with self.subTest(task=task.__name__):
                env = task()
                try:
                    _, info = env.reset(seed=0)
                    self.assertEqual(info["episode_reward"], 0)
                    self.assertEqual(info["last_reward"], 0)
                    total = 0.0
                    for action in ([0, 2, 0], [4, 2, 1], [4, 2, 0]):
                        _, reward, terminated, _, info = env.step(np.array(action))
                        self.assertFalse(terminated)
                        total += reward
                        self.assertEqual(info["episode_reward"], total)
                        self.assertEqual(info["last_reward"], reward)
                    _, info = env.reset(seed=0)
                    self.assertEqual(info["episode_reward"], 0)
                    self.assertEqual(info["last_reward"], 0)
                finally:
                    env.close()

    def test_node_bonus_and_final_hit_are_included_once(self):
        env = WallgenNavpointEnv()
        try:
            env.reset(seed=0)
            env.target_visibility_reward_earned = True
            env.target_alignment_reward_earned = True
            index = next(index for index in range(12) if index not in env.visited)
            env.agent.position = env.layout.nodes[index].copy()
            _, visit_reward, _, _, info = env.step(np.array([0, 2, 0]))
            self.assertAlmostEqual(visit_reward, env.node_reward - env.step_penalty)
            self.assertEqual(info["episode_reward"], visit_reward)
            env.agent.position = Vector2(50, 50)
            env.target.position = Vector2(150, 50)
            env.agent.aim_x, env.agent.aim_y = 1, 0
            _, hit_reward, terminated, _, info = env.step(np.array([0, 2, 1]))
            self.assertTrue(terminated)
            self.assertAlmostEqual(hit_reward, env.hit_reward - env.step_penalty)
            self.assertEqual(info["episode_reward"], visit_reward + hit_reward)
            self.assertEqual(info["last_reward"], hit_reward)
            with self.assertRaises(gym.error.ResetNeeded):
                env.step(np.array([0, 2, 0]))
            self.assertEqual(env.episode_reward, visit_reward + hit_reward)
        finally:
            env.close()

    def test_timeout_keeps_final_score_until_reset(self):
        env = WallgenNavpointEnv(max_steps=1)
        try:
            env.reset(seed=0)
            _, reward, _, truncated, info = env.step(np.array([0, 2, 0]))
            self.assertTrue(truncated)
            self.assertEqual(info["episode_reward"], reward)
            self.assertIn(f"{reward:+.3f}", env._score_hud()[0])
            self.assertIn("Step: 1/1", env._score_hud()[1])
        finally:
            env.close()


class RewardHudRenderingTests(unittest.TestCase):
    def test_both_watch_views_render_live_score_without_changing_state(self):
        for task in (MechanicsEnv, WallgenNavpointEnv):
            with self.subTest(task=task.__name__):
                with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
                    env = task(render_mode="human")
                try:
                    env.reset(seed=0)
                    _, reward, _, _, _ = env.step(np.array([0, 2, 0]))
                    before = env._observation()
                    renderer = env.simulation
                    assert isinstance(renderer, Renderer)
                    with patch.object(renderer, "render", wraps=renderer.render) as render:
                        env.render()
                        env.render()
                        self.assertEqual(
                            render.call_args.kwargs["hud_lines"],
                            (
                                f"Score (episode reward): {reward:+.3f}",
                                f"Last reward: {reward:+.3f} | Step: 1/600",
                            ),
                        )
                    self.assertEqual(env.episode_reward, reward)
                    self.assertEqual(env.simulation.step_count, 1)
                    np.testing.assert_array_equal(before, env._observation())
                    env.reset(seed=1)
                    self.assertIn("+0.000", env._score_hud()[0])
                finally:
                    env.close()

    def test_hud_is_drawn_over_walls_and_is_optional(self):
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            renderer = Renderer()
        try:
            lines = ("Score (episode reward): +1.234", "Last reward: -0.001 | Step: 42/600")
            panel_top = renderer.screen.get_height() - 2 * renderer.vision_font.get_linesize() - 22
            y = panel_top + 2
            renderer.walls = [Wall(Vector2(0, y), Vector2(300, y))]
            renderer.render()
            self.assertEqual(renderer.screen.get_at((12, y))[:3], (120, 135, 155))
            renderer.render(hud_lines=lines)
            self.assertEqual(renderer.screen.get_at((12, y))[:3], (16, 18, 24))
            label = renderer.vision_font.render(lines[0], True, (235, 240, 245))
            bright_pixels = [
                (x, y) for x in range(label.get_width()) for y in range(label.get_height())
                if label.get_at((x, y)).a == 255
            ]
            self.assertTrue(bright_pixels)
            x, y = bright_pixels[0]
            self.assertEqual(
                renderer.screen.get_at((16 + x, panel_top + 6 + y))[:3],
                (235, 240, 245),
            )
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
