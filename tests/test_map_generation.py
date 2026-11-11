import argparse
import contextlib
import io
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gymnasium.utils.env_checker import check_env

from models import ppo
import training_random_map
from entities import Agent, Bullet, ShootingThreat, circle_overlaps_boxes
from environments import STAGES, HuntEnv, default_map
from map_generation import generate_map, parse_map_seeds
from renderer import Renderer

IDLE = np.array([0, 1, 0])


class MapSeedParsingTests(unittest.TestCase):
    def test_integer_inclusive_range_singleton_and_list(self):
        self.assertEqual(list(parse_map_seeds("3")), [0, 1, 2, 3])
        self.assertEqual(list(parse_map_seeds("0")), [0])
        self.assertEqual(parse_map_seeds("[3]"), (3,))
        self.assertEqual(parse_map_seeds("[7, 2, 7, 11]"), (7, 2, 7, 11))
        large_range = parse_map_seeds("1000000000")
        self.assertIsInstance(large_range, range)
        self.assertEqual(large_range[-1], 1000000000)

    def test_invalid_cli_seeds_are_reported(self):
        for value in ["-1", "[]", "[1,-2]", "[1,2.0]", "true", "[false]", "null", "3.0", "abc", "[1,"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_map_seeds(value)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as invalid:
            training_random_map.parse_args(["--map-seeds", "[]"])
        self.assertEqual(invalid.exception.code, 2)

    def test_cli_and_wrappers_include_combat_and_separate_checkpoint(self):
        args = training_random_map.parse_args(["--map-seeds", "[2,7]"])
        self.assertEqual(args.map_seeds, (2, 7))
        self.assertEqual(list(training_random_map.parse_args([]).map_seeds), list(range(10)))
        self.assertEqual(args.model, "angular_deep_sets_ppo.pt")
        with (
            patch.object(training_random_map, "train_ppo") as train,
            patch.object(training_random_map, "watch_ppo") as watch,
        ):
            training_random_map.train(args)
            training_random_map.watch(args)
        train.assert_called_once_with(args, shoot_back=True, map_seeds=(2, 7))
        watch.assert_called_once_with(args, shoot_back=True, map_seeds=(2, 7))


class MapGeometryTests(unittest.TestCase):
    def test_seed_reproducibility_variety_boundaries_and_clearance(self):
        signatures = set()
        for seed in range(250):
            layout = generate_map(seed)
            np.testing.assert_array_equal(layout.boxes, generate_map(seed).boxes)
            signatures.add(layout.boxes.tobytes())
            self.assertEqual((layout.width, layout.height), (800.0, 800.0))
            np.testing.assert_array_equal(layout.boxes[:4], default_map().boxes[:4])
            self.assertIn(len(layout.walls), range(10, 14))
            for index, box in enumerate(layout.boxes[4:]):
                xmin, ymin, xmax, ymax = box
                self.assertGreaterEqual(min(xmin - 20, ymin - 20, 780 - xmax, 780 - ymax), 48.0)
                self.assertTrue(np.all(box[2:] - box[:2] >= 40))
                self.assertTrue(np.all(box[2:] - box[:2] <= 150))
                for other in layout.boxes[4:4 + index]:
                    separation = np.maximum(other[:2] - box[2:], box[:2] - other[2:])
                    self.assertGreaterEqual(float(separation.max()), 96.0 - 1e-9)
        self.assertEqual(len(signatures), 250)

    def test_free_space_is_connected_at_body_clearance(self):
        radius = Agent().radius + 4.0
        axis = np.arange(30.0, 780.0, 10.0)
        centers = np.stack(np.meshgrid(axis, axis), axis=-1)
        for seed in range(50):
            boxes = generate_map(seed).boxes
            nearest = np.clip(centers[:, :, None, :], boxes[None, None, :, :2], boxes[None, None, :, 2:])
            free = np.all(np.sum((centers[:, :, None, :] - nearest) ** 2, axis=-1) >= radius ** 2, axis=-1)
            row, col = np.argwhere(free)[0]
            queue = deque([(int(row), int(col))])
            visited = {(int(row), int(col))}
            while queue:
                row, col = queue.popleft()
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < len(axis) and 0 <= nc < len(axis) and free[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            self.assertEqual(len(visited), int(free.sum()), f"Disconnected playable space on seed {seed}")

    def test_invalid_generator_and_environment_config(self):
        for seed in [-1, True, 2.5]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_map(seed)
        for seeds in [[], (-1,), (True,), range(0), range(-1, 2), (1.5,)]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                HuntEnv(map_seeds=seeds)
        with self.assertRaisesRegex(ValueError, "either"):
            HuntEnv(layout=default_map(), map_seeds=(0,))


class GeneratedMapEnvironmentTests(unittest.TestCase):
    def test_map_seed_cycles_and_explicit_reset_restarts(self):
        for sequence in [range(3), (6,), (2, 7, 2)]:
            with self.subTest(sequence=sequence):
                env = HuntEnv(map_seeds=sequence, shoot_back=True)
                for index in range(len(sequence) * 2 + 1):
                    _, info = env.reset(seed=50 if index == 0 else None)
                    expected = sequence[index % len(sequence)]
                    self.assertEqual(info["map_seed"], expected)
                    self.assertEqual(env.map_seed, expected)
                    np.testing.assert_array_equal(env.layout.boxes, generate_map(expected).boxes)
                env.reset(seed=50)
                self.assertEqual(env.map_seed, sequence[0])

    def test_maps_do_not_depend_on_spawn_seed(self):
        first = HuntEnv(map_seeds=(13,), shoot_back=True)
        second = HuntEnv(map_seeds=(13,), shoot_back=True)
        first.reset(seed=1)
        second.reset(seed=999)
        np.testing.assert_array_equal(first.layout.boxes, second.layout.boxes)
        self.assertFalse(np.array_equal(first.agent.position, second.agent.position))

    def test_gym_api_and_curriculum_stages_use_safe_spawns(self):
        check_env(HuntEnv(map_seeds=(1, 5), shoot_back=True, max_steps=10), skip_render_check=True)
        env = HuntEnv(map_seeds=range(5), shoot_back=True, replay_probability=0.0, max_steps=30)
        for stage in range(len(STAGES)):
            env.set_stage(stage)
            obs, _ = env.reset()
            self.assertEqual(env.episode_stage, stage)
            self.assertIsInstance(env.threat, ShootingThreat)
            self.assertEqual(env.threat.speed, STAGES[stage].threat_speed)
            for body in [env.agent, env.threat]:
                self.assertFalse(circle_overlaps_boxes(body.position, body.radius, env.layout.boxes))
            self.assertTrue(env.observation_space.contains(obs))
            for _ in range(30):
                obs, _, terminated, truncated, _ = env.step(IDLE)
                self.assertTrue(env.observation_space.contains(obs))
                if terminated or truncated:
                    break

    def test_threat_respawn_keeps_map_but_death_autoreset_advances_cycle(self):
        env = HuntEnv(map_seeds=(3, 5), shoot_back=True, max_steps=100, replay_probability=0.0)
        env.reset(seed=7)
        layout = env.layout
        env.threat.health = 25.0
        env.bullets.append(Bullet(env.threat.position.copy(), np.array([900.0, 0.0]), 25.0, 900.0))
        env.step(IDLE)
        self.assertEqual(env.episode_stats["kills"], 1.0)
        self.assertIs(env.layout, layout)
        self.assertEqual(env.map_seed, 3)
        env.agent.health = 10.0
        env.threat_bullets.append(Bullet(env.agent.position.copy(), np.array([450.0, 0.0]), 10.0, 600.0))
        _, _, terminated, _, info, final_obs = ppo._step_autoreset(env, IDLE)
        self.assertTrue(terminated)
        self.assertIsNotNone(final_obs)
        self.assertEqual(info["episode"]["map_seed"], 3)
        self.assertEqual(info["episode"]["deaths"], 1.0)
        self.assertEqual(env.map_seed, 5)
        self.assertIsNot(env.layout, layout)
        self.assertEqual(env.agent.health, env.agent.max_health)


class GeneratedMapTrainingTests(unittest.TestCase):
    def test_workers_keep_combat_and_advance_maps_on_death(self):
        envs = ppo.VecEnv(
            2, {"map_seeds": (2, 7), "shoot_back": True, "max_steps": 400, "replay_probability": 0.0},
            seed=9,
        )
        finished = set()
        try:
            for _ in range(400):
                _, _, terminated, _, infos, final_obs = envs.step(np.stack([IDLE, IDLE]))
                for index, info in enumerate(infos):
                    if "episode" not in info:
                        continue
                    episode = info["episode"]
                    self.assertTrue(terminated[index])
                    self.assertEqual(episode["map_seed"], 2)
                    self.assertEqual(episode["deaths"], 1.0)
                    self.assertEqual(episode["damage_taken"], 100.0)
                    self.assertGreaterEqual(
                        episode["threat_shots"],
                        math.ceil(Agent().max_health / ShootingThreat().bullet_damage),
                    )
                    self.assertIsNotNone(final_obs[index])
                    finished.add(index)
                if len(finished) == 2:
                    break
            self.assertEqual(finished, {0, 1})
        finally:
            envs.close()

    def test_worker_and_rendered_autoresets_follow_seed_cycle(self):
        with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            envs = ppo.VecEnv(
                2, {"map_seeds": (2, 7), "shoot_back": True, "max_steps": 1}, seed=9, render=True
            )
            try:
                local = envs.local_env
                if local is None:
                    self.fail("First environment must render")
                for expected, next_seed in [(2, 7), (7, 2), (2, 7)]:
                    _, _, _, truncated, infos, final_obs = envs.step(np.stack([IDLE, IDLE]))
                    self.assertTrue(truncated.all())
                    self.assertTrue(all(info["episode"]["map_seed"] == expected for info in infos))
                    self.assertTrue(all(obs is not None for obs in final_obs))
                    self.assertEqual(local.map_seed, next_seed)
                    np.testing.assert_array_equal(local.layout.boxes, generate_map(next_seed).boxes)
            finally:
                envs.close()

    def test_training_updates_resume_and_watch_cycle(self):
        args = training_random_map.parse_args([
            "--no-dashboard", "--map-seeds", "[2,7]", "--num-envs", "2",
            "--max-steps", "12", "--rollout-steps", "4", "--minibatch-size", "8",
            "--epochs", "1", "--hidden-size", "16", "--total-episodes", "4",
        ])
        with tempfile.TemporaryDirectory() as directory, patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)):
            training_random_map.train(args)
            checkpoint = ppo.load_checkpoint(ppo.resolve_model_path(args.model))
            self.assertGreater(checkpoint["updates"], 0)
            args.total_episodes = checkpoint["episodes"] + 2
            training_random_map.train(args)
            resumed = ppo.load_checkpoint(ppo.resolve_model_path(args.model))
            self.assertGreater(resumed["updates"], checkpoint["updates"])
            self.assertGreater(resumed["global_step"], checkpoint["global_step"])
            watched_seeds = []

            class RecordingRenderer(Renderer):
                def reset(self, **kwargs):
                    result = super().reset(**kwargs)
                    watched_seeds.append(self.map_seed)
                    return result

            args.episodes, args.fps = 3, 0
            with (
                patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}),
                patch.object(ppo, "Renderer", RecordingRenderer),
            ):
                training_random_map.watch(args)
            self.assertEqual(watched_seeds, [2, 7, 2])


if __name__ == "__main__":
    unittest.main()
