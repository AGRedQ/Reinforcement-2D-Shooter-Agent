import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pygame
import yaml
from pygame import Vector2
from stable_baselines3.common.env_checker import check_env

from entities import Wall
from side_implementations import train
from side_implementations.exploration_env import ExplorationEnv, ExplorationSettings


class ExplorationTests(unittest.TestCase):
    def settings(self, **overrides):
        return ExplorationSettings(**(train.load_config()["environment"] | overrides))

    def make_env(self, **overrides):
        env = ExplorationEnv(self.settings(**overrides))
        self.addCleanup(env.close)
        env.reset(seed=7)
        return env

    def test_gym_contract_and_hidden_visit_history(self):
        env = self.make_env()
        check_env(env)
        observation, _ = env.reset(seed=0)
        self.assertEqual(observation.shape, (18,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(len(env.simulation.entities), 1)
        self.assertEqual(env.total_tiles, 192)
        env.visited.update({(0, 0), (1, 0), (2, 0)})
        np.testing.assert_array_equal(observation, env._observation())
        np.testing.assert_allclose(observation[-2:], [
            env.agent.position.x / 800, env.agent.position.y / 600,
        ])

    def test_rays_measure_nearest_walls_boundaries_and_finite_range(self):
        env = self.make_env(ray_range=200)
        env.agent.position = Vector2(100, 100)
        env.simulation.walls = []
        readings = env.ray_distances()
        self.assertEqual(len(readings), 16)
        self.assertEqual(readings[0], 1)
        self.assertEqual(readings[4], 1)
        self.assertAlmostEqual(readings[8], 0.5)
        self.assertAlmostEqual(readings[12], 0.5)
        self.assertAlmostEqual(readings[10], math.sqrt(2) / 2)
        env.simulation.walls = [
            Wall(Vector2(400, 0), Vector2(400, 600)),
            Wall(Vector2(150, 0), Vector2(150, 600)),
            Wall(Vector2(200, 0), Vector2(200, 600)),
        ]
        self.assertEqual(env.ray_distances()[0], 0.25)
        self.assertTrue(env.observation_space.contains(env._observation()))
        env.simulation.walls = [Wall(Vector2(400, 0), Vector2(400, 600))]
        self.assertEqual(env.ray_distances()[0], 1)

    def test_movement_standing_diagonals_and_shared_collisions(self):
        env = self.make_env(max_steps=1000)
        env.simulation.walls = []
        origin = Vector2(100, 100)
        for action, (x, y) in enumerate(env.movement_actions):
            env.agent.position = origin.copy()
            env.step(action)
            expected = Vector2(x, y)
            if expected.length_squared():
                expected.scale_to_length(200 / 60)
            self.assertLess((env.agent.position - origin - expected).length(), 1e-10)
            self.assertEqual(env.agent.shoot_cooldown, 0)
            self.assertEqual((env.agent.aim_x, env.agent.aim_y), (1, 0))
        env.simulation.walls = [Wall(Vector2(150, 0), Vector2(150, 600))]
        env.agent.position = origin.copy()
        for _ in range(60):
            env.step(4)
        self.assertAlmostEqual(env.agent.position.x, 140)
        env.simulation.walls = []
        env.agent.position = Vector2(790, 100)
        env.step(4)
        self.assertEqual(env.agent.position.x, 790)

    def test_first_visits_decay_revisits_and_reset(self):
        env = self.make_env()
        env.simulation.walls = []
        env.agent.position = Vector2(25, 25)
        env.visited = {(0, 0)}
        self.assertEqual(env.step(0)[1], -0.001)
        env.agent.position = Vector2(49, 25)
        _, reward, _, _, info = env.step(4)
        self.assertAlmostEqual(reward, 0.009)
        self.assertEqual(info["exploration_reward"], 0.01)
        env.agent.position = Vector2(99, 25)
        _, reward, _, _, info = env.step(4)
        self.assertAlmostEqual(reward, 0.01 / (1 + 0.5 / 191) - 0.001)
        self.assertLess(info["exploration_reward"], 0.01)
        self.assertEqual(info["new_tiles_visited"], 2)
        env.agent.position = Vector2(25, 25)
        self.assertEqual(env.step(0)[1], -0.001)
        env.visited = {
            (x, y) for x in range(env.tile_columns) for y in range(env.tile_rows)
        } - {(15, 11)}
        env.agent.position = Vector2(775, 575)
        info = env.step(0)[4]
        self.assertAlmostEqual(info["exploration_reward"], 0.01 / (1 + 0.5 * 190 / 191))
        self.assertEqual(info["tile_coverage"], 1)
        _, info = env.reset(seed=7)
        self.assertEqual(info["tiles_visited"], 1)
        self.assertEqual(info["exploration_reward"], 0)
        self.assertEqual(info["episode_reward"], 0)

    def test_fixed_maps_random_maps_spawns_and_seed_logging(self):
        for mode in ("fixed", "random"):
            for random_spawn in (False, True):
                with self.subTest(mode=mode, random_spawn=random_spawn):
                    with tempfile.TemporaryDirectory() as directory:
                        log = Path(directory) / "seeds.jsonl"
                        env = ExplorationEnv(self.settings(
                            map_mode=mode, map_seed=4, random_spawn=random_spawn,
                        ), seed_log=log)
                        self.addCleanup(env.close)
                        sequences = []
                        for _ in range(2):
                            sequence = []
                            for episode in range(4):
                                _, info = env.reset(seed=123 if episode == 0 else None)
                                spawn = env.agent.position.copy()
                                self.assertGreaterEqual(spawn.x, env.agent.radius)
                                self.assertLessEqual(spawn.x, 800 - env.agent.radius)
                                for wall in env.simulation.walls:
                                    edge = wall.end - wall.start
                                    t = max(0, min(1, (spawn - wall.start).dot(edge) / edge.length_squared()))
                                    self.assertGreaterEqual(spawn.distance_to(wall.start + t * edge), env.agent.radius)
                                sequence.append((info["map_seed"], list(spawn)))
                            sequences.append(sequence)
                        self.assertEqual(sequences[0], sequences[1])
                        seeds = {row[0] for row in sequences[0]}
                        self.assertEqual(len(seeds), 4 if mode == "random" else 1)
                        if mode == "fixed":
                            self.assertEqual(seeds, {4})
                            positions = {tuple(row[1]) for row in sequences[0]}
                            self.assertEqual(len(positions), 4 if random_spawn else 1)
                        rows = [json.loads(line) for line in log.read_text().splitlines()]
                        self.assertEqual(len(rows), 8)
                        self.assertEqual(rows[0]["spawn"], sequences[0][0][1])
                        self.assertEqual(rows[0]["map_seed"], sequences[0][0][0])

    def test_limits_and_invalid_inputs(self):
        env = self.make_env(max_steps=2)
        self.assertEqual(env.step(0)[2:4], (False, False))
        self.assertEqual(env.step(0)[2:4], (False, True))
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(0)
        env.reset()
        for action in (-1, 9, 0.5, np.array([1, 2])):
            with self.assertRaises(ValueError):
                env.step(action)
        for overrides in (
            {"ray_range": 0}, {"ray_range": float("inf")}, {"tile_size": 0},
            {"max_steps": -1}, {"random_spawn": "false"}, {"map_mode": "other"},
            {"reward_decay": -0.1}, {"map_seed": True}, {"step_penalty": float("nan")},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.settings(**overrides)
        for size in (10, 30):
            with self.assertRaises(ValueError):
                ExplorationEnv(replace(env.settings, tile_size=size))

    def test_defaults_and_cli_overrides(self):
        with patch.object(train, "train") as run:
            train.main([])
        config, directory = run.call_args.args
        expected = train.load_config()
        expected["run_dir"] = str(Path(expected["run_dir"]))
        self.assertEqual(config, expected)
        self.assertEqual(directory, Path(config["run_dir"]))
        with patch.object(train, "train") as run:
            train.main(["train", "--map-mode", "random", "--random-spawn", "--steps", "1000"])
        config = run.call_args.args[0]
        self.assertEqual(config["environment"]["map_mode"], "random")
        self.assertTrue(config["environment"]["random_spawn"])
        self.assertEqual(config["train"]["steps"], 1000)

    def test_real_ppo_training_saved_evaluation_and_rendering(self):
        config = train.load_config()
        config["environment"].update(map_mode="random", random_spawn=True, max_steps=4)
        config["train"].update(steps=16, seed=3)
        config["train"]["ppo"].update(n_steps=8, batch_size=4, n_epochs=1, verbose=0)
        config["evaluate"]["episodes"] = 2
        config["watch"]["episodes"] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            source = root / "config.yaml"
            source.write_text(yaml.safe_dump(config), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(["train", "--config", str(source), "--run-dir", str(run)])
                train.main(["evaluate", "--config", str(source), "--run-dir", str(run)])
            self.assertTrue((run / "final.zip").exists())
            self.assertTrue((run / "progress.csv").exists())
            self.assertTrue((run / "episodes.monitor.csv").exists())
            saved = json.loads((run / "config.json").read_text())
            self.assertEqual(saved["actual_steps"], 16)
            self.assertEqual(saved["config"]["run_dir"], str(run))
            self.assertEqual(len((run / "map_seeds.jsonl").read_text().splitlines()), 5)
            output = run / "evaluation_10000.json"
            metrics = json.loads(output.read_text())
            self.assertEqual(metrics["episodes"], 2)
            self.assertTrue(all(row["episode_steps"] == 4 for row in metrics["outcomes"]))
            # Saved environment settings win over changes to today's config.
            config["environment"].update(max_steps=10, ray_range=500)
            source.write_text(yaml.safe_dump(config), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(["evaluate", "--config", str(source), "--run-dir", str(run)])
            self.assertEqual(metrics, json.loads(output.read_text()))
            with contextlib.redirect_stdout(io.StringIO()):
                train.main([
                    "evaluate", "--config", str(source), "--run-dir", str(run),
                    "--map-mode", "fixed", "--map-seed", "77", "--no-random-spawn",
                ])
            self.assertEqual(
                [row["map_seed"] for row in json.loads(output.read_text())["outcomes"]],
                [77, 77],
            )
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                train.main(["train", "--config", str(source), "--run-dir", str(run)])
            with (
                patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train.main(["watch", "--config", str(source), "--run-dir", str(run)])
            self.assertFalse(pygame.display.get_init())


if __name__ == "__main__":
    unittest.main()
