import argparse
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from stable_baselines3 import PPO

from map_generation import generate_map
from training import train_mechanics
from training import wallgen_navpoint_training as training
from training.config import load_defaults


class NavigationTrainingTests(unittest.TestCase):
    def test_seed_selection_validates_scalar_range_and_explicit_arrays(self):
        self.assertEqual(training.map_seed_option("5"), 5)
        self.assertEqual(training.map_seed_option("[5]"), [5])
        for value in ("[]", "[true]", "[1.5]", "true", "1.5", "[-1]", "random"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                training.map_seed_option(value)

    def test_maps_cycle_and_reset_geometry_spawns_and_visit_state(self):
        for selection, expected in (
            (5, [1, 2, 3, 4, 5, 1]),
            ([5], [5] * 6),
            ([7, 2, 9], [7, 2, 9, 7, 2, 9]),
        ):
            env = training.TrainingMapEnv(selection, 4, max_steps=1)
            try:
                for index, selected in enumerate(expected):
                    observation, info = env.reset(seed=42 if index == 0 else None)
                    self.assertEqual(info["map_seed"], selected)
                    self.assertEqual(info["nodes_visited"], 1)
                    self.assertTrue(env.observation_space.contains(observation))
                    layout = generate_map(
                        env.simulation.bounds, agent_radius=env.agent.radius,
                        seed=selected, max_walls=4, columns=4, rows=3,
                    )
                    self.assertEqual(env.layout.edges, layout.edges)
                    self.assertEqual(env.agent.position, layout.nodes[layout.agent_spawn])
                    self.assertEqual(env.target.position, layout.nodes[layout.threat_spawn])
                    self.assertEqual(
                        [(wall.start, wall.end) for wall in env.simulation.walls],
                        [(wall.start, wall.end) for wall in layout.walls],
                    )
                    env.visited.update(range(12))
                    env.step(np.array([0, 2, 0]))
                self.assertEqual(env.reset(seed=42)[1]["map_seed"], expected[0])
            finally:
                env.close()

    def test_random_maps_are_reproducible_and_all_vector_resets_are_logged(self):
        for selection in (0, -10):
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "seeds.jsonl"
                env = training.make_vector_env(
                    selection, 4, n_envs=2, max_steps=1, seed_log=log,
                )
                try:
                    env.seed(123)
                    env.reset()
                    for _ in range(3):
                        env.step(np.array([[0, 2, 0], [0, 2, 0]]))
                    records = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(len(records), 8)
                    for index in range(2):
                        seeds = [row["map_seed"] for row in records if row["env_index"] == index]
                        self.assertEqual(len(set(seeds)), 4)
                        self.assertTrue(all(1 <= seed < 2**31 for seed in seeds))
                        replay = training.TrainingMapEnv(selection, 4, max_steps=1)
                        try:
                            repeated = [
                                replay.reset(seed=123 + index if episode == 0 else None)[1]["map_seed"]
                                for episode in range(4)
                            ]
                            self.assertEqual(seeds, repeated)
                        finally:
                            replay.close()
                finally:
                    env.close()

    def test_vector_environments_stagger_and_repeat_selected_seeds(self):
        env = training.make_vector_env([1, 2, 3], 4, n_envs=3, max_steps=1)
        try:
            env.seed(0)
            env.reset()
            self.assertEqual([info["map_seed"] for info in env.reset_infos], [1, 2, 3])
            env.step(np.array([[0, 2, 0]] * 3))
            self.assertEqual([info["map_seed"] for info in env.reset_infos], [2, 3, 1])
        finally:
            env.close()

    def test_no_arguments_load_yaml_and_cli_overrides_defaults(self):
        defaults = load_defaults("wallgen_navpoint")
        defaults["train"].update(
            steps=40, map_seed=[3, 7], device="cpu", eval_every=40,
        )
        defaults["n_envs"] = 2
        defaults["ppo"].update(n_steps=10, batch_size=10)
        with tempfile.TemporaryDirectory() as directory:
            defaults["common"]["run_dir"] = str(Path(directory) / "run")
            for cli, expected in (([], [3, 7]), (["--map-seed", "5"], 5)):
                with (
                    patch.object(training, "load_defaults", return_value=defaults),
                    patch("sys.argv", ["training", *cli]),
                    patch.object(training, "train") as train,
                ):
                    training.main()
                args = train.call_args.args[0]
                self.assertEqual(args.command, "train")
                self.assertEqual(args.steps, 40)
                self.assertEqual(args.map_seed, expected)
                self.assertEqual(args.ppo, defaults["ppo"])

    def test_mechanics_defaults_and_overrides_reach_ppo_and_environment(self):
        defaults = load_defaults("train_mechanics")
        defaults["train"].update(steps=12, seed=9)
        defaults["common"]["no_walls"] = True
        defaults["common"]["max_steps"] = 15
        defaults["ppo"].update(n_steps=6, batch_size=6)
        with tempfile.TemporaryDirectory() as directory:
            defaults["common"]["model"] = str(Path(directory) / "policy.zip")
            for cli, walls in (([], False), (["--walls"], True)):
                with (
                    patch.object(train_mechanics, "load_defaults", return_value=defaults),
                    patch("sys.argv", ["training", *cli]),
                    patch.object(train_mechanics, "check_env"),
                    patch.object(train_mechanics, "PPO") as ppo,
                    patch.object(train_mechanics, "MechanicsEnv", wraps=train_mechanics.MechanicsEnv) as env,
                ):
                    train_mechanics.main()
                self.assertEqual(env.call_args.kwargs, {"use_walls": walls, "max_steps": 15})
                self.assertEqual(ppo.call_args.kwargs["n_steps"], 6)
                self.assertEqual(ppo.call_args.kwargs["seed"], 9)
                ppo.return_value.learn.assert_called_once_with(total_timesteps=12, log_interval=1)

    def test_small_training_run_saves_seeds_and_can_evaluate_saved_settings(self):
        defaults = load_defaults("wallgen_navpoint")
        defaults["train"].update(
            steps=8, map_seed=[1, 2], max_steps=2, device="cpu",
            eval_every=8, eval_episodes=1,
        )
        defaults["n_envs"] = 2
        defaults["ppo"].update(n_steps=2, batch_size=4, n_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            defaults["common"]["run_dir"] = str(run)
            with (
                patch.object(training, "load_defaults", return_value=defaults),
                patch("sys.argv", ["training"]),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                training.main()
            config = training.load_config(run)
            self.assertEqual(config["actual_steps"], 8)
            self.assertEqual(config["max_steps"], 2)
            self.assertEqual(config["map_seed"], [1, 2])
            records = [json.loads(line) for line in (run / "map_seeds.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual({row["map_seed"] for row in records}, {1, 2})
            for checkpoint in ("final.zip", "best.zip"):
                self.assertTrue((run / checkpoint).is_file())
            evaluations = [
                json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(evaluations), 2)
            for metrics in evaluations:
                self.assertEqual(metrics["requested_episodes"], 1)
                self.assertEqual(metrics["episodes"], 2)
                self.assertEqual(Counter(metrics["map_seeds"]), Counter({1: 1, 2: 1}))
            with (
                patch("sys.argv", ["training", "evaluate", "--run-dir", str(run), "--episodes", "3"]),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                training.main()
            metrics = json.loads((run / "evaluation_final_20000.json").read_text())
            self.assertEqual(metrics["requested_episodes"], 3)
            self.assertEqual(metrics["episodes"], 4)
            self.assertEqual(Counter(metrics["map_seeds"]), Counter({1: 2, 2: 2}))

    def test_legacy_runs_keep_fixed_seed_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config.json").write_text(json.dumps({
                "schema_version": 1, "columns": 4, "rows": 3,
                "observation_size": 66, "map_seed": 0,
            }))
            self.assertEqual(training.load_config(run)["map_seed"], [0])

    def test_random_training_saves_concrete_seeds_and_repeats_evaluation(self):
        defaults = load_defaults("wallgen_navpoint")
        defaults["train"].update(
            steps=8, map_seed=0, max_steps=2, device="cpu",
            eval_every=8, eval_episodes=2,
        )
        defaults["n_envs"] = 2
        defaults["ppo"].update(n_steps=2, batch_size=4, n_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            defaults["common"]["run_dir"] = str(run)
            with (
                patch.object(training, "load_defaults", return_value=defaults),
                patch("sys.argv", ["training"]),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                training.main()
            records = [json.loads(line) for line in (run / "map_seeds.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(len({row["map_seed"] for row in records}), 6)
            self.assertTrue(all(row["map_seed"] > 0 for row in records))
            config = training.load_config(run)
            self.assertEqual(config["actual_steps"], 8)
            self.assertEqual(config["map_seed"], 0)
            model = PPO.load(run / "final.zip", device="cpu")
            env = training.make_vector_env(
                config["map_seed"], config["max_walls"],
                n_envs=config["n_envs"], max_steps=config["max_steps"],
            )
            try:
                first = training.evaluate(model, env, episodes=4, seed=100)
                second = training.evaluate(model, env, episodes=4, seed=100)
                self.assertEqual(first, second)
                self.assertEqual(len(set(first["map_seeds"])), 4)
            finally:
                env.close()

    def test_cuda_request_never_silently_falls_back_to_cpu(self):
        with patch.object(training.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                training.require_device("cuda")
            training.require_device("cpu")

    def test_cli_requires_exact_rollouts_and_does_not_overwrite_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            for args in (
                ["--steps", "500001"],
                ["--eval-every", "50001"],
                ["--run-dir", directory],
                ["--max-walls", "-1"],
            ):
                with (
                    self.subTest(args=args),
                    patch("sys.argv", ["wallgen_navpoint_training", "train", *args]),
                    patch("sys.stderr", new_callable=io.StringIO),
                    self.assertRaises(SystemExit) as raised,
                ):
                    training.main()
                self.assertEqual(raised.exception.code, 2)

    def test_evaluation_seeds_and_records_combat_not_just_reward(self):
        env = training.make_vector_env(0, 4)
        try:
            model = PPO("MlpPolicy", env, device="cpu", seed=0, n_steps=2, batch_size=16)
            first = training.evaluate(model, env, episodes=2, seed=10_000)
            second = training.evaluate(model, env, episodes=2, seed=10_000)
            self.assertEqual(first, second)
            self.assertEqual(first["episodes"], 2)
            self.assertAlmostEqual(first["hit_rate"] + first["timeout_rate"], 1)
            self.assertGreaterEqual(first["mean_node_coverage"], 1 / 12)
            self.assertLessEqual(first["mean_node_coverage"], 1)
        finally:
            env.close()

    def test_evaluation_balances_complete_map_cycles_across_workers(self):
        for selection, episodes, workers, expected in (
            (128, 100, 8, Counter(range(1, 129))),
            (4, 100, 8, Counter({seed: 25 for seed in range(1, 5)})),
            ([7, 2, 9], 7, 2, Counter({7: 3, 2: 3, 9: 3})),
            ([5, 0], 1, 8, Counter({5: 1, 0: 1})),
            ([0], 3, 1, Counter({0: 3})),
            ([7, 7, 2], 4, 3, Counter({7: 4, 2: 2})),
        ):
            with self.subTest(selection=selection, episodes=episodes, workers=workers):
                env = training.make_vector_env(
                    selection, 4, n_envs=workers, max_steps=1,
                )
                try:
                    for index, wrapped in enumerate(env.envs):
                        task = wrapped.unwrapped
                        assert isinstance(task, training.TrainingMapEnv)
                        task.max_steps = 1 + index % 3
                    model = Mock(spec=PPO, num_timesteps=0)
                    model.predict.side_effect = lambda observations, **kwargs: (
                        np.tile([0, 2, 0], (len(observations), 1)), None
                    )
                    offsets = env.get_attr("offset")
                    first = training.evaluate(model, env, episodes=episodes, seed=100)
                    second = training.evaluate(model, env, episodes=episodes, seed=100)
                    self.assertEqual(Counter(first["map_seeds"]), expected)
                    self.assertEqual(first["episodes"], sum(expected.values()))
                    self.assertEqual(first["requested_episodes"], episodes)
                    self.assertEqual(first, second)
                    self.assertEqual(env.get_attr("offset"), offsets)
                    self.assertEqual(first["hit_rate"], 0)
                finally:
                    env.close()

    def test_evaluation_rejects_nonpositive_episode_counts(self):
        env = training.make_vector_env([1], 4, n_envs=1, max_steps=1)
        try:
            for episodes in (0, -1):
                with self.subTest(episodes=episodes), self.assertRaisesRegex(ValueError, "positive integer"):
                    training.evaluate(Mock(spec=PPO), env, episodes=episodes, seed=100)
        finally:
            env.close()

    def test_evaluation_restores_worker_offsets_after_failure(self):
        env = training.make_vector_env(4, 4, n_envs=3, max_steps=1)
        try:
            offsets = env.get_attr("offset")
            with (
                patch.object(training, "evaluate_policy", side_effect=RuntimeError("evaluation failed")),
                self.assertRaisesRegex(RuntimeError, "evaluation failed"),
            ):
                training.evaluate(Mock(spec=PPO), env, episodes=10, seed=100)
            self.assertEqual(env.get_attr("offset"), offsets)
        finally:
            env.close()

    def test_checkpoint_selection_prioritizes_hits_over_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            env = training.make_vector_env(0, 4)
            self.addCleanup(env.close)
            callback = training.EvaluationCheckpoints(env, Path(directory), 50_000, 100)
            model = Mock(spec=PPO)
            callback.model = model
            metrics = {"hit_rate": 0.5, "mean_episode_steps": 400, "mean_reward": 0.1}
            with patch.object(training, "evaluate", return_value=metrics):
                callback.record()
            self.assertEqual(model.save.call_count, 2)
            metrics = {"hit_rate": 0.1, "mean_episode_steps": 500, "mean_reward": 0.9}
            with patch.object(training, "evaluate", return_value=metrics):
                callback.record()
            self.assertEqual(model.save.call_count, 3)
            best = json.loads((Path(directory) / "best_metrics.json").read_text())
            self.assertEqual(best["hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
