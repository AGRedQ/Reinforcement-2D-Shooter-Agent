"""Train, watch, or evaluate the isolated movement-only PPO baseline."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from renderer import Renderer
from side_implementations.exploration_env import ExplorationEnv, ExplorationSettings


DEFAULT_CONFIG = Path(__file__).with_name("configs.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Exploration configuration must be a mapping.")
    for section in ("environment", "train", "watch", "evaluate"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}.")
    ExplorationSettings(**config["environment"])
    return config


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer.")
    return number


def train(config: dict[str, Any], run_dir: Path) -> None:
    settings = ExplorationSettings(**config["environment"])
    options = config["train"]
    steps = options["steps"]
    ppo = options["ppo"]
    for name, minimum in (("n_steps", 2), ("batch_size", 2), ("n_epochs", 1)):
        if type(ppo[name]) is not int or ppo[name] < minimum:
            raise ValueError(f"ppo.{name} must be an integer >= {minimum}.")
    if type(options["log_interval"]) is not int or options["log_interval"] < 1:
        raise ValueError("log_interval must be a positive integer.")
    if ppo["n_steps"] % ppo["batch_size"]:
        raise ValueError("batch_size must divide n_steps for this single-environment task.")
    if steps % ppo["n_steps"]:
        raise ValueError("Training steps must be divisible by ppo.n_steps.")
    checker = ExplorationEnv(settings)
    try:
        check_env(checker)
    finally:
        checker.close()
    run_dir.mkdir(parents=True, exist_ok=False)
    env = Monitor(
        ExplorationEnv(settings, seed_log=run_dir / "map_seeds.jsonl"),
        filename=str(run_dir / "episodes"),
        info_keywords=("map_seed", "tiles_visited", "tile_coverage"),
    )
    logger = None
    saved = {"schema_version": 1, "config": config, "actual_steps": 0}
    try:
        (run_dir / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
        model = PPO(env=env, seed=options["seed"], device=options["device"], **ppo)
        logger = configure(str(run_dir), ["stdout", "csv"])
        model.set_logger(logger)
        model.learn(total_timesteps=steps, log_interval=options["log_interval"])
        model.save(run_dir / "final.zip")
        saved["actual_steps"] = model.num_timesteps
        (run_dir / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
        print(f"Saved movement-only policy after {model.num_timesteps} steps to {run_dir}.")
    finally:
        env.close()
        if logger is not None:
            logger.close()


def rollout(
    config: dict[str, Any], run_dir: Path, command: str,
    overrides: dict[str, Any],
) -> dict[str, Any] | None:
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if saved["schema_version"] != 1:
        raise ValueError("Unsupported exploration checkpoint configuration.")
    settings = ExplorationSettings(**(saved["config"]["environment"] | overrides))
    options = config[command]
    env = ExplorationEnv(settings, render_mode="human" if command == "watch" else None)
    outcomes = []
    try:
        model = PPO.load(run_dir / options["checkpoint"], env=env, device=options["device"])
        for episode in range(options["episodes"]):
            observation, _ = env.reset(seed=options["seed"] if episode == 0 else None)
            spawn = list(env.agent.position)
            done = False
            while not done:
                if command == "watch":
                    env.render()
                    if isinstance(env.simulation, Renderer) and not env.simulation.running:
                        return None
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated
            outcomes.append(info | {"spawn": spawn})
        metrics = {
            "episodes": len(outcomes), "seed": options["seed"],
            "environment": asdict(settings),
            "mean_tiles_visited": float(np.mean([row["tiles_visited"] for row in outcomes])),
            "mean_tile_coverage": float(np.mean([row["tile_coverage"] for row in outcomes])),
            "mean_reward": float(np.mean([row["episode_reward"] for row in outcomes])),
            "outcomes": outcomes,
        }
        if command == "evaluate":
            output = run_dir / f"evaluation_{options['seed']}.json"
            output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return metrics
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("train", "watch", "evaluate"), default="train")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--episodes", type=positive_int)
    parser.add_argument("--seed", type=nonnegative_int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--map-mode", choices=("fixed", "random"))
    parser.add_argument("--map-seed", type=nonnegative_int)
    parser.add_argument("--random-spawn", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.command == "train" and args.episodes is not None:
        parser.error("--episodes is only for watch/evaluate.")
    if args.command != "train" and args.steps is not None:
        parser.error("--steps is only for train.")
    config = load_config(args.config)
    options = config[args.command]
    for name in ("steps", "episodes", "seed", "device"):
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    for name, minimum in (
        ("seed", 0), ("steps" if args.command == "train" else "episodes", 1),
    ):
        if type(options[name]) is not int or options[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if type(config["num_threads"]) is not int or config["num_threads"] < 1:
        raise ValueError("num_threads must be a positive integer.")
    if options["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if options["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    torch.set_num_threads(config["num_threads"])
    overrides = {
        name: getattr(args, name) for name in ("map_mode", "map_seed", "random_spawn")
        if getattr(args, name) is not None
    }
    run_dir = args.run_dir if args.run_dir is not None else Path(config["run_dir"])
    if args.command == "train":
        config["run_dir"] = str(run_dir)
        config["environment"].update(overrides)
        train(config, run_dir)
    else:
        rollout(config, run_dir, args.command, overrides)


if __name__ == "__main__":
    main()
