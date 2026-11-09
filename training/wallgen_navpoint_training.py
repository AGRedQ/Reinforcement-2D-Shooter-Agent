"""Train, evaluate, or watch PPO on cycling or randomly generated maps."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from renderer import Renderer
from training.config import load_defaults, parse_configured_args
from training.train_mechanics import positive_int
from wallgen_navpoint_env import WallgenNavpointEnv


def non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Must be a non-negative integer.")
    return result


def map_seed_option(value: str) -> int | list[int]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("Use an integer or a JSON seed array, e.g. [1,2,3].") from error
    if type(result) is int:
        return result
    if (
        isinstance(result, list) and result
        and all(type(seed) is int and seed >= 0 for seed in result)
    ):
        return result
    raise argparse.ArgumentTypeError("Map seeds must be an integer or a nonempty array of non-negative integers.")


class TrainingMapEnv(WallgenNavpointEnv):
    """Cycle selected seeds, or generate a new seeded layout on every reset."""

    def __init__(
        self,
        map_seed: int | list[int],
        max_walls: int,
        *,
        max_steps: int,
        offset: int = 0,
        record_seed: Callable[[int], None] | None = None,
        render_mode: str | None = None,
    ):
        selection = map_seed_option(json.dumps(map_seed))
        self.map_seeds = (
            selection if isinstance(selection, list)
            else range(1, selection + 1) if selection > 0 else None
        )
        self.offset = offset
        self.episode_index = 0
        self.map_rng = random.Random()
        self.record_seed = record_seed
        initial_seed = self.map_seeds[offset % len(self.map_seeds)] if self.map_seeds else 0
        super().__init__(
            render_mode=render_mode, max_steps=max_steps,
            map_seed=initial_seed, max_walls=max_walls,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.episode_index = 0
            self.map_rng.seed(seed)
        selected = (
            self.map_seeds[(self.offset + self.episode_index) % len(self.map_seeds)]
            if self.map_seeds else self.map_rng.randrange(1, 2**31)
        )
        self.set_map(selected)
        observation, info = super().reset(seed=seed, options=options)
        self.episode_index += 1
        if self.record_seed is not None:
            self.record_seed(selected)
        return observation, info


def require_device(device: str) -> None:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Install requirements-cuda.txt "
                "in this interpreter or explicitly select --device cpu."
            )
        probe = torch.ones(1, device="cuda") * 2
        torch.cuda.synchronize()
        if probe.item() != 2:
            raise RuntimeError("CUDA computation check failed.")
        print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}", flush=True)


def make_vector_env(
    map_seed: int | list[int], max_walls: int, *,
    n_envs: int | None = None, max_steps: int | None = None,
    seed_log: Path | None = None,
) -> DummyVecEnv:
    defaults = load_defaults("wallgen_navpoint")
    count = defaults["n_envs"] if n_envs is None else n_envs
    steps = defaults["train"]["max_steps"] if max_steps is None else max_steps

    def factory(index: int) -> Callable[[], Monitor]:
        episode = 0

        def record(map_seed: int) -> None:
            nonlocal episode
            if seed_log is not None:
                with seed_log.open("a", encoding="utf-8") as file:
                    file.write(json.dumps({
                        "env_index": index, "episode": episode, "map_seed": map_seed,
                    }) + "\n")
            episode += 1

        return lambda: Monitor(TrainingMapEnv(
            map_seed, max_walls, max_steps=steps, offset=index,
            record_seed=record if seed_log is not None else None,
        ))

    return DummyVecEnv([factory(index) for index in range(count)])


def evaluate(
    model: PPO,
    env: DummyVecEnv,
    *,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("Evaluation episodes must be a positive integer.")
    tasks: list[TrainingMapEnv] = []
    for wrapped in env.envs:
        task = wrapped.unwrapped
        if not isinstance(task, TrainingMapEnv):
            raise TypeError("Expected the navigation training environment.")
        tasks.append(task)
    map_seeds = tasks[0].map_seeds
    if any(task.map_seeds != map_seeds for task in tasks):
        raise ValueError("Evaluation workers must share the same map selection.")
    total_episodes = episodes
    if map_seeds is not None:
        total_episodes = ((episodes + len(map_seeds) - 1) // len(map_seeds)) * len(map_seeds)
    outcomes: list[dict[str, Any]] = []

    def collect(local: dict[str, Any], _global: dict[str, Any]) -> None:
        if local["done"]:
            outcomes.append(local["info"])

    offsets = [task.offset for task in tasks]
    try:
        if map_seeds is not None:
            offset = 0
            for index, task in enumerate(tasks):
                task.offset = offset
                # Match SB3's static per-worker quotas with contiguous map-cycle slices.
                offset += (total_episodes + index) // env.num_envs
        env.seed(seed)
        rewards, lengths = evaluate_policy(
            model, env, n_eval_episodes=total_episodes, deterministic=True,
            callback=collect, return_episode_rewards=True,
        )
    finally:
        for task, offset in zip(tasks, offsets):
            task.offset = offset
    successful = [info for info in outcomes if info["is_success"]]
    return {
        "timesteps": model.num_timesteps,
        "requested_episodes": episodes,
        "episodes": len(outcomes),
        "seed": seed,
        "map_seeds": [info["map_seed"] for info in outcomes],
        "hit_rate": len(successful) / len(outcomes),
        "timeout_rate": 1 - len(successful) / len(outcomes),
        "mean_episode_steps": float(np.mean(lengths)),
        "mean_hit_steps": (
            float(np.mean([info["episode_steps"] for info in successful]))
            if successful else None
        ),
        "mean_reward": float(np.mean(rewards)),
        "mean_node_coverage": float(np.mean([info["node_coverage"] for info in outcomes])),
        "sighting_rate": float(np.mean([
            info["target_visibility_reward_earned"] or info["is_success"]
            for info in outcomes
        ])),
    }


class EvaluationCheckpoints(BaseCallback):
    def __init__(
        self, env: DummyVecEnv, run_dir: Path, every: int, episodes: int,
        seed: int | None = None,
    ):
        super().__init__()
        self.eval_env = env
        self.run_dir = run_dir
        self.every = every
        self.episodes = episodes
        self.seed = load_defaults("wallgen_navpoint")["eval_seed"] if seed is None else seed
        self.next_evaluation = 0
        self.best_score = (-1.0, float("-inf"))
        self.last_evaluation = -1

    def _on_rollout_start(self) -> None:
        if self.num_timesteps >= self.next_evaluation:
            self.record()
            self.next_evaluation = self.num_timesteps + self.every

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps != self.last_evaluation:
            self.record()

    def record(self) -> None:
        if not isinstance(self.model, PPO):
            raise TypeError("Navigation evaluations require a PPO model.")
        metrics = evaluate(
            self.model, self.eval_env, episodes=self.episodes, seed=self.seed
        )
        self.last_evaluation = self.num_timesteps
        self.model.save(self.run_dir / f"step_{self.num_timesteps}.zip")
        score = (metrics["hit_rate"], -metrics["mean_episode_steps"])
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.run_dir / "best.zip")
            (self.run_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
        with (self.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")
        print(f"Evaluation: {json.dumps(metrics)}", flush=True)


def train(args: argparse.Namespace) -> None:
    require_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    env = make_vector_env(
        args.map_seed, args.max_walls, n_envs=args.n_envs, max_steps=args.max_steps,
        seed_log=args.run_dir / "map_seeds.jsonl",
    )
    eval_env = None
    logger = None
    try:
        checker = TrainingMapEnv(args.map_seed, args.max_walls, max_steps=args.max_steps)
        try:
            check_env(checker)
        finally:
            checker.close()
        eval_env = make_vector_env(
            args.map_seed, args.max_walls, n_envs=args.n_envs, max_steps=args.max_steps
        )
        model = PPO(
            env=env, seed=args.seed, device=args.device, **args.ppo,
        )
        config = {
            "schema_version": 2,
            "map_seed": args.map_seed,
            "max_walls": args.max_walls,
            "max_steps": args.max_steps,
            "columns": WallgenNavpointEnv.columns,
            "rows": WallgenNavpointEnv.rows,
            "observation_size": WallgenNavpointEnv.observation_size,
            "seed": args.seed,
            "requested_steps": args.steps,
            "device": str(model.device),
            "torch_version": torch.__version__,
            "n_envs": args.n_envs,
            "n_steps": args.ppo["n_steps"],
            "batch_size": args.ppo["batch_size"],
            "ppo": args.ppo,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "num_threads": args.num_threads,
            "log_interval": args.log_interval,
            "logger_formats": args.logger_formats,
            "seed_log": "map_seeds.jsonl",
        }
        (args.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        layout_env = env.envs[0].unwrapped
        if not isinstance(layout_env, WallgenNavpointEnv):
            raise TypeError("Expected the navigation environment.")
        layout = layout_env.layout
        (args.run_dir / "map.json").write_text(json.dumps({
            "map_seed": layout_env.map_seed,
            "initial_preview_only": True,
            "nodes": [list(node) for node in layout.nodes],
            "edges": layout.edges,
            "walls": [[list(wall.start), list(wall.end)] for wall in layout.walls],
            "agent_spawn": layout.agent_spawn,
            "threat_spawn": layout.threat_spawn,
        }, indent=2), encoding="utf-8")
        logger = configure(str(args.run_dir), args.logger_formats)
        model.set_logger(logger)
        callback = EvaluationCheckpoints(
            eval_env, args.run_dir, every=args.eval_every, episodes=args.eval_episodes,
            seed=args.eval_seed,
        )
        print(
            f"Training {args.steps} steps on {model.device}; "
            f"map selection {args.map_seed}; {len(layout.nodes)} nodes; outputs: {args.run_dir}",
            flush=True,
        )
        model.learn(total_timesteps=args.steps, callback=callback, log_interval=args.log_interval)
        model.save(args.run_dir / "final.zip")
        config["actual_steps"] = model.num_timesteps
        (args.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Saved final policy after {model.num_timesteps} steps.", flush=True)
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
        if logger is not None:
            logger.close()


def load_config(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if (
        config["schema_version"] not in (1, 2)
        or config["columns"] != WallgenNavpointEnv.columns
        or config["rows"] != WallgenNavpointEnv.rows
        or config["observation_size"] != WallgenNavpointEnv.observation_size
    ):
        raise ValueError("This run uses an incompatible navigation observation format.")
    if config["schema_version"] == 1:
        # Legacy scalars named one fixed map, including seed zero.
        config["map_seed"] = [config["map_seed"]]
        config["max_steps"] = 600
    return config


def watch_or_evaluate(args: argparse.Namespace) -> None:
    require_device(args.device)
    config = load_config(args.run_dir)
    model = PPO.load(args.run_dir / args.checkpoint, device=args.device)
    if args.command == "evaluate":
        env = make_vector_env(
            config["map_seed"], config["max_walls"],
            n_envs=config["n_envs"], max_steps=config["max_steps"],
        )
        try:
            metrics = evaluate(model, env, episodes=args.episodes, seed=args.seed)
            output = args.run_dir / f"evaluation_{Path(args.checkpoint).stem}_{args.seed}.json"
            output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(json.dumps(metrics, indent=2))
        finally:
            env.close()
        return

    watching = TrainingMapEnv(
        render_mode="human", map_seed=config["map_seed"], max_walls=config["max_walls"],
        max_steps=config["max_steps"],
    )
    try:
        for episode in range(args.episodes):
            observation, _ = watching.reset(seed=args.seed if episode == 0 else None)
            done = False
            while not done:
                watching.render()
                if isinstance(watching.simulation, Renderer) and not watching.simulation.running:
                    return
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = watching.step(action)
                done = terminated or truncated
            watching.render()
    finally:
        watching.close()


def main() -> None:
    defaults = load_defaults("wallgen_navpoint")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    for option, converter in (
        ("steps", positive_int), ("seed", non_negative_int),
        ("map_seed", map_seed_option), ("max_walls", non_negative_int),
        ("max_steps", positive_int), ("eval_every", positive_int),
        ("eval_episodes", positive_int),
    ):
        value = defaults["train"][option]
        training.add_argument(
            "--" + option.replace("_", "-"), type=converter,
            default=json.dumps(value) if option == "map_seed" else str(value),
        )
    for command in ("watch", "evaluate"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--episodes", type=positive_int, default=str(defaults[command]["episodes"]))
        subparser.add_argument("--seed", type=non_negative_int, default=str(defaults[command]["seed"]))
        subparser.add_argument("--checkpoint", choices=("final.zip", "best.zip"), default=defaults[command]["checkpoint"])
    for command, subparser in commands.choices.items():
        subparser.add_argument("--run-dir", type=Path, default=defaults["common"]["run_dir"])
        subparser.add_argument("--device", choices=("cpu", "cuda"), default=defaults[command]["device"])
        subparser.set_defaults(num_threads=defaults["num_threads"])
    training.set_defaults(**{
        key: defaults[key]
        for key in ("n_envs", "ppo", "eval_seed", "log_interval", "logger_formats")
    })
    args = parse_configured_args(parser, defaults)
    if args.device not in ("cpu", "cuda"):
        parser.error("Device must be cpu or cuda.")
    if type(args.num_threads) is not int or args.num_threads <= 0:
        parser.error("num_threads must be a positive integer.")
    if args.command != "train" and args.checkpoint not in ("final.zip", "best.zip"):
        parser.error("Checkpoint must be final.zip or best.zip.")
    if args.command == "train":
        for name, value in (
            ("n_envs", args.n_envs), ("n_steps", args.ppo["n_steps"]),
            ("batch_size", args.ppo["batch_size"]), ("log_interval", args.log_interval),
        ):
            if type(value) is not int or value <= 0:
                parser.error(f"{name} must be a positive integer.")
        rollout_size = args.n_envs * args.ppo["n_steps"]
        if args.ppo["batch_size"] <= 1 or rollout_size <= 1 or rollout_size % args.ppo["batch_size"]:
            parser.error("PPO requires a rollout and batch size > 1, with batches dividing the rollout.")
        if type(args.eval_seed) is not int or args.eval_seed < 0:
            parser.error("eval_seed must be a non-negative integer.")
        if args.steps % rollout_size or args.eval_every % rollout_size:
            parser.error(f"--steps and --eval-every must be multiples of {rollout_size}.")
        if args.run_dir.exists():
            parser.error("Run directory already exists; choose a new --run-dir to avoid overwriting.")
    torch.set_num_threads(args.num_threads)
    if args.command == "train":
        train(args)
    else:
        watch_or_evaluate(args)


if __name__ == "__main__":
    main()
