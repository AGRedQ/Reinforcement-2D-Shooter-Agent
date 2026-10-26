"""Train, evaluate, or watch the Experiment 1 PPO policy."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from aim_env import AimEnv
from environment import Renderer
from scripted_bot import scripted_action


Policy = Callable[[np.ndarray], np.ndarray]


def evaluate_policy(
    policy: Policy, *, episodes: int = 100, seed: int = 10_000
) -> dict[str, int | float | None]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    env = AimEnv()
    successes = 0
    shots = []
    successful_shots = []
    hit_times = []
    returns = []
    totals = dict.fromkeys(
        ("episode_steps", "shots", "angular_error_sum_degrees", "aligned_steps",
         "aligned_shots", "ready_aligned_steps", "missed_fire_opportunities"), 0.0
    )
    no_shot_episodes = 0
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                observation, reward, terminated, truncated, info = env.step(policy(observation))
                episode_return += reward
                done = terminated or truncated
            shots.append(info["shots"])
            no_shot_episodes += int(info["shots"] == 0)
            for key in totals:
                totals[key] += info[key]
            returns.append(episode_return)
            if info["is_success"]:
                successes += 1
                successful_shots.append(info["shots"])
                hit_times.append(info["elapsed_time"])
    finally:
        env.close()
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_shots": float(np.mean(shots)),
        "mean_shots_on_success": float(np.mean(successful_shots)) if successes else None,
        "mean_time_to_hit_seconds": float(np.mean(hit_times)) if successes else None,
        "mean_return": float(np.mean(returns)),
        "mean_angular_error_degrees": totals["angular_error_sum_degrees"] / totals["episode_steps"],
        "aligned_step_fraction": totals["aligned_steps"] / totals["episode_steps"],
        "aligned_shot_fraction": totals["aligned_shots"] / totals["shots"] if totals["shots"] else None,
        "missed_fire_opportunity_fraction": (
            totals["missed_fire_opportunities"] / totals["ready_aligned_steps"]
            if totals["ready_aligned_steps"] else None
        ),
        "episodes_without_shots": no_shot_episodes,
        "no_shot_episode_fraction": no_shot_episodes / episodes,
        "total_steps": int(totals["episode_steps"]),
        "total_shots": int(totals["shots"]),
        "aligned_steps": int(totals["aligned_steps"]),
        "aligned_shots": int(totals["aligned_shots"]),
        "ready_aligned_steps": int(totals["ready_aligned_steps"]),
        "missed_fire_opportunities": int(totals["missed_fire_opportunities"]),
    }


def evaluate(model: PPO, episodes: int, seed: int, sampling_seed: int = 0) -> dict:
    if sampling_seed < 0:
        raise ValueError("sampling_seed must be non-negative.")
    # Evaluation is CPU-only; preserve the caller's Torch RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sampling_seed)
        stochastic = evaluate_policy(
            lambda obs: model.predict(obs, deterministic=False)[0],
            episodes=episodes, seed=seed,
        )
    rng = np.random.default_rng(seed)
    return {
        "evaluation_seed": seed,
        "sampling_seed": sampling_seed,
        "ppo": evaluate_policy(
            lambda obs: model.predict(obs, deterministic=True)[0],
            episodes=episodes, seed=seed,
        ),
        "ppo_stochastic": stochastic,
        "random": evaluate_policy(
            lambda _obs: rng.uniform(-1, 1, size=3).astype(np.float32),
            episodes=episodes, seed=seed,
        ),
        "scripted": evaluate_policy(scripted_action, episodes=episodes, seed=seed),
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train PPO, save it, then evaluate.")
    training.add_argument("--steps", type=positive_int, default=100_000)
    training.add_argument("--seed", type=int, default=0)
    for name in ("evaluate", "watch"):
        commands.add_parser(name)
    commands.choices["evaluate"].add_argument(
        "--output", type=Path, help="Save the evaluation JSON report to this path."
    )
    commands.choices["watch"].add_argument(
        "--stochastic", action="store_true",
        help="Sample learned actions instead of using deterministic predictions.",
    )
    for name, command in commands.choices.items():
        command.add_argument("--model", type=Path, default=Path("models") / "experiment1.zip")
        command.add_argument("--episodes", type=positive_int, default=5 if name == "watch" else 100)
        command.add_argument("--eval-seed", type=int, default=10_000)
        command.add_argument(
            "--sampling-seed", type=int, default=0,
            help="Seed for sampled actions; independent of scenario seeds.",
        )
    args = parser.parse_args()
    if args.eval_seed < 0 or (args.command == "train" and args.seed < 0):
        parser.error("Seeds must be non-negative.")
    if args.sampling_seed < 0:
        parser.error("--sampling-seed must be non-negative.")
    if args.model.suffix != ".zip":
        parser.error("--model must have a .zip extension.")
    torch.set_num_threads(1)
    if args.command == "train":
        env = Monitor(AimEnv())
        try:
            check_env(env.unwrapped)
            model = PPO(
                "MlpPolicy", env, seed=args.seed, device="cpu",
                n_steps=1024, batch_size=64, verbose=1,
            )
            model.learn(total_timesteps=args.steps)
            args.model.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.model)
        finally:
            env.close()
        print(f"Saved model to {args.model}")
    else:
        model = PPO.load(args.model, device="cpu")
    if args.command == "watch":
        env = AimEnv(render_mode="human")
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.sampling_seed)
                for episode in range(args.episodes):
                    observation, _ = env.reset(seed=args.eval_seed + episode)
                    done = False
                    while not done:
                        env.render()
                        if isinstance(env.simulation, Renderer) and not env.simulation.running:
                            return
                        observation, _, terminated, truncated, _ = env.step(
                            model.predict(observation, deterministic=not args.stochastic)[0]
                        )
                        done = terminated or truncated
                        if done:
                            env.render()
        finally:
            env.close()
    else:
        results = evaluate(model, args.episodes, args.eval_seed, args.sampling_seed)
        report = json.dumps(results, indent=2, allow_nan=False)
        print(report)
        if args.command == "train":
            args.model.with_suffix(".evaluation.json").write_text(report + "\n", encoding="utf-8")
        elif args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()