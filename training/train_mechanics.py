"""Train or watch mechanics PPO: python -m training.train_mechanics."""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from mechanics_env import MechanicsEnv
from renderer import Renderer
from training.config import load_defaults, parse_configured_args


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return result


def main() -> None:
    defaults = load_defaults("train_mechanics")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train PPO and save the policy.")
    training.add_argument("--steps", type=positive_int, default=str(defaults["train"]["steps"]))
    training.add_argument("--seed", type=int, default=str(defaults["train"]["seed"]))
    watching = commands.add_parser("watch", help="Watch a trained policy.")
    watching.add_argument("--episodes", type=positive_int, default=str(defaults["watch"]["episodes"]))
    watching.add_argument("--seed", type=int, default=str(defaults["watch"]["seed"]))
    for subparser in commands.choices.values():
        subparser.add_argument("--model", type=Path, default=defaults["common"]["model"])
        subparser.add_argument(
            "--no-walls", action="store_true",
            default=defaults["common"]["no_walls"], help="Use the open-arena variant."
        )
        subparser.add_argument("--walls", dest="no_walls", action="store_false")
        subparser.add_argument("--device", choices=("cpu", "cuda"), default=defaults["common"]["device"])
        subparser.add_argument("--max-steps", type=positive_int, default=str(defaults["common"]["max_steps"]))
    args = parse_configured_args(parser, defaults)
    if args.seed < 0:
        parser.error("Seed must be non-negative.")
    if args.model.suffix != ".zip":
        parser.error("--model must have a .zip extension.")

    if not isinstance(args.no_walls, bool):
        parser.error("no_walls must be a boolean.")
    if args.device not in ("cpu", "cuda"):
        parser.error("Device must be cpu or cuda.")
    if type(defaults["num_threads"]) is not int or defaults["num_threads"] <= 0:
        parser.error("num_threads must be a positive integer.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu.")
    torch.set_num_threads(defaults["num_threads"])
    if args.command == "train":
        env = Monitor(MechanicsEnv(use_walls=not args.no_walls, max_steps=args.max_steps))
        try:
            check_env(env.unwrapped)
            model = PPO(
                env=env, seed=args.seed, device=args.device, **defaults["ppo"],
            )
            model.learn(total_timesteps=args.steps, log_interval=defaults["log_interval"])
            args.model.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.model)
        finally:
            env.close()
        print(f"Saved model to {args.model}")
        return

    model = PPO.load(args.model, device=args.device)
    env = MechanicsEnv(render_mode="human", use_walls=not args.no_walls, max_steps=args.max_steps)
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            done = False
            while not done:
                env.render()
                if isinstance(env.simulation, Renderer) and not env.simulation.running:
                    return
                action = model.predict(observation, deterministic=True)[0]
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
