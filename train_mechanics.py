"""Train or watch a recurrent PPO policy in the discrete-action mechanics environment."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from mechanics_env import MechanicsEnv
from renderer import Renderer


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train recurrent PPO and save the policy.")
    training.add_argument("--steps", type=positive_int, default=100_000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--model", type=Path, default=Path("models/mechanics_lstm.zip"))
    watching = commands.add_parser("watch", help="Watch a trained policy.")
    watching.add_argument("--model", type=Path, default=Path("models/mechanics_lstm.zip"))
    watching.add_argument("--episodes", type=positive_int, default=5)
    watching.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be non-negative.")
    if args.model.suffix != ".zip":
        parser.error("--model must have a .zip extension.")

    torch.set_num_threads(1)
    if args.command == "train":
        env = Monitor(MechanicsEnv())
        try:
            check_env(env.unwrapped)
            model = RecurrentPPO(
                "MlpLstmPolicy", env, seed=args.seed, device="cpu",
                n_steps=1024, batch_size=64, verbose=1,
            )
            model.learn(total_timesteps=args.steps)
            args.model.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.model)
        finally:
            env.close()
        print(f"Saved model to {args.model}")
        return

    model = RecurrentPPO.load(args.model, device="cpu")
    env = MechanicsEnv(render_mode="human")
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            done = False
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            while not done:
                env.render()
                if isinstance(env.simulation, Renderer) and not env.simulation.running:
                    return
                action, lstm_states = model.predict(
                    observation,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_starts = np.array([done], dtype=bool)
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
