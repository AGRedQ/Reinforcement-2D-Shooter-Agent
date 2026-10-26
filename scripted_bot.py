"""Scripted Experiment 1 aimer and standalone Pygame playback."""

import argparse
import math

import numpy as np

from aim_env import AimEnv
from renderer import Renderer


def scripted_action(observation: np.ndarray) -> np.ndarray:
    """Aim directly at the target and only fire when the beam is aligned."""
    offset = observation[:2].astype(np.float64)
    distance = float(np.linalg.norm(offset))
    direction = offset / distance
    aim = observation[2:4]
    error = abs(math.atan2(
        aim[0] * direction[1] - aim[1] * direction[0],
        float(np.dot(aim, direction)),
    ))
    # Offsets use the 800x600 world's diagonal; target radius is 10 pixels.
    tolerance = math.asin(min(1.0, 10 / (distance * 1000)))
    return np.array(
        [direction[0], direction[1], 1 if error <= tolerance else -1],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.eval_seed < 0:
        parser.error("--eval-seed must be non-negative.")

    env = AimEnv(render_mode="human")
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.eval_seed + episode)
            done = False
            while not done:
                env.render()
                if isinstance(env.simulation, Renderer) and not env.simulation.running:
                    return
                observation, _, terminated, truncated, _ = env.step(
                    scripted_action(observation)
                )
                done = terminated or truncated
                if done:
                    env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()