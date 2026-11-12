"""Capture reproducible real Pygame frames without a desktop window.

Example: python render_preview.py --output-dir ../artifacts/sight
"""

import argparse
import math
import os
from pathlib import Path
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import pygame

from entities import Wall
from environments import MapLayout
from renderer import Renderer


def capture(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = (
        ("contact", (310., 410.), -0.95, (410., 270.)),
        ("occluded", (310., 400.), 0., (470., 400.)),
        ("range", (135., 160.), math.pi / 4, (300., 340.)),
    )
    for name, origin, angle, target in scenes:
        layout = None
        if name == "range":
            layout = MapLayout(800, 800, [Wall(0, 0, 800, 20), Wall(0, 780, 800, 20),
                                          Wall(0, 0, 20, 800), Wall(780, 0, 20, 800)])
        env = Renderer(layout=layout, fps=0, replay_probability=0, show_rays=False, show_visibility=True)
        try:
            env.reset(seed=7)
            env.agent.reset(np.array(origin), angle)
            env.threat.reset(np.array(target), 0, False)
            env.agent.sense(env.layout.boxes, [env.threat])
            env.render()
            pygame.image.save(env._screen, str(output_dir / f"{name}.png"))
            if name == "contact":
                env.show_rays = True
                env.render()
                pygame.image.save(env._screen, str(output_dir / "sensors.png"))
                env.show_rays = False
                start = time.perf_counter()
                for _ in range(60):
                    env.render()
                print(f"60-frame render average: {(time.perf_counter() - start) * 1000 / 60:.1f} ms")
                for i in range(90):
                    env.step(np.array([3, 1 if i < 40 else 2, int(i % 12 == 0)]))
                pygame.image.save(env._screen, str(output_dir / "movement.png"))
        finally:
            env.close()
    print(f"Saved previews to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sight"))
    capture(parser.parse_args().output_dir)
