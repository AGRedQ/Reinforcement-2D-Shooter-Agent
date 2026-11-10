"""Front end: the rendered environment and the live training dashboard.

``Renderer`` is a ``HuntEnv`` with a Pygame window, used both to watch/play and
during training. ``MetricsDashboard`` is re-exported for compatibility; its
Streamlit browser dashboard is implemented in ``dashboard.py``.

``python renderer.py`` lets you play the environment yourself: WASD moves relative
to your aim, Q/E or the arrow keys aim, Space fires. ``--stage N`` picks the stage.
``--map-seeds`` selects generated maps with the same seed-cycle rules as the
random-map trainer; combine with ``--shoot-back`` to play the full combat mode.
"""

from __future__ import annotations

import argparse
import numpy as np
import pygame

from dashboard import MetricsDashboard
from entities import CONE_RAY_INDICES, MOVE_DIRECTIONS
from environments import STAGES, HuntEnv

BACKGROUND = (24, 26, 32)
WALL_COLOR = (110, 115, 130)
WALL_RAY_COLOR = (70, 90, 70)
CONE_RAY_COLOR = (200, 200, 90)
CONE_RAY_HIT_COLOR = (255, 60, 60)
BULLET_COLOR = (255, 230, 120)
THREAT_BULLET_COLOR = (255, 80, 180)
WAYPOINT_COLOR = (120, 60, 55)
TEXT_COLOR = (220, 220, 220)
HUD_HEIGHT = 56


class Renderer(HuntEnv):
    """``HuntEnv`` with a Pygame front end.

    Keys while the window is focused: ``R`` toggles rays, ``F`` toggles the FPS cap.
    Closing the window stops rendering but keeps the environment running, so
    training is never interrupted by it.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, *args, fps: int = 30, render_every: int = 1, show_rays: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_mode = "human"
        self.fps = fps
        self.render_every = max(1, render_every)
        self.show_rays = show_rays
        self.cap_fps = fps > 0
        self.window_open = True
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        result = super().reset(seed=seed, options=options)
        self.render()
        return result

    def step(self, action):
        result = super().step(action)
        if self.steps % self.render_every == 0 or result[2] or result[3]:
            self.render()
        return result

    def render(self) -> None:
        if not self.window_open:
            return
        if self._screen is None:
            pygame.init()
            size = (int(self.layout.width), int(self.layout.height) + HUD_HEIGHT)
            self._screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Adaptive Combat Agent")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 16)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.show_rays = not self.show_rays
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.cap_fps = not self.cap_fps

        screen = self._screen
        screen.fill(BACKGROUND)
        for wall in self.layout.walls:
            pygame.draw.rect(screen, WALL_COLOR, pygame.Rect(wall.x, wall.y, wall.width, wall.height))
        if self.show_rays:
            self._draw_rays()
            if self.threat.waypoint is not None:
                pygame.draw.circle(screen, WAYPOINT_COLOR, self.threat.waypoint, 4)
        for bullet in self.bullets:
            pygame.draw.circle(screen, BULLET_COLOR, bullet.position, 3)
        for bullet in self.threat_bullets:
            pygame.draw.circle(screen, THREAT_BULLET_COLOR, bullet.position, 3)
        self._draw_body(self.threat.position, self.threat.radius, self.threat.color, self.threat.health / self.threat.max_health)
        agent = self.agent
        self._draw_body(agent.position, agent.radius, agent.color, agent.health / agent.max_health)
        pygame.draw.line(screen, (255, 255, 255), agent.position, agent.position + agent.forward * (agent.radius + 10), 3)
        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(self.fps if self.cap_fps else 0)

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
        self.window_open = False

    def _draw_rays(self) -> None:
        agent = self.agent
        origin = agent.position
        directions = agent.ray_directions()
        for direction, dist in zip(directions, agent.wall_distances):
            pygame.draw.line(self._screen, WALL_RAY_COLOR, origin, origin + direction * dist, 1)
        for k, i in enumerate(CONE_RAY_INDICES):
            if agent.enemy_detected[k]:
                end, color = origin + directions[i] * agent.enemy_distances[k], CONE_RAY_HIT_COLOR
            else:
                end, color = origin + directions[i] * agent.wall_distances[i], CONE_RAY_COLOR
            pygame.draw.line(self._screen, color, origin, end, 2)

    def _draw_body(self, position: np.ndarray, radius: float, color, health_fraction: float) -> None:
        pygame.draw.circle(self._screen, color, position, radius)
        top_left = position + np.array([-radius, -radius - 9])
        pygame.draw.rect(self._screen, (60, 20, 20), pygame.Rect(*top_left, radius * 2, 4))
        pygame.draw.rect(self._screen, (80, 220, 90), pygame.Rect(*top_left, radius * 2 * health_fraction, 4))

    def _draw_hud(self) -> None:
        y = self.layout.height
        pygame.draw.rect(self._screen, (16, 17, 21), pygame.Rect(0, y, self.layout.width, HUD_HEIGHT))
        stats = self.episode_stats
        replay = " (replay)" if stats.get("replay") else ""
        map_label = f" map {self.map_seed}" if self.map_seed is not None else ""
        lines = [
            f"stage {self.episode_stage}{replay}{map_label}   step {self.steps}/{self.max_steps}   "
            f"reward {stats.get('reward', 0.0):+.1f}   kills {int(stats.get('kills', 0))}",
            f"hits {int(stats.get('hits', 0))}/{int(stats.get('shots', 0))}   "
            f"wasted {int(stats.get('wasted_shots', 0))}   HP {self.agent.health:.0f}/{self.threat.health:.0f}   "
            f"[R] rays  [F] fps cap {'on' if self.cap_fps else 'off'}",
        ]
        for i, text in enumerate(lines):
            self._screen.blit(self._font.render(text, True, TEXT_COLOR), (10, y + 8 + i * 22))


def human_action() -> np.ndarray:
    """Keyboard action: WASD move (relative to aim), Q/E or arrows aim, Space fire."""
    keys = pygame.key.get_pressed()
    forward = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
    right = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
    move = 0
    if forward or right:
        target = np.array([forward, right], dtype=np.float64)
        move = int(np.argmax(MOVE_DIRECTIONS[1:] @ (target / np.linalg.norm(target)))) + 1
    aim = 1 + int(keys[pygame.K_e] or keys[pygame.K_RIGHT]) - int(keys[pygame.K_q] or keys[pygame.K_LEFT])
    return np.array([move, aim, int(keys[pygame.K_SPACE])], dtype=np.int64)


def main() -> None:
    from map_generation import MAP_SEEDS_HELP, parse_map_seeds

    parser = argparse.ArgumentParser(description="Play the hunting environment yourself.")
    parser.add_argument("--stage", type=int, default=0, choices=range(len(STAGES)))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--shoot-back", action="store_true", help="play against a shooting threat; death ends the episode")
    parser.add_argument("--map-seeds", type=parse_map_seeds, default=None, help=MAP_SEEDS_HELP + "; omitted: fixed arena")
    args = parser.parse_args()

    env = Renderer(
        stage=args.stage, max_steps=args.max_steps, replay_probability=0.0,
        shoot_back=args.shoot_back, map_seeds=args.map_seeds,
    )
    for episode in range(args.episodes):
        env.reset()
        done = False
        while not done and env.window_open:
            _, _, terminated, truncated, info = env.step(human_action())
            done = terminated or truncated
        if not env.window_open:
            break
        print(f"episode {episode}: {info['episode']}")
    env.close()


if __name__ == "__main__":
    main()
