"""Front end: the rendered environment and the live training dashboard.

``Renderer`` is a ``HuntEnv`` with a Pygame window, used both to watch/play and
during training. ``MetricsDashboard`` is re-exported for compatibility; its
Streamlit browser dashboard is implemented in ``dashboard.py``.

``python renderer.py`` lets you play the environment yourself: WASD moves relative
to your aim, Q/E or the arrow keys aim, Space fires. ``--stage N`` picks the stage.
``--map-seeds`` selects generated maps with the same seed-cycle rules as the
random-map trainer; combine with ``--shoot-back`` to play the full combat mode.
R toggles discrete rays, T toggles movement trails, and F toggles the frame cap.
Sight regions are not filled; rendering does not change policy observations.
Use ``--headless --screenshot PATH --seed N`` to capture a reproducible frame.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from dashboard import MetricsDashboard
from entities import CONE_RAY_INDICES, MOVE_DIRECTIONS
from environments import STAGES, HuntEnv

BACKGROUND = (20, 27, 35)
WALL_COLOR = (67, 80, 95)
WALL_RAY_COLOR = (76, 109, 121)
CONE_RAY_COLOR = (203, 172, 100)
CONE_RAY_HIT_COLOR = (255, 111, 101)
BULLET_COLOR = (255, 230, 120)
THREAT_BULLET_COLOR = (255, 80, 180)
WAYPOINT_COLOR = (120, 60, 55)
TEXT_COLOR = (220, 220, 220)
HUD_HEIGHT = 132
RENDER_SCALE = 2


class Renderer(HuntEnv):
    """``HuntEnv`` with a Pygame front end.

    Keys: ``R`` toggles discrete rays, ``T`` trails, ``F`` the FPS cap.
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
        self.show_trails = True
        self.cap_fps = fps > 0
        self.window_open = True
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None
        self._arena: pygame.Surface | None = None
        self._floor: pygame.Surface | None = None
        self._walls: pygame.Surface | None = None
        self._trails: tuple[deque[np.ndarray], deque[np.ndarray]] = (
            deque(maxlen=48), deque(maxlen=48),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        result = super().reset(seed=seed, options=options)
        for trail in self._trails:
            trail.clear()
        if self._screen is not None:
            self._prepare_surfaces()
        self.render()
        return result

    def step(self, action):
        result = super().step(action)
        for trail, actor in zip(self._trails, (self.agent, self.threat)):
            if trail and np.linalg.norm(actor.position - trail[-1]) > 80:
                trail.clear()  # Do not connect separate lives after a respawn.
            if not trail or np.linalg.norm(actor.position - trail[-1]) >= 8:
                trail.append(actor.position.copy())
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
            self._small_font = pygame.font.SysFont("dejavusans", 12)
            self._title_font = pygame.font.SysFont("dejavusans", 17, bold=True)
            self._prepare_surfaces()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.show_rays = not self.show_rays
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.cap_fps = not self.cap_fps
            if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                self.show_trails = not self.show_trails

        screen = self._screen
        self._arena.blit(self._floor, (0, 0))
        if self.show_trails:
            self._draw_trails()
        self._arena.blit(self._walls, (0, 0))
        if self.show_rays:
            self._draw_rays()
            if self.threat.waypoint is not None:
                pygame.draw.circle(
                    self._arena, WAYPOINT_COLOR, self._point(self.threat.waypoint),
                    4 * RENDER_SCALE, RENDER_SCALE,
                )
        for bullet in self.bullets:
            self._draw_bullet(bullet.position, bullet.velocity, BULLET_COLOR)
        for bullet in self.threat_bullets:
            self._draw_bullet(bullet.position, bullet.velocity, THREAT_BULLET_COLOR)
        self._draw_body(
            self.threat.position, self.threat.radius, self.threat.color,
            self.threat.health / self.threat.max_health,
            muted=not bool(self.agent.enemy_detected.any()),
        )
        agent = self.agent
        self._draw_body(agent.position, agent.radius, agent.color, agent.health / agent.max_health)
        tip = agent.position + agent.forward * (agent.radius + 12)
        base = agent.position + agent.forward * 8
        arrow = [tip, base + agent.right * 5, base - agent.right * 5]
        pygame.draw.polygon(self._arena, (225, 247, 255), [self._point(p) for p in arrow])
        arena_rect = (0, 0, int(self.layout.width), int(self.layout.height))
        pygame.transform.smoothscale(self._arena, arena_rect[2:], screen.subsurface(arena_rect))
        self._draw_hud()

        pygame.display.flip()
        self._clock.tick((self.fps or self.metadata["render_fps"]) if self.cap_fps else 0)

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
        self.window_open = False

    @staticmethod
    def _point(position) -> tuple[int, int]:
        return round(float(position[0]) * RENDER_SCALE), round(float(position[1]) * RENDER_SCALE)

    def _prepare_surfaces(self) -> None:
        size = self._point((self.layout.width, self.layout.height))
        self._arena = pygame.Surface(size).convert()
        self._floor = pygame.Surface(size).convert()
        self._floor.fill(BACKGROUND)
        for x in range(0, int(self.layout.width), 40):
            pygame.draw.line(self._floor, (26, 34, 43), self._point((x, 0)), self._point((x, self.layout.height)))
        for y in range(0, int(self.layout.height), 40):
            pygame.draw.line(self._floor, (26, 34, 43), self._point((0, y)), self._point((self.layout.width, y)))
        self._walls = pygame.Surface(size, pygame.SRCALPHA)
        for wall in self.layout.walls:
            rect = pygame.Rect(*self._point((wall.x, wall.y)), *self._point((wall.width, wall.height)))
            pygame.draw.rect(self._walls, (9, 15, 22, 130), rect.move(3 * RENDER_SCALE, 4 * RENDER_SCALE))
            pygame.draw.rect(self._walls, WALL_COLOR, rect)
            pygame.draw.rect(self._walls, (98, 115, 131), rect, RENDER_SCALE)
            pygame.draw.line(self._walls, (127, 143, 155), rect.topleft, rect.topright, RENDER_SCALE)

    def _draw_trails(self) -> None:
        for trail, color in zip(self._trails, ((53, 87, 108), (105, 77, 68))):
            for i, position in enumerate(trail):
                radius = 1.0 + 1.8 * (i + 1) / len(trail)
                pygame.draw.circle(self._arena, color, self._point(position), round(radius * RENDER_SCALE))

    def _draw_bullet(self, position: np.ndarray, velocity: np.ndarray, color) -> None:
        end = position - velocity / max(float(np.linalg.norm(velocity)), 1.0) * 12
        streak_color = tuple(round(c * 0.6) for c in color)
        pygame.draw.line(self._arena, streak_color, self._point(end), self._point(position), 3 * RENDER_SCALE)
        pygame.draw.circle(self._arena, color, self._point(position), 2 * RENDER_SCALE)

    def _draw_rays(self) -> None:
        agent = self.agent
        origin = agent.position
        directions = agent.ray_directions()
        for direction, dist in zip(directions, agent.wall_distances):
            pygame.draw.line(self._arena, WALL_RAY_COLOR, self._point(origin), self._point(origin + direction * dist), RENDER_SCALE)
        for k, i in enumerate(CONE_RAY_INDICES):
            if agent.enemy_detected[k]:
                end, color = origin + directions[i] * agent.enemy_distances[k], CONE_RAY_HIT_COLOR
            else:
                end, color = origin + directions[i] * agent.wall_distances[i], CONE_RAY_COLOR
            pygame.draw.line(self._arena, color, self._point(origin), self._point(end), 2 * RENDER_SCALE)

    def _draw_body(self, position: np.ndarray, radius: float, color, health_fraction: float, *, muted: bool = False) -> None:
        if muted:
            color = tuple(round(c * 0.35 + b * 0.65) for c, b in zip(color, BACKGROUND))
        center = self._point(position)
        pygame.draw.circle(self._arena, (10, 17, 24), self._point(position + (2, 3)), round((radius + 3) * RENDER_SCALE))
        pygame.draw.circle(self._arena, (27, 43, 54), center, round((radius + 3) * RENDER_SCALE))
        pygame.draw.circle(self._arena, color, center, round(radius * RENDER_SCALE))
        highlight = tuple(min(255, int(c * 0.6 + (45 if muted else 110))) for c in color)
        pygame.draw.circle(self._arena, highlight, center, round(radius * RENDER_SCALE), RENDER_SCALE)
        top_left = position + np.array([-radius - 2, -radius - 12])
        bar = pygame.Rect(*self._point(top_left), *self._point((radius * 2 + 4, 5)))
        pygame.draw.rect(self._arena, (12, 24, 31), bar.inflate(4, 4), border_radius=5)
        if health_fraction > 0:
            fill = bar.copy()
            fill.width = round(bar.width * np.clip(health_fraction, 0, 1))
            pygame.draw.rect(self._arena, (70, 101, 91) if muted else (114, 223, 173), fill, border_radius=4)

    def _draw_hud(self) -> None:
        y = int(self.layout.height)
        screen = self._screen
        pygame.draw.rect(screen, (13, 19, 27), pygame.Rect(0, y, self.layout.width, HUD_HEIGHT))
        pygame.draw.line(screen, (60, 77, 91), (0, y), (self.layout.width, y))
        stats = self.episode_stats
        replay = " (replay)" if stats.get("replay") else ""
        map_label = f" map {self.map_seed}" if self.map_seed is not None else ""
        screen.blit(self._title_font.render("ADAPTIVE COMBAT AGENT", True, (224, 234, 240)), (18, y + 12))
        detected = bool(self.agent.enemy_detected.any())
        status = "SENSOR CONTACT" if detected else "NO SENSOR CONTACT"
        color = (119, 228, 178) if detected else (152, 169, 183)
        label = self._small_font.render(status, True, color)
        screen.blit(label, (self.layout.width - label.get_width() - 18, y + 15))
        lines = [
            f"STAGE {self.episode_stage}{replay}{map_label}   STEP {self.steps}/{self.max_steps}   "
            f"REWARD {stats.get('reward', 0):+.1f}   KILLS {int(stats.get('kills', 0))}",
            f"HITS {int(stats.get('hits', 0))}/{int(stats.get('shots', 0))}   "
            f"WASTED {int(stats.get('wasted_shots', 0))}   HP {self.agent.health:.0f}/{self.threat.health:.0f}   "
            f"DAMAGE TAKEN {stats.get('damage_taken', 0):.0f}",
        ]
        for i, text in enumerate(lines):
            screen.blit(self._font.render(text, True, TEXT_COLOR), (18, y + 40 + i * 22))
        controls = (
            f"WASD move   Q/E aim   SPACE fire   R rays {'on' if self.show_rays else 'off'}   "
            f"T trails {'on' if self.show_trails else 'off'}   "
            f"F cap {'on' if self.cap_fps else 'off'}   ESC close"
        )
        screen.blit(self._small_font.render(controls, True, (158, 179, 190)), (18, y + 88))
        screen.blit(
            self._small_font.render("Discrete sensor rays only / undetected targets are dimmed for the observer", True, (122, 145, 161)),
            (18, y + 108),
        )


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
    parser.add_argument("--seed", type=int, default=None, help="reproduce initial spawns")
    parser.add_argument("--screenshot", type=Path, help="save the initial frame as PNG, then exit")
    parser.add_argument("--headless", action="store_true", help="capture a screenshot without a window")
    parser.add_argument("--shoot-back", action="store_true", help="play against a shooting threat; death ends the episode")
    parser.add_argument("--map-seeds", type=parse_map_seeds, default=None, help=MAP_SEEDS_HELP + "; omitted: fixed arena")
    args = parser.parse_args()
    if args.headless:
        if args.screenshot is None:
            parser.error("--headless requires --screenshot")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    env = Renderer(
        stage=args.stage, max_steps=args.max_steps, replay_probability=0.0,
        shoot_back=args.shoot_back, map_seeds=args.map_seeds,
    )
    try:
        if args.screenshot is not None:
            env.reset(seed=args.seed)
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(pygame.display.get_surface(), str(args.screenshot))
            print(f"Saved {args.screenshot}")
            return
        for episode in range(args.episodes):
            env.reset(seed=args.seed if episode == 0 else None)
            done = False
            while not done and env.window_open:
                _, _, terminated, truncated, info = env.step(human_action())
                done = terminated or truncated
            if not env.window_open:
                break
            print(f"episode {episode}: {info['episode']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
