"""Front end: the rendered environment and the live training dashboard.

``Renderer`` is a ``HuntEnv`` with a Pygame window, used both to watch/play and
during training. ``MetricsDashboard`` plots training metrics live in a separate
Matplotlib window running in its own process (so it never stalls training) and has
a "Stop" button that asks training to end after the current episodes.

``python renderer.py`` lets you play the environment yourself: WASD moves relative
to your aim, Q/E or the arrow keys aim, Space fires. ``--stage N`` picks the stage.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import signal
from collections import deque

import numpy as np
import pygame

from entities import CONE_RAY_INDICES, MOVE_DIRECTIONS
from environments import STAGES, HuntEnv

BACKGROUND = (24, 26, 32)
WALL_COLOR = (110, 115, 130)
WALL_RAY_COLOR = (70, 90, 70)
CONE_RAY_COLOR = (200, 200, 90)
CONE_RAY_HIT_COLOR = (255, 60, 60)
BULLET_COLOR = (255, 230, 120)
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
        if self.steps % self.render_every == 0 or result[3]:
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
        lines = [
            f"stage {self.episode_stage}{replay}   step {self.steps}/{self.max_steps}   "
            f"reward {stats.get('reward', 0.0):+.1f}   kills {int(stats.get('kills', 0))}",
            f"hits {int(stats.get('hits', 0))}/{int(stats.get('shots', 0))}   "
            f"wasted {int(stats.get('wasted_shots', 0))}   threat HP {self.threat.health:.0f}   "
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


# --------------------------------------------------------------------------- live dashboard


class MetricsDashboard:
    """Live training graphs in a separate window/process, with a stop button.

    ``metrics`` maps metric keys to subplot titles; subplots are laid out in that
    order. ``reference_lines`` draws dashed horizontal lines (e.g. a promotion
    threshold). Create it under ``if __name__ == "__main__":`` (spawned process).
    """

    def __init__(
        self,
        metrics: dict[str, str],
        reference_lines: dict[str, float] | None = None,
        title: str = "Training dashboard",
        max_points: int = 5000,
        smoothing: int = 10,
        columns: int = 4,
    ):
        self.metrics = dict(metrics)
        self.reference_lines = dict(reference_lines or {})
        self.title = title
        self.max_points = max_points
        self.smoothing = smoothing
        self.columns = columns
        self._queue: mp.Queue | None = None
        self._stop_event = None
        self._process: mp.Process | None = None

    def start(self) -> MetricsDashboard:
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._stop_event = ctx.Event()
        self._process = ctx.Process(
            target=_dashboard_loop,
            args=(
                self._queue,
                self._stop_event,
                self.title,
                self.metrics,
                self.reference_lines,
                self.max_points,
                self.smoothing,
                self.columns,
            ),
            daemon=True,
        )
        self._process.start()
        return self

    @property
    def is_open(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def log(self, step: int, **metrics: float) -> None:
        """Send metric values for ``step``. Ignored once the window has been closed."""
        unknown = set(metrics) - set(self.metrics)
        if unknown:
            raise KeyError(f"Unknown dashboard metrics: {sorted(unknown)}")
        if self.is_open:
            self._queue.put(("metrics", int(step), {k: float(v) for k, v in metrics.items()}))

    def set_info(self, text: str) -> None:
        if self.is_open:
            self._queue.put(("info", text))

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            self._queue.put(None)
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None

    def __enter__(self) -> MetricsDashboard:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


def _dashboard_loop(
    metrics_queue: mp.Queue,
    stop_event,
    title: str,
    metrics: dict[str, str],
    reference_lines: dict[str, float],
    max_points: int,
    smoothing: int,
    columns: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the training process
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    rows = math.ceil((len(metrics) + 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.6 * rows + 0.6), num=title)
    axes = list(np.atleast_1d(axes).flat)
    lines = {}
    history = {key: (deque(maxlen=max_points), deque(maxlen=max_points)) for key in metrics}
    for ax, (key, name) in zip(axes, metrics.items()):
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        raw, = ax.plot([], [], alpha=0.35, linewidth=1)
        smooth, = ax.plot([], [], color=raw.get_color(), linewidth=2)
        if key in reference_lines:
            ax.axhline(reference_lines[key], color="tab:red", linestyle="--", linewidth=1)
        lines[key] = (ax, raw, smooth)
    info_ax = axes[len(metrics)]
    info_ax.axis("off")
    info_text = info_ax.text(0.0, 1.0, "waiting for data...", va="top", family="monospace", fontsize=9)
    for ax in axes[len(metrics) + 1 :]:
        ax.axis("off")
    fig.supxlabel("environment steps", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    button = Button(fig.add_axes((0.80, 0.008, 0.18, 0.04)), "Stop after episode")

    def on_stop(_event) -> None:
        stop_event.set()
        button.label.set_text("Stopping...")
        fig.canvas.draw_idle()

    button.on_clicked(on_stop)
    plt.show(block=False)

    while plt.fignum_exists(fig.number):
        updated = False
        while True:
            try:
                item = metrics_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                plt.close(fig)
                return
            if item[0] == "info":
                info_text.set_text(item[1])
                updated = True
                continue
            _, step, values = item
            for key, value in values.items():
                steps, data = history[key]
                steps.append(step)
                data.append(value)
                updated = True

        if updated:
            for key, (ax, raw, smooth) in lines.items():
                steps, data = history[key]
                if not data:
                    continue
                x, y = np.asarray(steps), np.asarray(data)
                raw.set_data(x, y)
                k = min(smoothing, len(y))
                smooth.set_data(x[k - 1 :], np.convolve(y, np.ones(k) / k, mode="valid"))
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()
        plt.pause(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the hunting environment yourself.")
    parser.add_argument("--stage", type=int, default=0, choices=range(len(STAGES)))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = Renderer(stage=args.stage, max_steps=args.max_steps, replay_probability=0.0)
    for episode in range(args.episodes):
        env.reset()
        truncated = False
        while not truncated and env.window_open:
            _, _, _, truncated, info = env.step(human_action())
        if not env.window_open:
            break
        print(f"episode {episode}: {info['episode']}")
    env.close()


if __name__ == "__main__":
    main()
