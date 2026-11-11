"""Symmetric first-death duels, sharing Agent physics, sensing and rendering."""

from __future__ import annotations

import math

import numpy as np

from entities import Agent
from environments import HuntEnv, StageConfig
from renderer import Renderer, TEXT_COLOR


class SelfPlayEnv(HuntEnv):
    """Two Agent bodies; step accepts paired actions, and returns learner rewards.

    Both actions are chosen before either body moves. All projectiles are advanced
    before damage is applied, allowing mutual kills without first-player priority.
    Timeouts are terminal draws (no value bootstrap), not hunting truncations.
    Curriculum/replay are disabled; spawns are free, separated and randomly oriented.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update(shoot_back=True, stage=0, replay_probability=0.0, exploration_bonus=False)
        super().__init__(*args, **kwargs)
        self.threat = Agent(
            agent_id=1, angular_regions=self.agent.angular_regions, color=(255, 110, 90)
        )
        self._round_over = False

    def _threat_spawn_position(self, cfg: StageConfig, strict: bool = False) -> np.ndarray:
        for _ in range(1000):
            position = self._random_free_position()
            if np.linalg.norm(position - self.agent.position) >= 100:
                return position
        raise RuntimeError("Could not find separated self-play spawns.")

    def _spawn_threat(self, position: np.ndarray | None = None) -> None:
        if position is None:
            position = self._threat_spawn_position(self.stage_config)
        self.threat.reset(position, float(self.np_random.uniform(-math.pi, math.pi)))
        # Randomize role assignment so any spawn-sampling asymmetry is shared.
        if self.np_random.random() < 0.5:
            self.agent.position, self.threat.position = self.threat.position, self.agent.position
            self.agent.angle, self.threat.angle = self.threat.angle, self.agent.angle
        self._spawn_step = 0
        self._threat_detected = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._round_over = False
        result = super().reset(seed=seed, options=options)
        self.threat.sense(self.layout.boxes, [self.agent])
        self.episode_stats.update(win=0.0, loss=0.0, draw=0.0)
        return result

    def opponent_observation(self):
        state = self.threat.observation()
        if not self.bullet_sensing:
            return state
        return {
            "state": state,
            "bullets": self.threat.observe_bullets(self.layout.boxes, self.bullets),
        }

    def step(self, action) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict]:
        if self._round_over:
            raise RuntimeError("Round is over; call reset() before step().")
        actions = np.asarray(action)
        if actions.shape != (2, 3):
            raise ValueError("Self-play requires two (move, aim, fire) actions, shape (2, 3).")
        cfg, stats, boxes = self.reward_config, self.episode_stats, self.layout.boxes
        reward = 0.0
        visible = bool(self.agent.enemy_detected.any())
        previous = self.agent.position.copy()
        for actor, actor_action, projectiles in (
            (self.agent, actions[0], self.bullets),
            (self.threat, actions[1], self.threat_bullets),
        ):
            bullet = actor.apply_action(actor_action, self.dt, boxes)
            if bullet is not None:
                projectiles.append(bullet)
                stats["shots" if actor is self.agent else "threat_shots"] += 1
                if actor is self.agent and not visible:
                    stats["wasted_shots"] += 1
                    reward -= cfg.wasted_shot
        stats["distance"] += float(np.linalg.norm(self.agent.position - previous))
        hits = []
        for projectiles, target in ((self.bullets, self.threat), (self.threat_bullets, self.agent)):
            for bullet in projectiles:
                if bullet.update(self.dt, boxes, [target]) is not None:
                    hits.append((target, bullet.damage))
        for target, damage in hits:
            dealt = target.take_damage(damage)
            if target is self.threat:
                if dealt > 0:
                    stats["hits"] += 1
                    reward += cfg.hit
            else:
                stats["damage_taken"] += dealt
                reward -= cfg.damage_taken * dealt
        self.bullets = [bullet for bullet in self.bullets if bullet.alive]
        self.threat_bullets = [bullet for bullet in self.threat_bullets if bullet.alive]
        if not self.threat.alive:
            stats["kills"] = 1.0
            reward += cfg.kill_bonus
        if not self.agent.alive:
            stats["deaths"] = 1.0
            reward -= cfg.death
        self.steps += 1
        self.agent.sense(boxes, [self.threat])
        self.threat.sense(boxes, [self.agent])
        self._update_detection()
        self._round_over = not self.agent.alive or not self.threat.alive or self.steps >= self.max_steps
        stats["reward"] += reward
        info = {}
        if self._round_over:
            stats["win"] = float(self.agent.alive and not self.threat.alive)
            stats["loss"] = float(self.threat.alive and not self.agent.alive)
            stats["draw"] = float(not stats["win"] and not stats["loss"])
            delays = self._detection_delays + ([] if self._threat_detected else [self.steps])
            stats["detect_steps"] = float(np.mean(delays)) if delays else 0.0
            stats["length"] = float(self.steps)
            info["episode"] = dict(stats)
        return self._observation(), float(reward), self._round_over, False, info


class SelfPlayRenderer(Renderer, SelfPlayEnv):
    """Reuse the existing preview around symmetric duel mechanics."""

    def _draw_hud(self) -> None:
        super()._draw_hud()
        assert self._screen is not None and self._title_font is not None
        self._screen.fill((13, 19, 27), (10, int(self.layout.height) + 8, 350, 28))
        self._screen.blit(
            self._title_font.render("SELF-PLAY: BLUE vs FROZEN RED", True, TEXT_COLOR),
            (18, int(self.layout.height) + 12),
        )
