import math
import pygame
from dataclasses import dataclass
from pathlib import Path
import yaml
from pygame import Vector2
import random

from entities import Action, Agent, Entity, Player, Wall, fire_beam, turning_logic, move_with_walls


@dataclass
class Observation:
    positions: list[Vector2]

class Environment:
    def __init__(self):
        self.modifiers = {}
        self.bounds = Vector2(800, 600)
        self.step_count = 0
        self.entities: list[Entity] = []
        self.walls: list[Wall] = []
        self.move_speed = 200.0
        self.fire_interval = 0.2
        self.beam_duration = 0.08

        # modifiers
        with Path(__file__).with_name("configs.yaml").open("r") as f:
            self.modifiers = yaml.safe_load(f)
        self.random_spawn = self.modifiers.get("random_spawn", False)

    def state(self) -> tuple:
        return tuple(entity.position for entity in self.entities)

    def reset(self):
        self.step_count = 0
        for entity in self.entities:
            if isinstance(entity, Agent):
                entity.shoot_cooldown = 0.0
                entity.beam_remaining = 0.0
        if self.random_spawn:
            for entity in self.entities:
                entity.position = Vector2(
                    random.uniform(0, self.bounds.x),
                    random.uniform(0, self.bounds.y)
                )


    def step(self, actions: list[Action], dt: float = 1 / 60):
        """Apply one action per living Agent, in entity-list order."""
        agents = [entity for entity in self.entities if isinstance(entity, Agent)]
        if len(actions) != len(agents):
            raise ValueError("Provide exactly one action per living Agent.")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative.")
        for action in actions:
            if action.move_x not in (-1, 0, 1) or action.move_y not in (-1, 0, 1):
                raise ValueError("Movement components must be -1, 0, or 1.")
            if action.turn is None:
                if not math.isfinite(action.aim_x):
                    raise ValueError("aim_x must be finite.")
                if not math.isfinite(action.aim_y):
                    raise ValueError("aim_y must be finite.")
            elif action.turn not in (-2, -1, 0, 1, 2):
                raise ValueError("turn must be -2, -1, 0, 1, or 2.")
            if not isinstance(action.fire, bool):
                raise ValueError("fire must be a boolean.")

        for agent, action in zip(agents, actions):
            agent.shoot_cooldown = max(0.0, agent.shoot_cooldown - dt)
            agent.beam_remaining = max(0.0, agent.beam_remaining - dt)
            movement = Vector2(action.move_x, action.move_y)
            if movement.length_squared() > 1:
                movement.normalize_ip()
            if self.walls:
                bounds_walls = [
                    Wall(Vector2(0, 0), Vector2(self.bounds.x, 0)),
                    Wall(Vector2(self.bounds.x, 0), self.bounds.copy()),
                    Wall(self.bounds.copy(), Vector2(0, self.bounds.y)),
                    Wall(Vector2(0, self.bounds.y), Vector2(0, 0)),
                ]
                agent.position = move_with_walls(
                    agent.position, movement * self.move_speed * dt,
                    agent.radius, self.walls + bounds_walls,
                )
            else:
                agent.position += movement * self.move_speed * dt
            agent.position.x = max(
                agent.radius, min(self.bounds.x - agent.radius, agent.position.x)
            )
            agent.position.y = max(
                agent.radius, min(self.bounds.y - agent.radius, agent.position.y)
            )
            if action.turn is None:
                agent.update_aim(action.aim_x, action.aim_y, dt)
            else:
                agent.update_turn(action.turn, dt)

        for agent, action in zip(agents, actions):
            if agent in self.entities and action.fire and agent.shoot_cooldown <= 0:
                end = agent.position + (
                    Vector2(agent.aim_x, agent.aim_y) * self.bounds.length()
                )
                segment = fire_beam(agent.position, end, agent, self)
                if segment is not None:
                    agent.shot_start, agent.shot_end = segment
                    agent.beam_remaining = self.beam_duration
                agent.shoot_cooldown = self.fire_interval
        self.step_count += 1

    def on_hit(self, shooter: Agent, target: Entity):
        """Override this hook to customize death, hit effects, or rewards."""
        self.entities.remove(target)
