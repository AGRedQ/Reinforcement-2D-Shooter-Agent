import math
import pygame
from pygame import Vector2
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """aim_degree is a target heading; the agent turns toward it over time."""

    move_x: int
    move_y: int
    aim_degree: float
    fire: bool



class Entity:
    def __init__(self, position: Vector2):
        self.position = position
        self.radius = 10

class Threat(Entity):
    def __init__(self, position: Vector2):
        super().__init__(position)

class Agent(Entity):
    def __init__(self, position: Vector2):
        super().__init__(position)
        self.shoot_cooldown = 0.0
        self.aim_degree = 0.0
        self.aim_lerp_speed = 10.0
        self.shot_start = position.copy()
        self.shot_end = position.copy()
        self.beam_remaining = 0.0

    def update_aim(self, target_degree: float, dt: float):
        """Lerp along the shortest arc, with a time-based exponential weight."""
        difference = (target_degree - self.aim_degree + 180) % 360 - 180
        weight = -math.expm1(-self.aim_lerp_speed * dt)
        self.aim_degree = (self.aim_degree + difference * weight) % 360


class Player(Agent):
    def __init__(self, position: Vector2):
        super().__init__(position)

    def player_action(self) -> Action:
        """Read testing controls without changing simulation state."""
        keys = pygame.key.get_pressed()
        offset = Vector2(pygame.mouse.get_pos()) - self.position
        aim_degree = self.aim_degree
        if offset.length_squared() > 0:
            aim_degree = math.degrees(math.atan2(offset.y, offset.x)) % 360
        return Action(
            move_x=int(keys[pygame.K_d]) - int(keys[pygame.K_a]),
            move_y=int(keys[pygame.K_s]) - int(keys[pygame.K_w]),
            aim_degree=aim_degree,
            fire=bool(pygame.mouse.get_pressed()[0]),
        )