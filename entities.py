import math
import pygame
from pygame import Vector2
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """Aim is a target direction vector; (0, 0) keeps the current heading."""

    move_x: float
    move_y: float
    aim_x: float
    aim_y: float
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
        self.aim_lerp_speed = 10.0
        self.aim_x = 1.0
        self.aim_y = 0.0
        self.shot_start = position.copy()
        self.shot_end = position.copy()
        self.beam_remaining = 0.0

    def update_aim(self, aim_x: float, aim_y: float, dt: float):
        """Lerp along the shortest arc, with a time-based exponential weight."""
        if not math.isfinite(aim_x) or not math.isfinite(aim_y):
            raise ValueError("Aim components must be finite.")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative.")
        if aim_x == 0 and aim_y == 0:
            return
        target_degree = math.degrees(math.atan2(aim_y, aim_x)) % 360
        current_degree = math.degrees(math.atan2(self.aim_y, self.aim_x)) % 360
        difference = (target_degree - current_degree + 180) % 360 - 180
        weight = -math.expm1(-self.aim_lerp_speed * dt)
        new_degree = (current_degree + difference * weight) % 360
        self.aim_x = math.cos(math.radians(new_degree))
        self.aim_y = math.sin(math.radians(new_degree))


class Player(Agent):
    def __init__(self, position: Vector2):
        super().__init__(position)

    def player_action(self) -> Action:
        """Read testing controls without changing simulation state."""
        keys = pygame.key.get_pressed()
        offset = Vector2(pygame.mouse.get_pos()) - self.position
        aim = Vector2(self.aim_x, self.aim_y)
        if offset.length_squared() > 0:
            aim = offset.normalize()
        return Action(
            move_x=int(keys[pygame.K_d]) - int(keys[pygame.K_a]),
            move_y=int(keys[pygame.K_s]) - int(keys[pygame.K_w]),
            aim_x=aim.x,
            aim_y=aim.y,
            fire=bool(pygame.mouse.get_pressed()[0]),
        )