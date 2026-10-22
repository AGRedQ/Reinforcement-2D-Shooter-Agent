import math
import pygame
from pygame import Vector2
import random

from entities import Action, Agent, Entity, Player
from shooting import fire_beam


class Environment:
    def __init__(self):
        self.bounds = Vector2(800, 600)
        self.step_count = 0
        self.entities: list[Entity] = []
        self.move_speed = 200.0
        self.fire_interval = 0.2
        self.beam_duration = 0.08


    def state(self) -> tuple:
        return tuple(entity.position for entity in self.entities)

    def reset(self):
        self.step_count = 0
        for entity in self.entities:
            if isinstance(entity, Agent):
                entity.shoot_cooldown = 0.0
                entity.beam_remaining = 0.0
        if hasattr(self, 'modifiers') and self.modifiers.get('random_spawn', False):
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
            if not math.isfinite(action.aim_degree):
                raise ValueError("aim_degree must be finite.")
            if not isinstance(action.fire, bool):
                raise ValueError("fire must be a boolean.")

        for agent, action in zip(agents, actions):
            agent.shoot_cooldown = max(0.0, agent.shoot_cooldown - dt)
            agent.beam_remaining = max(0.0, agent.beam_remaining - dt)
            movement = Vector2(action.move_x, action.move_y)
            if movement.length_squared() > 1:
                movement.normalize_ip()
            agent.position += movement * self.move_speed * dt
            agent.position.x = max(
                agent.radius, min(self.bounds.x - agent.radius, agent.position.x)
            )
            agent.position.y = max(
                agent.radius, min(self.bounds.y - agent.radius, agent.position.y)
            )
            agent.update_aim(action.aim_degree, dt)

        for agent, action in zip(agents, actions):
            if agent in self.entities and action.fire and agent.shoot_cooldown <= 0:
                end = agent.position + (
                    Vector2(1, 0).rotate(agent.aim_degree) * self.bounds.length()
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


class Renderer(Environment):
    def __init__(self):
        super().__init__()
        pygame.display.init()
        self.screen = pygame.display.set_mode(
            (int(self.bounds.x), int(self.bounds.y))
        )
        pygame.display.set_caption(
            "RL Environment | WASD: move | Mouse: aim | Left-click: fire"
        )
        self.clock = pygame.time.Clock()
        self.running = True

    def render(self) -> bool:
        """Draw the current state; return False when the window is closed."""
        if not self.running:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.screen.fill((24, 26, 32))
        pygame.draw.rect(
            self.screen, (100, 110, 125), self.screen.get_rect(), width=2
        )

        for entity in self.entities:
            if isinstance(entity, Agent) and entity.beam_remaining > 0:
                pygame.draw.line(
                    self.screen,
                    (255, 235, 120),
                    entity.shot_start,
                    entity.shot_end,
                    width=2,
                )

        for entity in self.entities:
            if isinstance(entity, Player):
                color = (80, 180, 255)
            elif isinstance(entity, Agent):
                color = (240, 160, 70)
            else:
                color = (190, 195, 205)

            if isinstance(entity, Agent):
                gun_points = (
                    Vector2(entity.radius * 2.4, 0),
                    Vector2(entity.radius * 0.6, -entity.radius * 0.5),
                    Vector2(entity.radius * 0.6, entity.radius * 0.5),
                )
                pygame.draw.polygon(
                    self.screen,
                    color,
                    [
                        entity.position + point.rotate(entity.aim_degree)
                        for point in gun_points
                    ],
                )

            pygame.draw.circle(
                self.screen, color, entity.position, entity.radius
            )

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self):
        self.running = False
        pygame.display.quit()



def run_interactive_test():
    """Run the human-controlled demo; never called by training/evaluation."""
    env = Renderer()
    env.entities.extend([
        Entity(Vector2(400, 150)),
        Player(Vector2(200, 300)),
        Agent(Vector2(600, 300)),
    ])
    try:
        while env.render():
            actions = [
                entity.player_action()
                if isinstance(entity, Player)
                else Action(0, 0, entity.aim_degree, False)
                for entity in env.entities
                if isinstance(entity, Agent)
            ]
            env.step(actions, dt=env.clock.get_time() / 1000)
    finally:
        env.close()


if __name__ == "__main__":
    run_interactive_test()