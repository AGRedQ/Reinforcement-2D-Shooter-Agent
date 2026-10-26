from environment import Environment
import pygame
from entities import Agent, Player, Vector2, turning_logic

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

        for wall in self.walls:
            pygame.draw.line(
                self.screen, (120, 135, 155), wall.start, wall.end, width=6
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
                        turning_logic(
                            entity.position, entity.aim_x, entity.aim_y, point
                        )
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


