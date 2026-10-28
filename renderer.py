import math

import pygame
from environment import Environment
from entities import (
    Agent,
    Player,
    Vector2,
    has_line_of_sight,
    is_in_field_of_view,
    turning_logic,
    wall_intersection_fraction,
)

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
        self.vision_observer: Agent | None = None
        self.vision_half_angle_degrees = 45.0
        pygame.font.init()
        self.vision_font = pygame.font.Font(None, 24)

    def enable_field_of_view(
        self, observer: Agent, half_angle_degrees: float = 45.0
    ) -> None:
        """Enable unlimited-range vision rendering for an Agent."""
        if not math.isfinite(half_angle_degrees) or not 0 <= half_angle_degrees <= 180:
            raise ValueError("Field-of-view half-angle must be finite and in [0, 180].")
        self.vision_observer = observer
        self.vision_half_angle_degrees = half_angle_degrees

    def _draw_field_of_view(self) -> None:
        observer = self.vision_observer
        if observer is None or observer not in self.entities:
            return

        direction = Vector2(observer.aim_x, observer.aim_y)
        if direction.length_squared() == 0:
            return

        half_angle = self.vision_half_angle_degrees
        radius = self.bounds.length() * 2
        points = [observer.position]
        points.extend(
            observer.position + direction.rotate(angle) * radius
            for angle in (
                -half_angle + 2 * half_angle * index / 48
                for index in range(49)
            )
        )
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(
            overlay,
            (75, 145, 255, 35),
            [(round(point.x), round(point.y)) for point in points],
        )
        self.screen.blit(overlay, (0, 0))

        for entity in self.entities:
            if entity is observer:
                continue
            in_cone = is_in_field_of_view(
                observer.position,
                direction,
                entity.position,
                half_angle,
            )
            unobstructed = has_line_of_sight(
                observer.position, entity.position, self.walls
            )
            visible = in_cone and unobstructed
            line_end = entity.position
            if in_cone and not unobstructed:
                wall_times = [
                    time for wall in self.walls
                    if (
                        time := wall_intersection_fraction(
                            observer.position, entity.position, wall
                        )
                    ) is not None
                ]
                if wall_times:
                    line_end = observer.position + (
                        entity.position - observer.position
                    ) * min(wall_times)
            color = (
                (100, 235, 150)
                if visible
                else (225, 105, 105)
                if in_cone
                else (110, 115, 130)
            )
            pygame.draw.line(self.screen, color, observer.position, line_end, 1)

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
        self._draw_field_of_view()

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
            observer = self.vision_observer
            if observer is not None and observer in self.entities and entity is not observer:
                visible = (
                    is_in_field_of_view(
                        observer.position,
                        Vector2(observer.aim_x, observer.aim_y),
                        entity.position,
                        self.vision_half_angle_degrees,
                    )
                    and has_line_of_sight(
                        observer.position, entity.position, self.walls
                    )
                )
                if not visible:
                    marker = self.vision_font.render("?", True, (255, 100, 100))
                    marker_rect = marker.get_rect(
                        center=(round(entity.position.x), round(entity.position.y - entity.radius - 8))
                    )
                    self.screen.blit(marker, marker_rect)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self):
        self.running = False
        pygame.display.quit()
