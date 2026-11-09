import argparse

import pygame

from entities import Action, Agent, Entity, Player, Wall
from map_generation import generate_map
from pygame import Vector2
from renderer import Renderer, draw_navigation


def run_interactive_test(
    *, seed: int = 0, max_walls: int = 16, fixed_layout: bool = False
) -> None:
    env = Renderer()
    try:
        layout = None
        visited: set[int] = set()
        if fixed_layout:
            env.walls.extend([
                Wall(Vector2(300, 200), Vector2(500, 200)),
                Wall(Vector2(500, 200), Vector2(500, 400)),
            ])
            env.entities.extend([
                Entity(Vector2(400, 150)),
                Player(Vector2(200, 300)),
                Agent(Vector2(600, 300)),
            ])
        else:
            layout = generate_map(
                env.bounds, agent_radius=Agent(Vector2()).radius,
                seed=seed, max_walls=max_walls,
            )
            env.walls.extend(layout.walls)
            idle_spawn = next(
                index for index in range(len(layout.nodes))
                if index not in (layout.agent_spawn, layout.threat_spawn)
            )
            env.entities.extend([
                Entity(layout.nodes[layout.threat_spawn].copy()),
                Player(layout.nodes[layout.agent_spawn].copy()),
                Agent(layout.nodes[idle_spawn].copy()),
            ])
            pygame.display.set_caption(
                f"Navigation playground | Seed: {seed} | Walls: {len(layout.walls)}/{max_walls}"
                " | WASD: move | Mouse: aim | Left-click: fire"
            )
        player = next(entity for entity in env.entities if isinstance(entity, Player))
        env.enable_field_of_view(player, half_angle_degrees=45)

        def overlay(screen: pygame.Surface) -> None:
            if layout is not None:
                draw_navigation(screen, layout, visited, env.vision_font)

        while env.running:
            if layout is not None:
                visited.update(
                    index for index, node in enumerate(layout.nodes)
                    if player.position.distance_squared_to(node) <= player.radius ** 2
                )
            if not env.render(draw_overlay=overlay):
                break
            actions = [
                entity.player_action()
                if isinstance(entity, Player)
                else Action(0, 0, entity.aim_x, entity.aim_y, False)
                for entity in env.entities
                if isinstance(entity, Agent)
            ]
            env.step(actions, dt=env.clock.get_time() / 1000)
    finally:
        env.close()


def non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Must be a non-negative integer.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview connected procedural walls and navigation.")
    parser.add_argument("--seed", type=non_negative_int, default=0)
    parser.add_argument("--max-walls", type=non_negative_int, default=16)
    parser.add_argument("--fixed-layout", action="store_true", help="Use the original two-wall demo.")
    args = parser.parse_args()
    if args.fixed_layout and args.max_walls < 2:
        parser.error("--fixed-layout needs a wall limit of at least 2.")
    run_interactive_test(seed=args.seed, max_walls=args.max_walls, fixed_layout=args.fixed_layout)


if __name__ == "__main__":
    main()
