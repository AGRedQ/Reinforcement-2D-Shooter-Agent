from renderer import Renderer
from environment import Wall, Vector2, Entity, Player, Agent, Action

def run_interactive_test():
    env = Renderer()
    env.walls.extend([
        Wall(Vector2(300, 200), Vector2(500, 200)),
        Wall(Vector2(500, 200), Vector2(500, 400)),
    ])
    env.entities.extend([
        Entity(Vector2(400, 150)),
        Player(Vector2(200, 300)),
        Agent(Vector2(600, 300)),
    ])
    player = next(entity for entity in env.entities if isinstance(entity, Player))
    env.enable_field_of_view(player, half_angle_degrees=45)
    try:
        while env.render():
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

if __name__ == "__main__":
    run_interactive_test()
