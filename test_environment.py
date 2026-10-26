import math
import os
import runpy
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pygame
from pygame import Vector2

from entities import Action, Agent, Entity, Player, Wall, fire_beam
from environment import Environment
from renderer import Renderer


def heading(agent: Agent) -> float:
    return math.degrees(math.atan2(agent.aim_y, agent.aim_x)) % 360


class WallCollisionTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.agent = Agent(Vector2(100, 100))
        self.env.entities.append(self.agent)

    def test_large_step_stops_at_radius_and_can_move_away(self):
        self.env.walls.append(Wall(Vector2(200, 0), Vector2(200, 600)))
        self.env.step([Action(1, 0, 1, 0, False)], dt=2)
        self.assertAlmostEqual(self.agent.position.x, 190)
        self.env.step([Action(-1, 0, 1, 0, False)], dt=0.1)
        self.assertAlmostEqual(self.agent.position.x, 170)

    def test_diagonal_movement_slides_along_wall(self):
        self.env.walls.append(Wall(Vector2(200, 0), Vector2(200, 600)))
        self.env.step([Action(1, 1, 1, 0, False)], dt=1)
        self.assertAlmostEqual(self.agent.position.x, 190)
        self.assertAlmostEqual(self.agent.position.y, 100 + 200 / math.sqrt(2))
        self.env.step([Action(0, 1, 1, 0, False)], dt=0.1)
        self.assertAlmostEqual(self.agent.position.x, 190)
        self.assertAlmostEqual(self.agent.position.y, 120 + 200 / math.sqrt(2))

    def test_diagonal_wall_and_corner(self):
        self.env.walls.append(Wall(Vector2(0, 200), Vector2(400, 600)))
        self.env.step([Action(0, 1, 1, 0, False)], dt=1.5)
        wall = self.env.walls[0]
        distance = abs((self.agent.position - wall.start).cross(wall.end - wall.start))
        distance /= (wall.end - wall.start).length()
        self.assertAlmostEqual(distance, self.agent.radius)
        self.env.walls = [
            Wall(Vector2(200, 0), Vector2(200, 600)),
            Wall(Vector2(0, 200), Vector2(800, 200)),
        ]
        self.agent.position = Vector2(100, 100)
        self.env.step([Action(1, 1, 1, 0, False)], dt=2)
        self.assertAlmostEqual(self.agent.position.x, 190)
        self.assertAlmostEqual(self.agent.position.y, 190)

    def test_wall_endpoint_blocks_swept_body_and_point_wall(self):
        for end in (Vector2(200, 200), Vector2(200, 100)):
            with self.subTest(end=end):
                self.agent.position = Vector2(100, 100)
                self.env.walls = [Wall(Vector2(200, 100), end)]
                self.env.step([Action(1, 0, 1, 0, False)], dt=2)
                self.assertAlmostEqual(self.agent.position.x, 190)

    def test_movement_beyond_endpoint_is_unblocked(self):
        self.env.walls = [Wall(Vector2(200, 200), Vector2(200, 400))]
        self.env.step([Action(1, 0, 1, 0, False)], dt=1)
        self.assertEqual(self.agent.position, Vector2(300, 100))

    def test_reversed_wall_endpoints_and_approach_from_either_side(self):
        for start, end in (
            (Vector2(200, 0), Vector2(200, 600)),
            (Vector2(200, 600), Vector2(200, 0)),
        ):
            for x, move, expected in ((100, 1, 190), (300, -1, 210)):
                with self.subTest(start=start, x=x):
                    self.agent.position = Vector2(x, 100)
                    self.env.walls = [Wall(start, end)]
                    self.env.step([Action(move, 0, 1, 0, False)], dt=2)
                    self.assertAlmostEqual(self.agent.position.x, expected)

    def test_slide_into_world_boundary_cannot_escape_bounds(self):
        self.env.walls = [Wall(Vector2(200, 0), Vector2(200, 600))]
        self.env.step([Action(1, 1, 1, 0, False)], dt=10)
        self.assertAlmostEqual(self.agent.position.x, 190)
        self.assertAlmostEqual(self.agent.position.y, 590)

    def test_beam_stops_at_nearest_wall_and_still_pierces_front_targets(self):
        near = Entity(Vector2(150, 100))
        front = Entity(Vector2(170, 100))
        behind = Entity(Vector2(205, 100))
        self.env.entities.extend([near, front, behind])
        self.env.walls = [
            Wall(Vector2(300, 0), Vector2(300, 600)),
            Wall(Vector2(200, 0), Vector2(200, 600)),
        ]
        self.env.step([Action(0, 0, 1, 0, True)], dt=0)
        self.assertEqual(self.agent.shot_end, Vector2(200, 100))
        self.assertEqual(self.env.entities, [self.agent, behind])

    def test_beam_parallel_collinear_endpoint_and_reverse(self):
        cases = [
            (Vector2(100, 110), Vector2(400, 110), (Vector2(200, 100), Vector2(300, 100)), Vector2(400, 110)),
            (Vector2(100, 100), Vector2(400, 100), (Vector2(200, 100), Vector2(300, 100)), Vector2(200, 100)),
            (Vector2(400, 100), Vector2(100, 100), (Vector2(200, 100), Vector2(300, 100)), Vector2(300, 100)),
            (Vector2(100, 100), Vector2(400, 100), (Vector2(200, 100), Vector2(200, 200)), Vector2(200, 100)),
            (Vector2(100, 100), Vector2(400, 100), (Vector2(100, 50), Vector2(100, 150)), Vector2(100, 100)),
            (Vector2(100, 100), Vector2(400, 100), (Vector2(200, 100), Vector2(200, 100)), Vector2(200, 100)),
        ]
        for start, end, endpoints, expected in cases:
            with self.subTest(start=start, end=end, endpoints=endpoints):
                self.env.walls = [Wall(*endpoints)]
                result = fire_beam(start, end, self.agent, self.env)
                self.assertEqual(result, (start, expected))


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.agent = Agent(Vector2(100, 100))
        self.env.entities.append(self.agent)

    def test_environment_does_not_create_human_players(self):
        self.assertEqual(Environment().entities, [])
        with patch.object(Player, "player_action") as read_input:
            self.env.step([Action(0, 0, 1, 0, False)])
        read_input.assert_not_called()

    def test_movement_is_normalized_and_body_stays_in_bounds(self):
        self.env.step([Action(1, 1, 0, 1, False)], dt=0.1)
        self.assertAlmostEqual(
            self.agent.position.distance_to(Vector2(100, 100)), 20
        )
        self.assertAlmostEqual(heading(self.agent), 90 * -math.expm1(-1))
        self.env.step([Action(1, -1, 1, 0, False)], dt=10)
        self.assertEqual(self.agent.position, Vector2(790, 10))
        self.assertEqual(self.env.step_count, 2)

    def test_piercing_hit_calls_hook_and_removes_every_target(self):
        far = Entity(Vector2(400, 100))
        near = Entity(Vector2(200, 100))
        self.env.entities.extend([far, near])
        with patch.object(self.env, "on_hit", wraps=self.env.on_hit) as on_hit:
            self.env.step([Action(0, 0, 1, 0, True)], dt=0)
        self.assertEqual(on_hit.call_count, 2)
        on_hit.assert_any_call(self.agent, near)
        on_hit.assert_any_call(self.agent, far)
        self.assertNotIn(near, self.env.entities)
        self.assertNotIn(far, self.env.entities)
        self.assertEqual(self.agent.shot_end, Vector2(800, 100))
        self.assertGreater(self.agent.beam_remaining, 0)

    def test_custom_hit_hook_can_keep_target_alive(self):
        target = Entity(Vector2(200, 100))
        self.env.entities.append(target)
        with patch.object(self.env, "on_hit") as on_hit:
            self.env.step([Action(0, 0, 1, 0, True)], dt=0)
        on_hit.assert_called_once_with(self.agent, target)
        self.assertIn(target, self.env.entities)

    def test_ray_ignores_targets_behind_off_axis_or_beyond_wall(self):
        targets = [
            Entity(Vector2(50, 100)),
            Entity(Vector2(200, 111)),
            Entity(Vector2(900, 100)),
        ]
        self.env.entities.extend(targets)
        self.env.step([Action(0, 0, 1, 0, True)], dt=0)
        self.assertEqual(self.agent.shot_end, Vector2(800, 100))
        self.assertTrue(all(target in self.env.entities for target in targets))

    def test_ray_can_hit_edge_of_circle_and_downward_target(self):
        tangent = Entity(Vector2(200, 110))
        self.env.entities.append(tangent)
        self.env.step([Action(0, 0, 1, 0, True)], dt=0)
        self.assertNotIn(tangent, self.env.entities)
        self.assertEqual(self.agent.shot_end, Vector2(800, 100))
        target = Entity(Vector2(100, 200))
        self.env.entities.append(target)
        self.agent.aim_x, self.agent.aim_y = 0, 1
        self.env.step([Action(0, 0, 0, 1, True)], dt=self.env.fire_interval)
        self.assertNotIn(target, self.env.entities)
        self.assertAlmostEqual(self.agent.shot_end.y, 600)

    def test_held_fire_obeys_cooldown_and_beam_expires(self):
        target = Entity(Vector2(200, 100))
        self.env.entities.append(target)
        with patch.object(self.env, "on_hit") as on_hit:
            self.env.step([Action(0, 0, 1, 0, True)], dt=0)
            self.env.step([Action(0, 0, 1, 0, True)], dt=0.1)
            self.assertEqual(on_hit.call_count, 1)
            self.assertEqual(self.agent.beam_remaining, 0)
            self.env.step([Action(0, 0, 1, 0, True)], dt=0.1)
            self.assertEqual(on_hit.call_count, 2)


    def test_killed_agent_does_not_fire_later_in_same_step(self):
        target = Agent(Vector2(200, 100))
        self.env.entities.append(target)
        self.env.step([
            Action(0, 0, 1, 0, True),
            Action(0, 0, -1, 0, True),
        ], dt=0)
        self.assertEqual(self.env.entities, [self.agent])

    def test_invalid_inputs_fail_before_changing_state(self):
        cases = [
            ([], 0),
            ([Action(2, 0, 1, 0, False)], 0),
            ([Action(0, 0, float("nan"), 0, False)], 0),
            ([Action(0, 0, 1, float("inf"), False)], 0),
            ([Action(0, 0, 1, 0, False)], -1),
            ([Action(0, 0, 1, 0, False)], float("inf")),
        ]
        for actions, dt in cases:
            with self.subTest(actions=actions, dt=dt):
                with self.assertRaises(ValueError):
                    self.env.step(actions, dt)
                self.assertEqual(self.agent.position, Vector2(100, 100))
                self.assertEqual(self.env.step_count, 0)

    def test_reset_clears_shooting_state(self):
        self.env.step([Action(0, 0, 1, 0, True)])
        self.env.reset()
        self.assertEqual(self.agent.shoot_cooldown, 0)
        self.assertEqual(self.agent.beam_remaining, 0)
        self.assertEqual(self.env.step_count, 0)


class AimTests(unittest.TestCase):
    def test_agent_and_player_turn_gradually_toward_target(self):
        for entity_type in (Agent, Player):
            with self.subTest(entity_type=entity_type):
                env = Environment()
                agent = entity_type(Vector2(100, 100))
                env.entities.append(agent)
                direction = Vector2(1, 0).rotate(250)
                env.step([Action(0, 0, direction.x, direction.y, False)])
                turn = (heading(agent) + 180) % 360 - 180
                self.assertGreater(turn, -110)
                self.assertLess(turn, 0)
                self.assertAlmostEqual(turn, -110 * -math.expm1(-10 / 60))

    def test_wraparound_uses_shortest_arc_in_both_directions(self):
        for start, target in ((359, 1), (1, 359)):
            with self.subTest(start=start, target=target):
                agent = Agent(Vector2())
                agent.aim_x, agent.aim_y = Vector2(1, 0).rotate(start)
                direction = Vector2(1, 0).rotate(target)
                agent.update_aim(direction.x, direction.y, 0.1)
                turn = (heading(agent) - start + 180) % 360 - 180
                expected = (target - start + 180) % 360 - 180
                self.assertAlmostEqual(turn, expected * -math.expm1(-1))

    def test_frame_rate_does_not_change_turn_over_equal_time(self):
        results = []
        for fps in (30, 60, 120):
            agent = Agent(Vector2())
            direction = Vector2(1, 0).rotate(250)
            for _ in range(fps):
                agent.update_aim(direction.x, direction.y, 1 / fps)
            results.append(heading(agent))
        self.assertAlmostEqual(results[0], results[1])
        self.assertAlmostEqual(results[1], results[2])

    def test_zero_time_and_large_time_do_not_overshoot(self):
        agent = Agent(Vector2())
        agent.update_aim(0, 1, 0)
        self.assertEqual(heading(agent), 0)
        for _ in range(3):
            previous = heading(agent)
            agent.update_aim(0, 1, 0.1)
            self.assertGreater(heading(agent), previous)
            self.assertLess(heading(agent), 90)
        self.assertGreaterEqual(heading(agent), 90 * 0.95)
        agent.update_aim(0, 1, 100)
        self.assertAlmostEqual(heading(agent), 90)

    def test_target_magnitude_does_not_change_turn_or_unit_heading(self):
        for scale in (0.001, 1, 1000):
            with self.subTest(scale=scale):
                agent = Agent(Vector2())
                agent.update_aim(scale, scale, 0.1)
                self.assertAlmostEqual(heading(agent), 45 * -math.expm1(-1))
                self.assertAlmostEqual(math.hypot(agent.aim_x, agent.aim_y), 1)

    def test_zero_vector_keeps_heading_and_can_fire(self):
        env = Environment()
        agent = Agent(Vector2(100, 100))
        agent.aim_x, agent.aim_y = 0, 1
        target = Entity(Vector2(100, 200))
        env.entities.extend([agent, target])
        env.step([Action(0, 0, 0, 0, True)], dt=0.1)
        self.assertEqual((agent.aim_x, agent.aim_y), (0, 1))
        self.assertNotIn(target, env.entities)

    def test_invalid_second_action_does_not_update_first_agent(self):
        env = Environment()
        first = Agent(Vector2(100, 100))
        second = Agent(Vector2(200, 100))
        env.entities.extend([first, second])
        for component in (float("nan"), float("inf"), -float("inf")):
            for aim_x, aim_y in ((component, 1), (1, component)):
                with self.subTest(aim_x=aim_x, aim_y=aim_y):
                    with self.assertRaises(ValueError):
                        env.step([
                            Action(1, 0, 0, 1, True),
                            Action(0, 0, aim_x, aim_y, False),
                        ])
                    self.assertEqual(first.position, Vector2(100, 100))
                    self.assertEqual((first.aim_x, first.aim_y), (1, 0))
                    self.assertEqual(first.shoot_cooldown, 0)
                    self.assertEqual(env.step_count, 0)

    def test_shot_follows_current_heading_not_requested_heading(self):
        env = Environment()
        agent = Agent(Vector2(100, 100))
        expected_heading = 90 * -math.expm1(-0.1)
        current_target = Entity(
            agent.position + Vector2(100, 0).rotate(expected_heading)
        )
        requested_target = Entity(Vector2(100, 200))
        env.entities.extend([agent, current_target, requested_target])
        env.step([Action(0, 0, 0, 1, True)], dt=0.01)
        self.assertNotIn(current_target, env.entities)
        self.assertIn(requested_target, env.entities)
        direction = agent.shot_end - agent.shot_start
        self.assertAlmostEqual(
            math.degrees(math.atan2(direction.y, direction.x)), heading(agent)
        )


class BeamTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.shooter = Agent(Vector2(100, 100))
        self.env.entities.append(self.shooter)

    def test_clips_both_endpoints_without_changing_direction(self):
        start = Vector2(-100, -100)
        end = Vector2(1000, 450)
        target = Entity(Vector2(450, 175))
        self.env.entities.append(target)
        segment = fire_beam(start, end, self.shooter, self.env)
        assert segment is not None
        clipped_start, clipped_end = segment
        self.assertAlmostEqual(clipped_start.x, 100)
        self.assertAlmostEqual(clipped_start.y, 0)
        self.assertAlmostEqual(clipped_end.x, 800)
        self.assertAlmostEqual(clipped_end.y, 350)
        self.assertNotIn(target, self.env.entities)
        self.assertEqual(start, Vector2(-100, -100))
        self.assertEqual(end, Vector2(1000, 450))

    def test_segment_hits_bodies_but_not_entities_past_endpoints(self):
        target = Entity(Vector2(200, 105))
        before = Entity(Vector2(50, 100))
        after = Entity(Vector2(350, 100))
        self.env.entities.extend([target, before, after])
        fire_beam(Vector2(100, 100), Vector2(300, 100), self.shooter, self.env)
        self.assertNotIn(target, self.env.entities)
        self.assertEqual(self.env.entities, [self.shooter, before, after])

    def test_zero_length_segment_only_hits_bodies_containing_that_point(self):
        overlapping = Entity(Vector2(100, 100))
        other = Entity(Vector2(200, 100))
        self.env.entities.extend([overlapping, other])
        segment = fire_beam(
            self.shooter.position, self.shooter.position, self.shooter, self.env
        )
        self.assertEqual(segment, (Vector2(100, 100), Vector2(100, 100)))
        self.assertEqual(self.env.entities, [self.shooter, other])

    def test_segment_outside_world_never_triggers_hits(self):
        target = Entity(Vector2(-50, -50))
        self.env.entities.append(target)
        with patch.object(self.env, "on_hit") as on_hit:
            segment = fire_beam(
                Vector2(-100, -100), Vector2(-10, -10), self.shooter, self.env
            )
        self.assertIsNone(segment)
        on_hit.assert_not_called()

    def test_reverse_segment_keeps_direction_and_triggers_all_hits(self):
        targets = [Entity(Vector2(200, 100)), Entity(Vector2(400, 100))]
        self.env.entities.extend(targets)
        segment = fire_beam(
            Vector2(1000, 100), Vector2(-100, 100), self.shooter, self.env
        )
        self.assertEqual(segment, (Vector2(800, 100), Vector2(0, 100)))
        self.assertEqual(self.env.entities, [self.shooter])

    def test_invalid_coordinates_raise_before_triggering_hits(self):
        with self.assertRaises(ValueError):
            fire_beam(Vector2(float("nan"), 0), Vector2(), self.shooter, self.env)
        self.env.bounds.x = 0
        with self.assertRaises(ValueError):
            fire_beam(Vector2(), Vector2(100, 100), self.shooter, self.env)

class PlayerTests(unittest.TestCase):
    def test_controls_produce_action_without_moving_player(self):
        player = Player(Vector2(100, 100))
        keys = defaultdict(bool, {pygame.K_d: True, pygame.K_w: True})
        with (
            patch.object(pygame.key, "get_pressed", return_value=keys),
            patch.object(pygame.mouse, "get_pos", return_value=(100, 200)),
            patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)),
        ):
            action = player.player_action()
        self.assertEqual(action, Action(1, -1, 0, 1, True))
        self.assertEqual(player.position, Vector2(100, 100))

    def test_mouse_at_player_keeps_previous_aim(self):
        player = Player(Vector2(100, 100))
        player.aim_x, player.aim_y = Vector2(1, 0).rotate(45)
        with (
            patch.object(pygame.key, "get_pressed", return_value=defaultdict(bool)),
            patch.object(pygame.mouse, "get_pos", return_value=(100, 100)),
            patch.object(pygame.mouse, "get_pressed", return_value=(False, False, False)),
        ):
            self.assertEqual(
                player.player_action(),
                Action(0, 0, player.aim_x, player.aim_y, False),
            )


class RendererTests(unittest.TestCase):
    def test_interactive_demo_runs_controls_and_wall_blocks_hitscan(self):
        original_flip = pygame.display.flip
        original_on_hit = Environment.on_hit
        original_step = Environment.step
        keys = defaultdict(bool, {pygame.K_d: True})

        def finish_after_one_frame():
            original_flip()
            pygame.event.post(pygame.event.Event(pygame.QUIT))

        with (
            patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}),
            patch.object(pygame.key, "get_pressed", return_value=keys),
            patch.object(pygame.mouse, "get_pos", return_value=(600, 300)),
            patch.object(pygame.mouse, "get_pressed", return_value=(True, False, False)),
            patch.object(pygame.display, "flip", side_effect=finish_after_one_frame),
            patch.object(
                Environment, "on_hit", autospec=True, side_effect=original_on_hit
            ) as on_hit,
            patch.object(
                Environment, "step", autospec=True, side_effect=original_step
            ) as step,
        ):
            try:
                runpy.run_path(str(Path(__file__).with_name("interactive_test.py")), run_name="__main__")
                on_hit.assert_not_called()
                env = step.call_args.args[0]
                shooter = next(entity for entity in env.entities if isinstance(entity, Player))
                self.assertIsInstance(shooter, Player)
                self.assertGreater(shooter.position.x, 200)
                self.assertEqual(len(env.entities), 3)
                self.assertEqual(shooter.shot_end, Vector2(500, 300))
                self.assertEqual(len(env.walls), 2)
                self.assertFalse(env.running)
                self.assertFalse(pygame.display.get_init())
            finally:
                pygame.display.quit()

    def test_walls_are_drawn_as_segments_without_changing_simulation_state(self):
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            renderer = Renderer()
        try:
            self.assertEqual(renderer.walls, [])
            walls = [
                Wall(Vector2(50, 50), Vector2(150, 50)),
                Wall(Vector2(200, 50), Vector2(200, 150)),
                Wall(Vector2(250, 50), Vector2(350, 150)),
            ]
            endpoints = [(wall.start.copy(), wall.end.copy()) for wall in walls]
            renderer.walls.extend(walls)
            self.assertTrue(renderer.render())
            for point in ((100, 50), (200, 100), (300, 100)):
                self.assertEqual(
                    renderer.screen.get_at(point)[:3], (120, 135, 155)
                )
            self.assertEqual(renderer.screen.get_at((100, 60))[:3], (24, 26, 32))
            self.assertEqual(renderer.entities, [])
            self.assertEqual(renderer.step_count, 0)
            self.assertEqual(
                [(wall.start, wall.end) for wall in renderer.walls], endpoints
            )
        finally:
            renderer.close()

    def test_rendered_beam_ends_at_wall(self):
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            renderer = Renderer()
        try:
            agent = Agent(Vector2(100, 100))
            renderer.entities.append(agent)
            renderer.walls.append(Wall(Vector2(200, 50), Vector2(200, 150)))
            renderer.step([Action(0, 0, 1, 0, True)], dt=0)
            renderer.render()
            self.assertEqual(agent.shot_end, Vector2(200, 100))
            self.assertEqual(renderer.screen.get_at((190, 100))[:3], (255, 235, 120))
            self.assertEqual(renderer.screen.get_at((210, 100))[:3], (24, 26, 32))
        finally:
            renderer.close()

    def test_player_agent_guns_and_beam_are_drawn_without_movement(self):
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            renderer = Renderer()
        try:
            player = Player(Vector2(100, 100))
            player.aim_x, player.aim_y = 0, 1
            agent = Agent(Vector2(300, 100))
            renderer.entities.extend([player, agent])
            renderer.step([
                Action(0, 0, 0, 1, True),
                Action(0, 0, 1, 0, False),
            ], dt=0)
            self.assertTrue(renderer.render())
            screen = renderer.screen
            self.assertEqual(screen.get_at((100, 100))[:3], (80, 180, 255))
            self.assertEqual(screen.get_at((300, 100))[:3], (240, 160, 70))
            self.assertEqual(screen.get_at((100, 120))[:3], (80, 180, 255))
            self.assertEqual(screen.get_at((320, 100))[:3], (240, 160, 70))
            self.assertEqual(screen.get_at((100, 200))[:3], (255, 235, 120))
            self.assertEqual(player.position, Vector2(100, 100))
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            self.assertFalse(renderer.render())
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
