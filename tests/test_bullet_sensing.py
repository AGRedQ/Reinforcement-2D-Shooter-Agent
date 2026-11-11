import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

import training_random_map
from entities import Agent, Bullet, OBSERVATION_SIZE, RAY_LENGTH
from environments import HuntEnv, MapLayout
from models import ppo
from models.mlp import ActorCritic, batch_observations


EMPTY_BOXES = np.empty((0, 4))


def bullet(x, y, vx=-450.0, vy=0.0):
    return Bullet(np.array([x, y]), np.array([vx, vy]), 25.0, 600.0)


def observation(count):
    return {
        "state": np.zeros(OBSERVATION_SIZE, dtype=np.float32),
        "bullets": np.linspace(-0.9, 0.9, count * 4, dtype=np.float32).reshape(count, 4),
    }


class BulletSensingTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()
        self.agent.reset(np.zeros(2), 0.0)

    def test_continuous_cone_range_occlusion_and_alive_filter(self):
        between_rays = bullet(100.0, 8.0)
        edge = bullet(100 * math.cos(math.pi / 6), 100 * math.sin(math.pi / 6))
        dead = bullet(50.0, 0.0)
        dead.alive = False
        bullets = [
            between_rays, edge, bullet(-100.0, 0.0), bullet(100.0, 100.0),
            bullet(RAY_LENGTH + 1, 0.0), dead,
        ]
        visible = self.agent.observe_bullets(EMPTY_BOXES, bullets)
        self.assertEqual(visible.shape, (2, 4))
        wall = np.array([[40.0, -100.0, 60.0, 100.0]])
        self.assertEqual(self.agent.observe_bullets(wall, bullets).shape, (0, 4))

    def test_local_frame_relative_velocity_and_no_count_limit(self):
        self.agent.angle = math.pi / 2
        self.agent.velocity = np.array([0.0, 100.0])
        items = [bullet(0.0, 100.0, 0.0, -450.0) for _ in range(300)]
        sensed = self.agent.observe_bullets(EMPTY_BOXES, items)
        self.assertEqual(sensed.shape, (300, 4))
        np.testing.assert_allclose(sensed[0], [100 / 600, 0, -550 / 670, 0], atol=1e-7)
        self.assertTrue(np.all(np.abs(sensed) <= 1.0))

    def test_gym_space_step_reset_and_hostile_only(self):
        env = HuntEnv(shoot_back=True, bullet_sensing=True, max_steps=10)
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs["bullets"].shape, (0, 4))
        env.agent.reset(np.array([100.0, 100.0]), 0.0)
        env.bullets = [bullet(150.0, 100.0)]
        env.threat_bullets = [bullet(200.0, 100.0)]
        obs = env.step(np.array([0, 1, 0]))[0]
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(len(obs["bullets"]), 1)
        self.assertEqual(len(env.reset(seed=7)[0]["bullets"]), 0)

    def test_terminal_observation_survives_autoreset(self):
        env = HuntEnv(
            layout=MapLayout(800.0, 800.0, []), bullet_sensing=True,
            shoot_back=True, max_steps=1,
        )
        env.reset(seed=7)
        env.agent.reset(np.array([100.0, 100.0]), 0.0)
        env.threat_bullets = [bullet(300.0, 100.0)]
        obs, _, _, truncated, _, final_obs = ppo._step_autoreset(env, np.array([0, 1, 0]))
        self.assertTrue(truncated)
        assert final_obs is not None
        self.assertEqual(len(final_obs["bullets"]), 1)
        self.assertEqual(len(obs["bullets"]), 0)

    def test_rendered_and_worker_batches_preserve_variable_sets(self):
        with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            envs = ppo.VecEnv(2, {
                "layout": MapLayout(800.0, 800.0, []), "bullet_sensing": True,
                "shoot_back": True, "max_steps": 2,
            }, seed=7, render=True)
            try:
                local = envs.local_env
                assert local is not None
                local.agent.reset(np.array([100.0, 100.0]), 0.0)
                local.threat_bullets = [bullet(300.0 + i, 100.0) for i in range(70)]
                actions = np.array([[0, 1, 0], [0, 1, 0]])
                observations = envs.step(actions)[0]
                batch = batch_observations(list(observations))
                assert isinstance(batch, dict)
                self.assertEqual(batch["bullet_mask"].sum(dim=-1).tolist(), [70, 0])
                observations, _, _, truncated, _, final = envs.step(actions)
                self.assertTrue(truncated.all())
                self.assertEqual(len(final[0]["bullets"]), 70)
                self.assertEqual(len(observations[0]["bullets"]), 0)
            finally:
                envs.close()


class DeepSetsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.policy = ActorCritic(hidden_size=16, bullet_sensing=True)

    def test_permutation_and_padding_invariance(self):
        item = observation(9)
        original = batch_observations([item])
        shuffled = {"state": item["state"], "bullets": item["bullets"][::-1].copy()}
        reordered = batch_observations([shuffled])
        padded = batch_observations([item, observation(30)])
        assert isinstance(padded, dict)
        expected = self.policy._features(original)[0]
        torch.testing.assert_close(expected, self.policy._features(reordered)[0])
        torch.testing.assert_close(expected, self.policy._features(padded)[0])
        padded["bullets"][0, 9:] = 9999
        torch.testing.assert_close(expected, self.policy._features(padded)[0])

    def test_empty_sets_finite_outputs_and_zero_summary(self):
        obs = batch_observations([observation(0), observation(3), observation(0)])
        features = self.policy._features(obs)
        torch.testing.assert_close(features[[0, 2], -32:], torch.zeros(2, 32))
        actions, log_probs, values = self.policy.act(obs)
        self.assertEqual(actions.shape, (3, 3))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(values).all())
        all_empty = batch_observations([observation(0)])
        self.assertTrue(torch.isfinite(self.policy.value(all_empty)).all())

    def test_ppo_updates_encoder_and_handles_ragged_minibatches(self):
        obs = batch_observations([observation(n) for n in (0, 1, 5, 80)])
        with torch.no_grad():
            actions, log_probs, values = self.policy.act(obs)
        encoder = self.policy.bullet_encoder
        assert encoder is not None
        parameter = next(encoder.parameters())
        before = parameter.detach().clone()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        args = training_random_map.parse_args(["--epochs", "1", "--minibatch-size", "2"])
        stats = ppo.ppo_update(self.policy, optimizer, {
            "obs": obs, "actions": actions, "log_probs": log_probs,
            "advantages": torch.tensor([1.0, -1.0, 0.5, -0.5]),
            "returns": values + 1, "values": values,
        }, args)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        self.assertFalse(torch.equal(before, parameter))
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_roundtrip_legacy_and_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            optimizer = torch.optim.Adam(self.policy.parameters())
            ppo.save_checkpoint(path, self.policy, optimizer, ppo.Curriculum(),
                                {"global_step": 0, "episodes": 0, "updates": 0}, 16)
            checkpoint = ppo.load_checkpoint(path)
            self.assertEqual(checkpoint["architecture"], "deep_sets_v1")
            restored = ActorCritic(hidden_size=16, bullet_sensing=True)
            restored.load_state_dict(checkpoint["model"])
            obs = batch_observations([observation(5)])
            torch.testing.assert_close(restored.value(obs), self.policy.value(obs))
            legacy = ActorCritic(hidden_size=16)
            checkpoint["model"] = legacy.state_dict()
            checkpoint.pop("architecture")
            torch.save(checkpoint, path)
            loaded = ppo.load_checkpoint(path)
            ActorCritic(hidden_size=16).load_state_dict(loaded["model"])
            with patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)):
                args = training_random_map.parse_args(["--bullet-sensing", "--model", "policy"])
                with self.assertRaisesRegex(ValueError, "not a Deep Sets"):
                    training_random_map.train(args)

    def test_multiprocess_training_resume_and_watch(self):
        args = training_random_map.parse_args([
            "--bullet-sensing", "--no-dashboard", "--num-envs", "2",
            "--max-steps", "12", "--rollout-steps", "4", "--minibatch-size", "8",
            "--hidden-size", "16", "--epochs", "1", "--total-episodes", "4",
            "--save-every", "1", "--map-seeds", "[2,7]",
        ])
        self.assertNotEqual(args.model, training_random_map.DEFAULT_MODEL_NAME)
        with tempfile.TemporaryDirectory() as directory, patch.object(ppo, "CHECKPOINTS_DIR", Path(directory)):
            training_random_map.train(args)
            path = ppo.resolve_model_path(args.model)
            checkpoint = ppo.load_checkpoint(path)
            self.assertEqual(checkpoint["architecture"], "deep_sets_v1")
            self.assertGreater(checkpoint["updates"], 0)
            args.bullet_sensing = False
            args.total_episodes = checkpoint["episodes"] + 2
            training_random_map.train(args)
            self.assertGreater(ppo.load_checkpoint(path)["updates"], checkpoint["updates"])
            with patch.dict("os.environ", {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
                args.episodes, args.fps = 1, 0
                training_random_map.watch(args)


if __name__ == "__main__":
    unittest.main()
