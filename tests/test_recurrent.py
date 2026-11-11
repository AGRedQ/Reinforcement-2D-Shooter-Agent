"""Memory safety, temporal gradients, PPO and end-to-end experiment checks."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from environments import Curriculum
from models.mlp import ActorCritic, batch_observations
from models.ppo import save_checkpoint
from models.recurrent import ObservationHistory, RecurrentActorCritic, batch_histories
from training_bilstm_random_map import fingerprint, main, paired_comparison, parse_args, restore_policy


def observation(value=0.0, bullets=0):
    return {"state": np.full(64, value, dtype=np.float32),
            "bullets": np.full((bullets, 4), value, dtype=np.float32)}


class RecurrentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_history_rolls_resets_and_bootstraps_terminal_without_leakage(self):
        h = ObservationHistory([observation(1), observation(10)], 3)
        captured = h.snapshot()
        h.advance([observation(2), observation(20)], [False, False])
        h.advance([observation(3), observation(30)], [False, False])
        terminal = h.terminal_window(0, observation(4))
        self.assertEqual([o["state"][0] for o in terminal], [2, 3, 4])
        self.assertEqual(captured[0][0]["state"][0], 1)
        h.advance([observation(5), observation(40)], [True, False])
        self.assertEqual([o["state"][0] for o in h.snapshot()[0]], [5])
        self.assertEqual([o["state"][0] for o in h.snapshot()[1]], [20, 30, 40])

    def test_warm_start_preserves_both_policy_and_value_for_any_history(self):
        base = ActorCritic(hidden_size=16, bullet_sensing=True, angular_regions=True)
        recurrent = RecurrentActorCritic(hidden_size=16, lstm_hidden_size=8, context_length=4)
        recurrent.warm_start(base.state_dict())
        windows = [(observation(.2, 2),), (observation(-.5, 1), observation(.3, 3))]
        with torch.no_grad():
            expected = base.act(batch_observations([window[-1] for window in windows]), deterministic=True)
            actual = recurrent.act(batch_histories(windows), deterministic=True)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)

    def test_padding_and_other_episodes_do_not_affect_temporal_output(self):
        policy = RecurrentActorCritic(hidden_size=16, lstm_hidden_size=8, context_length=4)
        torch.nn.init.normal_(policy.memory_projection.weight, std=.1)
        window = (observation(.1, 1), observation(.2, 3))
        alone = batch_histories([window])
        together = batch_histories([window, (observation(.5, 8),) * 4])
        together["state"][0, 2:] = 999
        together["bullets"][0, 2:] = 999
        together["bullets"][0, 0, 1:] = 999  # masked bullet padding
        with torch.no_grad():
            torch.testing.assert_close(policy.value(alone)[0], policy.value(together)[0])

    def test_both_directions_and_past_observations_receive_gradients(self):
        policy = RecurrentActorCritic(hidden_size=16, lstm_hidden_size=8, context_length=4)
        torch.nn.init.normal_(policy.memory_projection.weight, std=.1)
        batch = batch_histories([(observation(.1, 2), observation(.2, 1), observation(.3, 3))])
        batch["state"].requires_grad_()
        batch["bullets"].requires_grad_()
        policy.value(batch).sum().backward()
        self.assertGreater(batch["state"].grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(batch["bullets"].grad[0, 0].abs().sum().item(), 0)
        for name in ("weight_ih_l0", "weight_ih_l0_reverse"):
            self.assertGreater(getattr(policy.memory, name).grad.abs().sum().item(), 0)
        self.assertGreater(policy.bullet_encoder.embedding[0].weight.grad.abs().sum().item(), 0)
        # Adding future data to a separate window cannot mutate a past decision.
        with torch.no_grad():
            before = policy.value(batch).clone()
            policy.value(batch_histories([(observation(.9, 1),) * 4]))
            torch.testing.assert_close(before, policy.value(batch))

    def test_rejects_overlong_histories(self):
        policy = RecurrentActorCritic(hidden_size=16, lstm_hidden_size=8, context_length=2)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            policy.value(batch_histories([(observation(),) * 3]))

    def test_paired_comparison_uses_same_trials(self):
        args = parse_args(["--steps", "512"])
        self.assertEqual(args.recurrent_type, "bilstm")
        metrics = dict(reward=1., kills=1., combat_success=1., survival=1., kill_episode=1., damage_taken=0., length=100.)
        rows = [dict(variant=variant, train_seed=seed, stage=4, map_seed=10, eval_seed=2, **metrics)
                for variant, seed in (("bilstm", 0), ("frozen", -1))]
        self.assertEqual(paired_comparison(rows, "bilstm", "frozen", 4, 100)["reward"]["mean_delta"], 0.)
        rows[-1]["eval_seed"] = 3
        with self.assertRaisesRegex(ValueError, "matching"):
            paired_comparison(rows, "bilstm", "frozen", 4, 100)

    def test_real_training_comparison_safe_save_resume_and_evaluate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            base = ActorCritic(hidden_size=16, bullet_sensing=True, angular_regions=True)
            save_checkpoint(source, base, torch.optim.Adam(base.parameters()), Curriculum(),
                            {"global_step": 100, "episodes": 10, "updates": 2}, 16)
            original = fingerprint(source)
            argv = ["--baseline", str(source), "--run-dir", str(root / "run"),
                    "--steps", "32", "--num-envs", "2", "--rollout-steps", "8",
                    "--minibatch-size", "8", "--epochs", "2", "--context-length", "4",
                    "--lstm-hidden-size", "8", "--max-steps", "8", "--eval-maps", "[10]",
                    "--eval-seeds", "100", "--eval-stages", "0", "--save-every", "1"]
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(argv)
            self.assertEqual(fingerprint(source), original)
            self.assertEqual(result["summary_by_stage"]["0"]["bilstm"]["episodes"], 1)
            policy, saved = restore_policy(root / "run/bilstm_seed_0.pt")
            self.assertEqual(saved["global_step"], 32)
            self.assertGreater(policy.memory_projection.weight.detach().norm().item(), 0)
            self.assertTrue((root / "run/comparison.html").exists())
            self.assertTrue((root / "run/evaluation.csv").exists())
            argv[argv.index("--steps") + 1] = "64"
            with contextlib.redirect_stdout(io.StringIO()):
                main([*argv, "--resume"])
                main(["--run-dir", str(root / "run"), "--evaluate-only"])
            _, saved = restore_policy(root / "run/bilstm_seed_0.pt")
            self.assertEqual(saved["global_step"], 64)
            self.assertEqual(fingerprint(source), original)
            manifest = json.loads((root / "run/experiment.json").read_text())
            self.assertEqual(manifest["source_sha256"], original)

    def test_scratch_training_never_warm_starts_and_hits_exact_partial_rollout_budget(self):
        from training_bilstm_scratch import main as scratch_main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["--run-dir", str(root), "--steps", "34", "--num-envs", "2",
                    "--rollout-steps", "8", "--minibatch-size", "8", "--epochs", "1",
                    "--context-length", "4", "--lstm-hidden-size", "8", "--hidden-size", "16",
                    "--max-steps", "8", "--save-every", "1", "--skip-evaluation",
                    "--compare-to", str(root / "DOES_NOT_EXIST.pt")]
            with patch.object(RecurrentActorCritic, "warm_start", side_effect=AssertionError("Pretrained weights forbidden")):
                with contextlib.redirect_stdout(io.StringIO()):
                    checkpoint = scratch_main(argv)
            policy, saved = restore_policy(checkpoint)
            self.assertEqual(saved["global_step"], 34)
            self.assertEqual(saved["updates"], 3)
            self.assertEqual(saved["initialization"], "random")
            self.assertIsNone(saved["source_sha256"])
            self.assertEqual(saved["curriculum"]["stage"], 0)
            self.assertGreater(policy.memory_projection.weight.detach().norm().item(), 0)
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["pretrained_steps"], 0)
            self.assertEqual(status["global_step"], 34)
            argv[argv.index("--steps") + 1] = "50"
            with contextlib.redirect_stdout(io.StringIO()):
                scratch_main([*argv, "--resume"])
            _, saved = restore_policy(checkpoint)
            self.assertEqual(saved["global_step"], 50)
            evaluation_baseline = root / "evaluation_baseline.pt"
            base = ActorCritic(hidden_size=16, bullet_sensing=True, angular_regions=True)
            save_checkpoint(evaluation_baseline, base, torch.optim.Adam(base.parameters()), Curriculum(),
                            {"global_step": 100, "episodes": 10, "updates": 2}, 16)
            argv.remove("--skip-evaluation")
            argv[argv.index("--compare-to") + 1] = str(evaluation_baseline)
            with contextlib.redirect_stdout(io.StringIO()):
                scratch_main([*argv, "--resume"])
            results = json.loads((root / "comparison.json").read_text())
            self.assertEqual(set(results["summary_by_stage"]["4"]),
                             {"frozen", "bilstm", "bilstm_current_only"})
            self.assertEqual(results["manifest"]["pretrained_steps"], 0)


if __name__ == "__main__":
    unittest.main()
