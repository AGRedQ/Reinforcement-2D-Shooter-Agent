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
from models.runtime import fingerprint, load_policy as restore_policy
from models.evaluation import paired_comparison
from models.curriculum_ppo import FinalStageWindow, training_settings
from training_bilstm_scratch import main, parse_args


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

    def scratch_argv(self, root, steps=34):
        return ["--model", str(root / "model.pt"), "--run-dir", str(root / "run"),
                "--steps", str(steps), "--num-envs", "2", "--rollout-steps", "8",
                "--minibatch-size", "8", "--epochs", "1", "--context-length", "4",
                "--lstm-hidden-size", "8", "--hidden-size", "16", "--max-steps", "8",
                "--save-every", "1", "--no-dashboard", "--device", "cpu", "--skip-evaluation"]

    def test_defaults_match_baseline_and_only_architecture_changes(self):
        import training_random_map
        baseline = training_random_map.parse_args([])
        recurrent = parse_args([])
        control = parse_args(["--architecture", "feedforward"])
        self.assertEqual(training_settings(baseline), training_settings(recurrent))
        self.assertEqual(training_settings(control), training_settings(recurrent))
        self.assertEqual(recurrent.num_envs, 8)
        self.assertEqual(recurrent.rollout_steps, 256)
        self.assertEqual(recurrent.minibatch_size, 512)
        self.assertTrue(recurrent.until_curriculum_complete)

    def test_final_stage_requires_qualifying_full_reward_window(self):
        gate = FinalStageWindow(2, 35)
        self.assertFalse(gate.complete)
        gate.record(dict(stage=4,replay=1,reward=100))
        gate.record(dict(stage=3,replay=0,reward=100))
        self.assertEqual(len(gate.rewards),0)
        gate.record(dict(stage=4,replay=0,reward=40))
        self.assertFalse(gate.complete)
        gate.record(dict(stage=4,replay=0,reward=30))
        self.assertTrue(gate.complete)
        gate.record(dict(stage=4,replay=0,reward=0))
        self.assertFalse(gate.complete)

    def test_scratch_exact_budget_resume_no_pretraining_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            argv=self.scratch_argv(root)
            argv += ["--compare-to",str(root/"DOES_NOT_EXIST.pt")]
            with patch.object(RecurrentActorCritic,"warm_start",side_effect=AssertionError("No pretrained weights")):
                with contextlib.redirect_stdout(io.StringIO()):
                    checkpoint=main(argv)
            policy,saved=restore_policy(checkpoint)
            self.assertEqual(saved["global_step"],34)
            self.assertEqual(saved["updates"],3)
            self.assertEqual(saved["initialization"],"random")
            self.assertIsNone(saved["source_sha256"])
            self.assertFalse(saved["curriculum_complete"])
            self.assertGreater(policy.memory_projection.weight.detach().norm().item(),0)
            argv[argv.index("--steps")+1]="50"
            with contextlib.redirect_stdout(io.StringIO()):
                main([*argv,"--resume"])
            _,saved=restore_policy(checkpoint)
            self.assertEqual(saved["global_step"],50)
            # Explicit changed learning rate must not silently alter a controlled run.
            with self.assertRaisesRegex(ValueError,"configuration differs"):
                with contextlib.redirect_stdout(io.StringIO()):
                    main([*argv,"--resume","--lr","0.0001"])

    def test_real_curriculum_completes_stage_four_instead_of_only_reaching_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            argv=self.scratch_argv(root,256)
            argv[argv.index("--max-steps")+1]="4"
            argv += ["--promotion-window","1","--promotion-reward","-100000"]
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint=main(argv)
            _,saved=restore_policy(checkpoint)
            self.assertTrue(saved["curriculum_complete"])
            self.assertEqual(saved["completed_stages"],[0,1,2,3,4])
            self.assertEqual([e["completed_stage"] for e in saved["stage_history"]],[0,1,2,3])
            self.assertLess(saved["global_step"],256)
            self.assertTrue(saved["final_stage"]["complete"])

    def test_post_training_comparison_and_evaluate_only_preserve_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            base=ActorCritic(hidden_size=16,bullet_sensing=True,angular_regions=True)
            baseline=root/"baseline.pt"
            save_checkpoint(baseline,base,torch.optim.Adam(base.parameters()),Curriculum(),
                            {"global_step":100,"episodes":10,"updates":2},16)
            original=fingerprint(baseline)
            argv=self.scratch_argv(root,16)
            argv.remove("--skip-evaluation")
            argv += ["--compare-to",str(baseline),"--eval-maps","[14]","--eval-seeds","100","--eval-stages","0","4"]
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint=main(argv)
                main([*argv,"--evaluate-only"])
            self.assertEqual(fingerprint(baseline),original)
            result=json.loads((root/"run/comparison.json").read_text())
            self.assertEqual(set(result["summary_by_stage"]["4"]),{"frozen","bilstm","bilstm_current_only"})
            self.assertFalse(result["manifest"]["fairness_audit"]["reference_training_config_recorded"])
            self.assertFalse(result["manifest"]["fairness_audit"]["controlled_architecture_only_claim"])
            self.assertEqual(result["manifest"]["candidate_steps"],16)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is inaccessible in this process")
    def test_cuda_training_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            argv=self.scratch_argv(root,16)
            argv[argv.index("--device")+1]="cuda"
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint=main(argv)
            policy,saved=restore_policy(checkpoint,"cuda")
            self.assertEqual(next(policy.parameters()).device.type,"cuda")
            self.assertTrue(saved["training_device"].startswith("cuda"))
            self.assertTrue(all(torch.isfinite(value).all() for value in saved["model"].values()))
            argv[argv.index("--steps")+1]="32"
            with contextlib.redirect_stdout(io.StringIO()):
                main([*argv,"--resume"])
            _,saved=restore_policy(checkpoint)
            self.assertEqual(saved["global_step"],32)

    def test_automatic_matched_control_uses_identical_settings_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            baseline=root/'historical.pt'
            base=ActorCritic(hidden_size=16,bullet_sensing=True,angular_regions=True)
            save_checkpoint(baseline,base,torch.optim.Adam(base.parameters()),Curriculum(),
                            {'global_step':100,'episodes':10,'updates':2},16)
            original=fingerprint(baseline)
            argv=self.scratch_argv(root,16)
            argv.remove('--skip-evaluation')
            argv += ['--train-control','--compare-to',str(baseline),'--eval-maps','[14]',
                     '--eval-seeds','100','--eval-stages','4']
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint=main(argv)
            _,candidate=restore_policy(checkpoint)
            _,control=restore_policy(root/'model_feedforward_control.pt')
            self.assertEqual(candidate['global_step'],control['global_step'])
            self.assertEqual(candidate['training_config'],control['training_config'])
            matched=json.loads((root/'run/matched_comparison/comparison.json').read_text())
            self.assertTrue(matched['manifest']['fairness_audit']['controlled_architecture_only_claim'])
            self.assertEqual(fingerprint(baseline),original)
            status=json.loads((root/'run/pipeline_status.json').read_text())
            self.assertEqual(status['phase'],'complete')


if __name__ == "__main__":
    unittest.main()
