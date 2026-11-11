import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import ppo
from training_report import TrainingReport


class TrainingReportTests(unittest.TestCase):
    def test_exact_intervals_partial_flush_resume_and_offline_html(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = {"episodes": 17, "global_step": 100, "updates": 2}
            report = TrainingReport(
                Path(directory), Path("model.pt"), {"reward": "Reward <test>", "loss": "Loss"},
                {"seed": 3}, progress,
            )
            for episode in range(1, 24):
                progress.update(episodes=17 + episode, global_step=100 + episode)
                report.record_episode(progress, {"reward": float(episode)})
                with report.csv_path.open(newline="") as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), episode // 10)
                report.ppo = {"loss": 0.25}
            report.finish(progress, "keyboard interrupt")
            with report.csv_path.open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["episodes"] for row in rows], ["27", "37", "40"])
            self.assertEqual([row["run_episodes"] for row in rows], ["10", "20", "23"])
            self.assertEqual([row["reward"] for row in rows], ["10.0", "20.0", "23.0"])
            self.assertEqual(rows[-1]["loss"], "0.25")
            markup = report.html_path.read_text(encoding="utf-8")
            self.assertIn("keyboard interrupt", markup)
            self.assertIn("Reward &lt;test&gt;", markup)
            self.assertIn("<svg", markup)
            self.assertNotIn("<script", markup)
            self.assertNotIn("https://", markup)
            self.assertIn("metrics.csv", markup)
            other = TrainingReport(
                Path(directory), Path("model.pt"), {"reward": "Reward"}, {}, progress
            )
            self.assertNotEqual(report.directory, other.directory)
            other.finish(progress, "no episodes")
            self.assertIn("No samples available", other.html_path.read_text())

    def test_nonfinite_metrics_are_blank_and_charts_preserve_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = {"episodes": 0, "global_step": 0, "updates": 0}
            report = TrainingReport(Path(directory), Path("test.pt"), {"reward": "Reward"}, {}, progress)
            for episode in range(1, 31):
                progress["episodes"] = episode
                report.record_episode(progress, {"reward": float("nan") if episode == 20 else 1.0})
            report.finish(progress, "done")
            with report.csv_path.open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[1]["reward"], "")
            self.assertEqual(report.html_path.read_text().count("<polyline"), 2)

    def test_training_saves_reports_without_dashboard_and_keeps_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ppo.parse_args([
                "--no-dashboard", "--num-envs", "4", "--max-steps", "2",
                "--total-episodes", "12", "--rollout-steps", "1",
                "--epochs", "1", "--minibatch-size", "4", "--hidden-size", "16",
                "--report-dir", str(root / "reports"),
            ])
            with patch.object(ppo, "CHECKPOINTS_DIR", root / "checkpoints"):
                ppo.train(args, shoot_back=True, map_seeds=(4,))
                checkpoint = ppo.load_checkpoint(ppo.resolve_model_path(args.model))
                self.assertEqual(checkpoint["episodes"], 12)
                ppo.train(args, shoot_back=True, map_seeds=(4,))
            directories = sorted((root / "reports").iterdir())
            self.assertEqual(len(directories), 2)
            samples = []
            for folder in directories:
                with (folder / "metrics.csv").open(newline="") as source:
                    rows = list(csv.DictReader(source))
                samples.append([int(row["episodes"]) for row in rows])
                self.assertIn("death_rate", rows[0])
                self.assertIn("entropy", rows[0])
                self.assertTrue((folder / "report.html").exists())
            self.assertIn([10, 12], samples)
            self.assertIn([16], samples)

    def test_interrupt_flushes_partial_sample_and_closes_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ppo.parse_args([
                "--no-dashboard", "--num-envs", "1", "--max-steps", "2",
                "--rollout-steps", "4", "--hidden-size", "16",
                "--report-dir", directory,
            ])
            original_step = ppo.VecEnv.step
            original_close = ppo.VecEnv.close
            calls = 0

            def interrupt(envs, actions):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt
                return original_step(envs, actions)

            with (
                patch.object(ppo, "CHECKPOINTS_DIR", Path(directory) / "checkpoints"),
                patch.object(ppo.VecEnv, "step", interrupt),
                patch.object(ppo.VecEnv, "close", autospec=True, side_effect=original_close) as close,
            ):
                ppo.train(args)
                close.assert_called_once()
            report_path = next(Path(directory).glob("*/report.html"))
            self.assertIn("keyboard interrupt", report_path.read_text())
            with report_path.with_name("metrics.csv").open(newline="") as source:
                self.assertEqual([row["episodes"] for row in csv.DictReader(source)], ["1"])

if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    unittest.main()
