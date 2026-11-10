import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from streamlit.testing.v1 import AppTest

import training
from dashboard import MetricsDashboard
from dashboard_app import export_csv, metric_figure, smoothed

APP_PATH = Path(__file__).with_name("dashboard_app.py")
METRICS = {**training.EPISODE_METRICS, **training.COMBAT_METRICS, **training.PPO_METRICS}


def bridge_request(dashboard: MetricsDashboard, path: str = "/snapshot", method: str = "GET"):
    if dashboard._server is None:
        raise RuntimeError("Dashboard bridge has not started")
    request = Request(
        f"http://127.0.0.1:{dashboard._server.server_port}{path}",
        headers={"Authorization": f"Bearer {dashboard._token}"},
        method=method,
    )
    with urlopen(request, timeout=2) as response:
        return json.load(response)


class DashboardBridgeTests(unittest.TestCase):
    def test_bounded_history_partial_updates_nonfinite_and_unknown_metrics(self):
        dashboard = MetricsDashboard({"reward": "Reward", "loss": "Loss"}, max_points=2)
        dashboard.log(1, reward=1.0, loss=2.0)
        dashboard.log(2, reward=math.nan)
        dashboard.log(3, reward=math.inf)
        snapshot = dashboard.snapshot()
        self.assertEqual(snapshot["history"]["reward"], [(2, None), (3, None)])
        self.assertEqual(snapshot["history"]["loss"], [(1, 2.0)])
        json.dumps(snapshot, allow_nan=False)
        with self.assertRaisesRegex(KeyError, "Unknown"):
            dashboard.log(4, typo=5.0)
        for kwargs in [{"metrics": {}}, {"metrics": {"x": "X"}, "max_points": 0}]:
            with self.assertRaises(ValueError):
                MetricsDashboard(**kwargs)

    def test_local_bridge_snapshot_authentication_and_stop(self):
        dashboard = MetricsDashboard({"reward": "Reward"})
        url = dashboard._start_bridge()
        try:
            dashboard.log(10, reward=2.5)
            dashboard.set_info("model     test.pt")
            snapshot = bridge_request(dashboard)
            self.assertEqual(snapshot["history"]["reward"], [[10, 2.5]])
            self.assertEqual(snapshot["info"], "model     test.pt")
            with self.assertRaises(HTTPError) as denied:
                urlopen(f"{url}/snapshot", timeout=2)
            self.assertEqual(denied.exception.code, 403)
            with self.assertRaises(HTTPError) as missing:
                bridge_request(dashboard, "/missing")
            self.assertEqual(missing.exception.code, 404)
            self.assertFalse(dashboard.stop_requested)
            self.assertTrue(bridge_request(dashboard, "/stop", "POST")["stop_requested"])
            self.assertTrue(dashboard.stop_requested)
            self.assertTrue(bridge_request(dashboard, "/stop", "POST")["stop_requested"])
            thread = dashboard._thread
        finally:
            dashboard.close()
        self.assertIsNotNone(thread)
        if thread is not None:
            self.assertFalse(thread.is_alive())
        with self.assertRaises(URLError):
            urlopen(f"{url}/snapshot", timeout=0.5)
        dashboard.close()

    def test_startup_failure_cleans_bridge(self):
        dashboard = MetricsDashboard({"reward": "Reward"})
        with patch("dashboard.subprocess.Popen", side_effect=OSError("cannot launch")):
            with self.assertRaisesRegex(OSError, "cannot launch"):
                dashboard.start()
        self.assertIsNone(dashboard._server)
        self.assertIsNone(dashboard._thread)
        self.assertIsNone(dashboard._output)

    def test_real_streamlit_start_health_and_shutdown(self):
        dashboard = MetricsDashboard({"reward": "Reward"})
        with patch("dashboard.webbrowser.open", return_value=True) as browser:
            dashboard.start()
            try:
                process = dashboard._process
                self.assertTrue(dashboard.is_open)
                self.assertIs(dashboard.start(), dashboard)
                browser.assert_called_once_with(dashboard.url)
                with urlopen(f"{dashboard.url}/_stcore/health", timeout=2) as response:
                    self.assertEqual(response.status, 200)
                dashboard.log(1, reward=3.0)
                self.assertEqual(bridge_request(dashboard)["history"]["reward"], [[1, 3.0]])
            finally:
                dashboard.close()
        if process is None:
            self.fail("Streamlit process was not created")
        self.assertIsNotNone(process.poll())
        self.assertFalse(dashboard.is_open)


class DashboardUITests(unittest.TestCase):
    def setUp(self):
        self.dashboard = MetricsDashboard(METRICS, reference_lines={"mean_reward": 35.0}, title="Combat training")
        url = self.dashboard._start_bridge()
        self.env = patch.dict(os.environ, {
            "SHOOTER_DASHBOARD_URL": url,
            "SHOOTER_DASHBOARD_TOKEN": self.dashboard._token,
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.dashboard.close)

    def app(self):
        return AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    def test_waiting_state_and_live_controls(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Combat training")
        self.assertEqual(app.metric[0].value, "--")
        self.assertEqual(len(app.tabs), 3)
        for step in range(3):
            self.dashboard.log(step, **{key: float(step) for key in METRICS})
        self.dashboard.set_info("model     test.pt\nsteps     2,048\nepisodes  16\nupdates   3\nSPS       900")
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "2,048")
        self.assertEqual(app.metric[4].value, "2.00")
        self.assertEqual(len(app.get("plotly_chart")), len(METRICS))
        app.slider[0].set_value(3).run()
        app.checkbox[0].uncheck().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.slider[0].value, 3)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(self.dashboard.stop_requested)
        self.assertTrue(app.button[0].disabled)
        self.assertIn("Finishing current episodes", app.info[0].value)

    def test_hunting_dashboard_without_combat_metrics(self):
        self.dashboard.metrics = {**training.EPISODE_METRICS, **training.PPO_METRICS}
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 2)

    def test_disconnected_training_is_visible(self):
        self.dashboard.close()
        app = self.app()
        self.assertFalse(app.exception)
        self.assertIn("Training has disconnected", app.warning[0].value)

    def test_smoothing_plot_threshold_and_csv_export(self):
        points = [(1, 1.0), (2, 3.0), (3, None), (4, 5.0)]
        self.assertEqual(smoothed(points, 2), [1.0, 2.0, None, 5.0])
        self.dashboard.log(1, mean_reward=1.0)
        self.dashboard.log(2, mean_reward=math.nan)
        snapshot = self.dashboard.snapshot()
        figure = metric_figure(snapshot, "mean_reward", 2, True)
        chart = figure.to_plotly_json()
        self.assertEqual(len(chart["data"]), 2)
        self.assertEqual(chart["layout"]["shapes"][0]["y0"], 35.0)
        self.assertEqual(len(metric_figure(snapshot, "mean_reward", 2, False).to_plotly_json()["data"]), 1)
        rows = list(csv.reader(io.StringIO(export_csv(snapshot))))
        self.assertEqual(rows[0], ["metric", "environment_step", "value"])
        self.assertEqual(rows[1:], [["mean_reward", "1", "1.0"], ["mean_reward", "2", ""]])


class DashboardTrainingTests(unittest.TestCase):
    def test_graceful_stop_saves_checkpoint_and_closes_dashboard_in_all_modes(self):
        dashboards = []

        class StopOnUpdateDashboard(MetricsDashboard):
            def start(self):
                dashboards.append(self)
                return super().start()

            def log(self, step: int, **metrics: float) -> None:
                super().log(step, **metrics)
                if "policy_loss" in metrics and not self.stop_requested:
                    bridge_request(self, "/stop", "POST")

        for combat, map_seeds in [(False, None), (True, None), (True, (2, 7))]:
            with self.subTest(combat=combat, map_seeds=map_seeds), tempfile.TemporaryDirectory() as directory:
                args = training.parse_args([
                    "--num-envs", "2", "--max-steps", "12",
                    "--rollout-steps", "4", "--minibatch-size", "8",
                    "--hidden-size", "16", "--epochs", "1",
                ])
                with (
                    patch.object(training, "MODELS_DIR", Path(directory)),
                    patch.object(training, "MetricsDashboard", StopOnUpdateDashboard),
                    patch("dashboard.webbrowser.open", return_value=True),
                ):
                    training.train(args, shoot_back=combat, map_seeds=map_seeds)
                    checkpoint = training.load_checkpoint(training.resolve_model_path(args.model))
                self.assertGreater(checkpoint["updates"], 0)
                self.assertGreaterEqual(checkpoint["episodes"], 2)
                self.assertTrue(dashboards[-1].stop_requested)
                self.assertFalse(dashboards[-1].is_open)
                self.assertIsNone(dashboards[-1]._server)
                self.assertIsNone(dashboards[-1]._thread)
                if map_seeds is not None:
                    self.assertIn("maps      (2, 7)", dashboards[-1].snapshot()["info"])

    def test_no_dashboard_does_not_import_streamlit_or_plotly(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, training; assert 'streamlit' not in sys.modules; assert 'plotly' not in sys.modules"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
