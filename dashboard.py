"""Non-blocking metrics bridge and lifecycle for the local Streamlit dashboard.

Training starts the dashboard automatically; ``--no-dashboard`` disables it.
The browser URL is printed and opened automatically. Closing the browser does
not stop training. "Stop after episode" finishes current episodes and saves the
checkpoint; Ctrl+C saves immediately. The server closes when training exits.
Only localhost is used; metric history is bounded in memory and can be exported
from the dashboard as CSV.
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypedDict
from urllib.error import URLError
from urllib.request import urlopen

LOGGER = logging.getLogger(__name__)


class DashboardSnapshot(TypedDict):
    title: str
    metrics: dict[str, str]
    reference_lines: dict[str, float]
    history: dict[str, list[tuple[int, float | None]]]
    info: str
    stop_requested: bool
    smoothing: int
    columns: int


class MetricsDashboard:
    """Keep the training-side API independent of Streamlit and browser clients."""

    def __init__(
        self,
        metrics: dict[str, str],
        reference_lines: dict[str, float] | None = None,
        title: str = "Training dashboard",
        max_points: int = 5000,
        smoothing: int = 10,
        columns: int = 2,
    ):
        if not metrics or min(max_points, smoothing, columns) < 1:
            raise ValueError("Metrics must be nonempty and max_points, smoothing, columns must be positive.")
        self.metrics = dict(metrics)
        self.reference_lines = dict(reference_lines or {})
        self.title = title
        self.max_points = max_points
        self.smoothing = smoothing
        self.columns = columns
        self.url: str | None = None
        self._history: dict[str, deque[tuple[int, float | None]]] = {
            key: deque(maxlen=max_points) for key in metrics
        }
        self._info = "Waiting for the first training update..."
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._token = secrets.token_urlsafe(32)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._output = None
        self._failure_reported = False

    def snapshot(self) -> DashboardSnapshot:
        with self._lock:
            return {
                "title": self.title,
                "metrics": dict(self.metrics),
                "reference_lines": dict(self.reference_lines),
                "history": {key: list(points) for key, points in self._history.items()},
                "info": self._info,
                "stop_requested": self.stop_requested,
                "smoothing": self.smoothing,
                "columns": self.columns,
            }

    def _start_bridge(self) -> str:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._handle()

            def do_POST(self) -> None:
                self._handle()

            def _handle(self) -> None:
                if self.headers.get("Authorization") != f"Bearer {dashboard._token}":
                    self.send_error(403)
                    return
                if self.command == "GET" and self.path == "/snapshot":
                    payload = dashboard.snapshot()
                elif self.command == "POST" and self.path == "/stop":
                    dashboard._stop_event.set()
                    payload = {"stop_requested": True}
                else:
                    self.send_error(404)
                    return
                data = json.dumps(payload, allow_nan=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args) -> None:
                LOGGER.debug(format, *args)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> MetricsDashboard:
        if self.is_open:
            return self
        if self._server is not None:
            raise RuntimeError("The dashboard process exited. Close it before starting another dashboard.")
        self._stop_event.clear()
        self._failure_reported = False
        bridge_url = self._start_bridge()
        try:
            with socket.socket() as port_socket:
                port_socket.bind(("127.0.0.1", 0))
                port = port_socket.getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
            env = {
                **os.environ,
                "SHOOTER_DASHBOARD_URL": bridge_url,
                "SHOOTER_DASHBOARD_TOKEN": self._token,
            }
            self._output = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
            self._process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run",
                    str(Path(__file__).with_name("dashboard_app.py")),
                    "--server.address", "127.0.0.1",
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                    "--theme.base", "dark",
                    "--theme.primaryColor", "#36D6B0",
                    "--theme.backgroundColor", "#0B1120",
                    "--theme.secondaryBackgroundColor", "#151F32",
                ],
                env=env,
                stdout=self._output,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    self._output.seek(0)
                    raise RuntimeError(
                        f"Streamlit exited during startup:\n{self._output.read()}\n"
                        "Install requirements.txt or use --no-dashboard."
                    )
                try:
                    with urlopen(f"{self.url}/_stcore/health", timeout=0.5) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise TimeoutError("Streamlit did not become ready within 30 seconds.")
        except (OSError, RuntimeError, KeyboardInterrupt):
            self.close()
            raise
        print(f"Live training dashboard: {self.url}", flush=True)
        try:
            if not webbrowser.open(self.url):
                LOGGER.warning("Could not open a browser automatically. Open %s to view the dashboard.", self.url)
        except webbrowser.Error as error:
            LOGGER.warning("Could not open a browser: %s. Open %s manually.", error, self.url)
        return self

    @property
    def is_open(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def log(self, step: int, **metrics: float) -> None:
        unknown = set(metrics) - set(self.metrics)
        if unknown:
            raise KeyError(f"Unknown dashboard metrics: {sorted(unknown)}")
        with self._lock:
            for key, value in metrics.items():
                number = float(value)
                self._history[key].append((int(step), number if math.isfinite(number) else None))
        if self._process is not None and not self.is_open and not self._failure_reported:
            LOGGER.error("Dashboard process exited unexpectedly. Training continues without it.")
            self._failure_reported = True

    def set_info(self, text: str) -> None:
        with self._lock:
            self._info = text

    def close(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._output is not None:
            self._output.close()
            self._output = None

    def __enter__(self) -> MetricsDashboard:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
