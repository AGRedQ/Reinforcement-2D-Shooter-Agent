"""Persistent metric samples and an offline HTML report for each training run."""

from __future__ import annotations

import csv
import html
import math
import tempfile
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORT_INTERVAL = 10


class TrainingReport:
    def __init__(
        self, root: Path, model: Path, metrics: dict[str, str], config: dict, progress: dict
    ):
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.directory = Path(tempfile.mkdtemp(prefix=f"{model.stem}-{stamp}-", dir=root))
        self.metrics = metrics
        self.config = config
        self.start_progress = dict(progress)
        self.started = datetime.now().isoformat(timespec="seconds")
        self.model = model.name
        self.csv_path = self.directory / "metrics.csv"
        self.html_path = self.directory / "report.html"
        self.pending = 0
        self.latest: dict = {}
        self.ppo: dict[str, float] = {}
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["episodes", "run_episodes", "environment_step", "updates", *metrics],
        )
        self._writer.writeheader()
        self._file.flush()
        print(f"Training reports: {self.directory}")

    def record_episode(self, progress: dict, values: dict[str, float]) -> None:
        self.pending += 1
        self.latest = {
            "episodes": progress["episodes"],
            "run_episodes": progress["episodes"] - self.start_progress["episodes"],
            "environment_step": progress["global_step"],
            "updates": progress["updates"],
            **values,
        }
        if self.pending == REPORT_INTERVAL:
            self._save_sample()

    def _save_sample(self) -> None:
        row = {**self.latest, **self.ppo}
        self._writer.writerow({
            key: value if math.isfinite(float(value)) else ""
            for key, value in row.items()
        })
        self._file.flush()
        self.pending = 0

    def finish(self, progress: dict, reason: str) -> None:
        try:
            if self.pending:
                self._save_sample()
        finally:
            self._file.close()
        with self.csv_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        details = {
            "Model": self.model,
            "Started": self.started,
            "Finished": datetime.now().isoformat(timespec="seconds"),
            "Stop reason": reason,
            "Starting progress": self.start_progress,
            "Final progress": progress,
            "Configuration": self.config,
        }
        summary = "".join(
            f"<dt>{html.escape(key)}</dt><dd>{html.escape(str(value))}</dd>"
            for key, value in details.items()
        )
        charts = "".join(self._chart(key, label, rows) for key, label in self.metrics.items())
        self.html_path.write_text(
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Training report</title><style>"
            "body{font:16px system-ui;margin:2rem;background:#0b1120;color:#e2e8f0}"
            "dd{margin-bottom:1rem;overflow-wrap:anywhere}dt{font-weight:bold}"
            ".charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}"
            "section{background:#151f32;padding:1rem;border-radius:8px}svg{width:100%;height:auto}"
            "text{fill:#cbd5e1;font:12px system-ui}a{color:#36d6b0}</style>"
            f"<h1>Training report: {html.escape(self.model)}</h1><dl>{summary}</dl>"
            "<p>Samples every 10 completed episodes in this run, plus a final partial sample. "
            "Episode metrics use the last up to 100 completed episodes of this run, including "
            "curriculum replay and all workers. PPO metrics are the latest completed update "
            "at sampling time; blanks mean unavailable/nonfinite. "
            "The stage is the current curriculum stage, not the replay episode's stage. "
            "Resumed runs have separate reports and retain cumulative episode numbering. "
            "These are training metrics, not an independent evaluation.</p>"
            "<p><a href='metrics.csv'>Download metrics CSV</a></p>"
            f"<div class='charts'>{charts}</div></html>",
            encoding="utf-8",
        )
        print(f"Training report saved: {self.html_path}")

    @staticmethod
    def _chart(key: str, label: str, rows: list[dict[str, str]]) -> str:
        points = [
            (float(row["episodes"]), float(row[key]) if row[key] else None)
            for row in rows
        ]
        values = [value for _, value in points if value is not None]
        title = html.escape(label)
        if not values:
            return f"<section><h2>{title}</h2><p>No samples available.</p></section>"
        x_min, x_max = points[0][0], points[-1][0]
        y_min, y_max = min(values), max(values)
        segments: list[str] = []
        segment: list[str] = []
        dots: list[str] = []
        for episode, value in points:
            if value is None:
                if segment:
                    segments.append(" ".join(segment))
                    segment = []
                continue
            x = 70 + 500 * (episode - x_min) / max(x_max - x_min, 1)
            y = 110 if y_max == y_min else 190 - 160 * (value - y_min) / (y_max - y_min)
            segment.append(f"{x:.2f},{y:.2f}")
            dots.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='#36d6b0'>"
                f"<title>Episode {episode:g}: {value:g}</title></circle>"
            )
        if segment:
            segments.append(" ".join(segment))
        lines = "".join(
            f"<polyline points='{segment}' fill='none' stroke='#36d6b0' stroke-width='2'/>"
            for segment in segments
        )
        return (
            f"<section><h2>{title}</h2><svg viewBox='0 0 600 245' role='img' aria-label='{title}'>"
            "<path d='M70 25V200H575' fill='none' stroke='#64748b'/>"
            f"<text x='2' y='35'>{y_max:.4g}</text><text x='2' y='195'>{y_min:.4g}</text>"
            f"<text x='70' y='220'>{x_min:g}</text><text x='540' y='220'>{x_max:g}</text>"
            f"<text x='245' y='240'>Completed episodes</text>{lines}{''.join(dots)}</svg>"
            f"<p>Latest: {values[-1]:.6g}</p></section>"
        )
