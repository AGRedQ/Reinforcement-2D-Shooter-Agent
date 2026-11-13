"""Freeze repository evidence, check reported metrics, and draw report figures.

Run from any directory. Requires the repository's saved primary experiment,
NumPy, PyTorch, Matplotlib, Gymnasium and Pygame. Existing policies are read only.
The LaTeX document can compile from the bundled figures without this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/shooter-report-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RUN = ROOT / "reports/bilstm_gpu_all_stages_20261112"
DATA, FIG = HERE / "data", HERE / "figures"
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from models.evaluation import METRICS, paired_comparison
from models.runtime import load_policy

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
COLORS = {"bilstm": "#176394", "frozen": "#b5571b", "bilstm_current_only": "#527645"}
LABELS = {"bilstm": "BiLSTM", "frozen": "Feedforward control", "bilstm_current_only": "BiLSTM, current only"}

files = {
    "matched_comparison/comparison.json": "matched_comparison.json",
    "matched_comparison/comparison_manifest.json": "matched_manifest.json",
    "matched_comparison/evaluation.csv": "evaluation.csv",
    "status.json": "bilstm_status.json",
    "feedforward_control/status.json": "feedforward_status.json",
    "source_manifest.json": "training_source_manifest.json",
    "experiment.json": "experiment.json",
    "comparison.json": "historical_comparison.json",
}
for source, destination in files.items():
    shutil.copy2(RUN / source, DATA / destination)
for parent, name in [(RUN, "bilstm_training.csv"), (RUN / "feedforward_control", "feedforward_training.csv")]:
    sources = list(parent.glob("*/metrics.csv"))
    if len(sources) != 1:
        raise ValueError(f"Expected one training CSV under {parent}, found {sources}")
    shutil.copy2(sources[0], DATA / name)

result = json.loads((DATA / "matched_comparison.json").read_text())
with (DATA / "evaluation.csv").open(newline="") as f:
    rows = [{key: (value if key == "variant" else float(value)) for key, value in row.items()}
            for row in csv.DictReader(f)]
assert len(rows) == 360
for stage, variants in result["summary_by_stage"].items():
    for variant, summary in variants.items():
        selected = [r for r in rows if r["stage"] == int(stage) and r["variant"] == variant]
        assert len(selected) == summary["episodes"] == 24
        assert len({(r["map_seed"], r["eval_seed"]) for r in selected}) == 24
        for metric in METRICS:
            assert np.isclose(np.mean([r[metric] for r in selected]), summary[metric])
        assert np.isclose(sum(r["hits"] for r in selected) / sum(r["shots"] for r in selected), summary["hit_rate"])
        assert np.isclose(1000 * sum(r["kills"] for r in selected) / sum(r["length"] for r in selected), summary["kills_per_1000_steps"])
    for other in ("frozen", "bilstm_current_only"):
        recomputed = paired_comparison(rows, "bilstm", other, int(stage))
        recorded = result["paired_differences_with_bootstrap_ci95"][stage][f"bilstm_minus_{other}"]
        for metric in METRICS:
            for key in ("mean_delta", "ci95_low", "ci95_high"):
                assert np.isclose(recomputed[metric][key], recorded[metric][key])

audit = {"evaluation_rows": len(rows), "raw_metrics_and_all_bootstrap_intervals_verified": True,
         "checkpoints": {}, "archived_source_matches_working_tree": True}
for key, filename in [("candidate_sha256", "bilstm_gpu_all_stages_20261112.pt"),
                      ("reference_sha256", "bilstm_gpu_all_stages_20261112_feedforward_control.pt")]:
    path = ROOT / "checkpoints" / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == result["manifest"][key]
    policy, saved = load_policy(path)
    audit["checkpoints"][filename] = {
        "sha256": digest, "parameters": sum(p.numel() for p in policy.parameters()),
        "global_step": saved["global_step"], "episodes": saved["episodes"],
        "updates": saved["updates"], "stage_history": saved.get("stage_history", []),
        "curriculum_complete": saved.get("curriculum_complete", False),
    }
for relative, digest in json.loads((DATA / "training_source_manifest.json").read_text())["files_sha256"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
(DATA / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")

fig = plt.figure(figsize=(7.05, 2.80), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=(0.16, 1.0), wspace=0.12)
legend_axis = fig.add_subplot(grid[0, :])
legend_axis.axis("off")
axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
line_styles = {"bilstm": "-", "frozen": "--"}
checkpoint_names = {
    "bilstm": "bilstm_gpu_all_stages_20261112.pt",
    "frozen": "bilstm_gpu_all_stages_20261112_feedforward_control.pt",
}
for filename, variant in [("bilstm_training.csv", "bilstm"), ("feedforward_training.csv", "frozen")]:
    with (DATA / filename).open(newline="") as f:
        records = list(csv.DictReader(f))
    x = np.array([float(r["environment_step"]) for r in records]) / 1e6
    axes[0].plot(x, [float(r["mean_reward"]) for r in records], color=COLORS[variant],
                 ls=line_styles[variant], lw=1.15, label=LABELS[variant])
    checkpoint = audit["checkpoints"][checkpoint_names[variant]]
    events = checkpoint["stage_history"]
    stage_x = [0, *[event["global_step"] for event in events], checkpoint["global_step"]]
    stage_y = [0, *[event["completed_stage"] + 1 for event in events], len(events)]
    axes[1].step(np.asarray(stage_x) / 1e6, stage_y, where="post",
                 color=COLORS[variant], ls=line_styles[variant], lw=1.4)
handles, labels = axes[0].get_legend_handles_labels()
legend = legend_axis.legend(handles, labels, loc="center", ncol=2, frameon=False,
                            handlelength=2.6, columnspacing=2.8)
axes[0].set(xlabel=r"Environment transitions ($10^6$)", ylabel="Mean episodic return",
            title="(a) Training returns")
axes[1].set(xlabel=r"Environment transitions ($10^6$)", ylabel="Current curriculum stage",
            title="(b) Curriculum progression", yticks=range(5), ylim=(-.15, 4.3))
for ax in axes:
    ax.grid(axis="y", alpha=.2)
    ax.set_xlim(0, 2.82)
    ax.tick_params(direction="out")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
legend_box = legend.get_window_extent(renderer)
assert all(not legend_box.overlaps(ax.get_tightbbox(renderer)) for ax in axes)
fig.savefig(FIG / "training_curves.pdf")
fig.savefig(FIG / "training_curves.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.55), layout="constrained")
x, width = np.arange(5), .25
for offset, variant in enumerate(("frozen", "bilstm", "bilstm_current_only")):
    y = [result["summary_by_stage"][str(s)][variant]["combat_success"] for s in range(5)]
    ax.bar(x + (offset - 1) * width, y, width, color=COLORS[variant], label=LABELS[variant])
ax.set(xlabel="Evaluation stage", ylabel="Combat success", xticks=x, ylim=(0, 1.04))
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_yticks([0, .25, .5, .75, 1])
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=1, frameon=False, fontsize=7.5)
fig.savefig(FIG / "stage_success.pdf")
plt.close(fig)

# Actual application frames, with angular sensing explicitly enabled.
import pygame
from renderer import Renderer
for map_seed in (2, 11):
    env = Renderer(fps=0, map_seeds=(map_seed,), stage=2, replay_probability=0,
                   shoot_back=True, angular_regions=True, bullet_sensing=True,
                   show_rays=True, show_visibility=False)
    try:
        env.reset(seed=7)
        surface = pygame.display.get_surface()
        pygame.image.save(surface.subsurface((0, 0, 800, 800)), str(FIG / f"arena_seed_{map_seed}.png"))
    finally:
        env.close()
print("Verified 360 raw trials, all stage summaries, bootstrap intervals, checkpoint hashes and archived source.")
print("Wrote report data and figures to", HERE)
