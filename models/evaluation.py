"""Fixed-trial evaluation and paired history-ablation reports, independent of training."""
from __future__ import annotations

import csv
import html
import json
import numpy as np
import torch

from environments import HuntEnv
from models.mlp import batch_observations
from models.recurrent import ObservationHistory, RecurrentActorCritic, batch_histories
from models.runtime import to_device, write_json

METRICS = ("reward", "kills", "combat_success", "survival", "kill_episode", "damage_taken", "length")

def evaluate(policy, args, variant, train_seed=-1, current_only=False):
    """Batch independent fixed map/spawn trials; each starts with a clean history."""
    policy.eval()
    device = next(policy.parameters()).device
    trials = [(stage, map_seed, seed) for stage in args.eval_stages
              for map_seed in args.eval_maps for seed in args.eval_seeds]
    rows = []
    recurrent = isinstance(policy, RecurrentActorCritic)
    for offset in range(0, len(trials), args.eval_envs):
        subset = trials[offset:offset + args.eval_envs]
        envs, observations = [], []
        try:
            for stage, map_seed, seed in subset:
                env = HuntEnv(stage=stage, max_steps=args.max_steps, shoot_back=True,
                              map_seeds=(map_seed,), replay_probability=0.0,
                              bullet_sensing=True, angular_regions=True)
                envs.append(env)
                observations.append(env.reset(seed=seed)[0])
            histories = ObservationHistory(observations, 1 if current_only else policy.context_length) if recurrent else None
            active = list(range(len(envs)))
            with torch.inference_mode():
                while active:
                    batch = (batch_histories([tuple(histories.windows[i]) for i in active]) if recurrent
                             else batch_observations([observations[i] for i in active]))
                    actions, _, _ = policy.act(to_device(batch, device), deterministic=True)
                    remaining = []
                    for action, i in zip(actions.cpu().numpy(), active):
                        obs, _, terminated, truncated, info = envs[i].step(action)
                        observations[i] = obs
                        if terminated or truncated:
                            episode = info["episode"]
                            stage, map_seed, seed = subset[i]
                            survived = float(not terminated)
                            rows.append({"variant": variant, "train_seed": train_seed, "stage": stage,
                                         "map_seed": map_seed, "eval_seed": seed, **episode,
                                         "combat_success": float(episode["kills"] >= 1 and survived),
                                         "survival": survived, "kill_episode": float(episode["kills"] >= 1)})
                        else:
                            if recurrent:
                                histories.windows[i].append(obs)
                            remaining.append(i)
                    active = remaining
        finally:
            for env in envs:
                env.close()
        print(f"Evaluated {variant} seed={train_seed}: {min(offset + args.eval_envs, len(trials))}/{len(trials)} episodes", flush=True)
    return rows


def paired_comparison(rows, first, second, stage, bootstrap_samples=4000):
    """Hierarchical paired bootstrap over training seeds and map/spawn trials.

    Intervals describe this finite pilot and do not remove bias from selecting
    maps, architecture, or training settings. Few training seeds remain a limit.
    """
    a = [row for row in rows if row["variant"] == first and row["stage"] == stage]
    b = [row for row in rows if row["variant"] == second and row["stage"] == stage]
    key = lambda row: (row["map_seed"], row["eval_seed"])
    seeds = sorted({row["train_seed"] for row in a})
    differences = []
    for seed in seeds:
        left = {key(row): row for row in a if row["train_seed"] == seed}
        right = {key(row): row for row in b if row["train_seed"] == (-1 if second == "frozen" else seed)}
        if set(left) != set(right) or not left:
            raise ValueError("Paired comparisons require matching evaluation trials")
        differences.append([[left[k][metric] - right[k][metric] for metric in METRICS] for k in sorted(left)])
    delta = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(419)
    sampled_seeds = rng.integers(0, len(seeds), (bootstrap_samples, len(seeds)))
    sampled_trials = rng.integers(0, delta.shape[1], (bootstrap_samples, delta.shape[1]))
    sampled = delta[sampled_seeds[:, :, None], sampled_trials[:, None, :]].mean(axis=(1, 2))
    return {metric: {"mean_delta": float(delta[:, :, i].mean()),
                     "ci95_low": float(np.quantile(sampled[:, i], 0.025)),
                     "ci95_high": float(np.quantile(sampled[:, i], 0.975))}
            for i, metric in enumerate(METRICS)}


def report_results(rows, args, manifest):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (args.run_dir / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    variants = sorted({row["variant"] for row in rows})
    summary = {}
    per_seed = {}
    for stage in args.eval_stages:
        summary[str(stage)] = {}
        for variant in variants:
            selected = [row for row in rows if row["stage"] == stage and row["variant"] == variant]
            values = {metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}
            shots = sum(row["shots"] for row in selected)
            values.update(episodes=len(selected), death_rate=1 - values["survival"],
                          hit_rate=sum(row["hits"] for row in selected) / shots if shots else 0.0,
                          kills_per_1000_steps=1000 * sum(row["kills"] for row in selected) / sum(row["length"] for row in selected))
            summary[str(stage)][variant] = values
            for seed in sorted({row["train_seed"] for row in selected}):
                subset = [row for row in selected if row["train_seed"] == seed]
                per_seed[f"stage{stage}/{variant}/seed{seed}"] = {
                    metric: float(np.mean([row[metric] for row in subset])) for metric in METRICS}
    recurrent = args.recurrent_type
    contrasts = {}
    for stage in args.eval_stages:
        contrasts[str(stage)] = {
            f"{recurrent}_minus_{other}": paired_comparison(rows, recurrent, other, stage)
            for other in ("frozen", "feedforward", f"{recurrent}_current_only") if other in variants and other != recurrent}
    primary = contrasts[str(args.eval_stages[-1])]
    reference = primary.get(f"{recurrent}_minus_frozen", {}).get("combat_success")
    memory = primary.get(f"{recurrent}_minus_{recurrent}_current_only", {}).get("combat_success")
    reference_verdict = ("No reference comparison available." if reference is None else
        "Success exceeds the reference on these trials." if reference["ci95_low"] > 0 else
        "Success is below the reference on these trials." if reference["ci95_high"] < 0 else
        "The reference success difference is uncertain; its 95% interval spans zero.")
    memory_verdict = ("No history ablation was performed." if memory is None else
        "Removing history reduces success on these trials; the ablation also changes the input distribution."
        if memory["ci95_low"] > 0 else "The history-ablation success difference is uncertain.")
    verdict = reference_verdict + " " + memory_verdict
    result = {"manifest": manifest, "success_definition": "At least one kill AND alive at the episode time limit",
              "primary_stage": args.eval_stages[-1], "summary_by_stage": summary, "per_training_seed": per_seed,
              "paired_differences_with_bootstrap_ci95": contrasts, "verdict": verdict, "reference_verdict": reference_verdict, "memory_verdict": memory_verdict,
              "limitations": ["Few training seeds; these bootstrap intervals reflect sampled trials, not general architecture superiority.",
                              "Current-only history is an input-distribution ablation, not a separate trained agent.",
                              "Same initial map/spawn seeds do not ensure identical subsequent trajectories.",
                              "Success is a reporting definition; the environment has no built-in win condition.",
                              "Evaluation maps are excluded from the new training run; the historical reference map selection may be unknown."]}
    write_json(args.run_dir / "comparison.json", result)
    table_rows = []
    for stage, values in summary.items():
        for variant, metrics in values.items():
            table_rows.append(f"<tr><td>{stage}</td><td>{html.escape(variant)}</td><td>{metrics['episodes']}</td>"
                              f"<td>{metrics['reward']:.2f}</td><td>{metrics['kills']:.2f}</td>"
                              f"<td>{metrics['combat_success']:.1%}</td><td>{metrics['death_rate']:.1%}</td>"
                              f"<td>{metrics['hit_rate']:.1%}</td><td>{metrics['damage_taken']:.1f}</td></tr>")
    ci_rows = []
    for stage, comparisons in contrasts.items():
        for name, metrics in comparisons.items():
            for metric in ("combat_success", "reward", "kills", "survival"):
                value = metrics[metric]
                ci_rows.append(f"<tr><td>{stage}</td><td>{html.escape(name)}</td><td>{metric}</td>"
                               f"<td>{value['mean_delta']:.4f}</td><td>[{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]</td></tr>")
    document = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Memory policy comparison</title>"
                "<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem;background:#111827;color:#e5e7eb}"
                "table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{padding:.6rem;border:1px solid #475569;text-align:left}"
                "pre{white-space:pre-wrap}a{color:#67e8f9}</style><h1>Past-window recurrent PPO experiment</h1>"
                f"<p><strong>{html.escape(verdict)}</strong></p>"
                "<p>Success = at least one kill and survival to the time limit. Higher reward, kills, success, survival and hit rate are desirable; lower deaths and damage are desirable. "
                "Evaluation is argmax, with fixed stages/maps/spawn seeds and no curriculum replay. Per-seed results and raw episodes are included in the linked files.</p>"
                "<p><a href='evaluation.csv'>Episodes CSV</a> · <a href='comparison.json'>Full results and intervals JSON</a></p>"
                "<table><tr><th>Stage</th><th>Policy</th><th>Episodes</th><th>Reward</th><th>Kills</th><th>Success</th><th>Deaths</th><th>Hit rate</th><th>Damage</th></tr>"
                + "".join(table_rows) + "</table><h2>Paired differences and 95% bootstrap intervals</h2>"
                "<p>First policy minus second. Rate differences are fractions (0.10 = 10 percentage points). Positive survival means fewer deaths. "
                "Training seeds and paired map/spawn trials are resampled; uncertainty with few training seeds remains poorly estimated.</p>"
                "<table><tr><th>Stage</th><th>Comparison</th><th>Metric</th><th>Mean difference</th><th>95% interval</th></tr>"
                + "".join(ci_rows) + "</table><h2>Configuration and provenance</h2><pre>"
                + html.escape(json.dumps(manifest, indent=2)) + "</pre><h2>Limits</h2><ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in result["limitations"]) + "</ul></html>")
    (args.run_dir / "comparison.html").write_text(document)
    print(json.dumps({"summary_by_stage": summary, "verdict": verdict}, indent=2), flush=True)
    print(f"Comparison saved: {args.run_dir / 'comparison.html'}", flush=True)
    return result
