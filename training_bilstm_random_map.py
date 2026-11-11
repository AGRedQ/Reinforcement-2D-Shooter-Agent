"""Warm-start a past-window BiLSTM and compare it with angular Deep Sets PPO.

Run (CPU, no dashboard; original checkpoint is never overwritten):
  python training_bilstm_random_map.py --steps 131072 --train-seeds 0 1 2
Quick pilot:
  python training_bilstm_random_map.py --steps 65536 --train-seeds 0 1
Resume to a larger TOTAL additional-step target:
  python training_bilstm_random_map.py --run-dir reports/memory-RUN --resume --steps 262144
Re-evaluate the run's saved policies:
  python training_bilstm_random_map.py --run-dir reports/memory-RUN --evaluate-only

Architecture: current angular state + per-frame Deep Sets bullet summary (96)
 -> one past-window BiLSTM -> zero-initialized residual projection + current
 features -> original 256-256 MLP -> three action heads and a value head.
Both LSTM directions see only history ending at the current observation. Windows
reset at episode boundaries; all encoder/LSTM weights train through full windows.
Default context is 32 observations (~1.03 seconds between first and last at 30 Hz).

Each training seed gets a feedforward continuation control and a recurrent policy,
starting from the SAME supplied weights and a fresh Adam optimizer. Both use the
same transition budget, PPO settings, map cycle and initial curriculum stage.
Evaluation includes the frozen source, the trained control, the trained recurrent
policy, and its current-observation-only history ablation, on identical map/seed/
stage trials with replay disabled. Success := >=1 kill AND alive at time limit.
No success/win condition is added to the environment's reward. Evaluation is
headless and deterministic (argmax); training remains stochastic. Results describe
this bounded-history residual architecture, not every possible LSTM architecture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from entities import ACTION_DIMS
from environments import Curriculum, HuntEnv, STAGES
from map_generation import parse_map_seeds
from models.mlp import ActorCritic, batch_observations
from models.ppo import VecEnv, compute_gae, load_checkpoint, ppo_update
from models.recurrent import ObservationHistory, RecurrentActorCritic, batch_histories

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / "checkpoints/angular_deep_sets_ppo.pt"
METRICS = ("reward", "kills", "combat_success", "survival", "kill_episode", "damage_taken", "length")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def build_policy(source: dict, args, recurrent: bool):
    if recurrent:
        policy = RecurrentActorCritic(source["hidden_size"], args.lstm_hidden_size,
                                      args.context_length, args.recurrent_type == "bilstm")
        if getattr(args, "initialization", "warm_start") != "random":
            policy.warm_start(source["model"])
    else:
        policy = ActorCritic(hidden_size=source["hidden_size"], bullet_sensing=True, angular_regions=True)
        if getattr(args, "initialization", "warm_start") != "random":
            policy.load_state_dict(source["model"])
    return policy


def save_policy(path, policy, optimizer, curriculum, progress, args, source_hash, seed):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
        "curriculum": curriculum.state_dict(), **progress,
        "architecture": "window_recurrent_deep_sets_v1" if isinstance(policy, RecurrentActorCritic) else "deep_sets_v1",
        "enemy_sensing": "angular_regions_v1", "obs_size": 64,
        "action_dims": list(ACTION_DIMS), "hidden_size": policy.trunk[0].out_features,
        "recurrent_config": policy.configuration() if isinstance(policy, RecurrentActorCritic) else None,
        "source_sha256": source_hash, "train_seed": seed,
        "initialization": getattr(args, "initialization", "warm_start"),
        "config": {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, (range, tuple)) else value
                   for key, value in vars(args).items()},
    }, temporary)
    os.replace(temporary, path)


def restore_policy(path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] == "window_recurrent_deep_sets_v1":
        policy = RecurrentActorCritic(**saved["recurrent_config"])
    elif saved["architecture"] == "deep_sets_v1":
        policy = ActorCritic(hidden_size=saved["hidden_size"], bullet_sensing=True, angular_regions=True)
    else:
        raise ValueError(f"Unsupported experiment checkpoint: {path}")
    policy.load_state_dict(saved["model"])
    return policy, saved


def train_variant(source, args, recurrent, seed, source_hash):
    label = f"{args.recurrent_type if recurrent else 'feedforward'}_seed_{seed}"
    path = args.run_dir / f"{label}.pt"
    torch.manual_seed(seed)
    policy = build_policy(source, args, recurrent)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    curriculum = Curriculum(threshold=args.promotion_reward, window=args.promotion_window)
    curriculum.load_state_dict(source["curriculum"])
    if args.train_stage is not None:
        curriculum.stage = args.train_stage
        curriculum.rewards.clear()
    progress = {"global_step": 0, "episodes": 0, "updates": 0}
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"{path} exists; use --resume or a new --run-dir")
        policy, saved = restore_policy(path)
        if saved["source_sha256"] != source_hash:
            raise ValueError("Resume source checkpoint fingerprint differs")
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        curriculum.load_state_dict(saved["curriculum"])
        progress = {key: saved[key] for key in progress}
    if progress["global_step"] >= args.steps:
        print(f"{label}: already trained {progress['global_step']:,} added steps", flush=True)
        return policy, path
    # Scratch runs retain a step-zero checkpoint as proof of initialization.
    if getattr(args, "initialization", "warm_start") == "random" and not path.exists():
        save_policy(path, policy, optimizer, curriculum, progress, args, source_hash, seed)
    policy.train()
    # Give both architectures the same starting action RNG, independent of construction.
    torch.manual_seed(seed + progress["global_step"])
    n, horizon = args.num_envs, args.rollout_steps
    envs = VecEnv(n, dict(max_steps=args.max_steps, stage=curriculum.stage,
                         replay_probability=args.replay_probability, shoot_back=True,
                         map_seeds=tuple(args.train_maps), bullet_sensing=True, angular_regions=True),
                  seed=seed + progress["global_step"])
    histories = ObservationHistory(list(envs.observations), args.context_length) if recurrent else None
    recent = deque(maxlen=100)
    started, start_step = time.monotonic(), progress["global_step"]
    fields = ["step", "episodes", "updates", "stage", "mean_reward", "mean_kills", "death_rate",
              "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance",
              "learning_rate", "steps_per_second", "memory_projection_norm"]
    log_path = args.run_dir / f"training_{label}.csv"
    append = args.resume and log_path.exists()
    print(f"Training {label}: {progress['global_step']:,} -> {args.steps:,} added steps; stage {curriculum.stage}", flush=True)
    try:
        with log_path.open("a" if append else "w", newline="") as log:
            writer = csv.DictWriter(log, fields)
            if not append:
                writer.writeheader()
            while progress["global_step"] < args.steps:
                # A shorter final rollout allows an exact transition budget.
                horizon = min(args.rollout_steps, (args.steps - progress["global_step"]) // n)
                if horizon < 1:
                    raise ValueError("Remaining training steps must be divisible by the environment count")
                observations = []
                actions_buffer = torch.empty(horizon, n, 3, dtype=torch.long)
                logs, values, rewards, dones = (torch.zeros(horizon, n) for _ in range(4))
                for t in range(horizon):
                    windows = histories.snapshot() if recurrent else None
                    observations.extend(windows if recurrent else list(envs.observations))
                    batch = batch_histories(windows) if recurrent else batch_observations(list(envs.observations))
                    with torch.no_grad():
                        actions, log_prob, value = policy.act(batch)
                    next_obs, reward, terminated, truncated, infos, final_obs = envs.step(actions.numpy())
                    reward = torch.tensor(reward)
                    for i in np.flatnonzero(truncated & ~terminated):
                        final_batch = (batch_histories([histories.terminal_window(i, final_obs[i])])
                                       if recurrent else batch_observations([final_obs[i]]))
                        with torch.no_grad():
                            reward[i] += args.gamma * policy.value(final_batch).item()
                    actions_buffer[t], logs[t], values[t] = actions, log_prob, value
                    rewards[t], dones[t] = reward, torch.tensor(terminated | truncated, dtype=torch.float32)
                    if recurrent:
                        histories.advance(list(next_obs), terminated | truncated)
                    progress["global_step"] += n
                    for info in infos:
                        if "episode" in info:
                            episode = info["episode"]
                            recent.append(episode)
                            progress["episodes"] += 1
                            if curriculum.record(episode):
                                envs.set_stage(curriculum.stage)
                batch = batch_histories(histories.snapshot()) if recurrent else batch_observations(list(envs.observations))
                with torch.no_grad():
                    last_values = policy.value(batch)
                advantages, returns = compute_gae(rewards, values, dones, last_values, args.gamma, args.gae_lambda)
                stats = ppo_update(policy, optimizer, {
                    "obs": batch_histories(observations) if recurrent else batch_observations(observations),
                    "actions": actions_buffer.reshape(-1, 3), "log_probs": logs.reshape(-1),
                    "advantages": advantages.reshape(-1), "returns": returns.reshape(-1), "values": values.reshape(-1),
                }, args)
                if not all(math.isfinite(value) for key, value in stats.items() if key != "explained_variance"):
                    raise FloatingPointError(f"Nonfinite PPO statistics: {stats}")
                progress["updates"] += 1
                sps = (progress["global_step"] - start_step) / max(time.monotonic() - started, 1e-9)
                row = {"step": progress["global_step"], "episodes": progress["episodes"],
                       "updates": progress["updates"], "stage": curriculum.stage, **stats,
                       "steps_per_second": sps,
                       "memory_projection_norm": float(policy.memory_projection.weight.detach().norm()) if recurrent else 0.0}
                if recent:
                    row.update(mean_reward=float(np.mean([e["reward"] for e in recent])),
                               mean_kills=float(np.mean([e["kills"] for e in recent])),
                               death_rate=float(np.mean([e["deaths"] for e in recent])))
                writer.writerow({key: value if not isinstance(value, float) or math.isfinite(value) else ""
                                 for key, value in row.items()})
                log.flush()
                status_path = getattr(args, "status_path", None)
                if status_path is not None:
                    write_json(status_path, {
                        "status": "training", "initialization": getattr(args, "initialization", "warm_start"),
                        "target_steps": args.steps, **progress, "stage": curriculum.stage,
                        "steps_per_second": sps,
                        "estimated_seconds_remaining": (args.steps - progress["global_step"]) / max(sps, 1e-9),
                        "checkpoint": str(path), "updated_at": datetime.now().isoformat(timespec="seconds"),
                    })
                if progress["updates"] % args.save_every == 0 or progress["global_step"] == args.steps:
                    save_policy(path, policy, optimizer, curriculum, progress, args, source_hash, seed)
                    print(f"{label}: step={progress['global_step']:,} reward={row.get('mean_reward', 0):.2f} "
                          f"kills={row.get('mean_kills', 0):.2f} deaths={row.get('death_rate', 0):.1%} SPS={sps:.0f}", flush=True)
    finally:
        try:
            save_policy(path, policy, optimizer, curriculum, progress, args, source_hash, seed)
        finally:
            envs.close()
    return policy, path


def evaluate(policy, args, variant, train_seed=-1, current_only=False):
    """Batch independent fixed map/spawn trials; each starts with a clean history."""
    policy.eval()
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
                    actions, _, _ = policy.act(batch, deterministic=True)
                    remaining = []
                    for action, i in zip(actions.numpy(), active):
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
            for other in ("frozen", "feedforward", f"{recurrent}_current_only") if other in variants}
    primary = contrasts[str(args.eval_stages[-1])]
    successes = [value["combat_success"] for value in primary.values()]
    if all(value["ci95_low"] > 0 for value in successes):
        verdict = "Preliminary evidence of higher kill-and-survive success and useful history in this pilot; replicate with more training seeds."
    elif (f"{recurrent}_minus_feedforward" in primary
          and primary[f"{recurrent}_minus_feedforward"]["combat_success"]["ci95_high"] < 0):
        verdict = "The recurrent policy performs worse than the matched feedforward continuation on the primary success metric in this pilot."
    else:
        verdict = "Inconclusive: this pilot does not establish that recurrence improves success and makes useful use of history."
    result = {"manifest": manifest, "success_definition": "At least one kill AND alive at the episode time limit",
              "primary_stage": args.eval_stages[-1], "summary_by_stage": summary, "per_training_seed": per_seed,
              "paired_differences_with_bootstrap_ci95": contrasts, "verdict": verdict,
              "limitations": ["Few training seeds; bootstrap intervals do not establish general superiority.",
                              "Current-only history is an input-distribution ablation, not a separate trained agent.",
                              "Same initial map/spawn seeds do not ensure identical subsequent trajectories.",
                              "Success is a reporting definition; the environment has no built-in win condition.",
                              "Held-out maps are held out from this continuation; the source training map history is not recorded in its checkpoint."]}
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--steps", type=int, default=131072, help="total ADDED transitions per policy and training seed")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--train-maps", type=parse_map_seeds, default="9")
    p.add_argument("--eval-maps", type=parse_map_seeds, default="[10,11,12,13]")
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[10000, 10001, 10002, 10003, 10004, 10005])
    p.add_argument("--eval-stages", nargs="+", type=int, choices=range(len(STAGES)), default=[4])
    p.add_argument("--eval-envs", type=int, default=8)
    p.add_argument("--train-stage", type=int, choices=range(len(STAGES)), default=None)
    p.add_argument("--recurrent-type", choices=("lstm", "bilstm"), default="bilstm")
    p.add_argument("--context-length", type=int, default=32)
    p.add_argument("--lstm-hidden-size", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--promotion-reward", type=float, default=35)
    p.add_argument("--promotion-window", type=int, default=100)
    p.add_argument("--replay-probability", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=16)
    args = p.parse_args(argv)
    positive = (args.steps, args.context_length, args.lstm_hidden_size, args.max_steps,
                args.num_envs, args.rollout_steps, args.minibatch_size, args.epochs, args.eval_envs,
                args.save_every, args.promotion_window)
    if min(positive) < 1 or args.minibatch_size < 2:
        p.error("Budgets and sizes must be positive; minibatch size must be >= 2")
    batch_size = args.num_envs * args.rollout_steps
    if args.steps % batch_size or batch_size % args.minibatch_size:
        p.error("--steps must divide into full rollouts, and rollout size into full minibatches")
    if len(set(args.train_seeds)) != len(args.train_seeds) or len(set(args.eval_seeds)) != len(args.eval_seeds):
        p.error("Seeds must be unique")
    if min(*args.train_seeds, *args.eval_seeds) < 0:
        p.error("Seeds must be nonnegative")
    if len(set(args.eval_maps)) != len(args.eval_maps) or len(set(args.eval_stages)) != len(args.eval_stages):
        p.error("Evaluation maps and stages must be unique")
    if set(args.train_maps) & set(args.eval_maps):
        p.error("Evaluation maps must be disjoint from this continuation's training maps")
    if not 0 <= args.replay_probability <= 1 or not 0 < args.lr or not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        p.error("Invalid learning rate, discount, GAE, or replay probability")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    if (args.resume or args.evaluate_only) and args.run_dir is None:
        raise ValueError("--resume/--evaluate-only requires --run-dir")
    args.run_dir = args.run_dir or ROOT / "reports" / f"memory-{datetime.now():%Y%m%d-%H%M%S}"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "experiment.json"
    if manifest_path.exists():
        if not (args.resume or args.evaluate_only):
            raise FileExistsError("Run already exists; use --resume or --evaluate-only")
        previous = json.loads(manifest_path.read_text())
        if args.evaluate_only:
            for key, value in previous["config"].items():
                if key not in ("run_dir", "resume", "evaluate_only"):
                    setattr(args, key, Path(value) if key == "baseline" else value)
        else:
            variable = {"steps", "lr", "resume", "evaluate_only", "run_dir"}
            for key, value in previous["config"].items():
                current = getattr(args, key)
                current = str(current) if isinstance(current, Path) else list(current) if isinstance(current, (range, tuple)) else current
                if key not in variable and current != value:
                    raise ValueError(f"Resume configuration differs for {key}; repeat the original setting")
    source_hash = fingerprint(args.baseline)
    source = load_checkpoint(args.baseline)
    if source.get("architecture") != "deep_sets_v1" or source.get("enemy_sensing") != "angular_regions_v1":
        raise ValueError("This experiment requires an angular Deep Sets baseline")
    if manifest_path.exists() and previous["source_sha256"] != source_hash:
        raise ValueError("The baseline changed since the experiment began")
    config = {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, (range, tuple)) else value
              for key, value in vars(args).items()}
    manifest = {"source_checkpoint": str(args.baseline.resolve()), "source_sha256": source_hash,
                "source_progress": {key: source[key] for key in ("global_step", "episodes", "updates", "curriculum")},
                "config": config, "torch_version": str(torch.__version__),
                "notes": ["CPU; all trainable weights updated; fresh Adam for each warm start.",
                          "Exact initial feedforward behavior is preserved by zero residual projection.",
                          "Fixed past-only windows, episode resets, no future observations."]}
    if args.evaluate_only:
        for seed in args.train_seeds:
            for variant in ("feedforward", args.recurrent_type):
                path = args.run_dir / f"{variant}_seed_{seed}.pt"
                _, saved = restore_policy(path)
                if saved["source_sha256"] != source_hash:
                    raise ValueError(f"Source fingerprint mismatch in {path}")
    write_json(manifest_path, manifest)
    print(f"Run directory: {args.run_dir}", flush=True)
    torch.manual_seed(0)
    frozen = build_policy(source, args, False)
    frozen_rows_path = args.run_dir / "frozen_evaluation.json"
    # Cache only this manifest's immutable evaluation protocol, never training scores.
    protocol = {key: config[key] for key in ("eval_maps", "eval_seeds", "eval_stages", "max_steps")}
    if frozen_rows_path.exists():
        cached = json.loads(frozen_rows_path.read_text())
        if cached["protocol"] != protocol or cached["source_sha256"] != source_hash:
            raise ValueError("Cached frozen evaluation protocol differs")
        rows = cached["rows"]
    else:
        rows = evaluate(frozen, args, "frozen")
        write_json(frozen_rows_path, {"protocol": protocol, "source_sha256": source_hash, "rows": rows})
    try:
        for seed in args.train_seeds:
            for recurrent in (False, True):
                label = args.recurrent_type if recurrent else "feedforward"
                if args.evaluate_only:
                    policy, _ = restore_policy(args.run_dir / f"{label}_seed_{seed}.pt")
                else:
                    policy, _ = train_variant(source, args, recurrent, seed, source_hash)
                rows.extend(evaluate(policy, args, label, seed))
                if recurrent:
                    rows.extend(evaluate(policy, args, f"{label}_current_only", seed, current_only=True))
                write_json(args.run_dir / "evaluation_progress.json", rows)
        return report_results(rows, args, manifest)
    finally:
        if fingerprint(args.baseline) != source_hash:
            raise RuntimeError("Source checkpoint changed during the experiment")


if __name__ == "__main__":
    main()
