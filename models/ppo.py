"""Shared PPO training and inference helpers for the random-map trainer."""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import signal
import time
from collections import deque
from collections.abc import Sequence
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dashboard import MetricsDashboard
from entities import ACTION_DIMS, ANGULAR_OBSERVATION_SIZE, OBSERVATION_SIZE
from environments import STAGES, Curriculum, HuntEnv
from models.mlp import ActorCritic, Observation, batch_observations, select_observations
from renderer import Renderer
from training_report import REPORTS_DIR, TrainingReport

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
DEFAULT_MODEL_NAME = "mlp_ppo_random_map.pt"
DEEP_SETS_MODEL_NAME = "deep_sets_ppo_random_map.pt"

EPISODE_METRICS = {
    "stage": "Curriculum stage",
    "mean_reward": "Mean reward (last 100 ep)",
    "mean_kills": "Mean kills (last 100 ep)",
    "mean_shots": "Mean shots (last 100 ep)",
    "mean_distance": "Mean distance travelled (last 100 ep)",
    "hit_rate": "Hit rate (last 100 ep)",
    "mean_wasted_shots": "Mean wasted shots (last 100 ep)",
    "mean_detect_steps": "Steps to first detection (last 100 ep)",
}
COMBAT_METRICS = {
    "mean_damage_taken": "Damage taken (last 100 ep)",
    "death_rate": "Death rate (last 100 ep)",
}
PPO_METRICS = {
    "policy_loss": "Policy loss",
    "value_loss": "Value loss",
    "entropy": "Entropy",
    "approx_kl": "Approx KL",
    "clip_fraction": "Clip fraction",
    "explained_variance": "Explained variance",
    "learning_rate": "Learning rate",
}
SELF_PLAY_METRICS = {
    "win_rate": "Win rate (last 100 rounds)",
    "loss_rate": "Loss rate (last 100 rounds)",
    "draw_rate": "Draw rate (last 100 rounds)",
    "opponent_generation": "Frozen opponent generation",
}
EXPLORATION_METRICS = {
    "mean_exploration_reward": "Exploration reward (last 100 ep)",
}


# --------------------------------------------------------------------------- vectorised envs


def _step_autoreset(env: HuntEnv, action: np.ndarray):
    obs, reward, terminated, truncated, info = env.step(action)
    final_obs = None
    if terminated or truncated:
        final_obs = obs
        obs, _ = env.reset()
    return obs, reward, terminated, truncated, info, final_obs


def _env_worker(remote: Connection, env_kwargs: dict, seed: int, self_play: bool = False) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    from self_play import SelfPlayEnv

    env = SelfPlayEnv(**env_kwargs) if self_play else HuntEnv(**env_kwargs)
    observation = env.reset(seed=seed)[0]
    remote.send((observation, env.opponent_observation()) if isinstance(env, SelfPlayEnv) else observation)
    while True:
        command, data = remote.recv()
        if command == "step":
            result = _step_autoreset(env, data)
            remote.send((result, env.opponent_observation()) if isinstance(env, SelfPlayEnv) else result)
        elif command == "set_stage":
            env.set_stage(data)
        elif command == "close":
            remote.close()
            return
        else:
            raise ValueError(f"Unknown command {command!r}")


class VecEnv:
    """Runs environments in worker processes; with ``render`` the first one runs locally in a ``Renderer``."""

    def __init__(self, num_envs: int, env_kwargs: dict, seed: int, render: bool = False, self_play: bool = False):
        from self_play import SelfPlayRenderer

        self.num_envs = num_envs
        self.self_play = self_play
        self.local_env: Renderer | None = None
        ctx = mp.get_context("spawn")
        self.remotes: list[Connection] = []
        self.processes: list[mp.Process] = []
        first_obs: list[Observation] = []
        self.opponent_observations: list[Observation] = []
        if render:
            self.local_env = (SelfPlayRenderer if self_play else Renderer)(**env_kwargs, fps=0)
            observation = self.local_env.reset(seed=seed)[0]
            first_obs.append(observation)
            if isinstance(self.local_env, SelfPlayRenderer):
                self.opponent_observations.append(self.local_env.opponent_observation())
        for i in range(len(first_obs), num_envs):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_env_worker, args=(child, env_kwargs, seed + i, self_play), daemon=True)
            process.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(process)
        for remote in self.remotes:
            data = remote.recv()
            if self_play:
                observation, opponent_observation = data
                first_obs.append(observation)
                self.opponent_observations.append(opponent_observation)
            else:
                first_obs.append(data)
        self.bullet_sensing = env_kwargs.get("bullet_sensing", False)
        if self.bullet_sensing:
            self.observations = first_obs
        else:
            arrays = []
            for observation in first_obs:
                if not isinstance(observation, np.ndarray):
                    raise TypeError("Flat vector environments must return array observations.")
                arrays.append(observation)
            self.observations = np.stack(arrays)

    def step(self, actions: np.ndarray):
        offset = 1 if self.local_env is not None else 0
        for remote, action in zip(self.remotes, actions[offset:]):
            remote.send(("step", action))
        results = [_step_autoreset(self.local_env, actions[0])] if self.local_env is not None else []
        if self.self_play:
            from self_play import SelfPlayRenderer

            self.opponent_observations = []
            if isinstance(self.local_env, SelfPlayRenderer):
                self.opponent_observations.append(self.local_env.opponent_observation())
        for remote in self.remotes:
            data = remote.recv()
            if self.self_play:
                result, opponent_observation = data
                results.append(result)
                self.opponent_observations.append(opponent_observation)
            else:
                results.append(data)
        obs, rewards, terminated, truncated, infos, final_obs = zip(*results)
        self.observations = list(obs) if self.bullet_sensing else np.stack(obs)
        return (
            self.observations,
            np.array(rewards, dtype=np.float32),
            np.array(terminated),
            np.array(truncated),
            list(infos),
            list(final_obs),
        )

    def set_stage(self, stage: int) -> None:
        if self.local_env is not None:
            self.local_env.set_stage(stage)
        for remote in self.remotes:
            remote.send(("set_stage", stage))

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass  # worker already gone
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        if self.local_env is not None:
            self.local_env.close()


# --------------------------------------------------------------------------- checkpoints


def resolve_model_path(name: str | None) -> Path:
    path = Path(name or DEFAULT_MODEL_NAME)
    if not path.suffix:
        path = path.with_suffix(".pt")
    return CHECKPOINTS_DIR / path.name


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    sensing = checkpoint.get("enemy_sensing", "rays_v1")
    if sensing not in ("rays_v1", "angular_regions_v1"):
        raise ValueError(f"{path} uses unsupported enemy sensing {sensing!r}")
    expected_size = ANGULAR_OBSERVATION_SIZE if sensing == "angular_regions_v1" else OBSERVATION_SIZE
    if checkpoint["obs_size"] != expected_size or tuple(checkpoint["action_dims"]) != ACTION_DIMS:
        raise ValueError(
            f"{path} was trained with obs size {checkpoint['obs_size']} / actions {checkpoint['action_dims']}, "
            f"but {sensing} requires {expected_size} / {ACTION_DIMS}."
        )
    architecture = checkpoint.get("architecture", "mlp")
    if architecture not in ("mlp", "deep_sets_v1"):
        raise ValueError(f"{path} uses unsupported architecture {architecture!r}")
    return checkpoint


def save_checkpoint(path: Path, policy: ActorCritic, optimizer, curriculum: Curriculum, progress: dict, hidden_size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "curriculum": curriculum.state_dict(),
            "obs_size": policy.obs_size,
            "enemy_sensing": "angular_regions_v1" if policy.angular_regions else "rays_v1",
            "action_dims": list(ACTION_DIMS),
            "hidden_size": hidden_size,
            "architecture": "deep_sets_v1" if policy.bullet_sensing else "mlp",
            **progress,
        },
        tmp,
    )
    os.replace(tmp, path)


# --------------------------------------------------------------------------- PPO


def ppo_update(policy: ActorCritic, optimizer, batch: dict, args) -> dict[str, float]:
    obs, actions, old_log_probs, advantages, returns, old_values = (
        batch[k] for k in ("obs", "actions", "log_probs", "advantages", "returns", "values")
    )
    size = len(actions)
    stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
    for _ in range(args.epochs):
        permutation = torch.randperm(size)
        for start in range(0, size, args.minibatch_size):
            idx = permutation[start : start + args.minibatch_size]
            log_probs, entropy, values = policy.evaluate(select_observations(obs, idx), actions[idx])
            log_ratio = log_probs - old_log_probs[idx]
            ratio = log_ratio.exp()
            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
            value_loss = 0.5 * (values - returns[idx]).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy_mean.item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clip_fraction"].append(((ratio - 1).abs() > args.clip).float().mean().item())

    result = {k: float(np.mean(v)) for k, v in stats.items()}
    variance = returns.var().item()
    result["explained_variance"] = float("nan") if variance == 0 else 1 - (returns - old_values).var().item() / variance
    result["learning_rate"] = optimizer.param_groups[0]["lr"]
    return result


def compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_values)
    for t in reversed(range(len(rewards))):
        next_values = last_values if t == len(rewards) - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def train(args, *, shoot_back: bool = False, map_seeds: Sequence[int] | None = None) -> None:
    shoot_back = shoot_back or args.self_play
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # the small MLP gains little from threads; leave the cores to env workers
    model_path = resolve_model_path(args.model)
    curriculum = Curriculum(threshold=args.promotion_reward, window=args.promotion_window)
    progress = {"global_step": 0, "episodes": 0, "updates": 0}
    hidden_size = args.hidden_size

    checkpoint = load_checkpoint(model_path) if model_path.exists() else None
    self_play_state = checkpoint.get("self_play") if checkpoint is not None else None
    if self_play_state is not None and not args.self_play:
        raise ValueError("This is a self-play checkpoint. Resume it with --self-play.")
    if args.self_play and checkpoint is not None and self_play_state is None:
        raise ValueError("Self-play must use a separate --model name; refusing to overwrite a combat checkpoint.")
    source = checkpoint
    if args.self_play and source is None:
        source_path = resolve_model_path(args.self_play_init)
        if source_path == model_path:
            raise ValueError("Self-play output must differ from its initialization checkpoint.")
        source = load_checkpoint(source_path)
        print(f"Initializing self-play weights from {source_path}; source will not be modified.")
    if args.self_play:
        if source is None or source.get("architecture") != "deep_sets_v1" or source.get("enemy_sensing") != "angular_regions_v1":
            raise ValueError("Self-play requires an angular-region Deep Sets initialization checkpoint.")
        hidden_size = source["hidden_size"]
    bullet_sensing = args.bullet_sensing
    angular_regions = args.angular_regions
    if args.self_play:
        bullet_sensing = angular_regions = True
    if checkpoint is not None:
        hidden_size = checkpoint["hidden_size"]
        checkpoint_sensing = checkpoint.get("architecture", "mlp") == "deep_sets_v1"
        if args.bullet_sensing_requested is True and not checkpoint_sensing:
            raise ValueError(
                "This is an MLP checkpoint, not a Deep Sets checkpoint. "
                "Use --bullet-sensing with a new --model name; existing weights are not migrated."
            )
        bullet_sensing = checkpoint_sensing
        checkpoint_regions = checkpoint.get("enemy_sensing", "rays_v1") == "angular_regions_v1"
        if args.angular_regions_requested is True and not checkpoint_regions:
            raise ValueError(
                "This checkpoint uses legacy enemy rays. Use --angular-regions with a new --model name; "
                "existing weights are not migrated."
            )
        angular_regions = checkpoint_regions
        if args.bullet_sensing_requested is False and bullet_sensing:
            raise ValueError("--no-bullet-sensing requires a new --model; this checkpoint uses Deep Sets.")
        if args.angular_regions_requested is False and angular_regions:
            raise ValueError("--no-angular-regions requires a new --model; this checkpoint uses angular regions.")
    policy = ActorCritic(hidden_size=hidden_size, bullet_sensing=bullet_sensing, angular_regions=angular_regions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    if args.self_play and checkpoint is None and source is not None:
        policy.load_state_dict(source["model"])
    if checkpoint is not None:
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if not args.self_play:
            curriculum.load_state_dict(checkpoint["curriculum"])
        progress = {k: checkpoint[k] for k in progress}
        print(f"Resuming {model_path} at step {progress['global_step']}, stage {curriculum.stage}")
    else:
        print(f"Creating new model {model_path}")

    opponent = copy.deepcopy(policy).eval().requires_grad_(False) if args.self_play else None
    opponent_generation = 0
    if self_play_state is not None and opponent is not None:
        opponent.load_state_dict(self_play_state["opponent"])
        opponent_generation = self_play_state["generation"]
        args.opponent_refresh = self_play_state["refresh_updates"]

    def save_training_checkpoint() -> None:
        extra = {}
        if opponent is not None:
            extra["self_play"] = {
                "opponent": opponent.state_dict(),
                "generation": opponent_generation,
                "refresh_updates": args.opponent_refresh,
            }
        save_checkpoint(model_path, policy, optimizer, curriculum, {**progress, **extra}, hidden_size)

    env_kwargs = {
        "max_steps": args.max_steps,
        "stage": curriculum.stage,
        "replay_probability": args.replay_probability,
        "shoot_back": shoot_back,
        "map_seeds": map_seeds,
        "bullet_sensing": bullet_sensing,
        "angular_regions": angular_regions,
        "exploration_bonus": args.exploration_bonus,
    }
    envs = VecEnv(
        args.num_envs, env_kwargs, seed=args.seed + progress["global_step"],
        render=args.render, self_play=args.self_play,
    )
    dashboard = None
    if not args.no_dashboard:
        dashboard = MetricsDashboard(
            {
                **EPISODE_METRICS,
                **(COMBAT_METRICS if shoot_back else {}),
                **PPO_METRICS,
                **(SELF_PLAY_METRICS if args.self_play else {}),
                **(EXPLORATION_METRICS if args.exploration_bonus else {}),
            },
            reference_lines={} if args.self_play else {"mean_reward": args.promotion_reward},
            title=f"Training - {model_path.name}",
        )

    n, t_max = args.num_envs, args.rollout_steps
    buf_actions = torch.zeros(t_max, n, len(ACTION_DIMS), dtype=torch.long)
    buf_log_probs, buf_values, buf_rewards, buf_dones = (torch.zeros(t_max, n) for _ in range(4))
    recent = deque(maxlen=100)

    stop_reason: str | None = None
    finished_after_stop: set[int] = set()
    start_time, start_step = time.time(), progress["global_step"]
    obs = batch_observations(list(envs.observations))
    report = None
    try:
        report = TrainingReport(
            args.report_dir, model_path,
            {**EPISODE_METRICS, **(COMBAT_METRICS if shoot_back else {}), **PPO_METRICS,
             **(SELF_PLAY_METRICS if args.self_play else {}),
             **(EXPLORATION_METRICS if args.exploration_bonus else {})},
            {**vars(args), "shoot_back": shoot_back, "map_seeds": map_seeds,
             "bullet_sensing": bullet_sensing, "angular_regions": angular_regions,
             "hidden_size": hidden_size},
            progress,
        )
        if dashboard is not None:
            dashboard.set_info(
                f"model     {model_path.name}\nsteps     {progress['global_step']:,}\nepisodes  {progress['episodes']:,}\n"
                f"maps      {map_seeds if map_seeds is not None else 'fixed arena'}"
            )
            dashboard.start()
        while True:
            rollout_observations: list[Observation] = []
            rollout_complete = True
            for t in range(t_max):
                rollout_observations.extend(envs.observations)
                with torch.no_grad():
                    actions, log_probs, values = policy.act(obs)
                    env_actions = actions.numpy()
                    if opponent is not None:
                        opponent_actions, _, _ = opponent.act(batch_observations(envs.opponent_observations))
                        env_actions = np.stack((env_actions, opponent_actions.numpy()), axis=1)
                next_obs, rewards, terminated, truncated, infos, final_obs = envs.step(env_actions)
                rewards = torch.as_tensor(rewards)
                # Time-limit truncation: the state still had a future, so bootstrap from it.
                for i in np.flatnonzero(truncated & ~terminated):
                    with torch.no_grad():
                        rewards[i] += args.gamma * policy.value(batch_observations([final_obs[i]])).item()

                buf_actions[t], buf_log_probs[t], buf_values[t] = actions, log_probs, values
                buf_rewards[t] = rewards
                buf_dones[t] = torch.as_tensor(terminated | truncated, dtype=torch.float32)
                obs = batch_observations(list(next_obs))
                progress["global_step"] += n

                if stop_reason is None and dashboard is not None and dashboard.stop_requested:
                    stop_reason = "stop button"
                ended = [i for i, info in enumerate(infos) if "episode" in info]
                for i in ended:
                    episode = infos[i]["episode"]
                    recent.append(episode)
                    progress["episodes"] += 1
                    if not args.self_play and curriculum.record(episode):
                        envs.set_stage(curriculum.stage)
                        print(f"Promoted to stage {curriculum.stage} at episode {progress['episodes']}")
                    report.record_episode(
                        progress, episode_metrics(
                            curriculum, recent, combat=shoot_back,
                            opponent_generation=opponent_generation if args.self_play else None,
                        )
                    )
                    if stop_reason is not None:
                        finished_after_stop.add(i)
                if ended:
                    if dashboard is not None:
                        dashboard.log(
                            progress["global_step"],
                            **episode_metrics(
                                curriculum, recent, combat=shoot_back,
                                opponent_generation=opponent_generation if args.self_play else None,
                            ),
                        )
                    if stop_reason is None and args.total_episodes and progress["episodes"] >= args.total_episodes:
                        stop_reason = f"reached {args.total_episodes} episodes"
                        finished_after_stop.update(ended)
                if stop_reason is not None and len(finished_after_stop) == n:
                    rollout_complete = False
                    break
            if not rollout_complete:
                break

            with torch.no_grad():
                last_values = policy.value(obs)
            advantages, returns = compute_gae(buf_rewards, buf_values, buf_dones, last_values, args.gamma, args.gae_lambda)
            batch = {
                "obs": batch_observations(rollout_observations),
                "actions": buf_actions.reshape(-1, len(ACTION_DIMS)),
                "log_probs": buf_log_probs.reshape(-1),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
                "values": buf_values.reshape(-1),
            }
            ppo_stats = ppo_update(policy, optimizer, batch, args)
            progress["updates"] += 1
            if opponent is not None and progress["updates"] % args.opponent_refresh == 0:
                opponent.load_state_dict(policy.state_dict())
                opponent_generation += 1
                print(f"Refreshed frozen opponent: generation {opponent_generation}, update {progress['updates']}")
            report.ppo = ppo_stats

            sps = (progress["global_step"] - start_step) / max(time.time() - start_time, 1e-9)
            if dashboard is None and progress["updates"] % 10 == 0 and recent:
                print(
                    f"step {progress['global_step']:>10,}  ep {progress['episodes']:>6,}  stage {curriculum.stage}  "
                    f"reward {np.mean([e['reward'] for e in recent]):7.2f}  "
                    f"kills {np.mean([e['kills'] for e in recent]):5.2f}  "
                    f"entropy {ppo_stats['entropy']:.3f}  SPS {sps:,.0f}",
                    flush=True,
                )
            if dashboard is not None:
                dashboard.log(progress["global_step"], **ppo_stats)
                dashboard.set_info(
                    f"model     {model_path.name}\n"
                    f"maps      {map_seeds if map_seeds is not None else 'fixed arena'}\n"
                    f"steps     {progress['global_step']:,}\n"
                    f"episodes  {progress['episodes']:,}"
                    + (f" / {args.total_episodes:,}" if args.total_episodes else "")
                    + f"\nupdates   {progress['updates']:,}\n"
                    + (
                        f"opponent  generation {opponent_generation}, refresh every {args.opponent_refresh} updates\n"
                        if args.self_play else
                        f"stage     {curriculum.stage} / {len(STAGES) - 1}\n"
                        f"promotion {len(curriculum.rewards)}/{curriculum.window} ep\n"
                    )
                    + f"SPS       {sps:,.0f}"
                    + (f"\n\nSTOPPING: {stop_reason}" if stop_reason else "")
                )
            if progress["updates"] % args.save_every == 0:
                save_training_checkpoint()
    except KeyboardInterrupt:
        stop_reason = "keyboard interrupt"
    finally:
        try:
            save_training_checkpoint()
        finally:
            try:
                if report is not None:
                    report.finish(progress, stop_reason or "training error")
            finally:
                try:
                    envs.close()
                finally:
                    if dashboard is not None:
                        dashboard.close()
    print(
        f"Training stopped ({stop_reason}). Saved {model_path} at step {progress['global_step']:,}, "
        f"episode {progress['episodes']:,}, stage {curriculum.stage}."
    )


def log_episode_metrics(dashboard: MetricsDashboard | None, progress: dict, curriculum: Curriculum, recent: deque) -> None:
    if dashboard is None:
        return
    dashboard.log(
        progress["global_step"],
        **episode_metrics(curriculum, recent, combat="death_rate" in dashboard.metrics),
    )


def episode_metrics(
    curriculum: Curriculum, recent: deque, *, combat: bool, opponent_generation: int | None = None
) -> dict[str, float]:
    def mean(key: str) -> float:
        return float(np.mean([e[key] for e in recent]))

    shots = sum(e["shots"] for e in recent)
    return dict(
        stage=float(curriculum.stage),
        mean_reward=mean("reward"),
        mean_kills=mean("kills"),
        mean_shots=mean("shots"),
        mean_distance=mean("distance"),
        **(
            {"mean_exploration_reward": mean("exploration_reward")}
            if recent and "exploration_reward" in recent[0] else {}
        ),
        hit_rate=sum(e["hits"] for e in recent) / shots if shots else 0.0,
        mean_wasted_shots=mean("wasted_shots"),
        mean_detect_steps=mean("detect_steps"),
        **(
            {"mean_damage_taken": mean("damage_taken"), "death_rate": mean("deaths")}
            if combat else {}
        ),
        **(
            {"win_rate": mean("win"), "loss_rate": mean("loss"), "draw_rate": mean("draw"),
             "opponent_generation": float(opponent_generation)}
            if opponent_generation is not None else {}
        ),
    )


# --------------------------------------------------------------------------- watch


def watch(args, *, shoot_back: bool = False, map_seeds: Sequence[int] | None = None) -> None:
    from self_play import SelfPlayRenderer

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}")
    checkpoint = load_checkpoint(model_path)
    duel = checkpoint.get("self_play")
    if args.self_play and duel is None:
        raise ValueError("--self-play --watch requires a self-play checkpoint.")
    if duel is not None and args.play_against:
        raise ValueError("Use --watch to view self-play duels; --play-against is not supported in duel mode.")
    bullet_sensing = checkpoint.get("architecture", "mlp") == "deep_sets_v1"
    angular_regions = checkpoint.get("enemy_sensing", "rays_v1") == "angular_regions_v1"
    if args.angular_regions_requested is True and not angular_regions:
        raise ValueError("--angular-regions requires an angular-region checkpoint; legacy checkpoints retain rays")
    if args.bullet_sensing_requested is True and not bullet_sensing:
        raise ValueError("--bullet-sensing requires a trained Deep Sets checkpoint")
    if args.angular_regions_requested is False and angular_regions:
        raise ValueError("--no-angular-regions cannot change a saved policy's observations.")
    if args.bullet_sensing_requested is False and bullet_sensing:
        raise ValueError("--no-bullet-sensing cannot change a saved policy's observations.")
    policy = ActorCritic(hidden_size=checkpoint["hidden_size"], bullet_sensing=bullet_sensing, angular_regions=angular_regions)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    stage = args.stage if args.stage is not None else checkpoint["curriculum"]["stage"]

    opponent = None
    if duel is not None:
        opponent = copy.deepcopy(policy).eval()
        opponent.load_state_dict(duel["opponent"])
    env = (SelfPlayRenderer if duel is not None else Renderer)(
        max_steps=args.max_steps, stage=stage, replay_probability=0.0,
        fps=args.fps, shoot_back=shoot_back, map_seeds=map_seeds,
        bullet_sensing=bullet_sensing,
        angular_regions=angular_regions,
        human_opponent=args.play_against, player_hacks=args.player_hacks,
    )
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode if map_seeds is None else (args.seed if episode == 0 else None)
            obs, _ = env.reset(seed=seed)
            done = False
            info = {}
            while not done and env.window_open:
                with torch.no_grad():
                    action, _, _ = policy.act(batch_observations([obs]), deterministic=not args.stochastic)
                    env_action = action[0].numpy()
                    if opponent is not None and isinstance(env, SelfPlayRenderer):
                        enemy_action, _, _ = opponent.act(
                            batch_observations([env.opponent_observation()]),
                            deterministic=not args.stochastic,
                        )
                        env_action = np.stack((env_action, enemy_action[0].numpy()))
                obs, _, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
            if not env.window_open:
                break
            print(f"episode {episode}: {info['episode']}")
    finally:
        env.close()


def parse_args(
    argv=None, *, description: str | None = __doc__,
    default_model: str = DEFAULT_MODEL_NAME, random_maps: bool = False,
    default_sensing: bool = False,
):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model", default=None,
        help=f"checkpoint name (default {default_model}, or {DEEP_SETS_MODEL_NAME} with --bullet-sensing)",
    )
    p.add_argument(
        "--bullet-sensing", action=argparse.BooleanOptionalAction, default=None,
        help="enable/disable bullet sensing for new policies; saved policies auto-select sensing",
    )
    p.add_argument(
        "--angular-regions", action=argparse.BooleanOptionalAction, default=None,
        help="enable/disable angular enemy regions for new policies; saved policies auto-select sensing",
    )
    p.add_argument(
        "--exploration-bonus", action="store_true",
        help="opt into capped anchor-relative search rewards (scripted-enemy training only)",
    )
    p.add_argument("--max-steps", type=int, default=1000, help="steps per episode")
    p.add_argument("--total-episodes", type=int, default=None, help="stop training after this many episodes")
    p.add_argument("--render", action="store_true", help="show one training environment live")
    p.add_argument("--no-dashboard", action="store_true", help="train without the Streamlit browser dashboard")
    p.add_argument(
        "--report-dir", type=Path, default=REPORTS_DIR,
        help="parent folder for per-run CSV metrics (every 10 episodes) and final offline HTML reports",
    )
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=256, help="steps per environment per PPO update")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--hidden-size", type=int, default=256, help="ignored when resuming")
    p.add_argument("--promotion-reward", type=float, default=35.0)
    p.add_argument("--promotion-window", type=int, default=100)
    p.add_argument("--replay-probability", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=25, help="PPO updates between checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-play", action="store_true", help="train symmetric 1v1 against a periodically frozen policy")
    p.add_argument(
        "--self-play-init", default="angular_deep_sets_ppo.pt",
        help="initial weights for a new self-play run; never overwritten, ignored when resuming",
    )
    p.add_argument("--opponent-refresh", type=int, default=50, help="PPO updates between frozen-opponent refreshes")
    p.add_argument("--watch", action="store_true", help="watch a saved model instead of training")
    p.add_argument("--play-against", action="store_true", help="control a Player threat against the saved agent")
    p.add_argument("--player-hacks", action="store_true", help="start --play-against with wallhack and agent rays")
    p.add_argument("--stage", type=int, default=None, choices=range(len(STAGES)), help="stage for --watch")
    p.add_argument("--episodes", type=int, default=5, help="episodes for --watch")
    p.add_argument("--fps", type=int, default=30, help="frame rate for --watch")
    p.add_argument("--stochastic", action="store_true", help="sample actions in --watch instead of argmax")
    if random_maps:
        from map_generation import MAP_SEEDS_HELP, parse_map_seeds

        p.add_argument("--map-seeds", type=parse_map_seeds, default="9", help=MAP_SEEDS_HELP)
    args = p.parse_args(argv)
    args.bullet_sensing_requested = args.bullet_sensing
    args.angular_regions_requested = args.angular_regions
    if args.bullet_sensing is None:
        args.bullet_sensing = default_sensing
    if args.angular_regions is None:
        args.angular_regions = default_sensing
    if args.exploration_bonus and (args.self_play or args.watch or args.play_against):
        p.error("--exploration-bonus is only supported for scripted-enemy training")
    if args.self_play and (
        args.bullet_sensing_requested is False or args.angular_regions_requested is False
    ):
        p.error("--self-play requires bullet sensing and angular regions")
    if args.player_hacks and not args.play_against:
        p.error("--player-hacks requires --play-against")
    if args.self_play and args.play_against:
        p.error("--self-play cannot be combined with --play-against")
    if min(args.num_envs, args.max_steps, args.rollout_steps, args.save_every, args.opponent_refresh) < 1:
        p.error("environment counts, step limits and save/refresh intervals must be positive")
    if args.model is None:
        if args.self_play:
            args.model = "self_play_ppo.pt"
        elif args.angular_regions:
            args.model = "angular_deep_sets_ppo.pt" if args.bullet_sensing else "angular_mlp_ppo.pt"
        else:
            args.model = DEEP_SETS_MODEL_NAME if args.bullet_sensing else default_model
    return args
