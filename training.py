"""PPO (MLP, three action heads) with curriculum learning for the hunting environment.

Train:  python training.py [--model NAME] [--max-steps 1000] [--total-episodes N] [--render]
Watch:  python training.py --watch [--model NAME] [--stage N]

``--model``: empty uses ``models/mlp_ppo_currilearning.pt``. A given name is looked up in
``models/`` (``.pt`` is added if it has no extension); training resumes from it if it
exists, otherwise a new model is created under that name.

Training automatically opens a local Streamlit/Plotly dashboard with live
metrics, CSV export and a graceful Stop button. ``--no-dashboard`` disables it.
Closing the browser does not interrupt training; the server exits with training.
"""

from __future__ import annotations

import argparse
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
from torch.distributions import Categorical

from dashboard import MetricsDashboard
from entities import ACTION_DIMS, OBSERVATION_SIZE
from environments import STAGES, Curriculum, HuntEnv
from renderer import Renderer

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_NAME = "mlp_ppo_currilearning.pt"

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


# --------------------------------------------------------------------------- model


def _layer_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Shared MLP trunk feeding one categorical head per action component plus a value head."""

    def __init__(self, obs_size: int = OBSERVATION_SIZE, action_dims=ACTION_DIMS, hidden_size: int = 256):
        super().__init__()
        gain = float(np.sqrt(2.0))
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_size, hidden_size), gain),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size), gain),
            nn.Tanh(),
        )
        self.action_heads = nn.ModuleList(_layer_init(nn.Linear(hidden_size, n), 0.01) for n in action_dims)
        self.value_head = _layer_init(nn.Linear(hidden_size, 1), 1.0)

    def _forward(self, obs: torch.Tensor) -> tuple[list[Categorical], torch.Tensor]:
        hidden = self.trunk(obs)
        return [Categorical(logits=head(hidden)) for head in self.action_heads], self.value_head(hidden).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dists, value = self._forward(obs)
        if deterministic:
            actions = torch.stack([d.probs.argmax(-1) for d in dists], dim=-1)
        else:
            actions = torch.stack([d.sample() for d in dists], dim=-1)
        log_prob = sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        return actions, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dists, value = self._forward(obs)
        log_prob = sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return log_prob, entropy, value


# --------------------------------------------------------------------------- vectorised envs


def _step_autoreset(env: HuntEnv, action: np.ndarray):
    obs, reward, terminated, truncated, info = env.step(action)
    final_obs = None
    if terminated or truncated:
        final_obs = obs
        obs, _ = env.reset()
    return obs, reward, terminated, truncated, info, final_obs


def _env_worker(remote: Connection, env_kwargs: dict, seed: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    env = HuntEnv(**env_kwargs)
    remote.send(env.reset(seed=seed)[0])
    while True:
        command, data = remote.recv()
        if command == "step":
            remote.send(_step_autoreset(env, data))
        elif command == "set_stage":
            env.set_stage(data)
        elif command == "close":
            remote.close()
            return
        else:
            raise ValueError(f"Unknown command {command!r}")


class VecEnv:
    """Runs environments in worker processes; with ``render`` the first one runs locally in a ``Renderer``."""

    def __init__(self, num_envs: int, env_kwargs: dict, seed: int, render: bool = False):
        self.num_envs = num_envs
        self.local_env: Renderer | None = None
        ctx = mp.get_context("spawn")
        self.remotes: list[Connection] = []
        self.processes: list[mp.Process] = []
        first_obs = []
        if render:
            self.local_env = Renderer(**env_kwargs, fps=0)
            first_obs.append(self.local_env.reset(seed=seed)[0])
        for i in range(len(first_obs), num_envs):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_env_worker, args=(child, env_kwargs, seed + i), daemon=True)
            process.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(process)
        first_obs.extend(remote.recv() for remote in self.remotes)
        self.observations = np.stack(first_obs)

    def step(self, actions: np.ndarray):
        offset = 1 if self.local_env is not None else 0
        for remote, action in zip(self.remotes, actions[offset:]):
            remote.send(("step", action))
        results = [_step_autoreset(self.local_env, actions[0])] if self.local_env is not None else []
        results.extend(remote.recv() for remote in self.remotes)
        obs, rewards, terminated, truncated, infos, final_obs = zip(*results)
        self.observations = np.stack(obs)
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
    return MODELS_DIR / path.name


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["obs_size"] != OBSERVATION_SIZE or tuple(checkpoint["action_dims"]) != ACTION_DIMS:
        raise ValueError(
            f"{path} was trained with obs size {checkpoint['obs_size']} / actions {checkpoint['action_dims']}, "
            f"but the environment now uses {OBSERVATION_SIZE} / {ACTION_DIMS}."
        )
    return checkpoint


def save_checkpoint(path: Path, policy: ActorCritic, optimizer, curriculum: Curriculum, progress: dict, hidden_size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "curriculum": curriculum.state_dict(),
            "obs_size": OBSERVATION_SIZE,
            "action_dims": list(ACTION_DIMS),
            "hidden_size": hidden_size,
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
    size = len(obs)
    stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
    for _ in range(args.epochs):
        permutation = torch.randperm(size)
        for start in range(0, size, args.minibatch_size):
            idx = permutation[start : start + args.minibatch_size]
            log_probs, entropy, values = policy.evaluate(obs[idx], actions[idx])
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
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)  # the small MLP gains little from threads; leave the cores to env workers
    model_path = resolve_model_path(args.model)
    curriculum = Curriculum(threshold=args.promotion_reward, window=args.promotion_window)
    progress = {"global_step": 0, "episodes": 0, "updates": 0}
    hidden_size = args.hidden_size

    checkpoint = load_checkpoint(model_path) if model_path.exists() else None
    if checkpoint is not None:
        hidden_size = checkpoint["hidden_size"]
    policy = ActorCritic(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    if checkpoint is not None:
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        curriculum.load_state_dict(checkpoint["curriculum"])
        progress = {k: checkpoint[k] for k in progress}
        print(f"Resuming {model_path} at step {progress['global_step']}, stage {curriculum.stage}")
    else:
        print(f"Creating new model {model_path}")

    env_kwargs = {
        "max_steps": args.max_steps,
        "stage": curriculum.stage,
        "replay_probability": args.replay_probability,
        "shoot_back": shoot_back,
        "map_seeds": map_seeds,
    }
    envs = VecEnv(args.num_envs, env_kwargs, seed=args.seed + progress["global_step"], render=args.render)
    dashboard = None
    if not args.no_dashboard:
        dashboard = MetricsDashboard(
            {
                **EPISODE_METRICS,
                **(COMBAT_METRICS if shoot_back else {}),
                **PPO_METRICS,
            },
            reference_lines={"mean_reward": args.promotion_reward},
            title=f"Training - {model_path.name}",
        )

    n, t_max = args.num_envs, args.rollout_steps
    buf_obs = torch.zeros(t_max, n, OBSERVATION_SIZE)
    buf_actions = torch.zeros(t_max, n, len(ACTION_DIMS), dtype=torch.long)
    buf_log_probs, buf_values, buf_rewards, buf_dones = (torch.zeros(t_max, n) for _ in range(4))
    recent = deque(maxlen=100)

    stop_reason: str | None = None
    finished_after_stop: set[int] = set()
    start_time, start_step = time.time(), progress["global_step"]
    obs = torch.as_tensor(envs.observations)
    try:
        if dashboard is not None:
            dashboard.set_info(
                f"model     {model_path.name}\nsteps     {progress['global_step']:,}\nepisodes  {progress['episodes']:,}\n"
                f"maps      {map_seeds if map_seeds is not None else 'fixed arena'}"
            )
            dashboard.start()
        while True:
            rollout_complete = True
            for t in range(t_max):
                with torch.no_grad():
                    actions, log_probs, values = policy.act(obs)
                next_obs, rewards, terminated, truncated, infos, final_obs = envs.step(actions.numpy())
                rewards = torch.as_tensor(rewards)
                # Time-limit truncation: the state still had a future, so bootstrap from it.
                for i in np.flatnonzero(truncated & ~terminated):
                    with torch.no_grad():
                        rewards[i] += args.gamma * policy.value(torch.as_tensor(final_obs[i])).item()

                buf_obs[t], buf_actions[t], buf_log_probs[t], buf_values[t] = obs, actions, log_probs, values
                buf_rewards[t] = rewards
                buf_dones[t] = torch.as_tensor(terminated | truncated, dtype=torch.float32)
                obs = torch.as_tensor(next_obs)
                progress["global_step"] += n

                if stop_reason is None and dashboard is not None and dashboard.stop_requested:
                    stop_reason = "stop button"
                ended = [i for i, info in enumerate(infos) if "episode" in info]
                for i in ended:
                    episode = infos[i]["episode"]
                    recent.append(episode)
                    progress["episodes"] += 1
                    if curriculum.record(episode):
                        envs.set_stage(curriculum.stage)
                        print(f"Promoted to stage {curriculum.stage} at episode {progress['episodes']}")
                    if stop_reason is not None:
                        finished_after_stop.add(i)
                if ended:
                    log_episode_metrics(dashboard, progress, curriculum, recent)
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
                "obs": buf_obs.reshape(-1, OBSERVATION_SIZE),
                "actions": buf_actions.reshape(-1, len(ACTION_DIMS)),
                "log_probs": buf_log_probs.reshape(-1),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
                "values": buf_values.reshape(-1),
            }
            ppo_stats = ppo_update(policy, optimizer, batch, args)
            progress["updates"] += 1

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
                    f"stage     {curriculum.stage} / {len(STAGES) - 1}\n"
                    f"promotion {len(curriculum.rewards)}/{curriculum.window} ep\n"
                    f"SPS       {sps:,.0f}"
                    + (f"\n\nSTOPPING: {stop_reason}" if stop_reason else "")
                )
            if progress["updates"] % args.save_every == 0:
                save_checkpoint(model_path, policy, optimizer, curriculum, progress, hidden_size)
    except KeyboardInterrupt:
        stop_reason = "keyboard interrupt"
    finally:
        save_checkpoint(model_path, policy, optimizer, curriculum, progress, hidden_size)
        envs.close()
        if dashboard is not None:
            dashboard.close()
    print(
        f"Training stopped ({stop_reason}). Saved {model_path} at step {progress['global_step']:,}, "
        f"episode {progress['episodes']:,}, stage {curriculum.stage}."
    )


def log_episode_metrics(dashboard: MetricsDashboard | None, progress: dict, curriculum: Curriculum, recent: deque) -> None:
    if dashboard is None:
        return

    def mean(key: str) -> float:
        return float(np.mean([e[key] for e in recent]))

    shots = sum(e["shots"] for e in recent)
    dashboard.log(
        progress["global_step"],
        stage=curriculum.stage,
        mean_reward=mean("reward"),
        mean_kills=mean("kills"),
        mean_shots=mean("shots"),
        mean_distance=mean("distance"),
        hit_rate=sum(e["hits"] for e in recent) / shots if shots else 0.0,
        mean_wasted_shots=mean("wasted_shots"),
        mean_detect_steps=mean("detect_steps"),
        **(
            {"mean_damage_taken": mean("damage_taken"), "death_rate": mean("deaths")}
            if "death_rate" in dashboard.metrics else {}
        ),
    )


# --------------------------------------------------------------------------- watch


def watch(args, *, shoot_back: bool = False, map_seeds: Sequence[int] | None = None) -> None:
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}")
    checkpoint = load_checkpoint(model_path)
    policy = ActorCritic(hidden_size=checkpoint["hidden_size"])
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    stage = args.stage if args.stage is not None else checkpoint["curriculum"]["stage"]

    env = Renderer(
        max_steps=args.max_steps, stage=stage, replay_probability=0.0,
        fps=args.fps, shoot_back=shoot_back, map_seeds=map_seeds,
    )
    for episode in range(args.episodes):
        seed = args.seed + episode if map_seeds is None else (args.seed if episode == 0 else None)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done and env.window_open:
            with torch.no_grad():
                action, _, _ = policy.act(torch.as_tensor(obs), deterministic=not args.stochastic)
            obs, _, terminated, truncated, info = env.step(action.numpy())
            done = terminated or truncated
        if not env.window_open:
            break
        print(f"episode {episode}: {info['episode']}")
    env.close()


def parse_args(
    argv=None, *, description: str | None = __doc__,
    default_model: str = DEFAULT_MODEL_NAME, random_maps: bool = False,
):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model, help=f"model name in models/ (default {default_model})")
    p.add_argument("--max-steps", type=int, default=1000, help="steps per episode")
    p.add_argument("--total-episodes", type=int, default=None, help="stop training after this many episodes")
    p.add_argument("--render", action="store_true", help="show one training environment live")
    p.add_argument("--no-dashboard", action="store_true", help="train without the Streamlit browser dashboard")
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
    p.add_argument("--watch", action="store_true", help="watch a saved model instead of training")
    p.add_argument("--stage", type=int, default=None, choices=range(len(STAGES)), help="stage for --watch")
    p.add_argument("--episodes", type=int, default=5, help="episodes for --watch")
    p.add_argument("--fps", type=int, default=30, help="frame rate for --watch")
    p.add_argument("--stochastic", action="store_true", help="sample actions in --watch instead of argmax")
    if random_maps:
        from map_generation import MAP_SEEDS_HELP, parse_map_seeds

        p.add_argument("--map-seeds", type=parse_map_seeds, default="9", help=MAP_SEEDS_HELP)
    return p.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.watch:
        watch(arguments)
    else:
        train(arguments)
