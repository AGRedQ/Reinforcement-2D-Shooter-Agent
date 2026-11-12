"""Scratch feedforward/(Bi)LSTM PPO with one shared configuration and CPU/CUDA path.

Uses the standard trainer's VecEnv, PPO objective, GAE, curriculum, dashboard and
reports. Architecture controls only observation history and the policy module.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
import os
from pathlib import Path
import time

import numpy as np
import torch

from dashboard import MetricsDashboard
from entities import ACTION_DIMS
from environments import Curriculum, STAGES
from models.mlp import ActorCritic, batch_observations
from models.ppo import (COMBAT_METRICS, EPISODE_METRICS, EXPLORATION_METRICS, PPO_METRICS,
                        VecEnv, compute_gae, episode_metrics, ppo_update)
from models.recurrent import ObservationHistory, RecurrentActorCritic, batch_histories
from models.runtime import configuration, cpu_tree, load_policy, resolve_device, to_device, write_json
from training_report import TrainingReport

TRAINING_FIELDS = (
    'seed', 'num_envs', 'rollout_steps', 'minibatch_size', 'epochs', 'lr', 'gamma',
    'gae_lambda', 'clip', 'ent_coef', 'vf_coef', 'max_grad_norm', 'hidden_size',
    'max_steps', 'map_seeds', 'promotion_reward', 'promotion_window',
    'replay_probability', 'bullet_sensing', 'angular_regions', 'exploration_bonus',
)


def training_settings(args):
    values = configuration(args)
    return {**{key: values[key] for key in TRAINING_FIELDS},
            'shoot_back': True, 'self_play': False, 'optimizer': 'Adam', 'adam_eps': 1e-5}


class FinalStageWindow:
    """The last stage must pass the SAME reward/window criterion as promotions."""
    def __init__(self, window, threshold, rewards=()):
        self.window, self.threshold = window, threshold
        self.rewards = deque(rewards, maxlen=window)

    def record(self, episode):
        if not episode['replay'] and int(episode['stage']) == len(STAGES) - 1:
            self.rewards.append(float(episode['reward']))

    @property
    def complete(self):
        return len(self.rewards) == self.window and float(np.mean(self.rewards)) >= self.threshold

    def state_dict(self):
        return {'rewards': list(self.rewards), 'window': self.window, 'threshold': self.threshold,
                'complete': self.complete}


def policy_batch(policy, items, device):
    batch = batch_histories(items) if isinstance(policy, RecurrentActorCritic) else batch_observations(items)
    return to_device(batch, device)


def new_policy(args):
    if args.architecture == 'feedforward':
        return ActorCritic(hidden_size=args.hidden_size, bullet_sensing=args.bullet_sensing,
                           angular_regions=args.angular_regions)
    return RecurrentActorCritic(hidden_size=args.hidden_size, lstm_hidden_size=args.lstm_hidden_size,
                               context_length=args.context_length, bidirectional=args.architecture == 'bilstm')


def save_training(path, policy, optimizer, curriculum, final_window, progress, history, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    recurrent = isinstance(policy, RecurrentActorCritic)
    torch.save({
        'model': cpu_tree(policy.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
        'architecture': 'window_recurrent_deep_sets_v1' if recurrent else ('deep_sets_v1' if policy.bullet_sensing else 'mlp'),
        'recurrent_config': policy.configuration() if recurrent else None,
        'enemy_sensing': 'angular_regions_v1' if policy.angular_regions else 'rays_v1',
        'obs_size': policy.obs_size, 'action_dims': list(ACTION_DIMS), 'hidden_size': args.hidden_size,
        'curriculum': curriculum.state_dict(), 'final_stage': final_window.state_dict(),
        'curriculum_complete': curriculum.is_final_stage and final_window.complete,
        'completed_stages': list(range(curriculum.stage)) + ([len(STAGES)-1] if final_window.complete else []),
        'stage_history': history, 'initialization': 'random', 'source_sha256': None,
        'training_config': training_settings(args), 'config': configuration(args),
        'training_device': str(next(policy.parameters()).device), **progress,
    }, temporary)
    os.replace(temporary, path)


def train(args):
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model_path = Path(args.checkpoint_path)
    settings = training_settings(args)
    curriculum = Curriculum(threshold=args.promotion_reward, window=args.promotion_window)
    final_window = FinalStageWindow(args.promotion_window, args.promotion_reward)
    progress = {'global_step': 0, 'episodes': 0, 'updates': 0}
    history = []
    if model_path.exists():
        if not args.resume:
            raise FileExistsError(f'{model_path} exists. Use --resume explicitly or a new --model.')
        policy, saved = load_policy(model_path, device)
        if saved.get('initialization') != 'random' or saved.get('source_sha256') is not None:
            raise ValueError('This trainer resumes only its own randomly initialized runs; pretrained weights are not used.')
        if 'training_config' not in saved:
            raise ValueError('This legacy checkpoint lacks the full configuration audit. Start a new run for a controlled comparison.')
        differences = {key: (saved['training_config'].get(key), value) for key, value in settings.items()
                       if saved['training_config'].get(key) != value}
        if differences:
            raise ValueError(f'Resume training configuration differs: {differences}')
        expected = new_policy(args)
        if type(expected) is not type(policy) or (isinstance(policy, RecurrentActorCritic) and expected.configuration() != policy.configuration()):
            raise ValueError('Resume architecture differs from checkpoint')
        curriculum.load_state_dict(saved['curriculum'])
        final_window = FinalStageWindow(args.promotion_window, args.promotion_reward, saved.get('final_stage', {}).get('rewards', ()))
        progress = {key: saved[key] for key in progress}
        history = saved.get('stage_history', [])
    else:
        policy, saved = new_policy(args).to(device), None
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    if saved is not None:
        optimizer.load_state_dict(saved['optimizer'])
    # Architecture construction consumes different RNG draws; reset action RNG
    # identically for the two scratch policies in this shared engine.
    torch.manual_seed(args.seed + progress['global_step'])
    print(f'Architecture={args.architecture} device={device} random initialization; step={progress["global_step"]:,} stage={curriculum.stage}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)
    print(f'Checkpoint: {model_path}\nEffective training settings: {settings}', flush=True)

    def save(path=model_path):
        save_training(path, policy, optimizer, curriculum, final_window, progress, history, args)

    def mastered():
        return curriculum.is_final_stage and final_window.complete

    def completed():
        return args.until_curriculum_complete and mastered() and progress['global_step'] >= args.min_steps

    def status(phase, reason=None):
        elapsed = max(time.monotonic() - started, 1e-9)
        stage_rewards = final_window.rewards if curriculum.is_final_stage else curriculum.rewards
        write_json(args.run_dir / 'status.json', {
            'status': phase, 'reason': reason, **progress, 'architecture': args.architecture,
            'initialization': 'random', 'pretrained_steps': 0, 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'stage': curriculum.stage, 'completed_stages': list(range(curriculum.stage)) + ([len(STAGES)-1] if mastered() else []),
            'curriculum_complete': mastered(), 'stage_window_episodes': len(stage_rewards),
            'stage_window_target': args.promotion_window,
            'stage_mean_reward': float(np.mean(stage_rewards)) if stage_rewards else None,
            'promotion_reward': args.promotion_reward, 'min_steps': args.min_steps, 'total_steps': args.total_steps,
            'steps_per_second': (progress['global_step'] - initial_step) / elapsed,
            'checkpoint': str(model_path), 'updated_at': datetime.now().isoformat(timespec='seconds'),
            'gpu_allocated_mb': torch.cuda.memory_allocated(device)/1024**2 if device.type == 'cuda' else 0,
            'gpu_peak_allocated_mb': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else 0,
        })

    started, initial_step = time.monotonic(), progress['global_step']
    if completed():
        status('trained', 'curriculum complete')
        return policy, {'reason': 'curriculum complete', **progress, 'curriculum_complete': True}
    if args.total_steps is not None and progress['global_step'] >= args.total_steps:
        status('trained', 'step limit')
        return policy, {'reason': 'step limit', **progress, 'curriculum_complete': mastered()}
    if progress['global_step'] % args.num_envs:
        raise ValueError('Resume step count must divide by the environment count')
    save()
    envs = VecEnv(args.num_envs, dict(max_steps=args.max_steps, stage=curriculum.stage,
                    replay_probability=args.replay_probability, shoot_back=True, map_seeds=args.map_seeds,
                    bullet_sensing=args.bullet_sensing, angular_regions=args.angular_regions,
                    exploration_bonus=args.exploration_bonus), seed=args.seed + progress['global_step'], render=args.render)
    recurrent = isinstance(policy, RecurrentActorCritic)
    histories = ObservationHistory(list(envs.observations), args.context_length) if recurrent else None
    metrics = {**EPISODE_METRICS, **COMBAT_METRICS, **PPO_METRICS,
               **(EXPLORATION_METRICS if args.exploration_bonus else {})}
    dashboard = None
    report = None
    recent = deque(maxlen=100)
    stop_reason, finished_after_stop = None, set()
    n = args.num_envs
    policy.train()
    try:
        report = TrainingReport(args.run_dir, model_path, metrics, configuration(args), progress)
        if not args.no_dashboard:
            dashboard = MetricsDashboard(metrics, reference_lines={'mean_reward': args.promotion_reward}, title=f'Training - {model_path.name}').start()
        obs = policy_batch(policy, histories.snapshot() if recurrent else list(envs.observations), device)
        while True:
            horizon = args.rollout_steps
            if args.total_steps is not None:
                horizon = min(horizon, (args.total_steps - progress['global_step']) // n)
            raw = []
            actions_buffer = torch.zeros(horizon, n, len(ACTION_DIMS), dtype=torch.long, device=device)
            logs, values, rewards, dones = (torch.zeros(horizon, n, device=device) for _ in range(4))
            stage_snapshots = []
            used = 0
            for t in range(horizon):
                raw.extend(histories.snapshot() if recurrent else list(envs.observations))
                with torch.no_grad():
                    actions, log_prob, value = policy.act(obs)
                next_obs, reward, terminated, truncated, infos, final_obs = envs.step(actions.cpu().numpy())
                reward = torch.as_tensor(reward, device=device)
                for i in np.flatnonzero(truncated & ~terminated):
                    terminal = [histories.terminal_window(i, final_obs[i])] if recurrent else [final_obs[i]]
                    with torch.no_grad():
                        reward[i] += args.gamma * policy.value(policy_batch(policy, terminal, device)).squeeze()
                actions_buffer[t], logs[t], values[t] = actions, log_prob, value
                rewards[t], dones[t] = reward, torch.as_tensor(terminated | truncated, device=device, dtype=torch.float32)
                if recurrent:
                    histories.advance(list(next_obs), terminated | truncated)
                obs = policy_batch(policy, histories.snapshot() if recurrent else list(next_obs), device)
                progress['global_step'] += n
                used += 1
                if stop_reason is None and ((dashboard is not None and dashboard.stop_requested) or (args.run_dir / 'STOP').exists()):
                    stop_reason = 'stop request'
                ended = []
                for i, info in enumerate(infos):
                    if 'episode' not in info:
                        continue
                    ended.append(i)
                    episode = info['episode']
                    recent.append(episode)
                    progress['episodes'] += 1
                    final_window.record(episode)
                    old_stage = curriculum.stage
                    if curriculum.record(episode):
                        envs.set_stage(curriculum.stage)
                        event = {'completed_stage': old_stage, **progress}
                        history.append(event)
                        stage_snapshots.append(old_stage)
                        print(f'Completed stage {old_stage}; promoted to {curriculum.stage} at step {progress["global_step"]:,}', flush=True)
                    report.record_episode(progress, episode_metrics(curriculum, recent, combat=True))
                    if stop_reason is not None:
                        finished_after_stop.add(i)
                if ended:
                    if stop_reason == 'curriculum complete' and not completed():
                        stop_reason = None
                        finished_after_stop.clear()
                    if dashboard is not None:
                        dashboard.log(progress['global_step'], **episode_metrics(curriculum, recent, combat=True))
                    if stop_reason is None and completed():
                        stop_reason = 'curriculum complete'
                        finished_after_stop.update(ended)
                    if stop_reason is None and args.total_episodes and progress['episodes'] >= args.total_episodes:
                        stop_reason = 'episode limit'
                        finished_after_stop.update(ended)
                if stop_reason is not None and len(finished_after_stop) == n:
                    break
            # Optimize partial final rollouts too; do not count discarded data as training.
            with torch.no_grad():
                last_values = policy.value(obs)
            advantages, returns = compute_gae(rewards[:used], values[:used], dones[:used], last_values, args.gamma, args.gae_lambda)
            stats = ppo_update(policy, optimizer, {
                'obs': policy_batch(policy, raw, device), 'actions': actions_buffer[:used].reshape(-1, len(ACTION_DIMS)),
                'log_probs': logs[:used].reshape(-1), 'advantages': advantages.reshape(-1),
                'returns': returns.reshape(-1), 'values': values[:used].reshape(-1),
            }, args)
            if not all(np.isfinite(value) for key, value in stats.items() if key != 'explained_variance'):
                raise FloatingPointError(f'Nonfinite PPO statistics: {stats}')
            progress['updates'] += 1
            report.ppo = stats
            for stage in stage_snapshots:
                save(model_path.with_name(f'{model_path.stem}_after_stage_{stage}.pt'))
            status('training')
            if dashboard is not None:
                dashboard.log(progress['global_step'], **stats)
                dashboard.set_info(f'model {model_path.name}\ndevice {device}\nsteps {progress["global_step"]:,}\nstage {curriculum.stage}/4\nfinal stage mastered {mastered()}')
            if progress['updates'] % 10 == 0:
                print(f'step={progress["global_step"]:,} episodes={progress["episodes"]} stage={curriculum.stage} '
                      f'reward={np.mean([e["reward"] for e in recent]) if recent else 0:.2f} '
                      f'final_window={len(final_window.rewards)}/{args.promotion_window} '
                      f'SPS={(progress["global_step"]-initial_step)/max(time.monotonic()-started,1e-9):.0f}', flush=True)
            if progress['updates'] % args.save_every == 0:
                save()
            if stop_reason is not None and len(finished_after_stop) == n:
                break
            if args.total_steps is not None and progress['global_step'] >= args.total_steps:
                stop_reason = 'step limit'
                break
    except KeyboardInterrupt:
        stop_reason = 'keyboard interrupt'
    except BaseException as error:
        status('failed', str(error))
        raise
    finally:
        try:
            save()
        finally:
            try:
                if report is not None:
                    report.finish(progress, stop_reason or 'training error')
            finally:
                envs.close()
                if dashboard is not None:
                    dashboard.close()
    status('trained' if stop_reason not in ('keyboard interrupt', 'stop request') else 'stopped', stop_reason)
    print(f'Training stopped: {stop_reason}; step={progress["global_step"]:,}; stage={curriculum.stage}; all stages complete={mastered()}', flush=True)
    return policy, {'reason': stop_reason, **progress, 'curriculum_complete': mastered()}
