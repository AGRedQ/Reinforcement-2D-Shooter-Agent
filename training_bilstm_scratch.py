"""Train angular Deep Sets + BiLSTM + MLP end to end from RANDOM initialization.

Start exactly two million environment transitions (all four envs combined):
  python -u training_bilstm_scratch.py --steps 2000000 --run-dir reports/bilstm-scratch-2m
Resume only this run's own scratch checkpoint:
  python -u training_bilstm_scratch.py --steps 2000000 --run-dir reports/bilstm-scratch-2m --resume

No checkpoint is read to initialize training. New bullet encoder, BiLSTM, residual
projection, MLP, heads and Adam state are created; curriculum begins at stage 0.
The residual projection uses the architecture's zero initialization, while other
network weights use their normal initializers. All layers train jointly via PPO.
The last rollout is shortened to hit the EXACT --steps target without overshoot.

Every 16 PPO updates a checkpoint is saved. status.json and training CSV record
live progress. On completion, evaluation compares the trained model, its history
ablation and --compare-to checkpoint, then writes comparison.html/JSON/CSV.
The comparison checkpoint is loaded ONLY AFTER training, for inference. It has a
different training history/budget, so this direct benchmark is not a matched
from-scratch architecture experiment. --skip-evaluation trains and saves only.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from environments import STAGES
from map_generation import parse_map_seeds
from models.ppo import load_checkpoint
from models.mlp import ActorCritic
from training_bilstm_random_map import (
    DEFAULT_BASELINE, ROOT, evaluate, fingerprint, report_results,
    train_variant, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, default=2000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--context-length', type=int, default=32)
    parser.add_argument('--lstm-hidden-size', type=int, default=48)
    parser.add_argument('--recurrent-type', choices=('lstm', 'bilstm'), default='bilstm')
    parser.add_argument('--train-maps', type=parse_map_seeds, default='9')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--rollout-steps', type=int, default=128)
    parser.add_argument('--minibatch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--clip', type=float, default=.2)
    parser.add_argument('--ent-coef', type=float, default=.01)
    parser.add_argument('--vf-coef', type=float, default=.5)
    parser.add_argument('--max-grad-norm', type=float, default=.5)
    parser.add_argument('--promotion-reward', type=float, default=35.)
    parser.add_argument('--promotion-window', type=int, default=100)
    parser.add_argument('--replay-probability', type=float, default=.2)
    parser.add_argument('--save-every', type=int, default=16)
    parser.add_argument('--compare-to', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--eval-maps', type=parse_map_seeds, default='[14,15,16,17]')
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[20000, 20001, 20002, 20003, 20004, 20005])
    parser.add_argument('--eval-stages', nargs='+', type=int, choices=range(len(STAGES)), default=[4])
    parser.add_argument('--eval-envs', type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.steps, args.hidden_size, args.context_length, args.lstm_hidden_size, args.max_steps,
           args.num_envs, args.rollout_steps, args.epochs, args.save_every, args.promotion_window, args.eval_envs) < 1:
        parser.error('Step counts and sizes must be positive')
    if args.minibatch_size < 2 or args.steps % args.num_envs:
        parser.error('Minibatches need >=2 samples and --steps must be divisible by --num-envs')
    for size in (args.num_envs * args.rollout_steps, args.steps % (args.num_envs * args.rollout_steps)):
        if size and size % args.minibatch_size == 1:
            parser.error('A minibatch with one sample would invalidate advantage normalization')
    if args.seed < 0 or min(args.eval_seeds) < 0:
        parser.error('Seeds must be nonnegative')
    if set(args.train_maps) & set(args.eval_maps):
        parser.error('Evaluation maps must be separate from training maps')
    if len(set(args.eval_maps)) != len(args.eval_maps) or len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error('Evaluation maps and seeds must be unique')
    if args.lr <= 0 or not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or not 0 <= args.replay_probability <= 1:
        parser.error('Invalid learning rate, discount, GAE, or replay probability')
    if args.resume and args.run_dir is None:
        parser.error('--resume requires --run-dir')
    args.run_dir = args.run_dir or ROOT / 'reports' / f'bilstm-scratch-{datetime.now():%Y%m%d-%H%M%S}'
    args.initialization = 'random'
    args.train_stage = None
    args.train_seeds = [args.seed]
    args.status_path = args.run_dir / 'status.json'
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / 'scratch_experiment.json'
    checkpoint_path = args.run_dir / f'{args.recurrent_type}_seed_{args.seed}.pt'
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(f'{checkpoint_path} exists; use a new directory or --resume')
    config = {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, (range, tuple)) else value
              for key, value in vars(args).items()}
    if args.resume:
        import json
        if not manifest_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError('Resume requires this scratch run’s manifest and checkpoint')
        previous = json.loads(manifest_path.read_text())
        for key in ('seed', 'hidden_size', 'context_length', 'lstm_hidden_size', 'recurrent_type',
                    'train_maps', 'num_envs', 'rollout_steps', 'max_steps'):
            if previous['config'][key] != config[key]:
                raise ValueError(f'Resume configuration differs for {key}')
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved.get('initialization') != 'random' or saved['source_sha256'] is not None:
            raise ValueError('Refusing to resume a checkpoint initialized from pretrained weights')
    source = {'hidden_size': args.hidden_size, 'curriculum': {'stage': 0, 'rewards': []}}
    manifest = {
        'initialization': 'random', 'pretrained_steps': 0, 'training_source_checkpoint': None,
        'config': config, 'torch_version': str(torch.__version__),
        'architecture': 'Deep Sets + one past-only BiLSTM/LSTM + residual projection + MLP',
        'notes': ['All layers train end to end; optimizer starts fresh on the initial run.',
                  'Curriculum starts at stage 0; exact environment-transition budget.',
                  'The comparison checkpoint is used only after training for evaluation.'],
    }
    write_json(manifest_path, manifest)
    print(f'FROM SCRATCH: {args.recurrent_type}, seed {args.seed}, target {args.steps:,} environment transitions', flush=True)
    print(f'Pretrained steps: 0. Training source checkpoint: NONE. Run: {args.run_dir}', flush=True)
    try:
        policy, path = train_variant(source, args, True, args.seed, None)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['global_step'] != args.steps:
            raise RuntimeError('Training did not finish at the requested exact step budget')
        completion = {'initialization': 'random', 'pretrained_steps': 0, 'checkpoint': str(path),
                      'target_steps': args.steps, 'global_step': saved['global_step'],
                      'episodes': saved['episodes'], 'updates': saved['updates'],
                      'stage': saved['curriculum']['stage']}
        if not args.skip_evaluation:
            write_json(args.status_path, {'status': 'evaluating', **completion})
            # No baseline checkpoint has been opened before this point.
            baseline_hash = fingerprint(args.compare_to)
            baseline = load_checkpoint(args.compare_to)
            if baseline.get('architecture') != 'deep_sets_v1' or baseline.get('enemy_sensing') != 'angular_regions_v1':
                raise ValueError('Comparison requires an angular Deep Sets checkpoint')
            manifest['comparison_checkpoint'] = str(args.compare_to)
            manifest['comparison_sha256'] = baseline_hash
            manifest['comparison_training_steps'] = baseline['global_step']
            frozen = ActorCritic(hidden_size=baseline['hidden_size'], bullet_sensing=True, angular_regions=True)
            frozen.load_state_dict(baseline['model'])
            rows = evaluate(frozen, args, 'frozen')
            rows.extend(evaluate(policy, args, args.recurrent_type, args.seed))
            rows.extend(evaluate(policy, args, f'{args.recurrent_type}_current_only', args.seed, current_only=True))
            report_results(rows, args, manifest)
            if fingerprint(args.compare_to) != baseline_hash:
                raise RuntimeError('Comparison checkpoint changed')
            write_json(manifest_path, manifest)
        write_json(args.status_path, {'status': 'complete', **completion,
                                      'finished_at': datetime.now().isoformat(timespec='seconds')})
        print(f'Complete: exactly {saved["global_step"]:,} transitions from scratch. Saved {path}', flush=True)
        return path
    except BaseException as error:
        current = {}
        if args.status_path.exists():
            import json
            current = json.loads(args.status_path.read_text())
        write_json(args.status_path, {**current, 'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                      'error': str(error), 'updated_at': datetime.now().isoformat(timespec='seconds')})
        raise


if __name__ == '__main__':
    main()
