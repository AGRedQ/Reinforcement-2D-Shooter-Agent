"""Scratch (Bi)LSTM + MLP training on CPU/CUDA, using the baseline PPO defaults.

Train all five stages on GPU (stage 4 must also pass the reward/window criterion):
  python training_bilstm_scratch.py --device cuda --no-dashboard
Match the reference's minimum experience, then continue until mastery:
  python training_bilstm_scratch.py --device cuda --min-steps 2501688 --no-dashboard
A bounded smoke run (does NOT claim curriculum completion):
  python training_bilstm_scratch.py --device cpu --steps 2048 --no-dashboard --skip-evaluation
Resume this trainer's own random-initialized run:
  python training_bilstm_scratch.py --device cuda --resume --no-dashboard
Matched scratch feedforward control, SAME engine and settings, only architecture changes:
  python training_bilstm_scratch.py --architecture feedforward --device cuda --no-dashboard
Watch/play: add --watch or --play-against, --model NAME, and optionally --stage N.
Re-evaluate without training: --evaluate-only --model NAME --run-dir reports/comparison.

No pretrained policy initializes training. The reference checkpoint is snapshotted
for later evaluation; its historical hyperparameters may be unknown. Manifests and
reports distinguish identical evaluation from a controlled training comparison.
"""
from __future__ import annotations

from datetime import datetime
import argparse
import copy
import json
from pathlib import Path
import shutil

import torch

from environments import STAGES
from map_generation import parse_map_seeds
from models.curriculum_ppo import policy_batch, train, training_settings
from models.evaluation import evaluate, report_results
from models.ppo import parse_args as shared_parse_args, resolve_model_path
from models.recurrent import ObservationHistory, RecurrentActorCritic
from models.runtime import configuration, fingerprint, load_policy, resolve_device, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / 'checkpoints/angular_deep_sets_ppo.pt'


def _arguments(parser):
    parser.set_defaults(model='__architecture_default__')
    parser.add_argument('--architecture', choices=('feedforward', 'lstm', 'bilstm'), default='bilstm')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N; explicit CUDA fails if unavailable')
    parser.add_argument('--context-length', type=int, default=32)
    parser.add_argument('--lstm-hidden-size', type=int, default=48)
    parser.add_argument('--until-curriculum-complete', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--total-steps', '--steps', dest='total_steps', type=int, default=None, help='hard transition limit; can stop before all stages are complete')
    parser.add_argument('--min-steps', type=int, default=0, help='minimum experience before curriculum-based stopping')
    parser.add_argument('--run-dir', type=Path, help='status, stage reports, reference snapshot and comparison outputs')
    parser.add_argument('--resume', action='store_true', help='explicitly resume this trainer’s own checkpoint')
    parser.add_argument('--train-control', action='store_true', help='after the recurrent run, train a fresh feedforward control for its exact transition budget')
    parser.add_argument('--compare-to', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--eval-maps', type=parse_map_seeds, default='[14,15,16,17]')
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[20000,20001,20002,20003,20004,20005])
    parser.add_argument('--eval-stages', nargs='+', type=int, choices=range(len(STAGES)), default=list(range(len(STAGES))))
    parser.add_argument('--eval-envs', type=int, default=8)


def parse_args(argv=None):
    args = shared_parse_args(argv, description=__doc__, random_maps=True, default_sensing=True, add_arguments=_arguments)
    if args.model == '__architecture_default__':
        args.model = f'{args.architecture}_angular_deep_sets_ppo.pt'
    path = Path(args.model)
    args.checkpoint_path = path if path.is_absolute() else resolve_model_path(args.model)
    if args.checkpoint_path.resolve() == args.compare_to.resolve():
        raise ValueError('Output model must differ from the protected reference checkpoint')
    if args.self_play:
        raise ValueError('Use training_random_map.py for self-play. This architecture comparison keeps the scripted opponent unchanged.')
    if args.train_control and args.architecture == 'feedforward':
        raise ValueError('--train-control requires a recurrent candidate')
    if not args.bullet_sensing or not args.angular_regions:
        raise ValueError('Keep angular regions and bullet sensing enabled for this angular Deep Sets architecture comparison')
    if min(args.context_length,args.lstm_hidden_size,args.minibatch_size,args.epochs,args.eval_envs,args.promotion_window) < 1:
        raise ValueError('Window, layer, minibatch, epoch and evaluation sizes must be positive')
    if args.minibatch_size < 2 or args.min_steps < 0 or (args.total_steps is not None and args.total_steps < 1):
        raise ValueError('Minibatches need >=2 samples, min steps >=0 and total steps >0')
    if args.total_steps is not None and args.total_steps % args.num_envs:
        raise ValueError('Total steps must divide by the environment count for an exact budget')
    if args.total_steps is not None and args.min_steps > args.total_steps:
        raise ValueError('Minimum steps cannot exceed a hard step limit')
    if args.lr <= 0 or args.seed < 0 or not 0 <= args.replay_probability <= 1 or not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError('Invalid optimizer, seed, replay or discount configuration')
    if len(set(args.eval_seeds)) != len(args.eval_seeds) or len(set(args.eval_maps)) != len(args.eval_maps) or len(set(args.eval_stages)) != len(args.eval_stages):
        raise ValueError('Evaluation seeds, maps and stages must be unique')
    if min(args.eval_seeds) < 0 or set(args.map_seeds) & set(args.eval_maps):
        raise ValueError('Use nonnegative evaluation seeds and maps excluded from training')
    args.recurrent_type = args.architecture
    args.run_dir = args.run_dir or args.report_dir / f'{Path(args.model).stem}-{datetime.now():%Y%m%d-%H%M%S}'
    if args.checkpoint_path.resolve() == (args.run_dir/'reference_baseline.pt').resolve():
        raise ValueError('Model output cannot overwrite the reference snapshot')
    return args


def reference_snapshot(args):
    snapshot=args.run_dir/'reference_baseline.pt'
    provenance=args.run_dir/'reference_origin.json'
    if provenance.exists():
        original=json.loads(provenance.read_text())
        if original['source_path'] != str(args.compare_to.resolve()):
            raise ValueError('This run directory contains another reference. Choose a new --run-dir for a different --compare-to.')
        if fingerprint(snapshot) != original['snapshot_sha256']:
            raise ValueError('Reference snapshot fingerprint differs from its provenance record')
    elif not snapshot.exists():
        shutil.copy2(args.compare_to,snapshot)
    if not provenance.exists():
        write_json(provenance,{'source_path':str(args.compare_to.resolve()),'snapshot_sha256':fingerprint(snapshot),
                               'created_at':datetime.now().isoformat(timespec='seconds')})
    return snapshot


def watch(args):
    from renderer import Renderer
    device = resolve_device(args.device)
    policy, saved = load_policy(args.checkpoint_path, device)
    if not policy.bullet_sensing or not policy.angular_regions or 'self_play' in saved:
        raise ValueError('Use training_random_map.py for legacy sensing or self-play checkpoints')
    policy.eval()
    stage = args.stage if args.stage is not None else saved['curriculum']['stage']
    env = Renderer(max_steps=args.max_steps, stage=stage, replay_probability=0.0, fps=args.fps,
                   shoot_back=True, map_seeds=args.map_seeds, bullet_sensing=True, angular_regions=True,
                   human_opponent=args.play_against, player_hacks=args.player_hacks)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed if episode == 0 else None)
            histories = ObservationHistory([obs], policy.context_length) if isinstance(policy,RecurrentActorCritic) else None
            done = False
            while not done and env.window_open:
                with torch.no_grad():
                    action,_,_ = policy.act(policy_batch(policy, histories.snapshot() if histories else [obs], device), deterministic=not args.stochastic)
                obs,_,terminated,truncated,info = env.step(action[0].cpu().numpy())
                done = terminated or truncated
                if histories:
                    histories.advance([obs],[done])
            if not env.window_open:
                break
            print(f'episode {episode}: {info["episode"]}')
    finally:
        env.close()


def compare(args, policy=None, training_result=None):
    device = resolve_device(args.device)
    if policy is None:
        policy, saved = load_policy(args.checkpoint_path, device)
    else:
        saved = torch.load(args.checkpoint_path,map_location='cpu',weights_only=True)
    args.recurrent_type = ('bilstm' if policy.bidirectional else 'lstm') if isinstance(policy,RecurrentActorCritic) else 'feedforward'
    if not policy.bullet_sensing or not policy.angular_regions or 'self_play' in saved:
        raise ValueError('Candidate must use angular Deep Sets scripted-combat observations')
    trained_maps=saved.get('training_config',{}).get('map_seeds',saved.get('config',{}).get('train_maps'))
    if trained_maps is not None and set(trained_maps) & set(args.eval_maps):
        raise ValueError('Evaluation maps overlap the checkpoint’s actual recorded training maps')
    reference_path = reference_snapshot(args)
    reference_hash = fingerprint(reference_path)
    reference, ref_saved = load_policy(reference_path,device)
    if not reference.bullet_sensing or not reference.angular_regions or 'self_play' in ref_saved:
        raise ValueError('The reference must be a scripted-opponent angular Deep Sets checkpoint')
    actual = saved.get('training_config',{})
    known = ref_saved.get('training_config')
    differences = {key: {'reference':known.get(key),'candidate':value} for key,value in actual.items()
                   if known.get(key) != value} if known is not None else None
    manifest = {
        'candidate_checkpoint':str(args.checkpoint_path),'candidate_sha256':fingerprint(args.checkpoint_path),
        'candidate_steps':saved['global_step'],'candidate_curriculum':saved['curriculum']['stage'],
        'candidate_curriculum_complete':saved.get('curriculum_complete',False),
        'reference_checkpoint':str(args.compare_to),'reference_snapshot':str(reference_path),
        'reference_sha256':reference_hash,'reference_steps':ref_saved['global_step'],
        'reference_current_sha256':fingerprint(args.compare_to) if args.compare_to.exists() else None,
        'reference_curriculum':ref_saved['curriculum']['stage'], 'training_result':training_result,
        'fairness_audit':{
            'same_evaluation_conditions':True, 'reference_training_config_recorded':known is not None,
            'training_config_differences':differences,
            'equal_transition_budget':saved['global_step']==ref_saved['global_step'],
            'same_random_initialization':saved.get('initialization') == ref_saved.get('initialization') == 'random',
            'same_training_device':saved.get('training_device') == ref_saved.get('training_device'),
            'controlled_architecture_only_claim':known is not None and not differences
                and saved['global_step']==ref_saved['global_step']
                and saved.get('initialization') == ref_saved.get('initialization') == 'random'
                and saved.get('training_device') == ref_saved.get('training_device'),
            'note':'Missing historical configuration cannot be reconstructed from model weights. Use a fresh feedforward control with the same saved settings and budget for an architecture-only study.',
        },
        'candidate_training_config':actual,'reference_training_config':known,'evaluation_config':{
            'maps':list(args.eval_maps),'seeds':args.eval_seeds,'stages':args.eval_stages,
            'max_steps':args.max_steps,'deterministic':True,'replay_probability':0.0,'exploration_bonus':False},
    }
    write_json(args.run_dir/'comparison_manifest.json',manifest)
    rows=evaluate(reference,args,'frozen')
    candidate_seed=actual.get('seed',saved.get('train_seed',args.seed))
    rows.extend(evaluate(policy,args,args.recurrent_type,candidate_seed))
    if isinstance(policy,RecurrentActorCritic):
        rows.extend(evaluate(policy,args,f'{args.recurrent_type}_current_only',candidate_seed,current_only=True))
    result=report_results(rows,args,manifest)
    if fingerprint(reference_path)!=reference_hash:
        raise RuntimeError('Reference snapshot changed during evaluation')
    return result


def main(argv=None):
    args=parse_args(argv)
    torch.set_num_threads(1)
    if args.watch or args.play_against:
        return watch(args)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.evaluate_only:
        return compare(args)
    if args.checkpoint_path.exists() and not args.resume:
        raise FileExistsError(f'{args.checkpoint_path} already exists. Choose a fresh --model or explicitly --resume.')
    snapshot=reference_snapshot(args) if not args.skip_evaluation else args.run_dir/'reference_baseline.pt'
    baseline_hash=fingerprint(snapshot) if snapshot.exists() else None
    write_json(args.run_dir/'experiment.json',{
        'initialization':'random','pretrained_steps':0,'training_source_checkpoint':None,
        'architecture':args.architecture,'training_config':training_settings(args),'config':configuration(args),
        'reference_sha256':baseline_hash,'reference_used_for_initialization':False,
    })
    if args.train_control:
        write_json(args.run_dir/'pipeline_status.json',{'phase':'candidate_training',
                   'candidate_checkpoint':str(args.checkpoint_path),
                   'historical_report':str(args.run_dir/'comparison.html'),
                   'matched_report':str(args.run_dir/'matched_comparison/comparison.html')})
    policy,outcome=train(args)
    if outcome['reason'] in ('keyboard interrupt','stop request'):
        if args.train_control:
            write_json(args.run_dir/'pipeline_status.json',{'phase':'candidate_stopped','candidate_result':outcome})
        return args.checkpoint_path
    control_path=None
    if args.train_control:
        control=copy.deepcopy(args)
        control.architecture=control.recurrent_type='feedforward'
        control_path=args.checkpoint_path.with_name(f'{args.checkpoint_path.stem}_feedforward_control.pt')
        control.checkpoint_path=control_path
        control.model=str(control_path)
        control.run_dir=args.run_dir/'feedforward_control'
        control.run_dir.mkdir(parents=True,exist_ok=True)
        control.total_steps=outcome['global_step']
        control.total_episodes=None
        control.min_steps=0
        control.until_curriculum_complete=False
        control.resume=args.resume and control_path.exists()
        write_json(args.run_dir/'pipeline_status.json',{'phase':'feedforward_control','candidate_steps':outcome['global_step'],
                   'control_target_steps':control.total_steps,'control_checkpoint':str(control_path)})
        _,control_outcome=train(control)
        if control_outcome['reason'] in ('keyboard interrupt','stop request'):
            write_json(args.run_dir/'pipeline_status.json',{'phase':'control_stopped','control_result':control_outcome})
            return args.checkpoint_path
    if not args.skip_evaluation:
        write_json(args.run_dir/'comparison_status.json',{'status':'evaluating','checkpoint':str(args.checkpoint_path)})
        compare(args,policy,outcome)
        if control_path is not None:
            matched=copy.deepcopy(args)
            matched.compare_to=control_path
            matched.run_dir=args.run_dir/'matched_comparison'
            matched.run_dir.mkdir(parents=True,exist_ok=True)
            compare(matched,policy,outcome)
        write_json(args.run_dir/'comparison_status.json',{'status':'complete','report':str(args.run_dir/'comparison.html')})
    if args.train_control:
        write_json(args.run_dir/'pipeline_status.json',{'phase':'complete','candidate_checkpoint':str(args.checkpoint_path),
                   'control_checkpoint':str(control_path),'matched_report':str(args.run_dir/'matched_comparison/comparison.html')})
    return args.checkpoint_path


if __name__=='__main__':
    main()
