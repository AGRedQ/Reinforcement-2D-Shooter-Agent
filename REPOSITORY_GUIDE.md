# Reinforcement 2D Shooter Agent: training and comparison guide

This repository trains an agent to move, aim, shoot, and survive in an 800 × 800 arena. Walls block bodies, bullets, and sensing. Policies receive numerical observations rather than rendered images. PPO learns from rewards gathered in parallel environments.

There are two entry points:

| Entry point | Purpose |
| --- | --- |
| [training_random_map.py](training_random_map.py) | Existing feedforward trainer; scripted combat, optional exploration reward, and self-play |
| [training_bilstm_scratch.py](training_bilstm_scratch.py) | GPU-capable scratch BiLSTM/LSTM training, a matched scratch feedforward control, all-stage completion, watch/play, and automatic evaluation |

The recurrent trainer never initializes from the reference checkpoint. All network weights and optimizer state start fresh. Its `--resume` option loads only a previously saved random-initialized run from this trainer with a matching recorded configuration.

The previous standalone warm-start experiment has been removed. Reusable training, device handling, and evaluation now live under `models/`.

## Contents

1. [Files and architecture](#1-files-and-architecture)
2. [Local setup and GPU access](#2-local-setup-and-gpu-access)
3. [Training settings and step terminology](#3-training-settings-and-step-terminology)
4. [Start training and complete all stages](#4-start-training-and-complete-all-stages)
5. [Checkpoints, progress, stopping, and resume](#5-checkpoints-progress-stopping-and-resume)
6. [Fair architecture comparisons](#6-fair-architecture-comparisons)
7. [Evaluation, reports, and interpretation](#7-evaluation-reports-and-interpretation)
8. [Watch, play, and rendering](#8-watch-play-and-rendering)
9. [Environment mechanics and observations](#9-environment-mechanics-and-observations)
10. [Exploration rewards and self-play](#10-exploration-rewards-and-self-play)
11. [Kaggle setup](#11-kaggle-setup)
12. [Tests and troubleshooting](#12-tests-and-troubleshooting)

## 1. Files and architecture

| File | Responsibility |
| --- | --- |
| [entities.py](entities.py) | Agent/opponent bodies, projectiles, collisions, actions, angular sensing, bullet features |
| [environments.py](environments.py) | Scripted combat environment, rewards, spawning, curriculum, optional exploration reward |
| [map_generation.py](map_generation.py) | Seeded obstacle generation and map-seed parsing |
| [models/mlp.py](models/mlp.py) | Feedforward actor-critic and Deep Sets bullet encoder |
| [models/recurrent.py](models/recurrent.py) | Episode-isolated observation windows and the one-layer LSTM/BiLSTM policy |
| [models/ppo.py](models/ppo.py) | Existing trainer, shared PPO loss/GAE, environment workers, common argument defaults |
| [models/curriculum_ppo.py](models/curriculum_ppo.py) | Shared CPU/CUDA scratch training engine for recurrent and feedforward control policies |
| [models/runtime.py](models/runtime.py) | Device transfers, portable checkpoint loading, hashes, atomic JSON files |
| [models/evaluation.py](models/evaluation.py) | Fixed-trial evaluation, history ablation, paired intervals, HTML/CSV/JSON comparison |
| [self_play.py](self_play.py) | Symmetric duels for the original trainer |
| [renderer.py](renderer.py) | Pygame arena, manual controls, player-versus-agent view |
| [visibility.py](visibility.py) | Wall-clipped geometry for optional sight shading |
| [dashboard.py](dashboard.py), [dashboard_app.py](dashboard_app.py) | Local Streamlit/Plotly dashboard, metric history, stop control |
| [training_report.py](training_report.py) | Per-run CSV samples and offline HTML training reports |
| [tests/](tests/) | Physics, sensing, PPO integration, CUDA, completion criteria, reporting, rendering, and self-play checks |

### Feedforward baseline

The angular Deep Sets policy uses 64 state values and a 32-value summary of currently visible hostile bullets:

```text
Angular state (64) + current bullet summary (32)
    → Linear(96, 256) → Tanh
    → Linear(256, 256) → Tanh
        ├─ movement head: 9 logits
        ├─ aim head:      3 logits
        ├─ fire head:    2 logits
        └─ value head:   1 scalar
```

The actor samples the three action distributions during training. The critic estimates discounted future reward. Watch/evaluation use each head's argmax unless explicitly configured otherwise.

### One BiLSTM before the decision MLP

The recurrent architecture keeps the bullet encoder and decision MLP, adding a temporal branch:

```text
Each past observation: state (64) + bullet summary (32)
    → window of up to 32 × 96 features, oldest first
    → ONE BiLSTM layer, 48 hidden units per direction
    → concatenate final forward/backward hidden states (96)
    → learned Linear(96, 96) projection
    → add current observation's 96 features
    → the same 256 → 256 decision MLP and output heads
```

The projection starts at zero. This is an additive memory architecture: it can initially use current features, then learn temporal corrections. On scratch runs, no pretrained encoder, MLP, head, or recurrent weights are loaded. All layers learn jointly through PPO. On the first optimizer step the zero projection blocks gradients into the LSTM; subsequent steps train it as the projection changes.

Both LSTM directions see only observations already available at the action time. There is no future-state lookahead. Both final hidden states summarize the full available window; using only the final output timestep would not give the same backward summary.

History continues across rollouts and clears at episode resets. Each PPO minibatch recomputes its complete raw windows, so gradients reach past bullet encodings and recurrent computations without reusing stale hidden states. Time padding and bullet padding are masked independently. The default 32 observations span about 1.03 seconds between the oldest and newest sample at 30 Hz. This is bounded temporal context, not unlimited memory.

Default parameter counts with 256-wide MLP and 48-unit directions are 97,807 for the angular feedforward policy and 163,183 for the BiLSTM variant.

## 2. Local setup and GPU access

Run commands from the repository root. Install dependencies in your Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you already have a working PyTorch/CUDA environment, use it rather than replacing its PyTorch build unnecessarily. Check actual GPU visibility:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No accessible CUDA GPU')"
```

In the inspected local setup, a GTX 1650 with 4 GB VRAM was accessible outside the agent's sandbox, and a CUDA LSTM forward/backward check succeeded. A sandboxed process may report no GPU even when the normal terminal can access it. Run the trainer in a normal GPU-enabled terminal, or grant the coding agent permission to run GPU commands outside that sandbox.

The new trainer supports:

| Setting | Behavior |
| --- | --- |
| `--device cuda` | Require a CUDA GPU; fail clearly if unavailable |
| `--device cuda:0` | Select a specific CUDA device |
| `--device cpu` | Force CPU execution |
| `--device auto` | Use CUDA when accessible, otherwise CPU; default |

Policy inference, PPO losses, gradient computation, optimizer updates, and rollout value/action tensors use the selected device. The game simulation remains NumPy/CPU work in spawned worker processes. Worker actions are copied to CPU; observations are moved to the learner's device. Environment workers do not initialize GPU policies.

Packed LSTM lengths are copied to CPU for PyTorch's packing operation, while features and model parameters remain on GPU. Checkpoints store CPU tensors for portability. See the official [PyTorch packed-sequence documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) and [CUDA multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).

The original `training_random_map.py` remains CPU-based. For a CUDA feedforward control that shares exactly the recurrent engine, use `training_bilstm_scratch.py --architecture feedforward --device cuda`.

## 3. Training settings and step terminology

The new trainer reads the same common defaults as `training_random_map.py`. Its feedforward control and recurrent policy run through one scratch training engine and the same PPO/GAE implementations.

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-envs` | 8 | Parallel game environments |
| `--rollout-steps` | 256 | Transitions collected per environment before learning |
| `--minibatch-size` | 512 | Transitions per optimizer update |
| `--epochs` | 4 | Passes through one collected rollout |
| `--lr` | 0.0003 | Adam learning rate |
| Adam epsilon | 0.00001 | Shared optimizer setting |
| `--gamma` | 0.99 | Reward discount |
| `--gae-lambda` | 0.95 | GAE weighting |
| `--clip` | 0.2 | PPO probability-ratio clipping |
| `--ent-coef` | 0.01 | Entropy bonus coefficient |
| `--vf-coef` | 0.5 | Value-loss coefficient |
| `--max-grad-norm` | 0.5 | Gradient norm limit |
| `--hidden-size` | 256 | Decision MLP width |
| `--max-steps` | 1000 | Maximum steps per episode |
| `--promotion-reward` | 35 | Mean reward required to pass a stage |
| `--promotion-window` | 100 | Qualifying completed episodes in a stage window |
| `--replay-probability` | 0.2 | Earlier-stage practice probability |
| `--map-seeds` | 9 | Cycle maps 0 through 9 inclusive |
| `--seed` | 0 | Policy/action and initial spawn seed |
| `--save-every` | 25 | PPO updates between checkpoint saves |
| Angular/bullet sensing | Enabled | Same 64-value state and bullet observations as the angular reference |
| Exploration reward | Disabled | Enable explicitly with `--exploration-bonus`; changes the training reward |
| `--context-length` | 32 | Maximum observations in a recurrent window |
| `--lstm-hidden-size` | 48 | Hidden units per LSTM direction |

### Collection versus learning

An **observation** is the policy input at a point in time. An **environment step** advances one environment after an action and produces a new observation and reward. A **transition** records that experience.

At the defaults:

```text
8 environments × 256 rollout steps = 2,048 new transitions
2,048 / 512 = 4 minibatches per epoch
4 minibatches × 4 epochs = 16 optimizer updates per PPO cycle
```

A **PPO update** is the entire learning cycle over the collected batch. Each minibatch performs zero-grad, backpropagation, gradient clipping, and one Adam step. The checkpoint `updates` counter counts PPO cycles, not individual Adam calls.

An **episode** ends at death or its 1,000-step time limit. It can cross several rollout boundaries. A rollout can also include multiple short episodes. Rollout boundaries do not reset the game or the BiLSTM's raw observation window. Overlapping windows do not multiply the environment-step count by 32.

PPO uses clipped policy loss, critic return error, and entropy:

```text
loss = policy_loss + 0.5 × value_loss − 0.01 × entropy
value_loss = 0.5 × mean((predicted_value − estimated_return)²)
```

Advantages are normalized per minibatch. GAE stops at episode boundaries. A time-limit truncation bootstraps from the final observation using that episode's history; death does not bootstrap. The code logs approximate KL but does not implement target-KL early stopping.

## 4. Start training and complete all stages

### Train the BiLSTM from scratch on GPU

Choose a new model name to avoid overwriting an existing run:

```bash
python -u training_bilstm_scratch.py \
  --model bilstm_full_seed0 \
  --device cuda \
  --run-dir reports/bilstm_full_seed0 \
  --no-dashboard
```

This starts at stage 0 with fresh parameters and Adam state. `--until-curriculum-complete` is enabled by default. There is no default hard transition ceiling, so it does not stop at two million steps while still on an earlier stage.

Add `--train-control` to automatically train a fresh feedforward policy after the recurrent run. The control receives the recurrent run’s exact final transition count with the same scientific settings, seed, and device. The trainer writes both the historical-reference comparison and `matched_comparison/comparison.html` against the control. Its checkpoint is named `*_feedforward_control.pt`; its progress and reports live in the run’s `feedforward_control/` directory. This costs an additional training run. `pipeline_status.json` tracks the phase. A control can hit its fixed budget before mastering every stage; that is an experimental outcome.

To require at least the current reference checkpoint's 2,501,688 environment steps before the curriculum stop is allowed:

```bash
python -u training_bilstm_scratch.py \
  --model bilstm_full_seed0 \
  --device cuda \
  --min-steps 2501688 \
  --train-control \
  --run-dir reports/bilstm_full_seed0 \
  --no-dashboard
```

Use the minimum-step setting consistently for a fresh control. The reference's step count is historical metadata, not a reconstruction of its original seed/maps/hyperparameters.

### What counts as completing every stage?

| Stage | Opponent speed | Requested spawn range | Main spawning/behavior rule |
| --- | --- | --- | --- |
| 0 | 0 | 80–250 | ±10 degrees; line of sight required |
| 1 | 60 | 100–350 | ±30 degrees; line of sight required |
| 2 | 120 | 150–500 | ±90 degrees; may be occluded |
| 3 | 180 | 200–700 | Any direction |
| 4 | 180 | 250–900 | Prefer hidden positions and evade aim |

Spawn constraints can be relaxed if the sampler cannot satisfy them. Even a stationary stage-0 combat opponent can shoot.

Stages 0–3 promote after their latest 100 qualifying episodes average at least 35 reward. Earlier-stage replay and episodes from a different stage do not count. Promotions clear the curriculum's stage reward window.

The final stage has no ordinary promotion. The new trainer maintains a separate stage-4 window and requires **100 non-replay stage-4 episodes with mean reward at least 35**. Merely reaching stage 4 is not completion. The minimum-step target, if set, must also be met. The criterion uses recent training episodes; final independent evaluation still matters.

A weak policy can take many steps or fail to meet this threshold. The trainer does not silently lower the threshold, skip stages, or call a budget-limited run complete.

### Optional hard limits

```bash
# Exactly two million transitions, even if mastery is not reached.
python -u training_bilstm_scratch.py --model bilstm_bounded \
  --device cuda --steps 2000000 --no-dashboard \
  --run-dir reports/bilstm_bounded

# Small CPU smoke run, not a learning-quality experiment.
python training_bilstm_scratch.py --model bilstm_smoke \
  --device cpu --steps 2048 --no-dashboard --skip-evaluation \
  --run-dir reports/bilstm_smoke
```

`--steps` aliases `--total-steps`. It is a hard limit; the final rollout is shortened so the target is exact. The target must divide by the environment count. A hard limit or episode limit can stop before mastery; the saved status and checkpoint explicitly record `curriculum_complete: false` in that case.

`--total-episodes N` is a cumulative episode target. Graceful stopping waits until every worker has finished an episode after the request, so episode totals can overshoot. All transitions collected in the new engine's final partial rollout are optimized.

## 5. Checkpoints, progress, stopping, and resume

For a relative `--model NAME`, checkpoints are stored under `checkpoints/`; `.pt` is appended when needed. Absolute `.pt` paths are also accepted by the new trainer. Its defaults are:

| Architecture | Default new-trainer model |
| --- | --- |
| `bilstm` | `bilstm_angular_deep_sets_ppo.pt` |
| `lstm` | `lstm_angular_deep_sets_ppo.pt` |
| `feedforward` | `feedforward_angular_deep_sets_ppo.pt` |

The new trainer requires explicit `--resume` if the output checkpoint already exists. It refuses to use the comparison checkpoint as its output. It also refuses to resume pretrained or legacy unaudited runs as if they were a new controlled scratch experiment. Old model files remain loadable for inference/evaluation through `models/runtime.py`.

Checkpoint contents include:

- Model and Adam state, observation/action dimensions, architecture and recurrent settings.
- Curriculum stage and reward window, final-stage qualification window, stage history, and completed-stage list.
- Step, episode, and PPO-update counters.
- `initialization: random`, no pretrained source hash, and the full effective training configuration.
- The actual learner device used.

Checkpoints are written atomically, initially, every 25 updates by default, and during cleanup. Passing a stage also saves an `*_after_stage_N.pt` snapshot. A snapshot marks the completed stage and current curriculum state; it is not an independently evaluated best policy.

The run directory contains:

```text
reports/bilstm_full_seed0/
    experiment.json                  # configuration and random initialization
    status.json                      # current progress/device/stage qualification
    reference_baseline.pt            # immutable reference snapshot for this run
    <unique training-report folder>/
        metrics.csv
        report.html
    comparison_manifest.json         # final provenance and fairness audit
    evaluation.csv
    comparison.json
    comparison.html
    comparison_status.json
```

The comparison files are created after training, or by `--evaluate-only`. Checkpoint weights normally live under `checkpoints/`, not inside the report directory. Preserve both the checkpoint and its report/configuration files when moving a run.

Monitor from another terminal:

```bash
cat reports/bilstm_full_seed0/status.json
nvidia-smi
```

For a logged foreground run, redirect output yourself, or use `tee` with shell pipe failure handling. Example:

```bash
set -o pipefail
python -u training_bilstm_scratch.py --model bilstm_full_seed0 \
  --device cuda --min-steps 2501688 --train-control --no-dashboard \
  --run-dir reports/bilstm_full_seed0 2>&1 | tee bilstm_full_seed0.log
```

Stop using Ctrl+C, the dashboard's stop button, or a run-local request file:

```bash
touch reports/bilstm_full_seed0/STOP
```

The request file and dashboard stop wait for episodes to finish and save. Ctrl+C saves during cleanup. Remove your `STOP` file before resuming. Resume with the same scientific settings:

```bash
python -u training_bilstm_scratch.py --model bilstm_full_seed0 \
  --device cuda --min-steps 2501688 --train-control --no-dashboard --resume \
  --run-dir reports/bilstm_full_seed0
```

Resume restores weights, Adam state, curriculum, and final-stage progress. It rejects changes to recorded PPO, seed, sensing, map, reward, or environment settings, and rejects architecture changes. Device changes are possible for portability and are recorded. Live environment states and full RNG trajectories are not restored, so resume is not bit-for-bit uninterrupted simulation.

## 6. Fair architecture comparisons

There are two distinct comparisons.

### A. Benchmark against the supplied historical checkpoint

The default reference is `checkpoints/angular_deep_sets_ppo.pt`. The inspected version has 2,501,688 steps and curriculum stage 4. It records dimensions, weights, optimizer state and progress, but not the complete original training configuration or proof that stage 4 passed the new final-stage criterion.

The new trainer snapshots this checkpoint before training and never loads its weights into the learner. Evaluation uses that snapshot, preventing later updates to the original file from silently changing the reference.

The new PPO/environment defaults match the current baseline trainer. However, **matching today's defaults does not establish what settings produced a historical checkpoint**. The final report therefore records unknown reference configuration rather than claiming an architecture-only experiment. A different total training budget is also reported explicitly.

### B. Controlled scratch feedforward versus BiLSTM

Use the new shared engine for both models. Keep seed, maps, rewards, optimizer, batch sizes, PPO coefficients, hardware, and transition budget identical. Change only architecture and its necessary temporal-layer settings.

For a fixed-budget experiment:

```bash
# Control: random-initialized feedforward angular Deep Sets policy.
python -u training_bilstm_scratch.py --architecture feedforward \
  --model ff_budget4m_seed0 --device cuda --seed 0 \
  --steps 4000000 --no-until-curriculum-complete --no-dashboard \
  --run-dir reports/ff_budget4m_seed0 --skip-evaluation

# Candidate: same settings, with the BiLSTM architecture.
python -u training_bilstm_scratch.py --architecture bilstm \
  --model bi_budget4m_seed0 --device cuda --seed 0 \
  --steps 4000000 --no-until-curriculum-complete --no-dashboard \
  --run-dir reports/bi_budget4m_seed0 --skip-evaluation

# Compare the two trained policies on identical held-out trials.
python training_bilstm_scratch.py --evaluate-only \
  --model bi_budget4m_seed0 --device cuda \
  --compare-to checkpoints/ff_budget4m_seed0.pt \
  --run-dir reports/matched_budget4m_seed0
```

Four million is an example budget, not a promise of mastery. If a policy is still on stage 2 at the cap, report that outcome rather than overriding its stage or changing its reward threshold. For an all-stage sample-efficiency experiment, let both run until mastery and compare steps required; training budgets will then differ by design.

Repeat with seeds 1, 2, and more independent seeds before drawing general conclusions. Evaluate held-out maps rather than tuning repeatedly on the same test set. A fresh control checkpoint records the same audit fields as the candidate; the report checks configuration, random initialization, actual device, and transition-budget equality before permitting an architecture-only claim.

Adding recurrence increases parameter count and computation. History ablation helps assess temporal dependence, but it does not by itself eliminate every capacity or optimization explanation.

## 7. Evaluation, reports, and interpretation

By default, after an ordinary training stop, evaluation compares:

1. The frozen reference snapshot.
2. The trained candidate.
3. For a recurrent candidate, the same weights with only the current observation instead of history.

The default evaluation uses all stages 0–4, maps 14–17, six spawn seeds 20000–20005, 1,000-step limits, deterministic argmax actions, no curriculum replay, and no exploration reward. Each policy receives the same map/stage/seed trials. Their subsequent trajectories naturally differ because their actions differ. Stage 4 is the default primary stage because it is last in the configured evaluation list.

Example custom evaluation:

```bash
python training_bilstm_scratch.py --evaluate-only \
  --model bilstm_full_seed0 --device cuda \
  --compare-to checkpoints/angular_deep_sets_ppo.pt \
  --eval-stages 2 4 --eval-maps '[14,15,16,17]' \
  --eval-seeds 20000 20001 20002 20003 20004 20005 \
  --run-dir reports/bilstm_custom_comparison
```

Training and evaluation maps must be disjoint. Earlier historical reference maps may be unknown; that limitation is included in the report. For evaluation-only use, retain the original training-map selection through `--map-seeds` when it differs from 0–9.

| Metric | Interpretation |
| --- | --- |
| Mean reward | Combat objective including shot/damage/death penalties; evaluation disables exploration shaping |
| Mean kills | Offensive performance per episode |
| Kills per 1,000 steps | Offensive rate accounting for different episode lengths |
| Hit rate | Total hits / total shots |
| Survival / death rate | Fraction surviving / dying in an episode |
| Combat success | At least one kill AND survival to the time limit |
| Damage taken | Health lost per episode |
| Detection delay | Delay to first detection of spawned opponents |

There is no native victory condition in scripted hunting: killed opponents respawn. Combat success is a reporting definition, not a new reward or environment termination rule.

Paired bootstrap intervals describe differences on sampled map/spawn trials. With one trained seed, they do not quantify variability across independent training runs. The report separates the reference success comparison from the history-ablation result: memory can matter even when the recurrent policy has lower reward than the baseline. Removing history also changes the input distribution, so avoid treating that ablation as a complete proof of superior memory.

The training dashboard and HTML training report are separate from final evaluation. Training metrics average up to the last 100 completed episodes across workers, including replay. CSV samples flush every 10 completed episodes plus a final partial sample. PPO statistics describe the latest completed update. The offline report includes configuration and stop reason.

A stage promotion changes difficulty and can lower training reward. A lower death rate can coexist with fewer kills or lower accuracy. Compare all metrics at fixed evaluation difficulty instead of choosing a single favorable score.

## 8. Watch, play, and rendering

The new trainer supports its recurrent checkpoints directly:

```bash
python training_bilstm_scratch.py --watch --model bilstm_full_seed0 \
  --device cuda --stage 4 --map-seeds '[14]' --episodes 5

python training_bilstm_scratch.py --play-against --model bilstm_full_seed0 \
  --device cuda --map-seeds '[4]' --episodes 5
```

These modes perform inference only and do not save policy updates. `--stochastic` samples actions instead of argmax. `--stage` chooses watch/play difficulty; it does not force the training start stage. Training starts at stage 0 or resumes its saved stage.

The original trainer watches existing feedforward/self-play checkpoints:

```bash
python training_random_map.py --watch --model angular_deep_sets_ppo \
  --map-seeds '[14]' --stage 4
```

Manual arena play and screenshots:

```bash
python renderer.py --shoot-back --map-seeds '[5]' --visibility
python renderer.py --headless --visibility --seed 7 \
  --screenshot /tmp/shooter-sight/seed-7.png
python render_preview.py --output-dir /tmp/shooter-sight
```

Normal view: WASD moves relative to aim during manual play, Q/E or arrows turn, Space fires, R toggles sensors, V sight shading, T trails, F FPS cap, and Escape closes the view. These movement keys do not override a trained agent during watch or rendered training.

Player-versus-agent mode: WASD moves your opponent in screen directions, mouse aims, left mouse fires, and H reveals hidden objects and agent rays. `--player-hacks` starts enabled. Display overrides never change AI inputs. R/V/T are disabled in that mode.

`--render` draws one training environment locally. Closing its arena leaves simulation running; use the stop control to end training. Continuous shading is optional and off by default; it is display geometry, not extra policy information. Angular fills follow the R sensor toggle.

## 9. Environment mechanics and observations

Actions are `MultiDiscrete([9,3,2])`: nine relative movement choices, three aim choices, and fire/hold. Diagonal movement is normalized. Turning happens before relative movement. Agent maximum speed is 220 units/second and turn speed is 200 degrees/second; velocity approaches the requested velocity gradually.

Agents and opponents have 100 health. Agent bullets deal 25 damage, travel at 900 units/second, have 900-unit range, and fire at most every 0.35 seconds. Scripted shooting opponents use 25-damage bullets at 450 units/second, 600-unit range, and a 0.8-second cooldown. A full-health opponent requires four hits.

Bullets use swept collisions, stopping at the first wall/target or their remaining range. Bodies slide along walls. There is no character-to-character collision resolution.

### Rewards

| Event | Default reward |
| --- | --- |
| Agent hit | +1 |
| Killing hit | Additional +3 |
| Shot created with no prior enemy detection | −0.1 |
| Health lost | −0.04 per point actually lost |
| Agent death | Additional −3 |

One full-health kill earns 7 points before penalties. Losing 25 health costs 1 point; losing all 100 costs 4 points plus 3 for death. A shot that misses a detected enemy has no separate miss penalty. Requesting fire during cooldown creates no bullet and has no wasted-shot penalty. Detection for the firing penalty is from before the current action.

Kills respawn opponents and clear remaining friendly bullets. Hostile bullets already in flight survive opponent respawns. Agent death terminates an episode; the time limit truncates it. If both happen together, death takes precedence for bootstrapping.

### Angular state layout

| Indices | Values |
| --- | --- |
| 0–31 | 32 wall-ray distances / 600 |
| 32–37 | Six enemy-region flags |
| 38–43 | Six enemy-region distances / 600 |
| 44–45 | Center-ray detection flag and normalized distance |
| 46–47 | Local forward/sideways velocity / maximum speed |
| 48 | Normalized firing cooldown |
| 49 | Normalized health |
| 50–63 | Previous movement/aim/fire as three one-hot groups |

Enemy regions cover boundaries −30, −15, −5, 0, 5, 15, 30 degrees and consider target silhouette and wall occlusion. A separate center ray supplies alignment information; it is not a firing gate.

Visible hostile bullets are a variable-length `(N,4)` array of local position and relative velocity. Positions are divided by 600; velocity components are divided by projectile speed plus agent maximum speed. Only live bullets within the 60-degree cone, within range, and unblocked by walls are observed. Friendly bullets are excluded.

Deep Sets applies a shared 4→32→32 network to each bullet, masked mean/max pooling plus bounded count `N/(N+1)`, then a 65→32 projection. Empty sets yield a zero summary. Padding ignores absent bullets without imposing a maximum count. Pooling is over bullets at one timestep; the LSTM supplies time context.

The older legacy-ray observation has 60 values with five target rays that can miss targets between rays. It remains supported in the original trainer, using explicit sensing switches, but the new comparison trainer keeps angular and bullet sensing enabled for both architectures.

Neither policy receives absolute coordinates, a full map, stage number, remaining time, or explicit target velocity. Visibility and finite memory still limit what can be inferred.

## 10. Exploration rewards and self-play

These are separate changes to the learning task. Do not enable them on only one side of a memory comparison.

Original trainer examples:

```bash
# A separate feedforward experiment; do not overwrite your reference model.
python training_random_map.py --model exploration_seed0 \
  --exploration-bonus --map-seeds 9 --no-dashboard

# Symmetric duel experiment with frozen-policy opponents.
python training_random_map.py --self-play --model self_play_new \
  --self-play-init angular_deep_sets_ppo --map-seeds '[4]' --no-dashboard

python training_random_map.py --self-play --watch --model self_play_new \
  --map-seeds '[4]' --stochastic
```

The optional exploration reward pays new maximum displacement from an invisible anchor while searching, at 0.0005 per unit up to 100 units per anchor, capped at +0.5 per episode. Anchors reset at spawn, kills, or after five simulated seconds. This rewards travel, not map novelty; revisiting routes can pay after a reset. It adds no observation inputs.

Self-play uses two identical Agent bodies. A frozen opponent samples actions and refreshes from the learner every 50 PPO updates by default. Only the learner supplies training data. First death ends the round; simultaneous death and timeouts are terminal draws, with no time-limit continuation bootstrap. New self-play runs initialize from the selected angular Deep Sets source under a separate output model; the source is preserved. Curriculum and replay are disabled.

The recurrent comparison entry point deliberately uses scripted combat, because self-play changes the opponent/task. Use the original entry point for self-play rather than silently comparing a duel policy with a scripted-combat reference.

## 11. Kaggle setup

Ensure your current branch includes the new files before cloning. With Internet enabled in a notebook:

```python
!git clone --branch quocanh --single-branch https://github.com/AGRedQ/Reinforcement-2D-Shooter-Agent.git /kaggle/working/shooter
%cd /kaggle/working/shooter
%pip install -q gymnasium pygame
```

Kaggle's existing PyTorch build may already support its selected GPU. Check it before replacing PyTorch. Select an available GPU accelerator and run the new trainer as a separate process so multiprocessing works:

```python
!SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -u training_bilstm_scratch.py \
  --device cuda --model bilstm_kaggle_seed0 --no-dashboard \
  --run-dir /kaggle/working/bilstm_kaggle_seed0
```

Use `--model /kaggle/working/bilstm_kaggle_seed0.pt` if you want the checkpoint directly beside exported notebook outputs. Otherwise it is under `/kaggle/working/shooter/checkpoints/`.

Save notebook outputs and keep both checkpoints and manifests/reports. Session interruption can occur before curriculum completion. Restore the model and repeat the same scientific settings with `--resume`. Selecting GPU enables CUDA computation in the new engine; actual speedup depends on CPU simulation and batch sizes. It does not change the original CPU trainer.

## 12. Tests and troubleshooting

Run tests after installing the declared dependencies:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m unittest discover -s tests -v
```

Dashboard tests need permission to open localhost sockets. CUDA tests are skipped when the test process cannot access CUDA. To exercise the actual local GPU path explicitly:

```bash
python -m unittest tests.test_recurrent.RecurrentTests.test_cuda_training_checkpoint_and_resume -v
```

The recurrent tests check matched baseline defaults, padding/reset isolation, temporal gradients, portable checkpoint/resume behavior, exact budgets, configuration-change rejection, full final-stage qualification, and post-training comparison. An inexpensive integration test lowers the reward threshold only inside its temporary test run; production thresholds are unchanged.

| Problem | What to check |
| --- | --- |
| CUDA requested but unavailable | Driver via `nvidia-smi`, PyTorch build, and process GPU permissions; avoid silently assuming CPU is GPU |
| CUDA out of memory | Stop and reduce context/minibatch/environment settings; apply the SAME scientific settings to a matched control and use a new run name |
| Existing output model | Use `--resume` for that audited run or choose a fresh model name |
| Resume configuration differs | Repeat the saved PPO/environment settings; do not quietly change the experiment |
| Old checkpoint lacks configuration | Use it for inference, or start a fresh audited run for a controlled study |
| Reached stage 4 but still training | The final 100 qualifying episodes must meet the threshold and any minimum-step requirement |
| Stopped at the step cap on stage 2 | The budget ended before mastery; this is not all-stage completion |
| Low training reward after promotion | Difficulty changed; inspect fixed-stage evaluation rather than comparing raw rolling means |
| Candidate survives but kills less | Report both defensive and offensive metrics; success alone is incomplete |
| Reference has no training configuration | The report permits an evaluation comparison but cannot certify an architecture-only training experiment |

`.gitignore` excludes new `.pt` files and `reports/`. Checkpoints already tracked in Git remain tracked. The original trainer automatically resumes its default angular checkpoint and can overwrite it; use separate model names for new baseline experiments. The new trainer protects the configured reference path and stores a snapshot for reproducible evaluation.
