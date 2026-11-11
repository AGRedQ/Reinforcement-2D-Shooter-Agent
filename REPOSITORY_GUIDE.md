# Understanding the Reinforcement 2D Shooter Agent

This guide explains the current merged implementation in beginner-friendly language. It covers the game, observations, policy networks, PPO training, checkpoints, visualization, and experiments. Commands assume your terminal is in this project folder.

The main entry point is **`training_random_map.py`**. It trains against shooting opponents on generated maps. The former `training.py` has been split into shared training code in `models/ppo.py` and network code in `models/mlp.py`.

**The current models are feedforward. No LSTM, BiLSTM, attention, or stored observation history has been implemented.** Deep Sets handles multiple bullets in the current observation; it does not add memory across time.

## Contents

1. [What the project does](#1-what-the-project-does)
2. [Files and responsibilities](#2-files-and-responsibilities)
3. [Game mechanics and one simulation step](#3-game-mechanics-and-one-simulation-step)
4. [Observations and sensing](#4-observations-and-sensing)
5. [Actions](#5-actions)
6. [Rewards and episode boundaries](#6-rewards-and-episode-boundaries)
7. [Maps, opponents, and curriculum](#7-maps-opponents-and-curriculum)
8. [Policy architectures](#8-policy-architectures)
9. [PPO and GAE](#9-ppo-and-gae)
10. [Parallel environments and checkpoints](#10-parallel-environments-and-checkpoints)
11. [Setup and running the project](#11-setup-and-running-the-project)
12. [Rendering and player controls](#12-rendering-and-player-controls)
13. [Dashboard, reports, and metrics](#13-dashboard-reports-and-metrics)
14. [Limitations and meaningful comparisons](#14-limitations-and-meaningful-comparisons)
15. [Tests and a reading path](#15-tests-and-a-reading-path)

## 1. What the project does

The project trains a blue circular agent to find, aim at, and shoot an opponent in an 800 × 800 arena. Walls block movement, projectiles, and line of sight. A neural network receives numerical sensor readings and chooses movement, turning, and firing actions.

The learning method is **reinforcement learning**: the agent gathers experience, receives rewards and penalties, and updates its policy to increase expected future reward. PPO is implemented directly in PyTorch.

There are two environment behaviors:

| Environment configuration | Opponent | Episode ending |
| --- | --- | --- |
| `HuntEnv()` defaults | Scripted `Threat` that cannot shoot | Time limit |
| `HuntEnv(shoot_back=True)` | Scripted `ShootingThreat` | Agent death or time limit |

`training_random_map.py` selects the second behavior and generated maps. The passive fixed-map environment remains available through the Python API. Killing an opponent respawns it within the same episode; it does not mean the episode has been won.

| Mode | Agent controller | Purpose |
| --- | --- | --- |
| Training | Policy being updated | Learn from interaction |
| Watch | Saved policy | Observe behavior without updating weights |
| Manual arena play | Keyboard | Explore mechanics and rendering |
| Play against agent | Saved policy; you control the opponent | Test the agent interactively |

## 2. Files and responsibilities

| File | Responsibility |
| --- | --- |
| [entities.py](entities.py) | Characters, bullets, walls, collision geometry, actions, and sensors |
| [environments.py](environments.py) | `HuntEnv`, rewards, spawning, episode statistics, and curriculum |
| [map_generation.py](map_generation.py) | Seeded obstacle generation and map-seed argument parsing |
| [models/mlp.py](models/mlp.py) | `ActorCritic`, `BulletEncoder`, and observation batching |
| [models/ppo.py](models/ppo.py) | Worker environments, PPO, GAE, checkpoint handling, training, and watch helpers |
| [training_random_map.py](training_random_map.py) | Public command-line entry point selecting combat and generated maps |
| [renderer.py](renderer.py) | Pygame rendering, manual play, player-versus-agent controls |
| [visibility.py](visibility.py) | Dense wall-clipped geometry for optional sight shading |
| [render_preview.py](render_preview.py) | Reproducible headless screenshots of sight and movement |
| [dashboard.py](dashboard.py) | Metrics history, local HTTP bridge, and Streamlit process lifecycle |
| [dashboard_app.py](dashboard_app.py) | Browser dashboard, Plotly charts, CSV download, and stop control |
| [training_report.py](training_report.py) | Persistent CSV samples and an offline HTML report for each run |
| [requirements.txt](requirements.txt) | Python dependencies |
| [tests/](tests/) | Geometry, environment, model, training, dashboard, report, and renderer tests |
| [checkpoints/](checkpoints/) | Saved policies included in the merged repository and newly saved models |
| [RENDERING.md](RENDERING.md) | Focused instructions for sight visualization |

The main flow is:

```text
training_random_map.py
    → models/ppo.py
        → environments.py → entities.py
        → models/mlp.py
        → renderer.py → visibility.py
        → dashboard.py → dashboard_app.py
        → training_report.py
```

`Renderer` inherits from `HuntEnv`, so rendering uses the same simulation rules. Drawing adds no sensor information to policy inputs.

## 3. Game mechanics and one simulation step

The default timestep is `1 / 30` second. A 1,000-step episode lasts about 33.3 simulated seconds if the agent survives. Training can run faster than real time.

Positions use screen coordinates: x increases rightward, y downward. Angle zero points right; increasing angles turn clockwise. Agent movement is relative to its aim.

| Property | Agent | Scripted shooting opponent |
| --- | --- | --- |
| Health | 100 | 100 |
| Body radius | 14 | 14 |
| Movement speed | Maximum 220 units/second | Curriculum-dependent, 0–180 |
| Shot damage | 25 | 25 |
| Bullet speed | 900 units/second | 450 units/second |
| Bullet range | 900 units | 600 units |
| Firing cooldown | 0.35 seconds | 0.8 seconds |

The agent turns at 200 degrees/second when a turning action is selected. Velocity approaches the desired velocity gradually; stopping does not instantly remove momentum. Circle-versus-wall collision resolution permits sliding along walls. There is no character-to-character collision resolution.

Bullets use swept path collision checks and stop at their remaining maximum range. This avoids checking only the projectile's endpoint and missing thin obstacles between frames.

`HuntEnv.step(action)` performs these operations:

1. Read whether the previous enemy sensors detected the opponent.
2. Turn and move the agent, update its cooldown, and create a requested shot if ready.
3. Penalize a newly created shot if the previous sensors detected no opponent.
4. Update the opponent's movement.
5. Advance agent bullets, award hits and kills, and respawn a killed opponent. A kill clears the agent's remaining bullets.
6. Let a shooting opponent fire if ready and permitted by its line-of-sight rules.
7. Advance hostile bullets, apply damage penalties, and detect agent death. Hostile bullets already in flight survive opponent respawns.
8. Increment the step count, refresh sensors, update statistics, and return the new observation.

The Gymnasium interface is:

```python
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(action)
```

A finished episode includes a statistics dictionary at `info["episode"]`. Call `reset()` after either ending flag is true. Training workers do this automatically.

## 4. Observations and sensing

The agent receives numerical observations, not pixels. All configurations include 32 wall rays around the full circle, with a 600-unit sensing limit. Rays rotate with the agent; ray 0 points forward. A normalized distance of `0.5` means 300 units, and `1.0` means no wall closer than 600 units.

Enemy sensing has two configurations, and bullet sensing is independently optional.

### Legacy enemy rays: 60 state values

Five wall-ray directions within the frontal 60-degree cone also detect enemies: 0°, +11.25°, +22.5°, −22.5°, and −11.25°, in that observation order. Enemies can fall between these discrete rays. A nearer wall blocks detection.

Indices below are zero-based and inclusive:

| Indices | Values | Count |
| --- | --- | --- |
| 0–31 | Wall distances / 600 | 32 |
| 32–36 | Enemy detection flags | 5 |
| 37–41 | Enemy distances / 600; 1 when undetected | 5 |
| 42–43 | Local forward and sideways velocity / maximum speed | 2 |
| 44 | Cooldown / firing cooldown | 1 |
| 45 | Health / maximum health | 1 |
| 46–59 | Previous action as three one-hot groups | 14 |
| **Total** | | **60** |

Flags are grouped together, followed by distances; they are not interleaved.

### Angular enemy regions: 64 state values

`--angular-regions` replaces the five enemy rays with six adjacent angular regions, using boundaries:

```text
−30°, −15°, −5°, 0°, +5°, +15°, +30°
```

The sensor checks the opponent's circular silhouette and wall occlusion within each region. It reports presence and nearest visible surface distance. This covers the cone without the legacy rays' angular gaps. The 32 wall rays remain unchanged.

A separate center ray reports whether the agent is directly aligned with a visible target. It supplies information; it does not prevent or authorize firing.

| Indices | Values | Count |
| --- | --- | --- |
| 0–31 | Wall distances / 600 | 32 |
| 32–37 | Enemy-region detection flags, left to right | 6 |
| 38–43 | Enemy-region distances / 600 | 6 |
| 44–45 | Center-ray detection flag and distance / 600 | 2 |
| 46–47 | Local velocity / maximum speed | 2 |
| 48 | Normalized cooldown | 1 |
| 49 | Normalized health | 1 |
| 50–63 | Previous action one-hot groups | 14 |
| **Total** | | **64** |

### Optional hostile-bullet observations

With `--bullet-sensing`, the environment returns a dictionary:

```text
state:   float32 array shaped (60,) or (64,)
bullets: float32 array shaped (N, 4)
```

Only live hostile bullets within the continuous 60-degree forward cone, within 600 units, and unblocked by walls are included. Friendly bullets are excluded. Each bullet supplies:

| Feature | Normalization |
| --- | --- |
| Local forward position offset | Divide by 600 |
| Local sideways position offset | Divide by 600 |
| Local forward relative velocity | Divide by projectile speed + agent maximum speed |
| Local sideways relative velocity | Same velocity scale |

Relative velocity is bullet velocity minus agent velocity. `N` varies, including zero. `batch_observations()` pads to the largest bullet count in that batch and produces a boolean `bullet_mask`. Padding is ignored by the encoder; bullets are not truncated to a fixed count.

### What remains unobserved

The policy receives no absolute position, absolute aim angle, full map, explicit opponent velocity, stage number, or remaining episode time. Enemy and bullet visibility remain limited by direction, range, and walls.

Previous action supplies a little recent context, but no previous observation is retained. This is a **partially observable** task: identical current inputs can correspond to different hidden situations.

## 5. Actions

An action is `[movement, aim, fire]`, represented by Gymnasium as `MultiDiscrete([9, 3, 2])`.

| Component | Choices |
| --- | --- |
| Movement | 0 stay; 1 forward; 2 forward-right; 3 right; 4 backward-right; 5 backward; 6 backward-left; 7 left; 8 forward-left |
| Aim | 0 turn left; 1 hold; 2 turn right |
| Fire | 0 hold fire; 1 request a shot |

For example, `[2, 0, 1]` moves forward-right, turns left, and requests firing. The aim changes before movement is calculated. Diagonal movement is normalized, so it is not faster than cardinal movement.

There are 54 possible combinations. The actor predicts three separate categorical distributions rather than a single distribution with 54 entries. Requesting fire during cooldown creates no bullet and incurs no wasted-shot penalty.

## 6. Rewards and episode boundaries

`RewardConfig` defines these defaults:

| Event | Reward change |
| --- | --- |
| Agent bullet hits opponent | +1 |
| That hit kills opponent | Additional +3 |
| Agent creates a bullet without prior enemy detection | −0.1 |
| Agent loses health in combat | −0.04 per health point actually lost |
| Agent dies in combat | Additional −3 |

Four hits kill a full-health opponent and earn `1 + 1 + 1 + 4 = 7`, before penalties. Losing 25 health costs 1 point. Losing all 100 health costs 4 points in damage penalties plus 3 for death.

There is no direct movement, discovery, exploration, or survival bonus. A visible-target shot that misses has no explicit miss penalty. A shot without detection can still hit later and earn a hit reward in addition to its original firing penalty. Detection for the firing penalty is measured before the current action, including any turning.

| Ending flag | Meaning | Value bootstrap during training |
| --- | --- | --- |
| `terminated` | Agent died in combat | No continuation value |
| `truncated` | Episode reached `max_steps` | Bootstrap from final observation if not also terminated |

An agent death on the last allowed step can set both flags. The death takes precedence for bootstrapping. Passive hunting never terminates through agent death.

Episode rewards reported in statistics are raw game rewards. The extra value bootstrap is used only in the training targets.

## 7. Maps, opponents, and curriculum

### Fixed and generated maps

`HuntEnv()` uses a fixed arena with boundary walls and seven interior obstacles. Passing `map_seeds` selects generated maps instead; an explicit `layout` and `map_seeds` cannot be combined.

Generated maps contain 6–9 rectangular obstacles in randomly selected cells of a 3 × 3 grid, plus boundary walls. Widths and heights vary from 40 to 150 units. Cell margins preserve corridors for the default body sizes. The map seed determines geometry independently of the spawn RNG.

The trainer's `--map-seeds` syntax is significant:

| Argument | Maps played |
| --- | --- |
| `--map-seeds 9` | Seeds 0 through 9 inclusive; the default |
| `--map-seeds '[9]'` | Only seed 9 |
| `--map-seeds '[2,7,11]'` | Seeds 2, 7, 11 repeatedly in that order |

Each worker starts at the first map and advances on episode reset. Opponent respawns do not advance maps. Explicitly reseeding `reset(seed=...)` restarts the map cycle. A resumed process starts its cycle again; the map cursor is not checkpointed. `--seed` controls spawn/action randomness, while `--map-seeds` selects geometry.

### Scripted opponents

`Threat` chooses destinations, steers with short wall-sensing whiskers, pauses, and replaces destinations when needed. At the final stage it can strafe when targeted. These rules are programmed; there is no trained opponent, A* planner, or self-play.

`ShootingThreat` adds line-of-sight firing with a 0.8-second cooldown and a delay after reset. Its bullet damage and speed do not become easier at stage 0: stage 0 makes it stationary, but it can still shoot in combat mode.

`Player` replaces scripted movement and shooting with keyboard and mouse input in `--play-against` mode. Its health, movement, and weapon defaults match the agent, including 900-unit/second bullets and a 0.35-second firing cooldown.

### Curriculum stages

| Stage | Opponent speed | Spawn angle relative to agent aim | Requested spawn distance | Extra rule |
| --- | --- | --- | --- | --- |
| 0 | 0 | ±10° | 80–250 | Require line of sight |
| 1 | 60 | ±30° | 100–350 | Require line of sight |
| 2 | 120 | ±90° | 150–500 | No visibility requirement |
| 3 | 180 | Anywhere | 200–700 | Wider search |
| 4 | 180 | Anywhere | 250–900 | Prefer hidden spawns and evade aim |

Spawn sampling may relax restrictions when suitable positions cannot be found. Distances are requested ranges, not guarantees on every spawn.

Promotion requires mean reward of at least 35 over the last 100 qualifying episodes at the current stage by default. Earlier-stage replay episodes and episodes from a different stage do not qualify. Promotion clears the reward window. Stage 4 is final.

Above stage 0, each reset has a 20% chance to replay a randomly selected earlier stage. This is practice on easier tasks, not an experience replay buffer. Stage changes take effect at subsequent resets. Since workers autoreset before the trainer processes promotion, an episode may already have started under the old stage.

## 8. Policy architectures

`ActorCritic` supports four configurations:

| New-policy flags | Enemy state size | Bullet summary | Decision MLP input |
| --- | --- | --- | --- |
| Neither flag | 60 | None | 60 |
| `--angular-regions` | 64 | None | 64 |
| `--bullet-sensing` | 60 | 32 | 92 |
| Both flags | 64 | 32 | 96 |

All use this decision network by default:

```text
Current state, optionally concatenated with bullet summary
    → Linear(input size, 256) → Tanh
    → Linear(256, 256) → Tanh
        ├─ Movement head: 9 logits
        ├─ Aim head:      3 logits
        ├─ Fire head:    2 logits
        └─ Value head:   1 scalar
```

The **actor** chooses actions. The **critic** predicts discounted future reward; its output is not a win probability. The trunk is shared, with three actor heads and one value head.

Training samples actions to encourage exploration. Watch and play-against choose each head's most probable action by default; `--stochastic` enables sampling. Joint log-probability is the sum of the three head log-probabilities.

### How Deep Sets encodes bullets

A shared network encodes every visible bullet independently:

```text
Each bullet's 4 features
    → Linear(4, 32) → Tanh
    → Linear(32, 32) → Tanh
    → masked mean pooling (32) + masked max pooling (32)
      + bounded count N/(N+1) (1)
    → Linear(65, 32) → Tanh
    → 32-value summary concatenated with current state
```

Pooling makes the result independent of bullet-list order. Empty sets produce a zero summary. PPO trains the bullet encoder jointly with the actor and critic.

**Pooling is across bullets at one timestep, not across time.** More sensors can improve decisions without creating temporal memory. No recurrent state is passed between actions in the current implementation.

## 9. PPO and GAE

The trainer repeatedly collects a rollout, estimates advantages, updates the policy, and collects fresh experience. Default batch size is `8 environments × 256 steps = 2,048 transitions`.

A **return** represents immediate and future rewards. With discount factor `gamma = 0.99`, later rewards receive progressively less weight. An **advantage** measures whether an outcome was better or worse than the critic expected.

`compute_gae()` implements Generalized Advantage Estimation. Away from boundaries:

```text
delta[t] = reward[t] + gamma × value[t+1] − value[t]
advantage[t] = delta[t] + gamma × lambda × advantage[t+1]
return[t] = advantage[t] + value[t]
```

Episode boundaries stop this recurrence. At a time limit, the final observation's discounted value is first added to the training reward. At death, no such continuation is added. Workers preserve the final observation separately from the reset observation.

PPO reevaluates collected actions under the updated policy. The probability ratio is `exp(new_log_probability − old_log_probability)`. The clipped policy objective discourages large changes; the critic learns through squared return error, and an entropy bonus encourages exploration:

```text
loss = clipped policy loss + 0.5 × value loss − 0.01 × entropy
```

The implementation standardizes advantages per minibatch, shuffles transitions each epoch, clips gradient norm, and uses Adam. It logs approximate KL but does not use a target-KL early stop. PPO clipping discourages large updates; it is not a strict bound on policy divergence.

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-envs` | 8 | Parallel environments |
| `--rollout-steps` | 256 | Steps per environment per rollout |
| `--epochs` | 4 | Passes over each rollout |
| `--minibatch-size` | 512 | Transitions per optimizer step |
| `--hidden-size` | 256 | MLP width for a new policy |
| `--lr` | 0.0003 | Adam learning rate |
| `--gamma` | 0.99 | Reward discount |
| `--gae-lambda` | 0.95 | GAE weighting |
| `--clip` | 0.2 | PPO clipping range |
| `--ent-coef` | 0.01 | Entropy coefficient |
| `--vf-coef` | 0.5 | Value-loss coefficient |
| `--max-grad-norm` | 0.5 | Gradient-norm limit |
| `--max-steps` | 1000 | Episode time limit |
| `--promotion-reward` | 35 | Curriculum promotion threshold |
| `--promotion-window` | 100 | Qualifying episodes for promotion |
| `--replay-probability` | 0.2 | Earlier-stage practice probability |
| `--save-every` | 25 | PPO updates between checkpoint saves |
| `--seed` | 0 | Initial training/watch RNG seed |

The current implementation runs tensors on CPU and sets PyTorch to one CPU thread during training. Installing a CUDA-enabled PyTorch build does not automatically move this trainer to the GPU.

## 10. Parallel environments and checkpoints

`VecEnv` starts environments with multiprocessing's `spawn` method and communicates over pipes. Without rendering, each environment runs in a worker. With `--render`, one runs locally through `Renderer`, and the others remain workers.

Workers automatically reset after death or a time limit. `global_step` counts transitions across all workers, not steps in just one arena. A rollout may span multiple episodes.

### Naming and default checkpoints

Models are resolved under the project-root `checkpoints/` directory. `.pt` is appended if the name has no extension. Only the basename is used: passing an absolute path does not select a checkpoint outside that directory.

| Flags when `--model` is omitted | Trainer's selected filename |
| --- | --- |
| Neither sensing flag | `mlp_ppo_raised_damage.pt` |
| `--bullet-sensing` | `deep_sets_ppo_random_map.pt` |
| `--angular-regions` | `angular_mlp_ppo.pt` |
| Both | `angular_deep_sets_ppo.pt` |

`models/ppo.py` also defines a shared-helper default of `mlp_ppo_random_map.pt`; the public wrapper overrides that default with `mlp_ppo_raised_damage.pt`.

Several checkpoints are already included in the merged repository. **Training automatically resumes and overwrites the selected checkpoint if it exists.** Use a new `--model` name for a separate experiment. A same-named checkpoint in the project root is not used; the trainer looks in `checkpoints/`.

### Saved state and compatibility

A checkpoint contains model weights, optimizer state, curriculum stage and reward window, observation size, action dimensions, hidden size, architecture, sensing metadata, and cumulative step/episode/update counts. Saving uses a temporary file followed by atomic replacement.

`architecture` is `mlp` or `deep_sets_v1`; `enemy_sensing` is `rays_v1` or `angular_regions_v1`. Older compatible metadata defaults to MLP and legacy rays. Loading restores the saved architecture and sensing automatically.

Adding `--bullet-sensing` to an existing MLP checkpoint or `--angular-regions` to a legacy-ray checkpoint is rejected. Those flags change the network input and require a new model name. Resuming retains checkpoint width and optimizer state but uses the requested learning rate.

RNG states, live environments, partial rollouts, map cursors, and the dashboard's recent episode history are not saved. Resume continues learning progress but does not reproduce an uninterrupted trajectory exactly. Map selection and environment settings must be supplied again.

The stop button and `--total-episodes` request a graceful stop: the trainer waits until every environment has finished an episode after the stop request. Cumulative episode totals can exceed the requested count. On resume, `--total-episodes` is a cumulative target, not an additional-episode budget. Ctrl+C saves through cleanup without waiting for all episodes. An incomplete final rollout is not optimized.

`.gitignore` excludes new `.pt` files and `reports/`. Checkpoints already tracked by Git remain tracked despite the ignore rule.

## 11. Setup and running the project

Create an environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python training_random_map.py --help
```

Dependencies are NumPy, PyTorch, Gymnasium, Pygame, Streamlit, and Plotly. The dashboard uses a local browser server; it is separate from the Pygame arena window.

### Start a new experiment

These commands use custom names. Each starts fresh only if its selected checkpoint does not already exist.

```bash
# Legacy-ray MLP, combat, maps 0 through 9.
python training_random_map.py --model experiment_mlp --total-episodes 500

# Legacy rays plus Deep Sets bullet sensing.
python training_random_map.py --model experiment_bullets \
  --bullet-sensing --total-episodes 500

# Angular sensing plus Deep Sets bullet sensing.
python training_random_map.py --model experiment_angular \
  --angular-regions --bullet-sensing --total-episodes 500
```

Append `--render` to show one arena. Append `--no-dashboard` for console training; persistent reports are still generated. Omitting `--total-episodes` allows training to continue until stopped.

A small training smoke run, rather than a performance experiment:

```bash
python training_random_map.py --model smoke_angular \
  --angular-regions --bullet-sensing --map-seeds '[2]' \
  --num-envs 2 --rollout-steps 16 --minibatch-size 32 \
  --epochs 1 --max-steps 100 --total-episodes 4 --no-dashboard
```

### Resume and watch

```bash
# Continue the experiment, with a cumulative target of 1,000 episodes.
python training_random_map.py --model experiment_angular \
  --map-seeds 9 --total-episodes 1000

# Watch the included angular Deep Sets policy.
python training_random_map.py --watch --model angular_deep_sets_ppo \
  --map-seeds '[2,7,11]' --episodes 5

# Choose a fixed evaluation stage and sample actions instead of taking argmax.
python training_random_map.py --watch --model angular_deep_sets_ppo \
  --map-seeds '[11]' --stage 4 --episodes 5 --stochastic
```

Watch disables curriculum replay and does not update weights or save checkpoints. `--stage` overrides watch/play difficulty; it does not set the training start stage. Checkpoint metadata restores sensing without repeating the sensing flags.

### Play manually or against a model

```bash
# Manual control of the blue agent in the fixed passive arena.
python renderer.py --stage 2 --visibility

# Manual control against a scripted shooting opponent on generated map 5.
python renderer.py --shoot-back --map-seeds '[5]'

# Control the opponent against the saved agent; inference only.
python training_random_map.py --play-against --model angular_deep_sets_ppo \
  --map-seeds '[4]' --episodes 5
```

## 12. Rendering and player controls

The renderer draws at twice the display resolution and downsamples for smoother edges. Walls, both sides' bullets, bodies, trails, health, and episode statistics are displayed. Generated-map changes refresh cached wall surfaces.

| Control | Normal arena view | Play-against mode |
| --- | --- | --- |
| WASD | Move blue agent relative to aim during manual play | Move your opponent character in screen directions |
| Q/E or left/right arrows | Turn blue agent during manual play | Aim with mouse instead |
| Space | Fire during manual play | Hold left mouse button to fire |
| R | Toggle agent sensor visualization | Disabled; H controls agent rays |
| V | Toggle continuous geometric sight shading | Disabled |
| T | Toggle trails | Disabled |
| F | Toggle FPS cap | Toggle FPS cap |
| H | No effect | Reveal hidden agent/bullets and agent rays |
| Escape or close window | Close rendering | End the interactive view |

Movement/fire keys do not override the saved agent during watch or rendered training. In play-against mode, normal visibility follows the player's 60-degree cone, range, and walls. `--player-hacks` starts with the H visibility toggle enabled and requires `--play-against`. It changes display visibility, not the AI's observations.

Discrete rays are enabled by default. For an angular policy, the R toggle also controls translucent wall-clipped region fills. Continuous sight shading is off by default; press V or pass `--visibility` to `renderer.py`. It remains disabled in player-versus-agent mode. In ordinary observer views, undetected targets are dimmed but can remain visible behind walls.

Headless screenshots use SDL's dummy driver through the renderer's CLI:

```bash
python renderer.py --headless --visibility --seed 7 \
  --screenshot /tmp/shooter-sight/seed-7.png
python render_preview.py --output-dir /tmp/shooter-sight
```

The preview script captures contact, occlusion, range, rays, and movement. Filled sight is display geometry, not a new policy sensor. Closing the arena during training leaves simulation running; the dashboard stop button controls graceful training shutdown.

## 13. Dashboard, reports, and metrics

### Live dashboard

`MetricsDashboard` stores bounded metric histories, exposes an authenticated localhost bridge, and launches a Streamlit browser dashboard with Plotly charts. The UI includes smoothing, raw-curve display, CSV download, and a stop-after-episode control. `--no-dashboard` avoids launching Streamlit; report writing remains active.

### Persistent reports

Every training run creates a unique directory under `reports/`, or under the parent chosen by `--report-dir`:

```text
reports/<model>-<timestamp>-<unique suffix>/
    metrics.csv
    report.html
```

CSV samples are flushed every 10 completed episodes in that run, plus one final partial sample if needed. Each row contains cumulative episodes, run episodes, environment steps, updates, rolling episode metrics, and the latest available PPO statistics. Nonfinite or unavailable metrics appear blank.

`report.html` is generated during cleanup on normal stop, Ctrl+C, or a handled training error. It includes configuration, starting/final progress, stop reason, and self-contained SVG charts. An abrupt process kill can prevent final HTML generation, though flushed CSV samples may remain. Resuming creates a new report directory.

Episode metrics average the last up to 100 completed episodes of the current run across all workers, including easier-stage replay. They are not an average of just the 10 episodes since the preceding CSV row. Dashboard export is a separate long-form history table with metric, environment step, and value columns.

| Metric | Interpretation |
| --- | --- |
| `mean_reward` | Raw episode reward; higher usually helps, but compare at the same difficulty |
| `mean_kills` | Opponents killed per episode |
| `hit_rate` | Total hits divided by total shots in the recent episode window; zero if no shots |
| `mean_wasted_shots` | Shots created without previous enemy detection |
| `mean_distance` | Agent path length per episode |
| `mean_detect_steps` | Mean per-episode delay to first detection for opponent spawns, including an undetected final spawn's elapsed delay |
| `mean_damage_taken` | Health lost per episode in combat |
| `death_rate` | Mean episode death indicator in combat; 0.65 means 65% of recent episodes ended with death |
| `stage` | Current curriculum stage, not necessarily the stage of replay episodes |
| `policy_loss` | PPO actor objective; not a direct agent-quality score |
| `value_loss` | Critic prediction error against estimated returns |
| `entropy` | Randomness across the three action distributions |
| `approx_kl` | Estimated policy change on the rollout |
| `clip_fraction` | Fraction of ratios outside PPO's clipping range |
| `explained_variance` | Fit of rollout value estimates to return targets; negative values indicate poor fit |
| `learning_rate` | Current optimizer learning rate |

Training charts mix changing policies, stages, maps, and replay episodes. A falling reward can accompany a curriculum promotion. Shorter episodes after death also change kills, shots, and distance totals. These charts describe training; they are not an independent benchmark.

## 14. Limitations and meaningful comparisons

The current implementation has several boundaries that matter when interpreting results:

| Limitation | Consequence |
| --- | --- |
| No observation history or recurrent state | The policy cannot explicitly remember where a hidden opponent was seen |
| No exploration bonus or coverage objective | Finding distant or hidden opponents is rewarded only through later combat outcomes |
| Legacy rays have angular gaps | A visible opponent can escape those sensors between rays |
| Angular sensing still has occlusion and range limits | A region sensor does not reveal an opponent behind a wall |
| Deep Sets pools bullet information into 32 values | It summarizes the current set and may discard useful detail |
| Scripted opponents | Good performance does not establish robustness against diverse human strategies |
| Fixed collection of training maps | Repeating seeds does not establish generalization to unseen layouts |
| No dedicated batch evaluation command | Watch prints episode statistics, but systematic paired evaluation needs additional tooling |
| Checkpoints omit RNG and environment state | Resume is not exact trajectory continuation |

There is no implemented LSTM/BiLSTM experiment or measured memory-improvement result in this version.

To assess a future memory model, keep observation features, reward rules, map selection, stage, and evaluation action mode consistent. Compare equal environment-transition budgets rather than only equal episode counts: combat deaths produce different episode lengths. Use multiple training seeds and held-out evaluation maps, and report reward, kills, deaths, damage, and detection delays.

Also compare the same recurrent policy with history available and with history reset or shortened. An improvement over a feedforward policy alone could result from additional parameters or training differences; a history ablation provides evidence about temporal information use. A BiLSTM for online action selection must process only observations available up to the decision, without future game states leaking into its input.

The included angular Deep Sets checkpoint uses the architecture and sensing supported by the merged source. It should not be treated as a matching baseline for a 60-input MLP environment without restoring its saved configuration. The filename alone does not describe the full training history or prove superiority.

## 15. Tests and a reading path

Run the suite from the project folder after installing dependencies:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  python -m unittest discover -s tests -v
```

The dummy drivers allow Pygame tests to run without a desktop or audio device. Dashboard integration tests additionally require permission to start and contact localhost servers. They mock automatic browser opening where appropriate.

| Test module | Coverage |
| --- | --- |
| `test_angular_regions.py` | Region coverage, center alignment, legacy compatibility, angular training/watch/play |
| `test_bullet_sensing.py` | Bullet visibility, variable sets, pooling invariance, encoder updates, checkpoint compatibility |
| `test_map_generation.py` | Geometry, connectivity, seed cycles, workers, generated-map training |
| `test_threat.py` | Shooting, cover, damage, death, autoreset, combat training |
| `test_player.py` | Human controls, shared combat behavior, visibility toggles, inference-only play |
| `test_renderer.py` | Simulation parity, both bullet colors, map refresh, angular fills |
| `test_visibility.py` | Sight geometry, occlusion, shading, and display toggles |
| `test_dashboard.py` | Local bridge, Streamlit UI/server, CSV export, graceful stop |
| `test_training_report.py` | CSV intervals, final partial samples, HTML, interruptions, resume reporting |

At the merge validation preceding this guide update, the 76-test suite was exercised, with dashboard failures caused by the validation environment resolved in a focused rerun. Rendered previews were inspected as well. That validates the exercised behavior; it does not measure learning quality or memory.

For reading the code, start with `training_random_map.py` to see the supported modes, then follow `HuntEnv.reset()` and `step()` in `environments.py`. Read `Agent.sense()`, `_sense_regions()`, `observe_bullets()`, and `observation()` to connect the game to its inputs. Next read `ActorCritic` and `BulletEncoder` in `models/mlp.py`, followed by rollout collection, `compute_gae()`, and `ppo_update()` in `models/ppo.py`. Finish with checkpoint loading, reporting, and rendering.

Useful terms:

| Term | Meaning here |
| --- | --- |
| Policy | Network rule that maps observations to action probabilities |
| Observation | Information exposed to the agent, smaller than the complete world state |
| Episode | Interaction from reset until death or time limit |
| Transition | One observation/action/reward/next-observation interaction |
| Rollout | Recently collected transitions used for one PPO update cycle |
| Value | Estimated discounted future reward |
| Advantage | Estimated outcome relative to the critic's expectation |
| Curriculum | Progression from easier to harder opponent configurations |
| Deep Sets | Order-independent encoding of a variable-size bullet set |
| Ablation | Controlled removal of a feature, such as history, to test its contribution |
