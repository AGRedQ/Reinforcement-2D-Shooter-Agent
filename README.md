# Reinforcement 2D Shooter Agent

A reinforcement learning project that trains agents to move, aim, shoot, and survive in a 2D arena using **Proximal Policy Optimization (PPO)**. The environment runs in Gymnasium, policies use PyTorch, and Pygame provides interactive playback and human-versus-agent combat.

Agents learn from numerical observations of walls, enemies, and incoming bullets. The project supports feedforward, LSTM, and BiLSTM policies, generated maps, a five-stage combat curriculum, and self-play.

For detailed architecture notes, experiment settings, and troubleshooting, see the [repository guide](REPOSITORY_GUIDE.md).

## Demo

<!-- For GitHub inline playback, replace the link below with the bare URL generated
     by uploading the demo in GitHub's Markdown editor. Repository video paths do not
     embed a player. Free GitHub plans require videos smaller than 10 MB. -->

[Watch the demo video](output/demo_video/video_demo.webm).

## Features

- **Generated arenas:** seeded obstacle layouts in an 800 × 800 arena, with walls that block movement, bullets, and sight.
- **Partial observations:** wall rays, angular enemy regions, health, velocity, cooldown, previous actions, and visible hostile bullets.
- **Deep Sets bullet encoder:** summarizes a variable number of visible projectiles for the policy.
- **Temporal policies:** LSTM and BiLSTM variants use windows of past observations; the BiLSTM receives only history available at action time.
- **Curriculum learning:** progressively faster, more distant, and evasive scripted opponents, with replay of earlier stages.
- **Self-play:** symmetric duels against a periodically refreshed frozen opponent policy.
- **Monitoring and evaluation:** a live Streamlit/Plotly dashboard, CSV metrics, offline HTML reports, and fixed-trial policy comparisons.

## Setup

Use Python 3.10 or newer. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Dependencies are NumPy, PyTorch, Gymnasium, Pygame, Streamlit, and Plotly. Interactive playback requires a graphical display. Training can run without a window or dashboard.

The scratch trainer supports CPU and CUDA. To check whether your PyTorch installation can access CUDA:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`--device auto` selects CUDA when available and otherwise uses CPU. `--device cuda` requires an accessible CUDA GPU. The game simulation runs on CPU with either setting. The original `training_random_map.py` trainer uses CPU.

## Quick start

### Watch a saved agent

The repository includes checkpoints under [`checkpoints/`](checkpoints/). Watch the angular Deep Sets policy on a generated map:

```bash
python training_random_map.py --watch \
  --model angular_deep_sets_ppo --map-seeds '[4]' --stage 4
```

Playback uses deterministic actions by default. Add `--stochastic` to sample actions, or `--episodes 10` to watch more rounds.

### Play against the agent

```bash
python training_random_map.py --play-against \
  --model angular_deep_sets_ppo --map-seeds '[4]'
```

Move with **WASD**, aim with the **mouse**, and hold **left click** to shoot. **H** toggles wallhack and agent sensing overlays; **Esc** exits.

To control the agent body against a scripted opponent instead:

```bash
python renderer.py --shoot-back --map-seeds '[4]' --stage 2
```

In this mode, **WASD** moves relative to aim, **Q/E** or **Left/Right** rotates aim, and **Space** fires. **R** toggles sensing rays, **V** toggles sight shading, **T** toggles trails, and **F** toggles the frame cap.

### Run a small training check

This bounded CPU run starts a fresh BiLSTM policy and collects 2,048 environment transitions:

```bash
python training_bilstm_scratch.py \
  --model bilstm_smoke --device cpu --steps 2048 \
  --run-dir reports/bilstm_smoke --no-dashboard --skip-evaluation
```

This checks the training workflow; it is too short to produce a strong agent. Use a fresh model name for a new run, or repeat the command with `--resume` to load its saved state.

## Training

### LSTM, BiLSTM, and matched feedforward experiments

[`training_bilstm_scratch.py`](training_bilstm_scratch.py) trains randomly initialized policies through a shared CPU/CUDA PPO engine. It defaults to BiLSTM and runs until all five curriculum stages meet the completion criterion:

```bash
python training_bilstm_scratch.py \
  --model bilstm_seed0 --device auto \
  --run-dir reports/bilstm_seed0 --no-dashboard
```

By default, promotion requires a mean reward of at least **35 over 100 qualifying episodes** at the current stage. Earlier-stage replay episodes do not count. The scratch trainer also requires the final stage to meet that threshold before declaring curriculum completion. There is no default hard step limit, so a run can continue for a long time if the threshold is not reached.

Useful options:

| Option | Purpose |
| --- | --- |
| `--architecture feedforward`, `lstm`, or `bilstm` | Choose the policy architecture; default: `bilstm` |
| `--steps 2000000` | Set a hard transition budget; may stop before curriculum completion |
| `--min-steps 2500000` | Require minimum experience before curriculum-based stopping |
| `--context-length 32` | Set the recurrent observation window; default: 32 |
| `--train-control` | After recurrent training, train a fresh feedforward control with the same transition budget |
| `--resume` | Restore this trainer's checkpoint with matching recorded training settings |
| `--skip-evaluation` | Skip reference comparison and its checkpoint requirement |

For example, resume the run above with:

```bash
python training_bilstm_scratch.py \
  --model bilstm_seed0 --device auto --resume \
  --run-dir reports/bilstm_seed0 --no-dashboard
```

Keep the original training settings when resuming. Weights, optimizer state, counters, and curriculum progress are restored; live environment states and full random trajectories are not restored.

### Original feedforward trainer

[`training_random_map.py`](training_random_map.py) supports scripted combat, optional exploration rewards, and self-play:

```bash
python training_random_map.py --model feedforward_seed0 --map-seeds 9
```

The dashboard starts automatically. Add `--no-dashboard` for terminal-only training or `--render` to display a training environment. `--total-episodes N` sets an episode limit. Exploration shaping is opt-in with `--exploration-bonus`.

**Checkpoint behavior:** the original trainer automatically resumes and overwrites an existing output checkpoint. Its default model is the included `angular_deep_sets_ppo.pt`; choose a separate `--model` name for a new experiment. The scratch trainer requires explicit `--resume` and protects its configured comparison checkpoint.

### Self-play

Start a separate duel experiment initialized from the included angular policy:

```bash
python training_random_map.py --self-play \
  --model self_play_seed0 --self-play-init angular_deep_sets_ppo \
  --map-seeds '[4]' --no-dashboard
```

The learner fights a frozen copy refreshed every 50 PPO updates by default. Self-play disables curriculum replay and exploration rewards. The initialization checkpoint is left unchanged; an existing self-play output resumes automatically.

Watch the resulting duel with:

```bash
python training_random_map.py --self-play --watch \
  --model self_play_seed0 --map-seeds '[4]' --stochastic
```

## Maps and policy inputs

Both trainers accept these map selections:

| Argument | Maps used |
| --- | --- |
| `--map-seeds 9` | Seeds 0 through 9, inclusive; the training default |
| `--map-seeds '[4]'` | Only seed 4 |
| `--map-seeds '[2,7,11]'` | The listed seeds in order |

Each worker advances its map cycle on episode reset. `--seed` controls initial spawn/action randomness separately from map geometry. Resuming starts the map cycle from its first seed again.

With the default angular and bullet sensing, the policy combines **64 state values** with a **32-value Deep Sets bullet summary**. The feedforward policy uses a two-layer, 256-unit decision MLP. Recurrent variants add a history branch before that MLP. The action space has three categorical heads: **9 movement choices, 3 aim choices, and 2 firing choices**.

## Reports and evaluation

Training opens a local dashboard unless `--no-dashboard` is supplied. It shows rewards, combat metrics, curriculum progress, and PPO statistics. Closing the browser leaves training running. The dashboard's stop button finishes current episodes and saves; **Ctrl+C** also saves during cleanup.

Runs write `metrics.csv` and a final `report.html` under `reports/`, including when the dashboard is disabled. The scratch trainer also writes configuration/progress files such as `experiment.json` and `status.json` to its `--run-dir`.

After scratch training, evaluation compares the candidate with `checkpoints/angular_deep_sets_ppo.pt` by default. It uses fixed stages, map seeds, and spawn seeds, and includes a current-observation-only ablation for recurrent policies. To evaluate a saved candidate separately:

```bash
python training_bilstm_scratch.py --evaluate-only \
  --model bilstm_seed0 --device auto \
  --run-dir reports/bilstm_seed0_evaluation --no-dashboard
```

Outputs include `comparison.html`, `comparison.json`, `evaluation.csv`, and a provenance manifest. Default evaluation maps are 14–17 and must be excluded from the candidate's training maps. Use `--compare-to` to select another compatible reference checkpoint.

Comparisons with historical checkpoints do not establish an architecture-only advantage when their training settings are unknown. `--train-control` produces an additional matched feedforward comparison under `matched_comparison/`. See the [comparison guide](REPOSITORY_GUIDE.md#6-fair-architecture-comparisons) for experimental details.

Relative model names resolve to `checkpoints/`, with `.pt` appended if omitted. New `.pt` files and `reports/` are ignored by Git; preserve checkpoints together with their run configurations and reports when archiving experiments.

## Repository layout

```text
.
├── entities.py                   # Bodies, bullets, collisions, actions, sensing
├── environments.py               # Gymnasium environment, rewards, curriculum
├── map_generation.py             # Seeded obstacle layouts
├── models/
│   ├── mlp.py                    # Feedforward actor-critic and Deep Sets encoder
│   ├── recurrent.py              # Observation history and LSTM/BiLSTM policies
│   ├── ppo.py                    # Original PPO trainer and shared utilities
│   ├── curriculum_ppo.py         # Shared CPU/CUDA scratch training engine
│   ├── evaluation.py             # Fixed-trial comparisons and reports
│   └── runtime.py                # Devices, checkpoint loading, provenance
├── training_random_map.py        # Feedforward combat and self-play entry point
├── training_bilstm_scratch.py     # Scratch architecture experiments
├── self_play.py                  # Symmetric duel environment
├── renderer.py                   # Pygame playback and manual controls
├── visibility.py                 # Wall-clipped sight geometry
├── render_preview.py             # Headless sight-preview captures
├── dashboard.py / dashboard_app.py
├── training_report.py            # CSV metrics and offline HTML summaries
├── checkpoints/                  # Included saved policies
├── tests/                        # Unit and integration tests
├── output/                       # Project PDF and presentation
├── requirements.txt
└── REPOSITORY_GUIDE.md            # Detailed training and comparison reference
```

## Tests

Run the test suite without a display or audio device:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m unittest discover -s tests -v
```

Tests cover physics, sensing, map generation, rendering, exploration rewards, self-play, recurrent history, checkpoint/resume behavior, reporting, and training integration. CUDA tests skip when a GPU is unavailable.

To inspect all command-line options:

```bash
python training_random_map.py --help
python training_bilstm_scratch.py --help
python renderer.py --help
```
