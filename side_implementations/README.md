# Movement-only exploration

This independent PPO experiment reuses the root simulation, collision handling,
wall intersection helper, map generator, and renderer. It does not inherit the
combat task. There is one freshly spawned Agent, no target, no turning actions,
and no firing. The shared renderer still draws the Agent's directional marker.
Existing combat trainers, configurations, and checkpoints are unchanged.

Run commands from the repository root using the existing Python environment:

```powershell
python -m side_implementations.train
python -m side_implementations.train train --map-mode fixed --map-seed 4 --run-dir ".\models\exploration-map4"
python -m side_implementations.train train --map-mode random --random-spawn --run-dir ".\models\exploration-random"
python -m side_implementations.train watch --run-dir ".\models\exploration-random"
python -m side_implementations.train evaluate --run-dir ".\models\exploration-random" --episodes 100
python -m side_implementations.train evaluate --run-dir ".\models\exploration-random" --map-mode fixed --map-seed 123 --no-random-spawn
```

Defaults live in [configs.yaml](configs.yaml), not the root YAML. Use `--config`
to select another complete configuration. Default training is one environment,
500,000 steps, CPU, map 0, fixed spawn, and a fresh MLP PPO. `--device cuda` is
available; it fails explicitly if CUDA is unavailable. Training refuses to
overwrite an existing run directory. There is no automatic training launch,
checkpoint ranking, or combat-policy reuse.

## Actions and observations

- `Discrete(9)`: stay, north, south, west, east, northwest, northeast,
  southwest, southeast. Diagonals use the shared normalized movement.
- Exactly 18 float32 values in `[0, 1]`: 16 ray distances followed by
  `x / arena_width`, `y / arena_height`.
- Rays start at the agent center, point east first, then advance clockwise in
  22.5-degree increments in screen coordinates. They are world-aligned.
- Rays report the nearest wall or arena boundary within 200 pixels by default.
  Distance is divided by range; 1 means no nearer intersection. These are
  center-to-wall distances, not body clearance.
- Visited tiles, map topology, targets, heading, and remaining time are **not**
  observations. Visited tiles and rays are drawn for humans during playback.

Visited history affects reward but is hidden from the feed-forward policy.
Thus this deliberately small baseline is partially observable: it cannot
reliably distinguish first visits from revisits based on its current input.

## Exploration reward

Coverage tiles are 50 x 50 pixels by default, independent of the larger map
generation cells. A tile is visited when the agent center enters it.
The spawn tile is marked visited without a bonus. Revisits never earn a bonus.
For a first visit:

```text
bonus = new_tile_reward / (1 + reward_decay * k / (total_tiles - 1))
reward = bonus - step_penalty
```

Here `k` counts previously rewarded tiles in this episode. Defaults give
`0.01` for the first new tile, declining gently toward `0.00667`, with a
`0.001` penalty on every step, including first visits. All history and reward
decay reset each episode. This is an episodic novelty bonus, not a learned
curiosity model. Negative total returns are possible even while exploring;
use tile coverage as the main diagnostic.

Tile size must exceed the agent diameter and divide each map generation cell
dimension. This keeps walls on tile edges and every tile center reachable in
the connected generated map. Default coverage denominator is 192 tiles.
Coverage includes the spawn tile; `new_tiles_visited` excludes it.

Physics and decisions remain 60 Hz. Episodes truncate after 600 physics steps
(10 simulated seconds), even if all tiles were visited. There is no action
repeat and no other reward or penalty.

## Maps, spawns, and outputs

`map_mode: fixed` uses exactly `map_seed` (including zero); unlike the older
navigation trainer, a positive number does **not** mean a seed range here.
`map_mode: random` generates a new seed on every reset. `random_spawn: false`
uses the generator's fixed agent spawn for that map. When enabled, spawn is
sampled continuously inside the arena with full-body wall clearance. Explicit
reset seeds reproduce both map and spawn sequences.

Each run saves:

- `config.json`: effective training settings and actual training steps.
- `final.zip`: the fresh trained PPO; watch/evaluate use this by default.
- `map_seeds.jsonl`: every training reset's concrete map seed and spawn,
  including vector-wrapper auto-resets at episode boundaries.
- `episodes.monitor.csv`: completed episode rewards, lengths, and coverage.
- `progress.csv`: PPO training statistics.
- `evaluation_<seed>.json`: explicit evaluation results, including per-episode
  map seeds, spawn coordinates, coverage, and rewards.

Watch/evaluate load environment settings from the saved run, then apply explicit
map/spawn CLI overrides; editing today's YAML does not alter old checkpoints'
sensor or reward settings. Evaluation uses deterministic actions and its own RNG
seed. A fixed map plus fixed spawn produces repeated identical episodes, so use
different maps or random spawns to test variability. Random evaluation seeds do
not by themselves guarantee maps absent from training; use explicit fixed seeds
and check the training seed log for held-out comparisons.

Training steps must divide into complete PPO rollouts; batch size must divide
the single environment's rollout length. Other unspecified algorithm options
retain Stable-Baselines3 defaults. No packages beyond the root requirements
are needed.

Tests:

```powershell
python -m unittest tests.test_exploration -q
```
