# Adaptive 2D Shooter Agent

## Classes

- Entity: Entities are the base of all characters and agents in this project
- Agent: The Agent with Reinforcement Learning capabilities. Inherit Entity
- Player: Inheriting from Agent, it can do what the Agent can, but with player control capabilities.

- Environment: The main Environment in which the Entities are in. Used for training headlessly.
- Renderer: Using Pygame library, render the Environment and all its citizens at a fixed speed.

## Playground walls

Run `python -m tests.interactive_test --seed 0 --max-walls 16` to preview procedural
walls and navigation nodes in the player-controlled playground. Change the seed
to generate another reproducible layout. `--max-walls 0` gives an open arena;
`--fixed-layout` restores the original two-wall demo.

The generator uses an 8-by-6 grid of cell-center nodes. A randomized spanning
tree stays open, and walls are placed only across other connections, so every
node and both spawn points belong to one connected walkable graph. Cell spacing
must exceed the agent diameter, providing body clearance along every graph edge.
The player occupies the agent spawn; the stationary threat occupies a node at
maximum graph distance from it. The additional idle Agent uses another node.
Yellow nodes turn green on their first visit (within one player radius); the
spawn node starts visited. Thin lines show traversable connections. These are
debug overlays, not vision-restricted observations or automatic movement.

The wall limit counts interior segments, including separate collinear segments,
not the arena boundary. The generator uses up to the requested cap: at most 35
walls fit this grid while keeping its spanning tree open. Node visits in the
playground have no reward effect. The original mechanics training scenario stays
unchanged; the separate navigation task below uses a smaller generated map.

Validate generation, connectivity, movement clearance, and rendering with
`python -m unittest tests.test_map_generation tests.test_environment -q`.

Walls are stored separately in `Environment.walls` as
`Wall(start, end)` objects and drawn as six-pixel-wide lines. Other environments
start with no walls, preserving Experiment 1. Wall segments block circular
agent bodies and beams. Movement uses swept collision (including segment
endpoints) and slides along walls instead of tunneling through on large steps.
Bodies must start outside walls; contact is permitted. Up to eight contacts are
resolved per step, after which any remaining movement is discarded for safety.
The drawn six-pixel width is cosmetic: collision uses the mathematical segment.
Beams stop at the nearest wall, including endpoint and collinear intersections,
and still pierce targets before it. Targets centered on or behind the blocking
surface are protected, even if their circular body overlaps it.
Walls do not yet hide threats in observations or implement fog of war.

## Actions

- `Action(move_x, move_y, aim_x, aim_y, fire)` supplies one action per living Agent.
- Movement components remain discrete: `-1`, `0`, or `1`.
- Aim components specify an absolute target direction, not a turn rate. For example, `(1, 0)` points right and `(0, 1)` points down. Any finite nonzero vector is accepted; its magnitude does not affect turning speed.
- The Agent turns smoothly along the shortest arc and fires along its current aim. An aim vector of `(0, 0)` keeps its current heading.
- `fire` is a Boolean.

## Experiments and Goals
- Experiment 0: Set up and validate the simulation environment: movement, aiming, shooting, hit handling, rendering, configuration, and regression tests.

- Experiment 1: The Agent is able to consistently shoot threats while standing still.

- Experiment 2: The Agent is given or unlocked moving capabilities. Potentially we'll introduce walls.

- Experiment 3: Threats shoot back. Train the Agent to fight and survive, starting against scripted opponents and progressing to self-play against saved Agent policies.

- Experiment 4: Cone of Vision and Fog of War is introduced.

- Experiment 5: A small playable game for friends to play on and compete with the Agent.

## Experiment 2: Training with movement and vision

[mechanics_env.py](mechanics_env.py) adapts the reusable simulation for training
with the mechanics shown in the interactive playground. It controls one Agent
against one stationary target, with the original playground's two fixed walls and
45-degree half-angle field of view. The Agent's observation contains its
normalized position and heading, target-relative position only when the target
is both inside its field of view and unobstructed, a visibility indicator,
weapon cooldown, remaining episode time, a last-seen target offset and age, and
the known wall map. A hidden target has a zero current offset and a false
visibility indicator. The last-seen offset is derived from the target's stored
world position as the Agent moves; it expires two seconds after the last sighting
and never updates while the target is hidden.

Every reset samples wall-clear positions for the Agent and target, keeps them at
least 100 pixels apart, and randomizes the Agent's initial heading. Gymnasium's
seeded RNG makes the generated scenario reproducible for a given seed.

The three-value discrete action selects one of nine movement directions (including
idle), one of five turn commands (fast left, slow left, hold, slow right, fast
right), and whether to fire. Slow turns are 30 degrees per second and fast turns
are 180 degrees per second, allowing precise shots without continuously changing
an absolute aim target. The Agent moves and turns through the core
`Environment.step`; wall collision, line-of-sight checks, and beam blocking remain
implemented by the shared simulation. Episodes end on a hit or after 600 steps.
Rewards are +0.2 once after moving at least 75 pixels from the spawn, +1 for a
hit, +0.1 each for the first target sighting and first visible-target alignment,
-0.001 per missed shot, and -0.001 per step. The three milestone rewards are
each available once per episode, so the Agent cannot gain them by repeatedly
moving, toggling line of sight, or holding aim on the target.

```powershell
python -m training.train_mechanics train --steps 100000 --seed 0
python -m training.train_mechanics watch --model models\mechanics.zip --episodes 5
python -m training.train_mechanics train --no-walls --steps 100000 --seed 0 --model models\mechanics-open.zip
python -m training.train_mechanics watch --no-walls --model models\mechanics-open.zip --episodes 5
python -m unittest tests.test_mechanics_env -q
```

`--no-walls` uses the same scenario, CoV, actions, rewards, spawn randomization,
and observation shape, but removes wall collision and line-of-sight blocking.
Its wall-map observation entries are zero.

The memory fields increase the observation from 17 to 21 values. Existing
mechanics models use the old observation shape and must be retrained.

## Multi-map navigation training

Training entry points live in [training](training); run them as modules from
the repository root. [wallgen_navpoint_training.py](training/wallgen_navpoint_training.py)
trains a fresh PPO policy using [WallgenNavpointEnv](wallgen_navpoint_env.py).
The easy layout has 12 nodes on a 4-by-3 grid, 200-pixel wall segments, and up to
four walls by default. Map selection changes at episode resets:

- `--map-seed "[1,2,3,4,5]"` cycles those exact seeds in order, repeatedly.
- `--map-seed 5` cycles seeds 1 through 5.
- `--map-seed "[5]"` uses only seed 5. `[0]` preserves the original fixed map.
- `--map-seed 0` (or any negative integer) generates a fresh random map seed
  for every episode.

Vector workers start at staggered positions in the cycle, and each worker
repeats the whole list. Walls, connections, spawns, and visited-node state all
reset together. Spawns are fixed for each map seed; initial heading varies.
The target is stationary; episodes end on a hit or after `max_steps` (600 by default).

Both training entry points take their defaults from the `training` sections of
[configs.yaml](configs.yaml), including CLI options, PPO settings, logging,
thread count, and episode limits. Omitting the command uses the configured
`command` (default `train`); explicit CLI options override YAML. For example:

```powershell
python -m training.wallgen_navpoint_training
python -m training.wallgen_navpoint_training --map-seed 5 --device cpu
python -m training.train_mechanics
```

Set `training.wallgen_navpoint.train.map_seed` to a YAML array such as
`[1, 2, 3, 4, 5]`, a positive count, or `0` for random maps. The default `[0]`
retains the old fixed-map behavior. Navigation training defaults to CUDA;
select CPU in YAML or with `--device cpu` if CUDA is unavailable.
Mechanics `--no-walls` enables the open arena; `--walls` overrides a YAML
`no_walls: true` setting.

The observation has **66 float32 values**, all bounded by [-1, 1]:

- 0-12: the original combat, agent pose, cooldown, time, and target-memory fields.
- 13-48: 12 `(relative_x / width, relative_y / height, visited)` triples in
  row-major node order; visited flags are 0 or 1.
- 49-65: 17 open-connection flags, iterating nodes row-major and emitting the
  right connection then the downward connection where each exists.

Nodes and connectivity are globally known, but hidden target position is not
exposed by these fields. The policy still controls all movement and firing;
the graph does not choose routes for it. Fixed target spawns can be memorized;
training on a finite seed list alone does not demonstrate unseen-map generalization.

The old +0.2 displacement milestone is **replaced** by first-visit bonuses.
The spawn node starts visited without reward. Each other node gives `0.2 / 11`
once per episode when the agent center is within its body radius of the node.
The same visit update sets the observation flag and reward. Total exploration
bonus is at most +0.2; hit, visibility, alignment, shot and time rewards are
unchanged. Watch mode uses the same navigation overlay as the playground.
Both training tasks' Pygame watch views show the live **episode reward total**,
the latest step's reward, and the episode step count in a bottom-left HUD.
The score includes all bonuses and penalties, resets each episode, and includes
the terminal hit or timeout step. It is display/debug information only, also
available as `episode_reward` and `last_reward` in `info`; policy observations,
rewards, and saved-model compatibility are unchanged. The player-controlled
playground has no RL reward score.

For NVIDIA GPU training, activate the project virtual environment and install
the optional CUDA wheel after the base requirements:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
python -m training.wallgen_navpoint_training train --steps 500000 --device cuda --map-seed "[1,2,3,4,5]" --max-walls 4 --run-dir models\wallgen-navpoint
python -m training.wallgen_navpoint_training watch --run-dir models\wallgen-navpoint --checkpoint best.zip
python -m training.wallgen_navpoint_training evaluate --run-dir models\wallgen-navpoint --episodes 100 --seed 20000
python -m unittest tests.test_reward_hud tests.test_wallgen_navpoint_env tests.test_wallgen_training tests.test_map_generation tests.test_environment -q
```

CUDA training fails explicitly if CUDA is unavailable; `--device cpu` is an
explicit alternative. PyTorch's CUDA wheel includes its runtime; a compatible
NVIDIA driver is required, not a separate system CUDA toolkit. The simulation
still runs on CPU. Small MLP policies are not necessarily faster on GPU.

By default eight vectorized environments use the configured map selection. Each PPO rollout contains
2,000 transitions (250 per environment), with batch size 250 and two 128-unit
hidden layers. `--steps` and `--eval-every` must be multiples of 2,000 so a 500k
request trains on exactly 500,000 transitions. When changing YAML PPO settings,
these multiples become `n_envs * ppo.n_steps`; the batch size must divide that
rollout size. Evaluation steps do not count
toward that budget.

The default evaluates deterministically on at least 100 episodes at initialization, every
50k training steps, and after the final update. Evaluation resets to the same
map/heading seed streams starting at 10000. For finite map selections,
`--eval-episodes` (training) and `--episodes` (evaluation) are minimums rounded
up to complete map cycles. Every entry in the selected list receives the same
number of completed episodes, regardless of worker count or episode duration;
duplicate list entries retain their weighting. For example, 100 episodes over
4 maps gives 25 per map; 100 over 128 maps becomes 128, one per map.
This covers the training map selection, not a separate held-out map set.
Random-map evaluation keeps the exact requested episode count and generates
reproducible layouts from its own seed streams. Evaluation metrics include
`requested_episodes`, actual `episodes`, and the completed episodes' `map_seeds`.
Saved outputs include configuration, initial preview map geometry, CSV training
metrics, JSONL evaluation metrics, periodic checkpoints, `final.zip`, and
`best.zip` selected by hit rate then shorter mean episode duration, not reward.
Separate evaluation defaults to seed 20000. Watch/evaluate load map settings
and episode limits from the saved run, avoiding accidental mismatches. Watch
mode cycles maps too. Legacy runs keep their original fixed map, including seed 0.
Existing run directories are rejected rather than overwritten.

`map_seeds.jsonl` records every training episode reset, with `env_index`,
`episode` (zero-based per worker), and the concrete `map_seed`. It excludes
environment-checker and evaluation resets, but includes the next episode
automatically reset by SB3 at the final transition. Use the recorded seeds as
an explicit array to replay layouts with the saved wall/grid settings.
`map.json` is only the constructor's initial preview, labeled with its seed;
it is not a record of all training maps (random mode previews seed 0).

## Movement-only exploration baseline

[side_implementations](side_implementations/README.md) contains an isolated PPO
task with nine movement actions, 16 finite wall-distance rays plus position,
and gently diminishing first-visit tile rewards. It supports fixed or random
generated maps and optional random spawns, without targets, turning, or firing.
Visited tiles are tracked by the environment but hidden from the policy.
It reuses the shared simulation without changing the combat task.

```powershell
python -m side_implementations.train train
python -m side_implementations.train watch
```

Defaults are in [side_implementations/configs.yaml](side_implementations/configs.yaml).
See the task README for random-map commands, reward details, evaluation, and
the limitations of hiding exploration history from a feed-forward policy.

## Tests

All automated tests and the interactive playground are grouped in [tests](tests).
Run commands from the repository root:

```powershell
python -m unittest discover -s tests -t . -q
python -m unittest tests.test_reward_hud -q
python -m tests.interactive_test --seed 0 --max-walls 16
```

The playground is manual and is not included in automated test discovery.

## Archived Experiment 1

The retired stationary-target aiming task, its reward design, commands, and
historical results are documented in [documentation/experiment1.md](documentation/experiment1.md).
