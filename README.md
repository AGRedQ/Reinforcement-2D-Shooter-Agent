# Adaptive 2D Shooter Agent

## Classes

- Entity: Entities are the base of all characters and agents in this project
- Agent: The Agent with Reinforcement Learning capabilities. Inherit Entity
- Player: Inheriting from Agent, it can do what the Agent can, but with player control capabilities.

- Environment: The main Environment in which the Entities are in. Used for training headlessly.
- Renderer: Using Pygame library, render the Environment and all its citizens at a fixed speed.

## Playground walls

Run `python interactive_test.py` to see two sample wall segments in the player-controlled
playground. Walls are stored separately in `Environment.walls` as
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

## Experiment 1: Training

Install dependencies with `python -m pip install -r requirements.txt`.
The Gymnasium adapter in [aim_env.py](aim_env.py) reuses the simulation without
adding learning-specific rules to it. This is a baseline, not yet evidence that
the Agent has mastered Experiment 1.

- Each episode has one stationary Agent at the center and one stationary target.
  Reset restores both, randomizes initial aim and target position using Gymnasium's
  seeded RNG, keeps the target inside the bounds, and enforces an 80-pixel minimum
  center-to-center distance. The playground's random-spawn modifier is not used.
- Observation: six `float32` values: target offset x/y divided by the world
  diagonal, current aim x/y, cooldown divided by the firing interval, and the
  fraction of episode time remaining.
- Action: three `float32` values in `[-1, 1]`: target aim x/y and fire request
  (`> 0` fires; `<= 0` does not). Movement is always zero. Zero aim holds heading.
- Each step advances exactly `1/60` second. Reward is `+1` for a hit, `-0.01`
  per actual shot, and `-0.001` per step, plus an angular-accuracy bonus up to
  `+0.0005` per step and `-0.002` for a missed shooting opportunity.
  See [rewards.md](rewards.md) for all conditions, examples, and limitations.
- A hit terminates the episode. After 300 steps (5 simulated seconds), a miss
  truncates it. A hit on the final step counts as success, not timeout.
- PPO uses Stable-Baselines3's MLP policy and built-in GAE. There is no behavioral
  cloning, self-play, or automatic curriculum yet.

```powershell
python train.py train --steps 100000 --seed 0 --episodes 100
python train.py evaluate --model models\experiment1.zip --episodes 1000 --eval-seed 20000
python train.py evaluate --episodes 100 --sampling-seed 7 --output models\diagnostics.json
python train.py watch --model models\experiment1.zip --episodes 5
python train.py watch --model models\experiment1-shaped-500k.zip --episodes 10 --stochastic --sampling-seed 0
python scripted_bot.py --episodes 10 --eval-seed 10000
python -m unittest test_environment test_aim_env -q
```

Training runs headlessly on CPU, saves `models\experiment1.zip`, and writes
`models\experiment1.evaluation.json`. These outputs are git-ignored. PPO collects
complete 1024-step rollouts, so the actual training step count rounds up.
`watch` runs the same task in the Pygame playground; close the window to stop.
Playback is deterministic by default. Add `--stochastic` to sample from the
learned policy instead. `--eval-seed` chooses target positions and initial aims;
`--sampling-seed` (default 0) independently controls sampled actions. Repeating
both seeds with the same model and software reproduces stochastic playback.
Changing the sampling seed has no effect on deterministic actions. Playback
never trains or saves the model and preserves the caller's Torch CPU RNG state.
The original human-controlled demo remains available via `python interactive_test.py`.

The scripted aimer lives in [scripted_bot.py](scripted_bot.py). Run it directly
to watch the scripted baseline without loading or training a PPO model.
`--episodes` defaults to 5 and `--eval-seed` defaults to 10000, matching PPO
playback's scenario seeds. Close the Pygame window to stop early.
Evaluation imports the same scripted policy from this module.

Evaluation compares deterministic PPO (`ppo`), sampled PPO (`ppo_stochastic`),
a random policy, and a scripted aimer on
the same seed-indexed scenarios (default seeds start at 10000, separate from the
training RNG stream). It reports success rate, mean shots over all episodes,
mean shots on successful episodes, mean time to hit on successful episodes,
and mean return. Success-only metrics are `null` if there are no hits; do not
interpret them without the success rate. Use fresh evaluation seeds for final
testing after tuning and repeat training with multiple seeds before drawing
conclusions. The scripted baseline is an aiming reference, not a proven optimal
controller. A short smoke run verifies plumbing, not learning quality.

### Aiming and firing diagnostics

Evaluation includes these measurements for every policy. Observations and the
action space are unchanged. Existing models can be evaluated without retraining;
mean return uses the current reward rules in [rewards.md](rewards.md).

| JSON field | Meaning |
|---|---|
| `mean_angular_error_degrees` | Absolute angle from current aim to target center, averaged over all executed steps (0 aligned, 180 facing away). |
| `aligned_step_fraction` | Fraction of executed steps whose forward beam would intersect the target body, accounting for radius and distance. |
| `aligned_shot_fraction` | Aligned actual shots divided by all actual shots; `null` if no shots were fired. |
| `missed_fire_opportunity_fraction` | Aligned, cooldown-ready steps without a fire request divided by all aligned, ready steps; `null` if there were no opportunities. |
| `episodes_without_shots`, `no_shot_episode_fraction` | Count and fraction of episodes without any actual shots. |

Reports also include raw counts: `total_steps`, `total_shots`, `aligned_steps`,
`aligned_shots`, `ready_aligned_steps`, and `missed_fire_opportunities`.
Fractions pool counts across episodes rather than averaging episode fractions;
longer episodes contribute more steps. All step measurements include the final
step and use the heading **after turning**, when a shot would occur. Readiness
uses the cooldown after that step's time decrement, before any new shot.
A held fire request during cooldown is neither an actual shot nor a missed
firing opportunity. These are diagnostic correlations, not proof of a failure's
cause. In this one-target task, an aligned actual shot should end in a hit.
The scripted aimer decides whether to fire using the pre-turn observation, so
it can register a missed opportunity on the step that first turns into alignment;
this metric does not imply that the scripted policy is an optimal controller.

Both PPO modes use identical scenario seeds. Stochastic PPO samples from its
learned distribution, rather than using the mean action. `--sampling-seed`
(default 0) controls Torch's CPU action sampling separately from `--eval-seed`;
evaluation preserves the caller's Torch RNG state. Repeat sampling seeds to
assess sampling variability. Random actions use a separate NumPy RNG seeded
with `--eval-seed`. Watch mode also supports optional stochastic playback.
Evaluation prints JSON;
`--output` optionally saves it without retraining or replacing the model.

### First baseline result

This historical run used only hit, shot, and time rewards, before the aiming
bonus and missed-opportunity penalty were added.

With training seed 0, 100352 steps, and 100 evaluation scenarios starting at seed
10000, deterministic PPO achieved 11% success, random actions 27%, and the
scripted aimer 100%. The scripted aimer used one shot per success. This run used
Stable-Baselines3 2.9.0, Gymnasium 1.4.0, and PyTorch 2.14.1 on CPU.
The saved model verifies the training pipeline but does **not** meet the
Experiment 1 goal yet. Improving learning is the next experiment; these results
do not establish that PPO itself is unsuitable.

## References
- Reinforcement Learning Agent for a 2D Shooter Game (https://arxiv.org/pdf/2509.15042)
- Generalized Advantage Estimation, GAE (https://arxiv.org/abs/1506.02438)

## Some ideas
- Curriculum Learning, which is being used 
- Behavior Cloning using stripted bots, inspired by (2509.15042), not used yet.
