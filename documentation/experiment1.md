# Experiment 1: Stationary Target Aiming

Experiment 1 was the first reinforcement-learning task in this project. It
trained one stationary Agent to aim at and eliminate one stationary target in
the shared simulation. It was superseded by the mechanics environment, but this
document preserves its contract and results before its task-specific files are
removed.

## Scenario

- The Agent starts at the center of the 800 by 600 arena and cannot move.
- The target is stationary and is sampled with Gymnasium's seeded RNG, at least
  80 pixels from the Agent.
- The Agent's starting heading is randomized.
- Episodes run at 60 simulated frames per second.
- A hit terminates the episode. A miss after 300 steps (five seconds) truncates
  it. A hit on the final step is a success rather than a timeout.
- The task has no walls, field of view, fog of war, self-play, or curriculum.

## Observation and action spaces

The observation was a six-value `float32` vector:

1. Target x offset divided by the world diagonal.
2. Target y offset divided by the world diagonal.
3. Current aim x component.
4. Current aim y component.
5. Weapon cooldown divided by the fire interval.
6. Fraction of episode time remaining.

The action was a three-value continuous `float32` vector in `[-1, 1]`:

1. Absolute aim x direction.
2. Absolute aim y direction.
3. Fire request; values above zero request a shot.

The shared simulation smoothly turned toward nonzero aim vectors. A zero aim
vector held the current heading. A requested shot only counted when the weapon
was ready.

## Rewards

All reward terms applied on every executed step, including terminal steps.

| Term | Value | Condition |
|---|---:|---|
| Hit reward | `+1.0` | The shot eliminates the target. |
| Shot penalty | `-0.01` | A shot actually fires. |
| Step penalty | `-0.001` | Every executed step. |
| Aiming bonus | `+0.0005 * (1 - angular_error_degrees / 180)` | Every step, after smooth turning. |
| Missed-opportunity penalty | `-0.002` | The beam is aligned, the weapon is ready, and no fire is requested. |

Angular error was the shortest angle between current heading and the target
center. Alignment meant the forward beam would intersect the target's circular
body, including its radius and distance. The shaping terms were intended to
encourage accurate aim and firing when aligned, but they changed the objective;
success and shot-efficiency metrics were required to evaluate them.

The original baseline used only the hit, shot, and step terms. Later runs used
the aiming bonus and missed-opportunity penalty.

## PPO training

The trainer used Stable-Baselines3 PPO with `MlpPolicy` on CPU:

- Seed: configurable, default `0`.
- Rollout size: 1024 steps.
- Batch size: 64.
- Requested step counts rounded up to complete 1024-step rollouts.
- Output: a `.zip` policy model and a matching `.evaluation.json` report.

Historical commands:

```powershell
python train.py train --steps 100000 --seed 0 --episodes 100
python train.py train --steps 100000 --seed 0 --model models\experiment1-shaped.zip
python train.py watch --model models\experiment1.zip --episodes 5
python train.py watch --model models\experiment1-shaped-500k.zip --episodes 10 --stochastic --sampling-seed 0
```

## Evaluation and diagnostics

Evaluation ran deterministic PPO, stochastic PPO, random actions, and a
scripted aimer on the same seeded scenarios. Scenario seeds defaulted to
`10000`; sampled PPO actions used a separate configurable sampling seed.

Reported metrics included:

- success rate;
- mean shots across all episodes;
- mean shots on successful episodes;
- mean time to hit on successful episodes;
- mean return;
- mean angular error;
- aligned-step fraction;
- aligned-shot fraction;
- missed-fire-opportunity fraction;
- episodes without shots; and
- raw step, shot, alignment, and firing-opportunity counts.

Historical commands:

```powershell
python train.py evaluate --model models\experiment1.zip --episodes 1000 --eval-seed 20000
python train.py evaluate --episodes 100 --sampling-seed 7 --output models\diagnostics.json
python scripted_bot.py --episodes 10 --eval-seed 10000
python -m unittest test_environment test_aim_env -q
```

## Historical baseline

The original unshaped baseline trained with seed `0` for 100,352 actual steps.
Across 100 evaluation scenarios beginning with seed `10000`, deterministic PPO
achieved 11% success, random actions achieved 27%, and the scripted aimer
achieved 100% success with one shot per successful episode. The run verified
the training pipeline but did not satisfy Experiment 1's learning goal.

## Files formerly associated with Experiment 1

- `aim_env.py`: Gymnasium task definition.
- `train.py`: PPO train, evaluate, and watch entry point.
- `scripted_bot.py`: scripted aiming baseline and playback.
- `test_aim_env.py`: Experiment 1 task tests.
- `documentation/rewards.md`: detailed reward and shaping notes.
- `models/experiment1*.zip` and matching evaluation JSON reports: trained
  artifacts, ignored by Git.
