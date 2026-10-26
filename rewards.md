# Experiment 1 rewards

Implemented in [AimEnv.step](aim_env.py). All terms are added on every executed
step, including the hit or timeout step. One step is 1/60 simulated second.

## Reward terms

| Term | Value | Condition |
|---|---|---|
| Hit reward | +1.0 | The target is eliminated by this step's shot. |
| Shot penalty | -0.01 | A shot actually fires. Holding fire during cooldown does not count. |
| Time penalty | -0.001 | Every executed step. |
| Aiming bonus | 0.0005 x (1 - angular_error_degrees / 180) | Every executed step, using the current heading after smooth turning. |
| Missed shooting opportunity penalty | -0.002 | The beam is aligned, the weapon is ready, and no fire request is made. |

The coefficients are the `hit_reward`, `shot_penalty`, `step_penalty`,
`aim_reward`, and `missed_opportunity_penalty` class attributes in
[aim_env.py](aim_env.py).

## Aiming and shooting definitions

- Angular error is the shortest absolute angle between the current gun direction
  and the direction to the target center, from 0 to 180 degrees.
- "Closer aim" means a smaller angular error, not a smaller physical distance.
  The bonus is +0.0005 at 0 degrees, +0.000375 at 45 degrees, +0.00025 at
  90 degrees, and zero at 180 degrees.  
- This is an accuracy bonus on each step, not a reward only for improvement and
  not potential-based shaping.
- Alignment means the forward beam segment would intersect the target's circular
  body. It accounts for target radius and distance, using the same alignment
  calculation as the diagnostics.
- Readiness is checked after this step's cooldown decrement, before a new shot.
  A fire request is the action's third component being greater than zero.
- A missed opportunity requires all three conditions: alignment, readiness, and
  no fire request. Misaligned steps and cooldown-blocked steps receive no
  missed-opportunity penalty.
- Aim, alignment, and aiming reward use the heading after turning, matching the
  heading used for firing. The target's last position is used on a successful
  terminal step, even though the target has been removed from the simulation.

## Examples

| Situation | Total step reward |
|---|---|
| Perfect aim, ready, no fire request | -0.001 + 0.0005 - 0.002 = **-0.0025** |
| Perfect aim, cooling down, no actual shot | -0.001 + 0.0005 = **-0.0005** |
| 90-degree error, no shot | -0.001 + 0.00025 = **-0.00075** |
| 90-degree error, actual missed shot | -0.001 + 0.00025 - 0.01 = **-0.01075** |
| Perfect aim, actual shot hits | +1 - 0.01 - 0.001 + 0.0005 = **+0.9895** |

The target hit terminates the episode. A miss after 300 steps truncates it;
a hit on step 300 takes priority. There is no separate timeout reward or penalty.

## Scale and limitations

The aiming bonus is smaller than the time penalty, so every non-hit step still
has negative reward. At most +0.15 aiming bonus can accumulate over 300 steps.
Missed-opportunity penalties can total -0.6. Perfectly aligned waiting without
firing for all 300 steps therefore returns -0.75.

This discourages waiting, but does not guarantee the desired policy. The Agent
could avoid missed-opportunity penalties by aiming away or holding fire. Use
success rate, shot efficiency, angular error, and firing-opportunity diagnostics
together to evaluate learning. These extra rewards intentionally change the
training objective; improvement must be measured, not assumed.

## Baseline compatibility

The original baseline used only hit, shot, and time terms. Its saved model and
reports have not been overwritten or retrained. New runs use the two additional
terms automatically; use a different model path to preserve the baseline:

```powershell
python train.py train --steps 100000 --seed 0 --model models\experiment1-shaped.zip
```

Observations, actions, and episode endings are unchanged, so old models still
load. Evaluating an old model now calculates mean return under the new rewards,
while historical reports retain the old returns. Do not compare those returns
directly. Success and behavior metrics remain comparable on the same scenarios.
Retrain to learn from the new rewards.
