# Understanding the Reinforcement 2D Shooter Agent

This guide describes the original fixed-map, 60-input MLP implementation.
The current training entry point is `training_random_map.py`, with PPO in
`models/ppo.py`, policies in `models/mlp.py`, and the Streamlit dashboard in
`dashboard.py`. It also supports generated maps, shooting threats, Deep Sets
bullet sensing, and optional 64-input angular sensing. See the module docstring
and `python training_random_map.py --help` for current commands and defaults.
The architecture, commands, rewards, and limitations below are historical.

This guide explains that original source code in beginner-friendly language. Start with sections 1–5 for the main idea; the later sections explain learning, running the project, and its limitations.

## Contents

1. [What this project does](#1-what-this-project-does)
2. [The four files and their jobs](#2-the-four-files-and-their-jobs)
3. [How one game step works](#3-how-one-game-step-works)
4. [What the agent can see](#4-what-the-agent-can-see)
5. [What the agent can do](#5-what-the-agent-can-do)
6. [Rewards: how the agent knows it did well](#6-rewards-how-the-agent-knows-it-did-well)
7. [The target and the difficulty stages](#7-the-target-and-the-difficulty-stages)
8. [The neural network](#8-the-neural-network)
9. [How training works: PPO and GAE](#9-how-training-works-ppo-and-gae)
10. [Parallel environments, saving, and resuming](#10-parallel-environments-saving-and-resuming)
11. [How to run the project](#11-how-to-run-the-project)
12. [How to read the training metrics](#12-how-to-read-the-training-metrics)
13. [Related knowledge and a learning path](#13-related-knowledge-and-a-learning-path)
14. [Limitations and useful improvements](#14-limitations-and-useful-improvements)
15. [What was verified for this guide](#15-what-was-verified-for-this-guide)

## 1. What this project does

The project trains an AI-controlled character to **find, aim at, and shoot a target in a 2D arena**.

Imagine teaching someone to play a simple game through practice:

1. Give them information about their surroundings.
2. Let them choose how to move, turn, and shoot.
3. Give points for useful results.
4. Let them practice many times and adjust their strategy.

That is the basic idea of **reinforcement learning**, or **RL**.

Here, the learner is a blue circular character. The target is an orange-red circle. Rectangular walls block movement, bullets, and sight. The default arena is 800 × 800 world units, with four boundary walls and seven interior obstacles.

The target follows programmed movement rules. It does not learn and does not shoot back. The learner cannot die in the current environment; its health stays full. Killing a target immediately creates another target, so a training episode can contain many kills.

**The goal is to collect reward during a fixed-length practice session.** By default, a session lasts 1,000 simulation steps, equivalent to about 33.3 seconds of simulated game time. Training can run faster than real time.

There are three ways to use the project:

| Mode | Who controls the blue character? | Purpose |
| --- | --- | --- |
| Manual play | You, using the keyboard | Understand the arena and controls |
| Training | A neural network that is being updated | Learn a strategy through experience |
| Watch | A saved neural network | Observe what a trained model does |

## 2. The four files and their jobs

The repository currently contains four tracked Python source files. There is no dependency manifest or saved model included in the inspected checkout.

| File | Simple explanation | Important parts |
| --- | --- | --- |
| [entities.py](entities.py) | The game objects and their physics | `Agent`, `Threat`, `Bullet`, `Wall`, movement, sensors, collisions |
| [environments.py](environments.py) | The game rules and scoring system | `HuntEnv`, `RewardConfig`, `STAGES`, `Curriculum`, spawning |
| [training.py](training.py) | The learner and its practice routine | `ActorCritic`, `VecEnv`, `compute_gae`, `ppo_update`, checkpoints, command-line options |
| [renderer.py](renderer.py) | The game window and training graphs | `Renderer`, keyboard controls, `MetricsDashboard` |

The main dependency direction is:

```text
entities.py
    ↑
environments.py
    ↑
renderer.py
    ↑
training.py  (also directly uses entities.py and environments.py)
```

`Renderer` inherits from `HuntEnv`: it uses the same game rules and adds a Pygame window. Training normally uses environments without drawing them. With `--render`, one environment is displayed.

The learning algorithm is implemented directly using PyTorch. The code does not delegate PPO training to Stable-Baselines3 or another RL training framework.

## 3. How one game step works

The game advances in small fixed time intervals. The default `dt` is `1 / 30` second.

The main interaction is:

```text
Current sensor values
        ↓
Neural network chooses [move, aim, fire]
        ↓
Environment updates the game
        ↓
New sensor values + reward
        ↓
Choose the next action
```

Inside `HuntEnv.step(action)`, the order is:

1. Remember whether the previous sensor readings detected the target.
2. Turn and move the learner; create a bullet if it fires and its cooldown allows it.
3. Apply the wasted-shot penalty if a bullet was created while the previous readings detected no target.
4. Move the target according to its rules.
5. Move bullets, check collisions, apply damage, and award hit/kill rewards.
6. If the target dies, clear existing bullets and respawn the target.
7. Refresh the learner's sensors and update statistics.
8. Return the next observation, reward, and episode status.

The Gymnasium interface returns:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

| Result | Meaning in this repository |
| --- | --- |
| `observation` | The next 60 numbers available to the learner |
| `reward` | Points earned or lost during this step |
| `terminated` | Always `False`; there is no death or success condition ending the task |
| `truncated` | `True` when the episode reaches its step limit |
| `info` | Includes completed episode statistics under `info["episode"]` at the time limit |

`reset()` starts a new episode: it clears bullets and statistics, selects a stage, places the learner and target, and returns the first observation.

### Movement and collisions

Positions are `(x, y)`: x increases to the right, y increases downward. Angles use radians. An angle of zero points right on the screen; increasing the angle turns clockwise.

The learner's maximum speed is 220 world units per second. Velocity gradually approaches the commanded velocity, so movement has a little inertia. Choosing “stand still” slows it down rather than instantly setting velocity to zero.

Characters are circles and walls are axis-aligned rectangles. Collision handling pushes a character out of a wall and removes the part of its velocity pointing into that wall. This allows sliding along walls. There is no character-to-character collision resolution in the current step logic.

Bullets travel at 900 units per second. The code checks their path during each step, rather than only checking the final position. This **swept collision** approach helps a fast bullet hit a thin wall instead of jumping through it between frames.

## 4. What the agent can see

The learner does **not** receive the rendered game image. It receives a NumPy array containing **60 numbers**, calculated in `Agent.sense()` and `Agent.observation()`.

Think of a robot carrying distance sensors rather than a camera.

### Wall sensors

There are 32 rays distributed around the learner in a full circle. Each measures the nearest wall distance, up to 600 world units.

For example, a reading of `0.5` means the nearest wall on that ray is 300 units away. A reading of `1.0` means there is no wall closer than the 600-unit sensor limit.

The rays rotate with the learner's aim. Ray 0 always points forward.

### Target sensors

Only rays inside a nominal 60° forward cone can detect the target. Because rays are spaced 11.25° apart, **five rays** satisfy that cone rule: 0°, ±11.25°, and ±22.5° relative to the aim.

For each of those rays, the observation contains a detection flag and a target distance. A wall blocks detection if it is closer than the target.

This is discrete ray sensing: a target can be inside the cone but fall between rays, especially when far away. “Inside the cone” does not guarantee detection.

### The exact observation layout

Indices below are zero-based and inclusive.

| Indices | Values | Count | Meaning |
| --- | --- | --- | --- |
| 0–31 | Wall distances / 600 | 32 | Nearby obstacle information |
| 32–36 | Target detection flags | 5 | `1` if the corresponding forward ray detects the target, else `0` |
| 37–41 | Target distances / 600 | 5 | Distance on each target ray; `1` when undetected |
| 42–43 | Local velocity / maximum speed | 2 | Forward and sideways motion relative to current aim |
| 44 | Cooldown / firing cooldown | 1 | `0` means ready to fire |
| 45 | Health / maximum health | 1 | Currently always `1` during normal play |
| 46–59 | Previous action encoded as three one-hot groups | 14 | Last movement, aim, and firing choices |
| **Total** | | **60** | |

“One-hot” means one position is `1` and the others are `0`. For example, three aim choices need three numbers; “hold aim” is `[0, 1, 0]`. The previous-action encoding contains **three ones**, one per action group.

The target flags are grouped together, followed by the target distances; they are not alternating flag/distance pairs. Their ray order is the order selected from the 32 wall rays: forward, +11.25°, +22.5°, −22.5°, −11.25°.

Most inputs are between 0 and 1. Local velocity can be between −1 and 1. Putting values on similar scales is called **normalization** and makes network inputs easier to handle.

### What the learner does not know

It receives no absolute position, absolute aim angle, full map, target coordinates, target velocity, bullet positions, stage number, or remaining episode time.

This is **partial observability**: the learner's information does not completely describe the world. An undetected target might be behind a wall, behind the learner, too far away, or between sensor rays.

The network has no recurrent memory. Including the previous action adds a small amount of recent context, but it does not provide a history of where the target was seen.

## 5. What the agent can do

Each action is a three-number array:

```python
[movement, aim, fire]
```

Gymnasium describes this as `MultiDiscrete([9, 3, 2])`.

| Component | Choices |
| --- | --- |
| Movement | `0` stand still; `1` forward; `2` forward-right; `3` right; `4` backward-right; `5` backward; `6` backward-left; `7` left; `8` forward-left |
| Aim | `0` turn left; `1` hold; `2` turn right |
| Fire | `0` hold fire; `1` request a shot |

For example:

```python
[2, 0, 1]  # Move diagonally forward-right, turn left, and request a shot.
```

Movement is relative to the learner's aim, not fixed screen directions. The code changes the aim first, then calculates movement using that updated orientation. Diagonal directions are normalized so moving diagonally is not faster.

There are `9 × 3 × 2 = 54` possible combinations. The network produces separate movement, aim, and fire probabilities rather than one list of 54 choices.

Firing has a 0.35-second cooldown, so repeatedly requesting fire does not create a bullet on every step. Each bullet deals 25 damage; a target starts with 100 health, requiring four hits.

## 6. Rewards: how the agent knows it did well

`RewardConfig` defines three scoring rules:

| Event | Reward |
| --- | --- |
| A bullet hits the target | `+1.0` |
| That hit kills the target | Additional `+3.0` |
| A bullet is fired while no target ray detected an enemy in the preceding observation | `−0.1` |

A normal hit earns `+1`. A killing hit earns `+4`. Four hits that kill one full-health target therefore earn **7 points before shot penalties**:

```text
1 + 1 + 1 + (1 + 3) = 7
```

An example episode with three full-health kills and ten penalized shots earns `3 × 7 − 10 × 0.1 = 20` points, assuming no other hits.

There is no direct reward for moving, seeing the target, or approaching it. Those behaviors matter only if they help produce hits and kills later.

The name `wasted_shot` is narrower than ordinary “missed shot”:

- A visible-target shot that misses receives no explicit miss penalty.
- A shot fired without detection can still hit later, receiving both the initial penalty and the eventual hit reward.
- Holding fire while the gun is cooling down creates no bullet and receives no wasted-shot penalty.
- Visibility for this penalty comes from before the current action, even if the learner turns during that action.

This illustrates **reward design**: the learner optimizes the score you actually implement. It does not automatically understand a human goal such as “use good combat tactics.”

## 7. The target and the difficulty stages

### The target is a scripted opponent

`Threat.update()` controls the target. At moving stages, it generally:

1. Picks a random destination, trying to find a clear straight path.
2. Moves toward it.
3. Uses eight short obstacle rays, like whiskers, to steer away from walls.
4. Pauses briefly after reaching a destination.
5. Chooses a replacement destination if it takes too long.

At the final stage, it also moves sideways when the learner appears to be aiming at it with clear line of sight. This evasion is programmed behavior. There is no pathfinding algorithm such as A*, and the target has no trained neural network.

### Curriculum learning: practice easy tasks first

**Curriculum learning** here means starting with an easy stationary target and gradually making the task harder.

| Stage | Target speed | Spawn angle relative to aim | Spawn distance | Extra behavior |
| --- | --- | --- | --- | --- |
| 0 | 0 | ±10° | 80–250 | Requires clear line of sight |
| 1 | 60 | ±30° | 100–350 | Requires clear line of sight |
| 2 | 120 | ±90° | 150–500 | May spawn behind an obstacle |
| 3 | 180 | Anywhere | 200–700 | Faster target, wider search |
| 4 | 180 | Anywhere | 250–900 | Prefers hidden spawns and evades aim |

Speeds are world units per simulated second. Spawn distances are requested sampling ranges; walls, arena bounds, and fallback logic constrain actual placements.

The sampler tries to satisfy a stage's rules. If it cannot, it relaxes the angle and visibility requirements, eventually falling back to a free position. A “prefers hidden” spawn is therefore not guaranteed to be hidden.

### When does the stage increase?

By default, `Curriculum.record()` promotes the learner after the mean reward of its most recent **100 qualifying completed episodes** at the current stage reaches **35**.

Five full-health kills with no penalties yield 35 points, which helps put that threshold in context. Promotion still uses the actual episode reward, including partial hits and penalties.

On promotion, the reward window clears. The final stage has no further promotion.

At stages above 0, each reset has a 20% chance to practice a randomly chosen earlier stage. These practice episodes do not count toward promotion. This is earlier-stage replay, not a stored experience replay buffer.

Stage changes apply when an environment next resets; they do not change the difficulty halfway through an active episode. Because training automatically resets finished environments before processing promotion statistics, an environment that just finished may already have started another episode at the old stage.

## 8. The neural network

`ActorCritic` is a **multilayer perceptron**, or **MLP**: a neural network made from fully connected layers.

Its default structure is:

```text
60 sensor values
        ↓
256 neurons + Tanh
        ↓
256 neurons + Tanh
        ├── Movement head: 9 scores
        ├── Aim head:      3 scores
        ├── Fire head:     2 scores
        └── Value head:    1 number
```

A “head” is simply an output branch. `Tanh` is a nonlinear activation that lets the network learn relationships more complex than a straight line.

The first two layers are shared. Their output feeds two roles:

| Role | Plain-language question | Output |
| --- | --- | --- |
| Actor / policy | “What should I do?” | Probabilities for movement, aim, and firing |
| Critic / value function | “How much future reward do I expect from here?” | A value estimate |

The critic estimates future reward, not the probability of winning.

During training, actions are sampled from the three categorical distributions. This allows **exploration**: trying different actions and discovering useful behavior. During normal watch mode, the model chooses the most probable action in each group. `--stochastic` makes watch mode sample instead.

The three choices share the same sensor processing, but their joint probability is factorized:

```text
P(move, aim, fire | observation)
    = P(move | observation)
    × P(aim | observation)
    × P(fire | observation)
```

The implementation sums the three log-probabilities, which is equivalent to taking the log of that product.

There is no image-processing CNN, recurrent LSTM, or attention mechanism in this model. Its input is already a compact numerical description of the surroundings.

## 9. How training works: PPO and GAE

### The basic training cycle

`train()` repeats this routine:

1. Use the current policy to play several environments.
2. Store observations, actions, action log-probabilities, rewards, value estimates, and episode boundaries.
3. Estimate which actions led to better or worse results than expected.
4. Update the actor and critic using that recent experience.
5. Collect a fresh batch and repeat.

By default, there are 8 environments and 256 steps per environment per collection cycle:

```text
8 × 256 = 2,048 transitions per PPO update
```

A **transition** is one observation/action/reward/next-observation interaction. A **rollout** is a collected sequence of transitions; it does not need to contain a whole episode. An **episode** is one complete practice session.

### Return and advantage

An immediate reward tells the learner what happened now. A **return** also accounts for rewards that arrive later.

With discount factor `gamma = 0.99`, a reward farther in the future receives progressively less weight:

```text
return ≈ reward now + 0.99 × next reward + 0.99² × following reward + ...
```

An **advantage** asks whether an outcome was better or worse than the critic expected. If the critic expected a return of 2 and the estimated return is 5, the advantage is approximately `+3`. The actor should tend to make that action more likely in similar situations.

### GAE: sharing credit across time

Suppose turning left helps the learner find a target and hit it several steps later. Learning needs to associate the later reward with earlier useful decisions. The code uses **Generalized Advantage Estimation**, or **GAE**, to combine prediction errors across time; its `lambda` setting controls their weighting. See the [original GAE paper](https://arxiv.org/abs/1506.02438).

Optional math, matching `compute_gae()` away from episode boundaries:

```text
delta[t]     = reward[t] + gamma × value[t+1] − value[t]
advantage[t] = delta[t] + gamma × lambda × advantage[t+1]
return[t]    = advantage[t] + value[t]
```

The code works backward through the rollout. Episode-boundary masks prevent one episode's rewards from carrying into another episode's advantage calculation.

### PPO: improving the policy without an oversized update

The project uses **Proximal Policy Optimization**, or **PPO**. It collects experience with the current policy, then performs several minibatch learning passes over that batch. The clipped objective limits the incentive for very large changes in action probability. See the [original PPO paper](https://arxiv.org/abs/1707.06347).

In `ppo_update()`, the main comparison is:

```text
ratio = probability of the recorded action under the updated policy
        / probability of that action when the data was collected
```

With default `clip = 0.2`, the objective uses a clipped ratio between 0.8 and 1.2. This is an objective constraint, not a guarantee that every probability or parameter stays within a hard limit.

The combined loss has three jobs:

| Part | Purpose |
| --- | --- |
| Policy loss | Encourage actions with positive advantage and discourage those with negative advantage |
| Value loss | Improve the critic's prediction of returns |
| Entropy bonus | Encourage the policy to keep some variety in its choices |

The implementation combines them as:

```text
loss = policy_loss + 0.5 × value_loss − 0.01 × entropy
```

PyTorch calculates gradients, and the Adam optimizer adjusts network weights. Gradient clipping limits the overall gradient norm to 0.5. Advantages are normalized within each minibatch.

Each default batch receives 4 epochs of training with minibatches of 512. With 2,048 transitions, that is 4 minibatches per epoch and 16 optimizer steps per completed rollout.

PPO here is **on-policy**: a batch is used for the current update, then replaced with freshly collected experience. The repository has no long-lived replay buffer of old transitions.

### Why time-limit handling matters

An episode ends because its practice time runs out, even though the character could continue earning reward. The training loop therefore adds `gamma × value(final_observation)` to the last reward used for learning, then prevents GAE from crossing into the reset episode. The reported episode score remains the raw game reward. This follows the distinction between termination and truncation described in [Gymnasium's time-limit documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## 10. Parallel environments, saving, and resuming

### Several copies of the game, one shared learner

`VecEnv` normally runs eight separate environment worker processes. They communicate with the main training process through pipes.

There is **one shared neural network**, not eight independently trained agents. It selects a batch of actions, gathers results from all game copies, and learns from their combined experience. The collector waits for every environment at each step, so this is synchronous collection.

With `--render`, the first environment runs in the main process with a Pygame window. The remaining environments stay in worker processes. The Matplotlib dashboard runs in another process and receives metrics through a queue.

The network and training tensors stay on the CPU in the current implementation. There is no CUDA/device option; the code sets PyTorch's CPU thread count to one and leaves other cores available for environment workers.

### Checkpoints

The default model path is:

```text
models/mlp_ppo_currilearning.pt
```

The spelling `currilearning` is the actual default filename.

`--model my_agent` uses `models/my_agent.pt`. Model names are resolved inside the repository's `models/` directory; passing another directory path does not select a different output folder.

Checkpoints contain the network weights, optimizer state, curriculum stage and reward window, input/action dimensions, hidden-layer size, and progress counters.

Training resumes automatically if the selected checkpoint exists. It restores the saved hidden size and uses the learning rate from the current command. To start a separate experiment, choose a new model name.

Saving happens every 25 PPO updates by default, and when the training loop exits through its cleanup block. The code writes a temporary file and then replaces the checkpoint file, reducing the chance of leaving a partially written checkpoint.

Resuming is not an exact replay of the previous run: the current game states, unfinished rollout, and random-generator states are not saved.

### Stopping

- The dashboard's **Stop after episode** button requests a stop after each environment has completed an episode following the request.
- `--total-episodes N` requests a stop when the cumulative completed-episode count reaches N. Parallel workers can make the final count exceed N; the count also includes episodes restored from a checkpoint.
- `Ctrl+C` is caught in the training loop and triggers checkpoint saving and cleanup.
- Closing the game rendering window or the dashboard window does not itself request that training stop.

If a stop interrupts a rollout, the code discards that incomplete rollout rather than performing one last PPO update on it. For a short training experiment, make sure episodes are long enough to allow at least one complete rollout before stopping.

## 11. How to run the project

These commands are derived from the imports and command-line parsers. Full installation and training were not executed while writing this guide.

### Set up a Python environment

Run these commands from the repository directory on Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy gymnasium torch pygame matplotlib
```

The code uses modern Python type syntax, so use Python 3.10 or newer. This is a source-syntax requirement, not a tested dependency compatibility matrix. The repository does not pin package versions.

Pygame is imported by `training.py` through `renderer.py`, even when training without windows. Matplotlib is imported inside the dashboard process and is needed when using that dashboard. GUI modes also require a working desktop/display environment.

### Play manually first

```bash
python renderer.py --stage 0
```

| Key | Action |
| --- | --- |
| W / S | Move forward / backward relative to aim |
| A / D | Move left / right relative to aim |
| Q / E or left / right arrows | Turn aim left / right |
| Space | Fire |
| R | Toggle sensor rays |
| F | Toggle the rendering frame-rate cap |
| Escape | Close the game window |

Try `--stage 4` to see the hardest target behavior.

### Train a new model

```bash
python training.py --model first_agent --total-episodes 200
```

This opens the metrics dashboard. Add `--render` to see one training environment.

For training without either window:

```bash
python training.py --model first_agent --total-episodes 200 --no-dashboard
```

For a smaller initial experiment with fewer workers:

```bash
python training.py --model quick_try --num-envs 2 --total-episodes 10 --no-dashboard
```

Ten episodes are enough to try the workflow, but do not establish that the agent has learned a good strategy.

### Resume and watch

```bash
# Resume the existing checkpoint; without an episode limit, training continues until stopped.
python training.py --model first_agent --no-dashboard

# Watch a saved model at the stage stored in its checkpoint.
python training.py --watch --model first_agent

# Watch the same model on stage 4 for three episodes.
python training.py --watch --model first_agent --stage 4 --episodes 3
```

Watch mode requires an existing checkpoint. It does not update the model. `--stage` in `training.py` affects watch mode only; it does not set the starting training stage.

Use these commands to see all available options:

```bash
python training.py --help
python renderer.py --help
```

### Common settings, translated

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-envs` | 8 | Number of game copies collecting experience |
| `--max-steps` | 1000 | Steps per episode |
| `--rollout-steps` | 256 | Steps collected per environment before an update |
| `--lr` | 0.0003 | Size of optimizer updates |
| `--gamma` | 0.99 | Weight given to future rewards |
| `--gae-lambda` | 0.95 | Weighting of longer chains of prediction errors in GAE |
| `--clip` | 0.2 | PPO clipping amount |
| `--ent-coef` | 0.01 | Strength of the exploration bonus |
| `--hidden-size` | 256 | Width of each shared hidden layer; restored from checkpoint on resume |
| `--promotion-reward` | 35 | Mean score needed for promotion |
| `--promotion-window` | 100 | Qualifying episodes required in the promotion window |
| `--replay-probability` | 0.2 | Chance of selecting an earlier stage on reset |
| `--save-every` | 25 | Completed PPO updates between periodic saves |
| `--seed` | 0 | Starting seed for random behavior |

## 12. How to read the training metrics

Judge progress using gameplay results together with learning diagnostics. A lower training loss alone does not prove better gameplay.

| Metric | Easy interpretation |
| --- | --- |
| Mean reward | Average episode score; higher is usually better within the same task settings |
| Mean kills | Average targets eliminated per episode |
| Mean shots | How frequently the learner actually fires |
| Hit rate | Total hits divided by total shots in the recent episode window |
| Mean wasted shots | Shots fired without prior sensor detection; lower can indicate more selective firing |
| Mean distance | Movement per episode; more movement is not automatically better |
| Steps to first detection | Average delay from target spawn to initial detection; includes the current undetected target's waiting time when an episode ends |
| Stage | Current curriculum level, which can differ from a worker's still-running episode stage |
| Entropy | Variety in the policy's action choices |
| Policy loss | The actor's optimization objective; not a direct gameplay score |
| Value loss | Error in predicting estimated returns |
| Approx KL | A diagnostic of policy change on the collected batch |
| Clip fraction | Fraction of recorded action probability ratios outside the PPO clipping range |
| Explained variance | How well stored value estimates explain estimated returns; nearer 1 indicates a better fit, while negative values indicate a poor fit |
| SPS | Aggregate environment steps collected per wall-clock second |

The episode metrics use the latest 100 completed episodes, including earlier-stage practice episodes and possibly episodes from before a promotion. The promotion calculation uses its own qualifying-episode window. Therefore, the displayed mean reward need not match the score used for promotion.

A score drop after promotion can reflect a harder task. Also, hit rate can be reduced by bullets cleared on a kill or still in flight at an episode boundary, so it is not a pure measure of aiming accuracy.

Without the dashboard, the program prints periodic summaries every 10 PPO updates once completed episode data is available.

## 13. Related knowledge and a learning path

### Concepts connected to this repository

| Concept | Meaning here |
| --- | --- |
| Reinforcement learning | Learning behavior from interactions and rewards |
| Supervised learning | Learning from supplied correct answers; this repo instead lets the agent discover useful actions |
| Policy | The rule, represented by the actor network, that maps observations to action probabilities |
| State vs. observation | The full game state includes everything; the observation exposes only selected sensor readings |
| Model-free learning | The learner improves from experience without learning an explicit predictor of the game's physics |
| Exploration vs. exploitation | Trying varied actions versus choosing actions already believed useful |
| Actor–critic | Learning an action policy alongside a predictor of future reward |
| Reward shaping | Adding intermediate feedback, such as hit rewards, instead of rewarding only a final outcome |
| Curriculum learning | Changing task difficulty as the learner improves |
| Ray casting | Sending a mathematical line through the map to find the first intersection |
| AABB | Axis-aligned bounding box: the rectangle representation used for walls |
| Coordinate transforms | Converting forward/sideways movement into screen x/y movement |
| Backpropagation | Computing how network weights contributed to the training loss |
| Generalization | Performing well beyond the particular map and situations used during training |

The simulator itself knows object positions to calculate physics, sensing, and target rules. “Model-free” describes the learner's method; it does not mean the environment has no rules or internal state.

### A practical reading order

1. **Play the game.** Turn the sensor rays on and observe when the target becomes detectable.
2. **Read `RewardConfig` and `HuntEnv.step()`.** Understand the task being rewarded before studying the optimizer.
3. **Read `Agent.sense()`, `Agent.observation()`, and `Agent.apply_action()`.** Connect the learner's inputs and outputs to visible behavior.
4. **Read `Threat.update()` and `STAGES`.** Separate programmed difficulty from learned behavior.
5. **Read `ActorCritic`.** Follow the 60 inputs through the shared layers into the four output branches.
6. **Read the rollout loop in `train()`.** Follow how game experience becomes a training batch.
7. **Read `compute_gae()` and `ppo_update()`.** Study advantage estimation and policy optimization after the game loop is clear.
8. **Read `VecEnv` and checkpoint functions.** These support speed and persistence rather than defining the learning objective.

Useful small experiments are to compare stage 0 and stage 4 manually, compare a saved model's deterministic and stochastic watch modes, or train separate model names with one setting changed. Record reward, kills, and hit rate under the same evaluation conditions before drawing conclusions.

## 14. Limitations and useful improvements

These observations describe the current implementation. Suggested improvements have not been applied to the source code.

| Current behavior or limitation | Why it matters | Possible improvement |
| --- | --- | --- |
| Target does not attack; learner cannot die | Results demonstrate hunting, aiming, and navigation rather than a full combat duel | Add incoming attacks, damage, and survival objectives |
| One fixed map | Good performance may depend on the training layout | Evaluate on different maps or randomize layouts |
| Sparse target rays and no recurrent memory | Targets can fall between rays or disappear without a remembered location | Compare denser sensing, stacked observations, or a recurrent policy |
| No target velocity or bullet observations | Leading a moving target must be inferred from limited context | Experiment with extra sensor history or carefully chosen additional observations |
| Wasted-shot penalty uses the previous observation | Turning onto a target and firing in the same step can still be penalized | Decide whether to assess visibility before or after the action |
| Health input is constant | One input currently carries no changing information | Remove it for this task or implement damage mechanics |
| No pinned dependencies, automated test suite, or evaluation report | Setup and performance are harder to reproduce | Add dependency versions, focused environment checks, and repeatable evaluation |
| Dashboard history is kept in memory only | Closing a run loses its graphs | Save episode metrics and settings to files |
| Resume restores learning state but not full simulation state | Continuing a checkpoint changes the exact trajectory of experience | Save configuration and random states if exact reproducibility is needed |
| Argument values have little validation | Zero workers or invalid training sizes can fail; a one-sample minibatch can make `adv.std()` invalid | Validate positive counts and suitable minibatch sizes |
| Stop discards an incomplete rollout | Some collected experience is counted but not used for a PPO update | Optionally support a correctly sized final update |
| Bullet range is checked after movement | With custom settings, a bullet may collide slightly beyond its remaining range during the final step | Limit its swept path to the remaining range |

There is no trained checkpoint or benchmark result in this checkout. The source shows how learning is intended to work; it does not establish that the default configuration reaches the final stage or a particular skill level.

## 15. What was verified for this guide

The analysis covered all four Python files. Lightweight checks confirmed:

- All four files pass Python syntax compilation.
- The default observation contains 60 `float32` values and the checked sample stays inside the declared range.
- There are five target-sensing rays at the angles listed above.
- A directly visible target is detected, and an intervening wall blocks detection in a simple scenario.
- Four 25-damage bullet hits remove a target's 100 health.

The available interpreter during analysis was Python 3.13.16 with NumPy installed. Gymnasium, PyTorch, Pygame, and Matplotlib were unavailable, so full environment integration, window rendering, multiprocessing, and PPO training were not run. Setup commands and training behavior above were checked against the source rather than demonstrated end to end.

The external links in the learning sections provide background on PPO, GAE, and Gymnasium time limits. Repository-specific details come from the local implementation.
