"""Combat PPO training with generated maps, curriculum and the live dashboard.

Train: python training_random_map.py --map-seeds 9 [--render]
Flag exploration: python training_random_map.py --exploration-bonus --map-seeds '[4]' --render
Legacy sensing: python training_random_map.py --no-bullet-sensing --no-angular-regions --map-seeds 9
Watch Deep Sets: python training_random_map.py --watch --bullet-sensing --map-seeds '[2,7,11]'
Fight Deep Sets: python training_random_map.py --play-against --bullet-sensing --map-seeds '[4]'
Fight MLP: python training_random_map.py --play-against --model mlp_ppo_raised_damage
Angular regions: python training_random_map.py --angular-regions --bullet-sensing --map-seeds 9 --render
Watch angular: python training_random_map.py --watch --model angular_deep_sets_ppo
Self-play: python training_random_map.py --self-play --map-seeds '[4]' --render
Watch duel: python training_random_map.py --self-play --watch --map-seeds '[4]' --stochastic
Watch: python training_random_map.py --watch --map-seeds '[2,7,11]'
Play:  python renderer.py --shoot-back --map-seeds '[5]'

Map generation uses stratified random rectangular obstacles in a 3x3 grid.
Each map has 6-9 obstacles with randomized sizes and positions. Slot margins
leave connected corridors, at least 48 units wide (default bodies are 28 units).

--map-seeds N       loops seeds 0 through N, inclusive (default: 9).
--map-seeds '[N]'   plays only seed N.
--map-seeds '[N,M]' loops exactly the listed seeds, in order; duplicates are allowed.
Quote arrays containing spaces. Each worker starts at the first seed and advances
on episode reset, not threat respawn. Different workers have independent cycles
and spawn RNGs. --seed controls spawns/actions, not the selected map geometry.
Watch follows the same cycle. Each new run, including checkpoint resume, starts
the map cycle from the first seed; pass --map-seeds again to retain your selection.

Includes shooting threats, damage/death, threat respawns, curriculum replay,
PPO updates, --render, and the Streamlit/Plotly dashboard with graceful Stop and
CSV export. --no-dashboard disables the dashboard.

Every training run also creates a unique folder under reports/ (override with
--report-dir PATH). metrics.csv is flushed every 10 completed episodes, even
without the dashboard. Metrics use the last up to 100 episodes across workers,
including replay; PPO statistics are from the latest completed update.
On normal stop or Ctrl+C, remaining episodes are saved as a partial sample and
report.html provides an offline summary and charts. Resumes create separate
reports with cumulative episode numbers. This does not change checkpoint saving.

Default checkpoint: checkpoints/angular_deep_sets_ppo.pt. Training automatically resumes
it if present. --model NAME can fine-tune a compatible hunting/combat checkpoint;
that checkpoint will be overwritten.

Bullet sensing and angular regions are enabled by default for new policies.
--no-bullet-sensing and --no-angular-regions disable them independently.
With both disabled the default is mlp_ppo_raised_damage.pt; bullet sensing alone
uses deep_sets_ppo_random_map.pt, and angular regions alone uses angular_mlp_ppo.pt.
Existing checkpoints restore their architecture/sensing from metadata when no
explicit sensing switches are supplied. Explicit incompatible switches produce
an error rather than migrating weights. Use --model NAME for a separate experiment.

--exploration-bonus opts scripted-enemy training into anchor-relative travel
shaping, without adding observation inputs. An invisible anchor is placed at
spawn, after each kill, and every 5 simulated seconds. New maximum displacement
earns 0.0005 per pixel up to 100 pixels (+0.05 per anchor), capped at +0.5 per
episode across all anchors. Only transitions with no enemy detected at either
end earn the bonus; terminal deaths earn none. Combat movement still advances
the maximum, so it cannot be claimed later. Returning incurs no penalty.
The interval resets after each new anchor; reaching 100 pixels does not reset it.
Repeated routes can pay after interval resets: this rewards travel, not novelty.
Reports/dashboard show exploration reward separately; total reward includes it.
The switch must be supplied again on resume. Self-play/watch/play do not use it.

Deep Sets senses all live hostile bullets within the continuous 60-degree viewing
cone and 600-unit range, occluded by walls. Each bullet supplies normalized local
position and relative velocity. A shared 4-32-32 MLP, masked mean/max pooling and
bounded count feed a 65-32 projection, concatenated with the existing observations.
Empty sets produce a zero summary. Batches pad dynamically without truncating
bullets; all encoder weights train jointly with PPO. Damage and rewards are unchanged.

--play-against loads the checkpoint for inference only: you control a Player
instead of the scripted threat. WASD moves in screen space, mouse aims, and hold
left click to shoot (Agent defaults: speed 220, acceleration 12, health 100,
25 damage, 900 bullet speed/range, 0.35-second cooldown; independent of curriculum).
Normal view shows the agent/bullets only within
your 60-degree cone, 600-unit range and line of sight. H toggles wallhack plus
agent rays; --player-hacks starts enabled. Hacks never change AI observations.
Player deaths respawn as in hunting; agent death/time limit resets the round.
--episodes sets the round count. No training or checkpoint writes occur.

Angular sensing gives new policies six gap-free enemy regions with boundaries
-30,-15,-5,0,5,15,30 degrees relative to aim. Enemy circle overlap and wall
occlusion determine each region's presence and nearest visible surface distance.
A separate zero-width center ray supplies its own flag/distance (alignment, not
a shooting gate). Wall rays, bullet sensing, weapons and rewards are unchanged.
The base observation grows from 60 to 64 values: walls 32, region flags 6,
region distances 6, center flag/distance 2, velocity 2, cooldown/health 2,
previous action 14. With Deep Sets the decision MLP receives 96 values.
Defaults: angular_deep_sets_ppo.pt with --bullet-sensing, otherwise angular_mlp_ppo.pt.
This starts fresh; legacy checkpoints keep their old ray observations. Resume,
watch and --play-against restore sensing from metadata with --model NAME.

--self-play trains 1v1 using two identical Agent bodies, angular regions and Deep
Sets bullet sensing on both sides. A new run copies weights (not optimizer or
training progress) from --self-play-init angular_deep_sets_ppo by default, leaving
that source unchanged. The separate output, self_play_ppo.pt, resumes automatically.
Missing/incompatible source checkpoints produce an error; no silent fresh start.
Curriculum and replay are disabled. Both fighters spawn separated with random
headings; neither receives privileged observations or world coordinates.
The red opponent samples actions from a frozen policy, refreshed from the blue
learner every --opponent-refresh PPO updates (default 50), only between rollouts.
Only blue supplies PPO training data. Refreshes can occur during a round.
First death ends the round; simultaneous deaths and time limits are terminal
draws, without respawns or timeout value bootstrapping. Existing hit, kill,
damage, death and wasted-shot rewards apply; no exploration or memory is added.
Reports/dashboard include win/loss/draw rates and opponent generation. Rates are
against changing opponents, not a fixed benchmark. Checkpoints retain the frozen
opponent and refresh schedule for resume/watch. Use --stochastic when watching
to match training action sampling; otherwise both sides use argmax.
R/T/F retain their preview controls. --play-against is not supported for duels.
"""

from models.ppo import parse_args as parse_training_args
from models.ppo import train as train_ppo
from models.ppo import watch as watch_ppo

DEFAULT_MODEL_NAME = "mlp_ppo_raised_damage.pt"


def parse_args(argv=None):
    return parse_training_args(
        argv, description=__doc__, default_model=DEFAULT_MODEL_NAME, random_maps=True,
        default_sensing=True,
    )


def train(args) -> None:
    train_ppo(args, shoot_back=True, map_seeds=args.map_seeds)


def watch(args) -> None:
    watch_ppo(args, shoot_back=True, map_seeds=args.map_seeds)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.watch or arguments.play_against:
        watch(arguments)
    else:
        train(arguments)
