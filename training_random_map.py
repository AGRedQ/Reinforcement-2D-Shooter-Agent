"""Combat PPO training with generated maps, curriculum and the live dashboard.

Train: python training_random_map.py --map-seeds 9 [--render]
Deep Sets: python training_random_map.py --bullet-sensing --map-seeds 9 [--render]
Watch Deep Sets: python training_random_map.py --watch --bullet-sensing --map-seeds '[2,7,11]'
Fight Deep Sets: python training_random_map.py --play-against --bullet-sensing --map-seeds '[4]'
Fight MLP: python training_random_map.py --play-against --model mlp_ppo_raised_damage
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

Default checkpoint: checkpoints/mlp_ppo_raised_damage.pt. Training automatically resumes
it if present. --model NAME can fine-tune a compatible hunting/combat checkpoint;
that checkpoint will be overwritten.

--bullet-sensing selects a separate default: checkpoints/deep_sets_ppo_random_map.pt.
This starts a new policy, not a migration of the existing MLP. Resume/watch with
--bullet-sensing, or --model deep_sets_ppo_random_map; checkpoint metadata restores
the architecture automatically. Existing MLP checkpoints remain supported.

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
"""

from models.ppo import parse_args as parse_training_args
from models.ppo import train as train_ppo
from models.ppo import watch as watch_ppo

DEFAULT_MODEL_NAME = "mlp_ppo_raised_damage.pt"


def parse_args(argv=None):
    return parse_training_args(
        argv, description=__doc__, default_model=DEFAULT_MODEL_NAME, random_maps=True
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
