"""Combat PPO training with generated maps, curriculum and the live dashboard.

Train: python training_random_map.py --map-seeds 9 [--render]
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

Default checkpoint: models/mlp_ppo_random_map.pt. Training automatically resumes
it if present. --model NAME can fine-tune a compatible hunting/combat checkpoint;
that checkpoint will be overwritten. Other trainers keep their fixed maps.
"""

from training import parse_args as parse_training_args
from training import train as train_ppo
from training import watch as watch_ppo

DEFAULT_MODEL_NAME = "mlp_ppo_random_map.pt"


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
    if arguments.watch:
        watch(arguments)
    else:
        train(arguments)
