"""Train the agent against a moving threat that shoots back.

Train: python training_threat.py [--model NAME] [--total-episodes N] [--render]
Watch: python training_threat.py --watch [--model NAME] [--stage N]
Play:  python renderer.py --shoot-back [--stage N]

Uses the shared hunting curriculum, observations, actions and PPO implementation.
Threats fire every 0.8 seconds with clear line of sight, using slower, 10-damage
bullets. Walls block shots. Agent damage costs 0.04 reward per health point;
death costs another 3 reward and terminates the episode. Surviving until the time
limit truncates it. Threats respawn after kills; the agent resets each episode.

The default checkpoint is models/mlp_ppo_threat.pt, separate from hunting models.
An existing compatible hunting checkpoint can also be used with --model NAME
to continue training it in combat mode (this overwrites that checkpoint).

Training opens a local Streamlit dashboard with interactive charts, damage and
death metrics, CSV export, and a Stop-after-episode button. Use --no-dashboard
to disable it. The dashboard server closes when training exits.
"""

from training import parse_args as parse_training_args
from training import train as train_ppo
from training import watch as watch_ppo

DEFAULT_MODEL_NAME = "mlp_ppo_threat.pt"


def parse_args(argv=None):
    return parse_training_args(argv, description=__doc__, default_model=DEFAULT_MODEL_NAME)


def train(args) -> None:
    train_ppo(args, shoot_back=True)


def watch(args) -> None:
    watch_ppo(args, shoot_back=True)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.watch:
        watch(arguments)
    else:
        train(arguments)
