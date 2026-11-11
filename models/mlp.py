"""MLP actor-critic model used by the PPO trainer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from entities import ACTION_DIMS, OBSERVATION_SIZE


def _layer_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Shared MLP trunk feeding one categorical head per action component plus a value head."""

    def __init__(self, obs_size: int = OBSERVATION_SIZE, action_dims=ACTION_DIMS, hidden_size: int = 256):
        super().__init__()
        gain = float(np.sqrt(2.0))
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_size, hidden_size), gain),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size), gain),
            nn.Tanh(),
        )
        self.action_heads = nn.ModuleList(_layer_init(nn.Linear(hidden_size, n), 0.01) for n in action_dims)
        self.value_head = _layer_init(nn.Linear(hidden_size, 1), 1.0)

    def _forward(self, obs: torch.Tensor) -> tuple[list[Categorical], torch.Tensor]:
        hidden = self.trunk(obs)
        return [Categorical(logits=head(hidden)) for head in self.action_heads], self.value_head(hidden).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dists, value = self._forward(obs)
        if deterministic:
            actions = torch.stack([d.probs.argmax(-1) for d in dists], dim=-1)
        else:
            actions = torch.stack([d.sample() for d in dists], dim=-1)
        log_prob = sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        return actions, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dists, value = self._forward(obs)
        log_prob = sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return log_prob, entropy, value
