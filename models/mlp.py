"""MLP actor-critic model used by the PPO trainer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from entities import ACTION_DIMS, ANGULAR_OBSERVATION_SIZE, BULLET_FEATURE_SIZE, OBSERVATION_SIZE

Observation = np.ndarray | dict[str, np.ndarray]
TensorObservation = torch.Tensor | dict[str, torch.Tensor]


def batch_observations(observations: list[Observation]) -> TensorObservation:
    """Pad only to this batch's largest set; never truncate projectiles."""
    first = observations[0]
    if isinstance(first, np.ndarray):
        arrays = []
        for observation in observations:
            if not isinstance(observation, np.ndarray):
                raise TypeError("Cannot mix flat and bullet observations in one batch")
            arrays.append(observation)
        return torch.as_tensor(np.stack(arrays))
    states, bullets = [], []
    for observation in observations:
        if not isinstance(observation, dict):
            raise TypeError("Cannot mix flat and bullet observations in one batch")
        states.append(observation["state"])
        bullets.append(torch.as_tensor(observation["bullets"]))
    counts = torch.tensor([len(items) for items in bullets])
    max_count = max(1, int(counts.max().item()))
    padded = torch.zeros(len(bullets), max_count, BULLET_FEATURE_SIZE)
    for i, items in enumerate(bullets):
        padded[i, :len(items)] = items
    return {
        "state": torch.as_tensor(np.stack(states)),
        "bullets": padded,
        "bullet_mask": torch.arange(max_count)[None, :] < counts[:, None],
    }


def select_observations(observations: TensorObservation, indices: torch.Tensor) -> TensorObservation:
    if isinstance(observations, torch.Tensor):
        return observations[indices]
    return {key: value[indices] for key, value in observations.items()}


def _layer_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class BulletEncoder(nn.Module):
    """Shared per-bullet MLP followed by masked, order-independent pooling."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(BULLET_FEATURE_SIZE, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
        )
        self.projection = nn.Sequential(nn.Linear(65, 32), nn.Tanh())

    def forward(self, bullets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(bullets)
        counts = mask.sum(dim=-1, keepdim=True)
        mean = embeddings.masked_fill(~mask[..., None], 0.0).sum(dim=-2) / counts.clamp_min(1)
        maximum = embeddings.masked_fill(~mask[..., None], -torch.inf).amax(dim=-2)
        maximum = torch.where(counts > 0, maximum, torch.zeros_like(maximum))
        # Bounded count preserves cardinality without assuming a maximum set size.
        count_feature = counts.to(embeddings.dtype) / (counts + 1)
        summary = self.projection(torch.cat([mean, maximum, count_feature], dim=-1))
        return torch.where(counts > 0, summary, torch.zeros_like(summary))


class ActorCritic(nn.Module):
    """Shared MLP trunk feeding one categorical head per action component plus a value head."""

    def __init__(
        self, obs_size: int = OBSERVATION_SIZE, action_dims=ACTION_DIMS,
        hidden_size: int = 256, bullet_sensing: bool = False,
        angular_regions: bool = False,
    ):
        super().__init__()
        self.bullet_sensing = bullet_sensing
        self.angular_regions = angular_regions
        self.obs_size = ANGULAR_OBSERVATION_SIZE if angular_regions else obs_size
        self.bullet_encoder = BulletEncoder() if bullet_sensing else None
        gain = float(np.sqrt(2.0))
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(self.obs_size + (32 if bullet_sensing else 0), hidden_size), gain),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size), gain),
            nn.Tanh(),
        )
        self.action_heads = nn.ModuleList(_layer_init(nn.Linear(hidden_size, n), 0.01) for n in action_dims)
        self.value_head = _layer_init(nn.Linear(hidden_size, 1), 1.0)

    def _features(self, obs: TensorObservation) -> torch.Tensor:
        if self.bullet_encoder is not None:
            if not isinstance(obs, dict):
                raise TypeError("Deep Sets policy requires state, bullets, and bullet_mask")
            return torch.cat([
                obs["state"], self.bullet_encoder(obs["bullets"], obs["bullet_mask"]),
            ], dim=-1)
        if not isinstance(obs, torch.Tensor):
            raise TypeError("MLP policy requires flat observations")
        return obs

    def _forward(self, obs: TensorObservation) -> tuple[list[Categorical], torch.Tensor]:
        hidden = self.trunk(self._features(obs))
        return [Categorical(logits=head(hidden)) for head in self.action_heads], self.value_head(hidden).squeeze(-1)

    def value(self, obs: TensorObservation) -> torch.Tensor:
        return self.value_head(self.trunk(self._features(obs))).squeeze(-1)

    def act(self, obs: TensorObservation, deterministic: bool = False):
        dists, value = self._forward(obs)
        if deterministic:
            actions = torch.stack([d.probs.argmax(-1) for d in dists], dim=-1)
        else:
            actions = torch.stack([d.sample() for d in dists], dim=-1)
        log_prob = torch.stack([d.log_prob(actions[..., i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
        return actions, log_prob, value

    def evaluate(self, obs: TensorObservation, actions: torch.Tensor):
        dists, value = self._forward(obs)
        log_prob = torch.stack([d.log_prob(actions[..., i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
        entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)
        return log_prob, entropy, value
