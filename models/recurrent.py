"""Past-window (Bi)LSTM between Deep Sets features and the existing decision MLP.

Both directions see only an already observed prefix ending at the action time.
There is no lookahead, persistent hidden state, or stale rollout hidden state.
PPO recomputes the entire window, including its bullet encodings, for gradients.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from entities import ANGULAR_OBSERVATION_SIZE
from models.mlp import ActorCritic, Observation, batch_observations


class ObservationHistory:
    """Raw observations, oldest first; episode resets cannot leak into a window."""

    def __init__(self, observations: list[Observation], context_length: int):
        if context_length < 1 or not observations:
            raise ValueError("History needs observations and a positive context length")
        self.context_length = context_length
        self.windows = [deque([obs], maxlen=context_length) for obs in observations]

    def snapshot(self) -> list[tuple[Observation, ...]]:
        return [tuple(window) for window in self.windows]

    def advance(self, observations: list[Observation], dones) -> None:
        if len(observations) != len(self.windows) or len(dones) != len(self.windows):
            raise ValueError("One observation and done flag are required per environment")
        for window, obs, done in zip(self.windows, observations, dones):
            if done:
                window.clear()
            window.append(obs)

    def terminal_window(self, index: int, final_obs: Observation) -> tuple[Observation, ...]:
        """Append the terminal observation BEFORE clearing the episode's history."""
        return tuple((*self.windows[index], final_obs)[-self.context_length:])


def batch_histories(windows: list[tuple[Observation, ...]]) -> dict[str, torch.Tensor]:
    """Right-pad time and bullets independently; lengths exclude temporal padding."""
    if not windows or any(not window for window in windows):
        raise ValueError("Cannot batch an empty history")
    lengths = torch.tensor([len(window) for window in windows], dtype=torch.long)
    time_size = int(lengths.max())
    empty = {
        "state": np.zeros(ANGULAR_OBSERVATION_SIZE, dtype=np.float32),
        "bullets": np.empty((0, 4), dtype=np.float32),
    }
    flat = [obs for window in windows for obs in (*window, *((empty,) * (time_size - len(window))))]
    batch = batch_observations(flat)
    if not isinstance(batch, dict):
        raise TypeError("The recurrent experiment requires angular Deep Sets observations")
    return {
        **{key: value.reshape(len(windows), time_size, *value.shape[1:]) for key, value in batch.items()},
        "lengths": lengths,
    }


class RecurrentActorCritic(ActorCritic):
    """One recurrent layer before the unchanged two-layer decision MLP.

    A zero-initialized residual projection preserves the supplied checkpoint's
    exact initial policy/value. It then learns a temporal correction. This is
    an additive memory experiment, avoiding replacement of pretrained features.
    h_n contains both full-window endpoints, unlike output[:, -1]'s backward half.
    """

    def __init__(self, hidden_size: int = 256, lstm_hidden_size: int = 48,
                 context_length: int = 32, bidirectional: bool = True):
        if min(hidden_size, lstm_hidden_size, context_length) < 1:
            raise ValueError("Layer widths and context length must be positive")
        super().__init__(hidden_size=hidden_size, bullet_sensing=True, angular_regions=True)
        self.context_length = context_length
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.memory = nn.LSTM(96, lstm_hidden_size, num_layers=1,
                              batch_first=True, bidirectional=bidirectional)
        self.memory_projection = nn.Linear(lstm_hidden_size * (2 if bidirectional else 1), 96)
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)

    def _features(self, obs) -> torch.Tensor:
        if not isinstance(obs, dict) or "lengths" not in obs or obs["state"].ndim != 3:
            raise TypeError("Recurrent policy requires batch_histories() input")
        lengths = obs["lengths"]
        if torch.any(lengths < 1) or torch.any(lengths > obs["state"].shape[1]):
            raise ValueError("Invalid temporal lengths")
        if obs["state"].shape[1] > self.context_length:
            raise ValueError("History exceeds this checkpoint's context length")
        features = super()._features(obs)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.memory(packed)
        memory = torch.cat((hidden[-2], hidden[-1]), dim=-1) if self.bidirectional else hidden[-1]
        current = features[torch.arange(len(lengths), device=features.device), lengths - 1]
        return current + self.memory_projection(memory)

    def warm_start(self, state_dict: dict) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed = {name for name in self.state_dict() if name.startswith(("memory.", "memory_projection."))}
        if set(missing) != allowed or unexpected:
            raise ValueError(f"Incompatible warm start: missing={missing}, unexpected={unexpected}")

    def configuration(self) -> dict:
        return {
            "hidden_size": self.trunk[0].out_features,
            "lstm_hidden_size": self.lstm_hidden_size,
            "context_length": self.context_length,
            "bidirectional": self.bidirectional,
        }
