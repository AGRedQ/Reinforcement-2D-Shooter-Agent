"""Explicit device selection and portable checkpoints for training/evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is inaccessible. Check nvidia-smi, the PyTorch CUDA build, and GPU permissions; use --device cpu explicitly for CPU training.")
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Only CPU and CUDA devices are supported")
    return device


def to_device(observations, device):
    if isinstance(observations, torch.Tensor):
        return observations.to(device)
    return {key: value.to(device) for key, value in observations.items()}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def configuration(args):
    return {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, (range, tuple)) else value
            for key, value in vars(args).items()}


def load_policy(path: Path, device="cpu"):
    from entities import ACTION_DIMS, ANGULAR_OBSERVATION_SIZE, OBSERVATION_SIZE
    from models.mlp import ActorCritic
    from models.recurrent import RecurrentActorCritic

    saved = torch.load(path, map_location="cpu", weights_only=True)
    architecture = saved.get("architecture", "mlp")
    angular = saved.get("enemy_sensing", "rays_v1") == "angular_regions_v1"
    if tuple(saved["action_dims"]) != ACTION_DIMS or saved["obs_size"] != (ANGULAR_OBSERVATION_SIZE if angular else OBSERVATION_SIZE):
        raise ValueError("Checkpoint observations/actions do not match this environment")
    if architecture == "window_recurrent_deep_sets_v1":
        if not angular:
            raise ValueError("Recurrent checkpoints require angular observations")
        policy = RecurrentActorCritic(**saved["recurrent_config"])
    elif architecture in ("mlp", "deep_sets_v1"):
        policy = ActorCritic(hidden_size=saved["hidden_size"], bullet_sensing=architecture == "deep_sets_v1", angular_regions=angular)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    policy.load_state_dict(saved["model"])
    return policy.to(device), saved
