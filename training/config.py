"""Required YAML defaults shared by training entry points."""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def load_defaults(name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = config["training"][name]
    if not isinstance(settings, dict):
        raise ValueError(f"training.{name} must be a mapping in {path}.")
    return settings


def parse_configured_args(
    parser: argparse.ArgumentParser, settings: dict[str, Any]
) -> argparse.Namespace:
    arguments = sys.argv[1:]
    if not arguments or (arguments[0].startswith("--") and arguments[0] != "--help"):
        arguments = [settings["command"], *arguments]
    return parser.parse_args(arguments)
