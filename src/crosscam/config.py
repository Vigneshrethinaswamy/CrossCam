"""Configuration loading and validation."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    cameras = config.get("cameras", [])
    camera_ids = [camera.get("id") for camera in cameras]
    if not camera_ids or any(not camera_id for camera_id in camera_ids):
        raise ValueError("At least one camera with a non-empty id is required")
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Camera ids must be unique")

    known = set(camera_ids)
    for edge in config.get("transitions", []):
        if edge.get("from") not in known or edge.get("to") not in known:
            raise ValueError(f"Transition references unknown camera: {edge}")
        if edge.get("min_seconds", 0) < 0:
            raise ValueError("Transition min_seconds must be non-negative")
        if edge.get("max_seconds", 0) < edge.get("min_seconds", 0):
            raise ValueError("Transition max_seconds must be >= min_seconds")

    association = config.get("association", {})
    accept = association.get("accept_threshold", 0.78)
    defer = association.get("defer_threshold", 0.55)
    if not 0 <= defer <= accept <= 1:
        raise ValueError("Require 0 <= defer_threshold <= accept_threshold <= 1")
