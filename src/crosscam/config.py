"""Configuration loading and validation."""
import math
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
    seen_edges: set[tuple[str, str]] = set()
    for edge in config.get("transitions", []):
        source = edge.get("from")
        target = edge.get("to")
        if source not in known or target not in known:
            raise ValueError(f"Transition references unknown camera: {edge}")
        key = (source, target)
        if key in seen_edges:
            raise ValueError(f"Duplicate directed camera-pair edge: {source} -> {target}")
        seen_edges.add(key)

        minimum = float(edge.get("min_seconds", 0))
        maximum = float(edge.get("max_seconds", 0))
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("Transition times must be finite")
        if minimum < 0:
            raise ValueError("Transition min_seconds must be non-negative")
        if maximum < minimum:
            raise ValueError("Transition max_seconds must be >= min_seconds")
        prior = edge.get("prior")
        if prior is not None and (
            not math.isfinite(float(prior)) or not 0 <= float(prior) <= 1
        ):
            raise ValueError("Transition prior must be between 0 and 1")

    seen_relationships: set[tuple[str, str]] = set()
    allowed_states = {"known_overlap", "known_non_overlap", "unknown"}
    for relation in config.get("camera_relationships", []):
        pair = relation.get("cameras", [])
        if len(pair) != 2 or pair[0] not in known or pair[1] not in known:
            raise ValueError(f"Camera relationship must reference two known cameras: {relation}")
        if pair[0] == pair[1]:
            raise ValueError("Camera relationships must connect different cameras")
        key = tuple(sorted(pair))
        if key in seen_relationships:
            raise ValueError(f"Duplicate camera relationship: {key}")
        seen_relationships.add(key)
        if relation.get("state") not in allowed_states:
            raise ValueError(f"Invalid camera relationship state: {relation.get('state')}")

    association = config.get("association", {})
    accept = float(association.get("accept_threshold", 0.78))
    defer = float(association.get("defer_threshold", 0.55))
    margin = float(association.get("min_margin", 0.0))
    horizon = float(association.get("widen_search_horizon_seconds", 180.0))
    time_decay = float(association.get("time_decay_seconds", 90.0))
    widen_score = float(association.get("widen_if_best_score_below", accept))
    if not 0 <= defer <= accept <= 1:
        raise ValueError("Require 0 <= defer_threshold <= accept_threshold <= 1")
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("association.min_margin must be between 0 and 1")
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError("association.widen_search_horizon_seconds must be non-negative")
    if not math.isfinite(time_decay) or time_decay <= 0:
        raise ValueError("association.time_decay_seconds must be greater than zero")
    if not math.isfinite(widen_score) or not 0 <= widen_score <= 1:
        raise ValueError("association.widen_if_best_score_below must be between 0 and 1")
