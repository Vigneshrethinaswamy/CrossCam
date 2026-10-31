from copy import deepcopy
from pathlib import Path

import pytest

from crosscam.config import load_config, validate_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_is_valid():
    config = load_config(CONFIG_PATH)
    assert config["association"]["widen_search_horizon_seconds"] == 180


def test_duplicate_directed_transition_is_rejected():
    config = load_config(CONFIG_PATH)
    broken = deepcopy(config)
    broken["transitions"].append(deepcopy(broken["transitions"][0]))
    with pytest.raises(ValueError, match="Duplicate directed"):
        validate_config(broken)


def test_duplicate_undirected_relationship_is_rejected():
    config = load_config(CONFIG_PATH)
    broken = deepcopy(config)
    broken["camera_relationships"] = [
        {"cameras": ["cam_01", "cam_02"], "state": "known_overlap"},
        {"cameras": ["cam_02", "cam_01"], "state": "known_non_overlap"},
    ]
    with pytest.raises(ValueError, match="Duplicate camera relationship"):
        validate_config(broken)


def test_unknown_relationship_state_is_rejected():
    config = load_config(CONFIG_PATH)
    broken = deepcopy(config)
    broken["camera_relationships"] = [
        {"cameras": ["cam_01", "cam_02"], "state": "maybe"}
    ]
    with pytest.raises(ValueError, match="Invalid camera relationship"):
        validate_config(broken)



def test_widening_horizon_is_required_in_config():
    config = load_config(CONFIG_PATH)
    broken = deepcopy(config)
    del broken["association"]["widen_search_horizon_seconds"]
    with pytest.raises(ValueError, match="widen_search_horizon_seconds is required"):
        validate_config(broken)
