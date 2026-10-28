from pathlib import Path

import numpy as np
import pytest

from crosscam.appearance import hsv_histogram
from crosscam.association import associate
from crosscam.config import load_config
from crosscam.models import Tracklet
from crosscam.topology import CameraTopology

from scripts.associate_tracklets import parse_offsets, rank


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_hsv_histogram_is_normalized_and_deterministic():
    image = np.zeros((20, 10, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    feature = hsv_histogram(image)
    assert len(feature) == 64
    assert sum(feature) == pytest.approx(1.0)
    assert hsv_histogram(image) == feature


def test_hsv_histogram_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        hsv_histogram(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        hsv_histogram(np.zeros((10, 10, 3), dtype=np.uint8), bins=1)


def test_candidate_association_uses_time_topology_and_appearance():
    topology = CameraTopology.from_config(load_config(CONFIG_PATH))
    source = Tracklet("cam_01:1", "cam_01", 0, 5, [1.0, 0.0])
    target = Tracklet("cam_02:2", "cam_02", 10, 12, [1.0, 0.0])
    candidate = topology.generate_candidates(source, [target], widen_if_empty=False)[0]
    result = associate(source, candidate)
    assert candidate.elapsed_seconds == 5
    assert candidate.via_topology
    assert result.score > 0.9
    assert any(reason.startswith("appearance_component=") for reason in result.reasons)


def test_rank_applies_relative_offsets_and_keeps_heuristic_decisions():
    rows = [
        {"tracklet_id": "cam_01:1", "camera_id": "cam_01", "start_time": 0,
         "end_time": 5, "embedding": [1.0, 0.0]},
        {"tracklet_id": "cam_02:2", "camera_id": "cam_02", "start_time": 10,
         "end_time": 12, "embedding": [1.0, 0.0]},
    ]
    results = rank(rows, load_config(CONFIG_PATH), parse_offsets(["cam_01=-20"]))
    assert len(results) == 1
    assert results[0]["elapsed_seconds"] == 25
    assert results[0]["decision"] == "accept"
    assert results[0]["via_topology"]


def test_offsets_reject_non_finite_and_malformed_values():
    with pytest.raises(ValueError):
        parse_offsets(["cam_01=nan"])
    with pytest.raises(ValueError):
        parse_offsets(["cam_01"])
