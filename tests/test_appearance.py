from pathlib import Path

import numpy as np
import pytest

from crosscam.appearance import hsv_histogram
from crosscam.association import associate
from crosscam.config import load_config
from crosscam.models import Tracklet
from crosscam.topology import CameraTopology


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
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    topology = CameraTopology.from_config(load_config(config_path))
    source = Tracklet("cam_01:1", "cam_01", 0, 5, [1.0, 0.0])
    target = Tracklet("cam_02:2", "cam_02", 10, 12, [1.0, 0.0])
    candidate = topology.generate_candidates(source, [target], widen_if_empty=False)[0]
    result = associate(source, candidate)
    assert candidate.elapsed_seconds == 5
    assert candidate.via_topology
    assert result.score > 0.9
    assert any(reason.startswith("appearance_component=") for reason in result.reasons)
