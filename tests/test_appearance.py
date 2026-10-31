from pathlib import Path

import numpy as np
import pytest

from crosscam.appearance import hsv_histogram
from crosscam.association import associate, score_candidate
from crosscam.config import load_config
from crosscam.models import Tracklet
from crosscam.topology import CameraTopology
from scripts.associate_tracklets import parse_offsets, rank

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def rows_for_association():
    return [
        {"tracklet_id": "a", "camera_id": "cam_01", "start_time": 0,
         "end_time": 5, "embedding": [1.0, 0.0]},
        {"tracklet_id": "b", "camera_id": "cam_02", "start_time": 10,
         "end_time": 12, "embedding": [0.0, 1.0]},
        {"tracklet_id": "c", "camera_id": "cam_03", "start_time": 14,
         "end_time": 16, "embedding": [0.0, 1.0]},
    ]


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


def test_orthogonal_histograms_receive_zero_appearance_score():
    topology = CameraTopology.from_config(load_config(CONFIG_PATH))
    source = Tracklet("cam_01:1", "cam_01", 0, 5, [1.0, 0.0])
    target = Tracklet("cam_02:2", "cam_02", 10, 12, [0.0, 1.0])
    candidate = topology.generate_candidates(source, [target], widen_if_empty=False)[0]
    _, reasons = score_candidate(source, candidate)
    appearance_reason = next(
        reason for reason in reasons if reason.startswith("appearance_component=")
    )
    assert appearance_reason == "appearance_component=0.000"


def test_rank_uses_only_explicit_camera_offsets():
    rows = rows_for_association()[:2]
    config = load_config(CONFIG_PATH)
    results = rank(
        rows, config, parse_offsets(["cam_01=-20", "cam_02=0"])
    )
    assert len(results) == 1
    assert results[0]["elapsed_seconds"] == 25
    assert results[0]["timestamp_offsets_known"] is True
    assert results[0]["via_topology"] is True


def test_missing_camera_offset_remains_unknown_not_zero():
    rows = rows_for_association()[:2]
    results = rank(
        rows, load_config(CONFIG_PATH), parse_offsets(["cam_01=-20"])
    )
    assert len(results) == 1
    assert results[0]["elapsed_seconds"] is None
    assert results[0]["timestamp_offsets_known"] is False
    assert "temporal_evidence=unknown" in results[0]["reasons"]


def test_weak_candidate_triggers_union_widening():
    results = rank(
        rows_for_association(),
        load_config(CONFIG_PATH),
        parse_offsets(["cam_01=0", "cam_02=0", "cam_03=0"]),
    )
    source_results = [row for row in results if row["source_tracklet_id"] == "a"]
    assert {row["candidate_tracklet_id"] for row in source_results} == {"b", "c"}
    assert all(row["widened_search_used"] for row in source_results)


def test_offsets_reject_non_finite_malformed_and_duplicate_values():
    with pytest.raises(ValueError):
        parse_offsets(["cam_01=nan"])
    with pytest.raises(ValueError):
        parse_offsets(["cam_01"])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_offsets(["cam_01=0", "cam_01=1"])
