from crosscam.models import Tracklet, Transition
from crosscam.topology import CameraTopology


def t(tracklet_id, camera_id, start, end, embedding=None):
    return Tracklet(tracklet_id, camera_id, start, end, embedding or [])


def test_neighbor_transition_candidate_is_returned():
    topology = CameraTopology([Transition("cam_1", "cam_2", 2, 20, 0.9)])
    source = t("a", "cam_1", 0, 5)
    candidate = t("b", "cam_2", 10, 12)
    results = topology.generate_candidates(source, [candidate])
    assert len(results) == 1
    assert results[0].tracklet.tracklet_id == "b"
    assert results[0].via_topology is True


def test_widen_search_when_neighbor_missing():
    topology = CameraTopology([Transition("cam_1", "cam_2", 2, 20, 1.0)])
    source = t("a", "cam_1", 0, 5)
    unusual_route = t("b", "cam_4", 10, 12)
    results = topology.generate_candidates(source, [unusual_route], widen_if_empty=True)
    assert len(results) == 1
    assert results[0].via_topology is False


def test_no_fallback_when_disabled():
    topology = CameraTopology([Transition("cam_1", "cam_2", 2, 20, 1.0)])
    source = t("a", "cam_1", 0, 5)
    other = t("b", "cam_4", 10, 12)
    assert topology.generate_candidates(source, [other], widen_if_empty=False) == []
