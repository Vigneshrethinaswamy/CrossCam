import pytest

from crosscam.models import CameraRelationship, Tracklet, Transition
from crosscam.topology import CameraTopology


def t(
    tracklet_id, camera_id, start, end, embedding=None, *, offset_known=True
):
    return Tracklet(
        tracklet_id, camera_id, start, end, embedding or [],
        timestamp_offset_known=offset_known,
    )


def test_neighbor_transition_candidate_is_returned():
    topology = CameraTopology([Transition("cam_1", "cam_2", 2, 20, 0.9)])
    source = t("a", "cam_1", 0, 5)
    candidate = t("b", "cam_2", 10, 12)
    results = topology.generate_candidates(source, [candidate])
    assert len(results) == 1
    assert results[0].tracklet.tracklet_id == "b"
    assert results[0].via_topology is True
    assert results[0].topology_prior == 0.9


def test_widening_adds_candidates_even_when_weak_topology_candidate_exists():
    topology = CameraTopology([Transition("cam_1", "cam_2", 2, 20, 0.2)])
    source = t("a", "cam_1", 0, 5)
    weak_neighbor = t("b", "cam_2", 10, 12)
    unexpected_route = t("c", "cam_4", 14, 16)

    narrow = topology.generate_candidates(source, [weak_neighbor, unexpected_route])
    assert [item.tracklet.tracklet_id for item in narrow] == ["b"]

    widened = topology.generate_candidates(
        source, [weak_neighbor, unexpected_route], widen_search=True
    )
    assert {item.tracklet.tracklet_id for item in widened} == {"b", "c"}
    assert next(x for x in widened if x.tracklet.tracklet_id == "b").topology_prior == 0.2
    assert next(x for x in widened if x.tracklet.tracklet_id == "c").topology_prior is None


def test_widened_candidate_does_not_receive_maximal_prior():
    topology = CameraTopology([Transition("cam_1", "cam_2", 2, 20, 0.7)])
    source = t("a", "cam_1", 0, 5)
    other = t("b", "cam_3", 10, 12)
    candidate = topology.generate_candidates(
        source, [other], widen_search=True
    )[0]
    assert candidate.via_topology is False
    assert candidate.topology_prior is None


def test_no_widened_candidate_when_disabled():
    topology = CameraTopology([Transition("cam_1", "cam_2", 2, 20, 1.0)])
    source = t("a", "cam_1", 0, 5)
    other = t("b", "cam_4", 10, 12)
    assert topology.generate_candidates(source, [other]) == []


def test_known_overlap_allows_simultaneous_tracklets():
    topology = CameraTopology(
        [],
        [CameraRelationship("cam_1", "cam_2", "known_overlap")],
    )
    source = t("a", "cam_1", 1, 5)
    target = t("b", "cam_2", 2, 4)
    candidates = topology.generate_candidates(source, [target])
    assert len(candidates) == 1
    assert candidates[0].relationship_state == "known_overlap"


def test_known_non_overlap_excludes_simultaneous_tracklets_when_offsets_known():
    topology = CameraTopology(
        [],
        [CameraRelationship("cam_1", "cam_2", "known_non_overlap")],
    )
    source = t("a", "cam_1", 1, 5)
    target = t("b", "cam_2", 2, 4)
    assert topology.generate_candidates(
        source, [target], widen_search=True
    ) == []


def test_unknown_offsets_prevent_hard_simultaneity_exclusion():
    topology = CameraTopology(
        [],
        [CameraRelationship("cam_1", "cam_2", "known_non_overlap")],
    )
    source = t("a", "cam_1", 1, 5, offset_known=False)
    target = t("b", "cam_2", 2, 4, offset_known=True)
    candidates = topology.generate_candidates(
        source, [target], widen_search=True
    )
    assert len(candidates) == 1
    assert candidates[0].elapsed_seconds is None


def test_duplicate_transition_edges_are_rejected():
    edge = Transition("cam_1", "cam_2", 2, 20, 0.8)
    with pytest.raises(ValueError, match="duplicate"):
        CameraTopology([edge, edge])


def test_duplicate_camera_relationships_are_rejected():
    relation = CameraRelationship("cam_1", "cam_2", "known_overlap")
    with pytest.raises(ValueError, match="duplicate"):
        CameraTopology([], [relation, relation])
