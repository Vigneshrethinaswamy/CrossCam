from crosscam.association import associate, cosine_similarity, decide, score_candidate
from crosscam.models import Decision, Tracklet
from crosscam.topology import Candidate


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_decision_bands():
    assert decide(0.9) is Decision.ACCEPT
    assert decide(0.6) is Decision.DEFER
    assert decide(0.2) is Decision.REJECT


def test_association_has_explanation():
    source = Tracklet("a", "cam_1", 0, 5, [1.0, 0.0])
    target = Tracklet("b", "cam_2", 10, 12, [1.0, 0.0])
    candidate = Candidate(target, 0.9, 5.0, True, "known_transition")
    result = associate(source, candidate)
    assert result.source_tracklet_id == "a"
    assert result.candidate_tracklet_id == "b"
    assert 0 <= result.score <= 1
    assert result.reasons


def test_unknown_time_and_topology_evidence_are_neutral_and_explicit():
    source = Tracklet("a", "cam_1", 0, 5, [1.0, 0.0], timestamp_offset_known=False)
    target = Tracklet("b", "cam_2", 10, 12, [1.0, 0.0], timestamp_offset_known=False)
    candidate = Candidate(target, None, None, False, "unknown")
    score, reasons = score_candidate(source, candidate)
    assert score == 0.75
    assert "temporal_evidence=unknown" in reasons
    assert "topology_evidence=unknown" in reasons


def test_decision_margin_defers_ambiguous_top_match():
    source = Tracklet("a", "cam_1", 0, 5, [1.0, 0.0])
    target = Tracklet("b", "cam_2", 10, 12, [1.0, 0.0])
    candidate = Candidate(target, 1.0, 5.0, True, "known_transition")
    result = associate(
        source, candidate, min_margin=0.08, runner_up_score=0.76
    )
    assert result.decision is Decision.DEFER
    assert any("decision_margin_below_minimum" in reason for reason in result.reasons)
