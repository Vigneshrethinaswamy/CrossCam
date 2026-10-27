from crosscam.association import associate, cosine_similarity, decide
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
    candidate = Candidate(target, 1.0, 5.0, True)
    result = associate(source, candidate)
    assert result.source_tracklet_id == "a"
    assert result.candidate_tracklet_id == "b"
    assert 0 <= result.score <= 1
    assert result.reasons
