from crosscam.models import AssociationResult, Decision, Tracklet
from crosscam.store import TrackStore


def test_store_round_trip(tmp_path):
    path = tmp_path / "nested" / "test.sqlite"
    with TrackStore(path) as store:
        store.upsert_tracklet(Tracklet("t1", "cam_1", 1, 2, [0.2, 0.3]))
        result = AssociationResult("t1", "t2", 0.8, Decision.ACCEPT, ("test",))
        store.save_association(result)
        assert store.list_tracklets()[0]["tracklet_id"] == "t1"
        assert store.list_associations()[0]["decision"] == "accept"
