import pytest

from crosscam.detection import Detection
from crosscam.tracker import IoUTracker, intersection_over_union


def test_iou_for_overlapping_boxes():
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_tracker_keeps_id_for_overlapping_detections():
    tracker = IoUTracker(iou_threshold=0.2)
    first = tracker.update([Detection((0, 0, 20, 40), 0.9)])
    second = tracker.update([Detection((1, 1, 21, 41), 0.8)])
    assert first[0].track_id == second[0].track_id


def test_tracker_creates_new_id_for_non_overlapping_person():
    tracker = IoUTracker()
    first = tracker.update([Detection((0, 0, 10, 10), 0.9)])
    second = tracker.update([Detection((100, 100, 110, 110), 0.9)])
    assert first[0].track_id != second[0].track_id


def test_tracker_expires_lost_tracks_after_limit():
    tracker = IoUTracker(max_missed_frames=1)
    first = tracker.update([Detection((0, 0, 10, 10), 0.9)])[0]
    tracker.update([])
    tracker.update([])
    later = tracker.update([Detection((0, 0, 10, 10), 0.9)])[0]
    assert later.track_id > first.track_id
