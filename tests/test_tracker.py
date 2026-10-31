import pytest

from crosscam.detection import Detection
from crosscam.tracker import IoUTracker, intersection_over_union


def test_iou_for_overlapping_boxes():
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_tracker_keeps_id_for_overlapping_detections():
    tracker = IoUTracker(iou_threshold=0.2)
    first = tracker.update([Detection((0, 0, 20, 40), 0.9)], timestamp_seconds=0.0)
    second = tracker.update([Detection((1, 1, 21, 41), 0.8)], timestamp_seconds=0.2)
    assert first[0].track_id == second[0].track_id


def test_tracker_creates_new_id_for_non_overlapping_person():
    tracker = IoUTracker()
    first = tracker.update([Detection((0, 0, 10, 10), 0.9)], timestamp_seconds=0.0)
    second = tracker.update([Detection((100, 100, 110, 110), 0.9)], timestamp_seconds=0.2)
    assert first[0].track_id != second[0].track_id


def test_tracker_expiry_is_time_based_not_sample_count_based():
    tracker = IoUTracker(max_missed_seconds=2.0)
    first = tracker.update([Detection((0, 0, 10, 10), 0.9)], timestamp_seconds=0.0)[0]
    # Many missed samples over less than two seconds do not expire the track.
    for timestamp in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        tracker.update([], timestamp_seconds=timestamp)
    retained = tracker.update(
        [Detection((0, 0, 10, 10), 0.9)], timestamp_seconds=1.5
    )[0]
    assert retained.track_id == first.track_id

    expired = tracker.update(
        [Detection((0, 0, 10, 10), 0.9)], timestamp_seconds=4.1
    )[0]
    assert expired.track_id > first.track_id


def test_tracker_rejects_non_monotonic_timestamps():
    tracker = IoUTracker()
    tracker.update([], timestamp_seconds=1.0)
    with pytest.raises(ValueError, match="monotonic"):
        tracker.update([], timestamp_seconds=0.5)
