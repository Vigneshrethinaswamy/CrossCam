import pytest

from scripts.evaluate_videos import resolve_retention_seconds


def test_default_retention_is_seconds():
    assert resolve_retention_seconds(3.0) == pytest.approx(10 / 3)


def test_legacy_frame_retention_converts_using_sample_rate():
    assert resolve_retention_seconds(3.0, max_missed_frames=10) == pytest.approx(10 / 3)


def test_seconds_and_legacy_frames_cannot_be_combined():
    with pytest.raises(ValueError, match="only one"):
        resolve_retention_seconds(3.0, max_missed_seconds=2.0, max_missed_frames=10)


def test_invalid_retention_is_rejected():
    with pytest.raises(ValueError):
        resolve_retention_seconds(0.0)
    with pytest.raises(ValueError):
        resolve_retention_seconds(3.0, max_missed_frames=-1)
    with pytest.raises(ValueError):
        resolve_retention_seconds(3.0, max_missed_seconds=float("nan"))
