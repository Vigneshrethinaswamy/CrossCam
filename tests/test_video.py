import cv2
import numpy as np
import pytest

from crosscam.video import FramePacket, FrameSampler, VideoSource


def packet(index: int, timestamp: float) -> FramePacket:
    return FramePacket("cam_01", index, timestamp, np.zeros((8, 8, 3), dtype=np.uint8))


def test_frame_sampler_uses_video_timestamps():
    sampler = FrameSampler(2)
    selected = [i for i in range(6) if sampler.should_sample(packet(i, i / 4))]
    assert selected == [0, 2, 4]


def test_frame_sampler_rejects_non_positive_fps():
    with pytest.raises(ValueError, match="greater than zero"):
        FrameSampler(0)


def test_video_source_reads_timestamped_frames(tmp_path):
    video_path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (16, 16),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable in this environment")
    for value in (0, 80, 160):
        frame = np.full((16, 16, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = list(VideoSource(video_path, "cam_test").frames())
    assert len(frames) == 3
    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0.0, 0.2, 0.4])
    assert all(frame.camera_id == "cam_test" for frame in frames)
