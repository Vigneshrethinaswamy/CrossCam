"""Recorded-video input and deterministic timestamped frame sampling."""
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame_index: int
    timestamp_seconds: float
    frame_bgr: NDArray[np.uint8]


class VideoSource:
    """Read a local video file and yield frames with timestamps derived from FPS."""

    def __init__(self, path: str | Path, camera_id: str) -> None:
        self.path = Path(path)
        self.camera_id = camera_id
        if not camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

    def frames(self) -> Iterator[FramePacket]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video file: {self.path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise ValueError(f"Video reports an invalid FPS value: {fps}")

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield FramePacket(
                    camera_id=self.camera_id,
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    frame_bgr=frame,
                )
                frame_index += 1
        finally:
            capture.release()


class FrameSampler:
    """Keep at most the requested number of frames per second of video time."""

    def __init__(self, target_fps: float) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        self.interval_seconds = 1.0 / target_fps
        self._next_timestamp = 0.0

    def should_sample(self, packet: FramePacket) -> bool:
        if packet.timestamp_seconds + 1e-9 < self._next_timestamp:
            return False
        self._next_timestamp = packet.timestamp_seconds + self.interval_seconds
        return True
