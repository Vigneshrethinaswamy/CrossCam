"""Replay a local video through sampled person detection and camera-local tracking."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from crosscam.detection import PersonDetector
from crosscam.tracker import IoUTracker
from crosscam.video import FrameSampler, VideoSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Path to a local video file")
    parser.add_argument("--camera-id", default="cam_01", help="Configured camera identifier")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Target sampled video FPS")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.45, help="Person detection threshold")
    parser.add_argument("--display", action="store_true", help="Display sampled frames with track IDs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        print(f"Video file not found: {args.video}", file=sys.stderr)
        return 2
    if args.sample_fps <= 0:
        print("--sample-fps must be greater than zero", file=sys.stderr)
        return 2

    detector = PersonDetector(args.device, args.confidence)
    tracker = IoUTracker()
    sampler = FrameSampler(args.sample_fps)
    processed = 0
    total_detections = 0
    peak_tracks = 0

    print(f"Camera: {args.camera_id}")
    print(f"Video: {args.video}")
    print(f"Inference device: {detector.device}")
    print("Loading frames; pretrained weights may download on the first run.")

    try:
        for packet in VideoSource(args.video, args.camera_id).frames():
            if not sampler.should_sample(packet):
                continue
            detections = detector.predict(packet.frame_bgr)
            tracked = tracker.update(detections)
            processed += 1
            total_detections += len(detections)
            peak_tracks = max(peak_tracks, len({person.track_id for person in tracked}))

            print(
                f"frame={packet.frame_index} t={packet.timestamp_seconds:.2f}s "
                f"detections={len(detections)} active_tracks={peak_tracks}"
            )

            if args.display:
                for person in tracked:
                    x1, y1, x2, y2 = (int(value) for value in person.box)
                    cv2.rectangle(packet.frame_bgr, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(
                        packet.frame_bgr,
                        f"{args.camera_id}:{person.track_id} {person.detection_score:.2f}",
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 220, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("CrossCam local tracking (press q to stop)", packet.frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if args.display:
            cv2.destroyAllWindows()

    print(
        f"Replay complete: sampled_frames={processed}, "
        f"person_detections={total_detections}, peak_people_in_frame={peak_tracks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
