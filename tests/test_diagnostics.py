from scripts.diagnose_tracks import build_diagnostics


def row(video, track_id, timestamp, frame_index, confidence=0.8):
    return {
        "video": video,
        "camera_id": video.removesuffix(".mp4"),
        "track_id": track_id,
        "timestamp_seconds": timestamp,
        "frame_index": frame_index,
        "confidence": confidence,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 11.0,
        "y2": 22.0,
    }


def test_build_diagnostics_summarizes_tracks_and_singletons():
    rows = [
        row("clip.mp4", 1, 0.0, 0, 0.9),
        row("clip.mp4", 1, 0.5, 15, 0.7),
        row("clip.mp4", 2, 1.0, 30, 0.6),
    ]

    tracks, summary = build_diagnostics(rows)

    assert len(tracks) == 2
    first = next(track for track in tracks if track["track_id"] == 1)
    assert first["observations"] == 2
    assert first["observed_span_seconds"] == 0.5
    assert first["max_observation_gap_seconds"] == 0.5
    assert first["mean_detection_score"] == 0.8
    assert first["singleton_track"] is False

    video = summary["per_video"][0]
    assert video["track_count"] == 2
    assert video["singleton_tracks"] == 1
    assert video["singleton_track_percent"] == 50.0
    assert summary["input_rows"] == 3


def test_build_diagnostics_empty_input_is_safe():
    tracks, summary = build_diagnostics([])
    assert tracks == []
    assert summary["input_rows"] == 0
    assert summary["per_video"] == []
