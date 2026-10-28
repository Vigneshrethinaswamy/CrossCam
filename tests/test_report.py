from crosscam.report import build_report


def test_report_includes_runtime_and_limits_identity_claims():
    summary = {
        "device": "cpu",
        "videos_processed": 1,
        "sample_target_fps": 3,
        "confidence_threshold": 0.45,
        "iou_threshold": 0.3,
        "results": [{
            "video": "cam_a.mp4",
            "sampled_frames": 20,
            "total_person_detections": 12,
            "distinct_local_track_ids_observed": 3,
            "processing_sampled_fps": 2.5,
            "peak_torch_cuda_allocated_mb": None,
        }],
    }
    report = build_report(summary, {"input_rows": 12, "distinct_local_tracks": 3})
    assert "cam_a.mp4" in report
    assert "2.5" in report
    assert "ground-truth identity labels" in report
    assert "n/a" in report
