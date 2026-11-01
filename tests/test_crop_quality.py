from __future__ import annotations

import numpy as np
import pytest

from crosscam.crop_quality import (
    CropQualityConfig,
    assess_person_crop,
    box_iou,
    clip_box_to_frame,
    match_frame_detections_to_ground_truth,
    select_temporal_crops,
    select_temporal_indices,
    summarize_tracklet_purity,
)


def checkerboard_frame(height: int = 128, width: int = 96) -> np.ndarray:
    y, x = np.indices((height, width))
    pattern = ((x // 8 + y // 8) % 2) * 255
    frame = np.stack([pattern, pattern, pattern], axis=-1)
    return frame.astype(np.uint8)


def test_clip_box_returns_clipped_coordinates_and_visible_fraction():
    clipped, visible = clip_box_to_frame(
        (-10.0, 5.0, 50.0, 105.0), frame_width=100, frame_height=120
    )
    assert clipped == (0, 5, 50, 105)
    assert visible == pytest.approx(5 / 6)


def test_clip_box_rejects_degenerate_and_outside_boxes():
    assert clip_box_to_frame((10, 10, 10, 20), 100, 100) == (None, 0.0)
    assert clip_box_to_frame((-30, 0, -10, 20), 100, 100) == (None, 0.0)


def test_assess_person_crop_returns_accepted_in_memory_crop():
    frame = checkerboard_frame()
    config = CropQualityConfig(
        min_width=32,
        min_height=64,
        min_detection_score=0.4,
        min_visible_fraction=0.9,
        min_laplacian_variance=1.0,
    )

    crop, assessment = assess_person_crop(
        frame,
        (8, 8, 88, 120),
        0.95,
        config,
        frame_index=12,
        timestamp_seconds=0.4,
    )

    assert crop is not None
    assert crop.shape == (112, 80, 3)
    assert crop.dtype == np.uint8
    assert assessment.accepted is True
    assert assessment.rejection_reasons == ()
    assert assessment.laplacian_variance >= 1.0


def test_assess_person_crop_reports_multiple_rejection_reasons():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    config = CropQualityConfig(
        min_width=32,
        min_height=64,
        min_detection_score=0.7,
        min_visible_fraction=0.9,
        min_laplacian_variance=2.0,
    )

    crop, assessment = assess_person_crop(
        frame,
        (-10, 10, 20, 40),
        0.2,
        config,
        frame_index=1,
        timestamp_seconds=0.1,
    )

    assert crop is not None
    assert assessment.accepted is False
    assert set(assessment.rejection_reasons) == {
        "too_narrow",
        "too_short",
        "low_detection_score",
        "excessively_clipped",
        "low_sharpness",
    }


def test_assess_person_crop_rejects_box_outside_frame():
    crop, assessment = assess_person_crop(
        checkerboard_frame(),
        (200, 200, 240, 260),
        0.99,
        CropQualityConfig(),
        frame_index=0,
        timestamp_seconds=0.0,
    )
    assert crop is None
    assert assessment.accepted is False
    assert assessment.rejection_reasons == ("invalid_or_outside_frame",)


@pytest.mark.parametrize(
    "frame, box, score, frame_index, timestamp",
    [
        (np.zeros((10, 10), dtype=np.uint8), (0, 0, 5, 5), 0.9, 0, 0.0),
        (np.zeros((10, 10, 3), dtype=np.float32), (0, 0, 5, 5), 0.9, 0, 0.0),
        (np.zeros((0, 10, 3), dtype=np.uint8), (0, 0, 5, 5), 0.9, 0, 0.0),
    ],
)
def test_assessment_rejects_invalid_frames(frame, box, score, frame_index, timestamp):
    with pytest.raises((TypeError, ValueError)):
        assess_person_crop(
            frame, box, score, CropQualityConfig(),
            frame_index=frame_index, timestamp_seconds=timestamp,
        )


@pytest.mark.parametrize(
    "box, score, frame_index, timestamp",
    [
        ((0, 0, float("nan"), 10), 0.9, 0, 0.0),
        ((0, 0, 10, 10), float("inf"), 0, 0.0),
        ((0, 0, 10, 10), 1.1, 0, 0.0),
        ((0, 0, 10, 10), 0.9, -1, 0.0),
        ((0, 0, 10, 10), 0.9, 0, -0.1),
    ],
)
def test_assessment_rejects_invalid_observation_metadata(
    box, score, frame_index, timestamp
):
    with pytest.raises(ValueError):
        assess_person_crop(
            checkerboard_frame(), box, score, CropQualityConfig(),
            frame_index=frame_index, timestamp_seconds=timestamp,
        )


def test_temporal_indices_are_evenly_distributed_and_unique():
    indices = select_temporal_indices(list(range(9)), max_items=3)
    assert indices == [0, 4, 8]
    assert len(set(indices)) == 3


def test_temporal_indices_handle_unsorted_and_duplicate_timestamps():
    timestamps = [4.0, 0.0, 2.0, 2.0, 8.0]
    selected = select_temporal_indices(timestamps, max_items=3)
    assert selected == [1, 0, 4]
    assert select_temporal_indices(timestamps, max_items=10) == [1, 2, 3, 0, 4]


def test_temporal_indices_validate_inputs():
    assert select_temporal_indices([], 3) == []
    with pytest.raises(ValueError):
        select_temporal_indices([0.0, float("nan")], 2)
    with pytest.raises(ValueError):
        select_temporal_indices([0.0], 0)


def test_temporal_crop_selection_ignores_rejected_assessments():
    frame = checkerboard_frame()
    config = CropQualityConfig(min_width=10, min_height=10)
    assessments = [
        assess_person_crop(
            frame, (4, 4, 80, 120), 0.9, config,
            frame_index=index, timestamp_seconds=float(index),
        )[1]
        for index in range(6)
    ]
    rejected = assessments[2]
    assessments[2] = type(rejected)(
        **{
            **rejected.__dict__,
            "accepted": False,
            "rejection_reasons": ("test_rejection",),
        }
    )
    selected = select_temporal_crops(assessments, max_crops=3)
    assert len(selected) == 3
    assert all(item.accepted for item in selected)
    assert all(item.frame_index != 2 for item in selected)


def test_box_iou_is_correct_and_handles_no_overlap():
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_ground_truth_matching_is_one_to_one():
    matches = match_frame_detections_to_ground_truth(
        predicted_boxes=[(0, 0, 10, 10), (0, 0, 10, 10), (40, 40, 50, 50)],
        ground_truth_boxes=[(0, 0, 10, 10), (40, 40, 50, 50)],
        ground_truth_ids=["person-A", "person-B"],
        min_iou=0.5,
    )
    assert matches == ["person-A", None, "person-B"]


def test_ground_truth_matching_validates_alignment_and_threshold():
    with pytest.raises(ValueError):
        match_frame_detections_to_ground_truth([(0, 0, 10, 10)], [], ["id"])
    with pytest.raises(ValueError):
        match_frame_detections_to_ground_truth([], [], [], min_iou=1.1)


def test_ground_truth_matching_never_assigns_zero_overlap_even_at_zero_threshold():
    matches = match_frame_detections_to_ground_truth(
        predicted_boxes=[(0, 0, 10, 10)],
        ground_truth_boxes=[(20, 20, 30, 30)],
        ground_truth_ids=["person-A"],
        min_iou=0.0,
    )
    assert matches == [None]


def test_tracklet_purity_reports_mixed_identity_and_unmatched_coverage():
    result = summarize_tracklet_purity(["person-A", "person-A", "person-B", None])
    assert result.observation_count == 4
    assert result.matched_observations == 3
    assert result.unmatched_observations == 1
    assert result.dominant_identity == "person-A"
    assert result.identity_purity == pytest.approx(2 / 3)
    assert result.match_coverage == pytest.approx(0.75)


def test_tracklet_purity_handles_no_ground_truth_matches():
    result = summarize_tracklet_purity([None, None])
    assert result.dominant_identity is None
    assert result.identity_purity is None
    assert result.match_coverage == 0.0
    with pytest.raises(ValueError):
        summarize_tracklet_purity([])


def test_crop_quality_report_decodes_synthetic_video_without_saving_crops(tmp_path):
    import csv

    import cv2

    from crosscam.crop_quality import CropQualityConfig
    from scripts.diagnose_crop_quality import (
        build_tracklet_quality_report,
        read_observations,
    )

    video_dir = tmp_path / "videos"
    csv_dir = tmp_path / "evaluation"
    video_dir.mkdir()
    csv_dir.mkdir()
    video_path = video_dir / "cam_test.avi"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (96, 128),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable in this environment")
    frame = checkerboard_frame()
    for _ in range(5):
        writer.write(frame)
    writer.release()

    csv_path = csv_dir / "cam_test_tracks.csv"
    fields = [
        "video", "camera_id", "frame_index", "timestamp_seconds",
        "track_id", "confidence", "x1", "y1", "x2", "y2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=fields)
        output.writeheader()
        for frame_index in range(5):
            output.writerow({
                "video": video_path.name,
                "camera_id": "cam_test",
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / 5,
                "track_id": 7,
                "confidence": 0.95,
                "x1": 8,
                "y1": 8,
                "x2": 88,
                "y2": 120,
            })

    observations = read_observations(csv_dir)
    report = build_tracklet_quality_report(
        video_dir,
        observations,
        CropQualityConfig(
            min_width=32,
            min_height=64,
            min_detection_score=0.4,
            min_visible_fraction=0.9,
            min_observations=3,
            min_selected_crops=2,
            max_crops_per_tracklet=2,
            candidate_multiplier=2,
        ),
    )

    track = report["tracklets"][0]
    assert report["summary"]["tracklet_count"] == 1
    assert track["observation_count"] == 5
    assert track["decoded_candidate_count"] == 4
    assert track["selected_crop_count"] == 2
    assert track["crop_quality_eligible_for_next_stage"] is True
    assert track["selected_frame_indices"] == [0, 4]
    assert not list(tmp_path.rglob("*.jpg"))
    assert not list(tmp_path.rglob("*.png"))


def test_read_observations_rejects_duplicate_track_frame_rows(tmp_path):
    import csv

    from scripts.diagnose_crop_quality import read_observations

    csv_path = tmp_path / "cam_test_tracks.csv"
    fields = [
        "video", "camera_id", "frame_index", "timestamp_seconds",
        "track_id", "confidence", "x1", "y1", "x2", "y2",
    ]
    row = {
        "video": "cam_test.avi",
        "camera_id": "cam_test",
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "track_id": 1,
        "confidence": 0.9,
        "x1": 0,
        "y1": 0,
        "x2": 10,
        "y2": 20,
    }
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)

    with pytest.raises(ValueError, match="Duplicate observation"):
        read_observations(tmp_path)
