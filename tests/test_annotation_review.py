from __future__ import annotations

import pytest

from crosscam.annotation_review import (
    REVIEW_FIELDS,
    build_review_rows,
    validate_review_rows,
)


def make_manifest_rows() -> list[dict[str, str]]:
    rows = []
    tracklets = [("20", 5), ("11", 3), ("5", 5), ("16", 2)]
    frame = 0
    for track_id, count in tracklets:
        for index in range(count):
            rows.append(
                {
                    "video": "sample1.mp4",
                    "camera_id": "sample1",
                    "track_id": track_id,
                    "frame_index": str(frame),
                    "timestamp_seconds": f"{frame / 25:.2f}",
                    "detection_index": str(index),
                    "x1": "10",
                    "y1": "10",
                    "x2": "100",
                    "y2": "200",
                    "ground_truth_identity": f"sample1:GT-{int(track_id):02d}",
                }
            )
            frame += 1

    # An unrelated row must not enter the review set.
    rows.append(
        {
            "video": "sample2.mp4",
            "camera_id": "sample2",
            "track_id": "1",
            "frame_index": "0",
            "timestamp_seconds": "0.00",
            "detection_index": "0",
            "x1": "10",
            "y1": "10",
            "x2": "100",
            "y2": "200",
            "ground_truth_identity": "sample2:GT-01",
        }
    )
    return rows


def test_build_review_rows_selects_exact_four_tracklets_and_15_rows():
    source = make_manifest_rows()
    selected = build_review_rows(source)

    assert len(selected) == 15
    assert {
        (row["video"], row["track_id"]) for row in selected
    } == {
        ("sample1.mp4", "20"),
        ("sample1.mp4", "11"),
        ("sample1.mp4", "5"),
        ("sample1.mp4", "16"),
    }
    assert all(row["review_status"] == "pending" for row in selected)
    assert all(row["review_notes"] == "" for row in selected)
    assert all(field in selected[0] for field in REVIEW_FIELDS)


def test_build_review_rows_preserves_previous_labels_and_source_rows():
    source = make_manifest_rows()
    original = [row.copy() for row in source]

    selected = build_review_rows(source)

    assert source == original
    for row in selected:
        assert (
            row["previous_ground_truth_identity"] == row["ground_truth_identity"]
        )


def test_build_review_rows_rejects_missing_target_tracklet():
    source = [
        row
        for row in make_manifest_rows()
        if row["track_id"] != "16" or row["video"] != "sample1.mp4"
    ]

    with pytest.raises(ValueError, match="missing target tracklets"):
        build_review_rows(source)


def test_build_review_rows_rejects_duplicate_observations():
    source = make_manifest_rows()
    duplicate = source[0].copy()
    source.append(duplicate)

    with pytest.raises(ValueError, match="duplicate row keys"):
        build_review_rows(source)


def test_validate_review_rows_allows_reassignment_but_preserves_original_label():
    source = make_manifest_rows()
    review = build_review_rows(source)
    review[0]["ground_truth_identity"] = "sample1:GT-99"
    review[0]["review_status"] = "reassigned"
    review[0]["review_notes"] = "Reassigned after visual review."

    validate_review_rows(source, review)


def test_validate_review_rows_rejects_results_from_different_source_labels():
    source = make_manifest_rows()
    review = build_review_rows(source)
    review[0]["previous_ground_truth_identity"] = "sample1:GT-88"

    with pytest.raises(ValueError, match="previous identity label changed"):
        validate_review_rows(source, review)


def test_validate_review_rows_rejects_incomplete_or_extra_observations():
    source = make_manifest_rows()
    review = build_review_rows(source)

    with pytest.raises(ValueError, match="do not match"):
        validate_review_rows(source, review[:-1])

    extra = review + [review[0].copy()]
    with pytest.raises(ValueError, match="duplicate observation keys"):
        validate_review_rows(source, extra)
