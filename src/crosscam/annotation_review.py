"""Focused, local-only selection and validation for D4 annotation review."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

REVIEW_TRACKLETS = frozenset(
    {
        ("sample1.mp4", "20"),
        ("sample1.mp4", "11"),
        ("sample1.mp4", "5"),
        ("sample1.mp4", "16"),
    }
)

REQUIRED_FIELDS = frozenset(
    {
        "video",
        "camera_id",
        "frame_index",
        "timestamp_seconds",
        "detection_index",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "ground_truth_identity",
    }
)

REVIEW_FIELDS = (
    "previous_ground_truth_identity",
    "review_status",
    "review_notes",
)

COMPLETED_REVIEW_STATUSES = frozenset(
    {"confirmed", "reassigned", "explicit_unknown"}
)


def observation_key(row: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    """Return a stable key for one sampled detector/tracker observation."""
    return (
        str(row.get("video", "")).strip(),
        str(row.get("camera_id", "")).strip(),
        str(row.get("track_id", "")).strip(),
        str(row.get("frame_index", "")).strip(),
        str(row.get("detection_index", "")).strip(),
    )


def build_review_rows(
    source_rows: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Select the four agreed sample1 tracklets without mutating source rows."""
    copied_rows = [dict(row) for row in source_rows]
    if not copied_rows:
        raise ValueError("The source annotation manifest is empty")

    missing_fields = REQUIRED_FIELDS.difference(copied_rows[0])
    if missing_fields:
        raise ValueError(
            "The source manifest is missing required columns: "
            + ", ".join(sorted(missing_fields))
        )

    selected = [
        row
        for row in copied_rows
        if (row["video"].strip(), row["track_id"].strip()) in REVIEW_TRACKLETS
    ]

    present_tracklets = {
        (row["video"].strip(), row["track_id"].strip()) for row in selected
    }
    missing_tracklets = REVIEW_TRACKLETS.difference(present_tracklets)
    if missing_tracklets:
        missing_text = ", ".join(
            f"{video}/track-{track_id}"
            for video, track_id in sorted(missing_tracklets)
        )
        raise ValueError(f"Review manifest is missing target tracklets: {missing_text}")

    keys = [observation_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("The selected observations contain duplicate row keys")

    for row in selected:
        row["previous_ground_truth_identity"] = row.get(
            "ground_truth_identity", ""
        ).strip()
        row["review_status"] = "pending"
        row["review_notes"] = ""

    return selected


def validate_review_rows(
    source_rows: Iterable[Mapping[str, str]],
    review_rows: Iterable[Mapping[str, str]],
) -> None:
    """Ensure a resumable review file belongs to the current source manifest."""
    expected_rows = build_review_rows(source_rows)
    expected = {observation_key(row): row for row in expected_rows}
    actual_rows = [dict(row) for row in review_rows]
    actual_keys = [observation_key(row) for row in actual_rows]

    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("The review results contain duplicate observation keys")

    actual = {observation_key(row): row for row in actual_rows}
    if set(actual) != set(expected):
        raise ValueError(
            "The existing review results do not match the current 15-row target set"
        )

    for key, source_row in expected.items():
        review_row = actual[key]
        if (
            review_row.get("previous_ground_truth_identity", "").strip()
            != source_row["previous_ground_truth_identity"]
        ):
            raise ValueError(
                "The previous identity label changed for observation "
                f"{key}; refusing to resume this review file"
            )
        if not all(field in review_row for field in REVIEW_FIELDS):
            raise ValueError(
                "The existing review results are missing review metadata columns"
            )
