# D4 focused annotation review

This review narrows the second visual audit to the four mixed-label tracklets
identified in the 35-observation pilot. It does not edit the original annotation
manifest or the previously generated evaluation report.

## Target observations

The review manifest is built locally from:

`outputs/crop_quality/annotation_manifest.csv`

It selects all rows for these same-video tracklets:

- `sample1.mp4`, track ID `20` (five observations)
- `sample1.mp4`, track ID `11` (three observations)
- `sample1.mp4`, track ID `5` (five observations)
- `sample1.mp4`, track ID `16` (two observations)

Expected pilot selection: 15 observations. The script verifies that all four
target tracklets are present and rejects duplicate observation keys.

## Run on Windows

From the repository root, with the project virtual environment active:

```powershell
cd C:\Works\CrossCam
.\.venv\Scripts\Activate.ps1
python -m ruff check --no-cache src/crosscam/annotation_review.py scripts/review_tracklet_annotations.py tests/test_annotation_review.py
python -m pytest -q tests/test_annotation_review.py
python .\scripts\review_tracklet_annotations.py
```

Close the source or result CSV in Excel before running the viewer. The viewer
uses the local videos under `data\`; no frames, crops, or embeddings are saved.

## Viewer controls

The OpenCV window must have keyboard focus.

- `K`: confirm and keep the previous identity label.
- `R`: reassign; enter a label in the PowerShell terminal, e.g. `GT-13`
  or `sample1:GT-13`.
- `U`: mark the observation UNKNOWN.
- `S`: skip for now; it remains pending and will appear on a later run.
- `Q`: quit and save progress.

The reassign option accepts only labels prefixed for the current camera/video
(e.g. `sample1:GT-13`). Use UNKNOWN rather than guessing.

## Output and safety

The review output is local and git-ignored:

`outputs/crop_quality/annotation_review_results.csv`

It starts as a separate copy of the 15 selected rows and includes
`previous_ground_truth_identity`, `review_status`, and `review_notes`.
The current `ground_truth_identity` column holds the review decision.
Existing review output is validated and resumed rather than overwritten.

The following files are not modified:

- `outputs/crop_quality/annotation_manifest.csv`
- `outputs/crop_quality/annotation_evaluation_report.json`
- `outputs/crop_quality/annotation_manifest_evaluation.csv`

The labels are manual ground truth for this sample1 video only. This review
does not establish cross-camera identity matches. Mixed labels are audit
signals, not proof of tracker ID switches; visual confirmation is required.
