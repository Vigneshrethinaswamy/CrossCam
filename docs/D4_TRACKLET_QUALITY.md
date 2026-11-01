# D4 — Tracklet and crop quality

**Status: implementation started; dataset validation pending.** This branch adds a
crop-assessment utility, bounded temporal selection, and a metadata-only
diagnostics CLI. It does not replace the local tracker, extract embeddings,
change association scoring, or close D4.

## Why this stage exists

A camera-local track can contain missed detections, badly clipped boxes, tiny
person regions, blur, or even observations from more than one ground-truth
identity. If those observations are aggregated into one appearance descriptor,
the descriptor can become unreliable. Cross-camera retrieval should wait until
these properties have been measured.

## Implemented components

### `src/crosscam/crop_quality.py`

- Clamp xyxy boxes to frame bounds and compute a visible-area fraction.
- Reject invalid/out-of-frame boxes.
- Extract a BGR crop in memory and report crop width/height.
- Assess detector score, visible fraction, and Laplacian-variance sharpness.
- Select observations closest to evenly spaced temporal targets.
- Select accepted crops spread across a tracklet.
- Match predicted boxes to ground-truth boxes one-to-one within a frame using
  IoU, and summarize per-track identity purity plus annotation match coverage.

### `scripts/diagnose_crop_quality.py`

Reads the existing `*_tracks.csv` outputs and their source videos. It chooses a
bounded set of temporally distributed candidate observations per tracklet,
decodes each video sequentially, assesses candidates in memory, and writes a
JSON report containing only metadata:

- tracklet observation and candidate counts;
- accepted/rejected crop counts and rejection reasons;
- selected frame indices and timestamps;
- selected-crop detection score, visible fraction, and sharpness summaries;
- provisional crop-quality eligibility and explicit reasons.

**Raw crops and embeddings are not saved.** The CLI does not run OSNet and does
not calculate identity purity because the detector CSV has no ground-truth
identity labels. The ground-truth matching and purity functions are reusable
helpers; a separate labelled-data evaluation is still required.

## Provisional defaults

| Setting | Default | Meaning |
|---|---:|---|
| Minimum crop width | 32 px | Reject narrower clipped crops |
| Minimum crop height | 64 px | Reject shorter clipped crops |
| Minimum detector score | 0.45 | Reject lower detector scores |
| Minimum visible fraction | 0.75 | Reject crops with more than 25% estimated area clipped |
| Minimum Laplacian variance | 0.0 | Report sharpness; blur rejection disabled by default |
| Minimum track observations | 3 | Minimum sampled observations to be considered |
| Minimum selected crops | 2 | Minimum usable crops to be crop-quality eligible |
| Maximum selected crops | 5 | Bound downstream crop use per tracklet |
| Candidate multiplier | 4 | Assess up to 4 × maximum selected crops before final selection |

These are **engineering starting points, not validated thresholds**. They must
be reviewed on a representative development set. A Laplacian variance is only
a blur proxy; it is not a calibrated measure of person-crop quality.

Crop-quality eligibility means that the tracklet has enough sampled observations
and usable candidate crops for a later experiment. It does **not** mean that the
tracklet contains one person, that the identity is correct, or that checkpoint
usage is legally cleared.

## Run it locally

First, make sure the evaluation CSVs were generated from videos that you are
authorized to process. Do not use unrelated sample videos to make identity
accuracy claims.

```powershell
cd C:\Works\CrossCam

.\.venv\Scripts\python.exe -m ruff check --no-cache src/crosscam/crop_quality.py scripts/diagnose_crop_quality.py tests/test_crop_quality.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_crop_quality.py

.\.venv\Scripts\python.exe .\scripts\diagnose_crop_quality.py `
  --input-dir outputs\evaluation `
  --video-dir data `
  --output outputs\crop_quality\crop_quality_summary.json
```

The JSON output is under `outputs/`, which is Git-ignored. No video, raw crop,
embedding, or checkpoint is added to Git.

A small authorized sequence is recommended first. The CLI decodes the selected
video sequentially so it can assess candidate frames in one pass. Variable-frame-
rate footage requires presentation-timestamp validation; the current
`VideoSource` timestamp is derived from the reported FPS.

## What remains before D4 can close

- Run the new tests on the target Windows environment.
- Inspect output from a small, authorized, representative sequence.
- Compare detector box quality and crop rejection reasons manually.
- Use ground-truth boxes and identity labels to measure annotation match
  coverage and per-track identity purity; do not silently discard unmatched
  observations.
- Measure fragmentation, mixed-identity tracklets, observations per track,
  and usable crop counts across relevant camera views.
- Select crop-size, visibility, sharpness, minimum-observation, and aggregation
  rules from development evidence, not from the provisional defaults.
- Keep raw crops transient unless a separately approved, local-only debug
  workflow requires them; define retention and purge behavior before saving.

D3 remains open independently because downstream use rights for the current
MSMT17-trained checkpoint have not been explicitly cleared. No Market-1501
retrieval or OSNet embedding generation is part of this D4 implementation.
