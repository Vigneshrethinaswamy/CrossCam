# CrossCam 2.0 — Learned ReID Upgrade Architecture

## Status and objective

This document defines the next implementation milestone for CrossCam. The existing recorded-video MVP remains the baseline and must continue to work unchanged. The goal is to replace HSV histograms as the primary appearance descriptor with a pretrained person re-identification (ReID) encoder, then evaluate cross-camera retrieval and uncertainty using labeled, authorized footage.

This is an architecture and implementation plan, not a claim that cross-camera identity accuracy is already validated.

## Constraints

- Target: Windows, Python 3.11, RTX 3050 Laptop GPU with 4 GB VRAM, 16 GB system RAM.
- No cloud GPU; CPU fallback must remain supported.
- Recorded-video/offline processing first; no RTSP work in this milestone.
- Do not commit videos, person crops, embeddings, checkpoints, or other biometric/personal data.
- Preserve the current HSV pipeline as a baseline for fair comparison.
- Model weights must be downloaded only from a documented source after checking license, model card, and redistribution terms. Keep downloaded weights outside Git and record a checksum/source note.

## Proposed pipeline

`VideoSource -> FrameSampler -> PersonDetector -> CameraLocalTracker -> TrackletBuilder -> ReIDEncoder -> CandidateGenerator -> AssociationEngine -> TrackStore -> Evaluation + Dashboard`

### 1. Tracklet representation

Each tracklet will contain:
- camera ID and local track ID;
- first/last timestamps and observation count;
- normalized bounding-box/crop metadata;
- a robust aggregate appearance vector computed from several usable crops;
- descriptor provenance (encoder name/version, preprocessing version, crop count);
- quality diagnostics such as crop size, blur/occlusion flags where available.

Do not persist raw crops by default. If debug crops are enabled, make them opt-in and local-only with an explicit retention policy.

### 2. ReID encoder adapter

Create a narrow encoder interface so model choices can be swapped without rewriting association logic:

- `encode(crops) -> embeddings`;
- batch inference with configurable batch size;
- automatic `cuda` when available, otherwise CPU;
- inference-only mode and mixed precision only when supported and verified;
- deterministic output shape, finite values, L2-normalized embeddings;
- clear startup error if weights are missing or incompatible; never silently substitute a random or generic model.

Initial candidate: a compact pretrained OSNet variant (start by evaluating x0.25 or x0.5) from a reputable person-ReID implementation. Before implementation is locked, verify the exact model/weight source, license, preprocessing, and compatibility with Python 3.11/PyTorch on Windows. If this dependency is too brittle, select an alternative pretrained ReID checkpoint with clear licensing and a documented inference API. Do not train from scratch on the laptop.

### 3. Crop preprocessing and quality

- Convert detector boxes to safe image bounds and reject empty crops.
- Resize and normalize exactly as required by the selected checkpoint.
- Use several crops per tracklet when available; discard tiny or severely blurred crops.
- Aggregate normalized crop embeddings with a documented robust method (initially normalized mean); retain crop count and quality statistics.
- Unit-test edge cases: clipped boxes, empty crops, invalid tensors, NaN/Inf outputs, batch-size boundaries, and CPU fallback.

### 4. Candidate generation

Keep camera topology and elapsed-time gates as soft priors, not absolute identity rules. Generate candidates only when timing and camera transitions are plausible, but provide a configurable broad-search fallback. Require explicit camera offsets derived from recording metadata. The smoke-test offsets used for unrelated sample videos are not evidence of synchronization and must not be used for identity evaluation.

### 5. Association and uncertainty

Use cosine similarity between normalized ReID embeddings as the appearance feature. Combine it with elapsed-time compatibility and camera-topology priors in an explainable candidate ranker. Keep score components separately visible.

Important rules:
- Scores are ranking values, not probabilities, until calibrated on a held-out labeled set.
- Do not auto-merge global identities solely because a heuristic threshold was crossed.
- Preserve accept/defer/reject as review states. Defer ambiguous cases and show the evidence behind each suggestion.
- Prevent conflicting global assignments and consider one-to-one matching constraints within each association window.
- Log model version, preprocessing version, configuration, and score components for reproducibility.

### 6. Evaluation design

Use a small authorized dataset sequence with cross-camera identity labels and known camera timing before making identity-performance claims. Confirm the dataset's access rules and license before downloading or redistributing anything.

Report at minimum:
- detection and local-tracking diagnostics separately from ReID;
- cross-camera retrieval: Rank-1 and mAP where the annotation format supports them;
- candidate-link precision/recall and false-match rate at explicitly stated thresholds;
- selective risk versus coverage for accept/defer decisions;
- per-camera-pair results and failure examples;
- runtime, throughput, and peak VRAM on the target laptop.

Use a fixed, documented split or sequence protocol; avoid tuning thresholds on the final test set. Compare at least:
1. HSV baseline;
2. ReID appearance only;
3. ReID + topology/time priors;
4. uncertainty-aware review policy.

Do not report identity accuracy from unrelated videos, arbitrary offsets, or unlabeled footage.

## Implementation milestones

### R1 — Model and data preflight
- Confirm the dataset and license, annotation schema, synchronization, and camera IDs.
- Verify ReID model/checkpoint provenance and dependency compatibility.
- Add a small preflight script that checks checkpoint availability, device, embedding shape, and VRAM usage.

### R2 — Encoder adapter
- Add the encoder interface and the selected pretrained implementation.
- Add CPU/GPU inference, batched crop embedding, normalization, and validation tests.
- Keep HSV extraction working as a baseline.

### R3 — Tracklet integration
- Add learned embeddings to tracklet extraction with explicit encoder metadata.
- Cache only derived local experiment artifacts outside Git; invalidate cache when model/preprocessing changes.
- Update the association ranker to consume normalized embeddings and preserve explainable components.

### R4 — Evaluation and review
- Add retrieval/link metrics, camera-pair breakdowns, and error review exports.
- Compare the four variants above on labeled data.
- Update dashboard labels so uncalibrated scores cannot be mistaken for probabilities.

### R5 — Resource and reproducibility audit
- Measure throughput and peak VRAM on the RTX 3050.
- Test CPU fallback and bounded memory behavior.
- Run Ruff and the full pytest suite; document exact commands and measured results.

## Acceptance criteria

The upgrade is ready for a release only when:
- the original MVP tests and CLI flows remain functional;
- the encoder has deterministic shape/normalization tests and verified checkpoint provenance;
- CPU fallback works;
- model weights and personal data are excluded from Git;
- evaluation runs on authorized labeled data and reports reproducible metrics;
- dashboard scores are clearly labeled as uncalibrated unless calibration has been measured;
- no identity-accuracy claim is made without ground-truth labels.

## Explicit non-goals for this milestone

- Live RTSP ingestion or production deployment.
- Training a ReID model from scratch.
- Face recognition or facial identity matching.
- Automatic real-world identity confirmation without human review.
- Claiming generalization based only on the three existing unrelated sample videos.
