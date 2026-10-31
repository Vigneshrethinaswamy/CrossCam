# CrossCam 2.0 — Architecture Review and Decision-Gated ReID Plan

## Document status

**Status: Architecture under review — limited baseline-v1 correctness-fix work is approved under the scope recorded on PR #2; all other implementation remains gated.**

This document incorporates the independent architecture audit of the current MVP and the proposed learned person re-identification (ReID) upgrade. It is a design specification, not an implementation plan that authorizes coding immediately.

- The existing recorded-video MVP is preserved as **baseline-v0**.
- A separately measured **baseline-v1** will contain only approved correctness fixes to candidate generation, timing/overlap constraints, configuration, and result semantics.
- Learned ReID is a separate experimental variant. No improvement claim is valid until it is evaluated against the same labeled split and protocol.
- No encoder integration, broad dependency installation, model checkpoint download, or environment cleanup is authorized by this document alone.
- PR #2 remains a draft until the architecture gates are resolved.

## 1. Objective and non-goals

Build a local-first, offline, multi-camera person-tracklet association system that retrieves plausible cross-camera matches and communicates uncertainty honestly. The primary research question is whether learned person appearance embeddings improve cross-camera retrieval and reviewed link quality over the existing HSV descriptor, under measured constraints.

In scope:
- Recorded video from an initial set of four cameras.
- Camera-local person detection and tracking, tracklet extraction, appearance retrieval, constrained association, review workflow, evaluation, and dashboard.
- Reproducible experiments on the local Windows laptop.
- GPU use when available, with an explicit CPU option/fallback.
- Evidence-backed reporting of accuracy, uncertainty, resource use, and failure modes.

Explicitly out of scope for this milestone:
- Live RTSP ingestion or production deployment.
- Training a ReID encoder from scratch or fine-tuning before a validated baseline exists.
- Face recognition or facial identity matching.
- Automatically confirming or globally merging real-world identities without human review.
- Claiming cross-camera identity accuracy from the existing unrelated sample videos.
- Cloud GPU processing.

## 2. Target environment and operating constraints

- Windows, Python 3.11.
- NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM; 16 GB system RAM.
- Local processing only for this milestone.
- Preserve the current MVP CLI and tests.
- Do not commit source videos, raw person crops, embeddings, checkpoints, or other biometric/personal data.
- No silent switch to an unverified model, random weights, or CPU execution. Device choice and fallback must be visible in run metadata.
- Mixed precision stays disabled until numerical agreement and resource benefit are measured.
- Process cameras sequentially initially and reuse model instances; bounded batch sizes and OOM handling are required.

## 3. Architecture

### 3.1 Target pipeline

`RunManifest → VideoSource → FrameSampler → PersonDetector → CameraLocalTracker → TrackletBuilder/CropSelector → ReIDEncoder → EmbeddingStore → EligibilityFilter → AppearanceRetriever(top-k) → PairScorer → PairwiseAssigner → DecisionPolicy → ReviewQueue → IdentityHypotheses → Evaluation + Dashboard`

Cross-cutting services:
- **RunManifest:** immutable run ID, input video hashes, config hash, code revision, model/checkpoint hash, timestamps/offset source, and authorization reference.
- **EvalHarness:** split definition, label mapping, baseline variants, metrics, confidence intervals, and exported failure cases.
- **ArtifactPolicy:** local cache locations, retention period, purge behavior, and Git exclusion rules.
- **Diagnostics:** camera-pair coverage, tracklet purity/fragmentation, candidate-generation recall, runtime, RAM, and VRAM.

### 3.2 Component boundaries

Keep replaceable adapters for detector, local tracker, ReID encoder, pair scorer, and storage. Keep crop quality, embedding aggregation, eligibility checks, feature construction, decisions, and metrics as testable functions where practical.

The first ReID milestone keeps the current local tracker only if tracklet-purity and fragmentation measurements show it is usable. Do not add motion prediction preemptively; revisit it if measured fragmentation or identity contamination warrants it.

### 3.3 Stable identities and provenance

Use `(run_id, camera_id, local_track_id)` as the unique tracklet key. A persistent store must not overwrite records from separate runs. Every descriptor must record encoder/checkpoint ID, preprocessing version, crop count, aggregation method, and quality summary.

Association output must expose separate fields for:
- raw appearance similarity;
- timing/topology features;
- uncalibrated ranking score, if used;
- calibrated probability only if calibration has been evaluated;
- decision state and reason codes;
- assignment/conflict information.

Do not encode score explanations only as free-form strings.

## 4. Baselines and defect isolation

### baseline-v0 — frozen current behavior

Record the current behavior and outputs as an immutable comparison point before correctness fixes. Do not retroactively label its heuristic output as validated identity matches. Preserve its source revision and representative outputs.

### baseline-v1 — corrected heuristic baseline

Fix and test correctness issues before comparing learned ReID:
1. Candidate generation must not stop merely because one or more weak candidates exist; widening must add eligible candidates rather than hide them.
2. Widened candidates must not receive an artificially maximal topology prior.
3. Default topology settings must distinguish “unknown/no evidence” from a known transition; a constant prior must not be presented as useful topology evidence.
4. Explicitly model overlapping camera fields of view. Simultaneous tracklets can be valid cross-camera observations when overlap is possible; simultaneity is a hard exclusion only when the camera relationship and timing make a shared identity impossible.
5. Remove or implement dead configuration keys; eliminate hard-coded widening horizons; make decision margin behavior match the documentation.
6. Reject duplicate camera-pair edges/configurations rather than silently overwriting them.
7. Define time units clearly. Track retention should be based on elapsed seconds (or explicitly converted sampling cadence), not an unexplained count of sampled frames.
8. Make missing timestamp offsets explicit. Do not silently treat unknown offsets as zero.
9. Add regression tests for weak-candidate widening, topology priors, overlapping cameras, duplicate edges, and decision margins.

baseline-v1 must remain a heuristic baseline. Correcting its bugs does not make its scores probabilities or prove cross-camera identity accuracy.

## 5. ReID encoder strategy

### 5.1 Candidate, not a final selection

OSNet is the first candidate family to investigate, not an approved dependency yet. Before selection, compare a small, controlled bake-off:
- OSNet x1.0;
- OSNet-AIN x1.0;
- OSNet x0.25 (and x0.5 only if useful for the resource/quality trade-off).

Do not choose the smallest model by default. Select based on measured retrieval quality, checkpoint provenance, preprocessing correctness, and local resource constraints. Published model-zoo metrics are reference context only; they are not CrossCam performance.

### 5.2 Source, license, and checkpoint security gate

Before downloading or loading weights, document:
- upstream source repository and exact commit/release;
- model code license and retained copyright/license notices;
- exact checkpoint URL/source, model card, intended training domain, and redistribution terms;
- SHA-256 checksum and file size;
- required input color order, resize, normalization, and output dimension;
- checkpoint loading path using a state-dict-only format and safe loading options where supported;
- any training-dataset restrictions that affect use or redistribution.

No automatic checkpoint download in the processing pipeline. A missing or hash-mismatched checkpoint must fail clearly. Do not use unsafe deserialization options on unverified checkpoints. Keep model files outside Git and record the verified hash in a local or committed text manifest only when doing so does not redistribute the weight itself.

### 5.3 Dependency policy

Do not continue installing transitive dependencies to make `torchreid` import. The current `torchreid==0.2.5` install path pulled in missing packages and has not been validated for this environment. Before any environment cleanup:
- capture the current environment state;
- compare package changes against the known-good PyTorch/Torchvision CUDA setup;
- prepare a reversible change plan;
- obtain explicit approval before uninstalling or replacing packages.

Preferred direction to investigate: a narrow, audited inference adapter around minimal OSNet model code, with the upstream license and commit retained. This is conditional on source review and a verified checkpoint. ONNX Runtime is only an alternative if CUDA compatibility and embedding agreement are tested on the target machine.

### 5.4 Encoder contract

`encode(crops) -> embeddings`

The adapter must:
- accept RGB person crops and apply the exact checkpoint preprocessing;
- support bounded batches and explicit device selection;
- return a finite, predictable `(N, D)` tensor with L2-normalized rows;
- run in inference mode and report device, model ID, preprocessing ID, and batch size;
- validate empty crops, clipped boxes, malformed inputs, NaN/Inf outputs, and batch boundaries;
- have CPU/GPU consistency tests with a declared numerical tolerance;
- never silently replace missing weights with random initialization.

## 6. Tracklets, crops, and local tracking

### 6.1 Tracklet quality comes before ReID claims

A mixed-identity tracklet contaminates its embedding and can make any ReID model appear unreliable. Measure tracklet purity, observation count, fragmentation, and crop quality before evaluating cross-camera association.

### 6.2 Crop selection and aggregation

- Clamp bounding boxes to image bounds and reject empty or invalid crops.
- Exclude crops below a documented minimum size and flag severe blur/occlusion when measurable.
- Use several temporally distributed, usable crops per tracklet.
- Compare a normalized mean against a quality-weighted or robust aggregation only on the development split; do not assume mean aggregation is optimal.
- Store embeddings and crop metadata by default, not raw crops.
- Cache keys must include video hash, detector/tracker configuration, crop-selection version, encoder/checkpoint hash, and preprocessing version.
- Cache invalidation must be deterministic.

### 6.3 Local tracker limits

The current greedy IoU tracker has no motion prediction and can fragment tracks or switch IDs. For this milestone, measure those failures first. Express track retention in seconds, require a minimum number of observations for an eligible tracklet, and report low-purity/mixed tracklets. Add motion prediction or a different tracker only if error analysis justifies the complexity.

## 7. Time, topology, and candidate eligibility

### 7.1 Time semantics

- Define whether each timestamp is video-relative or wall-clock time.
- Record each camera offset, its source, and uncertainty.
- Unknown offset is an explicit state, not offset zero.
- Validate frame-rate assumptions. `frame_index / fps` is acceptable only for constant-frame-rate footage whose metadata is trustworthy; otherwise use presentation timestamps or document the limitation.
- Do not use offsets inferred from unrelated sample videos for identity evaluation.

### 7.2 Camera-pair overlap

Camera topology must distinguish at least: known overlapping, known non-overlapping, known transition, and unknown relationship. Unknown topology must not silently become a strong positive prior or an exclusion rule.

### 7.3 Two-stage candidate generation

**Stage A — hard eligibility constraints**
- Exclude same-camera pairs that represent the same-time duplicate tracklet hypotheses when the data model makes them mutually exclusive.
- Exclude simultaneous pairs across cameras only when trusted timing and known non-overlap make the pair physically impossible.
- Apply only documented hard impossibility constraints. Unknown transitions and atypical travel times remain eligible.

**Stage B — appearance retrieval**
- Retrieve top-k cross-camera candidates using embeddings within a broad, documented temporal search horizon.
- Choose k and any search window from development-set recall@k and workload measurements.
- Measure candidate recall with and without topology/time pruning. Topology must not silently discard plausible true matches.
- Treat elapsed time and topology as pairwise scorer features unless a hard physical impossibility is established.

## 8. Pair scoring, assignment, and decisions

### 8.1 Scoring

Start with a small, interpretable pairwise scorer fitted on labeled development data. Candidate features may include:
- aggregate/max cosine similarity;
- elapsed time and its uncertainty;
- topology state;
- source and destination tracklet quality;
- crop/observation counts.

Compare against the corrected heuristic baseline. Do not assume a global exponential time-decay constant or manually selected weights generalize across camera pairs. Use per-camera-pair timing distributions only if the data supports estimating them.

### 8.2 Assignment and conflicts

For each appropriate camera-pair/time window, use rectangular assignment with explicit dummy “no match” choices so the system is not forced to link every tracklet. SciPy's assignment routine is a candidate if its dependency is justified. Enforce one-to-one constraints only where the camera/time formulation supports them. Record conflicts and do not claim this pairwise assignment solves global identity clustering or triangle consistency.

Global graph clustering, multicut/correlation clustering, and cross-window global optimization are deferred until pairwise results justify them.

### 8.3 Decision policy

The user-facing decision states are:
- **Accept for review:** strong evidence under a threshold selected on development/calibration data, mutual-best support, sufficient margin, and no known conflict.
- **Defer:** ambiguous, insufficient-quality, poorly calibrated, or conflicting evidence.
- **Reject:** a hard physical contradiction or evidence below a validated rejection rule.

These are review recommendations, not automatic identity confirmation. Thresholds and margins must be selected on a calibration split and reported with coverage and error rates. Show component evidence for every candidate.

Do not display an uncalibrated rank score as a probability. If the dataset is large enough, evaluate calibration (e.g. reliability plot, Brier score, ECE, and identity-level bootstrap intervals) before exposing a probability. Otherwise retain uncalibrated scores and state that limitation.

## 9. Dataset and ground-truth gate

**No dataset has been approved yet.** Dataset selection is a blocking decision, not a task to defer until after implementation.

### Required dataset audit

For each candidate, record:
- license, access restrictions, permitted research/use, and redistribution terms;
- number of cameras, identities, sequences, annotated frames, and camera pairs;
- annotation format, identity continuity, bounding-box availability, timestamps, and synchronization;
- download size and feasibility on the target machine;
- whether the annotations can support the metrics and split plan below.

Do not assume any dataset's suitability or license from a paper summary alone. Confirm against the authoritative dataset source before downloading.

### Evaluation layers

- **L0 — encoder/preprocessing sanity:** use a license-cleared, labeled ReID benchmark to test color order, resizing, normalization, checkpoint loading, and retrieval metric implementation. Set a reproduction tolerance before running. This validates the pipeline, not performance on CrossCam footage.
- **L1 — labeled multi-camera benchmark:** use identity labels, camera IDs, boxes, and trustworthy timing to measure retrieval and pairwise linking.
- **L2 — own consented capture:** if needed, record a small, consented 3–4-camera sequence with a known camera layout, controlled synchronization, and manually checked identity labels. This is required to make claims about the user's own camera setup if public data is not representative.

Existing sample.mp4/sample1.mp4/sample2.mp4 files are not an identity-labeled evaluation set and must not be used to claim cross-camera identity accuracy.

### Ground-truth derivation and splits

- Match predicted boxes to annotated boxes by a declared IoU rule.
- Derive tracklet identity from matched annotations and report track purity; mark low-purity/mixed tracklets rather than quietly assigning a majority label.
- A true cross-camera link requires two eligible tracklets from different cameras with the same ground-truth identity.
- Split by identity and, where relevant, session/time to avoid leakage. Keep development, calibration, and final test roles separate. If sample size is too small, use grouped cross-validation and label the results exploratory.
- Do not tune thresholds on the final test split.

## 10. Evaluation and acceptance criteria

### Metrics

Report the metrics supported by the chosen annotations:
- ReID retrieval: Rank-1, mAP, and recall@k;
- candidate-generation recall before scoring, with/without topology pruning;
- pair-link precision, recall, false-match/false-merge rate, and per-camera-pair results;
- accept coverage versus selective risk for the review policy;
- tracklet purity, fragmentation, observation count, and crop-quality breakdown;
- latency/throughput, RAM, peak VRAM allocated and reserved, and device used;
- calibration metrics only if a calibration procedure is supported by sufficient data.

Use paired bootstrap intervals grouped by identity where feasible. Do not make an improvement claim unless the comparison protocol is fixed, the interval supports improvement, more than one relevant camera pair is represented, and false-merge behavior is not worsened at comparable coverage. If the number of identities is inadequate, report descriptive results only.

### Variants

1. baseline-v0: frozen current behavior.
2. baseline-v1: corrected heuristic candidate generation/association.
3. ReID-only: appearance retrieval/scoring.
4. ReID + time/topology features.
5. ReID + assignment/conflict handling.
6. Decision/review policy with accept/defer/reject.

Add each stage incrementally so gains and regressions can be attributed to a specific change.

### Minimum release gates

- Existing MVP tests and CLI flows remain functional.
- Regression tests cover the known candidate-generation and timing defects.
- Encoder source, license, preprocessing, checkpoint hash, and loading method are documented.
- Embeddings are finite, normalized, deterministic within declared tolerance, and correctly shaped.
- CPU execution works; GPU memory behavior is measured on the target laptop.
- Authorized labeled data and a reproducible split support reported metrics.
- Scores and dashboard labels distinguish uncalibrated similarity/ranking from calibrated probability.
- No automatic global identity merge; human review remains the final authority.
- No personal video, raw crop, embedding, or checkpoint is committed to Git.

## 11. Privacy, storage, and operational safety

- Default to no raw crop persistence. Debug crop saving must be opt-in, local-only, time-limited, and purgeable.
- Treat embeddings as biometric-derived personal data; do not upload them to external services or commit them to Git.
- Document consent/authorization, intended use, access controls, retention duration, and deletion procedure before own-footage collection.
- Extend ignore rules for generated arrays, model weights, serialized objects, debug images, and output/cache directories; verify with Git status and a repository scan before commits.
- Run manifests should record input hashes rather than embedding sensitive paths or personal content unnecessarily.
- Dashboard wording should say “candidate link,” “appearance similarity,” and “uncalibrated ranking score” as appropriate. Avoid “same person” unless a human has confirmed the hypothesis.

## 12. Reproducibility and resource checks

- Pin and document the Python and direct dependency versions after compatibility tests; do not rely on incidental transitive installs.
- Declare direct runtime dependencies such as PyTorch/Torchvision and pandas where they are actually required.
- Add a lock/constraints strategy appropriate for Windows and the selected CUDA wheel set.
- Record both PyTorch peak allocated and reserved memory, plus external GPU readings where available; allocated memory alone omits other GPU usage.
- Use bounded batches and retry smaller batches on OOM with explicit logs. Do not silently change device.
- Run full pytest and Ruff on the target environment, not only a separate Linux sandbox.

## 13. Decision gates — must be resolved before implementation

The following are deliberately open. They must be recorded in a decision log with rationale and evidence.

| Gate | Decision required | Required evidence |
|---|---|---|
| D1 — Dataset | Choose L0 benchmark and L1 benchmark, or approve a consented L2 capture plan | License/access audit, annotation and synchronization audit, size and camera/identity counts |
| D2 — Baselines | Confirm frozen baseline-v0 and exact scope of baseline-v1 fixes | Baseline revision/output snapshot and tests for each known defect |
| D3 — Encoder | Select OSNet x1.0, OSNet-AIN x1.0, x0.25/x0.5, or an evidence-backed alternative | Source/license/checkpoint hash, preprocessing verification, L0 results and local resource measurements |
| D4 — Tracklets | Define minimum observations, crop size/quality rules, aggregation and purity threshold | Development-set purity/fragmentation and crop-quality analysis |
| D5 — Timing/topology | Define offset source/uncertainty and camera-pair overlap states | Recording metadata and camera-layout evidence |
| D6 — Association | Choose top-k, scorer features, assignment windows, and conflict policy | Candidate recall, development-set comparisons, and failure review |
| D7 — Review policy | Set false-merge target, coverage goal, margins, and defer behavior | Calibration results or explicit statement that probability calibration is unsupported |
| D8 — Privacy | Set retention/purge period and consent/authorization process | Written local data-handling policy |
| D9 — Storage | Decide whether TrackStore is implemented now or removed from the first milestone | Required dashboard/evaluation use cases and unique-key/run-isolation design |
| D10 — Environment | Approve dependency changes and the exact reproducible environment plan | Before/after package diff and rollback plan |


## D1 — Dataset decision: NVIDIA PhysicalAI Smart Spaces (approved for the initial benchmark)

### Selected dataset and pinned revision

**Initial labeled multi-camera benchmark (L1):** NVIDIA PhysicalAI Smart Spaces, `MTMC_Tracking_2024/train/scene_002`, pinned to dataset revision `483cf8a1ad0bc1ea871628268e2fc39d142438ad`.

- [Official dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
- [Pinned scene directory](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/tree/483cf8a1ad0bc1ea871628268e2fc39d142438ad/MTMC_Tracking_2024/train/scene_002)
- [Pinned ground truth](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/blob/483cf8a1ad0bc1ea871628268e2fc39d142438ad/MTMC_Tracking_2024/train/scene_002/ground_truth.txt)
- [Pinned shared calibration](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/blob/483cf8a1ad0bc1ea871628268e2fc39d142438ad/MTMC_Tracking_2024/train/scene_002/calibration_2025_format.json)

The dataset card identifies the data as synthetically generated and states that the multi-camera dataset is time-synchronized. Its 2024 annotation schema is camera ID, global object ID, zero-based frame index, 2D box `xmin ymin width height`, and projected world `x y`. The completed local audit found 1,932,341 valid observations, nine cameras (11–19), and 23 global IDs; all 23 IDs appear in every camera. No malformed rows or invalid/non-positive boxes were found. The downloaded ground truth is 80.18 MB and its SHA-256 matches the pinned source: `e7039be4ff3d3fa1b499dd3b39d42081b5affe0fe7e1df0440e6a457893e2f3e`.

### Calibration schema and camera mapping

The shared `calibration_2025_format.json` is 23,584 bytes (source SHA-256 `ffd0fbc4a0c6015f85520501109f83193d396f0831a16a5395003187d1a70274`). It uses `calibrationType: cartesian` and a `sensors` array with entries `type: camera`; sensor IDs `Camera_0011`–`Camera_0019` map by numeric suffix to annotation IDs 11–19 and folders `camera_0011`–`camera_0019`. Each entry reports 30 FPS and 1920×1080 dimensions (as string-valued attributes), plus intrinsic, extrinsic, camera, and homography matrices. The 2024 data does not provide all of the newer schema’s camera coordinates, direction, or scale-factor metadata, as the dataset card cautions. The per-camera `calibration.json` listings are empty at this revision; use the shared scene calibration file.

Annotation frame indices divided by 30 provide a provisional synchronized timeline because the card states synchronization and calibration reports 30 FPS. There are no per-camera wall-clock timestamps in this annotation format. Before interpreting travel times, confirm video frame rate and presentation timestamps from the selected files; retain timing uncertainty in the run manifest.

### Camera subset decision

Use cameras **12, 14, 17, and 19** as the first manageable four-view subset. The prior annotation audit ranked these highest for pairwise same-frame identity overlap. That makes them sensible for an initial same-time cross-view appearance/retrieval test, with six camera pairs and all 23 labeled identities represented. It does not establish that they are the best set for camera-transition association: selecting on simultaneous overlap can favor views that see the same people at once. Keep topology and travel-time claims provisional, and include a lower-overlap/transition pair in a later experiment if the videos and calibration support it.

The four listed videos total approximately **1.0 GB** (254.79, 261.14, 249.79, and 234.40 MB respectively), which is feasible for staged local processing on the target machine. Do not fetch them until implementation approval. Keep all data and derived crops/embeddings local and outside Git.

### License, attribution, and limits

The pinned Hugging Face dataset card displays **Creative Commons Attribution 4.0 International (CC BY 4.0)** for the dataset. The license permits sharing and adaptation, including commercial use, subject to attribution: credit NVIDIA / the dataset, link the license, indicate modifications, and avoid implying endorsement. Use a citation from the dataset card (the 8th AI City Challenge paper is listed for the 2024 release) in any public report. Recheck the pinned card and terms before redistribution. The footage is synthetic, so this benchmark tests behavior on synthetic people and does not establish performance or safety for real-world person tracking.

### D1 status

**Closed for the initial benchmark choice:** scene_002 at the pinned revision, using cameras 12, 14, 17, and 19 for the first controlled four-camera study. Annotation schema, identity coverage, calibration mapping, dataset synchronization claim, file hash, size, and displayed license have been checked. Video decoding/PTS validation remains a preflight item for after approval to download the selected streams; no videos or model checkpoints have been downloaded.
## 14. Milestones after the gates are approved

- **A0 — Freeze and correct baseline:** capture baseline-v0; specify baseline-v1; fix and test candidate generation, overlap, topology prior, timing, and config defects.
- **A1 — Source and environment verification:** review encoder code/checkpoint provenance; finalize a reversible dependency plan; pass the L0 model/preprocessing sanity gate.
- **A2 — Dataset and labels:** complete license/annotation audit, identity-disjoint split, and label/track-purity pipeline.
- **A3 — Tracklets and cache:** implement crop selection, quality-aware aggregation, provenance, cache invalidation, and bounded memory behavior.
- **A4 — Retrieval and association:** implement eligibility, top-k retrieval, scorer comparison, assignment, and decision policy.
- **A5 — Review and dashboard:** add human-confirmed identity hypotheses, transparent evidence, and risk/coverage reporting.
- **A6 — Final audit:** resource, privacy, reproducibility, regression, and documentation review.

A milestone is not complete merely because its code runs; its stated tests and evidence gates must pass.

## 15. Current stop line

Until D1–D10 are resolved, the permitted work is architecture review, source/license research, dataset feasibility assessment, decision-log updates, and non-mutating inspection of the existing implementation. Do not start ReID implementation, download checkpoints, continue installing `torchreid` dependencies, alter the local environment, or merge PR #2.

## 16. Decision log

| ID | Status | Current position |
|---|---|---|
| D1 Dataset | CLOSED | NVIDIA PhysicalAI Smart Spaces, pinned 2024 scene_002; cameras 12/14/17/19 for the initial four-view benchmark; synthetic-only validity and timestamp caveats documented |
| D2 Baselines | IN PROGRESS — BASELINE-V1 ONLY | baseline-v0 remains frozen; isolated correctness fixes and regression tests are being implemented on `codex/baseline-v1-correctness`; ReID, downloads, dependency/schema changes, and PR #2 merge remain blocked |
| D3 Encoder | OPEN | OSNet family is a candidate, not selected |
| D4 Tracklets | OPEN | Measure purity/fragmentation before locking rules |
| D5 Timing/topology | OPEN | Explicit offsets, uncertainty, and overlap states required |
| D6 Association | PROPOSED | Eligibility → top-k retrieval → learned/fit pair scorer → assignment |
| D7 Review policy | PROPOSED | Accept/defer/reject; no automatic global merge |
| D8 Privacy | OPEN | Local-only; no raw crops by default; retention duration to be decided |
| D9 Storage | OPEN | Decide whether TrackStore is needed in first milestone |
| D10 Environment | OPEN | No package removal/install changes authorized |
