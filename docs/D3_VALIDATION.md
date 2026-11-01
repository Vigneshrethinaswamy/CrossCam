# D3 validation record — OSNet x0.25

**Status: OPEN.** The standalone encoder and repository adapter have passed synthetic CUDA smoke testing, target-Windows tests, and CPU/CUDA numerical parity under explicit IEEE FP32 settings. D3 remains open because downstream rights for the pretrained weights are unresolved.

## Verified checkpoint identity

- Architecture: OSNet x0.25, 512-dimensional embedding.
- Training dataset: MSMT17, `combineall=True`.
- Filename: `osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`.
- Size: 9,336,983 bytes.
- SHA-256: `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.
- The user verified the same hash against the local file and the corresponding Hugging Face file entry.

The Hugging Face file entry displays an MIT license label. This is recorded as source metadata, not as a conclusive determination of downstream weight rights.

## Target-laptop synthetic CUDA smoke test

The user ran the standalone architecture with safe `torch.load(..., weights_only=True)`, strict state-dictionary loading, and synthetic tensors on the NVIDIA RTX 3050 Laptop GPU.

| Check | Result |
|---|---:|
| State-dictionary tensors | 567 |
| Classifier weight | (4101, 512) |
| Feature projection weight | (512, 128) |
| Strict state-dict loading | All keys matched |
| Feature dimension | 512 |
| Batch 1 output | (1, 512), finite |
| Batch 1 latency | 17.336 ms |
| Batch 1 peak allocated / reserved | 20.38 / 26.00 MiB |
| Batch 4 output | (4, 512), finite |
| Batch 4 latency | 17.442 ms |
| Batch 4 peak allocated / reserved | 30.14 / 48.00 MiB |

The test explicitly L2-normalized embeddings before checking unit norms. These are synthetic-input encoder measurements, not Market-1501 retrieval metrics or end-to-end video throughput.

## Adapter implementation verification

On the user's Windows environment, after the adapter branch reached commit `bb5db36486eed29589d511e6a569e15029f40620`:

- `pytest -q`: **65 passed in 8.20s**.
- `ruff check --no-cache .`: **All checks passed**.
- `git diff --check`: passed.
- Working tree: clean and synchronized with `origin/codex/osnet-adapter-benchmark`.

The upstream vendored `src/crosscam/_osnet.py` was preserved unchanged. Ruff ignores only that vendored file; project-owned files are linted.

## Preprocessing

The adapter contract is RGB float input in [0, 1], resized to 256 × 128 (height × width), ImageNet mean/std normalization, and explicit L2 normalization of the 512-D eval-mode features. This matches the documented Torchreid model-zoo input convention and transform specification. The smoke test used this preprocessing on synthetic data; it did not validate real image crops.

## Target-laptop CPU/CUDA parity validation (2026-11-01)

The user ran the adapter against the verified checkpoint on Windows with
PyTorch `2.14.1+cu130` and an NVIDIA GeForce RTX 3050 Laptop GPU. Both CPU
and CUDA inference succeeded. Inputs were deterministic synthetic RGB float
tensors with the adapter's required shape `(N, 3, 256, 128)`, scaled to
[0, 1]. Batch sizes 1 and 4 were tested.

The first comparison using the backend's reported default precision setting
(`none`) missed the strict parity thresholds:

| Batch | Maximum absolute difference | Minimum cosine similarity |
|---:|---:|---:|
| 1 | 0.0005249679 | 0.9999974966 |
| 4 | 0.0011393726 | 0.9999884367 |

The user repeated the same test after explicitly selecting IEEE FP32 for the
global backend, CUDA matmul, cuDNN, and cuDNN convolution precision settings.
The parity thresholds were not changed.

| Measurement | Batch 1 | Batch 4 |
|---|---:|---:|
| Output shape | (1, 512) | (4, 512) |
| Outputs finite | Yes | Yes |
| CPU maximum norm error | 0.0 | 5.960464477539063e-08 |
| CUDA maximum norm error | 5.960464477539063e-08 | 0.0 |
| Maximum absolute CPU/CUDA difference | 1.7881393432617188e-07 | 5.364418029785156e-07 |
| Minimum cosine similarity | 1.0000001192092896 | 1.0 |
| Result | PASS | PASS |

Acceptance criteria for both batches were maximum absolute difference
`<= 5e-4`, minimum cosine similarity `>= 0.99999`, and maximum unit-norm
error `<= 1e-5`. All passed under the explicit IEEE FP32 settings.

This establishes CPU/CUDA numerical consistency for the tested synthetic
inputs, checkpoint, software stack, devices, and batch sizes under the
recorded precision configuration. The diagnostic shows that the explicit
precision setting resolved the observed discrepancy; it does not identify
which lower-level operation caused the original default-setting difference.
The test does not establish real-crop preprocessing equivalence, retrieval
quality, or end-to-end video performance. The strict IEEE setting was applied
inside the validation process; the adapter's production precision policy has
not been changed. Profile latency and memory under this setting before
adopting it as the production default.

## Synthetic precision/performance benchmark

The repository now includes two reproducible local scripts:

- `scripts/validate_osnet_parity.py`: CPU/CUDA numerical consistency for synthetic inputs.
- `scripts/benchmark_osnet_precision.py`: compares backend-default precision
  (`none`) against explicit IEEE FP32 on CPU and CUDA, using deterministic
  synthetic RGB float inputs at batch sizes 1, 4, 8, and 16.

The performance script verifies checkpoint size and SHA-256, warms up each
case, synchronizes CUDA around timed calls, validates finite normalized
512-D outputs, and records median/mean/p95 latency, images per second,
peak allocated/reserved VRAM, and incremental peak memory where available.
Timing covers `OSNetEncoder.encode()`, including its device transfers and
CPU output return. CPU results are included under both precision labels for
completeness; the CUDA FP32 backend settings are the material comparison.

The CPU/CUDA precision benchmark has now been run on the target laptop. Its
first results are recorded below and superseded for precision-mode comparison
by the controlled three-round CUDA experiment.
Run it with:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_osnet_precision.py
```

Use `--iterations 10` for a shorter initial run (minimum 5), and
`--batches 1 4` for a reduced smoke benchmark. Synthetic inputs are not
person images and these measurements are not ReID retrieval metrics.

## Initial synthetic precision/performance run (2026-11-01; preliminary)

The user ran the first version of `scripts/benchmark_osnet_precision.py` on
the target RTX 3050 with 5 warm-up and 20 measured iterations at batch sizes
1, 4, 8, and 16. All 16 CPU/CUDA × precision × batch cases completed with
finite, normalized 512-D outputs and no CUDA OOM.

Initial CUDA median results were:

| Batch | Default precision | IEEE FP32 |
|---:|---:|---:|
| 1 | 28.95 ms | 25.76 ms |
| 4 | 28.05 ms | 25.52 ms |
| 8 | 27.14 ms | 26.35 ms |
| 16 | 28.46 ms | 28.27 ms |

These figures are **preliminary, not production conclusions**. The first run
executed all default cases before all IEEE cases, so precision mode and run
order were confounded. The CUDA caching allocator also retained reserved
blocks across batch configurations, making the old per-case reserved-memory
figures unsuitable for comparison. CPU timings were noisy and should not be
used to conclude that one precision mode is faster.

The benchmark was revised in commit
`2e1709308af7af6b9310c8662d414a6babda9dd6` to remove Ruff findings, pair
precision runs by device and batch size, and clear unused CUDA cache between
cases. It now reports warm-up allocation/reservation, measured peak allocation
and reservation, and the increase above the warm-up baseline separately.
The revised four-way CPU/CUDA benchmark was run by the user; Ruff passed,
and all 16 cases completed with valid outputs. Its initial results are kept
as preliminary context; the later three-round CUDA-only experiment below is
the controlled repeatability comparison.

## Three-round CUDA precision repeatability benchmark

A dedicated CUDA-only script is available at
`scripts/repeat_osnet_cuda_benchmark.py`. It uses one verified OSNet encoder
instance and fixed synthetic RGB float32 tensors generated once per batch size,
then reuses the same tensors across every precision mode and round.

The sequence is deliberately balanced:

| Round | Precision order within each batch | Batch order |
|---:|---|---|
| 1 | Default → IEEE | 1, 4, 8, 16 |
| 2 | IEEE → Default | 16, 8, 4, 1 |
| 3 | Default → IEEE | 1, 4, 8, 16 |

Each case warms up independently, synchronizes CUDA around timed calls, checks
finite normalized 512-D embeddings, and records median/mean/p95 latency,
throughput, effective precision settings, and CUDA allocator memory before and
during measurement. A JSON report is written by default to
`outputs/osnet_cuda_repeatability.json`.

Run on the target Windows machine:

```powershell
.\.venv\Scripts\python.exe -m ruff check --no-cache scripts/repeat_osnet_cuda_benchmark.py
.\.venv\Scripts\python.exe .\scripts\repeat_osnet_cuda_benchmark.py
```

Optional shorter diagnostic:

```powershell
.\.venv\Scripts\python.exe .\scripts\repeat_osnet_cuda_benchmark.py --warmup 3 --iterations 10
```

The default is 5 warm-ups and 20 measured iterations per case (24 cases total).
Three rounds provide descriptive repeatability evidence only, not a statistical
significance test. Performance results remain **pending** until this script is
run on the target laptop. No precision policy change is made to the adapter.

## Three-round CUDA precision repeatability results (2026-11-01)

The user ran `scripts/repeat_osnet_cuda_benchmark.py` on the target Windows
laptop with PyTorch `2.14.1+cu130`, CUDA runtime 13.0, and an NVIDIA RTX 3050
Laptop GPU. The script verified the checkpoint size and SHA-256, reused one
CUDA encoder instance, and reused fixed synthetic inputs for batch sizes 1, 4,
8, and 16 over all three rounds.

- 24/24 configurations passed output shape, finiteness, and unit-normalization checks.
- 480 timed inference iterations completed (20 per configuration).
- No CUDA OOM occurred.
- Ruff reported all checks passed for the repeatability script.
- The JSON report was saved locally to `outputs/osnet_cuda_repeatability.json`; this path is Git-ignored.

### CUDA median latency by round

| Batch | Round 1 default | Round 1 IEEE | Round 2 default | Round 2 IEEE | Round 3 default | Round 3 IEEE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42.080 ms | 41.449 ms | 41.325 ms | 43.859 ms | 42.161 ms | 44.282 ms |
| 4 | 44.871 ms | 43.136 ms | 43.104 ms | 43.851 ms | 49.713 ms | 47.747 ms |
| 8 | 45.569 ms | 45.594 ms | 45.596 ms | 44.941 ms | 47.200 ms | 44.982 ms |
| 16 | 49.937 ms | 54.373 ms | 50.011 ms | 49.945 ms | 48.744 ms | 48.921 ms |

### Paired precision comparison

The percentage is (IEEE median − default median) / default median × 100.
Negative means IEEE was faster; positive means IEEE was slower.

| Batch | Median paired difference | Observed round range | Round-to-round spread |
|---:|---:|---:|---:|
| 1 | +5.03% | −1.50% to +6.13% | 7.63 percentage points |
| 4 | −3.87% | −3.95% to +1.73% | 5.69 percentage points |
| 8 | −1.44% | −4.70% to +0.05% | 4.75 percentage points |
| 16 | +0.36% | −0.13% to +8.88% | 9.01 percentage points |

**Interpretation:** Neither precision mode was consistently faster across batch
sizes and rounds. The paired effects changed sign, and the round-to-round
spread was larger than several of the observed median differences. Three
rounds are descriptive repeatability evidence, not a statistical significance
test; no speed advantage is claimed for either mode.

### CUDA memory

The measured PyTorch peak allocated memory was the same under both precision
modes for each batch size:

| Batch | Peak allocated | Peak reserved |
|---:|---:|---:|
| 1 | 20.76 MiB | 26 MiB |
| 4 | 32.00 MiB | 48 MiB |
| 8 | 47.00 MiB | 70 MiB |
| 16 | 75.13 MiB | 110 MiB |

The allocator's reserved pool was already established during warm-up. The
measured incremental peak reserved memory was 0 MiB in these cases; this does
not mean the process used zero reserved VRAM. The model's warm-up baseline
allocated memory was about 17.13 MiB, and the measured incremental allocated
peak ranged from 3.63 MiB to 58.00 MiB.

### Precision decision

Use explicit IEEE FP32 as the **validated parity configuration** for the next
controlled experiments, because the earlier default-setting CPU/CUDA
comparison missed the declared numerical parity limits while IEEE passed.
This is not a claim that IEEE is faster: the three-round timing results show
no consistent speed winner. The production adapter has not been modified to
force IEEE precision. Before changing runtime defaults, benchmark the complete
pipeline and confirm the relevant numerical behavior with representative
image crops once checkpoint use is cleared.

The benchmark inputs were synthetic; this experiment does not measure person
ReID accuracy, real-crop preprocessing quality, or cross-camera retrieval
performance.

## Rights and limitations

The Torchreid code repository is MIT-licensed. Separately, the model author clarified that pretrained models are for research use because some training datasets may not permit commercial use. MSMT17 access is controlled through a signed agreement. These facts do not establish a complete downstream grant for this pretrained checkpoint. The local checkpoint must therefore remain private and must not be used for the Market-1501 evaluation until its use for that purpose is explicitly cleared.

Market-1501 was acquired separately for local research-only, non-commercial evaluation, with no redistribution. Its retrieval runner has not been executed. No images, crops, embeddings, videos, or checkpoint files are committed to Git.

## Remaining D3 acceptance checks

- [x] Run adapter CPU inference on the target Windows laptop using the verified checkpoint.
- [x] Compare CPU and CUDA outputs for deterministic synthetic batch sizes 1 and 4 under explicit IEEE FP32 settings; both batches passed max absolute difference ≤ 5e-4, minimum cosine similarity ≥ 0.99999, and maximum unit-norm error ≤ 1e-5.
- [x] Complete the three-round CUDA precision repeatability run: 24/24 cases passed, 480 timed inference iterations completed, and no CUDA OOM occurred. No consistent performance winner was observed; IEEE FP32 remains the validated parity configuration, not a speed optimization.
- [ ] Establish explicit downstream usage permission for the MSMT17 pretrained checkpoint, or replace it with weights whose rights and training-data provenance are adequately documented.
- [ ] Only after checkpoint-use rights are cleared, run the local Market-1501 retrieval protocol and report Rank-1, mAP, and runtime. Do not fine-tune for this L0 evaluation.
