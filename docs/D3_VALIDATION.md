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

## Rights and limitations


The Torchreid code repository is MIT-licensed. Separately, the model author clarified that pretrained models are for research use because some training datasets may not permit commercial use. MSMT17 access is controlled through a signed agreement. These facts do not establish a complete downstream grant for this pretrained checkpoint. The local checkpoint must therefore remain private and must not be used for the Market-1501 evaluation until its use for that purpose is explicitly cleared.

Market-1501 was acquired separately for local research-only, non-commercial evaluation, with no redistribution. Its retrieval runner has not been executed. No images, crops, embeddings, videos, or checkpoint files are committed to Git.

## Remaining D3 acceptance checks

- [x] Run adapter CPU inference on the target Windows laptop using the verified checkpoint.
- [x] Compare CPU and CUDA outputs for deterministic synthetic batch sizes 1 and 4 under explicit IEEE FP32 settings; both batches passed max absolute difference ≤ 5e-4, minimum cosine similarity ≥ 0.99999, and maximum unit-norm error ≤ 1e-5.
- [ ] Establish explicit downstream usage permission for the MSMT17 pretrained checkpoint, or replace it with weights whose rights and training-data provenance are adequately documented.
- [ ] Only after checkpoint-use rights are cleared, run the local Market-1501 retrieval protocol and report Rank-1, mAP, and runtime. Do not fine-tune for this L0 evaluation.
