# CrossCam OSNet adapter

This is a minimal inference adapter for the locally held OSNet x0.25 MSMT17
checkpoint. It is independent of Torchreid's top-level imports and training
engine. No checkpoint is downloaded by CrossCam.

## Model and checkpoint

- Architecture: Torchreid OSNet x0.25, 512-dimensional feature output.
- Training domain: MSMT17, combineall checkpoint named
  `osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`.
- Expected local checkpoint size: 9,336,983 bytes.
- Expected SHA-256: `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.
- Upstream source: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid), commit [f8cd150fdf77e8d9e1ed143b7f308c2c609ded50](https://github.com/KaiyangZhou/deep-person-reid/commit/f8cd150fdf77e8d9e1ed143b7f308c2c609ded50), file `torchreid/models/osnet.py`.
- Upstream license: MIT; full notice is retained in `OSNET_LICENSE.txt`.

The vendored architecture source is from the pinned commit, with a one-line
copyright/license pointer added at its top. It includes unused upstream
pretrained-download helpers. CrossCam only constructs with `pretrained=False`
and does not call those helpers. The adapter itself accepts RGB float tensors
in [0, 1], already resized to (256, 128), and applies Torchreid's standard
ImageNet mean/std. It then explicitly L2-normalizes the raw eval-mode 512-D
feature vectors. No BGR conversion, implicit resizing, or augmentation occurs.

The user verified the checkpoint hash against both the local file and the
corresponding Hugging Face file entry. The file entry displays an MIT license
label, but the checkpoint's downstream terms are not treated as conclusively
cleared: the model author has clarified that pretrained models are for research
use because some training datasets may restrict commercial use, and MSMT17
access requires a signed agreement. Keep these weights local and out of the
Market-1501 evaluation until explicit downstream permission is established.

The user ran a standalone synthetic CUDA smoke test on the target RTX 3050:
567 state tensors loaded strictly; feature dimension 512; batch-1 latency
17.336 ms with 20.38 MiB peak allocated / 26 MiB reserved; batch-4 latency
17.442 ms with 30.14 MiB allocated / 48 MiB reserved. Outputs were finite.
These are synthetic-input measurements, not retrieval metrics.

The repository adapter passed the target-Windows suite (65 tests) and Ruff
(all checks passed) at commit bb5db36486eed29589d511e6a569e15029f40620.
The user subsequently verified CPU fallback and CPU/CUDA numerical parity on
the target laptop for deterministic synthetic batches of 1 and 4. Both passed
the declared tolerances after explicitly setting global FP32, CUDA matmul,
cuDNN, and cuDNN convolution precision to IEEE. Maximum absolute differences
were 1.7881393432617188e-07 (batch 1) and 5.364418029785156e-07 (batch 4);
minimum cosine similarities were 1.0000001192092896 and 1.0 respectively.
The adapter itself has not been changed to force IEEE precision; this setting
was applied in the validation process. Profile the production latency/resource
cost before adopting it as a runtime policy. The preprocessing contract is
RGB, 256x128, ImageNet mean/std, then explicit L2 normalization; no real image
crops or retrieval benchmark have been evaluated.

## Synthetic precision/performance benchmark

Run the local benchmark with:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_osnet_precision.py
```

It compares backend-default and IEEE FP32 settings on CPU and CUDA with
synthetic inputs at batch sizes 1, 4, 8, and 16. It reports median/mean/p95
latency, throughput, and CUDA memory peaks. The target laptop completed all
16 configurations with valid outputs and no OOM. Those first-run timings are
preliminary; the controlled three-round CUDA experiment below is the stronger
precision-mode comparison. No Market-1501 or person imagery is used.

## CUDA precision repeatability benchmark

Use the focused three-round benchmark after pulling the latest branch:

```powershell
.\.venv\Scripts\python.exe .\scripts\repeat_osnet_cuda_benchmark.py
```

It compared default versus IEEE FP32 in alternating order over three rounds,
with identical synthetic inputs and batch sizes 1, 4, 8, and 16. All 24 cases
passed and no CUDA OOM occurred. The paired timing effect changed direction
across batches and rounds, so neither precision mode is consistently faster.
The detailed JSON is saved locally at `outputs/osnet_cuda_repeatability.json`.
The experiment does not load Market-1501 or person imagery.

The initial validated runtime choice for parity-sensitive experiments is
explicit IEEE FP32. This is selected for numerical consistency, not for a
claimed speed advantage. The encoder's default runtime behavior is unchanged
until a full-pipeline profile justifies a deliberate policy change.

## Use

Instantiate `OSNetConfig(checkpoint_path=..., expected_sha256=...)` and pass it
to `OSNetEncoder`. Use device `auto` to select CUDA when available and CPU
otherwise; the selected device and adapter version are available in run metadata.
CUDA failure at inference is surfaced; it does not trigger an unreported device
switch.

The local checkpoint remains outside Git. MSMT17 checkpoint data-use terms are
not resolved here, so neither weights nor embeddings are redistributed.


## D3 validation status

See [D3 validation record](D3_VALIDATION.md) for the measured smoke-test and
CPU/CUDA parity results, checkpoint provenance, rights caveat, and remaining
acceptance checks. D3 remains open pending explicit checkpoint-use clearance.
