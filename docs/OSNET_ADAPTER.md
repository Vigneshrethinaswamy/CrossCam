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

The repository adapter has since passed the target-Windows suite (65 tests)
and Ruff (all checks passed) at commit bb5db36486eed29589d511e6a569e15029f40620.
CPU fallback and CPU/GPU numerical parity have not yet been measured. The
preprocessing contract is RGB, 256x128, ImageNet mean/std, then explicit L2
normalization; it is consistent with the published Torchreid transform
convention, but no real image crops have been evaluated.

## Use

Instantiate `OSNetConfig(checkpoint_path=..., expected_sha256=...)` and pass it
to `OSNetEncoder`. Use device `auto` to select CUDA when available and CPU
otherwise; the selected device and adapter version are available in run metadata.
CUDA failure at inference is surfaced; it does not trigger an unreported device
switch.

The local checkpoint remains outside Git. MSMT17 checkpoint data-use terms are
not resolved here, so neither weights nor embeddings are redistributed.


## D3 validation status

See [D3 validation record](D3_VALIDATION.md) for the measured smoke-test results,
checkpoint provenance, rights caveat, and remaining acceptance checks. D3 remains
open pending target CPU/GPU parity and explicit checkpoint-use clearance.
