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

The checkpoint hash and strict state-dict load protect integrity and shape/key
compatibility. The model-zoo source confirms the size and transform convention;
the exact original local Torchreid installation used for the supplied smoke test
was unavailable for direct inspection in this workspace. Therefore, rerun the
adapter against the checkpoint on the target laptop before treating preprocessing
parity as closed.

## Use

Instantiate `OSNetConfig(checkpoint_path=..., expected_sha256=...)` and pass it
to `OSNetEncoder`. Use device `auto` to select CUDA when available and CPU
otherwise; the selected device and adapter version are available in run metadata.
CUDA failure at inference is surfaced; it does not trigger an unreported device
switch.

The local checkpoint remains outside Git. MSMT17 checkpoint data-use terms are
not resolved here, so neither weights nor embeddings are redistributed.
