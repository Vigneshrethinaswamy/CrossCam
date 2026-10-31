"""Verified local OSNet x0.25 inference adapter (no Torchreid package imports)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ._osnet import osnet_x0_25

EXPECTED_CHECKPOINT_SHA256 = (
    "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"
)
INPUT_HEIGHT = 256
INPUT_WIDTH = 128
FEATURE_DIM = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class OSNetConfig:
    """All choices that affect checkpoint identity, preprocessing, or execution."""

    checkpoint_path: Path
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 4

    def __post_init__(self) -> None:
        if len(self.expected_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.expected_sha256
        ):
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


class OSNetEncoder:
    """Load only the verified state dict and emit normalized 512-D embeddings."""

    def __init__(
        self,
        config: OSNetConfig,
        *,
        model_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model_id = "osnet_x0_25_msmt17_combineall"
        self.preprocessing_id = "rgb_256x128_imagenet_norm_v1"
        self._model_factory = model_factory or (
            lambda num_classes: osnet_x0_25(
                num_classes=num_classes, pretrained=False, loss="softmax"
            )
        )
        state = self._load_verified_state(config.checkpoint_path, config.expected_sha256)
        classifier = state.get("classifier.weight")
        feature_layer = state.get("fc.0.weight")
        if not isinstance(classifier, Tensor) or classifier.ndim != 2:
            raise ValueError("checkpoint is missing a valid classifier.weight")
        if tuple(classifier.shape[1:]) != (FEATURE_DIM,):
            raise ValueError("checkpoint feature dimension is not 512")
        if not isinstance(feature_layer, Tensor) or tuple(feature_layer.shape) != (
            FEATURE_DIM,
            128,
        ):
            raise ValueError("checkpoint does not match OSNet x0.25 feature layer")
        model = self._model_factory(int(classifier.shape[0]))
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        self._model = model
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose device='cpu'")
        return torch.device(requested)

    @staticmethod
    def _load_verified_state(path: Path, expected_sha256: str) -> dict[str, Tensor]:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"OSNet checkpoint not found: {checkpoint}")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("OSNet checkpoint SHA-256 does not match expected_sha256")
        obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict):
            raise ValueError("checkpoint must contain a state-dictionary object")
        state = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not isinstance(state, dict) or not state:
            raise ValueError("checkpoint has no state dictionary")
        if any(
            not isinstance(key, str) or not isinstance(value, Tensor)
            for key, value in state.items()
        ):
            raise ValueError("state dictionary must contain only string keys and tensors")
        if any(key.startswith("module.") for key in state):
            raise ValueError("DataParallel module. prefixes are not accepted")
        return state

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "checkpoint_sha256": self.config.expected_sha256,
            "preprocessing_id": self.preprocessing_id,
            "input_shape": (3, INPUT_HEIGHT, INPUT_WIDTH),
            "feature_dim": FEATURE_DIM,
            "device": str(self.device),
            "batch_size": self.config.batch_size,
            "l2_normalized": True,
        }

    @torch.inference_mode()
    def encode(self, rgb: Tensor) -> Tensor:
        """Encode RGB float images (N,3,256,128), scaled to [0,1]."""
        if not isinstance(rgb, Tensor):
            raise TypeError("rgb must be a torch.Tensor")
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError("rgb must have shape (N, 3, 256, 128)")
        if rgb.shape[0] == 0:
            raise ValueError("rgb batch must not be empty")
        if not rgb.is_floating_point():
            raise TypeError("rgb must be a floating-point tensor scaled to [0,1]")
        if not bool(torch.isfinite(rgb).all()):
            raise ValueError("rgb contains NaN or Inf")
        if bool((rgb < 0).any()) or bool((rgb > 1).any()):
            raise ValueError("rgb values must be in [0,1]")

        outputs = []
        mean = self._mean.to(device=self.device)
        std = self._std.to(device=self.device)
        for start in range(0, rgb.shape[0], self.config.batch_size):
            batch = rgb[start : start + self.config.batch_size].to(
                device=self.device, dtype=torch.float32
            )
            features = self._model((batch - mean) / std)
            if not isinstance(features, Tensor) or features.ndim != 2:
                raise RuntimeError("OSNet must return a 2-D feature tensor in eval mode")
            if features.shape != (batch.shape[0], FEATURE_DIM):
                raise RuntimeError("OSNet returned an unexpected feature shape")
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("OSNet produced NaN or Inf features")
            features = features.float()
            norms = torch.linalg.vector_norm(features, ord=2, dim=1)
            if bool((norms <= 1e-12).any()):
                raise RuntimeError("OSNet produced a zero-norm feature")
            features = F.normalize(features, p=2, dim=1, eps=1e-12)
            if not bool(torch.isfinite(features).all()):
                raise RuntimeError("normalized OSNet features are not finite")
            outputs.append(features.cpu())
        return torch.cat(outputs, dim=0)

