"""Compact pretrained person detector backed by Torchvision SSDLite."""
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Detection:
    """One person bounding box in source-frame pixel coordinates."""

    box: tuple[float, float, float, float]
    score: float
    class_id: int = 1


class PersonDetector:
    """Run a pretrained SSDLite detector; dependencies are imported only on use."""

    def __init__(self, device: str = "auto", confidence_threshold: float = 0.45) -> None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1]")
        try:
            import torch
            from torchvision.models.detection import (
                SSDLite320_MobileNet_V3_Large_Weights,
                ssdlite320_mobilenet_v3_large,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Person detection requires CUDA-enabled PyTorch and Torchvision. "
                "Follow docs/ENVIRONMENT.md."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self._torch = torch
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights).to(self.device)
        self.model.eval()
        self.transforms: Any = self.weights.transforms()

    def predict(self, frame_bgr: NDArray[np.uint8]) -> list[Detection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with three channels")
        height, width = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        image = self._torch.from_numpy(np.ascontiguousarray(frame_rgb.transpose(2, 0, 1)))
        image = image.to(dtype=self._torch.float32).div(255.0)
        transformed = self.transforms(image).to(self.device)

        with self._torch.inference_mode():
            output = self.model([transformed])[0]

        # The SSDLite weights transform resizes the image to 320 x 320.
        scale_x, scale_y = width / 320.0, height / 320.0
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()

        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            if int(label) != 1 or float(score) < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            x1 = min(max(x1 * scale_x, 0.0), float(width))
            x2 = min(max(x2 * scale_x, 0.0), float(width))
            y1 = min(max(y1 * scale_y, 0.0), float(height))
            y2 = min(max(y2 * scale_y, 0.0), float(height))
            if x2 > x1 and y2 > y1:
                detections.append(Detection((x1, y1, x2, y2), float(score)))
        return detections
