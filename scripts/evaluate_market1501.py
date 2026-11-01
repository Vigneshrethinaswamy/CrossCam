"""Run the frozen OSNet adapter on an authorized local Market-1501 copy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscam.reid import EXPECTED_CHECKPOINT_SHA256, OSNetConfig, OSNetEncoder
from crosscam.retrieval import evaluate_market1501, market1501_identity_camera


def _images(folder: Path):
    for path in sorted(folder.glob("*.jpg")):
        pid, camera = market1501_identity_camera(path.name)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        yield tensor, pid, camera


def _encode(folder: Path, encoder: OSNetEncoder, batch_size: int):
    features, pids, cameras, pending = [], [], [], []
    for tensor, pid, camera in _images(folder):
        pending.append(tensor)
        pids.append(pid)
        cameras.append(camera)
        if len(pending) == batch_size:
            features.append(encoder.encode(torch.stack(pending)).numpy())
            pending.clear()
    if pending:
        features.append(encoder.encode(torch.stack(pending)).numpy())
    if not features:
        raise ValueError(f"No JPG images found in {folder}")
    return np.concatenate(features), pids, cameras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    base = args.dataset_root
    if (base / "Market-1501-v15.09.15").is_dir():
        base = base / "Market-1501-v15.09.15"
    readme = base / "readme.txt"
    if not readme.is_file():
        parser.error("Dataset readme.txt is required; verify source and terms first.")
    terms = readme.read_text(encoding="utf-8", errors="replace").lower()
    if not all(term in terms for term in ("research only", "distribute", "commercial")):
        parser.error("Dataset usage notice was not recognized; review terms first.")
    query_dir, gallery_dir = base / "query", base / "bounding_box_test"
    if not query_dir.is_dir() or not gallery_dir.is_dir():
        parser.error("Expected query/ and bounding_box_test/ directories.")

    encoder = OSNetEncoder(
        OSNetConfig(
            checkpoint_path=args.checkpoint,
            expected_sha256=args.sha256,
            device=args.device,
            batch_size=args.batch_size,
        )
    )
    query = _encode(query_dir, encoder, args.batch_size)
    gallery = _encode(gallery_dir, encoder, args.batch_size)
    metrics = evaluate_market1501(
        query[0], query[1], query[2], gallery[0], gallery[1], gallery[2]
    )
    print(json.dumps({"metadata": encoder.metadata, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
