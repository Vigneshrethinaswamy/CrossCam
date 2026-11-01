from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F

from crosscam.reid import (
    EXPECTED_CHECKPOINT_SHA256,
    FEATURE_DIM,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OSNetConfig,
    OSNetEncoder,
)

EXPECTED_SIZE = 9_336_983
MAX_ABS_LIMIT = 5e-4
MIN_COSINE_LIMIT = 0.99999
MAX_NORM_ERROR = 1e-5


def verify_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.stat().st_size != EXPECTED_SIZE:
        raise ValueError(
            f"Unexpected checkpoint size: {path.stat().st_size}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    actual = digest.hexdigest()

    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint hash mismatch.\n"
            f"Expected: {EXPECTED_CHECKPOINT_SHA256}\n"
            f"Actual:   {actual}"
        )

    return actual


def make_synthetic_input(batch_size: int) -> torch.Tensor:
    # Deterministic synthetic RGB float tensors in [0, 1].
    # These already have the adapter's required 256x128 dimensions.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261101 + batch_size)

    return torch.rand(
        (batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH),
        generator=generator,
        dtype=torch.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate OSNet CPU/CUDA embedding consistency."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            r"C:\CrossCamData\models"
            r"\osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_"
            r"stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
        ),
    )
    args = parser.parse_args()

    print("=" * 65)
    print("CrossCam D3 - OSNet CPU/CUDA parity validation")
    print("=" * 65)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. CPU execution can be tested separately, "
            "but CPU/CUDA parity cannot be measured."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Diagnostic mode: request strict IEEE FP32 computation.
    # These settings affect this Python process only.
    print("\nFP32 precision settings before diagnostic:")

    print("Global:", torch.backends.fp32_precision)
    print("CUDA matmul:", torch.backends.cuda.matmul.fp32_precision)
    print("cuDNN:", torch.backends.cudnn.fp32_precision)

    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"

    print("\nFP32 precision settings applied:")
    print("Global:", torch.backends.fp32_precision)
    print("CUDA matmul:", torch.backends.cuda.matmul.fp32_precision)
    print("cuDNN:", torch.backends.cudnn.fp32_precision)

    checkpoint_hash = verify_checkpoint(args.checkpoint)
    print(f"Checkpoint size: {args.checkpoint.stat().st_size:,} bytes")
    print(f"Checkpoint SHA-256: {checkpoint_hash}")
    print("Checkpoint identity: VERIFIED")

    # Load identical verified weights into separate device instances.
    print("\nLoading CPU encoder...")
    cpu_encoder = OSNetEncoder(
        OSNetConfig(
            checkpoint_path=args.checkpoint,
            expected_sha256=EXPECTED_CHECKPOINT_SHA256,
            device="cpu",
            batch_size=4,
        )
    )

    print("\nLoading CUDA encoder...")
    cuda_encoder = OSNetEncoder(
        OSNetConfig(
            checkpoint_path=args.checkpoint,
            expected_sha256=EXPECTED_CHECKPOINT_SHA256,
            device="cuda",
            batch_size=4,
        )
    )

    print(f"CPU device: {cpu_encoder.metadata['device']}")
    print(f"CUDA device: {cuda_encoder.metadata['device']}")

    all_passed = True
    results = []

    for batch_size in (1, 4):
        print(f"\n--- Batch size: {batch_size} ---")

        images = make_synthetic_input(batch_size)

        with torch.inference_mode():
            cpu_embeddings = cpu_encoder.encode(images)

            cuda_embeddings = cuda_encoder.encode(images)

            torch.cuda.synchronize()

        expected_shape = (batch_size, FEATURE_DIM)

        shape_ok = (
            tuple(cpu_embeddings.shape) == expected_shape
            and tuple(cuda_embeddings.shape) == expected_shape
        )

        finite_ok = (
            bool(torch.isfinite(cpu_embeddings).all())
            and bool(torch.isfinite(cuda_embeddings).all())
        )

        cpu_norms = torch.linalg.vector_norm(
            cpu_embeddings, ord=2, dim=1
        )
        cuda_norms = torch.linalg.vector_norm(
            cuda_embeddings, ord=2, dim=1
        )

        cpu_norm_error = float((cpu_norms - 1.0).abs().max())
        cuda_norm_error = float((cuda_norms - 1.0).abs().max())

        max_norm_error = max(cpu_norm_error, cuda_norm_error)

        max_abs_diff = float(
            (cpu_embeddings - cuda_embeddings).abs().max()
        )

        cosine_values = F.cosine_similarity(
            cpu_embeddings,
            cuda_embeddings,
            dim=1,
            eps=1e-8,
        )

        min_cosine = float(cosine_values.min())

        parity_ok = (
            max_abs_diff <= MAX_ABS_LIMIT
            and min_cosine >= MIN_COSINE_LIMIT
        )

        norms_ok = max_norm_error <= MAX_NORM_ERROR

        batch_passed = (
            shape_ok and finite_ok and norms_ok and parity_ok
        )

        result = {
            "batch_size": batch_size,
            "shape": list(cpu_embeddings.shape),
            "finite": finite_ok,
            "cpu_max_norm_error": cpu_norm_error,
            "cuda_max_norm_error": cuda_norm_error,
            "max_absolute_difference": max_abs_diff,
            "minimum_cosine_similarity": min_cosine,
            "passed": batch_passed,
        }

        results.append(result)
        all_passed = all_passed and batch_passed

        print(json.dumps(result, indent=2))
        print("RESULT:", "PASS" if batch_passed else "FAIL")

    print("\n" + "=" * 65)
    print("FINAL D3 NUMERICAL PARITY RESULT")
    print("=" * 65)

    print(json.dumps({
        "checkpoint_verified": True,
        "cpu_execution": "PASS",
        "cuda_execution": "PASS",
        "batch_results": results,
        "overall_parity": "PASS" if all_passed else "FAIL",
        "acceptance": {
            "max_absolute_difference_lte": MAX_ABS_LIMIT,
            "minimum_cosine_similarity_gte": MIN_COSINE_LIMIT,
            "maximum_norm_error_lte": MAX_NORM_ERROR,
        },
    }, indent=2))

    if not all_passed:
        print("\nD3 parity gate NOT PASSED.")
        return 1

    print("\nD3 numerical parity gate PASSED.")
    print("This is a synthetic-input validation, not a retrieval benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
