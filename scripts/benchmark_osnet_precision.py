from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import Tensor

from crosscam.reid import (
    EXPECTED_CHECKPOINT_SHA256,
    FEATURE_DIM,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OSNetConfig,
    OSNetEncoder,
)

EXPECTED_CHECKPOINT_SIZE = 9_336_983
BATCH_SIZES = (1, 4, 8, 16)
PRECISION_MODES = ("default", "ieee")


def verify_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.stat().st_size != EXPECTED_CHECKPOINT_SIZE:
        raise ValueError(
            f"Unexpected checkpoint size: {path.stat().st_size:,} bytes"
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    actual = digest.hexdigest()
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Checkpoint SHA-256 mismatch. "
            f"Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual}"
        )
    return actual


def set_precision(mode: str) -> None:
    """Set this process's PyTorch FP32 precision policy explicitly."""
    if mode not in PRECISION_MODES:
        raise ValueError(f"Unsupported precision mode: {mode}")

    value = "none" if mode == "default" else "ieee"
    torch.backends.fp32_precision = value
    torch.backends.cuda.matmul.fp32_precision = value
    torch.backends.cudnn.fp32_precision = value
    torch.backends.cudnn.conv.fp32_precision = value


def make_input(batch_size: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261101 + batch_size)
    return torch.rand(
        (batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH),
        generator=generator,
        dtype=torch.float32,
    )


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def validate_output(output: Tensor, batch_size: int) -> None:
    expected_shape = (batch_size, FEATURE_DIM)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape {tuple(output.shape)}; "
            f"expected {expected_shape}"
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Encoder produced NaN or Inf values")

    norms = torch.linalg.vector_norm(output, ord=2, dim=1)
    max_norm_error = float((norms - 1.0).abs().max())
    if max_norm_error > 1e-5:
        raise RuntimeError(
            f"Output embeddings are not unit-normalized: {max_norm_error}"
        )


def benchmark_case(
    encoder: OSNetEncoder,
    device: str,
    precision: str,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    set_precision(precision)
    images = make_input(batch_size)

    # Release unused cached CUDA blocks between cases so one batch size does
    # not inflate the next case's reserved-memory reading.
    if device == "cuda":
        synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warm up kernels and allocator before collecting measurements.
    with torch.inference_mode():
        for _ in range(warmup):
            encoder.encode(images)
        synchronize(device)

    if device == "cuda":
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_allocated = None
        baseline_reserved = None

    samples_ms: list[float] = []
    last_output: Tensor | None = None

    with torch.inference_mode():
        for _ in range(iterations):
            synchronize(device)
            start = time.perf_counter()
            last_output = encoder.encode(images)
            synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    if last_output is None:
        raise RuntimeError("No benchmark iterations were executed")
    validate_output(last_output, batch_size)

    median_ms = statistics.median(samples_ms)
    mean_ms = statistics.mean(samples_ms)
    p95_ms = percentile_95(samples_ms)
    throughput = batch_size / (median_ms / 1000.0)

    memory = {
        "warmup_allocated_vram_mib": None,
        "warmup_reserved_vram_mib": None,
        "measured_peak_allocated_vram_mib": None,
        "measured_peak_reserved_vram_mib": None,
        "measured_peak_allocated_increase_mib": None,
        "measured_peak_reserved_increase_mib": None,
    }

    if device == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        mib = 1024**2
        memory = {
            "warmup_allocated_vram_mib": baseline_allocated / mib,
            "warmup_reserved_vram_mib": baseline_reserved / mib,
            "measured_peak_allocated_vram_mib": peak_allocated / mib,
            "measured_peak_reserved_vram_mib": peak_reserved / mib,
            "measured_peak_allocated_increase_mib": max(
                0, peak_allocated - baseline_allocated
            ) / mib,
            "measured_peak_reserved_increase_mib": max(
                0, peak_reserved - baseline_reserved
            ) / mib,
        }

    return {
        "device": device,
        "precision": precision,
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_ms_median": median_ms,
        "latency_ms_mean": mean_ms,
        "latency_ms_p95": p95_ms,
        "throughput_images_per_second": throughput,
        **memory,
        "output_valid": True,
        "status": "PASS",
    }

def run_benchmark(
    encoder: OSNetEncoder,
    device: str,
    precision: str,
    batch_sizes: tuple[int, ...],
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for batch_size in batch_sizes:
        print(
            f"Running device={device}, precision={precision}, "
            f"batch={batch_size} ...",
            flush=True,
        )
        try:
            result = benchmark_case(
                encoder=encoder,
                device=device,
                precision=precision,
                batch_size=batch_size,
                warmup=warmup,
                iterations=iterations,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if device != "cuda":
                raise
            torch.cuda.empty_cache()
            result = {
                "device": device,
                "precision": precision,
                "batch_size": batch_size,
                "status": "CUDA_OUT_OF_MEMORY",
                "error": str(exc).splitlines()[0],
            }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OSNet CPU/CUDA inference using synthetic inputs only."
        )
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Measured iterations per test case (minimum 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up iterations per test case.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=list(BATCH_SIZES),
        help="Batch sizes to benchmark (default: 1 4 8 16).",
    )
    args = parser.parse_args()

    if args.iterations < 5:
        parser.error("--iterations must be at least 5")
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if not args.batches or any(size < 1 for size in args.batches):
        parser.error("--batches must contain positive integers")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this benchmark requires both CPU and CUDA")

    checkpoint_hash = verify_checkpoint(args.checkpoint)

    print("=" * 72)
    print("CrossCam OSNet precision and performance benchmark")
    print("=" * 72)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"Checkpoint bytes: {args.checkpoint.stat().st_size:,}")
    print(f"Checkpoint SHA-256: {checkpoint_hash}")
    print(f"Warm-up iterations: {args.warmup}")
    print(f"Measured iterations: {args.iterations}")
    print(f"Batch sizes: {args.batches}")
    print("Input: deterministic synthetic RGB float32, shape (N,3,256,128)")
    print("Scope: adapter encode() latency including device transfer and CPU output")

    maximum_batch = max(args.batches)
    print("\nLoading verified CPU encoder...", flush=True)
    cpu_encoder = OSNetEncoder(
        OSNetConfig(
            checkpoint_path=args.checkpoint,
            expected_sha256=EXPECTED_CHECKPOINT_SHA256,
            device="cpu",
            batch_size=maximum_batch,
        )
    )

    print("Loading verified CUDA encoder...", flush=True)
    cuda_encoder = OSNetEncoder(
        OSNetConfig(
            checkpoint_path=args.checkpoint,
            expected_sha256=EXPECTED_CHECKPOINT_SHA256,
            device="cuda",
            batch_size=maximum_batch,
        )
    )

    print(f"CPU device: {cpu_encoder.metadata['device']}")
    print(f"CUDA device: {cuda_encoder.metadata['device']}")

    all_results: list[dict[str, object]] = []

    # Pair default and IEEE runs for each batch size to reduce run-order bias.
    # CPU measurements are included under both labels for completeness;
    # CUDA FP32 precision controls are primarily relevant to GPU kernels.
    for device, encoder in (
        ("cpu", cpu_encoder),
        ("cuda", cuda_encoder),
    ):
        for batch_size in args.batches:
            for precision in PRECISION_MODES:
                all_results.extend(
                    run_benchmark(
                        encoder=encoder,
                        device=device,
                        precision=precision,
                        batch_sizes=(batch_size,),
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )

    print("\n" + "=" * 72)
    print("FINAL BENCHMARK RESULTS (JSON)")
    print("=" * 72)
    print(json.dumps({
        "checkpoint_sha256": checkpoint_hash,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cpu_threads": torch.get_num_threads(),
        "input": "deterministic synthetic RGB float32 (N,3,256,128)",
        "scope": "OSNetEncoder.encode() including transfers and CPU output",
        "results": all_results,
    }, indent=2))

    oom_count = sum(
        result.get("status") == "CUDA_OUT_OF_MEMORY"
        for result in all_results
    )
    failed_count = sum(
        result.get("status") != "PASS"
        and result.get("status") != "CUDA_OUT_OF_MEMORY"
        for result in all_results
    )

    print(
        f"\nCompleted: {len(all_results)} test cases; "
        f"CUDA OOM cases: {oom_count}; other failures: {failed_count}"
    )

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
