from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

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
ROUND_PRECISION_ORDERS = {
    1: ("default", "ieee"),
    2: ("ieee", "default"),
    3: ("default", "ieee"),
}
ROUND_BATCH_ORDERS = {
    1: (1, 4, 8, 16),
    2: (16, 8, 4, 1),
    3: (1, 4, 8, 16),
}


def verify_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    size = path.stat().st_size
    if size != EXPECTED_CHECKPOINT_SIZE:
        raise ValueError(
            f"Unexpected checkpoint size: {size:,} bytes"
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


def set_precision(mode: str) -> dict[str, str]:
    """Set and return the effective FP32 settings for one benchmark case."""
    if mode not in ("default", "ieee"):
        raise ValueError(f"Unsupported precision mode: {mode}")

    value = "none" if mode == "default" else "ieee"
    torch.backends.fp32_precision = value
    torch.backends.cuda.matmul.fp32_precision = value
    torch.backends.cudnn.fp32_precision = value
    torch.backends.cudnn.conv.fp32_precision = value

    return {
        "global": str(torch.backends.fp32_precision),
        "cuda_matmul": str(torch.backends.cuda.matmul.fp32_precision),
        "cudnn": str(torch.backends.cudnn.fp32_precision),
        "cudnn_conv": str(torch.backends.cudnn.conv.fp32_precision),
    }


def make_inputs() -> dict[int, Tensor]:
    """Create each fixed synthetic tensor once and reuse it for every round."""
    inputs: dict[int, Tensor] = {}
    for batch_size in BATCH_SIZES:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20261101 + batch_size)
        inputs[batch_size] = torch.rand(
            (batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH),
            generator=generator,
            dtype=torch.float32,
        )
    return inputs


def synchronize() -> None:
    torch.cuda.synchronize()


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def validate_embeddings(output: Tensor, batch_size: int) -> dict[str, Any]:
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
            f"Embeddings are not unit-normalized: {max_norm_error}"
        )
    return {
        "shape": list(output.shape),
        "finite": True,
        "max_unit_norm_error": max_norm_error,
    }


def benchmark_case(
    encoder: OSNetEncoder,
    images: Tensor,
    batch_size: int,
    round_number: int,
    precision: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    effective_settings = set_precision(precision)

    # Clear only unused allocator blocks before each case. The model remains
    # resident on the GPU. Warm-up is outside the timed measurement.
    synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(warmup):
            encoder.encode(images)
        synchronize()

    warmup_peak_allocated = torch.cuda.max_memory_allocated()
    warmup_peak_reserved = torch.cuda.max_memory_reserved()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    # Reset peak counters after warm-up so measured peaks are distinguishable.
    torch.cuda.reset_peak_memory_stats()
    samples_ms: list[float] = []
    output: Tensor | None = None

    with torch.inference_mode():
        for _ in range(iterations):
            synchronize()
            started = time.perf_counter()
            output = encoder.encode(images)
            synchronize()
            samples_ms.append((time.perf_counter() - started) * 1000.0)

    if output is None:
        raise RuntimeError("No benchmark iterations were executed")

    output_check = validate_embeddings(output, batch_size)
    median_ms = statistics.median(samples_ms)
    mean_ms = statistics.mean(samples_ms)
    p95_ms = percentile_95(samples_ms)
    mib = 1024**2

    measured_peak_allocated = torch.cuda.max_memory_allocated()
    measured_peak_reserved = torch.cuda.max_memory_reserved()

    return {
        "round": round_number,
        "precision": precision,
        "batch_size": batch_size,
        "effective_precision_settings": effective_settings,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_ms_median": median_ms,
        "latency_ms_mean": mean_ms,
        "latency_ms_p95": p95_ms,
        "throughput_images_per_second": batch_size / (median_ms / 1000.0),
        **output_check,
        "warmup_peak_allocated_vram_mib": warmup_peak_allocated / mib,
        "warmup_peak_reserved_vram_mib": warmup_peak_reserved / mib,
        "baseline_allocated_vram_mib": baseline_allocated / mib,
        "baseline_reserved_vram_mib": baseline_reserved / mib,
        "measured_peak_allocated_vram_mib": measured_peak_allocated / mib,
        "measured_peak_reserved_vram_mib": measured_peak_reserved / mib,
        "measured_peak_allocated_increase_mib": max(
            0, measured_peak_allocated - baseline_allocated
        ) / mib,
        "measured_peak_reserved_increase_mib": max(
            0, measured_peak_reserved - baseline_reserved
        ) / mib,
        "status": "PASS",
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (int(item["round"]), int(item["batch_size"]), str(item["precision"])): item
        for item in results
    }
    paired: list[dict[str, Any]] = []

    for round_number in (1, 2, 3):
        for batch_size in BATCH_SIZES:
            default = by_key[(round_number, batch_size, "default")]
            ieee = by_key[(round_number, batch_size, "ieee")]
            default_ms = float(default["latency_ms_median"])
            ieee_ms = float(ieee["latency_ms_median"])
            delta_pct = (ieee_ms - default_ms) / default_ms * 100.0
            paired.append({
                "round": round_number,
                "batch_size": batch_size,
                "default_median_ms": default_ms,
                "ieee_median_ms": ieee_ms,
                "ieee_vs_default_percent": delta_pct,
                "interpretation": (
                    "ieee_faster" if delta_pct < 0
                    else "default_faster" if delta_pct > 0
                    else "equal"
                ),
            })

    aggregate = []
    for batch_size in BATCH_SIZES:
        default_values = [
            float(by_key[(round_number, batch_size, "default")]["latency_ms_median"])
            for round_number in (1, 2, 3)
        ]
        ieee_values = [
            float(by_key[(round_number, batch_size, "ieee")]["latency_ms_median"])
            for round_number in (1, 2, 3)
        ]
        paired_deltas = [
            float(item["ieee_vs_default_percent"])
            for item in paired
            if item["batch_size"] == batch_size
        ]
        aggregate.append({
            "batch_size": batch_size,
            "default_median_of_round_medians_ms": statistics.median(default_values),
            "default_round_min_ms": min(default_values),
            "default_round_max_ms": max(default_values),
            "ieee_median_of_round_medians_ms": statistics.median(ieee_values),
            "ieee_round_min_ms": min(ieee_values),
            "ieee_round_max_ms": max(ieee_values),
            "paired_ieee_vs_default_percent_median": statistics.median(paired_deltas),
            "paired_percent_min": min(paired_deltas),
            "paired_percent_max": max(paired_deltas),
            "round_to_round_paired_spread_percentage_points": (
                max(paired_deltas) - min(paired_deltas)
            ),
        })

    return {
        "paired_round_comparisons": paired,
        "per_batch_summary": aggregate,
        "interpretation_note": (
            "Negative IEEE-vs-default percentage means IEEE was faster. "
            "Three rounds are descriptive repeatability evidence, not a "
            "statistical significance test."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Three-round CUDA-only OSNet precision repeatability benchmark "
            "using synthetic inputs."
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/osnet_cuda_repeatability.json"),
    )
    args = parser.parse_args()

    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.iterations < 5:
        parser.error("--iterations must be at least 5")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this CUDA-only benchmark cannot run")

    checkpoint_hash = verify_checkpoint(args.checkpoint)
    inputs = make_inputs()

    print("=" * 76)
    print("CrossCam OSNet — three-round CUDA precision repeatability benchmark")
    print("=" * 76)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"Checkpoint bytes: {args.checkpoint.stat().st_size:,}")
    print(f"Checkpoint SHA-256: {checkpoint_hash}")
    print(f"Warm-up iterations per case: {args.warmup}")
    print(f"Measured iterations per case: {args.iterations}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print("Synthetic inputs: fixed per batch size and reused across all rounds")
    print("Round 1: default -> IEEE")
    print("Round 2: IEEE -> default")
    print("Round 3: default -> IEEE")
    print("No videos, person crops, or retrieval datasets are read.")

    print("\nLoading one verified CUDA encoder instance...", flush=True)
    encoder = OSNetEncoder(
        OSNetConfig(
            checkpoint_path=args.checkpoint,
            expected_sha256=EXPECTED_CHECKPOINT_SHA256,
            device="cuda",
            batch_size=max(BATCH_SIZES),
        )
    )
    print(f"Encoder device: {encoder.metadata['device']}")
    print(f"Encoder metadata: {json.dumps(encoder.metadata, default=str)}")

    all_results: list[dict[str, Any]] = []

    for round_number in (1, 2, 3):
        precision_order = ROUND_PRECISION_ORDERS[round_number]
        batch_order = ROUND_BATCH_ORDERS[round_number]
        print(
            f"\n=== ROUND {round_number}: "
            f"{' -> '.join(precision_order)}; batches={batch_order} ===",
            flush=True,
        )

        for batch_size in batch_order:
            images = inputs[batch_size]
            for precision in precision_order:
                print(
                    f"Round {round_number}: precision={precision}, "
                    f"batch={batch_size}",
                    flush=True,
                )
                try:
                    result = benchmark_case(
                        encoder=encoder,
                        images=images,
                        batch_size=batch_size,
                        round_number=round_number,
                        precision=precision,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    result = {
                        "round": round_number,
                        "precision": precision,
                        "batch_size": batch_size,
                        "status": "CUDA_OUT_OF_MEMORY",
                        "error": str(exc).splitlines()[0],
                    }
                all_results.append(result)
                print(json.dumps(result, indent=2), flush=True)

    summary = summarize(
        [item for item in all_results if item.get("status") == "PASS"]
    ) if len([x for x in all_results if x.get("status") == "PASS"]) == 24 else None

    report = {
        "checkpoint_sha256": checkpoint_hash,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "synthetic_input": "fixed RGB float32 tensors, shape (N,3,256,128)",
        "batch_sizes": list(BATCH_SIZES),
        "round_precision_orders": {
            str(k): list(v) for k, v in ROUND_PRECISION_ORDERS.items()
        },
        "round_batch_orders": {
            str(k): list(v) for k, v in ROUND_BATCH_ORDERS.items()
        },
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "results": all_results,
        "summary": summary,
        "completed_cases": len(all_results),
        "passed_cases": sum(x.get("status") == "PASS" for x in all_results),
        "cuda_oom_cases": sum(
            x.get("status") == "CUDA_OUT_OF_MEMORY" for x in all_results
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print("REPEATABILITY SUMMARY")
    print("=" * 76)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved JSON report: {args.output}")
    print(
        f"Cases passed: {report['passed_cases']}/{report['completed_cases']}; "
        f"CUDA OOM: {report['cuda_oom_cases']}"
    )

    return 0 if report["passed_cases"] == 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())
