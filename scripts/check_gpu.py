"""Verify that the installed PyTorch build can execute on an NVIDIA CUDA device."""
import sys


def main() -> int:
    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"Missing deep-learning dependency: {exc}")
        print("Install the CUDA build using docs/ENVIRONMENT.md.")
        return 2

    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"Bundled CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("GPU check FAILED. Install a compatible CUDA-enabled PyTorch build.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / (1024 ** 3):.2f} GiB")

    # A real CUDA operation catches failures that a device-availability check alone may miss.
    tensor = torch.randn((512, 512), device="cuda")
    result = tensor @ tensor
    torch.cuda.synchronize()
    assert result.device.type == "cuda"
    print("CUDA tensor operation: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
