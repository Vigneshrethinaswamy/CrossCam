# Environment setup

## Windows + Python 3.11

Create a project-local virtual environment and install the base app dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Install CUDA-enabled PyTorch

The default PyPI index may install CPU-only PyTorch wheels. For the tested Windows/Python 3.11 setup on an NVIDIA GPU with a compatible driver, install from the official PyTorch CUDA 13.0 wheel index:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

Then verify actual GPU execution:

```powershell
.\.venv\Scripts\python.exe scripts\check_gpu.py
```

The CUDA runtime bundled with PyTorch is separate from the CUDA version displayed by `nvidia-smi`. Do not infer GPU support from the driver alone; `torch.cuda.is_available()` and a real tensor operation must pass.

If you need another PyTorch/CUDA combination, consult the official PyTorch installation selector and choose a build compatible with your Windows/Python version and installed NVIDIA driver. Avoid mixing CPU and CUDA wheels in the same environment.

## Hardware notes

The development target is an RTX 3050 Laptop GPU with 4 GB VRAM and 16 GB system RAM. Start with pretrained compact models, modest input dimensions, sampled frames, and CPU-side association. Measure actual peak VRAM and latency before increasing concurrency.
