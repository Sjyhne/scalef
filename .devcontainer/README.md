# scalef Dev Container

Python 3.11 dev container with **CUDA** and **tiny-cuda-nn** support for GPU-accelerated training and inference.

## What's included

- **Base**: Python 3.11 (Bookworm)
- **CUDA**: NVIDIA CUDA feature with toolkit (nvcc) for building tiny-cuda-nn
- **PyTorch**: Installed via project deps (use a [CUDA index](https://pytorch.org/get-started/locally/) in `postCreateCommand` if you need a specific CUDA build)
- **tiny-cuda-nn**: Installed from NVlabs repo (`bindings/torch`) in `postCreateCommand`
- **OpenCV** system libs and dev tools (black, ruff, pytest)

## Host requirements (for GPU)

- NVIDIA GPU and drivers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install.html) so the container can use the GPU

`runArgs` includes `--gpus all`; if no GPU is present, the container still builds and runs (CPU-only).

## Reopen in container

1. Open this repo in VS Code or Cursor.
2. Command Palette → **Dev Containers: Reopen in Container** (or accept the prompt when opening the folder).

First build can take a while while the CUDA feature and tiny-cuda-nn compile.

## CUDA / nvJitLink

New shells automatically source `scripts/set_cuda_libs.sh` so PyTorch and tinycudann find the right CUDA/nvJitLink libs. If you see `undefined symbol: __nvJitLinkComplete_*`, run:

```bash
source scripts/set_cuda_libs.sh
```
