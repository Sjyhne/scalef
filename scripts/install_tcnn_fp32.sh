#!/usr/bin/env bash
# Rebuild tiny-cuda-nn in the project venv with fp32 kernels (TCNN_HALF_PRECISION=0).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PIP="${ROOT}/.venv/bin/pip"
VENV_PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${VENV_PIP}" ]]; then
  echo "Missing ${ROOT}/.venv — create it first: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TCNN_HALF_PRECISION=0
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-90}"

"${VENV_PIP}" install -q setuptools wheel
"${VENV_PIP}" uninstall -y tinycudann 2>/dev/null || true
"${VENV_PIP}" install --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

"${VENV_PYTHON}" -c "
import torch, tinycudann as tcnn
net = tcnn.Network(4, 1, {'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None', 'n_neurons': 16, 'n_hidden_layers': 1})
y = net(torch.zeros(1, 4, device='cuda'))
print('tinycudann native dtype:', y.dtype)
assert y.dtype == torch.float32, 'expected fp32 kernels'
print('OK: fp32 tinycudann installed')
"
