#!/usr/bin/env bash
# Build and install tiny-cuda-nn against the CUDA toolkit that matches PyTorch.
#
# The devcontainer's nvidia-cuda feature pins a toolkit version independently of
# the wheel PyTorch was built with. When those majors disagree, torch's
# cpp_extension refuses to build and tiny-cuda-nn silently never installs --
# which is why `import tinycudann` failed despite postCreateCommand claiming to
# install it. Rather than depend on the feature, this pulls nvcc/nvvm/cccl from
# the same NVIDIA wheel family torch already uses, so the versions cannot drift.
set -euo pipefail

ARCH="${TCNN_CUDA_ARCHITECTURES:-90}"
TCNN_REF="${TCNN_REF:-master}"

TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"
if [[ -z "${TORCH_CUDA}" ]]; then
  echo "PyTorch has no CUDA build; nothing to do." >&2
  exit 1
fi
MAJOR="${TORCH_CUDA%%.*}"
echo "PyTorch CUDA ${TORCH_CUDA} -> targeting cu${MAJOR}, sm_${ARCH}"

# nvcc ships split across three wheels; cccl supplies the <nv/target> headers
# that cuda_fp16.h includes, and nvvm supplies cicc. Missing either fails late.
python -m pip install --quiet \
  "nvidia-cuda-nvcc~=${MAJOR}.0" \
  "nvidia-nvvm~=${MAJOR}.0" \
  "nvidia-cuda-cccl~=${MAJOR}.0"

SITE="$(python -c 'import site; print(site.getusersitepackages())')"
CUDA_ROOT="${SITE}/nvidia/cu${MAJOR}"
if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  CUDA_ROOT="${SITE}/nvidia/cu${MAJOR}"
fi
[[ -x "${CUDA_ROOT}/bin/nvcc" ]] || { echo "nvcc not found under ${CUDA_ROOT}" >&2; exit 1; }

# nvcc expects lib64/, and the linker needs unversioned .so names that the
# runtime wheels do not ship.
ln -sfn "${CUDA_ROOT}/lib" "${CUDA_ROOT}/lib64"
for lib in libcudart libcublas libcublasLt libcurand libnvrtc; do
  target="$(ls "${CUDA_ROOT}/lib/${lib}.so."* 2>/dev/null | head -1 || true)"
  [[ -n "${target}" ]] && ln -sfn "${target}" "${CUDA_ROOT}/lib/${lib}.so"
done

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_ROOT}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${CUDA_ROOT}/include:${CPATH:-}"
export TCNN_CUDA_ARCHITECTURES="${ARCH}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

nvcc --version | tail -2
python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVlabs/tiny-cuda-nn/@${TCNN_REF}#subdirectory=bindings/torch"

python - <<'PY'
import torch, tinycudann as tcnn
enc = tcnn.Encoding(2, {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2,
                        "log2_hashmap_size": 15, "base_resolution": 16,
                        "per_level_scale": 1.5, "interpolation": "Linear"}).cuda()
y = enc(torch.rand(128, 2, device="cuda"))
y.sum().backward()
assert torch.isfinite(y).all()
print(f"tinycudann OK: encoding output {tuple(y.shape)} {y.dtype}")
PY
