#!/usr/bin/env bash
# Set LD_LIBRARY_PATH so PyTorch and tinycudann find the right CUDA / nvJitLink libs.
# Fixes: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
# See: https://github.com/pytorch/pytorch/issues/134929
#
# Usage (run after activating your venv):
#   source scripts/set_cuda_libs.sh
# Or:  . scripts/set_cuda_libs.sh

# Prefer libnvJitLink from the active Python environment (matches PyTorch's bundled cusparse).
NVJITLINK_SO=$(python3 -c "
import sys, os
for sp in getattr(sys, 'path', []):
    if 'site-packages' in sp:
        cand = os.path.join(sp, 'nvidia', 'nvjitlink', 'lib', 'libnvJitLink.so.12')
        if os.path.isfile(cand):
            print(cand)
            break
" 2>/dev/null)

if [[ -n "$NVJITLINK_SO" && -f "$NVJITLINK_SO" ]]; then
  export LD_PRELOAD="$NVJITLINK_SO${LD_PRELOAD:+:$LD_PRELOAD}"
  NVJITLINK_DIR=$(dirname "$NVJITLINK_SO")
  export LD_LIBRARY_PATH="$NVJITLINK_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "set_cuda_libs.sh: using nvJitLink (LD_PRELOAD): $NVJITLINK_SO"
fi

# System CUDA (for nvcc, headers, and libs when building tinycudann).
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
if [[ -d "$CUDA_ROOT" ]]; then
  LIB64="$CUDA_ROOT/lib64"
  if [[ -d "$LIB64" ]]; then
    export LD_LIBRARY_PATH="$LIB64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CUDA_HOME="$CUDA_ROOT"
    echo "set_cuda_libs.sh: CUDA_HOME=$CUDA_HOME, prepended $LIB64 to LD_LIBRARY_PATH"
  fi
else
  echo "set_cuda_libs.sh: CUDA not found at $CUDA_ROOT (optional unless building tinycudann)" >&2
fi

