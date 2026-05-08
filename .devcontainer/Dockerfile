# Python 3.11 devcontainer; CUDA is added via the nvidia-cuda feature in devcontainer.json.
# This image adds system deps needed for OpenCV and leaves nvcc/toolkit to the feature.
FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# OpenCV and other runtime libs
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Source project CUDA/nvJitLink helper in new shells (workspace is mounted at run time)
RUN echo '[ -f /workspaces/scalef/scripts/set_cuda_libs.sh ] && . /workspaces/scalef/scripts/set_cuda_libs.sh' >> /home/vscode/.bashrc
