# The harness's one environment (uv.lock) in a Linux x86 image, for GPU clusters that run jobs in
# containers, first the DESI lab's SPARK cluster (Cluster-CI). Built and pushed to
# ghcr.io/colombo-hci-lab/deeptrace-bench by .github/workflows/image.yml.
FROM python:3.12-slim-bookworm

# The NVIDIA container toolkit mounts the host's driver; torch's cu128 wheels carry the CUDA runtime,
# so no CUDA base image is needed and the image works on any driver new enough for CUDA 12.8.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# git for upstream checkouts, gcc for Cluster-CI's NVSHMEM stub, the rest for opencv and onnxruntime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git gcc libc6-dev libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

# The lock goes into the image's own Python, so plain `python3` runs the harness (Cluster-CI checks
# `import torch` on it before every job). --inexact keeps the image's pip, which Cluster-CI uses.
WORKDIR /opt/deeptrace-bench
COPY pyproject.toml uv.lock ./
RUN UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --inexact --no-install-project \
    && rm -rf /root/.cache

# Tools Cluster-CI's runner calls inside the container. On the image's PATH, they don't depend on its
# install into the job's home volume, which failed on 2026-10-16. Pinned, unlike the runner's own.
RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin uv tool install dvc==3.67.1 --with dvc-http \
    && UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin uv tool install \
        "git+https://github.com/UNIL-DESI/dvc-viewer.git@13e3074031543eedca78396bda4077224e4e2ef8" \
    && chmod -R a+rX /opt/uv-tools \
    && rm -rf /root/.cache

WORKDIR /workspace
