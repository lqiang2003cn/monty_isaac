# vision_comp: Vision model server (SAM2, VGGT, Grounding DINO).
# Build context: docker_all/components/vision_comp/
#
# Ubuntu 22.04 ships Python 3.10 natively — no conda.
# BuildKit pip cache mounts ensure torch (4GB) is downloaded only once ever.
#
# Layering strategy (most stable first):
#   1. System packages (rarely changes)
#   2. Torch + torchvision (changes only on version bump)
#   3. Vision model deps — SAM2, VGGT, GDINO (changes rarely)
#   4. App code (changes often — sub-second rebuild)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Aliyun apt mirror (faster in China)
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git ca-certificates cmake build-essential \
        libgl1 libglib2.0-0 \
        libusb-1.0-0 libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Aliyun pip mirror (used as default for non-torch packages)
RUN pip3 config set global.index-url http://mirrors.aliyun.com/pypi/simple/ && \
    pip3 config set global.trusted-host mirrors.aliyun.com

# ---------------------------------------------------------------------------
# Layer 2 — PyTorch (4GB, cached via BuildKit mount)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121/ \
        --extra-index-url http://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com

# ---------------------------------------------------------------------------
# Layer 3a — PyPI packages: OpenCV, matplotlib, RealSense (changes rarely)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
        opencv-python-headless \
        matplotlib \
        pyrealsense2

# ---------------------------------------------------------------------------
# Layer 3b — Orbbec SDK (built from local clone; PyPI package ships macOS-only binary)
# ---------------------------------------------------------------------------
RUN pip3 install pybind11
COPY pyorbbecsdk_src /tmp/pyorbbecsdk
RUN cd /tmp/pyorbbecsdk && mkdir -p build && cd build \
    && cmake -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) .. \
    && make -j$(nproc) && make install \
    && cd /tmp/pyorbbecsdk \
    && mkdir -p install/lib/pyorbbecsdk \
    && pip3 install . \
    && rm -rf /tmp/pyorbbecsdk

# ---------------------------------------------------------------------------
# Layer 3c — Vision model deps: SAM2 + Grounding DINO (local submodules), transformers<5
# Run: git submodule update --init --recursive (sam2_src, groundingdino_src).
# VGGT is optional and installed in 3d when reachable.
# ---------------------------------------------------------------------------
COPY sam2_src /tmp/sam2_build
COPY groundingdino_src /tmp/groundingdino_build
COPY patches /tmp/patches
RUN cd /tmp/groundingdino_build && patch -p1 -i /tmp/patches/groundingdino_ms_deform_attn_cpu.patch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-build-isolation /tmp/sam2_build \
    && pip3 install --no-build-isolation /tmp/groundingdino_build \
    && pip3 install "transformers<5" \
    && rm -rf /tmp/sam2_build /tmp/groundingdino_build

# ---------------------------------------------------------------------------
# Layer 3d — VGGT (optional; only for --depth-source vggt)
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    GIT_SSL_NO_VERIFY=1 pip3 install --no-build-isolation \
        "vggt @ git+https://github.com/facebookresearch/vggt.git" \
    || echo "WARNING: vggt skipped (GitHub unreachable); use --depth-source realsense"

# ---------------------------------------------------------------------------
# Layer 4 — Application code (changes often, fast rebuild)
# ---------------------------------------------------------------------------
RUN pip3 install --upgrade pip setuptools
COPY app /opt/vision_app
RUN pip3 install --no-deps /opt/vision_app

RUN mkdir -p /data /models /vision_shm /var/log/monty

WORKDIR /workspace

CMD ["python3", "-m", "vision_server"]
