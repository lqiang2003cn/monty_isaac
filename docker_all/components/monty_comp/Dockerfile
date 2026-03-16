# syntax=docker/dockerfile:1
# monty_comp: Thousand Brains Project Monty + LQTech extensions.
# Build context: docker_all/  (same as other components).
#
# Multi-stage build:
#   1. conda-env  — creates the tbp.monty conda environment (heavy, cached)
#   2. runtime    — installs tbp.monty + monty_ext into the env
#
# Vision models (SAM2, VGGT, GDINO) run in a separate container (vision_comp).
# monty_ext._bridge connects to vision_comp via TCP.
#
# tbp.monty is a git submodule at third_party/tbp.monty (upstream, read-only).
# monty_ext is our extension package at components/monty_comp/app/monty_ext.

# ---------------------------------------------------------------------------
# Stage 1 — conda environment (heaviest layer, cached across rebuilds)
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS conda-env

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates git bzip2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Accept conda channel ToS (required by recent conda versions)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy upstream environment.yml and strip the editable pip install line
# (we install tbp.monty separately as a non-editable package).
COPY third_party/tbp.monty/environment.yml /tmp/environment_orig.yml
RUN grep -v '\-e \.\[' /tmp/environment_orig.yml > /tmp/environment.yml

# Create the conda environment. This is the slowest step (~10-20 min first
# time) but is cached unless environment.yml changes.
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda env create -f /tmp/environment.yml \
    && conda clean -afy

# ---------------------------------------------------------------------------
# Stage 2 — install packages into the env
# ---------------------------------------------------------------------------
FROM conda-env AS runtime

# Runtime library needed by pyrealsense2 (installed separately to avoid
# invalidating the conda-env cache when adding new apt packages).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libusb-1.0-0 libopengl0 libegl1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Copy tbp.monty source and apply our patches (we don't push to upstream).
# Patches: displacement_matching (empty-edge handling), object_model (k_n guard).
COPY third_party/tbp.monty /opt/tbp.monty
COPY third_party/tbp.monty.diff /opt/tbp.monty.diff
RUN cd /opt/tbp.monty && git apply /opt/tbp.monty.diff && rm /opt/tbp.monty.diff

# Install tbp.monty as a non-editable package. Uses --no-build-isolation so pip
# sees the conda env's packages (numpy, torch, etc.). Upstream updates: rebuild
# after `git submodule update --remote` (patch will reapply).
RUN conda run --no-capture-output -n tbp.monty \
    pip install --no-build-isolation /opt/tbp.monty

# Lightweight deps that work with tbp.monty's Python 3.8 + torch 1.13
RUN conda run --no-capture-output -n tbp.monty \
    pip install --no-build-isolation \
        opencv-python-headless \
        pyrealsense2 \
        pyzmq \
        plotly

# Copy and install our extension package (lightweight, fast rebuild).
COPY components/monty_comp/app /opt/monty_ext_pkg
RUN conda run --no-capture-output -n tbp.monty \
    pip install --no-deps /opt/monty_ext_pkg

# Monty data and results directories (bind-mounted at runtime)
RUN mkdir -p /data /results /var/log/monty /models /vision_shm

# Activate the conda env in every interactive shell session
RUN conda init bash \
    && echo 'conda activate tbp.monty' >> /root/.bashrc

WORKDIR /workspace

# Default: keep container alive for `docker compose exec` usage.
# Run experiments with:
#   docker compose exec monty_comp conda run -n tbp.monty python run.py experiment=...
CMD ["sleep", "infinity"]
