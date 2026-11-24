# This file was adapted from the original in the process of creating the imp-act adaption of JaxMARL under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

# Use a CUDA runtime image
FROM nvidia/cuda:12.6.2-base-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget curl ca-certificates git build-essential bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Install uv (standalone binary) 
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Clone repositories
RUN git clone https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git && \
    cd imp-act-JaxMARL && \
    git checkout main && \
    git clone https://github.com/AI-for-Infrastructure-Management/imp-act.git && \
    cd imp-act && \
    git checkout main

# Create the venv .venv with Python 3.10 and install the pinned dependencies of the GPU version
# (uv.lock, versions of the final runs); the venv is needed to avoid conflicts with system cuda/python packages
WORKDIR /workspace/imp-act-JaxMARL
RUN uv sync --frozen --no-cache --all-extras --group cuda
# make sure Python in this venv is used by default
ENV PATH="/workspace/imp-act-JaxMARL/.venv/bin:$PATH"

# Clean up
RUN rm -rf \
    ~/.cache \
    /root/.cache \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/* \
    /usr/share/doc/* \
    /usr/share/man/* \
    /usr/share/info/* \
    /var/cache/apt/* && \
    apt-get autoremove -y && apt-get clean

# Set final working directory
WORKDIR /workspace/imp-act-JaxMARL