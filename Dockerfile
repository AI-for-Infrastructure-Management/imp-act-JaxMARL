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

# Create venv and install Python 3.10 with uv
# venv is needed to avoid conflicts with system cuda/python packages
WORKDIR /workspace/imp-act-JaxMARL
RUN uv python install 3.10 && uv venv --python 3.10 .venv
# make sure Python in this venv is used by default
ENV PATH="/workspace/imp-act-JaxMARL/.venv/bin:$PATH"

# Install Python Packages

# 1. Modify Jax dependency for CUDA 
RUN sed -i 's/jax==\(.*\)/jax[cuda12]==\1/' /workspace/imp-act-JaxMARL/pyproject.toml

# 2. Install JaxMARL
RUN cd /workspace/imp-act-JaxMARL && \
    uv pip install --no-cache-dir -e ".[algs]"
# Undo the Jax dependency modification
RUN sed -i 's/jax\[cuda12\]==\(.*\)/jax==\1/' /workspace/imp-act-JaxMARL/pyproject.toml

# 3. Install imp-act
RUN cd /workspace/imp-act-JaxMARL/imp-act && \
    uv pip install -r requirements/requirements.txt && \
    uv pip install -e .

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