# Use a CUDA runtime image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget curl bzip2 vim git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Set PATH so conda is accessible everywhere
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Clone repositories
RUN git clone https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git && \
    cd imp-act-JaxMARL && \
    git checkout imp_act_adaption && \
    git clone https://github.com/AI-for-Infrastructure-Management/imp-act.git && \
    cd imp-act && \
    git checkout 99-updating-the-jax-implementation

# Create Conda environment (impact-jaxmarl-env)
RUN conda env create -f imp-act-JaxMARL/conda_environment.yaml

# Set conda environment path (for clarity)
ENV CONDA_DEFAULT_ENV=impact-jaxmarl-env
ENV PATH=$CONDA_DIR/envs/impact-jaxmarl-env/bin:$PATH

# Install JaxMARL
# Modify JaxMARL dependency for CUDA
RUN sed -i 's/jax==0.4.30/jax[cuda12]==0.4.30/' imp-act-JaxMARL/pyproject.toml
# 2. Install JaxMARL with pip install -e ".[algs]"
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate impact-jaxmarl-env && \
    cd imp-act-JaxMARL && \
    pip install --no-cache-dir -e '.[algs]'"
# Undo the JaxMARL dependency modification
RUN sed -i 's/jax[cuda12]==0.4.30/jax==0.4.30/' imp-act-JaxMARL/pyproject.toml

# Install imp-act
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate impact-jaxmarl-env && \
    cd imp-act-JaxMARL/imp-act && \
    poetry install --with dev,vis"

# Clean up
RUN rm -rf ~/.cache/pip

# Persist environment variables
RUN printenv > /etc/environment