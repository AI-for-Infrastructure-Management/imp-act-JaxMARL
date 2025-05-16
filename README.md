# Adaptation of JAXMARL for imp-act

## Installation

### 1. Clone this repository

To clone the this repository and the imp-act repository, run:
```bash
git clone https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git
cd imp-act-JaxMARL && git checkout imp_act_adaption
git clone https://github.com/AI-for-Infrastructure-Management/imp-act.git
```

### 2. Create a virtual environment

Option A: Create a conda environment using the environment YAML file,
```bash
conda env create -f conda_environment.yaml
conda activate impact-jaxmarl-env
```

Option B: Create a virtual environment `impact-jaxmarl-env` using venv/poetry etc.
```bash
# create `impact-jaxmarl-env` with python=3.10
pip install poetry==1.7.1 lockfile==0.12.2
```

### 3.Install dependencies 

**Option A: CPU version**

- Install dependencies for JaxMARL using pyproject.toml
```bash
pip install -e ".[algs]"
```

- Install dependencies for imp-act
```bash
cd imp-act && git checkout main
poetry install --with dev,vis,jax
```
<details>
<summary>Optional: Troubleshooting and Additional Notes</summary>

- If `cmake` is not installed, you can install it using:
    ```bash
    sudo apt-get install cmake  # For Ubuntu/Debian
    brew install cmake          # For macOS
    ```
</details>

**Option B: GPU version**

To install the GPU version jax, we need to install the `jax[cuda12]` version. 
In pyproject.toml, remove "jax==0.4.30" and use "jax[cuda12]==0.4.30" for GPU installation

```bash
pip install -e ".[algs]"

# check JAX GPU (should return somthing like [cuda(id=0)])
python -c "import jax; print(jax.devices())"
```

<details> 
<summary>vast.ai GPU instance</summary>

Create a GPU instance on vast.ai using this [link](https://cloud.vast.ai?ref_id=113803&template_id=b48cde0d602acbd9d886c815750df9b6),

It uses the `nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04` and filters GPUs with CUDA >=12.8. After creating an instance, use the above commands for installation.

```bash
cd imp-act-JaxMARL && conda activate impact-jaxmarl-env
```
</details>

## Run 
To reproduce the results in the paper please refer to the [REPRODUCING.md](REPRODUCING.md) file.

# Licence Disclaimer
This file was modified from the original JaxMARL repository in the process of creating the imp-act adaption of JaxMARL. The original repository can be found at [JaxMARL](https://github.com/FLAIROx/JaxMARL). 

It is used under the Apache License 2.0. The original license can be found at [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) in addition to the licence file in this repository. [JaxMARL License](LICENSE).