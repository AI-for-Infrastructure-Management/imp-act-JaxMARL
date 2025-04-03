# Adaptation of JAXMARL for imp-act

## Installation

##### 1. Clone this repository

The imp-act repository is included as a submodule. To clone this repository and its submodules, run:
```bash
git clone --recurse-submodules -j8 https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git
cd imp-act-JaxMARL && git submodule update --init --recursive && git checkout imp_act_adaption
```

##### 2. Create a virtual environment

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

##### 3. Install dependencies

- Install dependencies for JAXMARL using pyproject.toml
```bash
pip install -e .[algs]
```
If that does not work, try `pip install -e ".[algs]"`

- Install dependencies for imp-act
```bash
cd imp-act && git checkout 99-updating-the-jax-implementation-daniel-dev
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

##### 4. Run the example script
```bash
cd ..
python getting-started/example_mpe.py
python getting-started/example_road_env.py
```