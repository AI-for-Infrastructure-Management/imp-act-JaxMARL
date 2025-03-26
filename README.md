# Adaptation of JAXMARL for imp-act

## Installation

### 1. Clone this repository (with imp-act submodules)
```bash
git clone --recurse-submodules -j8 https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git
cd imp-act-JaxMARL
```

### 2. Install dependencies for imp-act and JaxMARL

- Create a new conda environment `impact-jaxmarl-env`
```bash
conda env create -f conda_environment.yaml
conda activate impact-jaxmarl-env
```

- Install dependencies for JAXMARL using pyproject.toml
```bash
pip install -e .[algs]
```
If does not work, try `pip install -e ".[algs]"`

- Install dependencies for imp-act
```bash
cd imp-act && git checkout 99-updating-the-jax-implementation-daniel-dev
poetry install --with dev,vis,jax
```

### 3. Run the example script
```bash
cd ..
python getting-started/example_mpe.py
python getting-started/example_road_env.py
```