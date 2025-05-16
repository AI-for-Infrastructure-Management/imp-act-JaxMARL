# Adaptation of JAXMARL for imp-act

This repository provides the adaptation of JaxMARL for the IMP-act environments. It corresponds to the `imp-act-JaxMARL` repository referenced in the main "IMP-act: Benchmarking MARL for Infrastructure Management Planning at Scale with JAX" paper.

* **Code Repositories:**
    * This Repository (JaxMARL Adaptation): [AI-for-Infrastructure-Management/imp-act-JaxMARL](https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL)
    * Main IMP-act Environment (Dependency): [AI-for-Infrastructure-Management/imp-act](https://github.com/AI-for-Infrastructure-Management/imp-act)
* **Reproducibility:** To reproduce the results presented in the main IMP-act paper using this adapted JaxMARL codebase, please refer to the [REPRODUCING.md](REPRODUCING.md) file.

## Requirements

1.  **Prerequisites:**
    * Python: 3.10 (as specified for the Conda environment or when creating with venv/poetry).
    * Poetry: 1.7.1 (if using Poetry for environment management or manual installation step).
    * Conda (if using Option A for environment setup).
    * Git.
    * Ensure JAX is installed correctly for your specific hardware (CPU/GPU). Detailed JAX installation notes are included in the installation steps.

2.  **Installation:**

    * **1. Clone Repositories:**
        ```bash
        git clone [https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git](https://github.com/AI-for-Infrastructure-Management/imp-act-JaxMARL.git)
        cd imp-act-JaxMARL && git checkout imp_act_adaption
        git clone [https://github.com/AI-for-Infrastructure-Management/imp-act.git](https://github.com/AI-for-Infrastructure-Management/imp-act.git)
        ```

    * **2. Create a virtual environment:**
        Choose one of the following options:

        * **Option A: Using Conda**
            ```bash
            conda env create -f conda_environment.yaml
            conda activate impact-jaxmarl-env
            ```

        * **Option B: Using venv/Poetry etc.**
            ```bash
            # Create a virtual environment 'impact-jaxmarl-env' with python=3.10
            # For example, using venv:
            # python3.10 -m venv impact-jaxmarl-env
            # source impact-jaxmarl-env/bin/activate  # On Linux/macOS
            # .\impact-jaxmarl-env\Scripts\activate # On Windows

            # Install poetry and lockfile if not already in your global/base environment
            # (or install them after activating the new empty virtual environment)
            pip install poetry==1.7.1 lockfile==0.12.2
            ```

    * **3. Install dependencies:**
        Ensure your chosen virtual environment (`impact-jaxmarl-env`) is activated. Navigate to the `imp-act-JaxMARL` directory if you're not already there.

        * **Option A: CPU version**
            ```bash
            # Install dependencies for JaxMARL (from imp-act-JaxMARL directory)
            pip install -e ".[algs]"

            # Install dependencies for imp-act
            cd imp-act && git checkout main # Ensure you are in the imp-act directory
            poetry install --with dev,vis,jax
            cd .. # Return to the imp-act-JaxMARL directory
            ```
            <details>
            <summary>Optional: Troubleshooting and Additional Notes</summary>

            - If `cmake` is not installed (it might be a dependency for some packages), you can install it using:
                ```bash
                sudo apt-get install cmake  # For Ubuntu/Debian
                brew install cmake          # For macOS
                # Or download from cmake.org for other systems
                ```
            </details>

        * **Option B: GPU version**
            To install the GPU version of JAX, you typically need to specify the CUDA version. For example, for JAX 0.4.23 (check your `pyproject.toml` for the exact version used by `.[algs]` or update as needed):
            In `pyproject.toml` (within the `imp-act-JaxMARL` directory), you might need to adjust the `jax` and `jaxlib` dependencies. For example, change:
            `"jax==0.4.23"` to `"jax[cuda12_pip]==0.4.23"` (or the specific CUDA version you have, e.g., `cuda11_pip`).
            Refer to the [official JAX installation guide](https://github.com/google/jax#installation) for the correct pip wheels for your CUDA and JAX version.

            ```bash
            # After modifying pyproject.toml if necessary:
            pip install -e ".[algs]"

            # Check JAX GPU (should return something like [cuda(id=0)] or similar GPU device)
            python -c "import jax; print(jax.devices())"

            # Install dependencies for imp-act (GPU version of JAX will be handled by its own poetry config)
            cd imp-act && git checkout main # Ensure you are in the imp-act directory
            poetry install --with dev,vis,jax # This will install JAX according to imp-act's pyproject.toml
            cd .. # Return to the imp-act-JaxMARL directory
            ```
            <details>
            <summary>vast.ai GPU instance</summary>

            If using a vast.ai GPU instance, you can create one using this [link](https://cloud.vast.ai?ref_id=113803&template_id=b48cde0d602acbd9d886c815750df9b6).
            This template might use an image like `nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04` (example, actual version might differ). After creating and connecting to an instance, activate your Conda environment (if used) and follow the GPU installation commands above.
            ```bash
            # Example if you set up conda and cloned inside the instance
            # cd imp-act-JaxMARL 
            # conda activate impact-jaxmarl-env 
            # Then run the GPU installation steps
            ```
            </details>

## Results
![Figure 3](docs/imgs/imp_act_results.png)
Normalized best policy returns, for all tested IMP-act environments and MARL algorithms over 10 training seeds. Returns are normalized with respect to the baseline heuristic policy $\text{H}_\text{PS}$.

Best performance per algorithm in terms of expected return, 95% CI, and required VRAM for each environment. The best performance per environment is highlighted in bold, and performances within their 95% CI are marked with *.

### **Toy-Example** ($\text{H}_\text{PS}=-274\text{M}$)

| Algorithm           | $\Delta \bar{R}^{\pi}_0$ | 95% CI             | VRAM (GB) |
| :------------------ | -----------------------: | :----------------: | -----------: |
| VDN                 | *+22.09%                 | [21.35, 22.81]   | 0.52         |
| QMIX                | *+21.37%                 | [20.54, 22.15]   | 1.55         |
| PQN-VDN             | **+22.72%**                 | [21.98, 23.46]   | 0.16         |
| MAPPO               | +19.22%                  | [18.38, 20.04]   | 0.85         |
| IPPO                | +20.54%                  | [19.76, 21.30]   | 1.87         |
| $\text{VDN}_{\text{BA}}$ | +23.04%                  | [20.69, 25.31]   | --           |

### **Cologne** ($\text{H}_\text{PS}=-8.2\text{B}$)

| Algorithm           | $\Delta \bar{R}^{\pi}_0$ | 95% CI             | VRAM (GB) |
| :------------------ | -----------------------: | :----------------: | -----------: |
| VDN                 | **+22.88%** | [22.59, 23.16]   | 5.94         |
| QMIX                | +21.48%                  | [21.17, 21.80]   | 7.72         |
| PQN-VDN             | +20.01%                  | [19.70, 20.30]   | 0.77         |
| MAPPO               | +17.85%                  | [17.51, 18.17]   | 13.13        |
| IPPO                | +12.62%                  | [12.30, 12.93]   | 1.29         |
| $\text{VDN}_{\text{BA}}$ | +24.57%                  | [23.67, 25.42]   | --           |

### **CologneBonn-Dusseldorf** ($\text{H}_\text{PS}=-33.1\text{B}$)

| Algorithm           | $\Delta \bar{R}^{\pi}_0$ | 95% CI             | VRAM (GB) |
| :------------------ | -----------------------: | :----------------: | -----------: |
| VDN                 | **+24.91%** | [24.71, 25.10]   | 12.09        |
| QMIX                | +20.19%                  | [19.97, 20.40]   | 10.37        |
| PQN-VDN             | +21.24%                  | [21.04, 21.45]   | 2.29         |
| MAPPO               | +3.89%                   | [3.67, 4.10]     | 16.45        |
| IPPO                | -15.03%                  | [-15.31, -14.75] | 2.14         |
| $\text{VDN}_{\text{BA}}$ | +25.70%                  | [25.09, 26.29]   | --           |



## Training & Evaluation
To reproduce the results presented in the main IMP-act paper using this adapted JaxMARL codebase, please refer to the [REPRODUCING.md](REPRODUCING.md) file in this repository.

## License
This file was modified from the original JaxMARL repository in the process of creating the imp-act adaptation of JaxMARL.
* The original JaxMARL repository can be found at [JaxMARL](https://github.com/FLAIROx/JaxMARL).
* This adaptation, and the original JaxMARL, are used under the Apache License 2.0. The original license can be found at [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* A copy of the Apache 2.0 license is also included in the [LICENSE](LICENSE) file in this repository, reflecting the license of the original JaxMARL components.
