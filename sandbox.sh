#!/bin/bash

# usage: bash sandbox.sh

home_path="/home/$USER/prateek/imp-act-JaxMARL"
source "${home_path}/.venv/bin/activate"
PYTHON="${home_path}/.venv/bin/python"

# wandb sync /scratch/pbhustali/imp-act-jaxmarl/wandb/offline-run-20251120_132552-bkg7iouc --project impact-jaxMARL-ToyExample-v2-validation

wandb sync /scratch/pbhustali/imp-act-jaxmarl/wandb/offline-run-20251120_142005-o6tgvxep --project impact-jaxMARL-ToyExample-v2-validation