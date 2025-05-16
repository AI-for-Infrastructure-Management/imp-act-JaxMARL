#!/bin/bash

# Function to download and extract a model if it doesn't exist
download_and_extract_model() {
    local model_env=$1
    local model_type=$2
    local target_dir="data/models/${model_env}"
    local model_dir="${target_dir}/${model_type}"
    local zip_file="data/${model_env}-${model_type}.zip"
    local url="https://huggingface.co/datasets/AI-for-Infrastructure-Management/imp-act-benchmark-results/resolve/main/${model_env}-${model_type}.zip?download=true"
    
    # Check if the model directory already exists
    if [ ! -d "$model_dir" ]; then
        echo "Downloading ${model_env}-${model_type}..."
        mkdir -p "$target_dir"
        wget "$url" -O "$zip_file"
        unzip -o "$zip_file" -d "$target_dir/"
        rm "$zip_file"
        echo "${model_env}-${model_type} downloaded and extracted."
    else
        echo "${model_env}-${model_type} already exists, skipping."
    fi
}

# Download the wandb run data if it doesn't exist
if [ ! -d "data/wandb_run_data" ]; then
    echo "Downloading wandb run data..."
    wget "https://huggingface.co/datasets/AI-for-Infrastructure-Management/imp-act-benchmark-results/resolve/main/wandb_run_data.zip?download=true" -O data/wandb_run_data.zip
    unzip -o data/wandb_run_data.zip -d data/
    rm data/wandb_run_data.zip
    echo "wandb run data downloaded and extracted."
else
    echo "wandb run data already exists, skipping."
fi

# Models
# ToyExample-v2
echo "Checking ToyExample-v2 models..."
download_and_extract_model "ToyExample-v2" "vdn_rnn"
download_and_extract_model "ToyExample-v2" "qmix_rnn"
download_and_extract_model "ToyExample-v2" "pqn_rnn"
download_and_extract_model "ToyExample-v2" "mappo_rnn"
download_and_extract_model "ToyExample-v2" "ippo_rnn"

# Cologne-v1
echo "Checking Cologne-v1 models..."
download_and_extract_model "Cologne-v1" "vdn_rnn"
download_and_extract_model "Cologne-v1" "qmix_rnn"
download_and_extract_model "Cologne-v1" "pqn_rnn"
download_and_extract_model "Cologne-v1" "mappo_rnn"
download_and_extract_model "Cologne-v1" "ippo_rnn"

# CologneBonnDusseldorf-v1
echo "Checking CologneBonnDusseldorf-v1 models..."
download_and_extract_model "CologneBonnDusseldorf-v1" "vdn_rnn"
download_and_extract_model "CologneBonnDusseldorf-v1" "qmix_rnn"
download_and_extract_model "CologneBonnDusseldorf-v1" "pqn_rnn"
download_and_extract_model "CologneBonnDusseldorf-v1" "mappo_rnn"
download_and_extract_model "CologneBonnDusseldorf-v1" "ippo_rnn"

echo "All downloads completed!"