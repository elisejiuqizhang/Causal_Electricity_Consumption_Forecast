#!/bin/bash
# Environment setup script for Compute Canada (Narval)
# This script creates a conda environment with all required packages
# for the Causal Electricity Consumption Forecasting project

set -e  # Exit on error

echo "=============================================="
echo "Setting up environment on Compute Canada"
echo "=============================================="

# Load required modules
echo "Loading modules..."
module load python/3.10 cuda/11.8 scipy-stack

# Create conda environment directory in scratch (recommended for Compute Canada)
CONDA_ENV_PATH="$HOME/scratch/conda_envs/causal_forecast"
echo "Environment will be created at: $CONDA_ENV_PATH"

# Check if conda is installed, if not, install miniconda
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    cd $HOME
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    
    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
else
    echo "Conda found: $(which conda)"
fi

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# Remove existing environment if it exists
if conda env list | grep -q "causal_forecast"; then
    echo "Removing existing causal_forecast environment..."
    conda env remove -n causal_forecast -y
fi

# Create new environment
echo "Creating conda environment..."
conda create -n causal_forecast python=3.10 -y

# Activate environment
echo "Activating environment..."
conda activate causal_forecast

# Install PyTorch with CUDA support (compatible with cuda/11.8)
echo "Installing PyTorch with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install core scientific packages
echo "Installing core scientific packages..."
conda install numpy pandas scikit-learn matplotlib seaborn -y

# Install additional packages via pip
echo "Installing additional packages via pip..."
pip install --no-cache-dir \
    tensorboard \
    joblib \
    tqdm

# Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python -c "
import sys
print(f'Python version: {sys.version}')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')

import numpy as np
print(f'NumPy version: {np.__version__}')

import pandas as pd
print(f'Pandas version: {pd.__version__}')

import sklearn
print(f'Scikit-learn version: {sklearn.__version__}')

print('\\n✓ All packages installed successfully!')
"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  module load python/3.10 cuda/11.8 scipy-stack"
echo "  source activate causal_forecast"
echo ""
echo "Or in SLURM scripts:"
echo "  module load python/3.10 cuda/11.8 scipy-stack"
echo "  source \$HOME/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate causal_forecast"
echo ""
