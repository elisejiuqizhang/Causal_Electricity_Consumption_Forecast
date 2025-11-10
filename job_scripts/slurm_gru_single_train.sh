#!/bin/bash
#SBATCH --job-name=gru_single
#SBATCH --output=logs/gru_single_%A_%a.out
#SBATCH --error=logs/gru_single_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-35

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scratch

cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast/exps/forecast/per_region

SCRIPT="gru_single_train.py"

# Define parameter combinations
list_regions=("Toronto" "Peel" "Hamilton" "Brantford" "Waterloo" "London" "Oshawa" "Kingston" "Ottawa")
list_features=("F0" "F1" "F2" "F3")

# Calculate indices for this array task
n_features=${#list_features[@]}
n_regions=${#list_regions[@]}

region_idx=$((SLURM_ARRAY_TASK_ID / n_features))
feature_idx=$((SLURM_ARRAY_TASK_ID % n_features))

region=${list_regions[$region_idx]}
feature_set=${list_features[$feature_idx]}

echo "Running: Region $region, Feature set $feature_set"

python "$SCRIPT" \
    --region "$region" \
    --feature_set "$feature_set" \
    --scaler standard \
    --n_folds 3 \
    --window_size 26298 \
    --epochs 500 \
    --batch_size 64 \
    --lr 0.0001 \
    --train_ratio 0.93 \
    --val_ratio 0.07 \
    --early_stopping_patience 20 \
    --early_stopping_eps 0.0001 \
    --d_model 64 \
    --n_layers 4 \
    --dropout 0.1

echo "Completed: Region $region, Feature set $feature_set"
