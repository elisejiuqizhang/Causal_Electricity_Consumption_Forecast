#!/bin/bash
#SBATCH --job-name=tcn_multi
#SBATCH --output=logs/tcn_multi_%A_%a.out
#SBATCH --error=logs/tcn_multi_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-27

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate scratch

cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast/exps/forecast/multi_region

SCRIPT="tcn_multi_train.py"

# Define parameter combinations
list_feature_sets=("F0" "F1" "F2" "F3")
list_regions=(1 2 3 4 5 6 7)

# Calculate indices for this array task
n_features=${#list_feature_sets[@]}
n_regions=${#list_regions[@]}

region_idx=$((SLURM_ARRAY_TASK_ID / n_features))
feature_idx=$((SLURM_ARRAY_TASK_ID % n_features))

region=${list_regions[$region_idx]}
feature_set=${list_feature_sets[$feature_idx]}

echo "Running: Region list $region, Feature set $feature_set"

python "$SCRIPT" \
    --region_list "$region" \
    --feature_set "$feature_set" \
    --scaler standard \
    --n_folds 5 \
    --window_size 17568 \
    --epochs 500 \
    --batch_size 64 \
    --lr 0.0001 \
    --train_ratio 0.93 \
    --val_ratio 0.07 \
    --early_stopping_patience 20 \
    --early_stopping_eps 0.0001 \
    --hidden_channels 64 \
    --levels 4 \
    --kernel_size 3 \
    --dilation_base 2 \
    --dropout 0.1

echo "Completed: Region list $region, Feature set $feature_set"
