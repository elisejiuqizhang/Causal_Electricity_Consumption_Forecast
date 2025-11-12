#!/bin/bash
# Run OOD inference for all models on Toronto and Ottawa
# This script tests pretrained models on identified extreme weather windows

set -e

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# Configuration
REGIONS=("Toronto" "Ottawa")
MODELS=("gru" "tcn" "patchtst")
FEATURE_SET="F2"  # Change this if your models use different feature set
FOLD=0  # Default fold to use

# Model hyperparameters (must match training configuration)
# GRU
GRU_D_MODEL=64
GRU_N_LAYERS=4
GRU_DROPOUT=0.1

# TCN
TCN_HIDDEN_CHANNELS=64
TCN_LEVELS=4
TCN_KERNEL_SIZE=3
TCN_DILATION_BASE=2
TCN_DROPOUT=0.1

# PatchTST
PATCHTST_D_MODEL=64
PATCHTST_N_HEADS=4
PATCHTST_N_LAYERS=3
PATCHTST_PATCH_LEN=16
PATCHTST_PATCH_STRIDE=8
PATCHTST_DROPOUT=0.1

# Common parameters
INPUT_LENGTH=168
HORIZON=24
STRIDE=1
BATCH_SIZE=64

# Training config folder name
TRAINING_CONFIG="bs64_ep500_lr0.0001_tr0.93_vr0.07_pat20_esep0.0001"

# Seeds (adjust based on your training)
GRU_SEED=97
TCN_SEED=97
PATCHTST_SEED=597

# OOD analysis directory
OOD_DIR="${PROJECT_ROOT}/outputs/ood_analysis"

# Function to run inference
run_inference() {
    local model=$1
    local region=$2
    local ood_file=$3
    local seed=$4
    
    echo "=========================================="
    echo "Running ${model} inference for ${region}"
    echo "OOD file: ${ood_file}"
    echo "=========================================="
    
    case $model in
        gru)
            python exps/ood_inference/gru_ood_inference.py \
                --region "$region" \
                --feature_set "$FEATURE_SET" \
                --fold "$FOLD" \
                --seed "$seed" \
                --d_model "$GRU_D_MODEL" \
                --n_layers "$GRU_N_LAYERS" \
                --dropout "$GRU_DROPOUT" \
                --input_length "$INPUT_LENGTH" \
                --horizon "$HORIZON" \
                --stride "$STRIDE" \
                --batch_size "$BATCH_SIZE" \
                --training_config "$TRAINING_CONFIG" \
                --ood_file "$ood_file"
            ;;
        tcn)
            python exps/ood_inference/tcn_ood_inference.py \
                --region "$region" \
                --feature_set "$FEATURE_SET" \
                --fold "$FOLD" \
                --seed "$seed" \
                --hidden_channels "$TCN_HIDDEN_CHANNELS" \
                --levels "$TCN_LEVELS" \
                --kernel_size "$TCN_KERNEL_SIZE" \
                --dilation_base "$TCN_DILATION_BASE" \
                --dropout "$TCN_DROPOUT" \
                --input_length "$INPUT_LENGTH" \
                --horizon "$HORIZON" \
                --stride "$STRIDE" \
                --batch_size "$BATCH_SIZE" \
                --training_config "$TRAINING_CONFIG" \
                --ood_file "$ood_file"
            ;;
        patchtst)
            python exps/ood_inference/patchtst_ood_inference.py \
                --region "$region" \
                --feature_set "$FEATURE_SET" \
                --fold "$FOLD" \
                --seed "$seed" \
                --d_model "$PATCHTST_D_MODEL" \
                --n_heads "$PATCHTST_N_HEADS" \
                --n_layers "$PATCHTST_N_LAYERS" \
                --patch_len "$PATCHTST_PATCH_LEN" \
                --patch_stride "$PATCHTST_PATCH_STRIDE" \
                --dropout "$PATCHTST_DROPOUT" \
                --input_length "$INPUT_LENGTH" \
                --horizon "$HORIZON" \
                --stride "$STRIDE" \
                --batch_size "$BATCH_SIZE" \
                --training_config "$TRAINING_CONFIG" \
                --ood_file "$ood_file"
            ;;
        *)
            echo "Error: Unknown model type '$model'"
            exit 1
            ;;
    esac
    
    echo ""
}

# Main execution
echo "=========================================="
echo "OOD Inference Runner"
echo "=========================================="
echo "Regions: ${REGIONS[@]}"
echo "Models: ${MODELS[@]}"
echo "Feature Set: $FEATURE_SET"
echo "Fold: $FOLD"
echo "=========================================="
echo ""

# Check if OOD windows exist
for region in "${REGIONS[@]}"; do
    ood_file="${OOD_DIR}/ood_windows_${region}_val.csv"
    if [ ! -f "$ood_file" ]; then
        echo "Error: OOD file not found: $ood_file"
        echo "Please run identify_ood_weather.py first to generate OOD windows"
        exit 1
    fi
done

# Run inference for each combination
for region in "${REGIONS[@]}"; do
    ood_file="${OOD_DIR}/ood_windows_${region}_val.csv"
    
    for model in "${MODELS[@]}"; do
        # Determine seed based on model
        case $model in
            gru)
                seed=$GRU_SEED
                ;;
            tcn)
                seed=$TCN_SEED
                ;;
            patchtst)
                seed=$PATCHTST_SEED
                ;;
        esac
        
        run_inference "$model" "$region" "$ood_file" "$seed"
    done
done

echo "=========================================="
echo "All OOD inference runs completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - outputs/ood_inference/gru/"
echo "  - outputs/ood_inference/tcn/"
echo "  - outputs/ood_inference/patchtst/"
echo ""
echo "Each directory contains:"
echo "  - *_ood_metrics.csv: Per-window metrics summary"
echo "  - *_ood_predictions.csv: Detailed predictions for each timestep"
echo "  - *_summary.txt: Text summary with statistics"
