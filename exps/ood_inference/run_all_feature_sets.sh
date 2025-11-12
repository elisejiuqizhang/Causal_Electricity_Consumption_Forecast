#!/bin/bash

# ============================================================================
# OOD Inference for All Feature Sets
# ============================================================================
# This script runs OOD inference for F0, F1, F2, F3 on all models and regions
# ============================================================================

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

echo "============================================"
echo "OOD Inference Runner - All Feature Sets"
echo "============================================"
echo "Models: gru, tcn, patchtst"
echo "Regions: Toronto, Ottawa"
echo "Feature Sets: F0, F1, F2, F3"
echo "Fold: 0"
echo "============================================"
echo ""

# Configuration
REGIONS=("Toronto" "Ottawa")
MODELS=("gru" "tcn" "patchtst")
FEATURE_SETS=("F0" "F1" "F2" "F3")
FOLD=0

# GRU/TCN configuration
GRU_SEED=97
TCN_SEED=97

# PatchTST configuration
PATCHTST_SEED=597

# Model hyperparameters
GRU_D_MODEL=64
GRU_N_LAYERS=4
GRU_DROPOUT=0.1

TCN_HIDDEN_CHANNELS=64
TCN_LEVELS=4
TCN_KERNEL_SIZE=3
TCN_DILATION_BASE=2
TCN_DROPOUT=0.1

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

# OOD analysis directory
OOD_DIR="${PROJECT_ROOT}/outputs/ood_analysis"

# Function to run inference
run_inference() {
    local model=$1
    local region=$2
    local feature_set=$3
    local ood_file=$4
    local seed=$5
    
    echo "=========================================="
    echo "Running ${model} inference for ${region} - ${feature_set}"
    echo "OOD file: ${ood_file}"
    echo "=========================================="
    
    case $model in
        gru)
            python exps/ood_inference/gru_ood_inference.py \
                --region "$region" \
                --feature_set "$feature_set" \
                --fold $FOLD \
                --seed $seed \
                --ood_file "$ood_file" \
                --d_model $GRU_D_MODEL \
                --n_layers $GRU_N_LAYERS \
                --dropout $GRU_DROPOUT \
                --input_length $INPUT_LENGTH \
                --horizon $HORIZON \
                --stride $STRIDE \
                --batch_size $BATCH_SIZE
            ;;
        tcn)
            python exps/ood_inference/tcn_ood_inference.py \
                --region "$region" \
                --feature_set "$feature_set" \
                --fold $FOLD \
                --seed $seed \
                --ood_file "$ood_file" \
                --hidden_channels $TCN_HIDDEN_CHANNELS \
                --levels $TCN_LEVELS \
                --kernel_size $TCN_KERNEL_SIZE \
                --dilation_base $TCN_DILATION_BASE \
                --dropout $TCN_DROPOUT \
                --input_length $INPUT_LENGTH \
                --horizon $HORIZON \
                --stride $STRIDE \
                --batch_size $BATCH_SIZE
            ;;
        patchtst)
            python exps/ood_inference/patchtst_ood_inference.py \
                --region "$region" \
                --feature_set "$feature_set" \
                --fold $FOLD \
                --seed $seed \
                --ood_file "$ood_file" \
                --d_model $PATCHTST_D_MODEL \
                --n_heads $PATCHTST_N_HEADS \
                --n_layers $PATCHTST_N_LAYERS \
                --patch_len $PATCHTST_PATCH_LEN \
                --patch_stride $PATCHTST_PATCH_STRIDE \
                --dropout $PATCHTST_DROPOUT \
                --input_length $INPUT_LENGTH \
                --horizon $HORIZON \
                --stride $STRIDE \
                --batch_size $BATCH_SIZE
            ;;
        *)
            echo "Unknown model: $model"
            return 1
            ;;
    esac
    
    echo ""
}

# Counter for tracking progress
total_runs=$((${#MODELS[@]} * ${#REGIONS[@]} * ${#FEATURE_SETS[@]}))
current_run=0

# Loop through all combinations
for feature_set in "${FEATURE_SETS[@]}"; do
    for region in "${REGIONS[@]}"; do
        # Construct OOD file path
        ood_file="${OOD_DIR}/ood_windows_${region}_val.csv"
        
        # Check if OOD file exists
        if [ ! -f "$ood_file" ]; then
            echo "WARNING: OOD file not found: $ood_file"
            echo "Skipping ${region} for feature set ${feature_set}"
            continue
        fi
        
        for model in "${MODELS[@]}"; do
            current_run=$((current_run + 1))
            echo ""
            echo "========================================"
            echo "Progress: ${current_run}/${total_runs}"
            echo "Feature Set: ${feature_set}"
            echo "========================================"
            
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
            
            # Run inference
            if run_inference "$model" "$region" "$feature_set" "$ood_file" "$seed"; then
                echo "✓ ${model} ${region} ${feature_set} completed successfully"
            else
                echo "✗ ${model} ${region} ${feature_set} failed"
                # Continue with other runs instead of exiting
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "All OOD inference runs completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - outputs/ood_inference/{model}/{Region}_{FeatureSet}_fold0_*"
echo ""
echo "Next steps:"
echo "  1. Generate comparison reports for each feature set:"
echo "     python exps/ood_inference/compare_ood_normal.py --regions Toronto Ottawa --models gru tcn patchtst --feature_set F0 --fold 0"
echo "     python exps/ood_inference/compare_ood_normal.py --regions Toronto Ottawa --models gru tcn patchtst --feature_set F1 --fold 0"
echo "     python exps/ood_inference/compare_ood_normal.py --regions Toronto Ottawa --models gru tcn patchtst --feature_set F3 --fold 0"
echo ""
echo "  2. Compare feature sets:"
echo "     python exps/ood_inference/compare_feature_sets.py"
echo ""
