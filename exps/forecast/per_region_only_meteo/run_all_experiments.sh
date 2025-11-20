#!/bin/bash
# Unified script to run  uni-variate forecasting experiments (without meteorological features) locally with optimal resource usage

set -e  # Exit on error

# ============================================
# Configuration
# ============================================

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname $(dirname "$(dirname "$SCRIPT_DIR")"))"
cd "$PROJECT_ROOT"

# Experiment directories
SINGLE_REGION_DIR="exps/forecast/per_region_only_meteo"

# Parallel jobs - adjust based on your CPU cores and GPU availability
# Recommendation: Use 2-4 jobs for GPU experiments to avoid OOM
NUM_PARALLEL_JOBS=${1:-4}  # Default to 4, can override with first argument

# Region and feature configurations
SINGLE_REGIONS=("Toronto" "Peel" "Hamilton" "Brantford" "Waterloo" "London" "Oshawa" "Kingston" "Ottawa")
FEATURE_SETS=("F0" "F1" "F2" "F3")
# FEATURE_SETS=("F2")

HORIZONS=(1 2 4 6 12 24)

# Model types
MODELS=("patchtst" "tcn" "gru")

# Common hyperparameters (as string for export compatibility)
COMMON_ARGS="--scaler standard --n_folds 15 --window_size 17568 --epochs 1000 --batch_size 32 --lr 0.0001 --train_ratio 0.93 --val_ratio 0.07 --early_stopping_patience 35 --early_stopping_eps 0.00005 --seed 697"

# Model-specific hyperparameters (as simple variables for export)
PATCHTST_D_MODEL=64
PATCHTST_N_HEADS=4
PATCHTST_N_LAYERS=4
PATCHTST_PATCH_LEN=16
PATCHTST_PATCH_STRIDE=8
PATCHTST_DROPOUT=0.1

TCN_HIDDEN_CHANNELS=64
TCN_LEVELS=5
TCN_KERNEL_SIZE=3
TCN_DILATION_BASE=2
TCN_DROPOUT=0.1

GRU_D_MODEL=64
GRU_N_LAYERS=4
GRU_DROPOUT=0.1

# ============================================
# Helper Functions
# ============================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

get_model_args() {
    local model=$1
    case $model in
        patchtst)
            echo "--d_model $PATCHTST_D_MODEL --n_heads $PATCHTST_N_HEADS --n_layers $PATCHTST_N_LAYERS --patch_len $PATCHTST_PATCH_LEN --patch_stride $PATCHTST_PATCH_STRIDE --dropout $PATCHTST_DROPOUT"
            ;;
        tcn)
            echo "--hidden_channels $TCN_HIDDEN_CHANNELS --levels $TCN_LEVELS --kernel_size $TCN_KERNEL_SIZE --dilation_base $TCN_DILATION_BASE --dropout $TCN_DROPOUT"
            ;;
        gru)
            echo "--d_model $GRU_D_MODEL --n_layers $GRU_N_LAYERS --dropout $GRU_DROPOUT"
            ;;
    esac
}

# ============================================
# Experiment Runner Functions
# ============================================

run_single_region_experiment() {
    local model=$1
    local region=$2
    local feature=$3
    local horizon=$4
    
    local script="${model}_single_train.py"
    local model_args=$(get_model_args "$model")
    
    log_message "Running single-region: Model=$model, Region=$region, Feature=$feature, Horizon=$horizon"
    
    # Run from project root, specify full path to script
    python "$PROJECT_ROOT/$SINGLE_REGION_DIR/$script" \
        --region "$region" \
        --feature_set "$feature" \
        --horizon "$horizon" \
        $COMMON_ARGS \
        $model_args
}

# Export functions and variables for GNU parallel
export -f run_single_region_experiment
export -f get_model_args
export -f log_message
export PROJECT_ROOT
export SINGLE_REGION_DIR
export COMMON_ARGS
# Export model hyperparameters
export PATCHTST_D_MODEL PATCHTST_N_HEADS PATCHTST_N_LAYERS PATCHTST_PATCH_LEN PATCHTST_PATCH_STRIDE PATCHTST_DROPOUT
export TCN_HIDDEN_CHANNELS TCN_LEVELS TCN_KERNEL_SIZE TCN_DILATION_BASE TCN_DROPOUT
export GRU_D_MODEL GRU_N_LAYERS GRU_DROPOUT

# ============================================
# Main Execution
# ============================================

print_summary() {
    echo ""
    echo "======================================"
    echo "Experiment Configuration Summary"
    echo "======================================"
    echo "Parallel Jobs: $NUM_PARALLEL_JOBS"
    echo "Models: ${MODELS[*]}"
    echo "Single Regions: ${SINGLE_REGIONS[*]}"
    echo "Feature Sets: ${FEATURE_SETS[*]}"
    echo "Horizons: ${HORIZONS[*]}"
    echo ""
    echo "Total Single-Region Jobs: $((${#MODELS[@]} * ${#SINGLE_REGIONS[@]} * ${#FEATURE_SETS[@]} * ${#HORIZONS[@]}))"
    echo "Total Jobs: $((${#MODELS[@]} * ${#SINGLE_REGIONS[@]} * ${#FEATURE_SETS[@]} * ${#HORIZONS[@]}))"
    echo "======================================"
    echo ""
}

run_all_experiments() {
    log_message "Starting all experiments with $NUM_PARALLEL_JOBS parallel jobs"
    
    # Run single-region experiments
    log_message "========== SINGLE-REGION EXPERIMENTS =========="
    for model in "${MODELS[@]}"; do
        log_message "Running single-region experiments for model: $model"
        parallel --jobs "$NUM_PARALLEL_JOBS" --line-buffer \
            run_single_region_experiment "$model" ::: "${SINGLE_REGIONS[@]}" ::: "${FEATURE_SETS[@]}" ::: "${HORIZONS[@]}"
    done
    
    log_message "All experiments completed successfully!"
}

run_single_region_only() {
    log_message "Running only single-region experiments with $NUM_PARALLEL_JOBS parallel jobs"
    
    for model in "${MODELS[@]}"; do
        log_message "Running single-region experiments for model: $model"
        parallel --jobs "$NUM_PARALLEL_JOBS" --line-buffer \
            run_single_region_experiment "$model" ::: "${SINGLE_REGIONS[@]}" ::: "${FEATURE_SETS[@]}" ::: "${HORIZONS[@]}"
    done
    
    log_message "Single-region experiments completed!"
}

run_specific_model() {
    local model=$1
    log_message "Running experiments for model: $model"
    
    log_message "Single-region experiments for $model"
    parallel --jobs "$NUM_PARALLEL_JOBS" --line-buffer \
        run_single_region_experiment "$model" ::: "${SINGLE_REGIONS[@]}" ::: "${FEATURE_SETS[@]}" ::: "${HORIZONS[@]}"
    
    log_message "Experiments for $model completed!"
}

# ============================================
# Usage Menu
# ============================================

show_usage() {
    cat << EOF
Usage: $0 [NUM_JOBS] [OPTION]

Run forecasting experiments with optimal parallelization.

Arguments:
  NUM_JOBS    Number of parallel jobs (default: 4)
              Recommended: 2-4 for GPU experiments, 4-8 for CPU

Options:
  all         Run all experiments (default)
  patchtst    Run only PatchTST experiments 
  tcn         Run only TCN experiments 
  gru         Run only GRU experiments 
  summary     Show experiment summary without running
  help        Show this help message

Examples:
  $0                          # Run all with 4 parallel jobs
  $0 2                        # Run all with 2 parallel jobs
  $0 4 single                  # Run single-region only with 4 jobs
  $0 8 gru                    # Run GRU experiments with 8 jobs
  $0 summary                  # Show what would be run

EOF
}

# ============================================
# Main Entry Point
# ============================================

OPTION=${2:-all}

case $OPTION in
    all)
        print_summary
        run_all_experiments
        ;;
    single)
        print_summary
        run_single_region_only
        ;;
    patchtst|tcn|gru)
        print_summary
        run_specific_model "$OPTION"
        ;;
    summary)
        print_summary
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Error: Unknown option '$OPTION'"
        echo ""
        show_usage
        exit 1
        ;;
esac
