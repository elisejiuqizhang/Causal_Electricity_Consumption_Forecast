#!/bin/bash

# Train all models on long window (2018-01-01 to 2023-03-10)
# For OOD inference on test period (2023-03-11 to 2024-03-10)
# Uses GNU parallel for concurrent training

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "❌ Error: GNU parallel is not installed"
    echo "Install with: sudo apt-get install parallel"
    exit 1
fi

# Regions to train
REGIONS=("Toronto" "Ottawa")

# Feature sets
FEATURE_SETS=("F0" "F1" "F2" "F3")

# Model types
MODELS=("gru" "tcn" "patchtst")

# Training script
SCRIPT="exps/forecast/per_region/train_long_window.py"

# Number of parallel jobs (adjust based on your GPU memory)
# RTX 3090 24GB can typically handle 2-3 models simultaneously
N_JOBS=6

echo "================================================================================"
echo "Training All Models on Long Window (2018-01-01 to 2023-03-10)"
echo "================================================================================"
echo "Regions: ${REGIONS[@]}"
echo "Feature Sets: ${FEATURE_SETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Total configurations: $((${#REGIONS[@]} * ${#FEATURE_SETS[@]} * ${#MODELS[@]}))"
echo "Parallel jobs: $N_JOBS"
echo "================================================================================"
echo ""

# Save current working directory
WORK_DIR="$(pwd)"

# Create log directory
LOG_DIR="$WORK_DIR/outputs/training_logs_long_window"
mkdir -p "$LOG_DIR"

# Function to train a single model
train_model() {
    local model=$1
    local region=$2
    local feature_set=$3
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $model | $region | $feature_set"
    
    # Create log file name
    LOG_FILE="$LOG_DIR/${model}_${region}_${feature_set}.log"
    
    # Activate conda environment and run training
    # Use the current working directory to find the script
    cd "$WORK_DIR" || exit 1
    
    # Run training and capture output
    python "$SCRIPT" \
        --model_type "$model" \
        --region "$region" \
        --feature_set "$feature_set" \
        --seed 97 \
        --train_start "2018-01-01" \
        --train_end "2023-03-10" \
        --input_length 168 \
        --horizon 24 \
        --batch_size 64 \
        --epochs 500 \
        --lr 0.0001 \
        --early_stopping_patience 25 \
        --early_stopping_eps 0.0001 \
        --scaler standard \
        > "$LOG_FILE" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Successfully trained: $model | $region | $feature_set"
        echo "SUCCESS" >> "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed: $model | $region | $feature_set (exit code: $exit_code)"
        echo "FAILED (exit code: $exit_code)" >> "$LOG_FILE"
    fi
    
    return $exit_code
}

# Export function and variables for parallel
export -f train_model
export SCRIPT
export LOG_DIR
export WORK_DIR

# Generate all combinations and run with GNU parallel
# --no-notice: suppress citation notice
# --line-buffer: print output as it arrives
# --tagstring: add tag to identify which job produced output
parallel -j $N_JOBS --no-notice --line-buffer --tagstring "[{1}|{2}|{3}]" --joblog "$LOG_DIR/parallel_jobs.log" \
    train_model {1} {2} {3} \
    ::: "${MODELS[@]}" \
    ::: "${REGIONS[@]}" \
    ::: "${FEATURE_SETS[@]}"

echo ""
echo "================================================================================"
echo "✅ All training jobs completed!"
echo "================================================================================"
echo "Job log saved to: $LOG_DIR/parallel_jobs.log"
echo "Individual logs saved to: $LOG_DIR/"
echo ""
echo "Summary:"
grep -c "SUCCESS" "$LOG_DIR"/*.log | awk -F: '{s+=$2} END {print "  Successful: " s}'
grep -c "FAILED" "$LOG_DIR"/*.log | awk -F: '{s+=$2} END {print "  Failed: " s}'
echo "================================================================================"
