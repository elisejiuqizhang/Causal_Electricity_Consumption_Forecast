#!/bin/bash

# Run OOD inference for all models trained on long window
# Evaluates on OOD windows from test period with all feature sets

MODELS=("gru" "tcn" "patchtst")
REGIONS=("Toronto" "Ottawa")

SCRIPT="exps/ood_inference_test/ood_inference_long_window.py"

echo "================================================================================"
echo "OOD Inference - Long Window Models"
echo "================================================================================"
echo "Models: ${MODELS[@]}"
echo "Regions: ${REGIONS[@]}"
echo "Feature Sets: F0, F1, F2, F3 (all tested per model)"
echo "Total configurations: $((${#MODELS[@]} * ${#REGIONS[@]}))"
echo "================================================================================"
echo ""

TOTAL=$((${#MODELS[@]} * ${#REGIONS[@]}))
CURRENT=0

for model in "${MODELS[@]}"; do
    for region in "${REGIONS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "--------------------------------------------------------------------------------"
        echo "[$CURRENT/$TOTAL] Running: $model | $region (testing F0, F1, F2, F3)"
        echo "--------------------------------------------------------------------------------"
        
        python $SCRIPT \
            --model_type $model \
            --region $region \
            --seed 97 \
            --test_start "2023-03-11" \
            --test_end "2024-03-10" \
            --input_length 168 \
            --horizon 24 \
            --batch_size 64 \
            --scaler standard
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed: $model | $region"
        else
            echo "❌ Failed: $model | $region"
        fi
    done
done

echo ""
echo "================================================================================"
echo "✅ All OOD inference jobs completed!"
echo "================================================================================"
echo "Results saved in: outputs/ood_inference_test/"
echo ""
echo "Next step: Compare results across feature sets"
echo "================================================================================"
