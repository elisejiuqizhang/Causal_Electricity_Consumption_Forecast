#!/bin/bash

cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast/exps/forecast/multi_region

SCRIPT="tcn_multi_train.py"
export SCRIPT

list_feature_sets=("F0" "F1" "F2" "F3")
list_regions=(1 2 3 4 5 6 7)

run_single(){
    python "$SCRIPT" \
    --region_list "$1" \
    --feature_set "$2" \
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
}

export -f run_single
parallel --jobs 4 run_single ::: "${list_regions[@]}" ::: "${list_feature_sets[@]}"
