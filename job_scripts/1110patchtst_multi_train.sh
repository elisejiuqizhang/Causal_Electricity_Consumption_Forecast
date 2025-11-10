cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast/exps/forecast/per_region

SCRIPT="patchtst_multi_train.py"
export SCRIPT

list_feature_sets=("F0" "F1" "F2" "F3")
list_regions=(1 2 3 4 5 6 7)

run_single(){
    python "$SCRIPT" \
    --region_ilist "$1" \
    --feature_set "$2" 
}
export -f run_single
parallel --jobs 4 run_single ::: "${list_regions[@]}" ::: "${list_feature_sets[@