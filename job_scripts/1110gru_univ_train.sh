cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/Causal_Electricity_Consumption_Forecast/exps/forecast/per_region

SCRIPT="gru_single_train.py"
export SCRIPT

list_regions=("Toronto" "Peel" "Hamilton" "Brantford" "Waterloo" "London" "Oshawa" "Kingston" "Ottawa")
list_features=("F0" "F1" "F2" "F3")

run_single(){
    python "$SCRIPT" \
    --region "$1" \
    --feature_set "$2" 
}

export -f run_single
parallel --jobs 4 run_single ::: "${list_regions[@]}" ::: "${list_features[@]}"