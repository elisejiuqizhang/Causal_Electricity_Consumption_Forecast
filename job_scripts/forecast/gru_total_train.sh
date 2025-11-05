python exps/forecast/train/train_forecast.py \
  --mode total \
  --city-electricity-csv "Ottawa=data/ieso_hourly/ieso_residential_Ottawa.csv.gz" --city-electricity-csv "Toronto=data/ieso_hourly/ieso_residential_Toronto.csv.gz" --city-electricity-csv "Hamilton=data/ieso_hourly/ieso_residential_Hamilton.csv.gz" \
  --city-meteo-csv      "Ottawa=data/era5/ottawa_era5_timeseries.csv.gz" --city-meteo-csv      "Toronto=data/era5/toronto_era5_timeseries.csv.gz" --city-meteo-csv      "Hamilton=data/era5/hamilton_era5_timeseries.csv.gz" \
  --features "t2m_degC, tp_mm, net_radiation_Wm2" \
  --history 168 --horizon 24 \
  --epochs 500 --batch-size 32 \
  --model gru \
  --model-kwarg 'hidden_size=256' --model-kwarg 'num_layers=2'   --model-kwarg 'output_size=1'\
  --exp-name total_sum_3cities_h168_h24_gru
