# see how the weather variables propagate across cities
import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import contextlib
import time

from utils.data_utils.datetime_utils import add_hourly_calendar_features
from utils.data_utils.info_cities import list_cities, dict_city_coords
from utils.data_utils.processing import min_max_scale, corr
from utils.causality.mxmap import MXMap

from tigramite import plotting as tp

#output dir
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'results_mxmap_meteo_cities')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# var of interest
meteo_var = 't2m_degC'  # temperature in degC

# load data
ERA5_DIR = os.path.join(ROOT, 'data', 'era5')
list_dfs = []
for city_name in list_cities:
    era5_df = pd.read_csv(os.path.join(ERA5_DIR, f'{city_name.replace(" ", "_").replace(".","").lower()}_era5_timeseries.csv.gz'), parse_dates=['time'])
    era5_df = era5_df[['time', meteo_var]]
    era5_df = era5_df.rename(columns={meteo_var: city_name})
    list_dfs.append(era5_df)
df = list_dfs[0]
for other_df in list_dfs[1:]:
    df = pd.merge(df, other_df, on='time', how='inner')

df = df.set_index('time')


# create MXMap object
model=MXMap(df, tau=3, emd=5, score_type='corr', bivCCM_thres=0.75, pcm_thres=0.5, knn=10, L=4000, method='vanilla')

file_dir = os.path.join(OUTPUT_DIR, f'{meteo_var}_lag{model.tau}_emd{model.emd}')

start_time = time.time()
ch=model.fit()
time_spent = time.time() - start_time

model.draw_graph(file_dir)

print('Time spent:', time_spent)
with open(file_dir+'_time.txt', 'w') as f:
    f.write(str(time_spent))

print('ch:', ch)
with open(file_dir+'_ch.txt', 'w') as f:
    f.write(str(ch))

# print the stats in phase one - determine the order
print('Phase 1 stats:')
print(model.phase1_stats)
with open(file_dir+'_phase1_stats.txt', 'w') as f:
    f.write(str(model.phase1_stats))

# print the stats in phase two - determine the PCM
print('Phase 2 stats:')
print(model.phase2_stats)
with open(file_dir+'_phase2_stats.txt', 'w') as f:
    f.write(str(model.phase2_stats))