import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import contextlib
import graphviz

from utils.data_utils.datetime_utils import add_hourly_calendar_features
from utils.data_utils.info_cities import list_cities, dict_city_coords
from utils.data_utils.processing import min_max_scale, corr
from utils.causality.mxmap import MXMap

#output dir
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'results_mxmap_meteo2demande')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# city of interest
city_idx=0 # between 0 and 24
city_name=list_cities[city_idx]
city_coords=dict_city_coords[city_name]

is_avg_ieso=True

# load data
ERA5_DIR = os.path.join(ROOT, 'data', 'era5')
IESO_DIR = os.path.join(ROOT, 'data', 'ieso_hourly')
era5_df = pd.read_csv(os.path.join(ERA5_DIR, f'{city_name.replace(" ", "_").lower()}_era5_timeseries.csv.gz'), parse_dates=['time'])
if is_avg_ieso:
    ieso_df = pd.read_csv(os.path.join(IESO_DIR, f'ieso_residential_avg_per_premise_{city_name.replace(" ", "_")}.csv.gz'), parse_dates=['TIMESTAMP'])
else:
    ieso_df = pd.read_csv(os.path.join(IESO_DIR, f'ieso_residential_{city_name.replace(" ", "_")}.csv.gz'), parse_dates=['TIMESTAMP'])

# print(list(era5_df.columns)) #'time', 'tcw', 'tcc', 't2m_degC', 'u10_ms', 'v10_ms', 'd2m_degC', 'ssr', 'net_sw_Wm2', 'tp_mm', 'e_mm', 'skt', 'lw_down_Wm2', 'lw_up_Wm2', 'net_lw_Wm2', 'net_radiation_Wm2', 'city'
# print(list(ieso_df.columns)) #'TIMESTAMP', 'TOTAL_CONSUMPTION', 'PREMISE_COUNT'   or    'TIMESTAMP', 'TOTAL_CONSUMPTION', 'PREMISE_COUNT', 'AVG_CONSUMPTION_PER_PREMISE'

# variables of interest
meteo_vars = [
    'tp_mm',    
    'e_mm',         
    'tcw', 
    't2m_degC',# in m/s
    'u10_ms',# in m/s
    'v10_ms',# in m/s
    'net_radiation_Wm2'  # in W/m2
]

if is_avg_ieso:
    ieso_var = 'AVG_CONSUMPTION_PER_PREMISE'  # in kWh
else:
    ieso_var = 'TOTAL_CONSUMPTION'  # in kWh


# reduce to time+vars of interest
era5_df = era5_df[['time'] + meteo_vars]
ieso_df = ieso_df[['TIMESTAMP', ieso_var]]

# merge on time
df = pd.merge(era5_df, ieso_df, left_on='time', right_on='TIMESTAMP', how='inner')
df = df.drop(columns=['TIMESTAMP'])

# min_max scale all variables
for col in df.columns:
    if col != 'time':
        df[col], min_val, max_val = min_max_scale(df[col])

# # examine for NaNs
# print(f'Number of NaNs in the merged dataframe for {city_name}:')
# print(df.isna().sum())
# # locate NaNs
# nan_indices = df[df.isna().any(axis=1)].index
# if len(nan_indices) > 0:
#     print(f'Found {len(nan_indices)} rows with NaNs in the merged dataframe for {city_name}. Dropping these rows.')
#     df = df.drop(index=nan_indices)

# choose index between 1 and 51738 (inclusive) - from reading the nan_indices list
df = df[1:51739]  # to avoid any potential issues with lag 0 in PCMCI

OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'results_mxmap_meteo2demande')
os.makedirs(OUTPUT_DIR, exist_ok=True)
file_dir=os.path.join(OUTPUT_DIR, f'mxmap_meteo2demande_{city_name.replace(" ", "_")}')

# create MXMap object
model=MXMap(df, tau=3, emd=5, score_type='corr', bivCCM_thres=0.75, pcm_thres=0.5, knn=10, L=5000, method='vanilla')

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