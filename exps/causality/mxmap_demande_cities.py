# pcmci, load all cities' electricity demand data (total, not the average one)
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
OUTPUT_DIR = os.path.join(ROOT, 'outputs', 'causality', 'results_mxmap_demande_cities')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# load data
IESO_DIR = os.path.join(ROOT, 'data', 'ieso_hourly')
list_dfs = []
for city_name in list_cities:
    ieso_df = pd.read_csv(os.path.join(IESO_DIR, f'ieso_residential_{city_name.replace(" ", "_")}.csv.gz'), parse_dates=['TIMESTAMP'])
    ieso_df = ieso_df[['TIMESTAMP', 'TOTAL_CONSUMPTION']]
    ieso_df = ieso_df.rename(columns={'TOTAL_CONSUMPTION': city_name})
    list_dfs.append(ieso_df)
df = list_dfs[0]
for other_df in list_dfs[1:]:
    df = pd.merge(df, other_df, on='TIMESTAMP', how='inner')

df = df.set_index('TIMESTAMP')

# create MXMap object
model=MXMap(df, tau=3, emd=5, score_type='corr', bivCCM_thres=0.75, pcm_thres=0.5, knn=10, L=5000, method='vanilla')
file_dir = os.path.join(OUTPUT_DIR, f'demande_lag{model.tau}_emd{model.emd}')

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